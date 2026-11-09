#!/usr/bin/env python3
"""
Qwen3.6-35B-A3B (MoE GGUF Q4_K_M) Multi-GPU Server for Kaggle Dual NVIDIA Tesla T4 (2x 16GB).
Repairs broken libcuda.so symlinks in Kaggle Ubuntu containers, compiles llama.cpp with CUDA sm_75,
caches build artifacts in /kaggle/working, validates Hugging Face GGUF artifacts, splits model
layers across 2x T4 GPUs (-sm layer) with FlashAttention (-fa on) and 262,144 context, verifies
the local /v1/models endpoint, and exposes an OpenAI-compatible /v1 endpoint via Cloudflare Tunnel.
"""

import os
import re
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

from typing import Any, Sequence

import requests

# ============================================================
# 0. CONFIGURATION (KAGGLE DUAL TESLA T4 x2 — QWEN 3.6 35B-A3B)
# ============================================================
MODEL_REPO = "HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive"
MODEL_FILE = "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf"

# 262,144 tokens (256K context window) split across 2x Tesla T4 GPUs (32 GB total VRAM)
CONTEXT = 262144
MIN_CONTEXT = 65536

PORT = 5000
ALIAS = "qwen3.6-35b"
STARTUP_TIMEOUT = 1800

# ============================================================
# KAGGLE PERSISTENCE ("Files only" or "Variables and Files")
# ============================================================
# In Kaggle, only /kaggle/working is persisted across sessions (~20 GiB quota).
# /kaggle/working is typically mounted with 'noexec'. We cache the built binaries
# in /kaggle/working, ensure chmod 0o755, and execute cloudflared from /usr/local/bin.
KAGGLE_WORKING = Path("/kaggle/working") if Path("/kaggle/working").exists() else (
    Path("/content") if Path("/content").exists() else Path.cwd()
)
KAGGLE_INPUT = Path("/kaggle/input")

SOURCE_DIR = KAGGLE_WORKING / "llama.cpp"
SERVER_BIN = SOURCE_DIR / "build/bin/llama-server"
CLOUDFLARED_CACHE = KAGGLE_WORKING / "cloudflared"
CLOUDFLARED = Path("/usr/local/bin/cloudflared")
SERVER_LOG = KAGGLE_WORKING / "llama_server.log"
TUNNEL_LOG = KAGGLE_WORKING / "cf_tunnel.log"

MODEL_DIR = Path("/tmp/models")

# ============================================================
# NOTE ON API KEY (DISABLED BY DEFAULT)
# ============================================================
# By default, llama-server runs WITHOUT an API key so execution never
# hangs on interactive prompts (getpass) or causes 401 header mismatches.
# Any client (Curl, Python openai SDK, NextChat, OpenWebUI) can pass
# any placeholder key (e.g. "sk-local") or none.
#
# Optional: To enforce Bearer token authentication, uncomment:
# import getpass
# API_KEY = getpass.getpass("Enter custom API Key (min 24 chars): ")
# env["LLAMA_API_KEY"] = API_KEY


def run(args: Sequence[Any], **kwargs: Any) -> subprocess.CompletedProcess:
    return subprocess.run([str(a) for a in args], check=True, **kwargs)


def tail(path: Path, lines: int = 60) -> str:
    if not path.exists():
        return "(log file not created yet)"
    with path.open(errors="replace") as f:
        return "".join(deque(f, maxlen=lines))


def stop_process(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def configure_ld_library_path() -> None:
    """Ensure dynamic linker finds llama.cpp shared libraries across builds and sessions."""
    ld_candidates = [
        SOURCE_DIR / "build/bin",
        SOURCE_DIR / "build/src",
        SOURCE_DIR / "build/common",
        SOURCE_DIR / "build/ggml/src",
        Path("/usr/local/cuda/lib64"),
        Path("/usr/local/cuda/compat"),
        Path("/usr/lib/x86_64-linux-gnu"),
    ]
    existing = [str(p) for p in ld_candidates if p.exists()]
    current = os.environ.get("LD_LIBRARY_PATH", "")
    parts = existing + ([current] if current else [])
    os.environ["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(parts))


# ============================================================
# 1. DEPENDENCIES, KAGGLE LIBCUDA.SO REPAIR & STALE PROCESS CLEANUP
# ============================================================
print("\n[1/6] Installing build dependencies and repairing libcuda.so for Kaggle 2x T4...")

run(["apt-get", "update", "-qq"])
run([
    "apt-get", "install", "-y", "-qq",
    "aria2", "git", "cmake", "build-essential",
    "pkg-config", "libssl-dev", "libcurl4-openssl-dev"
])
run([
    sys.executable, "-m", "pip", "install", "-q",
    "huggingface_hub", "psutil"
])

import psutil  # noqa: E402
from huggingface_hub import HfApi  # noqa: E402

# Repair circular symlink loop on libcuda.so inside Kaggle Ubuntu containers
libcuda_candidates = []
for search_root in [Path("/usr"), Path("/lib")]:
    if search_root.exists():
        for p in search_root.rglob("libcuda.so*"):
            try:
                if p.is_file() and not p.is_symlink():
                    libcuda_candidates.append(p)
            except OSError:
                pass

if libcuda_candidates:
    real_cuda = libcuda_candidates[0]
    print(f"Found physical libcuda binary at: {real_cuda}")
    target_dir = Path("/usr/lib/x86_64-linux-gnu")
    target_dir.mkdir(parents=True, exist_ok=True)
    for target_name in ["libcuda.so", "libcuda.so.1"]:
        dest = target_dir / target_name
        if dest != real_cuda:
            dest.unlink(missing_ok=True)
            shutil.copy2(real_cuda, dest)
    for dup in [
        Path("/usr/local/cuda/lib64/libcuda.so"),
        Path("/usr/local/cuda/lib64/libcuda.so.1"),
    ]:
        if dup.is_symlink():
            dup.unlink(missing_ok=True)
    run(["ldconfig"])
    configure_ld_library_path()

run([
    "nvidia-smi",
    "--query-gpu=index,name,memory.total,memory.free",
    "--format=csv"
])
print(
    f"Available System RAM: "
    f"{psutil.virtual_memory().available / 1024**3:.1f} GiB"
)

for proc in psutil.process_iter(["pid", "cmdline"]):
    try:
        argv = proc.info["cmdline"] or []
        if argv and argv[0] in {str(SERVER_BIN), str(CLOUDFLARED), str(CLOUDFLARED_CACHE)}:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except psutil.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. COMPILE OR REUSE LLAMA.CPP IN /kaggle/working
# ============================================================
print("\n[2/6] Checking for persisted llama-server in /kaggle/working...")

need_build = True
configure_ld_library_path()

if SERVER_BIN.exists():
    try:
        for bin_file in (SOURCE_DIR / "build/bin").glob("*"):
            if bin_file.is_file():
                bin_file.chmod(0o755)
        run([SERVER_BIN, "--version"])
        print("✔ Persisted llama-server found and operational! Skipping compilation.")
        need_build = False
    except Exception as exc:
        print(f"Existing binary check failed ({exc}); rebuilding...")
        need_build = True

if need_build:
    print("Cloning and compiling llama.cpp with CUDA (sm_75 for Tesla T4 x2)...")
    if not SOURCE_DIR.exists():
        run([
            "git", "clone",
            "https://github.com/ggml-org/llama.cpp.git",
            SOURCE_DIR
        ])
    elif not (SOURCE_DIR / ".git").exists():
        shutil.rmtree(SOURCE_DIR, ignore_errors=True)
        run([
            "git", "clone",
            "https://github.com/ggml-org/llama.cpp.git",
            SOURCE_DIR
        ])
    else:
        try:
            run(["git", "-C", SOURCE_DIR, "reset", "--hard", "HEAD"])
            run(["git", "-C", SOURCE_DIR, "clean", "-fd"])
            run(["git", "-C", SOURCE_DIR, "fetch", "origin", "master"])
            run(["git", "-C", SOURCE_DIR, "reset", "--hard", "origin/master"])
        except Exception as git_err:
            print(f"Warning: git pull/reset encountered {git_err}; keeping existing checkout.")

    cuda_cmake = SOURCE_DIR / "ggml/src/ggml-cuda/CMakeLists.txt"
    if cuda_cmake.exists() and Path("/usr/lib/x86_64-linux-gnu/libcuda.so").exists():
        cmake_text = cuda_cmake.read_text(errors="replace")
        if "CUDA::cuda_driver" in cmake_text:
            cuda_cmake.write_text(
                cmake_text.replace(
                    "CUDA::cuda_driver",
                    "/usr/lib/x86_64-linux-gnu/libcuda.so"
                )
            )

    run([
        "cmake",
        "-S", SOURCE_DIR,
        "-B", SOURCE_DIR / "build",
        "-DGGML_CUDA=ON",
        "-DCMAKE_CUDA_ARCHITECTURES=75",
        "-DCMAKE_BUILD_TYPE=Release"
    ])
    run([
        "cmake", "--build", SOURCE_DIR / "build",
        "--config", "Release",
        "--target", "llama-server",
        "-j", str(min(os.cpu_count() or 4, 8))
    ])
    configure_ld_library_path()
    SERVER_BIN.chmod(0o755)
    run([SERVER_BIN, "--version"])


# ============================================================
# 3. LOCATE PERSISTED MODEL OR DOWNLOAD FROM HUGGING FACE
# ============================================================
print("\n[3/6] Checking Qwen3.6-35B-A3B model location (Input / Working / Hugging Face)...")

api = HfApi()
info = api.model_info(MODEL_REPO, files_metadata=True)
item = next(
    (x for x in info.siblings if x.rfilename == MODEL_FILE),
    None
)
if item is None or item.size is None:
    raise RuntimeError(
        f"Model file not found or missing published byte size:\n"
        f"{MODEL_REPO}/{MODEL_FILE}"
    )

persisted_candidates = []
for root in [KAGGLE_INPUT, KAGGLE_WORKING / "models", MODEL_DIR]:
    if root.exists():
        for found in root.rglob(Path(MODEL_FILE).name):
            if found.is_file() and found.stat().st_size == item.size:
                persisted_candidates.append(found)

if persisted_candidates:
    model_path = persisted_candidates[0]
    print(f"✔ Found persisted model ready on disk: {model_path} (Zero download required!)")
else:
    if shutil.disk_usage(KAGGLE_WORKING).free > (item.size + 4 * 1024**3):
        MODEL_DIR = KAGGLE_WORKING / "models"
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / Path(MODEL_FILE).name
    local_size = model_path.stat().st_size if model_path.exists() else 0
    if local_size > item.size:
        raise RuntimeError(f"Local file is larger than remote artifact: {model_path}")

    missing_bytes = item.size - local_size
    print(
        f"- Downloading to {model_path}: {item.size / 1024**3:.2f} GiB\n"
        f"  Revision SHA: {info.sha}"
    )

    free_bytes = shutil.disk_usage(MODEL_DIR).free
    needed_bytes = missing_bytes + 4 * 1024**3

    if free_bytes < needed_bytes:
        raise RuntimeError(
            f"Insufficient disk space in {MODEL_DIR}: {free_bytes / 1024**3:.1f} GiB free; "
            f"approximately {needed_bytes / 1024**3:.1f} GiB required."
        )

    complete = (
        model_path.exists()
        and model_path.stat().st_size == item.size
        and not Path(str(model_path) + ".aria2").exists()
    )

    if not complete:
        url = (
            f"https://huggingface.co/{MODEL_REPO}/resolve/"
            f"{info.sha}/{MODEL_FILE}"
        )
        run([
            "aria2c",
            "--continue=true",
            "--auto-file-renaming=false",
            "--allow-overwrite=true",
            "--max-tries=5", "--retry-wait=10",
            "-x", "8", "-s", "8", "-k", "1M",
            url,
            "-d", MODEL_DIR,
            "-o", model_path.name
        ])

if model_path.stat().st_size != item.size:
    raise RuntimeError(f"Size mismatch after download: {model_path.name}")

with model_path.open("rb") as f:
    if f.read(4) != b"GGUF":
        raise RuntimeError(f"Invalid GGUF magic header: {model_path.name}")


# ============================================================
# 4. LAUNCH LLAMA-SERVER ACROSS 2x TESLA T4 GPUS (-sm layer)
# ============================================================
print("\n[4/6] Starting llama-server across 2x Tesla T4 GPUs (-sm layer)...")

total_threads = os.cpu_count() or 4
gen_threads = max(1, total_threads // 2)

cmd = [
    str(SERVER_BIN),
    "-m", str(model_path),
    "--alias", ALIAS,
    "--jinja",
    "-fa", "on",
    "--cache-reuse", "256",
    "-ngl", "99",
    "-sm", "layer",
    "-t", str(gen_threads),
    "-tb", str(total_threads),
    "-b", "2048",
    "-ub", "512",
    "-np", "1",
    "-c", str(CONTEXT),
    "--host", "127.0.0.1",
    "--port", str(PORT),
    "--temp", "0.7",
    "--no-webui",
]

server = None
tunnel = None
local_url = f"http://127.0.0.1:{PORT}"

try:
    with SERVER_LOG.open("w") as log:
        server = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT
        )

    deadline = time.monotonic() + STARTUP_TIMEOUT

    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(
                f"llama-server exited with code {server.returncode}.\n"
                + tail(SERVER_LOG)
            )

        try:
            response = requests.get(
                f"{local_url}/health",
                timeout=5
            )
            if response.status_code == 200:
                break
        except requests.RequestException:
            pass

        time.sleep(3)
    else:
        raise TimeoutError(
            "Server startup timed out.\n" + tail(SERVER_LOG)
        )

    # Verify /v1/models responds HTTP 200 before opening Cloudflare Tunnel
    models_resp = requests.get(
        f"{local_url}/v1/models",
        timeout=15
    )
    models_resp.raise_for_status()

    log_text = SERVER_LOG.read_text(errors="replace")
    main_contexts = re.findall(
        r"(?m)^.*llama_context:.*\bn_ctx_per_seq\s*=\s*(\d+)",
        log_text
    )
    actual_context = int(main_contexts[0]) if main_contexts else CONTEXT

    if actual_context < MIN_CONTEXT:
        raise RuntimeError(
            f"Allocated context ({actual_context:,}) is below floor ({MIN_CONTEXT:,})."
        )

    print(f"\nVerified effective context window: {actual_context:,} tokens")
    print("OpenAI /v1/models endpoint verified (HTTP 200 OK).")


    # ========================================================
    # 5. CLOUDFLARE TUNNEL (EXECUTED FROM /usr/local/bin)
    # ========================================================
    print("\n[5/6] Launching Cloudflare Tunnel...")

    if not CLOUDFLARED_CACHE.exists() or CLOUDFLARED_CACHE.stat().st_size < 1024 * 1024:
        cf_url = (
            "https://github.com/cloudflare/cloudflared/releases/"
            "latest/download/cloudflared-linux-amd64"
        )
        temporary = CLOUDFLARED_CACHE.with_suffix(".download")

        with requests.get(
            cf_url, stream=True, timeout=(30, 300)
        ) as response:
            response.raise_for_status()
            with temporary.open("wb") as f:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)

        temporary.replace(CLOUDFLARED_CACHE)

    shutil.copy2(CLOUDFLARED_CACHE, CLOUDFLARED)
    CLOUDFLARED.chmod(0o755)

    run([CLOUDFLARED, "--version"])

    with TUNNEL_LOG.open("w") as log:
        tunnel = subprocess.Popen(
            [str(CLOUDFLARED), "tunnel", "--url", local_url],
            stdout=log,
            stderr=subprocess.STDOUT
        )

    public_url = None
    for _ in range(90):
        if tunnel.poll() is not None:
            raise RuntimeError(
                "Cloudflare exited unexpectedly.\n" + tail(TUNNEL_LOG)
            )

        matches = re.findall(
            r"https://[a-zA-Z0-9-]+\.trycloudflare\.com",
            TUNNEL_LOG.read_text(errors="replace")
        )
        if matches:
            public_url = matches[0]
            break

        time.sleep(1)

    if public_url is None:
        raise RuntimeError(
            "Timed out waiting for trycloudflare.com URL.\n" + tail(TUNNEL_LOG)
        )


    # ========================================================
    # 6. OPENAI-COMPATIBLE ENDPOINT DETAILS
    # ========================================================
    print("\n[6/6] Inference Server is Online!")
    print("=" * 60)
    print(f"API BASE URL   : {public_url}/v1")
    print(f"MODEL          : {ALIAS}")
    print("API KEY        : (disabled / optional placeholder like 'sk-local')")
    print(f"EFFECTIVE CTX  : {actual_context:,} tokens (256K)")
    print("MULTI-GPU SPLIT: 2x Tesla T4 (-sm layer / FlashAttention ON)")
    print(f"SERVER LOG     : {SERVER_LOG}")
    print("=" * 60)

    run([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free",
        "--format=csv"
    ])

    print("\nRecent llama-server log lines:")
    print(tail(SERVER_LOG, 30))

except BaseException:
    stop_process(tunnel)
    stop_process(server)
    raise
