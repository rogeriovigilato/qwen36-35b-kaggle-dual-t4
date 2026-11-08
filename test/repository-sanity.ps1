$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

Write-Host "=================================================" -ForegroundColor Cyan
Write-Host "Running Repository Sanity & Leak Prevention Suite" -ForegroundColor Cyan
Write-Host "Target: $RepoRoot" -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan

$global:TotalTests = 0
$global:PassedTests = 0

function Assert-Test([string]$Name, [scriptblock]$Block) {
    $global:TotalTests++
    Write-Host -NoNewline "Testing: $Name ... "
    try {
        & $Block
        $global:PassedTests++
        Write-Host "PASS" -ForegroundColor Green
    }
    catch {
        Write-Host "FAIL: $_" -ForegroundColor Red
        throw $_
    }
}

Assert-Test "No personal paths or secrets in repo files" {
    $forbidden = @(
        [regex]"(?i)[ce]:[/\\](users[/\\]roger|antigravity-kit)",
        [regex]"ghp_[A-Za-z0-9]{20,}",
        [regex]"github_pat_[A-Za-z0-9_]{30,}"
    )

    $files = Get-ChildItem -Path $RepoRoot -Recurse -File | Where-Object {
        $_.FullName -notmatch '[\\/]\.git([\\/]|$)' -and $_.FullName -notmatch '[\\/]test[\\/]'
    }

    foreach ($file in $files) {
        $content = [System.IO.File]::ReadAllText($file.FullName, [System.Text.Encoding]::UTF8)
        foreach ($pattern in $forbidden) {
            if ($pattern.IsMatch($content)) {
                throw "Forbidden pattern found in $($file.FullName)"
            }
        }
    }
}

Assert-Test "kaggle/qwen36_t4x2_server.py has required multi-GPU & hardening flags" {
    $path = Join-Path $RepoRoot "kaggle\qwen36_t4x2_server.py"
    if (-not (Test-Path $path)) { throw "Missing $path" }
    $content = [System.IO.File]::ReadAllText($path, [System.Text.Encoding]::UTF8)

    $required = @(
        "-DCMAKE_CUDA_ARCHITECTURES=75",
        "-DGGML_CUDA=ON",
        "-sm layer",
        "-fa on",
        "262144",
        "/usr/local/bin/cloudflared",
        "Qwen3.6-35B-A3B"
    )

    foreach ($req in $required) {
        if (-not $content.Contains($req)) {
            throw "Server script missing required flag/string: $req"
        }
    }
}

Assert-Test "kaggle/Qwen36_35B_Kaggle_Dual_T4.ipynb is clean valid JSON with GPU metadata" {
    $path = Join-Path $RepoRoot "kaggle\Qwen36_35B_Kaggle_Dual_T4.ipynb"
    if (-not (Test-Path $path)) { throw "Missing $path" }
    $content = [System.IO.File]::ReadAllText($path, [System.Text.Encoding]::UTF8)
    $json = ConvertFrom-Json $content

    if ($json.cells.Count -lt 2) { throw "Notebook should have at least 2 cells" }

    foreach ($cell in $json.cells) {
        if ($cell.cell_type -eq "code") {
            if ($cell.outputs.Count -ne 0) {
                throw "Code cell contains outputs; should be clean before release"
            }
        }
    }

    if (-not $content.Contains("nvidiaTeslaT4")) {
        throw "Notebook metadata missing nvidiaTeslaT4 accelerator tag"
    }
}

Assert-Test "docs/assets/architecture-animated.svg is valid SVG with animations" {
    $path = Join-Path $RepoRoot "docs\assets\architecture-animated.svg"
    if (-not (Test-Path $path)) { throw "Missing $path" }
    $content = [System.IO.File]::ReadAllText($path, [System.Text.Encoding]::UTF8)

    if (-not ($content.Trim().StartsWith("<svg") -and $content.Trim().EndsWith("</svg>"))) {
        throw "SVG is not well-formed root container"
    }
    if (-not $content.Contains("<animate")) {
        throw "SVG missing animations"
    }
    if (-not ($content.Contains("GPU 0: NVIDIA TESLA T4") -and $content.Contains("GPU 1: NVIDIA TESLA T4"))) {
        throw "SVG missing dual T4 visual labels"
    }
}

Assert-Test "README.md and LICENSE have author Rogério Vigilato Rodrigues Filho and sections" {
    $readmePath = Join-Path $RepoRoot "README.md"
    $licensePath = Join-Path $RepoRoot "LICENSE"

    $readme = [System.IO.File]::ReadAllText($readmePath, [System.Text.Encoding]::UTF8)
    $license = [System.IO.File]::ReadAllText($licensePath, [System.Text.Encoding]::UTF8)

    if (-not $readme.Contains("Vigilato Rodrigues Filho")) {
        throw "README missing author Vigilato Rodrigues Filho"
    }
    if (-not $license.Contains("Vigilato Rodrigues Filho")) {
        throw "LICENSE missing author Vigilato Rodrigues Filho"
    }
    if (-not $readme.Contains("Quickstart")) { throw "README missing Quickstart" }
    if (-not $readme.Contains("Dual Tesla T4")) { throw "README missing Dual Tesla T4" }
    if (-not $readme.Contains("Cloudflare Tunnel")) { throw "README missing Cloudflare Tunnel" }
    if (-not $readme.Contains("OpenAI-compatible")) { throw "README missing OpenAI-compatible" }
}

Write-Host "=================================================" -ForegroundColor Cyan
Write-Host "All $global:PassedTests / $global:TotalTests tests PASSED successfully!" -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Cyan
