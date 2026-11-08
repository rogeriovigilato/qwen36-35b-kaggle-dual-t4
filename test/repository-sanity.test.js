import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '..');

test('Repository Sanity & Leak Prevention', async (t) => {
  await t.test('No personal paths or sensitive secrets leaked in repository', () => {
    const forbiddenPatterns = [
      /[ce]:[/\\](users[/\\]roger|antigravity-kit)/i,
      /ghp_[A-Za-z0-9]{20,}/,
      /github_pat_[A-Za-z0-9_]{30,}/,
      /eyJh[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}/
    ];

    function scanDir(dir) {
      const entries = fs.readdirSync(dir, { withFileTypes: true });
      for (const entry of entries) {
        if (entry.name === '.git' || entry.name === 'node_modules' || entry.name === 'test') continue;
        const fullPath = path.join(dir, entry.name);
        if (entry.isDirectory()) {
          scanDir(fullPath);
        } else if (entry.isFile()) {
          const content = fs.readFileSync(fullPath, 'utf8');
          for (const pattern of forbiddenPatterns) {
            assert.strictEqual(
              pattern.test(content),
              false,
              `Forbidden pattern ${pattern} found in ${path.relative(REPO_ROOT, fullPath)}`
            );
          }
        }
      }
    }

    scanDir(REPO_ROOT);
  });

  await t.test('kaggle/qwen36_t4x2_server.py contains required multi-GPU & hardening flags', () => {
    const scriptPath = path.join(REPO_ROOT, 'kaggle', 'qwen36_t4x2_server.py');
    assert.strictEqual(fs.existsSync(scriptPath), true, 'Server script must exist');
    const content = fs.readFileSync(scriptPath, 'utf8');

    assert.ok(content.includes('CMAKE_CUDA_ARCHITECTURES=75'), 'Must compile for Turing sm_75');
    assert.ok(content.includes('-DGGML_CUDA=ON'), 'Must compile with CUDA enabled');
    assert.ok(content.includes('-sm layer'), 'Must use layer split for dual GPUs');
    assert.ok(content.includes('-fa on'), 'Must enable FlashAttention');
    assert.ok(content.includes('262144'), 'Must support 256K context');
    assert.ok(content.includes('/usr/local/bin/cloudflared'), 'Must execute cloudflared from /usr/local/bin to avoid Kaggle noexec');
    assert.ok(content.includes('Qwen3.6-35B-A3B'), 'Must target Qwen3.6-35B-A3B');
  });

  await t.test('kaggle/Qwen36_35B_Kaggle_Dual_T4.ipynb is clean valid JSON with metadata', () => {
    const nbPath = path.join(REPO_ROOT, 'kaggle', 'Qwen36_35B_Kaggle_Dual_T4.ipynb');
    assert.strictEqual(fs.existsSync(nbPath), true, 'Notebook must exist');
    const content = fs.readFileSync(nbPath, 'utf8');
    const parsed = JSON.parse(content);

    assert.ok(Array.isArray(parsed.cells), 'Notebook must have cells array');
    assert.ok(parsed.cells.length >= 2, 'Notebook must have at least 2 cells');

    // Outputs must be clean (no huge pre-recorded execution dumps)
    for (const cell of parsed.cells) {
      if (cell.cell_type === 'code') {
        assert.deepStrictEqual(cell.outputs, [], 'Code cells should have empty outputs in repo');
        assert.strictEqual(cell.execution_count, null, 'Execution count should be null');
      }
    }

    assert.ok(
      content.includes('nvidiaTeslaT4') || content.includes('gpu'),
      'Notebook metadata should specify GPU accelerator'
    );
  });

  await t.test('docs/assets/architecture-animated.svg is well-formed SVG with animations', () => {
    const svgPath = path.join(REPO_ROOT, 'docs', 'assets', 'architecture-animated.svg');
    assert.strictEqual(fs.existsSync(svgPath), true, 'Architecture SVG must exist');
    const content = fs.readFileSync(svgPath, 'utf8');

    assert.ok(content.startsWith('<svg') && content.endsWith('</svg>'), 'Must be valid SVG container');
    assert.ok(content.includes('<animate'), 'Must contain SVG animations');
    assert.ok(content.includes('GPU 0: NVIDIA TESLA T4') && content.includes('GPU 1: NVIDIA TESLA T4'), 'Must illustrate dual T4 GPUs');
    assert.ok(content.includes('256K Context'), 'Must highlight 256K context capability');
  });

  await t.test('README.md and LICENSE have proper author attribution and complete guides', () => {
    const readmePath = path.join(REPO_ROOT, 'README.md');
    const licensePath = path.join(REPO_ROOT, 'LICENSE');

    assert.strictEqual(fs.existsSync(readmePath), true, 'README.md must exist');
    assert.strictEqual(fs.existsSync(licensePath), true, 'LICENSE must exist');

    const readme = fs.readFileSync(readmePath, 'utf8');
    const license = fs.readFileSync(licensePath, 'utf8');

    assert.ok(readme.includes('Rogério Vigilato Rodrigues Filho'), 'README must credit author');
    assert.ok(license.includes('Rogério Vigilato Rodrigues Filho'), 'LICENSE must attribute author');
    assert.ok(readme.includes('Quickstart'), 'README must contain Quickstart');
    assert.ok(readme.includes('Dual Tesla T4'), 'README must specify Dual Tesla T4');
    assert.ok(readme.includes('Cloudflare Tunnel'), 'README must detail Cloudflare Tunnel');
    assert.ok(readme.includes('OpenAI-compatible'), 'README must detail OpenAI compatibility');
    assert.ok(readme.includes('curl'), 'README must include curl example');
    assert.ok(readme.includes('openai'), 'README must include Python OpenAI client example');
  });
});
