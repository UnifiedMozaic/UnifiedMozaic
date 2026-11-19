// GPU（WebGPU）テスト用のページを作る: node tools/build-gpu-test.ts [出力先.html]
//
// tests/gpu/*.um を wasm にコンパイルし、ランタイムと一緒に1つの HTML に埋め込む。
// このページを WebGPU 対応のブラウザで開くと、各テストを GPU で実行して `// expect:` と照合する。
// 外部への通信はしない。

import { mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { build, compileToLL } from '../src/driver.ts';

const root = join(import.meta.dirname, '..');
const out = process.argv[2] ?? join(root, 'out', 'gpu-test.html');
const tmp = mkdtempSync(join(tmpdir(), 'um-gpu-'));

const tests = readdirSync(join(root, 'tests', 'gpu'))
  .filter((f) => f.endsWith('.um'))
  .sort()
  .map((f) => {
    const src = readFileSync(join(root, 'tests', 'gpu', f), 'utf8');
    const wasm = join(tmp, f.replace(/\.um$/, '.wasm'));
    build(compileToLL(src, 'wasm32'), 'wasm32', wasm);
    const expect = src
      .split('\n')
      .filter((l) => l.startsWith('// expect: '))
      .map((l) => l.slice('// expect: '.length).trim());
    return { name: f, expect, wasm: readFileSync(wasm).toString('base64') };
  });
rmSync(tmp, { recursive: true, force: true });

// ランタイムはモジュールの export を外して、そのままページのスクリプトに入れる。
const host = readFileSync(join(root, 'runtime', 'host.mjs'), 'utf8').replace(/^export /gm, '');
// </script> がデータの中に現れてもスクリプトが途中で終わらないようにする。
const data = JSON.stringify(tests).replace(/</g, '\\u003c');

const page = `<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>UnifiedMozaic GPU テスト</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 16px; line-height: 1.6; }
  h1 { font-size: 20px; }
  pre { background: #f4f4f4; padding: 12px; overflow-x: auto; white-space: pre-wrap; word-break: break-all; font-size: 13px; }
  button { font-size: 16px; padding: 8px 16px; }
  .ok { color: #0a7a2f; font-weight: 600; }
  .ng { color: #b00020; font-weight: 600; }
  @media (prefers-color-scheme: dark) { body { background: #111; color: #eee; } pre { background: #222; } }
</style>
</head>
<body>
<h1>UnifiedMozaic GPU テスト</h1>
<p>kernel を CPU と GPU（WebGPU）の両方で実行して、結果が一致するかを確かめます。外部への通信はしません。</p>
<p id="summary">実行中…</p>
<button id="copy" hidden>結果をコピー</button>
<pre id="log"></pre>
<script type="module">
${host}

const TESTS = ${data};
const log = document.getElementById('log');
const lines = [];
const say = (s) => { lines.push(s); log.textContent = lines.join('\\n'); };
const decode = (b64) => Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));

async function run(test, useGpu) {
  const out = [];
  const exports = await instantiate(decode(test.wasm), { print: (s) => out.push(s), gpu: useGpu });
  const start = performance.now();
  await exports.main(useGpu ? 1 : 0);
  return { out, ms: performance.now() - start, gpu: exports.um_gpu };
}

let failed = 0;
say('ブラウザ: ' + navigator.userAgent);
say('WebGPU: ' + (navigator.gpu ? 'あり' : 'なし') + ' / JSPI: ' + (jspi ? 'あり' : 'なし'));
try {
  const adapter = navigator.gpu ? await navigator.gpu.requestAdapter() : null;
  const info = adapter?.info;
  say('GPU: ' + (adapter ? [info?.vendor, info?.architecture, info?.device, info?.description].filter(Boolean).join(' / ') || '(名前は非公開)' : '取得できず'));
} catch (e) {
  say('GPU: 取得時にエラー: ' + e.message);
}
say('');

for (const test of TESTS) {
  for (const useGpu of [false, true]) {
    const label = test.name + (useGpu ? ' [GPU]' : ' [CPUのみ]');
    try {
      const r = await run(test, useGpu);
      if (useGpu && !r.gpu) {
        failed++;
        say('未実行 ' + label + ': この環境では GPU を使えません（WebGPU か JSPI がない）');
        continue;
      }
      const same = r.out.length === test.expect.length && r.out.every((v, i) => v === test.expect[i]);
      if (!same) failed++;
      say((same ? 'OK   ' : 'NG   ') + label + ' ' + r.ms.toFixed(1) + 'ms' + (useGpu ? ' (GPU実行 ' + r.gpu.dispatches() + '回)' : ''));
      if (!same) {
        say('  期待: ' + test.expect.join(' '));
        say('  実際: ' + r.out.join(' '));
      }
    } catch (e) {
      failed++;
      say('NG   ' + label + ': ' + (e && e.message ? e.message : e));
    }
  }
}

const summary = document.getElementById('summary');
summary.textContent = failed === 0 ? 'すべて成功しました。' : failed + ' 件が失敗または未実行です。下の結果を送ってください。';
summary.className = failed === 0 ? 'ok' : 'ng';
const copy = document.getElementById('copy');
copy.hidden = false;
copy.onclick = async () => {
  try {
    await navigator.clipboard.writeText(lines.join('\\n'));
    copy.textContent = 'コピーしました';
  } catch {
    copy.textContent = 'コピーできませんでした（長押しで選択してください）';
  }
};
</script>
</body>
</html>
`;

writeFileSync(out, page);
console.log(`${out} (${tests.length} tests, ${(page.length / 1024).toFixed(0)} KiB)`);
