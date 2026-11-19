// テストランナー（依存なし）: node tests/run.ts
//
//   examples/*.um      `// expect: 行` と出力を照合。wasm32 で実行、ホストのネイティブでも実行。
//                      `// stdin: 行` があれば、その行を標準入力として与える（何回でも書ける。1つにつき1行）。
//                      ネイティブの実行は一時フォルダの中で行うので、writeFile で相対パスに書いたファイルはそこにできる。
//                      wasm32 では、ファイルはメモリ上の入れ物に読み書きする。
//                      AddressSanitizer か valgrind があれば、メモリ検査つきでも実行する
//                      （解放後使用・範囲外アクセス・二重解放・解放漏れを検出）。
//                      arm64-macos はオブジェクト生成まで確認。
//   tests/gpu/*.um     GPU（WebGPU）用のテスト。ここでは wasm32 向けにコンパイルして WGSL を生成し、
//                      CPU だけで実行して `// expect:` と照合する。GPU での実行はブラウザで行う。
//   tests/msl/         MSL の出力を、Metal なしで確かめる。clang++ があれば: tests/gpu と examples の kernel の MSL を
//                      スタブヘッダ（metal_stub.h）つきで C++14 として構文検査し、tests/msl/kernels.um の kernel の MSL を
//                      C++ として実行して、CPU 実行（ネイティブ）とチェックサムを比べる。Metal コンパイラでの確認ではない。
//   tests/io/*.um      標準入力の届き方に左右される動作。pipe_driver.c がパイプに書く（「待たない」設定のパイプ、
//                      1行を2回に分けて書く）。ネイティブだけ。
//   tests/threads/*.um スレッド（ネイティブだけ）。`// expect:` か `// panic:` と照合する。`// sorted` があれば出力を
//                      並べ替えてから比べる。メモリ検査つきと、ThreadSanitizer（データ競合の検出）つきでも実行する。
//   tests/errors/*.um  `// error: コード` の診断がすべて出ることを確認。
//   tests/panics/*.um  `// panic: メッセージ` で停止することを確認。

import { execFileSync, spawnSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { CompileError, CompileErrors } from '../src/diag.ts';
import { build, CLANG, compileMsl, compileToLL, compileWgsl } from '../src/driver.ts';
import { smallMagnitudes } from '../src/codegen.ts';
// @ts-ignore 型定義のないJSモジュール
import { instantiate, UmPanic } from '../runtime/host.mjs';

const root = join(import.meta.dirname, '..');
const tmp = mkdtempSync(join(tmpdir(), 'um-test-'));
let passed = 0;
const failures: string[] = [];

function ok(name: string): void {
  passed++;
  console.log(`ok   ${name}`);
}

/** この環境の clang が wasm32 を出力できるか。できなければ wasm32 のテストは飛ばす。 */
let hasWasm = true;
let skippedWasm = 0;

function fail(name: string, detail: string): void {
  if (!hasWasm && name.includes('wasm32')) {
    skippedWasm++;
    return;
  }
  failures.push(name);
  console.log(`FAIL ${name}\n     ${detail.replace(/\n/g, '\n     ')}`);
}

function directives(src: string, key: string): string[] {
  return src
    .split('\n')
    .filter((l) => l.startsWith(`// ${key}: `))
    .map((l) => l.slice(key.length + 5).trim());
}

/** `// stdin: 行` の並び。`// stdin:` だけの行は空行。前後の空白はそのまま渡す。 */
function stdinLines(src: string): string[] {
  const out: string[] = [];
  for (const l of src.split('\n')) {
    const m = /^\/\/ stdin:(?: (.*))?$/.exec(l.replace(/\r$/, ''));
    if (m) out.push(m[1] ?? '');
  }
  return out;
}

/** 標準入力として与えるテキスト（各行の末尾に改行）。 */
function stdinText(src: string): string {
  return stdinLines(src).map((l) => l + '\n').join('');
}

function files(dir: string): string[] {
  return readdirSync(join(root, dir))
    .filter((f) => f.endsWith('.um'))
    .sort()
    .map((f) => join(dir, f));
}

async function runWasm(src: string, tag: string): Promise<{ out: string[]; panic?: string }> {
  const wasm = join(tmp, `${tag}.wasm`);
  build(compileToLL(src, 'wasm32'), 'wasm32', wasm);
  const out: string[] = [];
  // 標準入力は `// stdin:` の行。ファイルはメモリ上の入れ物（実行ごとに空から始まる）。
  const lines = stdinLines(src);
  const disk = new Map<string, Uint8Array>();
  const exports = await instantiate(readFileSync(wasm), {
    print: (s: string) => out.push(s),
    readLine: () => lines.shift() ?? null,
    inputReady: () => true,
    readFile: (path: string) => disk.get(path) ?? null,
    writeFile: (path: string, bytes: Uint8Array) => {
      disk.set(path, bytes);
      return true;
    },
  });
  try {
    await (exports.main as () => unknown)();
  } catch (e) {
    if (e instanceof UmPanic) return { out, panic: (e as Error).message };
    throw e;
  }
  return { out };
}

function runNative(src: string, tag: string): { out: string[]; panic?: string } {
  const exe = join(tmp, `${tag}.bin`);
  build(compileToLL(src, 'host'), 'host', exe);
  const r = spawnSync(exe, [], { encoding: 'utf8', input: stdinText(src), cwd: tmp });
  const out = r.stdout.split('\n').filter((l) => l !== '');
  return r.status === 101 ? { out, panic: r.stderr.trim() } : { out };
}

type MemCheck = 'asan' | 'valgrind' | 'macos' | null;

/**
 * メモリ検査つきで実行する。解放後使用・範囲外アクセス・二重解放・解放漏れがあれば report に内容が入る。
 * AddressSanitizer があればそれを、なければ valgrind を使う。
 * macOS の AddressSanitizer は解放漏れを調べられないので、解放漏れだけ標準の leaks コマンドで別に調べる（'macos'）。
 */
function runMemChecked(src: string, tag: string, tool: 'asan' | 'valgrind' | 'macos', quiet = false): { out: string[]; report: string } {
  const exe = join(tmp, `${tag}.mem`);
  let r;
  if (tool === 'macos') {
    build(compileToLL(src, 'host', true), 'host', exe, '1', true, quiet);
    r = spawnSync(exe, [], { encoding: 'utf8', input: stdinText(src), cwd: tmp, env: { ...process.env, ASAN_OPTIONS: 'halt_on_error=1' } });
    if (r.status === 0) {
      const plain = join(tmp, `${tag}.leaks`);
      build(compileToLL(src, 'host'), 'host', plain, '1');
      const l = spawnSync('leaks', ['-q', '--atExit', '--', plain], { encoding: 'utf8', input: stdinText(src), cwd: tmp });
      if (l.status !== 0) {
        // leaks は報告を標準出力に書く。プログラム自身の出力のあとに続く。
        const text = l.stdout.slice(l.stdout.search(/^Process /m) >= 0 ? l.stdout.search(/^Process /m) : 0) + l.stderr;
        return { out: r.stdout.split('\n').filter((x) => x !== ''), report: text.trim().split('\n').slice(0, 14).join('\n') || `leaks exited with ${l.status ?? l.signal}` };
      }
    }
  } else if (tool === 'asan') {
    build(compileToLL(src, 'host', true), 'host', exe, '1', true, quiet);
    r = spawnSync(exe, [], { encoding: 'utf8', input: stdinText(src), cwd: tmp, env: { ...process.env, ASAN_OPTIONS: 'detect_leaks=1:halt_on_error=1' } });
  } else {
    build(compileToLL(src, 'host'), 'host', exe, '1');
    r = spawnSync('valgrind', ['-q', '--error-exitcode=97', '--leak-check=full', '--errors-for-leak-kinds=all', exe], { encoding: 'utf8', input: stdinText(src), cwd: tmp });
  }
  const out = r.stdout.split('\n').filter((l) => l !== '');
  const bad = tool !== 'valgrind' ? r.status !== 0 : r.status === 97 || r.signal !== null;
  return { out, report: bad ? r.stderr.trim().split('\n').slice(0, 14).join('\n') : '' };
}

function detectMemCheck(): MemCheck {
  const probe = 'export function main(): i32 { return 0; }';
  try {
    if (!runMemChecked(probe, 'probe_asan', 'asan', true).report) return 'asan';
  } catch {
    // AddressSanitizer のランタイムがない
  }
  try {
    if (process.platform === 'darwin' && !runMemChecked(probe, 'probe_macos', 'macos', true).report) return 'macos';
  } catch {
    // AddressSanitizer も leaks もない
  }
  try {
    if (spawnSync('valgrind', ['--version']).status === 0 && !runMemChecked(probe, 'probe_valgrind', 'valgrind').report) {
      return 'valgrind';
    }
  } catch {
    // valgrind がない
  }
  return null;
}

function same(a: string[], b: string[]): boolean {
  return a.length === b.length && a.every((x, i) => x === b[i]);
}

const hasClang = spawnSync(CLANG, ['--version']).status === 0;
if (!hasClang) {
  console.error('clang が見つかりません。LLVM (clang, wasm-ld) をインストールしてください。');
  process.exit(2);
}

try {
  build(compileToLL('export function main(): i32 { return 0; }', 'wasm32'), 'wasm32', join(tmp, 'probe.wasm'), '2', false, true);
} catch {
  hasWasm = false;
  console.log('note: この clang は wasm32 を出力できないため、wasm32 のテストは飛ばします。');
  console.log('      macOS では `brew install llvm lld` で入れてください（入っていれば wasm32 には自動でそれを使います）。');
}

const memCheck = detectMemCheck();
if (!memCheck) console.log('note: AddressSanitizer も valgrind も leaks も使えないため、メモリ検査つきの実行は省略します');

for (const f of files('examples')) {
  const src = readFileSync(join(root, f), 'utf8');
  const want = directives(src, 'expect');
  const tag = f.replace(/[^a-z0-9]/gi, '_');

  try {
    const w = await runWasm(src, tag);
    if (w.panic) fail(`${f} [wasm32]`, w.panic);
    else if (same(w.out, want)) ok(`${f} [wasm32]`);
    else fail(`${f} [wasm32]`, `expected: ${want.join(' ')}\nactual:   ${w.out.join(' ')}`);
  } catch (e) {
    fail(`${f} [wasm32]`, String(e));
  }

  try {
    const n = runNative(src, tag);
    if (n.panic) fail(`${f} [host]`, n.panic);
    else if (same(n.out, want)) ok(`${f} [host]`);
    else fail(`${f} [host]`, `expected: ${want.join(' ')}\nactual:   ${n.out.join(' ')}`);
  } catch (e) {
    fail(`${f} [host]`, String(e));
  }

  if (memCheck) {
    try {
      const a = runMemChecked(src, tag, memCheck);
      if (a.report) fail(`${f} [host, memory check]`, a.report);
      else if (same(a.out, want)) ok(`${f} [host, memory check]`);
      else fail(`${f} [host, memory check]`, `expected: ${want.join(' ')}\nactual:   ${a.out.join(' ')}`);
    } catch (e) {
      fail(`${f} [host, memory check]`, String(e));
    }
  }

  try {
    const obj = join(tmp, `${tag}.o`);
    if (process.platform === 'darwin') {
      build(compileToLL(src, 'arm64-macos'), 'arm64-macos', obj);
    } else {
      build(compileToLL(src, 'arm64-macos'), 'arm64-macos', obj);
      const desc = execFileSync('file', ['-b', obj], { encoding: 'utf8' });
      if (!/Mach-O 64-bit.*arm64/.test(desc)) throw new Error(`unexpected object format: ${desc.trim()}`);
    }
    ok(`${f} [arm64-macos build]`);
  } catch (e) {
    fail(`${f} [arm64-macos build]`, String(e));
  }
}

for (const f of files('tests/gpu')) {
  const src = readFileSync(join(root, f), 'utf8');
  const want = directives(src, 'expect');
  const tag = f.replace(/[^a-z0-9]/gi, '_');
  try {
    const shaders = compileWgsl(src);
    if (shaders.size === 0) throw new Error('no dispatchable kernel found');
    for (const [name, text] of shaders) {
      // 生成物の形だけの確認（括弧の対応）。WGSL として正しいかはブラウザでの実行で確かめる。
      const depth = [...text].reduce((d, c) => d + (c === '{' ? 1 : c === '}' ? -1 : 0), 0);
      if (depth !== 0 || !text.includes('fn um_main(')) throw new Error(`malformed WGSL for kernel '${name}'`);
    }
    const w = await runWasm(src, tag);
    if (w.panic) fail(`${f} [wasm32, CPU only]`, w.panic);
    else if (same(w.out, want)) ok(`${f} [wasm32, CPU only]`);
    else fail(`${f} [wasm32, CPU only]`, `expected: ${want.join(' ')}\nactual:   ${w.out.join(' ')}`);
  } catch (e) {
    fail(`${f} [wasm32, CPU only]`, String(e));
  }
}

// MSL: Metal コンパイラの代わりに、C++14 として構文検査し、スタブヘッダで実行して CPU と照合する。
{
  const hasCxx = spawnSync('clang++', ['--version']).status === 0;
  if (!hasCxx) {
    console.log('note: clang++ が使えないため、MSL の確認は省略します');
  } else {
    const mslDir = join(root, 'tests', 'msl');
    const out = join(tmp, 'msl');
    mkdirSync(out);
    const cxx = ['-std=c++14', '-ffp-contract=off', '-Wall', '-Wextra', '-Wno-unknown-attributes', '-Wno-unknown-pragmas', '-Wno-unused-parameter', '-I', mslDir];

    // 1. tests/gpu と examples の kernel の MSL が、スタブつきの C++14 として通ること
    for (const f of [...files('tests/gpu'), ...files('examples')]) {
      const src = readFileSync(join(root, f), 'utf8');
      const tag = f.replace(/[^a-z0-9]/gi, '_');
      let shaders: Map<string, string>;
      try {
        shaders = compileMsl(src);
      } catch (e) {
        fail(`${f} [msl syntax]`, String(e));
        continue;
      }
      for (const [name, text] of shaders) {
        try {
          const base = join(out, `${tag}_${name}`);
          writeFileSync(`${base}.metal`, text);
          writeFileSync(`${base}.cpp`, `#include <cmath>\n#include <cstdint>\n#include "metal_stub.h"\n#include "${base}.metal"\n`);
          const r = spawnSync('clang++', [...cxx, '-Werror', '-fsyntax-only', `${base}.cpp`], { encoding: 'utf8' });
          if (r.status === 0) ok(`${f} kernel ${name} [msl syntax, C++14 + stub]`);
          else fail(`${f} kernel ${name} [msl syntax, C++14 + stub]`, r.stderr.split('\n').slice(0, 14).join('\n'));
        } catch (e) {
          fail(`${f} kernel ${name} [msl syntax]`, String(e));
        }
      }
    }

    // 2. tests/msl/kernels.um: ネイティブ（CPU）の出力と、MSL を C++ として実行した出力が一致すること
    const f = 'tests/msl/kernels.um';
    try {
      const src = readFileSync(join(root, f), 'utf8');
      const cpu = runNative(src, 'msl_kernels');
      if (cpu.panic) throw new Error(cpu.panic);
      if (cpu.out.length === 0) throw new Error('no output from the CPU run');
      const dir = join(out, 'kernels');
      mkdirSync(dir);
      for (const [name, text] of compileMsl(src)) writeFileSync(join(dir, `${name}.metal`), text);
      const exe = join(out, 'harness');
      const c = spawnSync('clang++', [...cxx, '-O1', '-I', dir, join(mslDir, 'harness.cpp'), '-o', exe], { encoding: 'utf8' });
      if (c.status !== 0) throw new Error(c.stderr.split('\n').slice(0, 20).join('\n'));
      const r = spawnSync(exe, [], { encoding: 'utf8' });
      const msl = r.stdout.split('\n').filter((l) => l !== '');
      if (same(msl, cpu.out)) ok(`${f} [msl as C++ vs CPU: ${cpu.out.length} checksums]`);
      else fail(`${f} [msl as C++ vs CPU]`, `cpu: ${cpu.out.join(' ')}\nmsl: ${msl.join(' ')}`);
    } catch (e) {
      fail(`${f} [msl as C++ vs CPU]`, String(e));
    }
  }
}

// f8 / f4 の値の表が、コンパイラ（定数の変換）とランタイム（実行時の変換）で同じであること。
{
  const header = readFileSync(join(root, 'runtime/smallfloat.h'), 'utf8');
  for (const bits of [8, 4] as const) {
    const body = header.slice(header.indexOf(`um_f${bits}_mag[`)).split('{')[1].split('}')[0];
    const fromHeader = body.split(',').map((x) => Number(x.trim()));
    if (same(fromHeader.map(String), smallMagnitudes(bits).map(String))) ok(`f${bits} table matches the runtime`);
    else fail(`f${bits} table matches the runtime`, 'runtime/smallfloat.h and smallMagnitudes() differ');
  }
}

// 複数ファイルのプログラム。tests/modules/basic は実行し、tests/modules/errors/<名前>/main.um はエラーを確かめる。
{
  const f = 'tests/modules/basic/main.um';
  const path = join(root, f);
  const src = readFileSync(path, 'utf8');
  const want = directives(src, 'expect');
  for (const target of ['wasm32', 'host'] as const) {
    try {
      const exe = join(tmp, `modules_basic.${target === 'wasm32' ? 'wasm' : 'bin'}`);
      build(compileToLL(src, target, false, path), target, exe);
      let out: string[];
      if (target === 'wasm32') {
        out = [];
        const exports = await instantiate(readFileSync(exe), (s: string) => out.push(...s.split('\n')));
        (exports.main as () => number)();
        out = out.filter((l) => l !== '');
      } else {
        out = spawnSync(exe, [], { encoding: 'utf8' }).stdout.split('\n').filter((l) => l !== '');
      }
      if (same(out, want)) ok(`${f} [${target}]`);
      else fail(`${f} [${target}]`, `expected: ${want.join(' ')}\nactual:   ${out.join(' ')}`);
    } catch (e) {
      fail(`${f} [${target}]`, String(e));
    }
  }
  for (const name of readdirSync(join(root, 'tests/modules/errors')).sort()) {
    const ef = `tests/modules/errors/${name}/main.um`;
    const epath = join(root, ef);
    const esrc = readFileSync(epath, 'utf8');
    const ewant = directives(esrc, 'error');
    const wantFile = directives(esrc, 'file')[0];
    try {
      compileToLL(esrc, 'wasm32', false, epath);
      fail(ef, `compiled without error; expected: ${ewant.join(', ')}`);
    } catch (e) {
      const errs = e instanceof CompileErrors ? e.errors : e instanceof CompileError ? [e] : null;
      if (!errs) {
        fail(ef, String(e));
        continue;
      }
      const got = errs.map((x) => x.code);
      const fileOk = !wantFile || errs.every((x) => x.pos.file === wantFile);
      if (got.length === ewant.length && ewant.every((c) => got.includes(c)) && fileOk) ok(ef);
      else fail(ef, `expected: ${ewant.join(', ')}${wantFile ? ` in ${wantFile}` : ''}\nactual:   ${errs.map((x) => `${x.code} in ${x.pos.file ?? 'main'} (${x.message})`).join(', ')}`);
    }
  }
}

for (const f of files('tests/errors')) {
  const src = readFileSync(join(root, f), 'utf8');
  const want = directives(src, 'error');
  // 既定は wasm32。ターゲットによって変わるエラーは `// target: host` で指定する。
  const target = directives(src, 'target')[0] === 'host' ? 'host' : 'wasm32';
  try {
    compileToLL(src, target);
    fail(f, `compiled without error; expected: ${want.join(', ')}`);
  } catch (e) {
    const errs = e instanceof CompileErrors ? e.errors : e instanceof CompileError ? [e] : null;
    if (!errs) {
      fail(f, String(e));
      continue;
    }
    const got = errs.map((x) => x.code);
    const missing = want.filter((c) => !got.includes(c));
    if (missing.length === 0 && got.length === want.length) ok(f);
    else fail(f, `expected: ${want.join(', ')}\nactual:   ${errs.map((x) => `${x.code} (${x.message})`).join(', ')}`);
  }
}

// 標準入力の届き方に左右される動作（ネイティブだけ）。tests/io/pipe_driver.c が、ファイル名と同じやり方で
// パイプに書く: nonblocking.um = 「待たない」設定のパイプ、partial.um = 1行を2回に分けて書く。
if (process.platform !== 'win32') {
  const driver = join(tmp, 'pipe_driver');
  let hasDriver = true;
  try {
    execFileSync(CLANG, ['-O1', '-o', driver, join(root, 'tests/io/pipe_driver.c')], { stdio: ['ignore', 'ignore', 'pipe'] });
  } catch (e) {
    hasDriver = false;
    fail('tests/io/pipe_driver.c [host]', String(e));
  }
  for (const f of hasDriver ? files('tests/io') : []) {
    const src = readFileSync(join(root, f), 'utf8');
    const want = directives(src, 'expect');
    const mode = f.replace(/^.*[\\/]/, '').replace(/\.um$/, '');
    try {
      const exe = join(tmp, `io_${mode}.bin`);
      build(compileToLL(src, 'host'), 'host', exe);
      const r = spawnSync(driver, [mode, exe], { encoding: 'utf8', cwd: tmp, timeout: 60000 });
      const out = r.stdout.split('\n').filter((l) => l !== '');
      if (r.status === 0 && same(out, want)) ok(`${f} [host, piped stdin]`);
      else fail(`${f} [host, piped stdin]`, `expected: ${want.join(' ')}\nactual:   ${out.join(' ')} (exit ${r.status ?? r.signal})`);
    } catch (e) {
      fail(`${f} [host, piped stdin]`, String(e));
    }
  }
}

// 後ろで宣言した定数を次々に使う長い並びでも、コンパイラが止まらないこと（再帰が深くならない）。
{
  const name = 'forward chain of 3000 constants';
  const n = 3000;
  let src = '';
  for (let i = 0; i < n; i++) src += `const C${i}: i32 = C${i + 1} + 1;\n`;
  src += `const C${n}: i32 = 0;\nexport function main(): i32 { print(C0); return 0; }\n`;
  try {
    const r = runNative(src, 'const_chain');
    if (same(r.out, [String(n)])) ok(`${name} [host]`);
    else fail(`${name} [host]`, `expected: ${n}\nactual:   ${r.out.join(' ')}`);
  } catch (e) {
    fail(`${name} [host]`, String(e));
  }
  // 同じ長さの循環は、あふれずに `const-cycle` になる。
  let cyc = '';
  for (let i = 0; i < n; i++) cyc += `const C${i}: i32 = C${(i + 1) % n} + 1;\n`;
  cyc += 'export function main(): i32 { return C0; }\n';
  try {
    compileToLL(cyc, 'host');
    fail('cycle of 3000 constants', 'compiled without error; expected: const-cycle');
  } catch (e) {
    const errs = e instanceof CompileErrors ? e.errors : e instanceof CompileError ? [e] : null;
    if (errs && errs.length === 1 && errs[0].code === 'const-cycle') ok('cycle of 3000 constants');
    else fail('cycle of 3000 constants', errs ? errs.map((x) => `${x.code} (${x.message})`).join(', ') : String(e));
  }
}

// wasm のホスト（runtime/host.mjs）の入出力の口。コンパイラを通さず、口をそのまま export する小さな wasm で確かめる
// （wasm32 を出力できない環境でも実行できる）。
{
  const name = 'runtime/host.mjs input and file hooks';
  try {
    const str = (s: string) => [s.length, ...Buffer.from(s)];
    const section = (id: number, body: number[]) => [id, body.length, ...body];
    const I32 = 0x7f;
    const imports = ['um_host_input_ready', 'um_host_write_file', 'um_host_read_file'];
    const bytes = Uint8Array.from([
      0, 0x61, 0x73, 0x6d, 1, 0, 0, 0,
      // 型: () -> i32、(i32, i32, i32, i32) -> i32、(i32, i32) -> i32
      ...section(1, [3, 0x60, 0, 1, I32, 0x60, 4, I32, I32, I32, I32, 1, I32, 0x60, 2, I32, I32, 1, I32]),
      ...section(2, [imports.length, ...imports.flatMap((n, i) => [...str('env'), ...str(n), 0, i])]),
      ...section(5, [1, 0, 1]),
      ...section(7, [imports.length + 1, ...str('memory'), 2, 0, ...imports.flatMap((n, i) => [...str(n), 0, i])]),
    ]);
    const problems: string[] = [];
    // 口を渡さないとき: 入力は終わっているので inputReady は true（ネイティブの入力の終わりと同じ）。
    const bare = await instantiate(bytes, { gpu: false });
    if ((bare.um_host_input_ready as () => number)() !== 1) problems.push('inputReady() without hooks must be true');
    const seen: string[] = [];
    const hooked = await instantiate(bytes, {
      gpu: false,
      readLine: () => null,
      readFile: (path: string) => (seen.push(`read ${path}`), new Uint8Array(3)),
      writeFile: (path: string) => (seen.push(`write ${path}`), true),
    });
    if ((hooked.um_host_input_ready as () => number)() !== 1) problems.push('inputReady() with only a readLine hook must be true');
    const memory = new Uint8Array((hooked.memory as WebAssembly.Memory).buffer);
    const write = hooked.um_host_write_file as (p: number, n: number, d: number, dn: number) => number;
    const read = hooked.um_host_read_file as (p: number, n: number) => number;
    // UTF-8 として読めない名前は、口を呼ばずに失敗にする（別の名前に化けたファイルを開かない）。
    memory.set([0xff, 0x61], 16);
    if (write(16, 2, 0, 1) !== 0) problems.push('writeFile with a non-UTF-8 name must fail');
    if (read(16, 2) !== -1) problems.push('readFile with a non-UTF-8 name must fail');
    memory.set([0x61, 0, 0x62], 16);
    if (write(16, 3, 0, 1) !== 0 || read(16, 3) !== -1) problems.push('a name with a NUL byte must fail');
    memory.set(Buffer.from('é.bin'), 16);
    if (write(16, 6, 0, 1) !== 1 || read(16, 6) !== 3) problems.push('a UTF-8 name must reach the hooks');
    if (!same(seen, ['write é.bin', 'read é.bin'])) problems.push(`hooks saw: ${seen.join(', ')}`);
    if (problems.length === 0) ok(name);
    else fail(name, problems.join('\n'));
  } catch (e) {
    fail(name, String(e));
  }
}

// SIMD（ネイティブ）。macOS の arm64 では NEON の sdot を使う経路と、汎用の経路（UM_GENERIC_SIMD=1。
// wasm32 やほかの CPU と同じ LLVM のベクトル命令）の両方で実行して、同じ結果になることを確かめる。
for (const f of files('tests/simd')) {
  const src = readFileSync(join(root, f), 'utf8');
  const want = directives(src, 'expect');
  const tag = f.replace(/[^a-z0-9]/gi, '_');
  for (const generic of [false, true]) {
    const name = `${f} [host${generic ? ', generic SIMD' : ''}]`;
    try {
      if (generic) process.env.UM_GENERIC_SIMD = '1';
      const n = runNative(src, `${tag}${generic ? '_g' : ''}`);
      if (n.panic) fail(name, n.panic);
      else if (same(n.out, want)) ok(name);
      else fail(name, `expected: ${want.join(' ')}\nactual:   ${n.out.join(' ')}`);
    } catch (e) {
      fail(name, String(e));
    } finally {
      delete process.env.UM_GENERIC_SIMD;
    }
  }
}

// スレッド（ネイティブだけ。wasm32 にはスレッドがなく、合流の時点でその場で実行するので、待ち合わせる形は終わらない）。
{
  // ThreadSanitizer が使えるか。
  let tsan = false;
  try {
    const probe = join(tmp, 'probe_tsan');
    build(compileToLL('export function main(): i32 { return 0; }', 'host', 'thread'), 'host', probe, '1', 'thread', true);
    tsan = spawnSync(probe, []).status === 0;
  } catch {
    // ThreadSanitizer のランタイムがない
  }
  if (!tsan) console.log('note: ThreadSanitizer が使えないため、tests/threads の競合検査は省略します');
  for (const f of files('tests/threads')) {
    const src = readFileSync(join(root, f), 'utf8');
    const sorted = src.split('\n').some((l) => l.trim() === '// sorted');
    const norm = (xs: string[]) => (sorted ? [...xs].sort() : xs);
    const want = norm(directives(src, 'expect'));
    const wantPanic = directives(src, 'panic')[0];
    const tag = f.replace(/[^a-z0-9]/gi, '_');
    const judge = (name: string, out: string[], panic: string | undefined) => {
      if (wantPanic !== undefined) {
        if (panic === `panic: ${wantPanic}`) ok(name);
        else fail(name, `expected: panic: ${wantPanic}\nactual:   ${panic ?? 'no panic'}`);
      } else if (panic) fail(name, panic);
      else if (same(norm(out), want)) ok(name);
      else fail(name, `expected: ${want.join(' ').slice(0, 300)}\nactual:   ${norm(out).join(' ').slice(0, 300)}`);
    };
    try {
      const n = runNative(src, tag);
      judge(`${f} [host]`, n.out, n.panic);
    } catch (e) {
      fail(`${f} [host]`, String(e));
    }
    if (memCheck && wantPanic === undefined) {
      try {
        const a = runMemChecked(src, tag, memCheck);
        if (a.report) fail(`${f} [host, memory check]`, a.report);
        else judge(`${f} [host, memory check]`, a.out, undefined);
      } catch (e) {
        fail(`${f} [host, memory check]`, String(e));
      }
    }
    if (tsan) {
      try {
        const exe = join(tmp, `${tag}.tsan`);
        build(compileToLL(src, 'host', 'thread'), 'host', exe, '1', 'thread');
        const r = spawnSync(exe, [], { encoding: 'utf8', cwd: tmp, env: { ...process.env, TSAN_OPTIONS: 'halt_on_error=1:exitcode=66' }, timeout: 300000 });
        const out = r.stdout.split('\n').filter((l) => l !== '');
        if (r.status === 66 || /ThreadSanitizer/.test(r.stderr)) fail(`${f} [host, ThreadSanitizer]`, r.stderr.split('\n').slice(0, 20).join('\n'));
        else judge(`${f} [host, ThreadSanitizer]`, out, r.status === 101 ? r.stderr.trim() : undefined);
      } catch (e) {
        fail(`${f} [host, ThreadSanitizer]`, String(e));
      }
    }
  }
}

// RacyBuffer の clone() は要素ごとに不可分に読む。memcpy だと書き手のスレッドと競合するが、
// この Mac の ThreadSanitizer はそれを見つけないので、出力した LLVM IR で確かめる。
{
  const name = 'RacyBuffer.clone() reads elements atomically';
  try {
    const ll = compileToLL(
      'function f(b: RacyBuffer<u64>): u64 { const c = b.clone(); return c[0]; }\n' +
        'export function main(): i32 { const b = new RacyBuffer<u64>(4); print(f(b)); return 0; }\n',
      'host',
    );
    const at = ll.indexOf('define internal i64 @um.f(');
    const body = at < 0 ? '' : ll.slice(at, ll.indexOf('\n}\n', at));
    if (/@llvm\.memcpy/.test(body) || !/load atomic i64, ptr \S+ monotonic/.test(body)) fail(name, body.slice(0, 1500));
    else ok(name);
  } catch (e) {
    fail(name, String(e));
  }
}

// Metal（macOS の GPU）。試験中の経路なので、UM_METAL=1 を付けたときと同じ設定で作る。
// Mac では実際に GPU で実行して CPU の結果と比べる。それ以外では arm64 向けに作れることだけ確かめる。
for (const f of files('tests/metal')) {
  const src = readFileSync(join(root, f), 'utf8');
  const want = directives(src, 'expect');
  const tag = f.replace(/[^a-z0-9]/gi, '_');
  process.env.UM_METAL = '1';
  try {
    if (process.platform === 'darwin') {
      const n = runNative(src, `${tag}_metal`);
      if (n.panic) fail(`${f} [metal]`, n.panic);
      else if (same(n.out, want)) ok(`${f} [metal]`);
      else fail(`${f} [metal]`, `expected: ${want.join(' ')}\nactual:   ${n.out.join(' ')}`);
    } else {
      build(compileToLL(src, 'arm64-macos'), 'arm64-macos', join(tmp, `${tag}_metal.o`), '2', false, true);
      ok(`${f} [arm64-macos build with Metal calls]`);
    }
  } catch (e) {
    fail(`${f} [metal]`, String(e));
  } finally {
    delete process.env.UM_METAL;
  }
}

for (const f of files('tests/panics')) {
  const src = readFileSync(join(root, f), 'utf8');
  const want = `panic: ${directives(src, 'panic')[0]}`;
  const tag = f.replace(/[^a-z0-9]/gi, '_');
  // wasm32 と host は別々に確かめる（片方が作れない環境でも、もう片方は実行する）。
  try {
    const w = await runWasm(src, tag);
    if (w.panic === want) ok(`${f} [wasm32]`);
    else fail(`${f} [wasm32]`, `expected: ${want}\nactual:   ${w.panic ?? 'no panic'}`);
  } catch (e) {
    fail(`${f} [wasm32]`, String(e));
  }
  try {
    const n = runNative(src, tag);
    if (n.panic === want) ok(`${f} [host]`);
    else fail(`${f} [host]`, `expected: ${want}\nactual:   ${n.panic ?? 'no panic'}`);
  } catch (e) {
    fail(`${f} [host]`, String(e));
  }
}

rmSync(tmp, { recursive: true, force: true });
if (skippedWasm > 0) console.log(`\nwasm32 のテスト ${skippedWasm} 件は飛ばしました（この clang は wasm32 を出力できません）`);
console.log(`\n${passed} passed, ${failures.length} failed`);
process.exit(failures.length === 0 ? 0 : 1);
