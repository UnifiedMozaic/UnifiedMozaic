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
//   examples/wasm/*.um WebAssembly の組み込み（wasmtime。ネイティブだけ）。`// expect:` と照合する。
//   tests/wasm/*.um    同じく。`// expect:` か `// panic:` と照合する。wasmtime の C API が見つからなければ飛ばす。
//   examples/media/*.um 動画・音声の組み込み（FFmpeg。ネイティブだけ）。`// expect:` と照合する。
//   tests/media/*.um   同じく。`// expect:` か `// panic:` と照合する。素材は ffmpeg のコマンドで一時フォルダに作る
//                      （makeMediaClips）。FFmpeg のライブラリか ffmpeg のコマンドが見つからなければ飛ばす。

import { execFileSync, spawn, spawnSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { CompileError, CompileErrors } from '../src/diag.ts';
import { build, CLANG, compileMsl, compileToLL, compileWgsl, exePath, ffmpegAvailable, wasmtimeAvailable } from '../src/driver.ts';
import { smallMagnitudes } from '../src/codegen.ts';
// @ts-ignore 型定義のないJSモジュール
import { instantiate, UmPanic } from '../runtime/host.mjs';

const root = join(import.meta.dirname, '..');
const tmp = mkdtempSync(join(tmpdir(), 'um-test-'));
const windows = process.platform === 'win32';
/** clang と同じ場所の clang++（Windows の LLVM は PATH に入っていないことがある）。 */
const CLANGXX = CLANG.replace(/clang(\.exe)?$/i, (_m, ext: string | undefined) => `clang++${ext ?? ''}`);

/**
 * AddressSanitizer つきの実行に渡す環境変数。Windows の AddressSanitizer は解放漏れを調べられない
 * （detect_leaks を付けると止まる）。また、ランタイムの DLL を clang のリソースフォルダから読ませる。
 */
function asanEnv(options: string): NodeJS.ProcessEnv {
  if (!windows) return { ...process.env, ASAN_OPTIONS: options };
  const env: NodeJS.ProcessEnv = { ...process.env, ASAN_OPTIONS: options.replace(/(^|:)detect_leaks=1/, '').replace(/^:/, '') };
  try {
    const res = execFileSync(CLANG, ['-print-resource-dir'], { encoding: 'utf8' }).trim();
    const key = Object.keys(env).find((k) => k.toUpperCase() === 'PATH') ?? 'PATH';
    env[key] = `${join(res, 'lib', 'windows')};${env[key] ?? ''}`;
  } catch {
    // clang のリソースフォルダが分からない。DLL が PATH にあれば動く。
  }
  return env;
}
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
    // fileSize / readFileRange / appendFile は、readFile / writeFile の口で代わりに行われる。
    eprint: () => {},
  });
  try {
    await (exports.main as () => unknown)();
  } catch (e) {
    if (e instanceof UmPanic) return { out, panic: (e as Error).message };
    throw e;
  }
  return { out };
}

function runNative(src: string, tag: string, file?: string): { out: string[]; panic?: string } {
  const exe = join(tmp, `${tag}.bin`);
  // file はソースの場所（import を解決するため。なければ import は使えない）。
  build(compileToLL(src, 'host', false, file), 'host', exe);
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
    r = spawnSync(exe, [], { encoding: 'utf8', input: stdinText(src), cwd: tmp, env: asanEnv('detect_leaks=1:halt_on_error=1') });
  } else {
    build(compileToLL(src, 'host'), 'host', exe, '1');
    r = spawnSync('valgrind', ['-q', '--error-exitcode=97', '--leak-check=full', '--errors-for-leak-kinds=all', exe], { encoding: 'utf8', input: stdinText(src), cwd: tmp });
  }
  const out = r.stdout.split('\n').filter((l) => l !== '');
  const bad = tool !== 'valgrind' ? r.status !== 0 : r.status === 97 || r.signal !== null;
  // 何も書かずに失敗した（ランタイムを読めなかったなど）ときも、空でない報告にする。
  const report = bad ? r.stderr.trim().split('\n').slice(0, 14).join('\n') || `exited with ${r.status ?? r.signal}` : '';
  return { out, report };
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
else if (memCheck === 'asan' && windows) console.log('note: Windows の AddressSanitizer は解放漏れを調べられないため、メモリ検査では解放漏れを確かめません');

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
    build(compileToLL(src, 'arm64-macos'), 'arm64-macos', obj);
    if (process.platform !== 'darwin') {
      // Mach-O の 64 ビット（0xfeedfacf）で、CPU が arm64（0x0100000c）のオブジェクトか。
      const head = readFileSync(obj).subarray(0, 8);
      if (head.length < 8 || head.readUInt32LE(0) !== 0xfeedfacf || head.readUInt32LE(4) !== 0x0100000c) {
        throw new Error(`unexpected object format: ${head.toString('hex')}`);
      }
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
  const hasCxx = spawnSync(CLANGXX, ['--version']).status === 0;
  if (!hasCxx) {
    console.log('note: clang++ が使えないため、MSL の確認は省略します');
  } else {
    const mslDir = join(root, 'tests', 'msl');
    const out = join(tmp, 'msl');
    mkdirSync(out);
    // MSL の long は 64 ビット。Windows の C++ の long は 32 ビットなので、stub の 64 ビットの型（um_long）に読み替え、
    // リテラルの接尾辞 l / ul も ll / ull にする。
    const asCxx = (msl: string) =>
      windows ? msl.replace(/\blong\b/g, 'um_long').replace(/\b(\d[\dA-Fa-fxX]*)(u?)l\b/g, '$1$2ll') : msl;
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
          writeFileSync(`${base}.metal`, asCxx(text));
          writeFileSync(`${base}.cpp`, `#include <cmath>\n#include <cstdint>\n#include "metal_stub.h"\n#include "${base}.metal"\n`);
          const r = spawnSync(CLANGXX, [...cxx, '-Werror', '-fsyntax-only', `${base}.cpp`], { encoding: 'utf8' });
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
      for (const [name, text] of compileMsl(src)) writeFileSync(join(dir, `${name}.metal`), asCxx(text));
      const exe = exePath(join(out, 'harness'));
      const c = spawnSync(CLANGXX, [...cxx, '-O1', '-I', dir, join(mslDir, 'harness.cpp'), '-o', exe], { encoding: 'utf8' });
      if (c.status !== 0) throw new Error(c.stderr.split('\n').slice(0, 20).join('\n'));
      const r = spawnSync(exe, [], { encoding: 'utf8' });
      // harness は C++ の printf で書く（Windows では改行が \r\n になる）。
      const msl = r.stdout.split(/\r?\n/).filter((l) => l !== '');
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
  // 既定は wasm32。ターゲットによって変わるエラーは `// target: host`（または wasm32-plugin）で指定する。
  const t = directives(src, 'target')[0];
  const target = t === 'host' || t === 'wasm32-plugin' ? t : 'wasm32';
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
} else {
  // Windows: pipe_driver.c は POSIX 用なので、同じ書き方をここで行う。Windows のパイプに「待たない」設定はないので、
  // nonblocking.um は「間を空けて届く行を、readLine が待って受け取る」ことだけを確かめる。
  const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
  for (const f of files('tests/io')) {
    const src = readFileSync(join(root, f), 'utf8');
    const want = directives(src, 'expect');
    const mode = f.replace(/^.*[\\/]/, '').replace(/\.um$/, '');
    const name = `${f} [host, piped stdin]`;
    try {
      const exe = join(tmp, `io_${mode}.bin`);
      build(compileToLL(src, 'host'), 'host', exe);
      const child = spawn(exe, [], { cwd: tmp, stdio: ['pipe', 'pipe', 'inherit'] });
      let stdout = '';
      child.stdout.setEncoding('utf8');
      child.stdout.on('data', (d: string) => (stdout += d));
      const closed = new Promise<number | null>((r) => child.on('close', (code) => r(code)));
      const timer = setTimeout(() => child.kill(), 60000);
      if (mode === 'partial') {
        child.stdin.write('sto');
        await sleep(1000);
        child.stdin.end('p\n');
      } else {
        await sleep(300);
        child.stdin.write('first\n');
        await sleep(300);
        child.stdin.end('second\n');
      }
      const status = await closed;
      clearTimeout(timer);
      const out = stdout.split('\n').filter((l) => l !== '');
      if (status === 0 && same(out, want)) ok(name);
      else fail(name, `expected: ${want.join(' ')}\nactual:   ${out.join(' ')} (exit ${status})`);
    } catch (e) {
      fail(name, String(e));
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

// コマンドライン引数 args() と標準エラー出力 eprint()。引数は空白・空文字列・UTF-8 をそのまま渡す。
{
  const src = [
    'export function main(): i32 {',
    '  const a = args();',
    '  print(a.length);',
    '  for (const s of a) {',
    '    print("[" + s + "]");',
    '  }',
    '  eprint("err " + String(a.length));',
    '  return 0;',
    '}',
  ].join('\n');
  const argv = ['a b', '', 'é'];
  const want = ['3', '[a b]', '[]', '[é]'];
  try {
    const exe = join(tmp, 'args_test.bin');
    build(compileToLL(src, 'host'), 'host', exe);
    const r = spawnSync(exe, argv, { encoding: 'utf8', cwd: tmp });
    const out = r.stdout.split('\n').filter((l) => l !== '');
    if (same(out, want) && r.stderr === 'err 3\n') ok('args() and eprint() [host]');
    else fail('args() and eprint() [host]', `stdout: ${out.join(' | ')}\nstderr: ${r.stderr}`);
  } catch (e) {
    fail('args() and eprint() [host]', String(e));
  }
  if (hasWasm) {
    try {
      const wasm = join(tmp, 'args_test.wasm');
      build(compileToLL(src, 'wasm32'), 'wasm32', wasm);
      const out: string[] = [];
      const err: string[] = [];
      const exports = await instantiate(readFileSync(wasm), { gpu: false, print: (s: string) => out.push(s), eprint: (s: string) => err.push(s), args: argv });
      await (exports.main as () => unknown)();
      if (same(out, want) && same(err, ['err 3'])) ok('args() and eprint() [wasm32]');
      else fail('args() and eprint() [wasm32]', `stdout: ${out.join(' | ')}\nstderr: ${err.join(' | ')}`);
    } catch (e) {
      fail('args() and eprint() [wasm32]', String(e));
    }
  }
}

// Windows: 標準入力の先頭の UTF-8 の BOM（Windows PowerShell 5.1 がパイプで付ける）は読み飛ばす。2行目以降の BOM はそのまま。
if (windows) {
  const name = 'leading UTF-8 BOM on stdin is skipped [host, Windows]';
  try {
    const src = 'export function main(): i32 { const a = readLine(); if (a != null) { print(a.length); } const b = readLine(); if (b != null) { print(b.length); } return 0; }';
    const exe = join(tmp, 'bom_test.bin');
    build(compileToLL(src, 'host'), 'host', exe);
    const r = spawnSync(exe, [], { encoding: 'utf8', input: Buffer.from('﻿usi\r\n﻿x\r\n'), cwd: tmp });
    const out = r.stdout.split('\n').filter((l) => l !== '');
    if (same(out, ['3', '4'])) ok(name);
    else fail(name, `expected: 3 4\nactual:   ${out.join(' ')}`);
  } catch (e) {
    fail(name, String(e));
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
    const probe = exePath(join(tmp, 'probe_tsan'));
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

// wasm32-plugin: ホストの関数を使うプログラムは、ビルドの時点で失敗する（import を持たないことを確かめる）。
{
  const name = 'wasm32-plugin rejects host functions (print) [wasm32-plugin build]';
  try {
    const src = 'export function f(x: f64): void { print(x); }';
    build(compileToLL(src, 'wasm32-plugin'), 'wasm32-plugin', join(tmp, 'plugin_print.wasm'), '2', false, true);
    fail(name, 'built without error');
  } catch (e) {
    if (e instanceof CompileError && e.code === 'plugin-import' && /print/.test(e.message)) ok(name);
    else fail(name, String(e));
  }
}

// WebAssembly の組み込み（wasmtime。ネイティブだけ）。wasmtime の C API が見つからなければ飛ばす。
// examples/wasm/*.um は `// expect:`、tests/wasm/*.um は `// expect:` か `// panic:` と照合する。
if (!wasmtimeAvailable()) {
  console.log('note: wasmtime の C API が見つからないため、examples/wasm と tests/wasm は省略します（UM_WASMTIME を設定してください。docs/wasm.ja.md）');
} else {
  for (const f of [...files('examples/wasm'), ...files('tests/wasm')]) {
    const src = readFileSync(join(root, f), 'utf8');
    const want = directives(src, 'expect');
    const panic = directives(src, 'panic')[0];
    const tag = f.replace(/[^a-z0-9]/gi, '_');
    try {
      // `// plugin: 相対パス.um` は、実行の前に --target wasm32-plugin でビルドして、実行する場所に「名前.wasm」で置く。
      for (const rel of directives(src, 'plugin')) {
        const path = join(root, dirname(f), rel);
        const name = rel.replace(/^.*[\\/]/, '').replace(/\.um$/, '.wasm');
        build(compileToLL(readFileSync(path, 'utf8'), 'wasm32-plugin', false, path), 'wasm32-plugin', join(tmp, name));
      }
      const n = runNative(src, tag, join(root, f));
      if (panic !== undefined) {
        if (n.panic === `panic: ${panic}` && same(n.out, want)) ok(`${f} [host]`);
        else fail(`${f} [host]`, `expected: panic: ${panic}\nactual:   ${n.panic ?? 'no panic'} (output: ${n.out.join(' ')})`);
      } else if (n.panic) fail(`${f} [host]`, n.panic);
      else if (same(n.out, want)) ok(`${f} [host]`);
      else fail(`${f} [host]`, `expected: ${want.join(' ')}\nactual:   ${n.out.join(' ')}`);
    } catch (e) {
      fail(`${f} [host]`, String(e));
    }
  }
}

/**
 * 素材を作る ffmpeg のコマンド。UM_FFMPEG_HOME の bin にあればそれ、なければ PATH の ffmpeg。
 * 素材を作るためだけに使う（プログラムにリンクするのはライブラリのほう）。
 */
function ffmpegCommand(): string | null {
  const home = process.env.UM_FFMPEG_HOME;
  const cands = home ? [join(home, 'bin', windows ? 'ffmpeg.exe' : 'ffmpeg'), 'ffmpeg'] : ['ffmpeg'];
  for (const c of cands) {
    if (spawnSync(c, ['-hide_banner', '-version'], { stdio: 'ignore' }).status === 0) return c;
  }
  return null;
}

/**
 * tests/media と examples/media が読む素材を tmp に作る。どのコマも、画素の色にコマの番号 N を入れてある
 * （R = (N mod 16) * 16 + 8、G = floor(N / 16) * 16 + 8、B = 128）。読んだコマの番号を色から確かめられる。
 *   clip.mp4     H.264（使える符号器）320x240、25 コマ毎秒、4 秒（100 コマ）、キーフレームは 1 秒ごと。音は 440 Hz、44.1 kHz のステレオの AAC
 *   bframes.mp4  MPEG-4 Part 2 の B フレームつき（表示の順と復号の順が違う）、160x120、25 コマ毎秒、3 秒、キーフレームは 12 コマごと
 *   ten.mov      ProRes 422 の 10 ビット、64x64、24 コマ毎秒、1 秒
 *   broken.mp4   clip.mp4 の先頭の 3000 バイトだけ（途中で切れたファイル）
 *   garbage.mp4  動画ではないバイト列
 */
function makeMediaClips(ffmpeg: string): void {
  const enc = execFileSync(ffmpeg, ['-hide_banner', '-encoders'], { encoding: 'utf8' });
  const h264 = ['libopenh264', 'libx264', 'h264_mf', 'h264_videotoolbox'].find((e) => new RegExp(`\\s${e}\\s`).test(enc));
  const coded = (size: string, rate: number) =>
    `color=black:s=${size}:r=${rate},format=rgb24,geq=r='mod(N\\,16)*16+8':g='floor(N/16)*16+8':b=128`;
  const run = (args: string[]) => execFileSync(ffmpeg, ['-hide_banner', '-loglevel', 'error', '-y', ...args], { cwd: tmp, stdio: ['ignore', 'ignore', 'inherit'] });
  run(['-f', 'lavfi', '-i', coded('320x240', 25), '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=44100', '-t', '4',
    '-c:v', h264 ?? 'mpeg4', '-g', '25', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-ac', '2', '-b:a', '128k', '-shortest', 'clip.mp4']);
  run(['-f', 'lavfi', '-i', coded('160x120', 25), '-t', '3', '-c:v', 'mpeg4', '-bf', '2', '-g', '12', '-q:v', '2', '-pix_fmt', 'yuv420p', 'bframes.mp4']);
  run(['-f', 'lavfi', '-i', coded('64x64', 24), '-t', '1', '-c:v', 'prores_ks', '-profile:v', '2', '-pix_fmt', 'yuv422p10le', 'ten.mov']);
  writeFileSync(join(tmp, 'broken.mp4'), readFileSync(join(tmp, 'clip.mp4')).subarray(0, 3000));
  writeFileSync(join(tmp, 'garbage.mp4'), Buffer.from('this is not a video file
'.repeat(200)));
}

// 動画・音声の組み込み（FFmpeg。ネイティブだけ）。
const ffmpegCli = ffmpegAvailable() ? ffmpegCommand() : null;
if (!ffmpegAvailable()) {
  console.log('note: FFmpeg のライブラリが見つからないため、examples/media と tests/media は省略します（UM_FFMPEG_HOME を設定してください。docs/media.ja.md）');
} else if (!ffmpegCli) {
  console.log('note: 素材を作る ffmpeg のコマンドが見つからないため、examples/media と tests/media は省略します');
} else {
  makeMediaClips(ffmpegCli);
  for (const f of [...files('examples/media'), ...files('tests/media')]) {
    const src = readFileSync(join(root, f), 'utf8');
    const want = directives(src, 'expect');
    const panic = directives(src, 'panic')[0];
    const tag = f.replace(/[^a-z0-9]/gi, '_');
    try {
      const n = runNative(src, tag, join(root, f));
      if (panic !== undefined) {
        if (n.panic === `panic: ${panic}` && same(n.out, want)) ok(`${f} [host]`);
        else fail(`${f} [host]`, `expected: panic: ${panic}\nactual:   ${n.panic ?? 'no panic'} (output: ${n.out.join(' ')})`);
      } else if (n.panic) fail(`${f} [host]`, n.panic);
      else if (same(n.out, want)) ok(`${f} [host]`);
      else fail(`${f} [host]`, `expected: ${want.join(' | ')}\nactual:   ${n.out.join(' | ')}`);
    } catch (e) {
      fail(`${f} [host]`, String(e));
    }
  }
}

rmSync(tmp, { recursive: true, force: true });
if (skippedWasm > 0) console.log(`\nwasm32 のテスト ${skippedWasm} 件は飛ばしました（この clang は wasm32 を出力できません）`);
console.log(`\n${passed} passed, ${failures.length} failed`);
process.exit(failures.length === 0 ? 0 : 1);
