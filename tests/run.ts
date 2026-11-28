// テストランナー（依存なし）: node tests/run.ts
//
//   examples/*.um      `// expect: 行` と出力を照合。wasm32 で実行、ホストのネイティブでも実行。
//                      `// stdin: 行` があれば、その行を標準入力として与える（何回でも書ける。1つにつき1行）。
//                      ネイティブの実行は一時フォルダの中で行うので、writeFile で相対パスに書いたファイルはそこにできる。
//                      wasm32 では、ファイルはメモリ上の入れ物に読み書きする（今いるフォルダは /um、実行ファイルは
//                      /um/bin/prog.wasm。開いたままのファイルは readFile / writeFile の口で代わりに行われる）。
//                      AddressSanitizer か valgrind があれば、メモリ検査つきでも実行する
//                      （解放後使用・範囲外アクセス・二重解放・解放漏れを検出）。
//                      arm64-macos はオブジェクト生成まで確認。
//   tests/gpu/*.um     GPU（WebGPU）用のテスト。ここでは wasm32 向けにコンパイルして WGSL を生成し、
//                      CPU だけで実行して `// expect:` と照合する。GPU での実行はブラウザで行う。
//                      `// wgsl-has: kernel 文字列` / `// wgsl-lacks: kernel 文字列` があれば、その kernel の WGSL に
//                      文字列がある / ないことを確かめる。
//   tests/msl/         MSL の出力を、Metal なしで確かめる。clang++ があれば: tests/gpu と examples の kernel の MSL を
//                      スタブヘッダ（metal_stub.h）つきで C++14 として構文検査し、tests/msl/kernels.um の kernel の MSL を
//                      C++ として実行して、CPU 実行（ネイティブ）とチェックサムを比べる。Metal コンパイラでの確認ではない。
//   tests/metal/*.um と tests/wgpu/*.um
//                      UM_WGPU=1（wgpu-native）でネイティブに作り、GPU で実行して `// expect:` と照合する
//                      （どれも GPU の結果を CPU の結果と比べるプログラム）。wgpu-native が見つかり（UM_WGPU_HOME など。
//                      docs/wgpu.ja.md）、GPU が使えるときだけ。`// wgpu-error: コード` があれば、その診断でコンパイルが止まることを
//                      確かめる（64 ビットの整数や simdgroup の集計を gpu.dispatch で使う、Metal だけの機能）。
//                      `// wgsl: 名前` があれば、その kernel の WGSL を実行するフォルダに <名前>.wgsl として書く（gpu.loadKernel 用）。
//                      UM_TEST_WGPU_BACKENDS=vulkan,dx12 のように書くと、wgpu のバックエンドごとに実行する。
//                      `// cpu-build` があれば、GPU の経路のないビルド（UM_WGPU なしのネイティブと wasm32）でも実行して照合する。
//   tests/io/*.um      標準入力の届き方に左右される動作。pipe_driver.c がパイプに書く（「待たない」設定のパイプ、
//                      1行を2回に分けて書く）。ネイティブだけ。
//   tests/threads/*.um スレッド（ネイティブだけ）。`// expect:` か `// panic:` と照合する。`// sorted` があれば出力を
//                      並べ替えてから比べる。メモリ検査つきと、ThreadSanitizer（データ競合の検出）つきでも実行する。
//   tests/process/process.um
//                      processRun / processStart / processId（ネイティブ）。子は同じ実行ファイルを引数を変えて始める（窓は出さない）。
//   tests/errors/*.um  `// error: コード` の診断がすべて出ることを確認。
//   tests/panics/*.um  `// panic: メッセージ` で停止することを確認。
//   examples/wasm/*.um WebAssembly の組み込み（wasmtime。ネイティブだけ）。`// expect:` と照合する。
//   tests/wasm/*.um    同じく。`// expect:` か `// panic:` と照合する。wasmtime の C API が見つからなければ飛ばす。
//   tests/window/*.um と examples/window/*.um
//                      ウィンドウと音の組み込み（SDL3。ネイティブだけ）。SDL_VIDEO_DRIVER=offscreen と SDL_AUDIO_DRIVER=dummy で
//                      画面と音のデバイスなしに実行し、`// expect:` か `// panic:` と照合する（`// args:` はコマンドライン引数）。
//                      `// expect:` のない例はビルドだけ確かめる。SDL3 が見つからなければ飛ばす。
//   examples/media/*.um 動画・音声の組み込み（FFmpeg。ネイティブだけ）。`// expect:` と照合する。
//   tests/http/*.um    HTTP の組み込み（ネイティブ）。tests/http/server.mjs を手元（127.0.0.1）に立て、その場所を UM_TEST_HTTP で渡して
//                      実行し、`// expect:` か `// panic:` と照合する（外へは出ない）。AddressSanitizer があればメモリ検査つきでも実行する。
//   tests/media/*.um   同じく。`// expect:` か `// panic:` と照合する。素材は ffmpeg のコマンドで一時フォルダに作る
//                      （makeMediaClips）。FFmpeg のライブラリか ffmpeg のコマンドが見つからなければ飛ばす。
//
// UM_TEST_KNOWN_FAILURES=一覧のファイル を付けると、そこに理由つきで挙げた試験の失敗を「known」として表示し、
// 失敗には数えない（CI の特定の環境でだけ起きる、UM のせいではない失敗のため。.github/ci/known-failures-*.txt）。
//
// 手元で一部だけ確かめるとき（時間の短縮）:
//   UM_TEST_ONLY=正規表現   パス（examples/trig.um など）がこれに合うファイルの試験だけを行う
//   UM_TEST_SKIP=正規表現   パスがこれに合うファイルの試験を行わない（UM_TEST_ONLY と一緒にも使える）
//   UM_TEST_MEMCHECK=0      メモリ検査つきの実行（AddressSanitizer・valgrind・leaks）をすべて省く
//   例: UM_TEST_ONLY='trig|float_bits|errors/math-' node tests/run.ts
//       UM_TEST_SKIP='threads/handoff' node tests/run.ts

import { execFileSync, spawn, spawnSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { CompileError, CompileErrors } from '../src/diag.ts';
import { build, CLANG, compileMsl, compileToLL, compileWgsl, exePath, ffmpegAvailable, wasmtimeAvailable, sdl3Available } from '../src/driver.ts';
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

/**
 * 既知の失敗（環境変数 UM_TEST_KNOWN_FAILURES = 一覧のファイル）。CI の特定の環境でだけ、UM のせいではない理由で
 * 通らない試験を、理由つきで並べる（1 行に「試験の名前 | 理由」。# で始まる行と空行は読まない）。
 * 一覧にある試験が落ちたら `known` と理由を表示して、失敗には数えない（黙って飛ばさない）。通ったらそのことを表示する。
 * 名前のパスの区切りは / で書く（Windows の \ も / として比べる）。
 */
const knownFailures = new Map<string, string>();
if (process.env.UM_TEST_KNOWN_FAILURES) {
  for (const line of readFileSync(process.env.UM_TEST_KNOWN_FAILURES, 'utf8').split('\n')) {
    const t = line.trim();
    if (t === '' || t.startsWith('#')) continue;
    const bar = t.indexOf(' | ');
    if (bar < 0) throw new Error(`UM_TEST_KNOWN_FAILURES: expected "test name | reason": ${t}`);
    knownFailures.set(t.slice(0, bar).trim(), t.slice(bar + 3).trim());
  }
}
let knownFailed = 0;
const knownReason = (name: string): string | undefined => knownFailures.get(name.replace(/\\/g, '/'));

function ok(name: string): void {
  passed++;
  console.log(`ok   ${name}`);
  if (knownReason(name) !== undefined) console.log(`note: 既知の失敗として挙げた試験が通りました（一覧から外せます）: ${name}`);
}

/** この環境の clang が wasm32 を出力できるか。できなければ wasm32 のテストは飛ばす。 */
let hasWasm = true;
let skippedWasm = 0;

function fail(name: string, detail: string): void {
  if (!hasWasm && name.includes('wasm32')) {
    skippedWasm++;
    return;
  }
  const reason = knownReason(name);
  if (reason !== undefined) {
    knownFailed++;
    console.log(`known ${name}\n     ${detail.replace(/\n/g, '\n     ')}\n     既知の失敗: ${reason}`);
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

/** UM_TEST_ONLY / UM_TEST_SKIP（パスの正規表現）で、試験するファイルを選ぶ。パスの区切りは / で比べる。 */
const onlyRe = process.env.UM_TEST_ONLY ? new RegExp(process.env.UM_TEST_ONLY) : null;
const skipRe = process.env.UM_TEST_SKIP ? new RegExp(process.env.UM_TEST_SKIP) : null;
function selected(f: string): boolean {
  const p = f.replace(/\\/g, '/');
  return (!onlyRe || onlyRe.test(p)) && !(skipRe && skipRe.test(p));
}

function files(dir: string): string[] {
  return readdirSync(join(root, dir))
    .filter((f) => f.endsWith('.um'))
    .sort()
    .map((f) => join(dir, f))
    .filter(selected);
}

async function runWasm(src: string, tag: string): Promise<{ out: string[]; panic?: string }> {
  const wasm = join(tmp, `${tag}.wasm`);
  build(compileToLL(src, 'wasm32'), 'wasm32', wasm);
  const out: string[] = [];
  // 標準入力は `// stdin:` の行。ファイルはメモリ上の入れ物（実行ごとに空から始まる）。
  const lines = stdinLines(src);
  // 今いるフォルダは /um、実行ファイルは /um/bin/prog.wasm とする。名前は絶対パスにして入れ物の鍵にする。
  const disk = new Map<string, Uint8Array>();
  const dirs = new Set<string>(['/', '/um', '/um/bin']);
  const norm = (p: string): string => {
    const parts: string[] = [];
    for (const s of (p.startsWith('/') ? p : '/um/' + p).split('/')) {
      if (s === '..') parts.pop();
      else if (s !== '' && s !== '.') parts.push(s);
    }
    return '/' + parts.join('/');
  };
  const parent = (p: string): string => p.slice(0, Math.max(1, p.lastIndexOf('/')));
  const exports = await instantiate(readFileSync(wasm), {
    print: (s: string) => out.push(s),
    readLine: () => lines.shift() ?? null,
    inputReady: () => true,
    readFile: (path: string) => disk.get(norm(path)) ?? null,
    writeFile: (path: string, bytes: Uint8Array) => {
      if (!dirs.has(parent(norm(path))) || dirs.has(norm(path))) return false;
      disk.set(norm(path), bytes);
      return true;
    },
    // fileSize / readFileRange / appendFile と開いたままのファイルは、readFile / writeFile の口で代わりに行われる。
    listDir: (path: string) => {
      const d = norm(path);
      if (!dirs.has(d)) return null;
      return [...disk.keys(), ...dirs].filter((k) => k !== d && parent(k) === d).map((k) => k.slice(k.lastIndexOf('/') + 1));
    },
    makeDir: (path: string) => {
      const d = norm(path);
      for (let p = d; !dirs.has(p); p = parent(p)) {
        if (disk.has(p)) return false;
        dirs.add(p);
      }
      return true;
    },
    isDir: (path: string) => dirs.has(norm(path)),
    // 名前の変更（ファイルは置き換える。フォルダは中の名前ごと移す）と削除。
    fileRename: (from: string, to: string) => {
      const a = norm(from);
      const b = norm(to);
      if (!dirs.has(parent(b)) || dirs.has(b)) return false;
      const data = disk.get(a);
      if (data) {
        disk.delete(a);
        disk.set(b, data);
        return true;
      }
      if (!dirs.has(a) || disk.has(b) || b.startsWith(a + '/')) return false;
      for (const d of [...dirs]) {
        if (d === a || d.startsWith(a + '/')) {
          dirs.delete(d);
          dirs.add(b + d.slice(a.length));
        }
      }
      for (const [k, v] of [...disk]) {
        if (k.startsWith(a + '/')) {
          disk.delete(k);
          disk.set(b + k.slice(a.length), v);
        }
      }
      return true;
    },
    fileRemove: (path: string) => disk.delete(norm(path)),
    dirRemove: (path: string) => {
      const d = norm(path);
      if (!dirs.has(d) || d === '/') return false;
      for (const k of [...dirs]) if (k === d || k.startsWith(d + '/')) dirs.delete(k);
      for (const k of [...disk.keys()]) if (k.startsWith(d + '/')) disk.delete(k);
      return true;
    },
    cwd: '/um',
    exePath: '/um/bin/prog.wasm',
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

/**
 * 試験のプログラムを実行する（spawnSync に時間切れを付けたもの）。5 分で終わらなければ止めて、例外にする
 * （止まったままの試験があっても、ほかの試験は続き、どれが止まったかが失敗として分かる）。
 */
function runChild(cmd: string, args: string[], opts: Parameters<typeof spawnSync>[2] & { encoding: 'utf8' }) {
  const r = spawnSync(cmd, args, { timeout: 300000, ...opts });
  if (r.error) throw new Error(`${(r.error as NodeJS.ErrnoException).code === 'ETIMEDOUT' ? 'timed out after 300 s' : String(r.error)}: ${cmd}\n${String(r.stderr ?? '').trim().split('\n').slice(-15).join('\n')}`);
  return r;
}

/**
 * ネイティブで作って実行する。panic は停止（終了コード 101）の報告。
 * exit は、0 でも 101 でもない終わり方（シグナルなど）をしたときの、終わり方と標準エラー出力の最後（失敗の説明に添える）。
 */
function runNative(src: string, tag: string, file?: string): { out: string[]; panic?: string; exit?: string } {
  const exe = join(tmp, `${tag}.bin`);
  // file はソースの場所（import を解決するため。なければ import は使えない）。
  build(compileToLL(src, 'host', false, file), 'host', exe);
  const r = runChild(exe, [], { encoding: 'utf8', input: stdinText(src), cwd: tmp });
  const out = r.stdout.split('\n').filter((l) => l !== '');
  if (r.status === 101) return { out, panic: r.stderr.trim() };
  if (r.status === 0) return { out };
  return { out, exit: `exited with ${r.status ?? r.signal}\n${r.stderr.trim().split('\n').slice(-15).join('\n')}` };
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
    r = runChild(exe, [], { encoding: 'utf8', input: stdinText(src), cwd: tmp, env: { ...process.env, ASAN_OPTIONS: 'halt_on_error=1' } });
    if (r.status === 0) {
      const plain = join(tmp, `${tag}.leaks`);
      build(compileToLL(src, 'host'), 'host', plain, '1');
      const l = runChild('leaks', ['-q', '--atExit', '--', plain], { encoding: 'utf8', input: stdinText(src), cwd: tmp });
      if (l.status !== 0) {
        // leaks は報告を標準出力に書く。プログラム自身の出力のあとに続く。
        const text = l.stdout.slice(l.stdout.search(/^Process /m) >= 0 ? l.stdout.search(/^Process /m) : 0) + l.stderr;
        return { out: r.stdout.split('\n').filter((x) => x !== ''), report: text.trim().split('\n').slice(0, 14).join('\n') || `leaks exited with ${l.status ?? l.signal}` };
      }
    }
  } else if (tool === 'asan') {
    build(compileToLL(src, 'host', true), 'host', exe, '1', true, quiet);
    r = runChild(exe, [], { encoding: 'utf8', input: stdinText(src), cwd: tmp, env: asanEnv('detect_leaks=1:halt_on_error=1') });
  } else {
    build(compileToLL(src, 'host'), 'host', exe, '1');
    r = runChild('valgrind', ['-q', '--error-exitcode=97', '--leak-check=full', '--errors-for-leak-kinds=all', exe], { encoding: 'utf8', input: stdinText(src), cwd: tmp });
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

const memCheck = process.env.UM_TEST_MEMCHECK === '0' ? null : detectMemCheck();
if (process.env.UM_TEST_MEMCHECK === '0') console.log('note: UM_TEST_MEMCHECK=0 のため、メモリ検査つきの実行は省略します');
else if (!memCheck) console.log('note: AddressSanitizer も valgrind も leaks も使えないため、メモリ検査つきの実行は省略します');
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
    // `// wgsl-has: kernel 文字列` / `// wgsl-lacks: kernel 文字列`: その kernel の WGSL に文字列がある / ない。
    for (const [key, want] of [['wgsl-has', true], ['wgsl-lacks', false]] as const) {
      for (const d of directives(src, key)) {
        const sp = d.indexOf(' ');
        const name = d.slice(0, sp);
        const text = d.slice(sp + 1);
        const wgsl = shaders.get(name);
        if (wgsl === undefined) throw new Error(`${key}: no WGSL for kernel '${name}'`);
        if (wgsl.includes(text) !== want) throw new Error(`${key}: the WGSL of kernel '${name}' ${want ? 'lacks' : 'has'} '${text}'`);
      }
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
    if (selected(f)) try {
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
  for (const target of selected(f) ? ['wasm32', 'host'] as const : []) {
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
    if (!selected(ef)) continue;
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

// build の linkInputs（um build --link-input）。リンクしないターゲット（wasm32）では使わずに印を立てる。
// Windows では、llvm-rc で作ったアイコンのリソース（.res）を渡して、exe にアイコンが入ることを確かめる（llvm-rc がなければ飛ばす）。
{
  const src = 'export function main(): i32 { print("linked"); return 0; }';
  const name = 'build linkInputs are ignored when the target does not link [wasm32]';
  try {
    const dummy = join(tmp, 'link_input_dummy.bin');
    writeFileSync(dummy, 'x');
    const r = build(compileToLL(src, 'wasm32'), 'wasm32', join(tmp, 'link_input.wasm'), '2', false, false, [dummy]);
    if (r.linkInputsIgnored === true) ok(name);
    else fail(name, 'linkInputsIgnored was not set');
  } catch (e) {
    fail(name, String(e));
  }
  if (windows) {
    const llvmBin = [dirname(CLANG), 'C:\\Program Files\\LLVM\\bin'].find((d) => d !== '.' && existsSync(join(d, 'llvm-rc.exe')));
    const label = 'build linkInputs embed an icon resource in the exe [host, Windows]';
    if (!llvmBin || !existsSync(join(llvmBin, 'llvm-readobj.exe'))) {
      console.log('note: llvm-rc / llvm-readobj が見つからないため、exe にアイコンを埋め込むテストは飛ばします');
    } else {
      try {
        // 16×16・32 ビットの .ico を1枚だけ持つアイコンを作り、llvm-rc でリソースにする。
        const w = 16;
        const xor = Buffer.alloc(w * w * 4, Buffer.from([0xcc, 0x66, 0x33, 0xff]));
        const and = Buffer.alloc(4 * w);
        const bih = Buffer.alloc(40);
        bih.writeUInt32LE(40, 0); bih.writeInt32LE(w, 4); bih.writeInt32LE(w * 2, 8); bih.writeUInt16LE(1, 12); bih.writeUInt16LE(32, 14);
        bih.writeUInt32LE(xor.length + and.length, 20);
        const img = Buffer.concat([bih, xor, and]);
        const dir = Buffer.alloc(22);
        dir.writeUInt16LE(1, 2); dir.writeUInt16LE(1, 4); dir[6] = w; dir[7] = w; dir.writeUInt16LE(1, 10); dir.writeUInt16LE(32, 12);
        dir.writeUInt32LE(img.length, 14); dir.writeUInt32LE(22, 18);
        writeFileSync(join(tmp, 'app.ico'), Buffer.concat([dir, img]));
        writeFileSync(join(tmp, 'app.rc'), '1 ICON "app.ico"\n');
        execFileSync(join(llvmBin, 'llvm-rc.exe'), ['-fo', join(tmp, 'app.res'), join(tmp, 'app.rc')], { cwd: tmp, stdio: 'ignore' });
        const exe = build(compileToLL(src, 'host'), 'host', join(tmp, 'with_icon.exe'), '2', false, false, [join(tmp, 'app.res')]).output;
        const run = spawnSync(exe, [], { encoding: 'utf8', cwd: tmp });
        const res = execFileSync(join(llvmBin, 'llvm-readobj.exe'), ['--coff-resources', exe], { encoding: 'utf8' });
        if (run.stdout.trim() === 'linked' && /Type: ICON/.test(res) && /Type: GROUP_ICON/.test(res)) ok(label);
        else fail(label, `output: ${run.stdout.trim()}\nresources:\n${res.split('\n').filter((l) => /Type:/.test(l)).join('\n')}`);
      } catch (e) {
        fail(label, String(e));
      }
    }
    // コンソールの窓を出さない exe（Windows のサブシステム）。リンカへの指示（.drectve）だけを持つオブジェクトを clang で作って
    // linkInputs で渡すと、PE のサブシステムが WINDOWS（2）になり、パイプにつないだ標準出力と終了コードはそのまま使える。
    const glabel = 'build linkInputs can set the Windows GUI subsystem with a directive object [host, Windows]';
    try {
      writeFileSync(join(tmp, 'gui_subsystem.c'), '#pragma comment(linker, "/SUBSYSTEM:WINDOWS")\n#pragma comment(linker, "/ENTRY:mainCRTStartup")\n');
      execFileSync(CLANG, ['-c', '-o', join(tmp, 'gui_subsystem.obj'), join(tmp, 'gui_subsystem.c')], { stdio: ['ignore', 'ignore', 'pipe'] });
      const gsrc = 'export function main(): i32 { print(`gui ${args().length}`); return 7; }';
      const exe = build(compileToLL(gsrc, 'host'), 'host', join(tmp, 'gui_subsystem.exe'), '2', false, false, [join(tmp, 'gui_subsystem.obj')]).output;
      const pe = readFileSync(exe);
      const subsystem = pe.readUInt16LE(pe.readUInt32LE(0x3c) + 24 + 68);
      const run = spawnSync(exe, ['a', 'b'], { encoding: 'utf8', cwd: tmp });
      if (subsystem === 2 && run.status === 7 && run.stdout.trim() === 'gui 2') ok(glabel);
      else fail(glabel, `subsystem ${subsystem}, status ${run.status}, output: ${run.stdout.trim()}`);
    } catch (e) {
      fail(glabel, String(e));
    }
  }
}

// envVar（環境変数）。ネイティブは OS の環境（Windows は UTF-16 の環境ブロック）から、wasm32 は instantiate の env から読む。
// 値に日本語・空白・= を入れて、名前と値の文字の変換を確かめる。ない変数と空の名前は空。
{
  const src = [
    'export function main(): i32 {',
    '  print(envVar("UM_TEST_ENV_VALUE"));',
    '  print(envVar("UM_TEST_ENV_UNSET") == "");',
    '  print(envVar("") == "");',
    '  return 0;',
    '}',
  ].join('\n');
  const value = 'あい = う;x';
  const want = [value, 'true', 'true'];
  const name = 'envVar reads the process environment [host]';
  try {
    const exe = build(compileToLL(src, 'host'), 'host', join(tmp, 'env_var.bin')).output;
    const env: NodeJS.ProcessEnv = { ...process.env, UM_TEST_ENV_VALUE: value };
    delete env.UM_TEST_ENV_UNSET;
    const r = spawnSync(exe, [], { encoding: 'utf8', cwd: tmp, env });
    const out = r.stdout.split('\n').map((l) => l.replace(/\r$/, '')).filter((l) => l !== '');
    if (r.status === 0 && same(out, want)) ok(name);
    else fail(name, `status ${r.status}\nexpected: ${want.join(' | ')}\nactual:   ${out.join(' | ')}\n${r.stderr}`);
  } catch (e) {
    fail(name, String(e));
  }
  if (hasWasm) {
    const wname = 'envVar reads the env option of instantiate [wasm32]';
    try {
      const wasm = join(tmp, 'env_var.wasm');
      build(compileToLL(src, 'wasm32'), 'wasm32', wasm);
      const out: string[] = [];
      const exports = await instantiate(readFileSync(wasm), { gpu: false, print: (s: string) => out.push(s), env: { UM_TEST_ENV_VALUE: value } });
      await (exports.main as () => unknown)();
      if (same(out, want)) ok(wname);
      else fail(wname, `expected: ${want.join(' | ')}\nactual:   ${out.join(' | ')}`);
    } catch (e) {
      fail(wname, String(e));
    }
  }
}

// プロセス（processRun / processStart / processId）。tests/process/process.um をネイティブで作り、名前に日本語と空白を含む
// フォルダに置いて実行する（子は同じ実行ファイル。窓は出さない）。続けて、切り離した子（processStart）が、親が終わったあとも
// 残って印のファイルを書くことと、親の標準出力のパイプを持ち続けない（spawnSync が子を待たずに戻る）ことを確かめる。
// wasm32 では口がなければ失敗の値を返し、口があればそれを呼ぶことを確かめる。
if (selected('tests/process/process.um')) {
  const file = 'tests/process/process.um';
  const src = readFileSync(join(root, file), 'utf8');
  const want = directives(src, 'expect');
  const name = `${file} [host]`;
  try {
    const dir = join(tmp, 'プロセス の 試験');
    mkdirSync(dir, { recursive: true });
    const exe = build(compileToLL(src, 'host'), 'host', join(dir, 'process.bin')).output;
    const r = runChild(exe, [], { encoding: 'utf8', cwd: tmp });
    const out = r.stdout.split('\n').map((l) => l.replace(/\r$/, '')).filter((l) => l !== '');
    if (r.status === 0 && same(out, want)) ok(name);
    else fail(name, `status ${r.status}\nexpected: ${want.join(' | ')}\nactual:   ${out.join(' | ')}\n${r.stderr}`);

    const lname = 'processStart leaves a detached child that outlives the parent [host]';
    const marker = join(tmp, 'process_late_marker.txt');
    rmSync(marker, { force: true });
    const t0 = performance.now();
    const s = runChild(exe, ['startlate', marker], { encoding: 'utf8', cwd: tmp });
    const elapsed = performance.now() - t0;
    const early = existsSync(marker);
    let late = false;
    for (let i = 0; i < 200 && !late; i++) {
      await new Promise((res) => setTimeout(res, 50));
      late = existsSync(marker);
    }
    // 子は 1.5 秒待ってから書くので、親がそれより前に戻り、印がまだないなら、親は子を待っていない。
    if (s.status === 0 && elapsed < 1400 && !early && late) ok(lname);
    else fail(lname, `status ${s.status}, parent took ${elapsed.toFixed(0)} ms, marker before return: ${early}, marker later: ${late}`);
  } catch (e) {
    fail(name, String(e));
  }

  if (hasWasm) {
    const wsrc = [
      'export function main(): i32 {',
      '  const a: string[] = ["x y", "日本語", ""];',
      '  print(processRun("prog", a));',
      '  print(processStart("prog", a));',
      '  print(processId());',
      '  print(fileRename("a", "b"));',
      '  print(fileRemove("a"));',
      '  print(dirRemove("d"));',
      '  return 0;',
      '}',
    ].join('\n');
    const wname = 'process and file builtins call the instantiate options and fail without them [wasm32]';
    try {
      const wasm = join(tmp, 'process.wasm');
      build(compileToLL(wsrc, 'wasm32'), 'wasm32', wasm);
      const bare: string[] = [];
      await ((await instantiate(readFileSync(wasm), { gpu: false, print: (s: string) => bare.push(s) })).main as () => unknown)();
      const calls: string[] = [];
      const out: string[] = [];
      const exports = await instantiate(readFileSync(wasm), {
        gpu: false,
        print: (s: string) => out.push(s),
        processRun: (p: string, a: string[]) => (calls.push(`run ${p} ${JSON.stringify(a)}`), 5),
        processStart: (p: string, a: string[]) => (calls.push(`start ${p} ${a.length}`), true),
        processId: 1234,
        fileRename: (a: string, b: string) => (calls.push(`rename ${a} ${b}`), true),
        fileRemove: (p: string) => (calls.push(`remove ${p}`), true),
        dirRemove: (p: string) => (calls.push(`rmdir ${p}`), true),
      });
      await (exports.main as () => unknown)();
      const wantBare = ['-1', 'false', '0', 'false', 'false', 'false'];
      const wantOut = ['5', 'true', '1234', 'true', 'true', 'true'];
      const wantCalls = ['run prog ["x y","日本語",""]', 'start prog 3', 'rename a b', 'remove a', 'rmdir d'];
      if (same(bare, wantBare) && same(out, wantOut) && same(calls, wantCalls)) ok(wname);
      else fail(wname, `without options: ${bare.join(' ')}\nwith options: ${out.join(' ')}\ncalls: ${calls.join(' | ')}`);
    } catch (e) {
      fail(wname, String(e));
    }
  }
}

// dirRemove はリンクをたどらない。木の中に外のフォルダを指すリンク（Windows はジャンクション。管理者の権限が要らない）を置き、
// 木を消しても外のファイルが残ることを確かめる。消す名前そのものがリンクなら、リンクだけを消す。
{
  const name = 'dirRemove does not follow links out of the tree [host]';
  try {
    const src = [
      'export function main(): i32 {',
      '  const a = args();',
      '  print(dirRemove(a[0]));',
      '  print(isDir(a[0]));',
      '  return 0;',
      '}',
    ].join('\n');
    const exe = build(compileToLL(src, 'host'), 'host', join(tmp, 'dir_remove_links.bin')).output;
    const base = join(tmp, 'dir_remove_links');
    rmSync(base, { recursive: true, force: true });
    const outside = join(base, 'outside');
    const tree = join(base, 'tree');
    mkdirSync(outside, { recursive: true });
    mkdirSync(join(tree, 'sub', 'deep'), { recursive: true });
    writeFileSync(join(outside, 'keep.txt'), 'keep');
    writeFileSync(join(tree, 'sub', 'deep', 'x.txt'), 'x');
    writeFileSync(join(tree, 'y.txt'), 'y');
    symlinkSync(outside, join(tree, 'sub', 'link'), windows ? 'junction' : 'dir');
    symlinkSync(outside, join(base, 'rootlink'), windows ? 'junction' : 'dir');
    const r1 = runChild(exe, [tree], { encoding: 'utf8', cwd: tmp });
    const r2 = runChild(exe, [join(base, 'rootlink')], { encoding: 'utf8', cwd: tmp });
    const o1 = r1.stdout.split(/\r?\n/).filter((l) => l !== '');
    const o2 = r2.stdout.split(/\r?\n/).filter((l) => l !== '');
    const kept = existsSync(join(outside, 'keep.txt'));
    const gone = !existsSync(tree) && !existsSync(join(base, 'rootlink'));
    if (same(o1, ['true', 'false']) && same(o2, ['true', 'false']) && kept && gone) ok(name);
    else fail(name, `tree: ${o1.join(' ')}, link: ${o2.join(' ')}, outside kept: ${kept}, removed: ${gone}`);
  } catch (e) {
    fail(name, String(e));
  }
}

// HTTP の組み込み（runtime/http.c）。手元に立てた HTTP サーバー（tests/http/server.mjs。127.0.0.1 だけで待ち、外へは出ない）に対して、
// tests/http/*.um をネイティブで実行し、`// expect:` か `// panic:` と照合する。サーバーの場所は環境変数 UM_TEST_HTTP で渡す。
// 送受信は裏のスレッドで行うので、AddressSanitizer があればメモリ検査つきでも実行する（panic の試験は除く）。
// openssl のコマンドがあれば、自分で署名した証明書の HTTPS のサーバーも立て（UM_TEST_HTTPS）、検証で断られることを確かめる（tls.um）。
// wasm32 では要求がすぐに失敗の状態になり、理由を返すことも確かめる。
{
  const tlsDir = join(tmp, 'tls');
  mkdirSync(tlsDir, { recursive: true });
  const tlsKey = join(tlsDir, 'key.pem');
  const tlsCert = join(tlsDir, 'cert.pem');
  let tls = false;
  try {
    tls = spawnSync('openssl', ['req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-keyout', tlsKey, '-out', tlsCert, '-days', '2',
      '-subj', '/CN=127.0.0.1'], { stdio: 'ignore', timeout: 60000 }).status === 0 && existsSync(tlsCert);
  } catch {
    tls = false;
  }
  const serverEnv = tls ? { ...process.env, UM_TEST_TLS_KEY: tlsKey, UM_TEST_TLS_CERT: tlsCert } : process.env;
  const server = spawn(process.execPath, [join(root, 'tests', 'http', 'server.mjs')], { stdio: ['pipe', 'pipe', 'inherit'], env: serverEnv });
  try {
    const [port, tlsPort] = await new Promise<[number, number]>((resolvePort, reject) => {
      let text = '';
      const timer = setTimeout(() => reject(new Error('the test HTTP server did not start within 10 s')), 10000);
      server.stdout!.on('data', (d: Buffer) => {
        text += d.toString('utf8');
        const m = /port (\d+)(?: tlsport (\d+))?\n/.exec(text);
        if (m) {
          clearTimeout(timer);
          resolvePort([Number(m[1]), Number(m[2] ?? 0)]);
        }
      });
      server.on('exit', (code) => {
        clearTimeout(timer);
        reject(new Error(`the test HTTP server exited with ${code}`));
      });
    });
    process.env.UM_TEST_HTTP = `http://127.0.0.1:${port}`;
    if (tlsPort) process.env.UM_TEST_HTTPS = `https://127.0.0.1:${tlsPort}`;
    else console.log('note: openssl のコマンドが見つからないため、tests/http/tls.um（HTTPS の証明書の検証）は省略します');
    for (const f of files('tests/http')) {
      if (f.endsWith('tls.um') && !tlsPort) continue;
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
        else fail(`${f} [host]`, `expected: ${want.join(' | ')}\nactual:   ${n.out.join(' | ')}${n.exit ? `\n${n.exit}` : ''}`);
      } catch (e) {
        fail(`${f} [host]`, String(e));
      }
      if (memCheck === 'asan' && panic === undefined) {
        try {
          const m = runMemChecked(src, `${tag}_mem`, 'asan');
          if (m.report) fail(`${f} [asan]`, m.report);
          else if (same(m.out, want)) ok(`${f} [asan]`);
          else fail(`${f} [asan]`, `expected: ${want.join(' | ')}\nactual:   ${m.out.join(' | ')}`);
        } catch (e) {
          fail(`${f} [asan]`, String(e));
        }
      }
    }
  } catch (e) {
    fail('tests/http', String(e));
  } finally {
    delete process.env.UM_TEST_HTTP;
    delete process.env.UM_TEST_HTTPS;
    server.stdin!.end();
    server.kill();
  }
  if (hasWasm) {
    const wname = 'HTTP requests fail with a reason [wasm32]';
    const src = [
      'import { httpStart, httpState, httpStatus, httpWait, httpError, httpReadLine, httpRead, httpHeaders, httpCancel, httpClose } from "um:http";',
      'export function main(): i32 {',
      '  const id = httpStart("GET", "http://127.0.0.1:1/", "", "", 0);',
      '  print(`${httpState(id)} ${httpStatus(id)} ${httpWait(id, 0)}`);',
      '  print(httpError(id));',
      '  const line = httpReadLine(id);',
      '  print(`${line == null} ${httpRead(id).length} ${httpHeaders(id).length}`);',
      '  httpCancel(id);',
      '  httpClose(id);',
      '  return 0;',
      '}',
    ].join('\n');
    const want = ['3 0 false', 'HTTP requests are not available in wasm32 programs yet (build with --target host)', 'true 0 0'];
    try {
      const w = await runWasm(src, 'http_wasm');
      if (!w.panic && same(w.out, want)) ok(wname);
      else fail(wname, `expected: ${want.join(' | ')}\nactual:   ${w.out.join(' | ')}${w.panic ? `\n${w.panic}` : ''}`);
    } catch (e) {
      fail(wname, String(e));
    }
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

// 画像の読み込み（lib/png.um）。ネイティブと wasm32 で同じ結果になることを確かめる。
for (const f of files('tests/image')) {
  const src = readFileSync(join(root, f), 'utf8');
  const want = directives(src, 'expect');
  const tag = f.replace(/[^a-z0-9]/gi, '_');
  for (const target of ['host', 'wasm32'] as const) {
    const name = `${f} [${target}]`;
    try {
      const r = target === 'host' ? runNative(src, tag) : await runWasm(src, tag);
      if (r.panic) fail(name, r.panic);
      else if (same(r.out, want)) ok(name);
      else fail(name, `expected: ${want.join(' ')}\nactual:   ${r.out.join(' ')}`);
    } catch (e) {
      fail(name, String(e));
    }
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

// `// cpu-build` のある tests/wgpu のテストは、GPU の経路のないビルド（UM_WGPU なしのネイティブと wasm32）でも
// コンパイルして実行し、同じ出力になることを確かめる（gpu.prefer / gpu.available などはどのビルドでも書ける）。
for (const f of files('tests/wgpu')) {
  const src = readFileSync(join(root, f), 'utf8');
  if (!/^\/\/ cpu-build\s*$/m.test(src)) continue;
  const want = directives(src, 'expect');
  const tag = f.replace(/[^a-z0-9]/gi, '_');
  try {
    const n = runNative(src, `${tag}_cpu`, join(root, f));
    if (same(n.out, want)) ok(`${f} [host, no GPU]`);
    else fail(`${f} [host, no GPU]`, `expected: ${want.join(' ')}\nactual:   ${n.out.join(' ')}`);
  } catch (e) {
    fail(`${f} [host, no GPU]`, String(e));
  }
  try {
    const w = await runWasm(src, `${tag}_cpu`);
    if (same(w.out, want)) ok(`${f} [wasm32]`);
    else fail(`${f} [wasm32]`, `expected: ${want.join(' ')}\nactual:   ${w.out.join(' ')}`);
  } catch (e) {
    fail(`${f} [wasm32]`, String(e));
  }
}

// wgpu-native（UM_WGPU=1）。GPU で実行して CPU の結果と比べる（tests/metal と tests/wgpu）。
{
  process.env.UM_WGPU = '1';
  let gpuOk = false;
  try {
    const probe = 'export function main(): i32 { print(gpu.loadKernel("@compute @workgroup_size(1) fn um_main() {}") !== null); return 0; }';
    gpuOk = runNative(probe, 'wgpu_probe').out[0] === 'true';
    if (!gpuOk) console.log('note: wgpu-native は見つかったが GPU が使えないため、wgpu のテストは飛ばします');
  } catch (e) {
    console.log(`note: wgpu-native が見つからないため、wgpu のテストは飛ばします（docs/wgpu.ja.md。${String(e).split('\n')[0].slice(0, 120)}）`);
  }
  const backends = (process.env.UM_TEST_WGPU_BACKENDS ?? '').split(',').filter((b) => b !== '');
  if (gpuOk) {
    for (const f of [...files('tests/metal'), ...files('tests/wgpu')]) {
      const src = readFileSync(join(root, f), 'utf8');
      const want = directives(src, 'expect');
      const wantErr = directives(src, 'wgpu-error')[0];
      const tag = f.replace(/[^a-z0-9]/gi, '_');
      try {
        if (wantErr) {
          try {
            compileToLL(src, 'host');
            fail(`${f} [wgpu]`, `expected error ${wantErr}, but it compiled`);
          } catch (e) {
            const codes = e instanceof CompileErrors ? e.errors.map((x) => x.code) : e instanceof CompileError ? [e.code] : [];
            if (codes.includes(wantErr)) ok(`${f} [wgpu, error ${wantErr}]`);
            else fail(`${f} [wgpu]`, `expected error ${wantErr}, got: ${String(e)}`);
          }
          continue;
        }
        // `// plugin: 相対パス.um` は wasm の組み込みと一緒に使うテスト。wasmtime がなければ飛ばす。
        const plugins = directives(src, 'plugin');
        if (plugins.length > 0 && !wasmtimeAvailable()) continue;
        for (const rel of plugins) {
          const path = join(root, dirname(f), rel);
          const name = rel.replace(/^.*[\\/]/, '').replace(/\.um$/, '.wasm');
          build(compileToLL(readFileSync(path, 'utf8'), 'wasm32-plugin', false, path), 'wasm32-plugin', join(tmp, name));
        }
        for (const name of directives(src, 'wgsl')) {
          const text = compileWgsl(src, join(root, f), 64).get(name);
          if (!text) throw new Error(`no WGSL for kernel '${name}'`);
          writeFileSync(join(tmp, `${name}.wgsl`), text);
        }
        // `// fast-math` があれば、描画のシェーダを GPU のふつうの計算にしたビルド（UM_SHADER_FAST_MATH=1）でも照合する。
        const variants: [string, boolean][] = [['', false]];
        if (/^\/\/ fast-math\s*$/m.test(src)) variants.push([', fast math', true]);
        for (const [variant, fast] of variants) {
        const exe = join(tmp, `${tag}_wgpu${fast ? '_fast' : ''}.bin`);
        if (fast) process.env.UM_SHADER_FAST_MATH = '1';
        try {
          build(compileToLL(src, 'host', false, join(root, f)), 'host', exe);
        } finally {
          delete process.env.UM_SHADER_FAST_MATH;
        }
        for (const be of backends.length > 0 ? backends : ['']) {
          const label = be ? `${f} [wgpu ${be}${variant}]` : `${f} [wgpu${variant}]`;
          const env = { ...process.env, ...(be ? { UM_WGPU_BACKEND: be } : {}) };
          const r = spawnSync(exe, [], { encoding: 'utf8', input: stdinText(src), cwd: tmp, env, timeout: 300000 });
          const out = r.stdout.split('\n').filter((l) => l !== '');
          if (r.status !== 0) fail(label, `exit ${r.status}: ${r.stderr.trim().slice(0, 1500)}`);
          else if (same(out, want)) ok(label);
          else fail(label, `expected: ${want.join(' ')}\nactual:   ${out.join(' ')}`);
        }
        }
      } catch (e) {
        fail(`${f} [wgpu]`, String(e));
      }
    }
  }
  delete process.env.UM_WGPU;
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

// ウィンドウと音の組み込み（SDL3。ネイティブだけ）。画面と音のデバイスがなくても動くように、SDL の offscreen と dummy を使う。
if (!sdl3Available()) {
  console.log('note: SDL3 が見つからないため、examples/window と tests/window は省略します（UM_SDL3 を設定してください。docs/window.ja.md）');
} else {
  const env = { ...process.env, SDL_VIDEO_DRIVER: 'offscreen', SDL_AUDIO_DRIVER: 'dummy' };
  for (const f of [...files('examples/window'), ...files('tests/window')]) {
    const src = readFileSync(join(root, f), 'utf8');
    const want = directives(src, 'expect');
    const panic = directives(src, 'panic')[0];
    const argv = (directives(src, 'args')[0] ?? '').split(' ').filter((a) => a !== '');
    const exe = join(tmp, `${f.replace(/[^a-z0-9]/gi, '_')}.bin`);
    try {
      build(compileToLL(src, 'host', false, join(root, f)), 'host', exe);
      if (want.length === 0 && panic === undefined) {
        ok(`${f} [host, build]`);
        continue;
      }
      const r = spawnSync(exePath(exe), argv, { encoding: 'utf8', cwd: tmp, env, timeout: 60000 });
      const out = r.stdout.split('\n').map((l) => l.replace(/\r$/, '')).filter((l) => l !== '');
      const got = r.status === 101 ? r.stderr.trim().split('\n').map((l) => l.trim()).filter((l) => l.startsWith('panic:'))[0] : undefined;
      if (panic !== undefined) {
        if (got === `panic: ${panic}` && same(out, want)) ok(`${f} [host]`);
        else fail(`${f} [host]`, `expected: panic: ${panic}\nactual:   ${got ?? 'no panic'} (output: ${out.join(' | ')})`);
      } else if (r.status !== 0) fail(`${f} [host]`, `exit ${r.status ?? r.signal}\n${r.stderr.trim()}\noutput: ${out.join(' | ')}`);
      else if (same(out, want)) ok(`${f} [host]`);
      else fail(`${f} [host]`, `expected: ${want.join(' | ')}\nactual:   ${out.join(' | ')}`);
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
  writeFileSync(join(tmp, 'garbage.mp4'), Buffer.from('this is not a video file\n'.repeat(200)));
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
      else fail(`${f} [host]`, `expected: ${want.join(' | ')}\nactual:   ${n.out.join(' | ')}${n.exit ? `\n${n.exit}` : ''}`);
    } catch (e) {
      fail(`${f} [host]`, String(e));
    }
  }
}

// ブラウザ（docs/web.ja.md）。Playwright（とその Chromium）があれば、`build --web` の形でブラウザで開き、
// 画面の下の出力を `// expect:` と照合する。`// web-query: ?…` は URL の後ろに付ける（?gpu=0 で WebGPU を使わない、?args=…）。
// tests/web/*.um と、描画のテスト（tests/wgpu/draw*.um）を WebGPU あり（SwiftShader）となしで開く。
{
  let pw: { chromium: { launch(o: unknown): Promise<any> } } | null = null;
  try {
    pw = createRequire(import.meta.url)('playwright');
  } catch {
    console.log('note: Playwright が見つからないため、ブラウザのテスト（tests/web）は省略します');
  }
  if (pw) {
    const { createServer } = await import('node:http');
    const types: Record<string, string> = { '.html': 'text/html', '.mjs': 'text/javascript', '.wasm': 'application/wasm' };
    const webRoot = join(tmp, 'web');
    mkdirSync(webRoot);
    const server = createServer((req, res) => {
      const path = join(webRoot, decodeURIComponent((req.url ?? '/').split('?')[0]));
      if (!path.startsWith(webRoot) || !existsSync(path)) {
        res.writeHead(404).end();
        return;
      }
      res.writeHead(200, { 'content-type': types[path.slice(path.lastIndexOf('.'))] ?? 'application/octet-stream' }).end(readFileSync(path));
    });
    await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
    const port = (server.address() as { port: number }).port;
    const browser = await pw.chromium.launch({ args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-vulkan=swiftshader', '--use-webgpu-adapter=swiftshader', '--autoplay-policy=no-user-gesture-required'] });
    const cases: [string, string][] = [];
    for (const f of files('tests/web')) cases.push([f, directives(readFileSync(join(root, f), 'utf8'), 'web-query')[0] ?? '']);
    for (const f of files('tests/wgpu').filter((x) => /\/draw/.test(x))) cases.push([f, ''], [f, '?gpu=0']);
    for (const [f, query] of cases) {
      const label = `${f} [web${query.includes('gpu=0') ? ', no GPU' : ''}]`;
      try {
        const src = readFileSync(join(root, f), 'utf8');
        const want = directives(src, 'expect');
        const dir = join(webRoot, f.replace(/[^a-z0-9]/gi, '_'));
        mkdirSync(dir, { recursive: true });
        build(compileToLL(src, 'wasm32', false, join(root, f)), 'wasm32', join(dir, 'app.wasm'));
        for (const name of ['host.mjs', 'web.mjs']) writeFileSync(join(dir, name), readFileSync(join(root, 'runtime', name)));
        writeFileSync(join(dir, 'index.html'), readFileSync(join(root, 'runtime', 'web', 'index.html')));
        const page = await browser.newPage();
        await page.goto(`http://127.0.0.1:${port}/${dir.slice(webRoot.length + 1)}/index.html${query}`);
        await page.waitForFunction(() => (globalThis as any).document.body.dataset.exit !== undefined, null, { timeout: 120000 });
        const exit = await page.evaluate(() => (globalThis as any).document.body.dataset.exit);
        const out = String(await page.evaluate(() => (globalThis as any).document.getElementById('out').textContent)).split('\n').filter((l) => l !== '');
        const errText = String(await page.evaluate(() => (globalThis as any).document.getElementById('err').textContent)).trim();
        await page.close();
        if (exit !== '0') fail(label, `exit ${exit}: ${out.join(' | ')}\n${errText}`);
        else if (same(out, want)) ok(label);
        else fail(label, `expected: ${want.join(' | ')}\nactual:   ${out.join(' | ')}`);
      } catch (e) {
        fail(label, String(e));
      }
    }
    await browser.close();
    server.close();
  }
}

rmSync(tmp, { recursive: true, force: true });
if (skippedWasm > 0) console.log(`\nwasm32 のテスト ${skippedWasm} 件は飛ばしました（この clang は wasm32 を出力できません）`);
console.log(`\n${passed} passed, ${failures.length} failed${knownFailed > 0 ? `, ${knownFailed} known failures (UM_TEST_KNOWN_FAILURES)` : ''}`);
process.exit(failures.length === 0 ? 0 : 1);
