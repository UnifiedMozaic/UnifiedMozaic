import { execFileSync } from 'node:child_process';
import { copyFileSync, existsSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, extname, join, relative, resolve, sep } from 'node:path';
import { check } from './checker.ts';
import { generate, kernelUses64 } from './codegen.ts';
import { emitMsl } from './msl.ts';
import { emitWgsl, WEBGPU_CONST_ARRAY } from './wgsl.ts';
import type { TargetKind } from './codegen.ts';
import { parse } from './parser.ts';
import type { Decl, Module } from './ast.ts';
import { CompileError } from './diag.ts';

/**
 * wasm32-plugin: ホストのプログラムに組み込んで使う wasm（プラグイン）。import を一切持たない（入出力・GPU なし、
 * 停止は unreachable のトラップ）。export した関数は Buffer を (位置, 長さ) で受け取れ、メモリを確保する umAlloc / umFree も export する。
 * docs/wasm.ja.md の「UM で書いたプラグイン」を参照。
 */
export type TargetName = 'wasm32' | 'wasm32-plugin' | 'arm64-macos' | 'host';

export const TARGETS: Record<TargetName, { kind: TargetKind; ptrBits: 32 | 64 }> = {
  wasm32: { kind: 'wasm32', ptrBits: 32 },
  'wasm32-plugin': { kind: 'wasm32', ptrBits: 32 },
  'arm64-macos': { kind: 'native', ptrBits: 64 },
  host: { kind: 'native', ptrBits: 64 },
};

const RUNTIME_C = join(import.meta.dirname, '..', 'runtime', 'native.c');

/** macOS で Homebrew の LLVM が入っている場所（Apple Silicon と Intel）。 */
const BREW_CLANGS = ['/opt/homebrew/opt/llvm/bin/clang', '/usr/local/opt/llvm/bin/clang'];

/** Windows で LLVM の公式のインストーラ（`winget install LLVM.LLVM`）が clang を置く場所。 */
const WINDOWS_CLANGS = [join(process.env.ProgramFiles ?? 'C:\\Program Files', 'LLVM', 'bin', 'clang.exe')];

/** PATH に clang があるか（Windows では clang.exe）。 */
function clangOnPath(): boolean {
  const name = process.platform === 'win32' ? 'clang.exe' : 'clang';
  const sepChar = process.platform === 'win32' ? ';' : ':';
  return (process.env.PATH ?? process.env.Path ?? '').split(sepChar).some((d) => d !== '' && existsSync(join(d, name)));
}

/**
 * そのターゲットに使う clang。UM_CLANG があればそれ。macOS の wasm32 は、標準の clang が出力できないので、
 * Homebrew の LLVM があれば自動でそちらを使う（`brew install llvm lld`）。Windows で PATH に clang がなければ、
 * LLVM の公式のインストーラが入れた場所を使う（インストーラは既定では PATH に足さない）。ほかは PATH の clang。
 */
export function clangFor(target: TargetName): string {
  if (process.env.UM_CLANG) return process.env.UM_CLANG;
  if (TARGETS[target].kind === 'wasm32' && process.platform === 'darwin') {
    for (const c of BREW_CLANGS) if (existsSync(c)) return c;
  }
  if (process.platform === 'win32' && !clangOnPath()) {
    for (const c of WINDOWS_CLANGS) if (existsSync(c)) return c;
  }
  return 'clang';
}

/**
 * 使う clang（ネイティブ）。環境変数 UM_CLANG で差し替えられる（macOS の標準の clang は wasm32 を出力できないので、
 * Homebrew の LLVM を指定するときなどに使う）。
 */
export const CLANG = clangFor('host');

/**
 * ネイティブの実行ファイルの名前。Windows では拡張子のない名前に `.exe` を付ける
 * （拡張子がないと、Windows のほかのプログラムやシェルから実行ファイルとして起動できない）。
 */
export function exePath(path: string): string {
  return process.platform === 'win32' && extname(path) === '' ? `${path}.exe` : path;
}
const RUNTIME_METAL = join(import.meta.dirname, '..', 'runtime', 'metal.m');
const RUNTIME_WINDOW = join(import.meta.dirname, '..', 'runtime', 'window.c');
/** 音の出力（audioOpen など）。ウィンドウと同じく SDL3 を使い、window.c と一緒にリンクする。 */
const RUNTIME_AUDIO = join(import.meta.dirname, '..', 'runtime', 'audio.c');

/** SDL3 をリンクするための clang の引数と、実行ファイルの隣に置く DLL（Windows だけ）。 */
interface Sdl3 { args: string[]; dll?: string }

/**
 * ウィンドウの組み込み（runtime/window.c）が使う SDL3 を探す。
 *   1. 環境変数 UM_SDL3 = SDL3 を置いた場所（include/SDL3/SDL.h があるところ）。
 *      Windows では公式の SDL3-devel-*-VC.zip を展開した場所を指す（lib/x64 か lib/arm64 の SDL3.lib と SDL3.dll を使う）。
 *   2. pkg-config の sdl3（macOS の Homebrew、Linux のパッケージ）。
 *   3. macOS の Homebrew の決まった場所。
 */
function findSdl3(): Sdl3 {
  const hint = 'install SDL3 3.2.12 or later (macOS: `brew install sdl3`; Windows: unzip SDL3-devel-*-VC.zip from ' +
    'https://github.com/libsdl-org/SDL/releases and set UM_SDL3 to that folder; Linux: libsdl3-dev), or set UM_SDL3';
  const fromPrefix = (prefix: string): Sdl3 | null => {
    if (!existsSync(join(prefix, 'include', 'SDL3', 'SDL.h'))) return null;
    if (process.platform === 'win32') {
      const arch = process.arch === 'arm64' ? 'arm64' : 'x64';
      const lib = join(prefix, 'lib', arch);
      if (!existsSync(join(lib, 'SDL3.lib'))) return null;
      return { args: [`-I${join(prefix, 'include')}`, join(lib, 'SDL3.lib')], dll: join(lib, 'SDL3.dll') };
    }
    const lib = join(prefix, 'lib');
    return { args: [`-I${join(prefix, 'include')}`, `-L${lib}`, '-lSDL3', `-Wl,-rpath,${lib}`] };
  };
  const env = process.env.UM_SDL3;
  if (env) {
    const found = fromPrefix(resolve(env));
    if (!found) throw new Error(`SDL3 was not found in UM_SDL3=${env}; ${hint}`);
    return found;
  }
  if (process.platform !== 'win32') {
    try {
      const pc = (args: string[]) => execFileSync('pkg-config', args, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim();
      const flags = pc(['--cflags', '--libs', 'sdl3']).split(/\s+/).filter((a) => a !== '');
      const libdir = pc(['--variable=libdir', 'sdl3']);
      if (libdir) flags.push(`-Wl,-rpath,${libdir}`);
      return { args: flags };
    } catch {
      // pkg-config がない、または sdl3 が登録されていない。
    }
    if (process.platform === 'darwin') {
      for (const p of ['/opt/homebrew/opt/sdl3', '/usr/local/opt/sdl3']) {
        const found = fromPrefix(p);
        if (found) return found;
      }
    }
  }
  throw new Error(`this program opens a window, which needs SDL3, but SDL3 was not found; ${hint}`);
}

const RUNTIME_WASMTIME = join(import.meta.dirname, '..', 'runtime', 'wasm.c');

/** wasmtime をリンクするための clang の引数と、実行ファイルの隣に置く DLL（Windows だけ）。 */
interface Wasmtime { args: string[]; dll?: string }

/**
 * WebAssembly の組み込み（runtime/wasm.c）が使う wasmtime の C API を探す。
 *   1. 環境変数 UM_WASMTIME = C API を置いた場所（include/wasmtime.h があるところ）。
 *      公式のリリースの wasmtime-v*-<arch>-<os>-c-api（.zip / .tar.xz）を展開した場所を指す。
 *      Windows では lib/wasmtime.dll.lib でリンクし、lib/wasmtime.dll を実行ファイルの隣に写す。
 *      macOS・Linux では lib の共有ライブラリ（libwasmtime.dylib / .so）を rpath つきでリンクする。
 *   2. pkg-config の wasmtime（ソースから cmake でビルドして入れた場合など）。
 *   3. macOS の Homebrew の決まった場所（brew install wasmtime は C API も入れる）。
 */
function findWasmtime(): Wasmtime {
  const hint = 'download the C API from https://github.com/bytecodealliance/wasmtime/releases ' +
    '(wasmtime-v*-x86_64-windows-c-api.zip, -aarch64-macos-c-api.tar.xz, -x86_64-linux-c-api.tar.xz, ...), unpack it and set UM_WASMTIME to that folder';
  const fromPrefix = (prefix: string): Wasmtime | null => {
    if (!existsSync(join(prefix, 'include', 'wasmtime.h'))) return null;
    const lib = join(prefix, 'lib');
    if (process.platform === 'win32') {
      if (!existsSync(join(lib, 'wasmtime.dll.lib'))) return null;
      return { args: [`-I${join(prefix, 'include')}`, join(lib, 'wasmtime.dll.lib')], dll: join(lib, 'wasmtime.dll') };
    }
    const shared = process.platform === 'darwin' ? 'libwasmtime.dylib' : 'libwasmtime.so';
    if (!existsSync(join(lib, shared))) return null;
    return { args: [`-I${join(prefix, 'include')}`, `-L${lib}`, '-lwasmtime', `-Wl,-rpath,${lib}`] };
  };
  const env = process.env.UM_WASMTIME;
  if (env) {
    const found = fromPrefix(resolve(env));
    if (!found) throw new Error(`the wasmtime C API was not found in UM_WASMTIME=${env}; ${hint}`);
    return found;
  }
  if (process.platform !== 'win32') {
    try {
      const pc = (args: string[]) => execFileSync('pkg-config', args, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim();
      const flags = pc(['--cflags', '--libs', 'wasmtime']).split(/\s+/).filter((a) => a !== '');
      const libdir = pc(['--variable=libdir', 'wasmtime']);
      if (libdir) flags.push(`-Wl,-rpath,${libdir}`);
      return { args: flags };
    } catch {
      // pkg-config がない、または wasmtime が登録されていない。
    }
    if (process.platform === 'darwin') {
      for (const p of ['/opt/homebrew/opt/wasmtime', '/usr/local/opt/wasmtime']) {
        const found = fromPrefix(p);
        if (found) return found;
      }
    }
  }
  throw new Error(`this program runs WebAssembly modules, which needs the wasmtime C API, but it was not found; ${hint}`);
}

/** 生成した LLVM IR が WebAssembly の組み込みを使っているか（codegen は使ったときだけ宣言を出す）。 */
function usesWasm(ll: string): boolean {
  return ll.includes('declare i64 @um_wasm_compile(');
}

/** wasmtime が見つかるか（テストで、見つからなければ WebAssembly のテストを飛ばすのに使う）。 */
export function wasmtimeAvailable(): boolean {
  try {
    findWasmtime();
    return true;
  } catch {
    return false;
  }
}

/** wasm のバイナリの import の名前（"モジュール.名前"）。 */
export function wasmImports(bin: Uint8Array): string[] {
  const out: string[] = [];
  let p = 8;
  const leb = (): number => {
    let v = 0;
    let shift = 0;
    for (;;) {
      const b = bin[p++];
      v += (b & 127) * 2 ** shift;
      if ((b & 128) === 0) return v;
      shift += 7;
    }
  };
  const name = (): string => {
    const n = leb();
    const s = Buffer.from(bin.subarray(p, p + n)).toString('utf8');
    p += n;
    return s;
  };
  while (p < bin.length) {
    const id = bin[p++];
    const size = leb();
    const end = p + size;
    if (id === 2) {
      const count = leb();
      for (let i = 0; i < count; i++) {
        const mod = name();
        const field = name();
        out.push(`${mod}.${field}`);
        // 中身（種類と型）は読み飛ばす必要がないよう、このセクションを読み終えたら end に移る。
        const kind = bin[p++];
        if (kind === 0) leb();
        else if (kind === 1) { p++; const f = leb(); leb(); if (f & 1) leb(); }
        else if (kind === 2) { const f = leb(); leb(); if (f & 1) leb(); }
        else if (kind === 3) { p++; p++; }
        else { p++; leb(); }
      }
    }
    p = end;
  }
  return out;
}

/** プラグインが使ってしまった import を、UM の言葉で説明する。 */
function describeImport(imp: string): string {
  const field = imp.replace(/^env./, '');
  if (field.startsWith('um_print_')) return `print (${field})`;
  if (field === 'um_fmt_f64') return 'a conversion of f64 to string (um_fmt_f64)';
  if (field.startsWith('um_host_')) return `input / output (${field})`;
  if (field.startsWith('um_gpu_')) return `the GPU (${field})`;
  return imp;
}

/** 生成した LLVM IR がウィンドウか音の組み込みを使っているか（codegen は使ったときだけ宣言を出す）。 */
function usesWindow(ll: string): boolean {
  return ll.includes('declare i32 @um_window_open(');
}

/** SDL3 が見つかるか（テストで、ウィンドウと音の試験を飛ばすかを決めるため）。 */
export function sdl3Available(): boolean {
  try {
    findSdl3();
    return true;
  } catch {
    return false;
  }
}

/**
 * Metal（macOS の GPU）で kernel を実行する経路を使うか。
 * 環境変数 UM_METAL=1 を付けたときだけ有効にする（Metal と Foundation のフレームワークをリンクするため）。
 */
export function metalEnabled(target: TargetName): boolean {
  if (process.env.UM_METAL !== '1') return false;
  return target === 'arm64-macos' || (target === 'host' && process.platform === 'darwin');
}

const PRELUDE = join(import.meta.dirname, '..', 'runtime', 'prelude.um');
/**
 * ウィンドウの標準の型と関数（WindowEvent、pollWindowEvent など）。UM で書いてある。
 * ウィンドウを使わないプログラムに SDL3 をリンクさせないため、プログラムがこのファイルの名前を使うときだけ加える。
 */
const WINDOW_PRELUDE = join(import.meta.dirname, '..', 'runtime', 'window.um');
/**
 * WebAssembly の標準の型と関数（wasmLoad、wasmCall、WasmValue など）。UM で書いてある。
 * ウィンドウと同じく、プログラムがこのファイルの名前を使うときだけ加える（使わないプログラムは wasmtime をリンクしない）。
 */
const WASM_PRELUDE = join(import.meta.dirname, '..', 'runtime', 'wasm.um');
const RUNTIME_WASM_C = join(import.meta.dirname, '..', 'runtime', 'wasm_rt.c');

/** ファイル番号を、宣言とその中のメソッドに付ける。 */
function tag(d: Decl, file: number): void {
  d.file = file;
  if (d.kind === 'class') {
    if (d.ctor) d.ctor.file = file;
    for (const m of d.methods) m.file = file;
  }
}

/**
 * ソースを構文解析し、import されたファイルと標準の関数（runtime/prelude.um）を1つにまとめる。
 *
 * - import は `import { a, b } from "./x.um";` の形だけ。パスは相対で、最初のファイルのあるフォルダの外へは出られない。
 * - 名前はすべてのファイルで1つの空間を共有する（同じ名前を2つのファイルで定義するとエラー）。
 *   ただし他のファイルの名前を使うには、相手が `export` し、こちらが import する必要がある（型検査が確かめる）。
 * - wasm やネイティブの外へ公開されるのは、最初のファイルの `export function` だけ。
 * - 標準の関数は、プログラムが同じ名前を定義していれば加えない。
 */
export function parseProgram(src: string, file?: string): Module {
  const entry = parse(src);
  const merged: Module = { decls: [], importedNames: new Map() };
  const root = file ? dirname(realpathSync(resolve(file))) : undefined;
  const loaded = new Map<string, { id: number; mod: Module }>();
  if (file) loaded.set(realpathSync(resolve(file)), { id: 0, mod: entry });
  const texts = [src];

  const load = (mod: Module, id: number, dir: string | undefined): void => {
    for (const d of mod.decls) {
      tag(d, id);
      if (id !== 0 && d.kind === 'fn') d.exported = false;
      merged.decls.push(d);
    }
    const names = new Set<string>();
    merged.importedNames!.set(id, names);
    for (const imp of mod.imports ?? []) {
      if (!dir || !root) {
        throw new CompileError('import-path', 'imports need the program to be compiled from a file', imp.pos);
      }
      if (!/^\.\.?\//.test(imp.path) || !imp.path.endsWith('.um')) {
        throw new CompileError('import-path', `cannot import '${imp.path}'`, imp.pos, 'the path must be relative and end in .um, e.g. "./math.um"');
      }
      // シンボリックリンクをたどった先で判定する（フォルダの中のリンクが外を指していても読まない）。
      let abs = resolve(dir, imp.path);
      try {
        abs = realpathSync(abs);
      } catch {
        throw new CompileError('import-path', `cannot read '${imp.path}'`, imp.pos);
      }
      if (!abs.startsWith(root + sep)) {
        throw new CompileError('import-path', `'${imp.path}' is outside the folder of the main file`, imp.pos,
          'imported files must be in the same folder as the main file, or below it');
      }
      let target = loaded.get(abs);
      if (!target) {
        let text: string;
        try {
          text = readFileSync(abs, 'utf8');
          texts.push(text);
        } catch {
          throw new CompileError('import-path', `cannot read '${imp.path}'`, imp.pos);
        }
        // 診断に出すファイル名は、どの OS でも `lib/math.um` の形（区切りは /）。
        target = { id: loaded.size, mod: parse(text, relative(root, abs).split(sep).join('/')) };
        loaded.set(abs, target);
        load(target.mod, target.id, dirname(abs));
      }
      for (const n of imp.names) {
        const d = target.mod.decls.find((x) => x.name === n.name);
        if (!d) throw new CompileError('import-name', `'${imp.path}' has no '${n.name}'`, n.pos);
        if (!d.importable) {
          throw new CompileError('import-name', `'${n.name}' is not exported by '${imp.path}'`, n.pos, `write \`export\` in front of its declaration in '${imp.path}'`);
        }
        names.add(n.name);
      }
    }
  };
  load(entry, 0, root);

  // 同じ名前をプログラムが定義していれば、標準のものは入れない。定数は別:
  // 標準の関数・型と同じ名前の定数は、標準のものを黙って消さずに `duplicate` にする（型検査が報告する）。
  const defined = new Set(merged.decls.filter((d) => d.kind !== 'const').map((d) => d.name));
  const prelude = parse(readFileSync(PRELUDE, 'utf8'), 'prelude.um');
  const standard = [...prelude.decls];
  for (const [path, label] of [[WINDOW_PRELUDE, 'window.um'], [WASM_PRELUDE, 'wasm.um']]) {
    const extra = parse(readFileSync(path, 'utf8'), label);
    const names = extra.decls.map((d) => d.name).filter((n) => !defined.has(n));
    if (names.length === 0) continue;
    const used = new RegExp(`\\b(?:${names.join('|')})\\b`);
    if (texts.some((t) => used.test(t))) standard.push(...extra.decls);
  }
  for (const d of standard) {
    if (defined.has(d.name)) continue;
    tag(d, -1);
    merged.decls.push(d);
  }
  return merged;
}

/** ソースを検査し、LLVM IR（テキスト）を返す。エラーは CompileError / CompileErrors を投げる。 */
export function compileToLL(src: string, target: TargetName, sanitize: boolean | 'thread' = false, file?: string): string {
  const t = TARGETS[target];
  // GPU バックエンドがあるのは wasm32（WebGPU）と、UM_METAL=1 のときの Mac のネイティブ（Metal）。
  const metal = metalEnabled(target);
  // プラグインは GPU を使わない（kernel の dispatch は CPU の1スレッドで実行する）。
  const prog = check(parseProgram(src, file), t.ptrBits, target === 'wasm32' || metal);
  if (t.kind === 'native') {
    // 実行ファイルには入口が要る。
    const main = prog.fns.get('main')?.decl;
    if (!main || !main.exported || main.params.length !== 0) {
      throw new CompileError('no-main', 'a native program needs `export function main(): i32` with no parameters', main?.pos ?? { line: 1, col: 1 },
        'add `export function main(): i32 { ... return 0; }`');
    }
  }
  // SIMD の dot に NEON の sdot を直接使えるのは、macOS の arm64（Apple のチップはどれも dotprod を持つ）。
  // UM_GENERIC_SIMD=1 で汎用の経路にする（テストで両方の経路を確かめるため）。
  const appleArm = process.env.UM_GENERIC_SIMD !== '1' &&
    (target === 'arm64-macos' || (target === 'host' && process.platform === 'darwin' && process.arch === 'arm64'));
  const windows = target === 'host' && process.platform === 'win32';
  return generate(prog, t.kind, sanitize, metal, appleArm, windows, target === 'wasm32-plugin');
}

/**
 * dispatch できる kernel それぞれの WGSL を返す（確認用。wasm には同じものが埋め込まれる）。
 * 64 ビットの整数を使う kernel は WGSL にできない（WebGPU では CPU で実行する）ので含めない。
 */
export function compileWgsl(src: string, file?: string): Map<string, string> {
  const prog = check(parseProgram(src, file), 32, true);
  const out = new Map<string, string>();
  for (const sig of prog.fns.values()) {
    const d = sig.decl;
    const first = d.params[0]?.ty;
    if (d.isKernel && d.ret!.k === 'void' && first?.k === 'int' && first.bits === 32 && !first.signed && !kernelUses64(prog, d)) {
      try {
        // group.* を使う kernel は dispatchGroups 用の形にする。
        out.set(d.name, emitWgsl(prog, d, d.usesGroup === true).source);
      } catch (e) {
        // 定数の配列を使う kernel は WGSL にしない（wasm32 では CPU で実行する）。
        if (e instanceof CompileError && e.code === WEBGPU_CONST_ARRAY) continue;
        throw e;
      }
    }
  }
  return out;
}

/** dispatch できる kernel それぞれの MSL を返す（確認用。実行時の Metal は runtime/metal.m が同じ MSL をコンパイルする）。 */
export function compileMsl(src: string, file?: string): Map<string, string> {
  const prog = check(parseProgram(src, file), 64, true);
  const out = new Map<string, string>();
  for (const sig of prog.fns.values()) {
    const d = sig.decl;
    const first = d.params[0]?.ty;
    if (d.isKernel && d.ret!.k === 'void' && first?.k === 'int' && first.bits === 32 && !first.signed) {
      // group.* を使う kernel は dispatchGroups 用の形にする。
      out.set(d.name, emitMsl(prog, d, d.usesGroup === true));
    }
  }
  return out;
}

export interface BuildResult {
  output: string;
  /** リンクまで行わず、オブジェクトファイルで止めた場合 true。 */
  objectOnly: boolean;
}

// 外部コマンドは必ず execFile（シェルを通さない）で、引数を配列で渡す。
// ファイル名は絶対パスに直してから渡すので、'-' で始まる名前がオプションとして解釈されることはない。
function clang(args: string[], quiet = false, exe = CLANG): void {
  execFileSync(exe, ['-Wno-override-module', ...args], { stdio: ['ignore', 'inherit', quiet ? 'ignore' : 'inherit'] });
}

/**
 * sanitize はテスト用（quiet は clang のエラー出力を捨てる。対応状況の確認に使う）。true なら AddressSanitizer つきでビルドし、
 * 解放後使用・二重解放・解放漏れを実行時に検出する。'thread' なら ThreadSanitizer つき（データ競合を検出する）。
 * compileToLL にも同じ指定を渡すこと。
 */
export function build(ll: string, target: TargetName, outPath: string, optLevel = '2', sanitize: boolean | 'thread' = false, quiet = false): BuildResult {
  if (!/^[0-3sz]$/.test(optLevel)) throw new Error(`invalid optimization level: ${optLevel}`);
  const out = resolve(outPath);
  const dir = mkdtempSync(join(tmpdir(), 'um-'));
  try {
    const llPath = join(dir, 'module.ll');
    writeFileSync(llPath, ll, { mode: 0o600 });
    const opt = `-O${optLevel}`;

    if (target === 'wasm32' || target === 'wasm32-plugin') {
      const plugin = target === 'wasm32-plugin';
      clang([
        '--target=wasm32-unknown-unknown', opt, '-nostdlib', '-ffreestanding', '-fno-builtin',
        // メモリのコピーと初期化に wasm の命令（memory.copy / memory.fill）を使う。
        '-mbulk-memory',
        '-Wl,--no-entry',
        // スタックを線形メモリの先頭に置く。あふれたときにデータを壊さず、範囲外アクセスとして止まる。
        '-Wl,--stack-first',
        '-Wl,-z,stack-size=1048576',
        // プラグインでは、ランタイムが停止を import ではなくトラップにし、umAlloc / umFree を export する。
        // wasmtime はどの CPU でも SIMD を使えるので、ループをベクトル化できるよう simd128 を有効にする。
        ...(plugin ? ['-DUM_PLUGIN', '-msimd128'] : []),
        '-o', out, llPath, RUNTIME_WASM_C,
      ], quiet, clangFor('wasm32'));
      if (plugin) {
        const imports = wasmImports(readFileSync(out));
        if (imports.length > 0) {
          rmSync(out, { force: true });
          throw new CompileError('plugin-import', `a wasm32-plugin module cannot use host functions, but this program needs: ${imports.map(describeImport).join(', ')}`,
            { line: 1, col: 1 }, 'plugins run without any host functions: remove print, input / output, and conversions of f64 to string (String, template strings)');
        }
      }
      return { output: out, objectOnly: false };
    }

    if (target === 'arm64-macos' && process.platform !== 'darwin') {
      // macOS以外ではリンクに必要なSDKがないので、オブジェクトファイルまで。
      clang(['--target=arm64-apple-macos13', opt, '-c', '-o', out, llPath]);
      return { output: out, objectOnly: true };
    }

    if (process.platform === 'win32') {
      // Windows（clang の既定の MSVC ターゲット。リンクには Visual Studio か Build Tools の MSVC と Windows SDK を使う）。
      // スレッドは Win32 の API（runtime/win32.h）で、-pthread と libm は要らない。ASLR と DEP は MSVC のリンカの既定で有効。
      // メインスレッドのスタックは、Windows の既定の 1 MiB では探索の再帰が足りないので、macOS・Linux と同じ 8 MiB を予約する。
      const exe = exePath(out);
      const args = [opt, '-fstack-protector-strong', '-o', exe, llPath, RUNTIME_C, '-Wl,/STACK:8388608'];
      if (sanitize === 'thread') args.unshift('-fsanitize=thread', '-g');
      else if (sanitize) args.unshift('-fsanitize=address', '-g');
      const sdl = usesWindow(ll) ? findSdl3() : null;
      if (sdl) args.push(RUNTIME_WINDOW, RUNTIME_AUDIO, ...sdl.args);
      const wt = usesWasm(ll) ? findWasmtime() : null;
      if (wt) args.push(RUNTIME_WASMTIME, ...wt.args);
      clang(args, quiet);
      // SDL3.dll と wasmtime.dll は実行ファイルと同じ場所にないと起動できないので、隣に写す。
      if (sdl?.dll && existsSync(sdl.dll)) copyFileSync(sdl.dll, join(dirname(exe), 'SDL3.dll'));
      if (wt?.dll && existsSync(wt.dll)) copyFileSync(wt.dll, join(dirname(exe), 'wasmtime.dll'));
      return { output: exe, objectOnly: false };
    }

    const args = [opt, '-fstack-protector-strong', '-pthread', '-o', out, llPath, RUNTIME_C, '-lm'];
    if (target === 'arm64-macos') args.unshift('-arch', 'arm64');
    if (process.platform === 'linux') args.unshift('-fPIE', '-pie', '-Wl,-z,relro,-z,now', '-Wl,-z,noexecstack');
    if (sanitize === 'thread') args.unshift('-fsanitize=thread', '-fno-omit-frame-pointer', '-g');
    else if (sanitize) args.unshift('-fsanitize=address', '-fno-omit-frame-pointer', '-g');
    if (metalEnabled(target) && process.platform === 'darwin') {
      args.push('-fobjc-arc', RUNTIME_METAL, '-framework', 'Metal', '-framework', 'Foundation');
    }
    if (usesWindow(ll)) args.push(RUNTIME_WINDOW, RUNTIME_AUDIO, ...findSdl3().args);
    if (usesWasm(ll)) args.push(RUNTIME_WASMTIME, ...findWasmtime().args);
    clang(args, quiet);
    return { output: out, objectOnly: false };
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}
