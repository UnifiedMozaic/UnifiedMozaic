import { execFileSync } from 'node:child_process';
import { existsSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, relative, resolve, sep } from 'node:path';
import { check } from './checker.ts';
import { generate, kernelUses64 } from './codegen.ts';
import { emitMsl } from './msl.ts';
import { emitWgsl, WEBGPU_CONST_ARRAY } from './wgsl.ts';
import type { TargetKind } from './codegen.ts';
import { parse } from './parser.ts';
import type { Decl, Module } from './ast.ts';
import { CompileError } from './diag.ts';

export type TargetName = 'wasm32' | 'arm64-macos' | 'host';

export const TARGETS: Record<TargetName, { kind: TargetKind; ptrBits: 32 | 64 }> = {
  wasm32: { kind: 'wasm32', ptrBits: 32 },
  'arm64-macos': { kind: 'native', ptrBits: 64 },
  host: { kind: 'native', ptrBits: 64 },
};

const RUNTIME_C = join(import.meta.dirname, '..', 'runtime', 'native.c');
/**
 * 使う clang。環境変数 UM_CLANG で差し替えられる（macOS の標準の clang は wasm32 を出力できないので、
 * Homebrew の LLVM を指定するときなどに使う）。
 */
export const CLANG = process.env.UM_CLANG || 'clang';

/** macOS で Homebrew の LLVM が入っている場所（Apple Silicon と Intel）。 */
const BREW_CLANGS = ['/opt/homebrew/opt/llvm/bin/clang', '/usr/local/opt/llvm/bin/clang'];

/**
 * そのターゲットに使う clang。UM_CLANG があればそれ。macOS の wasm32 は、標準の clang が出力できないので、
 * Homebrew の LLVM があれば自動でそちらを使う（`brew install llvm lld`）。ほかは PATH の clang。
 */
export function clangFor(target: TargetName): string {
  if (process.env.UM_CLANG) return process.env.UM_CLANG;
  if (target === 'wasm32' && process.platform === 'darwin') {
    for (const c of BREW_CLANGS) if (existsSync(c)) return c;
  }
  return 'clang';
}
const RUNTIME_METAL = join(import.meta.dirname, '..', 'runtime', 'metal.m');

/**
 * Metal（macOS の GPU）で kernel を実行する経路を使うか。
 * 環境変数 UM_METAL=1 を付けたときだけ有効にする（Metal と Foundation のフレームワークをリンクするため）。
 */
export function metalEnabled(target: TargetName): boolean {
  if (process.env.UM_METAL !== '1') return false;
  return target === 'arm64-macos' || (target === 'host' && process.platform === 'darwin');
}

const PRELUDE = join(import.meta.dirname, '..', 'runtime', 'prelude.um');
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
        } catch {
          throw new CompileError('import-path', `cannot read '${imp.path}'`, imp.pos);
        }
        target = { id: loaded.size, mod: parse(text, relative(root, abs)) };
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
  for (const d of prelude.decls) {
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
  return generate(prog, t.kind, sanitize, metal, appleArm);
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

    if (target === 'wasm32') {
      clang([
        '--target=wasm32-unknown-unknown', opt, '-nostdlib', '-ffreestanding', '-fno-builtin',
        // メモリのコピーと初期化に wasm の命令（memory.copy / memory.fill）を使う。
        '-mbulk-memory',
        '-Wl,--no-entry',
        // スタックを線形メモリの先頭に置く。あふれたときにデータを壊さず、範囲外アクセスとして止まる。
        '-Wl,--stack-first',
        '-Wl,-z,stack-size=1048576',
        '-o', out, llPath, RUNTIME_WASM_C,
      ], quiet, clangFor('wasm32'));
      return { output: out, objectOnly: false };
    }

    if (target === 'arm64-macos' && process.platform !== 'darwin') {
      // macOS以外ではリンクに必要なSDKがないので、オブジェクトファイルまで。
      clang(['--target=arm64-apple-macos13', opt, '-c', '-o', out, llPath]);
      return { output: out, objectOnly: true };
    }

    const args = [opt, '-fstack-protector-strong', '-pthread', '-o', out, llPath, RUNTIME_C, '-lm'];
    if (target === 'arm64-macos') args.unshift('-arch', 'arm64');
    if (process.platform === 'linux') args.unshift('-fPIE', '-pie', '-Wl,-z,relro,-z,now', '-Wl,-z,noexecstack');
    if (sanitize === 'thread') args.unshift('-fsanitize=thread', '-fno-omit-frame-pointer', '-g');
    else if (sanitize) args.unshift('-fsanitize=address', '-fno-omit-frame-pointer', '-g');
    if (metalEnabled(target) && process.platform === 'darwin') {
      args.push('-fobjc-arc', RUNTIME_METAL, '-framework', 'Metal', '-framework', 'Foundation');
    }
    clang(args, quiet);
    return { output: out, objectOnly: false };
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}
