#!/usr/bin/env node
import { copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { basename, join } from 'node:path';
import { CompileError, CompileErrors, formatDiagnostic } from './diag.ts';
import type { Diagnostic } from './diag.ts';
import { build, compileMsl, compileToLL, compileWgsl, TARGETS } from './driver.ts';
import type { TargetName } from './driver.ts';

const USAGE = `UnifiedMozaic compiler

usage:
  um check <file.um> [--target T] [--json]
  um build <file.um> [--target T | --web] [-o <out>] [--emit-llvm] [-O<level>] [--link-input <file>]... [--json]

targets (T): wasm32 (default), wasm32-plugin (a wasm module with no imports, to load from a host), arm64-macos, host
  --emit-llvm   write LLVM IR text instead of a binary
  --web         build for the browser: write a folder (-o, default <name>-web) with app.wasm, index.html and the runtime
                (host.mjs, web.mjs). Open index.html over HTTP; windows become canvases (docs/web.ja.md)
  --emit-wgsl   also write the WGSL of each kernel as <out>.<kernel>.wgsl (wasm32 and UM_WGPU=1 native builds embed the same
                WGSL; gpu.loadKernel can load it as is, see docs/wgpu.ja.md)
  --emit-msl    also write the MSL of each kernel as <out>.<kernel>.metal (UM_METAL=1 native builds compile the same MSL at run time)
  --link-input  add a file to the native link as is (repeatable): a Windows resource (.res, e.g. made by llvm-rc to embed
                an icon in the .exe), an object file or a library. Ignored, with a note, when the target does not link
                (wasm32, wasm32-plugin, arm64-macos on a non-Mac)
  --json        print diagnostics as JSON on stdout
`;

function fail(msg: string): never {
  process.stderr.write(msg + '\n');
  process.exit(2);
}

function main(argv: string[]): void {
  const [cmd, ...rest] = argv;
  if (!cmd || cmd === '--help' || cmd === '-h') {
    process.stdout.write(USAGE);
    return;
  }
  if (cmd !== 'check' && cmd !== 'build') fail(`unknown command '${cmd}'\n\n${USAGE}`);

  let file: string | undefined;
  let target: TargetName = 'wasm32';
  let out: string | undefined;
  let emitLLVM = false;
  let web = false;
  let emitWgsl = false;
  let emitMsl = false;
  let json = false;
  let opt = '2';
  const linkInputs: string[] = [];

  for (let i = 0; i < rest.length; i++) {
    const a = rest[i];
    if (a === '--target') {
      const t = rest[++i];
      if (!t || !(t in TARGETS)) fail(`unknown target '${t}'; expected one of: ${Object.keys(TARGETS).join(', ')}`);
      target = t as TargetName;
    } else if (a === '-o') {
      out = rest[++i];
      if (!out) fail('-o needs a path');
    } else if (a === '--emit-llvm') {
      emitLLVM = true;
    } else if (a === '--web') {
      web = true;
    } else if (a === '--emit-wgsl') {
      emitWgsl = true;
    } else if (a === '--emit-msl') {
      emitMsl = true;
    } else if (a === '--link-input') {
      const p = rest[++i];
      if (!p) fail('--link-input needs a path');
      if (!existsSync(p)) fail(`cannot read the link input '${p}'`);
      linkInputs.push(p);
    } else if (a === '--json') {
      json = true;
    } else if (/^-O[0-3sz]$/.test(a)) {
      opt = a.slice(2);
    } else if (a.startsWith('-')) {
      fail(`unknown option '${a}'`);
    } else if (file === undefined) {
      file = a;
    } else {
      fail('only one source file is supported');
    }
  }
  if (!file) fail(`missing source file\n\n${USAGE}`);
  if (web) target = 'wasm32';

  let src: string;
  try {
    src = readFileSync(file, 'utf8');
  } catch {
    fail(`cannot read '${file}'`);
  }

  let ll: string;
  try {
    ll = compileToLL(src, target, false, file);
  } catch (e) {
    const errs = e instanceof CompileErrors ? e.errors : e instanceof CompileError ? [e] : null;
    if (!errs) throw e;
    const diags: Diagnostic[] = errs.map((x) => x.toDiagnostic(file));
    if (json) process.stdout.write(JSON.stringify({ ok: false, errors: diags }, null, 2) + '\n');
    else process.stderr.write(diags.map(formatDiagnostic).join('\n') + '\n');
    process.exit(1);
  }

  if (cmd === 'check') {
    if (json) process.stdout.write(JSON.stringify({ ok: true, errors: [] }) + '\n');
    return;
  }

  const stem = basename(file).replace(/\.um$/, '');
  if (emitLLVM) {
    const path = out ?? `${stem}.ll`;
    writeFileSync(path, ll);
    if (json) process.stdout.write(JSON.stringify({ ok: true, errors: [], output: path }) + '\n');
    return;
  }

  if (web) {
    // ブラウザで動かすフォルダ: app.wasm と、読み込むページ・ランタイム（runtime/web/index.html、host.mjs、web.mjs）。
    const dir = out ?? `${stem}-web`;
    mkdirSync(dir, { recursive: true });
    try {
      build(ll, target, join(dir, 'app.wasm'), opt);
    } catch {
      fail('error: the LLVM toolchain (clang) failed; see the messages above');
    }
    const rt = join(import.meta.dirname, '..', 'runtime');
    copyFileSync(join(rt, 'web', 'index.html'), join(dir, 'index.html'));
    copyFileSync(join(rt, 'host.mjs'), join(dir, 'host.mjs'));
    copyFileSync(join(rt, 'web.mjs'), join(dir, 'web.mjs'));
    if (json) process.stdout.write(JSON.stringify({ ok: true, errors: [], output: dir }) + '\n');
    else process.stderr.write(`wrote ${dir}/ (open index.html over HTTP, e.g. \`npx serve ${dir}\` or \`python3 -m http.server -d ${dir}\`)\n`);
    return;
  }

  const objectOnly = target === 'arm64-macos' && process.platform !== 'darwin';
  const path = out ?? (target === 'wasm32' || target === 'wasm32-plugin' ? `${stem}.wasm` : objectOnly ? `${stem}.o` : stem);
  if (emitWgsl) {
    for (const [name, text] of compileWgsl(src, file, TARGETS[target].ptrBits)) writeFileSync(`${path}.${name}.wgsl`, text);
  }
  if (emitMsl) {
    for (const [name, text] of compileMsl(src, file)) writeFileSync(`${path}.${name}.metal`, text);
  }
  let res;
  try {
    res = build(ll, target, path, opt, false, false, linkInputs);
  } catch {
    // clang 自身のエラー出力はすでに stderr に出ている。
    fail('error: the LLVM toolchain (clang) failed; see the messages above');
  }
  if (res.linkInputsIgnored && !json) {
    process.stderr.write(`note: --link-input is ignored for --target ${target}, which does not link a native executable\n`);
  }
  if (json) {
    process.stdout.write(JSON.stringify({ ok: true, errors: [], output: res.output, objectOnly: res.objectOnly, linkInputsIgnored: res.linkInputsIgnored === true }) + '\n');
  } else if (res.objectOnly) {
    process.stderr.write(`note: linking for macOS needs a Mac; wrote an object file: ${res.output}\n`);
  }
}

main(process.argv.slice(2));
