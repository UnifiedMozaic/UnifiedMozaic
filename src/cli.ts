#!/usr/bin/env node
import { readFileSync, writeFileSync } from 'node:fs';
import { basename } from 'node:path';
import { CompileError, CompileErrors, formatDiagnostic } from './diag.ts';
import type { Diagnostic } from './diag.ts';
import { build, compileMsl, compileToLL, compileWgsl, TARGETS } from './driver.ts';
import type { TargetName } from './driver.ts';

const USAGE = `UnifiedMozaic compiler

usage:
  um check <file.um> [--target T] [--json]
  um build <file.um> [--target T] [-o <out>] [--emit-llvm] [-O<level>] [--json]

targets (T): wasm32 (default), wasm32-plugin (a wasm module with no imports, to load from a host), arm64-macos, host
  --emit-llvm   write LLVM IR text instead of a binary
  --emit-wgsl   also write the WGSL of each kernel as <out>.<kernel>.wgsl (wasm32 and UM_WGPU=1 native builds embed the same
                WGSL; gpu.loadKernel can load it as is, see docs/wgpu.ja.md)
  --emit-msl    also write the MSL of each kernel as <out>.<kernel>.metal (UM_METAL=1 native builds compile the same MSL at run time)
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
  let emitWgsl = false;
  let emitMsl = false;
  let json = false;
  let opt = '2';

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
    } else if (a === '--emit-wgsl') {
      emitWgsl = true;
    } else if (a === '--emit-msl') {
      emitMsl = true;
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
    res = build(ll, target, path, opt);
  } catch {
    // clang 自身のエラー出力はすでに stderr に出ている。
    fail('error: the LLVM toolchain (clang) failed; see the messages above');
  }
  if (json) {
    process.stdout.write(JSON.stringify({ ok: true, errors: [], output: res.output, objectOnly: res.objectOnly }) + '\n');
  } else if (res.objectOnly) {
    process.stderr.write(`note: linking for macOS needs a Mac; wrote an object file: ${res.output}\n`);
  }
}

main(process.argv.slice(2));
