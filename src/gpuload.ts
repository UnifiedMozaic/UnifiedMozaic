// 実行時に kernel を読み込む組み込み（UM_WGPU=1 のネイティブ。docs/wgpu.ja.md の「実行時に kernel を読み込む」）。
//
//   gpu.loadKernel(wgsl: string): u32 | null
//       WGSL のソースを GPU に読み込み、kernel の番号を返す。GPU が使えないか、WGSL をコンパイルできなければ null。
//   gpu.runKernel(kernel: u32, count: u32, ...引数): bool
//       読み込んだ kernel を添字 0 から count 未満について GPU で実行する。引数は dispatch と同じ書き方で、
//       Buffer / RacyBuffer（書くなら `mut`）と値を、kernel の引数の順に並べる。実行を頼めたら true。
//       実行できなければ（GPU がない、束縛の並びが kernel と合わない）false で、Buffer は変わらない。
//
// 引数の並び（束縛）は、同じ型と並びの引数を持つ UM の kernel を `--emit-wgsl` で WGSL にしたときと同じ:
// binding 0 が UmParams（個数、1行の要素数、Buffer ごとの長さ、値の引数）、binding 1 からが Buffer。
// 実行は dispatch と同じく、結果を CPU が読むときに（必要なら）読み戻す。
//
// 型検査（このファイル）は呼び出しを `__gpu_load` / `__gpu_run` の呼び出しに書き換え、コード生成（codegen.ts の gpuLoadCall）が
// ランタイム（runtime/wgpu.c の um_gpu_load / um_gpu_run）の呼び出しにする。UM_WGPU=1 でないビルドでは、
// loadKernel はいつも null、runKernel はいつも false。

import { tyStr } from './ast.ts';
import type { Arg, Expr, FnDecl, Mode, Ty } from './ast.ts';
import type { Pos } from './diag.ts';
import type { FnSig } from './checker.ts';

/** 型検査器のうち、ここで使うもの（checker.ts の Checker）。 */
export interface GpuLoadChecker {
  noKernel(what: string, pos: Pos, hint?: string): void;
  err(code: string, message: string, pos: Pos, hint?: string): never;
  checkExpr(e: Expr, want?: Ty): Ty;
  checkArgs(sig: FnSig, args: Arg[], pos: Pos, recv?: Expr, kernelCall?: boolean): void;
  optionalOf(inner: Ty, pos: Pos): Ty;
  isShareable(t: Ty): boolean;
  structs: Map<string, { fields: { ty: Ty }[] }>;
}

const U32: Ty = { k: 'int', bits: 32, signed: false };
const BOOL: Ty = { k: 'bool' };
const STR: Ty = { k: 'heap', elem: { k: 'int', bits: 8, signed: false }, str: true };

/** `gpu.loadKernel` / `gpu.runKernel` の呼び出しか（同じ名前の変数があれば違う）。 */
export function isGpuLoadCall(e: Expr & { kind: 'mcall' }, lookup: (name: string) => unknown): boolean {
  return e.recv.kind === 'ident' && e.recv.name === 'gpu' && !lookup('gpu') && (e.name === 'loadKernel' || e.name === 'runKernel');
}

/** WGSL（32 ビットの値）で GPU に渡せる型か。8 / 16 ビットの整数は Buffer の要素としてだけ渡せる。 */
function wgslOk(c: GpuLoadChecker, t: Ty, elem: boolean): string | null {
  if (t.k === 'int') {
    if (t.bits === 64) return 'WGSL has no 64-bit integers';
    if (t.bits < 32 && !elem) return `${tyStr(t)} values cannot be passed directly; pass a \`${t.signed ? 'i32' : 'u32'}\``;
    return null;
  }
  if (t.k === 'float') return t.bits === 32 ? null : `WGSL kernels use f32, not ${tyStr(t)}`;
  if (t.k === 'struct' && hasWide(c, t)) return `${tyStr(t)} contains a type that WGSL does not have (64-bit integers or a float other than f32)`;
  if (t.k === 'bool') return elem ? 'a Buffer of bool cannot be sent to the GPU; use u32' : null;
  if (t.k === 'array' || t.k === 'struct') {
    if (!c.isShareable(t)) return `${tyStr(t)} cannot be sent to the GPU (it must not contain bool)`;
    return hasWide(c, t) ? `${tyStr(t)} contains a type that WGSL does not have (64-bit integers or a float other than f32)` : null;
  }
  return `${tyStr(t)} cannot be sent to the GPU`;
}

/** 型に 64 ビットの整数か f32 以外の浮動小数点が含まれるか。 */
function hasWide(c: GpuLoadChecker, t: Ty): boolean {
  if (t.k === 'int') return t.bits === 64;
  if (t.k === 'float') return t.bits !== 32;
  if (t.k === 'array') return hasWide(c, t.elem);
  if (t.k === 'struct') return (c.structs.get(t.name)?.fields ?? []).some((f) => hasWide(c, f.ty));
  return false;
}

export function inferGpuLoad(c: GpuLoadChecker, e: Expr & { kind: 'mcall' }): Ty {
  c.noKernel(`\`gpu.${e.name}\``, e.pos);
  const decl: FnDecl = { kind: 'fn', name: `gpu.${e.name}`, exported: false, params: [], body: [], pos: e.pos };
  const node = e as unknown as { kind: string; callee: string; args: Arg[] };
  if (e.name === 'loadKernel') {
    const ret = c.optionalOf(U32, e.pos);
    const sig: FnSig = { key: decl.name, name: decl.name, params: [{ name: 'wgsl', mode: 'none', ty: STR }], ret, exported: false, decl };
    c.checkArgs(sig, e.args, e.pos);
    node.kind = 'call';
    node.callee = '__gpu_load';
    return ret;
  }

  if (e.args.length < 2) {
    c.err('arg-count', '`gpu.runKernel` needs a kernel (from `gpu.loadKernel`) and an element count', e.pos,
      'e.g. `gpu.runKernel(k, w * h, src, mut dst, w, h, params)`; the other arguments are those of the kernel, in order');
  }
  const params: { name: string; mode: Mode; ty: Ty }[] = [
    { name: 'kernel', mode: 'none', ty: U32 },
    { name: 'count', mode: 'none', ty: U32 },
  ];
  e.args.slice(2).forEach((a, i) => {
    const t = c.checkExpr(a.expr);
    if (t.k === 'heap') {
      if (!(t.buffer || t.racy) || t.str) {
        c.err('type-mismatch', `\`gpu.runKernel\` takes Buffers, RacyBuffers and values, not ${tyStr(t)}`, a.expr.pos,
          'copy the data into a `Buffer<T>` first');
      }
      if (a.mode === 'own') c.err('arg-mode', 'a Buffer is lent to the kernel; pass it without `own` (with `mut` if the kernel writes it)', a.expr.pos);
      const bad = wgslOk(c, t.elem, true);
      if (bad) c.err('kernel-type', `this Buffer cannot be used by a WGSL kernel: ${bad}`, a.expr.pos);
    } else {
      if (a.mode !== 'none') c.err('arg-mode', 'values are passed by value; remove the mode', a.expr.pos);
      const lit = (a.expr.kind === 'float' && t.k === 'float' && t.bits === 64) || (a.expr.kind === 'int' && t.k === 'int' && t.bits === 64);
      const bad = wgslOk(c, t, false);
      if (bad) {
        c.err('kernel-type', `this value cannot be passed to a WGSL kernel: ${bad}`, a.expr.pos,
          lit ? 'write the type of a literal, e.g. `0.5 as f32` or `3 as u32`' : undefined);
      }
    }
    params.push({ name: `argument ${i + 3}`, mode: a.mode, ty: t });
  });
  const sig: FnSig = { key: decl.name, name: decl.name, params, ret: BOOL, exported: false, decl };
  c.checkArgs(sig, e.args, e.pos, undefined, true);
  node.kind = 'call';
  node.callee = '__gpu_run';
  return BOOL;
}
