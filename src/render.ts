// 描画の組み込み（GPU の三角形のラスタライザ。docs/render.ja.md）。UM_WGPU=1 のネイティブで使える。
//
//   gpu.beginImage(width: i32, height: i32, clear: u32): i32
//       width × height の画像（色と深度）を GPU に用意し、色を clear（0xAARRGGBB）で、深度を 1 で埋める。番号を返す。
//   gpu.draw(image: i32, vertexCount: u32, vs(引数...), fs(引数...)): void
//       vertex function vs を頂点の番号 0 から vertexCount 未満について実行し、3 つずつの頂点を三角形として
//       画像に塗る。画素の色は fragment function fs が決める。深度（position.z / position.w）が小さいものが残る。
//       vs(...) / fs(...) は呼び出しではなく、それぞれの引数の指定（vs は頂点の番号、fs は受け取る値を除いた引数）。
//   gpu.endImage(image: i32, mut pixels: Buffer<u32>): void
//       描き終えた画像の色を pixels に写し（0xAARRGGBB、行ごとに上から）、画像を片づける。
//
// 型検査（このファイル）は呼び出しを `__gpu_image_begin` / `__gpu_draw` / `__gpu_image_end` の呼び出しに書き換え、
// コード生成（codegen.ts の renderCall）がランタイム（runtime/wgpu.c の um_gpu_image_begin など）の呼び出しにする。
// __gpu_draw の引数は [image, vertexCount, vs の引数..., fs の引数...] で、vs と fs の名前と vs の引数の数を節に持つ。

import { tyEq, tyStr } from './ast.ts';
import type { Arg, Expr, FnDecl, Ty } from './ast.ts';
import type { Pos } from './diag.ts';
import type { FnSig } from './checker.ts';

/** 型検査器のうち、ここで使うもの（checker.ts の Checker）。 */
export interface RenderChecker {
  noKernel(what: string, pos: Pos, hint?: string): void;
  err(code: string, message: string, pos: Pos, hint?: string): never;
  checkArgs(sig: FnSig, args: Arg[], pos: Pos, recv?: Expr, kernelCall?: boolean): void;
  fns: Map<string, FnSig>;
  visible(name: string, pos: Pos): void;
}

const I32: Ty = { k: 'int', bits: 32, signed: true };
const U32: Ty = { k: 'int', bits: 32, signed: false };
const VOID: Ty = { k: 'void' };
const PIXELS: Ty = { k: 'heap', elem: U32, buffer: true };

export const RENDER_NAMES = ['beginImage', 'draw', 'endImage'];

/** `gpu.beginImage` / `gpu.draw` / `gpu.endImage` の呼び出しか（同じ名前の変数があれば違う）。 */
export function isRenderCall(e: Expr & { kind: 'mcall' }, lookup: (name: string) => unknown): boolean {
  return e.recv.kind === 'ident' && e.recv.name === 'gpu' && !lookup('gpu') && RENDER_NAMES.includes(e.name);
}

/** 引数の決まった組み込みの呼び出しを検査する。 */
function builtin(c: RenderChecker, name: string, params: { name: string; mode: 'none' | 'mut'; ty: Ty }[], ret: Ty, args: Arg[], pos: Pos): void {
  const decl: FnDecl = { kind: 'fn', name, exported: false, params: [], body: [], pos };
  c.checkArgs({ key: name, name, params, ret, exported: false, decl }, args, pos);
}

/** `vs(...)` / `fs(...)` の指定を検査し、シェーダの関数を返す。 */
function stageArg(c: RenderChecker, a: Arg, stage: 'vertex' | 'fragment'): { sig: FnSig; args: Arg[] } {
  const e = a.expr;
  const example = stage === 'vertex' ? '`vs(verts, camera)`' : '`fs(camera)`';
  if (a.mode !== 'none' || e.kind !== 'call') {
    c.err('type-mismatch', `this argument of \`gpu.draw\` must name the ${stage} function with its arguments, e.g. ${example}`, e.pos);
  }
  const sig = c.fns.get(e.callee);
  if (sig) c.visible(e.callee, e.pos);
  if (!sig || sig.decl.stage !== stage) {
    c.err('type-mismatch', `'${e.callee}' is not a ${stage} function`, e.pos,
      `declare it as \`${stage} function ${e.callee}(${stage === 'vertex' ? 'i: u32' : 'v: VOut'}, ...): ${stage === 'vertex' ? 'VOut' : 'vec4<f32>'}\``);
  }
  // 第1引数（頂点の番号 / 受け取る値）は GPU が渡すので、残りの引数を検査する。
  const rest: FnSig = { ...sig, params: sig.params.slice(1) };
  c.checkArgs(rest, e.args, e.pos, undefined, true);
  return { sig, args: e.args };
}

export function inferRender(c: RenderChecker, e: Expr & { kind: 'mcall' }): Ty {
  c.noKernel(`\`gpu.${e.name}\``, e.pos);
  const node = e as unknown as { kind: string; callee: string; args: Arg[]; vs?: string; fs?: string; vsArgc?: number };
  if (e.name === 'beginImage') {
    builtin(c, 'gpu.beginImage', [{ name: 'width', mode: 'none', ty: I32 }, { name: 'height', mode: 'none', ty: I32 }, { name: 'clear', mode: 'none', ty: U32 }],
      I32, e.args, e.pos);
    node.kind = 'call';
    node.callee = '__gpu_image_begin';
    return I32;
  }
  if (e.name === 'endImage') {
    builtin(c, 'gpu.endImage', [{ name: 'image', mode: 'none', ty: I32 }, { name: 'pixels', mode: 'mut', ty: PIXELS }], VOID, e.args, e.pos);
    node.kind = 'call';
    node.callee = '__gpu_image_end';
    return VOID;
  }
  // gpu.draw(image, vertexCount, vs(...), fs(...))
  if (e.args.length !== 4) {
    c.err('arg-count', `\`gpu.draw\` takes 4 arguments but ${e.args.length} were given`, e.pos,
      'write `gpu.draw(image, vertexCount, vs(...), fs(...))`');
  }
  builtin(c, 'gpu.draw', [{ name: 'image', mode: 'none', ty: I32 }, { name: 'vertexCount', mode: 'none', ty: U32 }], VOID, e.args.slice(0, 2), e.pos);
  const v = stageArg(c, e.args[2], 'vertex');
  const f = stageArg(c, e.args[3], 'fragment');
  const out = v.sig.ret;
  const got = f.sig.params[0].ty;
  if (!tyEq(out, got)) {
    c.err('type-mismatch', `fragment function '${f.sig.name}' takes ${tyStr(got)}, but vertex function '${v.sig.name}' returns ${tyStr(out)}`, e.args[3].expr.pos,
      'the fragment function receives what the vertex function returns');
  }
  node.kind = 'call';
  node.callee = '__gpu_draw';
  node.vs = v.sig.name;
  node.fs = f.sig.name;
  node.vsArgc = v.args.length;
  node.args = [e.args[0], e.args[1], ...v.args, ...f.args];
  return VOID;
}
