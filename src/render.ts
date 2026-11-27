// 描画の組み込み（三角形のラスタライザ。docs/render.ja.md）。
//
//   gpu.beginImage(width: i32, height: i32, clear: u32): GpuImage
//       width × height の画像（色と深度）を用意し、色を clear（0xAARRGGBB）で、深度を 1 で埋める。
//       GPU が使えれば（UM_WGPU=1 のビルドで GPU があれば）GPU に、なければ CPU に用意する。
//   gpu.draw(mut image: GpuImage, vertexCount: u32, vs(引数...), fs(引数...)): void
//       vertex function vs を頂点の番号 0 から vertexCount 未満について実行し、3 つずつの頂点を三角形として
//       画像に塗る。画素の色は fragment function fs が決める。深度（position.z / position.w）が小さいものが残る。
//       vs(...) / fs(...) は呼び出しではなく、それぞれの引数の指定（vs は頂点の番号、fs は受け取る値を除いた引数）。
//   gpu.endImage(mut image: GpuImage, mut pixels: Buffer<u32>): void
//       描き終えた画像の色を pixels に写す（0xAARRGGBB、行ごとに上から）。
//
// gpu.beginImage / gpu.endImage は標準の関数 gpuBeginImage / gpuEndImage（runtime/render.um）の呼び出しになる。
// gpu.draw は、vs と fs の組ごとにコンパイラが作る関数 umRenderDraw__vs__fs（drawSource。ドライバが型検査の前に加える）の
// 呼び出しになる。その関数は、画像が GPU にあれば gpu.rawDraw を呼び、なければ CPU で同じ描画を行う。
// gpu.rawBeginImage / gpu.rawDraw / gpu.rawEndImage（GPU の画像の番号を使う）は、これらの標準の関数の中でだけ使える。
// 型検査はそれらを `__gpu_image_begin` / `__gpu_draw` / `__gpu_image_end` の呼び出しに書き換え、コード生成
// （codegen.ts の renderCall）がランタイム（runtime/wgpu.c）の呼び出しにする。wgpu の経路のないビルドでは、
// __gpu_image_begin はいつも 0（CPU で描く）。
// __gpu_draw の引数は [image, vertexCount, vs の引数..., fs の引数...] で、vs と fs の名前と vs の引数の数を節に持つ。

import { tyEq, tyStr } from './ast.ts';
import type { Arg, Decl, Expr, FnDecl, StructDecl, Ty, TypeNode } from './ast.ts';
import type { Pos } from './diag.ts';
import type { FnSig } from './checker.ts';

/** 型検査器のうち、ここで使うもの（checker.ts の Checker）。 */
export interface RenderChecker {
  noKernel(what: string, pos: Pos, hint?: string): void;
  err(code: string, message: string, pos: Pos, hint?: string): never;
  checkArgs(sig: FnSig, args: Arg[], pos: Pos, recv?: Expr, kernelCall?: boolean): void;
  fns: Map<string, FnSig>;
  visible(name: string, pos: Pos): void;
  checkExpr(e: Expr, want?: Ty): Ty;
}

const I32: Ty = { k: 'int', bits: 32, signed: true };
const U32: Ty = { k: 'int', bits: 32, signed: false };
const VOID: Ty = { k: 'void' };
const PIXELS: Ty = { k: 'heap', elem: U32, buffer: true };
const IMAGE: Ty = { k: 'struct', name: 'GpuImage' };

/** 標準の関数（runtime/render.um）と、コンパイラが作る描画の関数のファイルの名前。gpu.raw* はこの中でだけ使える。 */
export const RENDER_FILES = ['render.um', 'render-draw.um'];

export const RENDER_NAMES = ['beginImage', 'draw', 'drawIndexed', 'endImage', 'beginFrame', 'endFrame', 'rawBeginImage', 'rawDraw', 'rawDrawIndexed', 'rawEndImage', 'rawBeginFrame', 'rawEndFrame'];

/** `gpu.beginImage` / `gpu.draw` / `gpu.endImage`（と gpu.raw*）の呼び出しか（同じ名前の変数があれば違う）。 */
export function isRenderCall(e: Expr & { kind: 'mcall' }, lookup: (name: string) => unknown): boolean {
  return e.recv.kind === 'ident' && e.recv.name === 'gpu' && !lookup('gpu') && RENDER_NAMES.includes(e.name);
}

/** 引数の決まった組み込みの呼び出しを検査する。 */
function builtin(c: RenderChecker, name: string, params: { name: string; mode: 'none' | 'mut'; ty: Ty }[], ret: Ty, args: Arg[], pos: Pos): void {
  const decl: FnDecl = { kind: 'fn', name, exported: false, params: [], body: [], pos };
  c.checkArgs({ key: name, name, params, ret, exported: false, decl }, args, pos);
}

/** `vs(...)` / `fs(...)` の指定を検査し、シェーダの関数を返す。 */
/**
 * vs(...) / fs(...) の指定を検査し、シェーダの関数を返す。skip は GPU が渡す先頭の引数の数
 * （fragment は受け取る値の 1 つ、vertex は頂点の番号の 1 つ、インスタンスのある描画では番号とインスタンスの番号の 2 つ）。
 * skip が null なら（標準の関数の中の gpu.rawDraw）、渡された引数の数から決める。
 */
function stageArg(c: RenderChecker, a: Arg, stage: 'vertex' | 'fragment', skip: number | null = 1): { sig: FnSig; args: Arg[]; instanced: boolean } {
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
  const n = skip ?? (e.args.length === sig.params.length - 2 ? 2 : 1);
  if (n === 2) {
    const inst = sig.params[1];
    if (!inst || !tyEq(inst.ty, U32)) {
      c.err('type-mismatch', `an instanced draw passes the instance number to the vertex function, but '${sig.name}' does not take it`, e.pos,
        `declare it as the second parameter: \`vertex function ${sig.name}(i: u32, inst: u32, ...)\``);
    }
  }
  // 先頭の引数（頂点の番号・インスタンスの番号 / 受け取る値）は GPU が渡すので、残りの引数を検査する。
  const rest: FnSig = { ...sig, params: sig.params.slice(n) };
  c.checkArgs(rest, e.args, e.pos, undefined, true);
  return { sig, args: e.args, instanced: n === 2 };
}

/** 描画の形。indexed: 頂点の番号を Buffer<u32> から読む（gpu.drawIndexed）。instanced: インスタンスの数を渡す。 */
export interface DrawKind {
  indexed: boolean;
  instanced: boolean;
}

function kindTag(k: DrawKind): string {
  return `${k.indexed ? 'X' : ''}${k.instanced ? 'I' : ''}`;
}

export function inferRender(c: RenderChecker, e: Expr & { kind: 'mcall' }): Ty {
  c.noKernel(`\`gpu.${e.name}\``, e.pos);
  const node = e as unknown as { kind: string; callee: string; args: Arg[]; vs?: string; fs?: string; vsArgc?: number };
  if (e.name.startsWith('raw')) {
    if (!RENDER_FILES.includes(e.pos.file ?? '')) {
      c.err('unknown-method', `\`gpu.${e.name}\` is not available`, e.pos, `use \`gpu.${e.name[3].toLowerCase()}${e.name.slice(4)}\``);
    }
    return inferRaw(c, e, node);
  }
  if (e.name === 'beginImage' || e.name === 'endImage' || e.name === 'beginFrame' || e.name === 'endFrame') {
    // 標準の関数（runtime/render.um と render_window.um）の呼び出しにする。引数の検査はその関数の型で行う。
    node.kind = 'call';
    node.callee = `gpu${e.name[0].toUpperCase()}${e.name.slice(1)}`;
    return c.checkExpr(e as unknown as Expr);
  }
  // gpu.draw(mut image, vertexCount, [instances,] vs(...), fs(...)) / gpu.drawIndexed(mut image, indices, [instances,] vs(...), fs(...))
  const indexed = e.name === 'drawIndexed';
  const usage = indexed ? '`gpu.drawIndexed(mut image, indices, vs(...), fs(...))` or `gpu.drawIndexed(mut image, indices, instances, vs(...), fs(...))`'
    : '`gpu.draw(mut image, vertexCount, vs(...), fs(...))` or `gpu.draw(mut image, vertexCount, instances, vs(...), fs(...))`';
  if (e.args.length !== 4 && e.args.length !== 5) {
    c.err('arg-count', `\`gpu.${e.name}\` takes 4 or 5 arguments but ${e.args.length} were given`, e.pos, `write ${usage}`);
  }
  const instanced = e.args.length === 5;
  const head: { name: string; mode: 'none' | 'mut'; ty: Ty }[] = [{ name: 'image', mode: 'mut', ty: IMAGE },
    indexed ? { name: 'indices', mode: 'none', ty: PIXELS } : { name: 'vertexCount', mode: 'none', ty: U32 }];
  if (instanced) head.push({ name: 'instances', mode: 'none', ty: U32 });
  builtin(c, `gpu.${e.name}`, head, VOID, e.args.slice(0, head.length), e.pos);
  const v = stageArg(c, e.args[head.length], 'vertex', instanced ? 2 : 1);
  const f = stageArg(c, e.args[head.length + 1], 'fragment');
  matchStages(c, v.sig, f.sig, e.args[head.length + 1].expr.pos);
  // vs と fs の組と描画の形ごとの描画の関数（drawSource。GPU か CPU で描く）の呼び出しにする。
  node.kind = 'call';
  node.callee = drawName(v.sig.name, f.sig.name, { indexed, instanced });
  node.args = [...e.args.slice(0, head.length), ...v.args, ...f.args];
  return c.checkExpr(e as unknown as Expr);
}

function matchStages(c: RenderChecker, vs: FnSig, fs: FnSig, pos: Pos): void {
  const out = vs.ret;
  const got = fs.params[0].ty;
  if (!tyEq(out, got)) {
    c.err('type-mismatch', `fragment function '${fs.name}' takes ${tyStr(got)}, but vertex function '${vs.name}' returns ${tyStr(out)}`, pos,
      'the fragment function receives what the vertex function returns');
  }
}

/** gpu.rawBeginImage / gpu.rawDraw / gpu.rawEndImage（GPU の画像の番号を使う。標準の関数の中だけ）。 */
function inferRaw(c: RenderChecker, e: Expr & { kind: 'mcall' }, node: { kind: string; callee: string; args: Arg[]; vs?: string; fs?: string; vsArgc?: number }): Ty {
  if (e.name === 'rawBeginImage') {
    builtin(c, 'gpu.rawBeginImage', [{ name: 'width', mode: 'none', ty: I32 }, { name: 'height', mode: 'none', ty: I32 }, { name: 'clear', mode: 'none', ty: U32 }],
      I32, e.args, e.pos);
    node.kind = 'call';
    node.callee = '__gpu_image_begin';
    return I32;
  }
  if (e.name === 'rawBeginFrame') {
    builtin(c, 'gpu.rawBeginFrame', [{ name: 'window', mode: 'none', ty: I32 }, { name: 'width', mode: 'none', ty: I32 }, { name: 'height', mode: 'none', ty: I32 },
      { name: 'clear', mode: 'none', ty: U32 }], I32, e.args, e.pos);
    node.kind = 'call';
    node.callee = '__gpu_frame_begin';
    return I32;
  }
  if (e.name === 'rawEndFrame') {
    builtin(c, 'gpu.rawEndFrame', [{ name: 'frame', mode: 'none', ty: I32 }], VOID, e.args, e.pos);
    node.kind = 'call';
    node.callee = '__gpu_frame_end';
    return VOID;
  }
  if (e.name === 'rawEndImage') {
    builtin(c, 'gpu.rawEndImage', [{ name: 'image', mode: 'none', ty: I32 }, { name: 'pixels', mode: 'mut', ty: PIXELS }], VOID, e.args, e.pos);
    node.kind = 'call';
    node.callee = '__gpu_image_end';
    return VOID;
  }
  // gpu.rawDraw(image, vertexCount, instances, vs(...), fs(...)) / gpu.rawDrawIndexed(image, indices, instances, vs(...), fs(...))
  const indexed = e.name === 'rawDrawIndexed';
  builtin(c, `gpu.${e.name}`, [{ name: 'image', mode: 'none', ty: I32 }, indexed ? { name: 'indices', mode: 'none', ty: PIXELS } : { name: 'vertexCount', mode: 'none', ty: U32 },
    { name: 'instances', mode: 'none', ty: U32 }], VOID, e.args.slice(0, 3), e.pos);
  const v = stageArg(c, e.args[3], 'vertex', null);
  const f = stageArg(c, e.args[4], 'fragment');
  matchStages(c, v.sig, f.sig, e.args[4].expr.pos);
  const n = node as typeof node & { indexed?: boolean; instanced?: boolean };
  n.kind = 'call';
  n.callee = '__gpu_draw';
  n.vs = v.sig.name;
  n.fs = f.sig.name;
  n.vsArgc = v.args.length;
  n.indexed = indexed;
  n.instanced = v.instanced;
  n.args = [e.args[0], e.args[1], e.args[2], ...v.args, ...f.args];
  return VOID;
}

/** vs と fs の組の描画の関数の名前。 */
export function drawName(vs: string, fs: string, kind: DrawKind): string {
  return `umRenderDraw${kindTag(kind)}__${vs}__${fs}`;
}

/** 型の書き方を UM のソースの文字列にする（シェーダの引数の型を、描画の関数の引数に写す）。 */
function typeText(t: TypeNode): string {
  switch (t.kind) {
    case 'name':
      return t.name;
    case 'generic':
      return `${t.name}<${t.args.map(typeText).join(', ')}>`;
    case 'array':
      return `${typeText(t.elem)}[${t.lenName ?? t.len}]`;
    case 'heap':
      return `${t.buffer ? 'Buffer' : t.racy ? 'RacyBuffer' : 'Array'}<${typeText(t.elem)}>`;
    case 'opt':
      return `${typeText(t.inner)} | null`;
  }
}

/** 受け渡す値のフィールドが整数（補間しない）か。f32 と vecN<f32> は補間する。 */
function isFlat(t: TypeNode): boolean {
  if (t.kind === 'name') return t.name === 'i32' || t.name === 'u32';
  if (t.kind === 'generic') return t.args.length === 1 && t.args[0].kind === 'name' && (t.args[0].name === 'i32' || t.args[0].name === 'u32');
  return false;
}

/**
 * プログラムの中の gpu.draw(…, vs(…), fs(…)) を探し、vs と fs の組ごとに描画の関数の UM のソースを作る（型検査の前）。
 * 組が見つからないか、vs / fs / 受け渡す struct の宣言が見つからなければ、その組は作らない（型検査がエラーを出す）。
 */
export function drawSources(decls: Decl[]): string {
  // vs と fs の組 → 使われた描画の形。
  const pairs = new Map<string, { vs: string; fs: string; kinds: Map<string, DrawKind> }>();
  const walk = (x: unknown): void => {
    if (!x || typeof x !== 'object') return;
    if (Array.isArray(x)) {
      for (const y of x) walk(y);
      return;
    }
    const n = x as Record<string, unknown> & { kind?: string };
    if (n.kind === 'mcall') {
      const m = n as unknown as Expr & { kind: 'mcall' };
      if (m.recv.kind === 'ident' && m.recv.name === 'gpu' && (m.name === 'draw' || m.name === 'drawIndexed') && (m.args.length === 4 || m.args.length === 5)) {
        const a = m.args[m.args.length - 2].expr;
        const b = m.args[m.args.length - 1].expr;
        if (a.kind === 'call' && b.kind === 'call') {
          const key = `${a.callee}|${b.callee}`;
          const pair = pairs.get(key) ?? { vs: a.callee, fs: b.callee, kinds: new Map() };
          pairs.set(key, pair);
          const kind = { indexed: m.name === 'drawIndexed', instanced: m.args.length === 5 };
          pair.kinds.set(kindTag(kind), kind);
        }
      }
    }
    for (const [k, v] of Object.entries(n)) {
      if (k !== 'pos' && k !== 'ty' && k !== 'sym') walk(v);
    }
  };
  walk(decls);
  const fns = new Map<string, FnDecl>();
  const structs = new Map<string, StructDecl>();
  for (const d of decls) {
    if (d.kind === 'fn') fns.set(d.name, d);
    if (d.kind === 'struct') structs.set(d.name, d);
  }
  let out = '';
  for (const { vs: vsName, fs: fsName, kinds } of pairs.values()) {
    const vs = fns.get(vsName);
    const fs = fns.get(fsName);
    if (!vs || !fs || vs.stage !== 'vertex' || fs.stage !== 'fragment' || vs.typeParams || fs.typeParams) continue;
    if (!vs.retNode || vs.retNode.kind !== 'name' || vs.params.length < 1 || fs.params.length < 1) continue;
    // fragment function が受け取る型が違えば、gpu.draw の検査がエラーにする（同じエラーを描画の関数から重ねて出さない）。
    const got = fs.params[0].tyNode;
    if (got.kind !== 'name' || got.name !== vs.retNode.name) continue;
    const st = structs.get(vs.retNode.name);
    if (!st || st.typeParams) continue;
    out += helperSource(vs, fs, st);
    for (const kind of kinds.values()) {
      if (vs.params.length < (kind.instanced ? 2 : 1)) continue;
      out += drawSource(vs, fs, st, kind);
    }
  }
  return out;
}

/**
 * vs と fs の組の描画の関数（GPU が使えれば gpu.rawDraw、使えなければ CPU のラスタライザ）の UM のソース。
 * CPU のラスタライザは GPU と同じ決まりで描く: クリップ空間で近い面（z ≥ 0）と遠い面（z ≤ w）で切り、
 * 画素の中心で塗る（辺の上は top-left rule）。補間は遠近を考えた重心座標（整数のフィールドは最初の頂点の値）、
 * 深度は小さいものが残る。fragment function の position は（画素の中心の x, y, 深度, 1 / w）。
 */
function helperSource(vs: FnDecl, fs: FnDecl, st: StructDecl): string {
  const tag = `${vs.name}__${fs.name}`;
  const out = st.name;
  const fArgs = fs.params.slice(1).map((_, i) => `f${i}`);
  // 引数の型はシェーダの宣言のまま（シェーダの引数は読むだけなので、渡し方の印はない）。
  const fTypes = fs.params.slice(1).map((p) => typeText(p.tyNode));
  const params = (names: string[], types: string[]) => names.map((n, i) => `, ${n}: ${types[i]}`).join('');
  const pass = (names: string[]) => names.map((n) => `, ${n}`).join('');
  const fields = st.fields.filter((f) => f.name !== 'position');
  const lerp = fields.map((f) => isFlat(f.tyNode) ? `${f.name}: prov.${f.name}` : `${f.name}: a.${f.name} + (b.${f.name} - a.${f.name}) * t`);
  const mix = fields.map((f) => isFlat(f.tyNode) ? `${f.name}: prov.${f.name}` : `${f.name}: a.${f.name} * wa + b.${f.name} * wb + c.${f.name} * wc`);
  return `
function umRenderLerp__${tag}(a: ${out}, b: ${out}, t: f32, prov: ${out}): ${out} {
  return { position: a.position + (b.position - a.position) * t${lerp.map((x) => `, ${x}`).join('')} };
}

function umRenderMix__${tag}(a: ${out}, b: ${out}, c: ${out}, wa: f32, wb: f32, wc: f32, prov: ${out}, pos: vec4<f32>): ${out} {
  return { position: pos${mix.map((x) => `, ${x}`).join('')} };
}

// 多角形を近い面（far = false: z ≥ 0）か遠い面（far = true: z ≤ w）で切る。
function umRenderClip__${tag}(poly: ${out}[], far: bool, prov: ${out}): ${out}[] {
  let res: ${out}[] = [];
  const n = poly.length;
  for (let k: usize = 0; k < n; k++) {
    const cur = poly[k];
    const nxt = poly[(k + 1) % n];
    const dc = far ? cur.position.w - cur.position.z : cur.position.z;
    const dn = far ? nxt.position.w - nxt.position.z : nxt.position.z;
    if (dc >= 0) {
      res.push(cur);
    }
    if ((dc >= 0) != (dn >= 0)) {
      res.push(umRenderLerp__${tag}(cur, nxt, dc / (dc - dn), prov));
    }
  }
  return res;
}

function umRenderTri__${tag}(img: mut GpuImage, a: ${out}, b: ${out}, c: ${out}, prov: ${out}${params(fArgs, fTypes)}): void {
  if (!(a.position.w > 0) || !(b.position.w > 0) || !(c.position.w > 0)) {
    return;
  }
  const fw = img.width as f32;
  const fh = img.height as f32;
  const aw = 1 / a.position.w;
  const bw = 1 / b.position.w;
  const cw = 1 / c.position.w;
  const ax = (a.position.x * aw * 0.5 + 0.5) * fw;
  const ay = (0.5 - a.position.y * aw * 0.5) * fh;
  const az = a.position.z * aw;
  const bx = (b.position.x * bw * 0.5 + 0.5) * fw;
  const by = (0.5 - b.position.y * bw * 0.5) * fh;
  const bz = b.position.z * bw;
  const cx = (c.position.x * cw * 0.5 + 0.5) * fw;
  const cy = (0.5 - c.position.y * cw * 0.5) * fh;
  const cz = c.position.z * cw;
  const area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
  if (!(area > 0 || area < 0)) {
    return;
  }
  if (area < 0) {
    // 内側の辺の関数が正になる向きにする（b と c を入れ替える）。
    umRenderTri__${tag}(mut img, a, c, b, prov${pass(fArgs)});
    return;
  }
  let x0 = Math.floor(Math.min(ax, Math.min(bx, cx)));
  let x1 = Math.ceil(Math.max(ax, Math.max(bx, cx)));
  let y0 = Math.floor(Math.min(ay, Math.min(by, cy)));
  let y1 = Math.ceil(Math.max(ay, Math.max(by, cy)));
  if (x0 < 0) {
    x0 = 0;
  }
  if (y0 < 0) {
    y0 = 0;
  }
  if (x1 > fw - 1) {
    x1 = fw - 1;
  }
  if (y1 > fh - 1) {
    y1 = fh - 1;
  }
  if (!(x1 >= x0 && y1 >= y0)) {
    return;
  }
  const width = img.width as u32;
  for (let py = y0 as i32; py <= (y1 as i32); py++) {
    const sy = (py as f32) + 0.5;
    for (let px = x0 as i32; px <= (x1 as i32); px++) {
      const sx = (px as f32) + 0.5;
      const e0 = (cx - bx) * (sy - by) - (cy - by) * (sx - bx);
      const e1 = (ax - cx) * (sy - cy) - (ay - cy) * (sx - cx);
      const e2 = (bx - ax) * (sy - ay) - (by - ay) * (sx - ax);
      if (renderInside(e0, cx - bx, cy - by) && renderInside(e1, ax - cx, ay - cy) && renderInside(e2, bx - ax, by - ay)) {
        const l0 = e0 / area;
        const l1 = e1 / area;
        const l2 = e2 / area;
        const z = l0 * az + l1 * bz + l2 * cz;
        const idx = (py as u32) * width + (px as u32);
        if (z < img.depth[idx]) {
          const p0 = l0 * aw;
          const p1 = l1 * bw;
          const p2 = l2 * cw;
          const s = p0 + p1 + p2;
          const v = umRenderMix__${tag}(a, b, c, p0 / s, p1 / s, p2 / s, prov, vec4(sx, sy, z, s));
          img.depth[idx] = z;
          img.color[idx] = renderPack(${fs.name}(v${pass(fArgs)}));
        }
      }
    }
  }
}
`;
}

/** 描画の形ごとの描画の関数: GPU の画像なら gpu.rawDraw / rawDrawIndexed、CPU の画像なら CPU で描く。 */
function drawSource(vs: FnDecl, fs: FnDecl, st: StructDecl, kind: DrawKind): string {
  const tag = `${vs.name}__${fs.name}`;
  const out = st.name;
  const skip = kind.instanced ? 2 : 1;
  const vArgs = vs.params.slice(skip).map((_, i) => `v${i}`);
  const fArgs = fs.params.slice(1).map((_, i) => `f${i}`);
  const vTypes = vs.params.slice(skip).map((p) => typeText(p.tyNode));
  const fTypes = fs.params.slice(1).map((p) => typeText(p.tyNode));
  const params = (names: string[], types: string[]) => names.map((n, i) => `, ${n}: ${types[i]}`).join('');
  const pass = (names: string[]) => names.map((n) => `, ${n}`).join('');
  const index = (k: string) => (kind.indexed ? `indices[${k}]` : k);
  const vsCall = (k: string) => `${vs.name}(${index(k)}${kind.instanced ? ', inst' : ''}${pass(vArgs)})`;
  const first = kind.indexed ? 'indices: Buffer<u32>' : 'count: u32';
  const raw = kind.indexed ? 'rawDrawIndexed' : 'rawDraw';
  return `
function ${drawName(vs.name, fs.name, kind)}(img: mut GpuImage, ${first}${kind.instanced ? ', instances: u32' : ''}${params(vArgs, vTypes)}${params(fArgs, fTypes)}): void {
  if (img.handle != 0) {
    gpu.${raw}(img.handle, ${kind.indexed ? 'indices' : 'count'}, ${kind.instanced ? 'instances' : '1'}, ${vs.name}(${vArgs.join(', ')}), ${fs.name}(${fArgs.join(', ')}));
    return;
  }
  const n: u32 = ${kind.indexed ? 'indices.length as u32' : 'count'};
  for (let inst: u32 = 0; inst < ${kind.instanced ? 'instances' : '1'}; inst++) {
    for (let i: u32 = 0; i + 3 <= n; i += 3) {
      const a = ${vsCall('i')};
      const b = ${vsCall('i + 1')};
      const c = ${vsCall('i + 2')};
      let poly: ${out}[] = [a, b, c];
      poly = umRenderClip__${tag}(poly, false, a);
      poly = umRenderClip__${tag}(poly, true, a);
      for (let k: usize = 1; k + 1 < poly.length; k++) {
        umRenderTri__${tag}(mut img, poly[0], poly[k], poly[k + 1], a${pass(fArgs)});
      }
    }
  }
}
`;
}
