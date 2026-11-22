import { isCollectiveLet, tyStr } from './ast.ts';
import type { Arg, Expr, FnDecl, Stmt, Sym, Ty } from './ast.ts';
import { isCopyTy, isPlace, passByPtr, simdInfo } from './checker.ts';
import type { FnSig, Program } from './checker.ts';
import { emitWgsl, gpuLayout, has64, WGSL_SIMDGROUP } from './wgsl.ts';
import { CompileError } from './diag.ts';
import type { Pos } from './diag.ts';
import { emitMsl } from './msl.ts';

export type TargetKind = 'wasm32' | 'native';

// um_panic に渡す原因コード。runtime/ 側の表と揃えること。
const PANIC_INDEX = 1;
const PANIC_DIV_ZERO = 2;
const PANIC_SHIFT = 3;
const PANIC_POP_EMPTY = 4;
const PANIC_OOM = 5;
const PANIC_DISPATCH = 6;
const PANIC_NO_GPU = 7;
// 8 = スレッドを起こせなかった（ランタイムが um_thread_start の中で停止する）。
const PANIC_THREAD_IO = 9;
const PANIC_GROUP = 10;
/** プラグインの export が、ホストから借りた Buffer を差し替えた（wasm32-plugin だけ）。 */
const PANIC_PLUGIN_BUFFER = 15;
// 11 = ウィンドウの関数をメインスレッド以外から呼んだ、12 = ウィンドウの関数の引数が正しくない（runtime/window.c が停止する）。
// 13 = wasm の番号（モジュール・インスタンス）か配列の範囲が正しくない、14 = 使っている最中の wasm のインスタンスを解放した
// （runtime/wasm.c が停止する）。
// 20 = 動画・音声の番号（読み手・書き手）が正しくない、21 = 使っている最中の読み手・書き手を閉じた（runtime/media.c が停止する）。
/** グループの大きさの上限（Metal の1グループのスレッド数の上限）。 */
const MAX_GROUP_SIZE = 1024;

// TBAA のタグ（ModuleGen.generate の末尾で定義）。
// HDR: 可変長配列・Buffer の管理情報（%um.Array のフィールド）の読み書き。
// VAL: 数値・bool として読み書きする値。集成体をまるごと読み書きするときはタグを付けない。
const HDR = '!3';
const VAL = '!4';
const isScalar = (t: Ty) => t.k === 'int' || t.k === 'float' || t.k === 'bool';

/**
 * 添字の式が「kernel の添字 i に定数を足し引きしたもの」なら、その定数を返す。
 * そうでなければ null。この形の添字は、dispatch の側で範囲内になる区間を前もって求められる。
 */
function affineOffset(e: Expr, idx: number): number | null {
  const isIdx = (x: Expr) => x.kind === 'ident' && x.sym!.id === idx;
  const lit = (x: Expr) => (x.kind === 'int' && x.value >= 0n && x.value <= 1_000_000n ? Number(x.value) : null);
  if (isIdx(e)) return 0;
  if (e.kind !== 'binary') return null;
  if (e.op === '+') {
    if (isIdx(e.left) && lit(e.right) !== null) return lit(e.right);
    if (isIdx(e.right) && lit(e.left) !== null) return lit(e.left);
  }
  if (e.op === '-' && isIdx(e.left) && lit(e.right) !== null) return -lit(e.right)!;
  return null;
}

/** 境界検査を省ける添字アクセスの計画。配列の変数ごとに、添字 i + c の c の範囲と、その配列を指す式を持つ。 */
interface SplitPlan {
  loopVar: Sym;
  arrays: Map<number, { min: number; max: number; base: Expr }>;
}

/**
 * for ループの本体で、境界検査をループの外へ出せる添字アクセスを探す。
 *
 * 条件:
 * - ループ変数 i は初期化部で宣言した 32 / 64 ビットの整数で、本体と条件式の中で書き換えられない。
 * - 配列 a は変数で、ループの中（条件・本体・更新）では「a[...]」と「a.length」の形でしか現れない。
 *   つまり長さが変わらず、移動も借用もされない。
 * - 添字は i、i + 定数、i - 定数 のどれか。
 *
 * 知らない形の式や文があれば、何もしない（null を返す）。
 */
function planSplit(s: Stmt & { kind: 'for' }): SplitPlan | null {
  if (!s.init || s.init.kind !== 'let' || !s.init.sym) return null;
  const loopVar = s.init.sym;
  const lt = loopVar.ty;
  if (lt.k !== 'int' || lt.bits < 32) return null;

  const escaped = new Set<number>(); // 許した形以外で現れた変数
  const found = new Map<number, { min: number; max: number; base: Expr }>();
  let ok = true;
  let collect = false;

  const ex = (e: Expr | undefined): void => {
    if (!e || !ok) return;
    switch (e.kind) {
      case 'int':
      case 'float':
      case 'bool':
      case 'str':
        return;
      case 'ident':
        if (e.sym!.id !== loopVar.id) escaped.add(e.sym!.id);
        return;
      case 'unary':
      case 'cast':
        return ex(e.expr);
      case 'binary':
        ex(e.left);
        return ex(e.right);
      case 'cond':
        ex(e.cond);
        ex(e.then);
        return ex(e.else);
      case 'call':
      case 'new':
      case 'newArray':
        return args(e.args);
      case 'dispatch':
        ex(e.count);
        if (e.group) ex(e.group);
        return args(e.args);
      case 'mcall':
        ex(e.recv);
        return args(e.args);
      case 'math':
        return e.args.forEach(ex);
      case 'field':
        if (e.base.kind === 'ident' && e.base.ty!.k === 'heap' && e.name === 'length') return;
        return ex(e.base);
      case 'index':
        if (e.base.kind === 'ident' && e.base.ty!.k === 'heap') {
          const c = affineOffset(e.index, loopVar.id);
          if (collect && c !== null) {
            const id = e.base.sym!.id;
            const o = found.get(id) ?? { min: 0, max: 0, base: e.base };
            found.set(id, { min: Math.min(o.min, c), max: Math.max(o.max, c), base: o.base });
          }
        } else {
          ex(e.base);
        }
        return ex(e.index);
      case 'object':
        return e.fields.forEach((f) => ex(f.value));
      case 'arrayLit':
        return e.elems.forEach(ex);
      default:
        ok = false;
    }
  };
  const args = (as: Arg[]): void => {
    for (const a of as) {
      // ループ変数を mut / own で渡すと、値が変わりうる。
      if (a.mode !== 'none' && a.expr.kind === 'ident' && a.expr.sym!.id === loopVar.id) ok = false;
      ex(a.expr);
    }
  };
  const st = (x: Stmt | undefined, mayAssignLoopVar: boolean): void => {
    if (!x || !ok) return;
    switch (x.kind) {
      case 'block':
        return x.body.forEach((b) => st(b, mayAssignLoopVar));
      case 'let':
        // ループの中で宣言した配列は、ループの前には存在しないので対象にしない。
        if (x.sym) escaped.add(x.sym.id);
        return ex(x.init);
      case 'assign':
        if (x.target.kind === 'ident' && x.target.sym!.id === loopVar.id && !mayAssignLoopVar) ok = false;
        ex(x.target);
        return ex(x.value);
      case 'if':
        ex(x.cond);
        st(x.then, mayAssignLoopVar);
        return st(x.else, mayAssignLoopVar);
      case 'while':
        ex(x.cond);
        return st(x.body, mayAssignLoopVar);
      case 'for':
        st(x.init, mayAssignLoopVar);
        ex(x.cond);
        st(x.step, mayAssignLoopVar);
        return st(x.body, mayAssignLoopVar);
      case 'forOf':
        if (x.sym) escaped.add(x.sym.id);
        ex(x.iter);
        return st(x.body, mayAssignLoopVar);
      case 'return':
        return ex(x.value);
      case 'expr':
        return ex(x.expr);
      case 'break':
      case 'continue':
        return;
      default:
        ok = false;
    }
  };

  ex(s.cond);
  st(s.step, true);
  collect = true;
  st(s.body, false);
  if (!ok) return null;
  for (const id of escaped) found.delete(id);
  return found.size > 0 ? { loopVar, arrays: found } : null;
}

/**
 * kernel の添字 i に対して、式の値が取りうる範囲 [i + min, i + max] を返す。分からなければ null。
 * 式は i、i ± 定数、または「i ± 定数 しか代入されない変数」（とそれに定数を足し引きしたもの）。
 */
function indexRange(e: Expr, idx: number, locals: Map<number, { min: number; max: number }>): { min: number; max: number } | null {
  const c = affineOffset(e, idx);
  if (c !== null) return { min: c, max: c };
  const lit = (x: Expr) => (x.kind === 'int' && x.value >= 0n && x.value <= 1_000_000n ? Number(x.value) : null);
  const local = (x: Expr) => (x.kind === 'ident' ? locals.get(x.sym!.id) ?? null : null);
  if (local(e)) return local(e);
  if (e.kind !== 'binary') return null;
  const shift = (r: { min: number; max: number } | null, by: number | null) => (r && by !== null ? { min: r.min + by, max: r.max + by } : null);
  if (e.op === '+') return shift(local(e.left), lit(e.right)) ?? shift(local(e.right), lit(e.left));
  if (e.op === '-') {
    const by = lit(e.right);
    return shift(local(e.left), by === null ? null : -by);
  }
  return null;
}

/**
 * kernel の本体で、「i ± 定数」しか代入されない変数を集め、それぞれの取りうる範囲を返す。
 * 初期値のない宣言や、それ以外の値の代入が1つでもある変数は含めない。
 */
function affineLocals(body: Stmt[], idx: number): Map<number, { min: number; max: number }> {
  const found = new Map<number, { min: number; max: number }>();
  const bad = new Set<number>();
  const assign = (id: number, value: Expr | undefined): void => {
    const c = value ? affineOffset(value, idx) : null;
    if (c === null) {
      bad.add(id);
      return;
    }
    const r = found.get(id);
    found.set(id, r ? { min: Math.min(r.min, c), max: Math.max(r.max, c) } : { min: c, max: c });
  };
  const st = (s: Stmt | undefined): void => {
    if (!s) return;
    switch (s.kind) {
      case 'block':
        return s.body.forEach(st);
      case 'let':
        if (s.sym) assign(s.sym.id, s.init);
        return;
      case 'assign':
        if (s.target.kind === 'ident') assign(s.target.sym!.id, s.value);
        return;
      case 'if':
        st(s.then);
        return st(s.else);
      case 'while':
        return st(s.body);
      case 'for':
        st(s.init);
        st(s.step);
        return st(s.body);
      case 'forOf':
        if (s.sym) bad.add(s.sym.id);
        return st(s.body);
    }
  };
  body.forEach(st);
  for (const id of bad) found.delete(id);
  return found;
}

/** kernel の本体に現れる式をすべて訪ねる。 */
function walkExprs(body: Stmt[], visit: (e: Expr) => void): void {
  const ex = (e: Expr | undefined): void => {
    if (!e) return;
    visit(e);
    switch (e.kind) {
      case 'unary':
      case 'cast':
        return ex(e.expr);
      case 'binary':
        ex(e.left);
        return ex(e.right);
      case 'cond':
        ex(e.cond);
        ex(e.then);
        return ex(e.else);
      case 'call':
        return e.args.forEach((a) => ex(a.expr));
      case 'mcall':
        ex(e.recv);
        return e.args.forEach((a) => ex(a.expr));
      case 'math':
        return e.args.forEach(ex);
      case 'field':
        return ex(e.base);
      case 'index':
        ex(e.base);
        return ex(e.index);
      case 'object':
        return e.fields.forEach((f) => ex(f.value));
      case 'arrayLit':
        return e.elems.forEach(ex);
    }
  };
  const st = (s: Stmt | undefined): void => {
    if (!s) return;
    switch (s.kind) {
      case 'block':
        return s.body.forEach(st);
      case 'let':
        return ex(s.init);
      case 'assign':
        ex(s.target);
        return ex(s.value);
      case 'if':
        ex(s.cond);
        st(s.then);
        return st(s.else);
      case 'while':
        ex(s.cond);
        return st(s.body);
      case 'for':
        st(s.init);
        ex(s.cond);
        st(s.step);
        return st(s.body);
      case 'forOf':
        ex(s.iter);
        return st(s.body);
      case 'return':
        return ex(s.value);
      case 'expr':
        return ex(s.expr);
    }
  };
  body.forEach(st);
}

/**
 * kernel（と、そこから呼ぶ補助の kernel）が 64 ビットの整数を使うか。WGSL には 64 ビットの整数がないので、
 * そうした kernel は WebGPU では実行せず CPU で実行する（Metal と CPU では使える）。
 */
export function kernelUses64(prog: Program, d: FnDecl, seen = new Set<string>()): boolean {
  if (seen.has(d.name)) return false;
  seen.add(d.name);
  if (d.params.some((p) => has64(prog, p.ty!)) || (d.ret && has64(prog, d.ret))) return true;
  let found = false;
  const visitStmt = (s: Stmt | undefined): void => {
    if (!s || found) return;
    if ((s.kind === 'let' || s.kind === 'forOf') && s.sym && has64(prog, s.sym.ty)) found = true;
    if (s.kind === 'block') s.body.forEach(visitStmt);
    else if (s.kind === 'if') {
      visitStmt(s.then);
      visitStmt(s.else);
    } else if (s.kind === 'while' || s.kind === 'forOf') visitStmt(s.body);
    else if (s.kind === 'for') {
      visitStmt(s.init);
      visitStmt(s.body);
    }
  };
  d.body.forEach(visitStmt);
  if (found) return true;
  walkExprs(d.body, (e) => {
    if (found) return;
    if (e.ty && has64(prog, e.ty)) found = true;
    else if (e.kind === 'call') {
      const sig = prog.fns.get(e.callee);
      if (sig && kernelUses64(prog, sig.decl, seen)) found = true;
    }
  });
  return found;
}

/** 文が group.barrier() か simdgroup の集計（`const s = group.simdSum(x)` など）を含むか。どちらも CPU では区切りになる。 */
function hasBarrier(s: Stmt | undefined): boolean {
  if (!s) return false;
  switch (s.kind) {
    case 'let':
      return isCollectiveLet(s);
    case 'expr':
      return s.expr.kind === 'math' && s.expr.fn === 'group.barrier';
    case 'block':
      return s.body.some(hasBarrier);
    case 'if':
      return hasBarrier(s.then) || hasBarrier(s.else);
    case 'while':
      return hasBarrier(s.body);
    case 'for':
      return hasBarrier(s.init) || hasBarrier(s.step) || hasBarrier(s.body);
    case 'forOf':
      return hasBarrier(s.body);
    default:
      return false;
  }
}

/** 本体を文の並びとして返す（{ } ならその中身）。 */
function stmtList(s: Stmt): Stmt[] {
  return s.kind === 'block' ? s.body : [s];
}

/**
 * barrier を使う kernel で、barrier をまたいで生きうる変数（スレッドごとに置き場所が要る）を集める。
 * 本体の直下と、barrier を含む文の直下で宣言された変数。それ以外（barrier の間の文の内側）は、
 * 1つのスレッドを続けて実行する間だけ生きるので、普通の変数でよい。
 */
function groupLevelSyms(list: Stmt[], out: Sym[]): void {
  for (const s of list) {
    if (s.kind === 'let' && !s.sym!.shared) out.push(s.sym!);
    if (!hasBarrier(s)) continue;
    switch (s.kind) {
      case 'block':
        groupLevelSyms(s.body, out);
        break;
      case 'if':
        groupLevelSyms(stmtList(s.then), out);
        if (s.else) groupLevelSyms(stmtList(s.else), out);
        break;
      case 'while':
        groupLevelSyms(stmtList(s.body), out);
        break;
      case 'for':
        if (s.init) groupLevelSyms([s.init], out);
        groupLevelSyms(stmtList(s.body), out);
        break;
      default:
        break;
    }
  }
}

/** kernel の共有配列（本体の直下の `let t: T[N] = group.shared();`）の並びと、合計のバイト数。 */
export function sharedLayout(d: FnDecl): { offsets: Map<number, number>; bytes: number } {
  const offsets = new Map<number, number>();
  let bytes = 0;
  for (const s of d.body) {
    if (s.kind !== 'let' || !s.sym?.shared) continue;
    offsets.set(s.sym.id, bytes);
    bytes += (s.sym.ty as Ty & { k: 'array' }).len * 4;
  }
  return { offsets, bytes };
}

// ============================================================================
// 値域の解析（範囲チェックの除去）
//
// 関数ごとに、添字の式とシフト量の式が取りうる整数の範囲を控えめに見積もり、「必ず範囲内」と
// 分かる所の範囲チェックを省く。健全さを最優先にする（確実なときだけ省く。間違えると範囲外アクセス
// になる）。固定長配列 `T[N]` の添字（長さがコンパイル時に分かる）と、`<<` / `>>` のシフト量
// （上限がビット幅）を対象にする。ヒープ配列は長さが実行時なので対象にしない。
//
// 範囲の出どころ: 型の範囲、リテラル、`x & 定数`・`x % 定数`・`Math.ctz/clz/popcount`・
// `Math.min/max`・足し引き（折り返さないときだけ）、そして `if (i < 定数)` などの比較で分かる絞り込み。
// ============================================================================

/** 整数の範囲 [min, max]（両端を含む）。null は「分からない」。 */
interface IRange {
  min: bigint | null;
  max: bigint | null;
}

const UNBOUNDED: IRange = { min: null, max: null };

function typeRange(t: Ty): IRange {
  if (t.k !== 'int') return UNBOUNDED;
  const b = BigInt(t.bits);
  return t.signed ? { min: -(1n << (b - 1n)), max: (1n << (b - 1n)) - 1n } : { min: 0n, max: (1n << b) - 1n };
}

/** 2つの範囲の共通部分（どちらの条件も満たす値の範囲）。 */
function meet(a: IRange, b: IRange): IRange {
  const lo = a.min === null ? b.min : b.min === null ? a.min : a.min > b.min ? a.min : b.min;
  const hi = a.max === null ? b.max : b.max === null ? a.max : a.max < b.max ? a.max : b.max;
  return { min: lo, max: hi };
}

/** 2つの範囲の和（どちらかの経路で取りうる値の範囲）。分からない端は null のまま。 */
function join(a: IRange, b: IRange): IRange {
  const lo = a.min === null || b.min === null ? null : a.min < b.min ? a.min : b.min;
  const hi = a.max === null || b.max === null ? null : a.max > b.max ? a.max : b.max;
  return { min: lo, max: hi };
}

/** 式が定数の整数なら、その値。そうでなければ null。最上位の定数を置き換えた値（fixed）も含む。 */
function constInt(e: Expr): bigint | null {
  if (e.kind === 'int') return e.value;
  if (e.kind === 'unary' && e.op === '-') {
    const x = constInt(e.expr);
    return x === null ? null : -x;
  }
  if (e.kind === 'cast') return constInt(e.expr); // 定数を型変換しても、範囲に収まっていれば値は同じ
  return null;
}

/** 変数ごとの、いま分かっている値の範囲。 */
type Facts = Map<number, IRange>;

/** 式の中で mut / own として渡される変数（評価の途中で値が変わりうる）。安全判断ではその範囲を信用しない。 */
function collectMutArgs(e: Expr | undefined, out: Set<number>): void {
  if (!e) return;
  const args = (as: Arg[]) => {
    for (const a of as) {
      if ((a.mode === 'mut' || a.mode === 'own') && a.expr.kind === 'ident' && a.expr.sym) out.add(a.expr.sym.id);
      collectMutArgs(a.expr, out);
    }
  };
  switch (e.kind) {
    case 'unary':
    case 'cast':
      return collectMutArgs(e.expr, out);
    case 'binary':
      collectMutArgs(e.left, out);
      return collectMutArgs(e.right, out);
    case 'cond':
      collectMutArgs(e.cond, out);
      collectMutArgs(e.then, out);
      return collectMutArgs(e.else, out);
    case 'call':
    case 'new':
    case 'newArray':
      return args(e.args);
    case 'mcall':
      collectMutArgs(e.recv, out);
      return args(e.args);
    case 'dispatch':
      collectMutArgs(e.count, out);
      if (e.group) collectMutArgs(e.group, out);
      return args(e.args);
    case 'thread':
      return args(e.args);
    case 'math':
      return e.args.forEach((x) => collectMutArgs(x, out));
    case 'field':
      return collectMutArgs(e.base, out);
    case 'index':
      collectMutArgs(e.base, out);
      return collectMutArgs(e.index, out);
    case 'object':
      return e.fields.forEach((f) => collectMutArgs(f.value, out));
    case 'arrayLit':
      return e.elems.forEach((x) => collectMutArgs(x, out));
  }
}

/** 文の中で値が変わる変数（代入先、++ / --、mut / own で渡す変数）をすべて集める。ループで信用を外すのに使う。 */
function collectMutated(s: Stmt | undefined, out: Set<number>): void {
  if (!s) return;
  switch (s.kind) {
    case 'let':
      collectMutArgs(s.init, out);
      return;
    case 'assign':
      if (s.target.kind === 'ident' && s.target.sym) out.add(s.target.sym.id);
      collectMutArgs(s.target, out);
      collectMutArgs(s.value, out);
      return;
    case 'if':
      collectMutArgs(s.cond, out);
      collectMutated(s.then, out);
      return collectMutated(s.else, out);
    case 'while':
      collectMutArgs(s.cond, out);
      return collectMutated(s.body, out);
    case 'for':
      collectMutated(s.init, out);
      collectMutArgs(s.cond, out);
      collectMutated(s.step, out);
      return collectMutated(s.body, out);
    case 'forOf':
      collectMutArgs(s.iter, out);
      return collectMutated(s.body, out);
    case 'return':
    case 'expr':
      return collectMutArgs(s.kind === 'return' ? s.value : s.expr, out);
    case 'block':
      return s.body.forEach((b) => collectMutated(b, out));
    case 'match':
      collectMutArgs(s.scrut, out);
      return s.arms.forEach((a) => collectMutated(a.body, out));
  }
}

/** 文が「素通りしない」（必ず return / break / continue で抜ける）かどうかを控えめに判断する。 */
function diverges(s: Stmt | undefined): boolean {
  if (!s) return false;
  if (s.kind === 'return' || s.kind === 'break' || s.kind === 'continue') return true;
  if (s.kind === 'block') return s.body.length > 0 && diverges(s.body[s.body.length - 1]);
  if (s.kind === 'if') return !!s.else && diverges(s.then) && diverges(s.else);
  return false;
}

/**
 * 関数の本体を順にたどり、「必ず範囲内」と分かる添字アクセスとシフトを集める。
 * safeIndex / safeShift に入った式は、コード生成が範囲チェックを省く。
 */
function analyzeBounds(d: FnDecl): { safeIndex: Set<Expr>; safeShift: Set<Expr> } {
  const safeIndex = new Set<Expr>();
  const safeShift = new Set<Expr>();

  const factOf = (facts: Facts, e: Expr & { kind: 'ident' }, blocked: Set<number>): IRange => {
    const tr = typeRange(e.ty!);
    if (!e.sym || blocked.has(e.sym.id)) return tr;
    const f = facts.get(e.sym.id);
    return f ? meet(tr, f) : tr;
  };

  // 式の範囲を求めつつ、範囲内と分かる添字・シフトを safeIndex / safeShift に入れる。
  const visit = (e: Expr | undefined, facts: Facts, blocked: Set<number>): IRange => {
    if (!e) return UNBOUNDED;
    switch (e.kind) {
      case 'int':
        return { min: e.value, max: e.value };
      case 'bool':
      case 'float':
      case 'str':
      case 'null':
        return UNBOUNDED;
      case 'ident':
        return factOf(facts, e, blocked);
      case 'cast': {
        const r = visit(e.expr, facts, blocked);
        const to = typeRange(e.ty!);
        // 値が変換先の型に収まるなら、整数どうしの変換は値を変えない。
        if (e.ty!.k === 'int' && e.expr.ty!.k === 'int' && r.min !== null && r.max !== null && to.min !== null && to.max !== null && r.min >= to.min && r.max <= to.max) {
          return r;
        }
        return to;
      }
      case 'unary':
        visit(e.expr, facts, blocked);
        return typeRange(e.ty!);
      case 'binary': {
        const lr = visit(e.left, facts, blocked);
        const rr = visit(e.right, facts, blocked);
        if (e.op === '<<' || e.op === '>>') {
          const vt = e.ty!;
          if (vt.k === 'int' && rr.min !== null && rr.min >= 0n && rr.max !== null && rr.max < BigInt(vt.bits)) {
            safeShift.add(e);
          }
        }
        return binRange(e, lr, rr);
      }
      case 'cond': {
        visit(e.cond, facts, blocked);
        const tf = narrow(e.cond, facts, true, blocked);
        const ef = narrow(e.cond, facts, false, blocked);
        const tr = visit(e.then, tf, blocked);
        const er = visit(e.else, ef, blocked);
        return join(tr, er);
      }
      case 'call':
      case 'new':
      case 'newArray':
        e.args.forEach((a) => visit(a.expr, facts, blocked));
        return typeRange(e.ty!);
      case 'mcall':
        visit(e.recv, facts, blocked);
        e.args.forEach((a) => visit(a.expr, facts, blocked));
        return typeRange(e.ty!);
      case 'dispatch':
        visit(e.count, facts, blocked);
        if (e.group) visit(e.group, facts, blocked);
        e.args.forEach((a) => visit(a.expr, facts, blocked));
        return UNBOUNDED;
      case 'thread':
        e.args.forEach((a) => visit(a.expr, facts, blocked));
        return UNBOUNDED;
      case 'math': {
        e.args.forEach((x) => visit(x, facts, blocked));
        const fn = e.fn;
        if (fn === 'ctz' || fn === 'clz' || fn === 'popcount') {
          const at = e.args[0].ty!;
          if (at.k === 'int') return { min: 0n, max: BigInt(at.bits) };
        }
        if ((fn === 'min' || fn === 'max') && e.ty!.k === 'int') {
          const a = visit(e.args[0], facts, blocked);
          const b = visit(e.args[1], facts, blocked);
          if (fn === 'min') {
            const hi = a.max === null ? b.max : b.max === null ? a.max : a.max < b.max ? a.max : b.max;
            const lo = a.min === null || b.min === null ? null : a.min < b.min ? a.min : b.min;
            return meet({ min: lo, max: hi }, typeRange(e.ty!));
          }
          const lo = a.min === null ? b.min : b.min === null ? a.min : a.min > b.min ? a.min : b.min;
          const hi = a.max === null || b.max === null ? null : a.max > b.max ? a.max : b.max;
          return meet({ min: lo, max: hi }, typeRange(e.ty!));
        }
        return typeRange(e.ty!);
      }
      case 'field':
        visit(e.base, facts, blocked);
        return typeRange(e.ty!);
      case 'index': {
        visit(e.base, facts, blocked);
        const ir = visit(e.index, facts, blocked);
        const bt = e.base.ty!;
        // 固定長配列だけ: 長さがコンパイル時に分かる。添字が [0, len) に必ず収まるなら省ける。
        if (bt.k === 'array' && ir.min !== null && ir.min >= 0n && ir.max !== null && ir.max < BigInt(bt.len)) {
          safeIndex.add(e);
        }
        return typeRange(e.ty!);
      }
      case 'object':
        e.fields.forEach((f) => visit(f.value, facts, blocked));
        return UNBOUNDED;
      case 'arrayLit':
        e.elems.forEach((x) => visit(x, facts, blocked));
        return UNBOUNDED;
    }
    return UNBOUNDED;
  };

  // 二項演算の範囲（折り返しが起きないときだけ正確に、そうでなければ型の範囲）。
  const binRange = (e: Expr & { kind: 'binary' }, lr: IRange, rr: IRange): IRange => {
    const tr = typeRange(e.ty!);
    const both = (a: IRange, b: IRange) => a.min !== null && a.max !== null && b.min !== null && b.max !== null;
    const inType = (lo: bigint, hi: bigint) => (tr.min !== null && tr.max !== null && lo >= tr.min && hi <= tr.max ? { min: lo, max: hi } : tr);
    switch (e.op) {
      case '&': {
        const cr = constInt(e.right);
        const cl = constInt(e.left);
        // x & C（C >= 0）の結果は [0, C] に収まる。
        if (cr !== null && cr >= 0n) return meet({ min: 0n, max: cr }, tr);
        if (cl !== null && cl >= 0n) return meet({ min: 0n, max: cl }, tr);
        if (lr.min !== null && lr.min >= 0n && rr.min !== null && rr.min >= 0n && lr.max !== null && rr.max !== null) {
          return { min: 0n, max: lr.max < rr.max ? lr.max : rr.max };
        }
        return tr;
      }
      case '%': {
        const c = constInt(e.right);
        if (c !== null && c !== 0n) {
          const m = (c < 0n ? -c : c) - 1n;
          // 符号なし、または左辺が 0 以上なら結果は [0, |C|-1]。符号つきで負もありうるなら [-(|C|-1), |C|-1]。
          if (!e.ty!.k || e.ty!.k !== 'int' || !e.ty!.signed || (lr.min !== null && lr.min >= 0n)) return meet({ min: 0n, max: m }, tr);
          return meet({ min: -m, max: m }, tr);
        }
        return tr;
      }
      case '+':
        if (both(lr, rr)) return inType(lr.min! + rr.min!, lr.max! + rr.max!);
        return tr;
      case '-':
        if (both(lr, rr)) return inType(lr.min! - rr.max!, lr.max! - rr.min!);
        return tr;
      case '*':
        if (both(lr, rr) && lr.min! >= 0n && rr.min! >= 0n) return inType(lr.min! * rr.min!, lr.max! * rr.max!);
        return tr;
      default:
        return tr;
    }
  };

  // 比較の条件から、真/偽それぞれの枝で分かる絞り込みを作る（元の facts は変えない）。
  const narrow = (cond: Expr, facts: Facts, truth: boolean, blocked: Set<number>): Facts => {
    const f = new Map(facts);
    const tighten = (sym: Sym, r: IRange) => {
      if (blocked.has(sym.id)) return;
      const cur = f.get(sym.id) ?? typeRange(sym.ty);
      f.set(sym.id, meet(cur, r));
    };
    const apply = (c: Expr, t: boolean): void => {
      if (c.kind === 'unary' && c.op === '!') return apply(c.expr, !t);
      if (c.kind !== 'binary') return;
      if (c.op === '&&') {
        if (t) { apply(c.left, true); apply(c.right, true); }
        return;
      }
      if (c.op === '||') {
        if (!t) { apply(c.left, false); apply(c.right, false); }
        return;
      }
      // 比較。片方が変数（型変換なしの ident）、もう片方が定数のときだけ扱う。
      let op = c.op;
      let varE = c.left;
      let constE = c.right;
      if (c.left.kind !== 'ident' && c.right.kind === 'ident') {
        // 定数 OP 変数 → 変数 OP' 定数 に向きを反転。
        varE = c.right;
        constE = c.left;
        op = { '<': '>', '<=': '>=', '>': '<', '>=': '<=', '==': '==', '!=': '!=' }[op] ?? op;
      }
      if (varE.kind !== 'ident' || !varE.sym) return;
      const k = constInt(constE);
      if (k === null) return;
      // 偽の枝では比較を否定する。
      const eff = t ? op : { '<': '>=', '<=': '>', '>': '<=', '>=': '<', '==': '!=', '!=': '==' }[op] ?? op;
      const sym = varE.sym;
      if (eff === '<') tighten(sym, { min: null, max: k - 1n });
      else if (eff === '<=') tighten(sym, { min: null, max: k });
      else if (eff === '>') tighten(sym, { min: k + 1n, max: null });
      else if (eff === '>=') tighten(sym, { min: k, max: null });
      else if (eff === '==') tighten(sym, { min: k, max: k });
    };
    apply(cond, truth);
    return f;
  };

  const cleared = (facts: Facts, drop: Set<number>): Facts => {
    const f = new Map(facts);
    for (const id of drop) f.delete(id);
    return f;
  };

  // 文を順にたどる。素通りするときは更新後の facts を、必ず抜けるときは null を返す。
  const walkStmt = (s: Stmt, facts: Facts): Facts | null => {
    switch (s.kind) {
      case 'let': {
        const blocked = new Set<number>();
        collectMutArgs(s.init, blocked);
        const r = s.init ? visit(s.init, facts, blocked) : UNBOUNDED;
        const f = cleared(facts, blocked);
        if (s.sym && s.sym.ty.k === 'int') f.set(s.sym.id, meet(r, typeRange(s.sym.ty)));
        return f;
      }
      case 'assign': {
        const blocked = new Set<number>();
        collectMutArgs(s.value, blocked);
        collectMutArgs(s.target, blocked);
        const r = visit(s.value, facts, blocked);
        visit(s.target, facts, blocked);
        const f = cleared(facts, blocked);
        if (s.target.kind === 'ident' && s.target.sym && s.target.sym.ty.k === 'int') {
          f.set(s.target.sym.id, meet(r, typeRange(s.target.sym.ty)));
        }
        return f;
      }
      case 'if': {
        const blocked = new Set<number>();
        collectMutArgs(s.cond, blocked);
        visit(s.cond, facts, blocked);
        const base = cleared(facts, blocked);
        const tf = narrow(s.cond, base, true, blocked);
        const ef = narrow(s.cond, base, false, blocked);
        const thenF = walkStmt(s.then, tf);
        const elseF = s.else ? walkStmt(s.else, ef) : ef;
        if (thenF === null && elseF === null) return null;
        if (thenF === null) return elseF;
        if (elseF === null) return thenF;
        return mergeFacts(thenF, elseF);
      }
      case 'while': {
        // 条件自身が書き換える変数だけ信用を外す（ループ変数は条件が毎回確かめるので、絞り込みは体の入口で有効）。
        const condMut = new Set<number>();
        collectMutArgs(s.cond, condMut);
        visit(s.cond, facts, condMut);
        const bodyMut = new Set<number>(condMut);
        collectMutated(s.body, bodyMut);
        const base = cleared(facts, bodyMut);
        walkStmt(s.body, narrow(s.cond, base, true, condMut));
        return base;
      }
      case 'for': {
        let f = s.init ? walkStmt(s.init, facts) : facts;
        if (f === null) f = facts;
        const condMut = new Set<number>();
        collectMutArgs(s.cond, condMut);
        const bodyMut = new Set<number>(condMut);
        collectMutated(s.step, bodyMut);
        collectMutated(s.body, bodyMut);
        const base = cleared(f, bodyMut);
        if (s.cond) visit(s.cond, base, condMut);
        const entry = s.cond ? narrow(s.cond, base, true, condMut) : base;
        walkStmt(s.body, entry);
        if (s.step) walkStmt(s.step, entry);
        return base;
      }
      case 'forOf': {
        visit(s.iter, facts, new Set());
        const mut = new Set<number>();
        collectMutated(s.body, mut);
        const base = cleared(facts, mut);
        walkStmt(s.body, base);
        return base;
      }
      case 'return':
        visit(s.value, facts, new Set());
        return null;
      case 'break':
      case 'continue':
        return null;
      case 'expr': {
        const blocked = new Set<number>();
        collectMutArgs(s.expr, blocked);
        visit(s.expr, facts, blocked);
        return cleared(facts, blocked);
      }
      case 'block': {
        let f: Facts | null = facts;
        for (const b of s.body) {
          f = walkStmt(b, f!);
          if (f === null) return null;
        }
        return f;
      }
      case 'match': {
        visit(s.scrut, facts, new Set());
        let out: Facts | null = null;
        for (const arm of s.arms) {
          const r = walkStmt(arm.body, facts);
          if (r === null) continue;
          out = out === null ? r : mergeFacts(out, r);
        }
        return out ?? facts;
      }
    }
    return facts;
  };

  const mergeFacts = (a: Facts, b: Facts): Facts => {
    const f: Facts = new Map();
    for (const [id, r] of a) {
      const rb = b.get(id);
      if (rb) f.set(id, join(r, rb));
    }
    return f;
  };

  let facts: Facts = new Map();
  for (const p of d.params) {
    if (p.sym && p.sym.ty.k === 'int') facts.set(p.sym.id, typeRange(p.sym.ty));
  }
  for (const s of d.body) {
    facts = walkStmt(s, facts) ?? new Map();
  }
  return { safeIndex, safeShift };
}

/**
 * f16 と f32 / f64 の変換。f16 を直接扱えないCPUでは、LLVM がこの名前の関数を呼ぶコードを出す。
 * 中身はランタイムの整数だけの関数（runtime/f16.h）に任せる。
 */
const F16_SHIMS = [
  'declare i32 @um_h2f(i16 zeroext) #0',
  'declare zeroext i16 @um_d2h(i64) #0',
  'define hidden float @__extendhfsf2(half %x) #0 {',
  '  %b = bitcast half %x to i16',
  '  %r = call i32 @um_h2f(i16 zeroext %b)',
  '  %f = bitcast i32 %r to float',
  '  ret float %f',
  '}',
  'define hidden half @__truncdfhf2(double %x) #0 {',
  '  %b = bitcast double %x to i64',
  '  %r = call zeroext i16 @um_d2h(i64 %b)',
  '  %h = bitcast i16 %r to half',
  '  ret half %h',
  '}',
  'define hidden half @__truncsfhf2(float %x) #0 {',
  '  %d = fpext float %x to double',
  '  %b = bitcast double %d to i64',
  '  %r = call zeroext i16 @um_d2h(i64 %b)',
  '  %h = bitcast i16 %r to half',
  '  ret half %h',
  '}',
  '',
];

/** スコープを抜けるときに解放する値。flag が真のときだけ生きている（移動済みなら偽）。 */
interface Droppable {
  ptr: string;
  flag: string;
  ty: Ty;
  /** スレッドの持ち手。ptr は { 持ち手, 引数の塊, 本数 } の置き場所。解放は「合流して戻り値を捨てる」。 */
  thread?: ThreadInfo;
}

/** スレッドの引数の塊の型（[0] が戻り値、[1..] が引数）と、関数の戻り値の型。 */
interface ThreadInfo {
  blk: string;
  ret: Ty;
}

/** RacyBuffer の要素の読み書きの並び。kernel の中は順序保証なし、外は monotonic（relaxed）。 */
function racyAlign(t: Ty): number {
  return (t.k === 'int' || t.k === 'float') ? Math.max(t.bits, 8) / 8 : 4;
}

function llTy(t: Ty): string {
  switch (t.k) {
    case 'int':
      return `i${t.bits}`;
    case 'float':
      // f8 / f4 は1バイトの符号（ビット列）として持つ。演算のたびに f32 へ展開する。
      return t.bits < 16 ? 'i8' : t.bits === 16 ? 'half' : t.bits === 32 ? 'float' : 'double';
    case 'bool':
      return 'i1';
    case 'void':
      return 'void';
    case 'struct':
      return `%S.${t.name}`;
    case 'array':
      return `[${t.len} x ${llTy(t.elem)}]`;
    case 'heap':
      return '%um.Array';
  }
}

/** f64 の値を最も近い f16 に丸めたビット列（偶数丸め）。runtime/f16.h の um_d2h と同じ手順。 */
export function halfBits(v: number): number {
  const buf = new DataView(new ArrayBuffer(8));
  buf.setFloat64(0, v);
  const d = buf.getBigUint64(0);
  const sign = Number((d >> 48n) & 0x8000n);
  const exp = Number((d >> 52n) & 0x7ffn);
  let man = d & 0xfffffffffffffn;
  if (exp === 0x7ff) return man === 0n ? sign | 0x7c00 : sign | 0x7e00 | Number(man >> 42n);
  if (exp === 0) return sign;
  let e = exp - 1023 + 15;
  man |= 1n << 52n;
  if (e >= 31) return sign | 0x7c00;
  let drop = 42n;
  if (e <= 0) {
    if (e < -10) return sign;
    drop = 42n + BigInt(1 - e);
    e = 0;
  }
  let kept = man >> drop;
  const rest = man & ((1n << drop) - 1n);
  const half = 1n << (drop - 1n);
  if (rest > half || (rest === half && (kept & 1n) === 1n)) kept++;
  let out = e === 0 ? Number(kept) : (e << 10) + (Number(kept) - 0x400);
  if (out >= 0x7c00) out = 0x7c00;
  return sign | out;
}

const F32T: Ty = { k: 'float', bits: 32 };
const isSmall = (t: Ty) => t.k === 'float' && t.bits < 16;

/**
 * f8 / f4 が表せる大きさ（0 以上）を小さい順に並べた表。番号がそのまま符号を除いたビット列になる。
 *   f8: 符号1・指数4・仮数3（OCP Microscaling の FP8 E4M3 と同じ）。番号 127 は NaN なので含めない。
 *   f4: 符号1・指数2・仮数1（同 FP4 E2M1）。
 * runtime/smallfloat.h の表と同じ値。
 */
export function smallMagnitudes(bits: 4 | 8): number[] {
  const out: number[] = [];
  if (bits === 8) {
    for (let code = 0; code < 127; code++) {
      const e = code >> 3;
      const m = code & 7;
      out.push(e === 0 ? 2 ** -6 * (m / 8) : 2 ** (e - 7) * (1 + m / 8));
    }
  } else {
    for (let code = 0; code < 8; code++) {
      const e = code >> 1;
      const m = code & 1;
      out.push(e === 0 ? m / 2 : 2 ** (e - 1) * (1 + m / 2));
    }
  }
  return out;
}

/** 値を f8 / f4 のビット列にする。runtime/smallfloat.h の um_f8_enc / um_f4_enc と同じ手順。 */
export function encodeSmall(v: number, bits: 4 | 8): number {
  if (Number.isNaN(v)) return bits === 8 ? 0x7f : 0;
  const mag = smallMagnitudes(bits);
  const sign = v < 0 || Object.is(v, -0) ? (bits === 8 ? 0x80 : 8) : 0;
  const a = Math.abs(v);
  let lo = 0;
  while (lo + 1 < mag.length && mag[lo + 1] <= a) lo++;
  if (lo + 1 === mag.length) return sign | lo;
  const mid = (mag[lo] + mag[lo + 1]) / 2;
  if (a < mid) return sign | lo;
  if (a > mid) return sign | (lo + 1);
  return sign | (lo & 1 ? lo + 1 : lo);
}

/**
 * 式の結果が、この式だけが持つ一時値か（使い終わったら解放する必要があるか）。
 * 変数やその中身（フィールド、要素）は持ち主が別にいる。一時値の中身（`f().name`、`f()[0]`）も、
 * 持ち主は元の一時値なので、ここでは false。
 */
function isOwnedTemp(e: Expr): boolean {
  return !isPlace(e) && e.kind !== 'field' && e.kind !== 'index';
}

/** RacyBuffer の要素を指す式か。 */
function isRacyElem(e: Expr): boolean {
  return e.kind === 'index' && e.base.ty!.k === 'heap' && e.base.ty!.racy === true;
}
const STR_TY: Ty = { k: 'heap', elem: { k: 'int', bits: 8, signed: false }, str: true };

/** LLVMの浮動小数点定数。10進表記は正確に表せない値で拒否されるため、常に16進のビット表現で出す。 */
function floatConst(v: number, t: Ty): string {
  if (t.k === 'float' && t.bits < 16) return String(encodeSmall(v, t.bits as 4 | 8));
  if (t.k === 'float' && t.bits === 16) return '0xH' + halfBits(v).toString(16).toUpperCase().padStart(4, '0');
  const x = t.k === 'float' && t.bits === 32 ? Math.fround(v) : v;
  const buf = new DataView(new ArrayBuffer(8));
  buf.setFloat64(0, x);
  return '0x' + buf.getBigUint64(0).toString(16).toUpperCase().padStart(16, '0');
}

class ModuleGen {
  prog: Program;
  target: TargetKind;
  /** 使った組み込み関数の宣言（重複なし）。 */
  intrinsics = new Set<string>();
  /** ウィンドウの組み込み（windowOpen など）を使ったか。使ったときだけランタイムの宣言を出す。 */
  usesWindow = false;
  /** WebAssembly の組み込み（wasmCompile など）を使ったか。使ったときだけランタイムの宣言を出す。 */
  usesWasm = false;
  /** 動画・音声の組み込み（mediaOpen など）を使ったか。使ったときだけランタイムの宣言を出す。 */
  usesMedia = false;
  /** GPU で実行する kernel。名前 → 番号と、WGSL を入れたグローバル定数。 */
  gpuKernels = new Map<string, { id: number; global: string; bytes: number }>();
  /** プログラムのどこかに f16 が現れるか。現れるときだけ変換の関数を出す。 */
  usesF16(): boolean {
    return this.prog.usesF16 === true;
  }

  /** モジュールの先頭に置くグローバル定数。 */
  globals: string[] = [];
  /** kernel を添字の区間について実行する関数（kernel の名前 → 関数の名前と、引数をまとめた型）。 */
  ranges = new Map<string, { fn: string; ctxTy: string }>();
  rangeFns: string[] = [];

  /** usize のLLVM型。 */
  usz: string;

  /** テスト用: AddressSanitizer（true）か ThreadSanitizer（'thread'）の計装を入れる。 */
  sanitize: boolean | 'thread';

  /**
   * kernel を GPU で実行する経路。wasm32 は WebGPU、macOS のネイティブは Metal（UM_METAL=1）、
   * ネイティブで UM_WGPU=1 なら wgpu-native（WGSL。runtime/wgpu.c）。なければ null。
   */
  gpu: 'webgpu' | 'metal' | 'wgpu' | null;

  /** wgpu のランタイムを直接呼ぶ組み込み（gpu.loadKernel / gpu.runKernel / gpu.ready / gpu.wait）を使ったか。 */
  usesGpuRuntime = false;

  /** wgpu の経路で、GPU を使う呼び出し（GPU で実行しうる dispatch、gpu.loadKernel）があるか。なければ wgpu をリンクしない。 */
  wgpuUsed(): boolean {
    return this.gpu === 'wgpu' && (this.gpuKernels.size > 0 || this.usesGpuRuntime);
  }

  /** macOS の arm64 向けか（SIMD の dot に NEON の sdot を直接使う。Apple の arm64 はどれも dotprod を持つ）。 */
  appleArm: boolean;

  /**
   * Windows 向けか。Windows では LLVM が大きなスタックフレームに必ず __chkstk の呼び出しを入れるので、
   * `"probe-stack"="inline-asm"` は付けない（Windows では、それが `inline-asm` という名前の関数の呼び出しになる）。
   */
  windows: boolean;

  /** wasm32-plugin（import なし。export の Buffer 引数を (位置, 長さ) で受ける。src/driver.ts の TargetName を参照）。 */
  plugin: boolean;

  constructor(prog: Program, target: TargetKind, sanitize: boolean | 'thread' = false, metal = false, appleArm = false, windows = false, plugin = false,
    wgpu = false) {
    this.prog = prog;
    this.target = target;
    this.appleArm = appleArm;
    this.windows = windows;
    this.plugin = plugin;
    // プラグインは GPU を使わない（dispatch は CPU の1スレッド）。
    this.gpu = target === 'wasm32' && !plugin ? 'webgpu' : metal ? 'metal' : wgpu ? 'wgpu' : null;
    this.sanitize = sanitize;
    this.usz = `i${prog.ptrBits}`;
  }

  isCopy(t: Ty): boolean {
    return isCopyTy(this.prog.structs, t);
  }

  /**
   * kernel を、添字の区間 [begin, end) について順に呼ぶ関数を作る。引数は ctx の構造体から読む。
   * ネイティブでは、ランタイムがこの関数を複数のスレッドから区間ごとに呼ぶ。
   *
   * 添字が i ± 定数 の形の Buffer アクセスがある kernel は、区間を3つに分ける:
   * その添字がすべて範囲内になる中央の区間は「検査なし版」で実行し（切り詰めがないのでベクトル化される）、
   * 両端だけ通常版で実行する。結果は通常版だけで実行した場合と同じ。
   */
  rangeFn(sig: FnSig, types: string[]): { fn: string; ctxTy: string } {
    let r = this.ranges.get(sig.key);
    if (r) return r;
    const d = sig.decl;
    const fn = `@um.range.${sig.key}`;
    const ctxTy = `{ ${types.length > 0 ? types.join(', ') : 'i32'} }`;
    const kernel = this.fnName(sig.key);
    const usz = this.usz;

    // Buffer の引数ごとに、i ± 定数 の添字の最小と最大を集める。
    const idx = d.params[0].sym!.id;
    const offsets = new Map<number, { min: number; max: number }>();
    const locals = affineLocals(d.body, idx);
    walkExprs(d.body, (e) => {
      // RacyBuffer のアトミック演算（r.fetchAdd(j, v) など）の添字も、検査なし版では切り詰めを省くので数える。
      const racy = e.kind === 'mcall' && e.key?.startsWith('Racy.') === true;
      const base = e.kind === 'index' ? e.base : e.kind === 'mcall' && racy ? e.recv : null;
      const index = e.kind === 'index' ? e.index : e.kind === 'mcall' && racy ? e.args[0].expr : null;
      if (!base || !index || base.kind !== 'ident' || base.ty!.k !== 'heap') return;
      const c = indexRange(index, idx, locals);
      if (c === null) return;
      const o = offsets.get(base.sym!.id) ?? { min: 0, max: 0 };
      offsets.set(base.sym!.id, { min: Math.min(o.min, c.min), max: Math.max(o.max, c.max) });
    });

    const out = [`define internal void ${fn}(i32 %begin, i32 %end, ptr %ctx) #0 {`, 'entry:'];
    types.forEach((t, i) => {
      out.push(`  %p${i} = getelementptr inbounds ${ctxTy}, ptr %ctx, i32 0, i32 ${i}`);
      out.push(`  %a${i} = load ${t}, ptr %p${i}`);
    });
    const args = (i: string) => [`i32 ${i}`, ...types.map((t, k) => `${t} %a${k}`)].join(', ');
    // from から to まで（i64）、callee を呼ぶループ。
    // atLeast: この区間では添字が必ずこの値以上であること（区間の求め方から成り立つ）を最適化に伝える。
    // kernel の中の `if (i > 0)` のような分岐が消え、ループがベクトル化されやすくなる。
    const loop = (name: string, from: string, to: string, callee: string, next: string, atLeast = 0) => {
      out.push(
        `  br label %${name}.cond`,
        `${name}.cond:`,
        `  %${name}.i = phi i64 [ ${from}, %${name}.pre ], [ %${name}.next, %${name}.body ]`,
        `  %${name}.more = icmp ult i64 %${name}.i, ${to}`,
        `  br i1 %${name}.more, label %${name}.body, label %${next}`,
        `${name}.body:`,
      );
      if (atLeast > 0) {
        this.intrinsics.add('declare void @llvm.assume(i1)');
        out.push(`  %${name}.low = icmp uge i64 %${name}.i, ${atLeast}`, `  call void @llvm.assume(i1 %${name}.low)`);
      }
      out.push(
        `  %${name}.i32 = trunc i64 %${name}.i to i32`,
        `  call void ${callee}(${args(`%${name}.i32`)})`,
        `  %${name}.next = add i64 %${name}.i, 1`,
        `  br label %${name}.cond`,
      );
    };

    out.push('  %b64 = zext i32 %begin to i64', '  %e64 = zext i32 %end to i64');
    if (offsets.size === 0) {
      out.push('  br label %all.pre', 'all.pre:');
      loop('all', '%b64', '%e64', kernel, 'done');
    } else {
      // 中央の区間 [lo, hi) を求める: すべての Buffer について 0 <= i + min かつ i + max < 長さ。
      this.intrinsics.add('declare i64 @llvm.umin.i64(i64, i64)');
      this.intrinsics.add('declare i64 @llvm.umax.i64(i64, i64)');
      this.intrinsics.add('declare i64 @llvm.usub.sat.i64(i64, i64)');
      let lo = '%b64';
      let hi = '%e64';
      let n = 0;
      d.params.slice(1).forEach((p, k) => {
        const o = offsets.get(p.sym!.id);
        if (!o) return;
        const id = n++;
        out.push(`  %lp${id} = getelementptr inbounds %um.Array, ptr %a${k}, i32 0, i32 1`);
        out.push(`  %len${id} = load ${usz}, ptr %lp${id}, !tbaa ${HDR}`);
        let len = `%len${id}`;
        if (usz !== 'i64') {
          out.push(`  %lenw${id} = zext ${usz} ${len} to i64`);
          len = `%lenw${id}`;
        }
        if (o.min < 0) {
          out.push(`  %lo${id} = call i64 @llvm.umax.i64(i64 ${lo}, i64 ${-o.min})`);
          lo = `%lo${id}`;
        }
        out.push(`  %lim${id} = call i64 @llvm.usub.sat.i64(i64 ${len}, i64 ${o.max})`);
        out.push(`  %hi${id} = call i64 @llvm.umin.i64(i64 ${hi}, i64 %lim${id})`);
        hi = `%hi${id}`;
      });
      // begin <= lo <= hi <= end になるように揃える。
      out.push(`  %lo = call i64 @llvm.umin.i64(i64 ${lo}, i64 %e64)`);
      out.push(`  %hi = call i64 @llvm.umax.i64(i64 ${hi}, i64 %lo)`);
      out.push('  br label %head.pre', 'head.pre:');
      loop('head', '%b64', '%lo', kernel, 'mid.pre');
      out.push('mid.pre:');
      const least = Math.max(0, ...[...offsets.values()].map((o) => -o.min));
      loop('mid', '%lo', '%hi', `${kernel}.fast`, 'tail.pre', least);
      out.push('tail.pre:');
      loop('tail', '%hi', '%e64', kernel, 'done');

      const fast = new FnGen(this, d);
      fast.fast = true;
      fast.fastLocals = locals;
      this.rangeFns.push(fast.generate('#40', '.fast'), '');
    }
    out.push('done:', '  ret void', '}', '');
    this.rangeFns.push(...out);

    r = { fn, ctxTy };
    this.ranges.set(sig.key, r);
    return r;
  }

  /** グループで実行する関数（kernel の名前 → 区間の関数の名前と、引数をまとめた型）。 */
  groupRanges = new Map<string, { fn: string; ctxTy: string }>();

  /**
   * dispatchGroups の CPU での実行: グループの番号の区間 [begin, end) について、グループを1つずつ実行する関数を作る。
   * ctx は { グループの大きさ, グループの数, 引数... }。共有配列はグループごとに 0 で埋めてから渡す。
   * barrier を使う kernel は、スレッドごとの変数の置き場所（グループの大きさ × 1スレッドぶん）を区間ごとに1回確保する。
   */
  groupRangeFn(sig: FnSig, types: string[]): { fn: string; ctxTy: string } {
    let r = this.groupRanges.get(sig.key);
    if (r) return r;
    const d = sig.decl;
    const fn = `@um.grange.${sig.key}`;
    const ctxTy = `{ ${['i32', 'i32', ...types].join(', ')} }`;
    const g = new FnGen(this, d);
    const body = g.generateGroup();
    const { bytes } = sharedLayout(d);
    const usz = this.usz;

    const out = [`define internal void ${fn}(i32 %begin, i32 %end, ptr %ctx) #0 {`, 'entry:'];
    if (bytes > 0) out.push(`  %sh = alloca [${bytes} x i8], align 16`);
    out.push(`  %ps = getelementptr inbounds ${ctxTy}, ptr %ctx, i32 0, i32 0`, '  %size = load i32, ptr %ps');
    out.push(`  %pc = getelementptr inbounds ${ctxTy}, ptr %ctx, i32 0, i32 1`, '  %count = load i32, ptr %pc');
    types.forEach((t, i) => {
      out.push(`  %p${i} = getelementptr inbounds ${ctxTy}, ptr %ctx, i32 0, i32 ${i + 2}`);
      out.push(`  %a${i} = load ${t}, ptr %p${i}`);
    });
    let scratch = 'null';
    if (g.ptTy) {
      // スレッドごとの変数の置き場所。グループの大きさは 1024 以下なので、大きくてもスタックではなくヒープに置く。
      // wasm32 では usize が i32 なので、広げる命令は要らない（i32 から i32 への zext は IR の誤り）。
      out.push(usz === 'i32' ? '  %sz = add i32 %size, 0' : `  %sz = zext i32 %size to ${usz}`);
      out.push(`  %scr = call ptr @um_alloc(${usz} %sz, ${usz} ptrtoint (ptr getelementptr (${g.ptTy}, ptr null, i32 1) to ${usz}))`);
      scratch = '%scr';
    }
    out.push(
      '  br label %cond',
      'cond:',
      '  %g = phi i32 [ %begin, %entry ], [ %gn, %body ]',
      '  %more = icmp ult i32 %g, %end',
      '  br i1 %more, label %body, label %done',
      'body:',
    );
    if (bytes > 0) {
      this.intrinsics.add('declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)');
      out.push(`  call void @llvm.memset.p0.i64(ptr align 16 %sh, i8 0, i64 ${bytes}, i1 false)`);
    }
    const args = ['i32 %g', 'i32 %size', 'i32 %count', `ptr ${bytes > 0 ? '%sh' : 'null'}`, `ptr ${scratch}`, ...types.map((t, k) => `${t} %a${k}`)];
    out.push(`  call void ${this.fnName(sig.key)}.group(${args.join(', ')})`, '  %gn = add i32 %g, 1', '  br label %cond', 'done:');
    if (g.ptTy) out.push('  call void @um_free(ptr %scr)');
    out.push('  ret void', '}', '');
    this.rangeFns.push(body, '', ...out);

    r = { fn, ctxTy };
    this.groupRanges.set(sig.key, r);
    return r;
  }

  /** f8 / f4 の変換をランタイムに頼んだか（宣言を出すかどうか）。 */
  usesSmall = false;
  smallTables = new Set<number>();

  /** f8 / f4 のビット列 → f32 の値の表を置き、その名前を返す。 */
  smallTable(bits: 4 | 8): string {
    const name = `@um.f${bits}.table`;
    if (!this.smallTables.has(bits)) {
      this.smallTables.add(bits);
      const mag = smallMagnitudes(bits);
      const half = bits === 8 ? 128 : 8;
      const vals: string[] = [];
      for (let code = 0; code < half * 2; code++) {
        const m = mag[code % half];
        // f8 の 0x7F / 0xFF は NaN。
        vals.push(`float ${floatConst(m === undefined ? NaN : code >= half ? -m : m, F32T)}`);
      }
      this.globals.push(`${name} = private unnamed_addr constant [${half * 2} x float] [${vals.join(', ')}]`);
    }
    return name;
  }

  /** バイト列を定数として置き、その名前を返す。同じ内容は1回だけ置く。 */
  byteGlobals = new Map<string, string>();
  bytesGlobal(bytes: Uint8Array): string {
    let text = '';
    for (const b of bytes) {
      text += b >= 0x20 && b < 0x7f && b !== 0x22 && b !== 0x5c ? String.fromCharCode(b) : `\\${b.toString(16).toUpperCase().padStart(2, '0')}`;
    }
    let g = this.byteGlobals.get(text);
    if (!g) {
      g = `@um.str.${this.byteGlobals.size}`;
      this.globals.push(`${g} = private unnamed_addr constant [${bytes.length} x i8] c"${text}"`);
      this.byteGlobals.set(text, g);
    }
    return g;
  }

  /** WGSL にできない kernel（simdgroup の集計を使う）。wgpu の経路では GPU の経路を作らず、CPU で実行する。 */
  noWgsl = new Set<string>();

  /** 出力した定数の配列（定数の名前 → グローバルの名前）。 */
  constArrays = new Map<string, string>();

  /**
   * 最上位の定数の配列を、読み取り専用のグローバル（LLVM の constant）として出力し、その名前を返す。
   * 置き場所はプログラムの読み取り専用のデータで、ヒープもコピーも使わない（添字の検査は固定長配列と同じ）。
   */
  constArray(sym: Sym): string {
    const g = sym.global!;
    let name = this.constArrays.get(g.name);
    if (!name) {
      name = `@um.k.${g.name}`;
      const init = (v: unknown, t: Ty): string => {
        if (t.k === 'array') return `[${(v as unknown[]).map((x) => `${llTy(t.elem)} ${init(x, t.elem)}`).join(', ')}]`;
        if (t.k === 'bool') return v ? 'true' : 'false';
        if (t.k === 'float') return floatConst(v as number, t);
        return String(v as bigint);
      };
      this.globals.push(`${name} = private unnamed_addr constant ${llTy(sym.ty)} ${init(g.value, sym.ty)}, align 16`);
      this.constArrays.set(g.name, name);
    }
    return name;
  }

  /**
   * kernel を WGSL / MSL に変換し、文字列定数として埋め込む。同じ kernel は1回だけ。
   * groups: dispatchGroups 用（グループの番号から添字を作る形）。同じ kernel でも普通の dispatch とは別のシェーダになる。
   */
  gpuKernel(name: string, groups = false): { id: number; global: string; bytes: number } | null {
    const key = groups ? `${name}#groups` : name;
    let k = this.gpuKernels.get(key);
    if (!k) {
      if (this.noWgsl.has(name)) return null;
      const decl = this.prog.fns.get(name)!.decl;
      // wasm32 と wgpu には WGSL、Metal には MSL のソースを埋め込む。実行時にランタイムがコンパイルする。
      let source: string;
      try {
        source = this.gpu === 'metal' ? emitMsl(this.prog, decl, groups) : emitWgsl(this.prog, decl, groups).source;
      } catch (err) {
        if (err instanceof CompileError && err.code === WGSL_SIMDGROUP) {
          // wasm32（WebGPU）では、simdgroup の集計はコンパイルエラーのまま。
          if (this.gpu === 'webgpu') {
            throw new CompileError('kernel-restriction', err.message.replace('by WebGPU (WGSL)', 'on WebGPU (wasm32)'), err.pos,
              'simdgroup functions run on Metal and on the CPU; build with `--target host`');
          }
          // wgpu では、simdgroup の集計を使う kernel は CPU でだけ実行する。
          this.noWgsl.add(name);
          return null;
        }
        throw err;
      }
      const bytes = Buffer.from(source, 'utf8');
      // 印字できる ASCII 以外と、引用符・バックスラッシュは \XX で書く。
      let text = '';
      for (const b of bytes) {
        text += b >= 0x20 && b < 0x7f && b !== 0x22 && b !== 0x5c ? String.fromCharCode(b) : `\\${b.toString(16).toUpperCase().padStart(2, '0')}`;
      }
      const global = `@um.shader.${name}${groups ? '.groups' : ''}`;
      this.globals.push(`${global} = private unnamed_addr constant [${bytes.length} x i8] c"${text}"`);
      k = { id: this.gpuKernels.size, global, bytes: bytes.length };
      this.gpuKernels.set(key, k);
    }
    return k;
  }

  /** 生成する関数に付ける属性。 */
  fnAttrs(): string {
    let a = 'nounwind';
    // ネイティブでは大きなスタックフレームがガードページを飛び越えないよう、スタックプローブを入れる。
    if (this.target === 'native' && !this.windows) a += ' "probe-stack"="inline-asm"';
    if (this.sanitize === 'thread') a += ' sanitize_thread';
    else if (this.sanitize) a += ' sanitize_address';
    return a;
  }

  /** 型の大きさ（配列の要素間隔）を表す定数式。 */
  sizeOf(t: Ty): string {
    return `ptrtoint (ptr getelementptr (${llTy(t)}, ptr null, i32 1) to ${this.usz})`;
  }

  /** ヒープを持つ構造体の解放関数。フィールドを順に解放する。 */
  /** 要素の型 → 「配列の全要素を解放する関数」の名前。 */
  elemDrops = new Map<string, string>();
  elemDropFns: string[] = [];

  /**
   * ヒープを持つ要素の配列について、全要素を解放する関数を作る（配列そのものの領域は解放しない）。
   * 引数は配列のヘッダ（%um.Array）の場所。
   */
  dropElems(elem: Ty): string {
    const key = tyStr(elem);
    let name = this.elemDrops.get(key);
    if (name) return name;
    name = `@um.dropelems.${this.elemDrops.size}`;
    this.elemDrops.set(key, name); // 先に登録する（要素が自分自身を含む型でも止まるように）
    const usz = this.usz;
    const et = llTy(elem);
    const out = [
      `define internal void ${name}(ptr %a) #0 {`,
      'entry:',
      '  %data = load ptr, ptr %a',
      '  %lp = getelementptr inbounds %um.Array, ptr %a, i32 0, i32 1',
      `  %len = load ${usz}, ptr %lp`,
      '  br label %cond',
      'cond:',
      `  %i = phi ${usz} [ 0, %entry ], [ %next, %body ]`,
      `  %more = icmp ult ${usz} %i, %len`,
      '  br i1 %more, label %body, label %done',
      'body:',
      `  %p = getelementptr inbounds ${et}, ptr %data, ${usz} %i`,
    ];
    if (elem.k === 'heap') {
      if (!this.isCopy(elem.elem)) out.push(`  call void ${this.dropElems(elem.elem)}(ptr %p)`);
      out.push('  %d = load ptr, ptr %p', '  call void @um_free(ptr %d)');
    } else if (elem.k === 'struct') {
      out.push(`  call void @um.drop.${elem.name}(ptr %p)`);
    }
    out.push(`  %next = add ${usz} %i, 1`, '  br label %cond', 'done:', '  ret void', '}', '');
    this.elemDropFns.push(...out);
    return name;
  }

  /** スレッドの入口（関数ごと）。名前 → 生成した関数の名前。 */
  threadEntries = new Map<string, string>();
  threadFns: string[] = [];

  /**
   * スレッドの入口: 引数の塊（blk 型。[0] に戻り値を書く）から引数を取り出して関数を呼ぶ。
   * 読み取り借用（ヒープを持つ型）は塊に場所が入っている。コピーされる型と own の値は塊の中にあり、
   * その場所を渡す（own は呼ばれた側が受け取って解放する）。
   */
  threadEntry(sig: FnSig, blk: string): string {
    let name = this.threadEntries.get(sig.key);
    if (name) return name;
    name = `@"um.thread entry.${sig.key}"`;
    this.threadEntries.set(sig.key, name);
    const out = [`define internal void ${name}(ptr %b) #0 {`, 'entry:'];
    const args: string[] = [];
    sig.params.forEach((p, i) => {
      out.push(`  %f${i} = getelementptr inbounds ${blk}, ptr %b, i32 0, i32 ${i + 1}`);
      if (passByPtr(p.mode, p.ty)) {
        if (p.mode === 'none' && !this.isCopy(p.ty)) {
          out.push(`  %a${i} = load ptr, ptr %f${i}`);
          args.push(`ptr %a${i}`);
        } else {
          args.push(`ptr %f${i}`);
        }
      } else {
        out.push(`  %a${i} = load ${llTy(p.ty)}, ptr %f${i}`);
        args.push(`${llTy(p.ty)} %a${i}`);
      }
    });
    const call = `${this.fnName(sig.key)}(${args.join(', ')})`;
    if (sig.ret.k === 'void') {
      out.push(`  call void ${call}`);
    } else {
      out.push(`  %r = call ${llTy(sig.ret)} ${call}`, `  %rp = getelementptr inbounds ${blk}, ptr %b, i32 0, i32 0`, `  store ${llTy(sig.ret)} %r, ptr %rp`);
    }
    out.push('  ret void', '}', '');
    this.threadFns.push(...out);
    return name;
  }

  dropFns(): string[] {
    const out: string[] = [];
    for (const s of this.prog.structs.values()) {
      if (this.isCopy({ k: 'struct', name: s.name })) continue;
      out.push(`define internal void @um.drop.${s.name}(ptr %p) #0 {`);
      s.fields.forEach((f, i) => {
        if (this.isCopy(f.ty)) return;
        out.push(`  %f${i} = getelementptr inbounds %S.${s.name}, ptr %p, i32 0, i32 ${i}`);
        if (f.ty.k === 'heap') {
          if (!this.isCopy(f.ty.elem)) out.push(`  call void ${this.dropElems(f.ty.elem)}(ptr %f${i})`);
          out.push(`  %d${i} = load ptr, ptr %f${i}`);
          out.push(`  call void @um_free(ptr %d${i})`);
        } else if (f.ty.k === 'struct') {
          out.push(`  call void @um.drop.${f.ty.name}(ptr %f${i})`);
        }
      });
      out.push('  ret void', '}', '');
    }
    return out;
  }

  /**
   * プラグイン（wasm32-plugin）の export の包み。ホストからは数と Buffer しか渡せないので、
   * 引数は数（整数・f32・f64・bool）か Buffer（借用か mut）、戻り値は数か void に限る。
   * Buffer の引数は、ホストから (要素の位置 i32, 要素の数 i32) の2つで受け取り、配列の管理情報をここで作って本体に貸す。
   * 領域はホストのもの（ふつうは umAlloc で確保したもの）で、本体は借りるだけ。本体が mut の Buffer を差し替えたら
   * （ホストの領域を解放してしまうので）停止（トラップ）する。
   */
  pluginExport(d: FnDecl, attr: string): string {
    const name = d.name;
    const bad = (pos: Pos, what: string): never => {
      throw new CompileError('plugin-export', `the exported function '${name}' ${what}`, pos,
        'a wasm32-plugin export can take numbers, bool and Buffer<T> (borrowed or `mut`), and return a number, bool or void');
    };
    const scalar = (t: Ty) => t.k === 'int' || t.k === 'bool' || (t.k === 'float' && (t.bits === 32 || t.bits === 64));
    const ret = d.ret!;
    if (ret.k !== 'void' && !scalar(ret)) bad(d.pos, `returns ${tyStr(ret)}, which the host cannot receive`);
    const params: string[] = [];
    const pre: string[] = [];
    const args: string[] = [];
    const post: string[] = [];
    d.params.forEach((p, i) => {
      const t = p.ty!;
      if (t.k === 'heap' && t.buffer && !t.racy && !t.str) {
        if (p.mode === 'own') bad(p.pos, `takes '${p.name}' as \`own\`; the host keeps the memory, so lend it instead`);
        params.push(`i32 %b${i}.at`, `i32 %b${i}.n`);
        pre.push(
          `  %b${i}.p = inttoptr i32 %b${i}.at to ptr`,
          `  %b${i}.h = alloca %um.Array`,
          `  %b${i}.v0 = insertvalue %um.Array undef, ptr %b${i}.p, 0`,
          `  %b${i}.v1 = insertvalue %um.Array %b${i}.v0, i32 %b${i}.n, 1`,
          `  %b${i}.v2 = insertvalue %um.Array %b${i}.v1, i32 %b${i}.n, 2`,
          `  store %um.Array %b${i}.v2, ptr %b${i}.h`,
        );
        args.push(`ptr %b${i}.h`);
        if (p.mode === 'mut') {
          post.push(
            `  %b${i}.after = load ptr, ptr %b${i}.h`,
            `  %b${i}.same = icmp eq ptr %b${i}.after, %b${i}.p`,
            `  br i1 %b${i}.same, label %b${i}.ok, label %replaced`,
            `b${i}.ok:`,
          );
        }
      } else if (scalar(t) && p.mode === 'none') {
        params.push(`${llTy(t)} %a${i}`);
        args.push(`${llTy(t)} %a${i}`);
      } else {
        bad(p.pos, `takes '${p.name}' of type ${p.mode === 'none' ? '' : `${p.mode} `}${tyStr(t)}, which the host cannot pass`);
      }
    });
    const rt = llTy(ret);
    const call = `call ${rt} ${this.fnName(this.declKey(d))}(${args.join(', ')})`;
    const out = [`define ${rt} @um.export.${name}(${params.join(', ')}) ${attr} {`, ...pre];
    out.push(ret.k === 'void' ? `  ${call}` : `  %r = ${call}`, ...post, ret.k === 'void' ? '  ret void' : `  ret ${rt} %r`);
    if (post.length > 0) out.push('replaced:', `  call void @um_panic(i32 ${PANIC_PLUGIN_BUFFER}, i32 ${d.pos.line})`, '  unreachable');
    out.push('}');
    return out.join('\n');
  }

  fnName(key: string): string {
    return `@um.${key}`;
  }

  /** 宣言から生成コード上の名前を作る。checker の FnSig.key と同じ規則。 */
  declKey(d: FnDecl): string {
    if (!d.cls) return d.name;
    return d.isCtor ? `${d.cls}.new` : `${d.cls}.${d.name}`;
  }

  paramSig(mode: string, ty: Ty): string {
    if (!passByPtr(mode as 'none' | 'mut' | 'own', ty)) return llTy(ty);
    // 借用規則により別名がないことが保証されるので noalias を付けられる。
    return mode === 'mut' ? 'ptr noalias' : 'ptr noalias readonly';
  }

  generate(): string {
    const out: string[] = [];
    out.push('; UnifiedMozaic が生成したLLVM IR。手で編集しない。');
    out.push('');

    // 可変長配列: 要素へのポインタ、長さ、容量。runtime/ の UmArray と同じ並び。
    out.push(`%um.Array = type { ptr, ${this.usz}, ${this.usz} }`);
    for (const s of this.prog.structs.values()) {
      out.push(`%S.${s.name} = type { ${s.fields.map((f) => llTy(f.ty)).join(', ')} }`);
    }
    out.push('');

    const bodies: string[] = [];
    const exportAttrs: string[] = [];
    let attrId = 100; // 100 以降は export 用。3〜 は import、40 は検査なし版の kernel
    for (const d of this.prog.decls) {
      let attr = '#0';
      if (d.exported && this.plugin) {
        // プラグインでは、ホストから呼べる形の包みを export する（本体はふつうの関数のまま）。
        const wattr = `#${attrId++}`;
        exportAttrs.push(`attributes ${wattr} = { ${this.fnAttrs()} "wasm-export-name"="${d.name}" }`);
        bodies.push(this.pluginExport(d, wattr), '');
      } else if (d.exported) {
        attr = `#${attrId++}`;
        exportAttrs.push(`attributes ${attr} = { ${this.fnAttrs()} "wasm-export-name"="${d.name}" }`);
      }
      // group.* を使う kernel は普通の関数としては呼べない。CPU では dispatchGroups がグループごとの関数（groupRangeFn）を作る。
      if (d.isKernel && d.usesGroup) continue;
      bodies.push(new FnGen(this, d).generate(attr));
      bodies.push('');
    }

    // ランタイム関数。wasmでは env モジュールからの import になる。
    const imports = ['void um_print_i64(i64)', 'void um_print_u64(i64)', 'void um_print_f64(double)', 'void um_print_bool(i32)', `void um_print_str(ptr, ${this.usz})`];
    // `gpu.prefer(on)` の印（0 でなければ、自動の dispatch は GPU が使えるかぎり GPU で実行する）。GPU の経路のないビルドでも覚える。
    out.push('@um.gpu_prefer = internal global i32 0, align 4');
    if (this.gpu === 'webgpu') {
      // GPU（WebGPU）での実行。実装は runtime/host.mjs。
      imports.push('i32 um_gpu_available()', 'i32 um_gpu_pick(i32, i32)', 'void um_gpu_dispatch(i32, ptr, i32, i32, ptr, i32, ptr, i32)');
      imports.push('void um_gpu_group(i32)');
    }
    if (this.gpu === 'wgpu') {
      // wgpu-native での実行。実装は runtime/wgpu.c。GPU を使う呼び出しがなければ宣言せず（ドライバは wgpu をリンクしない）、
      // Buffer の確保は普通のゼロ埋めの確保にする。
      if (this.wgpuUsed()) {
        out.push(`declare ptr @um_wgpu_alloc(${this.usz}, ${this.usz}) #0`);
        out.push('declare i32 @um_gpu_available() #0', 'declare i32 @um_gpu_pick(i32, i32) #0', 'declare void @um_gpu_dispatch(i32, ptr, i32, i32, ptr, i32, ptr, i32) #0');
        out.push('declare void @um_gpu_cpu_done(i32, i32) #0', 'declare void @um_gpu_flush() #0', 'declare void @um_gpu_group(i32) #0');
        out.push('declare i64 @um_gpu_submit(i32, ptr, i32, i32, ptr, i32, ptr, i32) #0', 'declare i32 @um_gpu_ready(i64) #0', 'declare void @um_gpu_wait(i64) #0');
        out.push(`declare i32 @um_gpu_load(ptr, ${this.usz}) #0`, 'declare i32 @um_gpu_run(i32, i32, ptr, i32, ptr, i32) #0');
        out.push('declare i32 @um_gpu_set_flags(i32) #0', 'declare i32 @um_gpu_pick_bufs(i32, i32, ptr, i32) #0');
      } else {
        out.push(`define internal ptr @um_wgpu_alloc(${this.usz} %n, ${this.usz} %s) #0 {`, `  %p = call ptr @um_alloc_zeroed(${this.usz} %n, ${this.usz} %s)`, '  ret ptr %p', '}');
      }
    }
    if (this.gpu === 'metal') {
      // Metal での実行。実装は runtime/metal.m（一緒にリンクする）。
      out.push(`declare ptr @um_alloc_shared(${this.usz}, ${this.usz}) #0`);
      out.push('declare i32 @um_gpu_available() #0', 'declare i32 @um_gpu_pick(i32, i32) #0', 'declare void @um_gpu_dispatch(i32, ptr, i32, i32, ptr, i32, ptr, i32) #0');
      out.push('declare void @um_gpu_cpu_done(i32, i32) #0', 'declare void @um_gpu_flush() #0', 'declare void @um_gpu_group(i32) #0');
      out.push('declare i64 @um_gpu_submit(i32, ptr, i32, i32, ptr, i32, ptr, i32) #0', 'declare i32 @um_gpu_ready(i64) #0', 'declare void @um_gpu_wait(i64) #0');
      out.push('declare i32 @um_gpu_set_flags(i32) #0', 'declare void @um_gpu_commit() #0', 'declare void @um_gpu_untracked(ptr) #0');
    }
    const importAttrs: string[] = [];
    imports.forEach((sig, i) => {
      const name = sig.slice(sig.indexOf(' ') + 1, sig.indexOf('('));
      out.push(`declare ${sig.replace(' ', ' @')} #${3 + i}`);
      importAttrs.push(`attributes #${3 + i} = { nounwind "wasm-import-module"="env" "wasm-import-name"="${name}" }`);
    });
    out.push('declare void @um_panic(i32, i32) #2');
    // スレッド。ネイティブは runtime/native.c（pthread）、wasm は runtime/wasm_rt.c（合流の時点でその場で実行する）。
    out.push(`declare ptr @um_thread_start(ptr, ptr, ${this.usz}, ${this.usz}, i32) #0`, 'declare void @um_thread_join(ptr) #0');
    out.push('declare i32 @um_thread_cpus() #0', 'declare void @um_thread_yield() #0', 'declare i32 @um_is_main() #0');
    // ヒープ。wasm ではランタイム（runtime/wasm_rt.c）を一緒にリンクするので import ではない。
    out.push(`declare ptr @um_alloc(${this.usz}, ${this.usz}) #0`);
    out.push(`declare ptr @um_alloc_zeroed(${this.usz}, ${this.usz}) #0`);
    out.push(`declare void @um_fill(ptr, ${this.usz}, ${this.usz}, i32, i64, i32) #0`);
    out.push(`declare ${this.usz} @um_grow_cap(${this.usz}) #0`);
    out.push(`declare ptr @um_grow_data(ptr, ${this.usz}, ${this.usz}, ${this.usz}) #0`);
    out.push('declare void @um_free(ptr) #0');
    out.push(`declare ptr @um_str_concat(ptr, ${this.usz}, ptr, ${this.usz}) #0`);
    out.push(`declare i32 @um_str_eq(ptr, ${this.usz}, ptr, ${this.usz}) #0`);
    out.push('declare ptr @um_str_from_i64(i64, ptr) #0');
    out.push('declare ptr @um_str_from_u64(i64, ptr) #0');
    out.push('declare ptr @um_str_from_f64(double, ptr) #0');
    // 入出力の組み込み（readLine など）。ネイティブは runtime/native.c、wasm は runtime/wasm_rt.c がホストの関数を呼ぶ。
    out.push('declare i32 @um_read_line(ptr, ptr) #0', 'declare i32 @um_input_ready() #0', 'declare double @um_clock() #0');
    out.push(`declare i32 @um_read_file(ptr, ${this.usz}, ptr, ptr) #0`, `declare i32 @um_write_file(ptr, ${this.usz}, ptr, ${this.usz}) #0`);
    out.push(`declare i32 @um_file_size(ptr, ${this.usz}, ptr) #0`, `declare i32 @um_read_file_range(ptr, ${this.usz}, i64, i64, ptr, ptr) #0`);
    out.push(`declare i32 @um_append_file(ptr, ${this.usz}, ptr, ${this.usz}) #0`, 'declare void @um_args(ptr, ptr) #0', `declare void @um_eprint(ptr, ${this.usz}) #0`);
    out.push(`declare i32 @um_read_file_into(ptr, ${this.usz}, i64, ptr, ${this.usz}, i64, i64, ptr, i32) #0`, `declare i32 @um_file_open(ptr, ${this.usz}, ptr, ${this.usz}, i32) #0`);
    out.push(`declare i32 @um_file_read(i32, i32, i64, ptr, ${this.usz}, i64, i64, ptr, i32) #0`, `declare i32 @um_file_write(i32, ptr, ${this.usz}, i64, i64, i32) #0`);
    out.push('declare i32 @um_file_seek(i32, i64, i32) #0', 'declare i32 @um_file_length(i32, ptr, i32) #0', 'declare void @um_file_close(i32, i32) #0');
    out.push(`declare i32 @um_list_dir(ptr, ${this.usz}, ptr, ptr) #0`, `declare i32 @um_make_dir(ptr, ${this.usz}) #0`, `declare i32 @um_is_dir(ptr, ${this.usz}) #0`);
    out.push(`declare void @um_path_query(i32, ptr, ${this.usz}, ptr, ptr) #0`);
    if (this.target === 'native') out.push('declare void @um_set_args(i32, ptr) #0');
    // ウィンドウ（runtime/window.c）。使ったときだけ宣言する。ドライバはこの宣言を見て window.c と SDL3 をリンクする。
    if (this.usesWindow) {
      out.push(`declare i32 @um_window_open(ptr, ${this.usz}, i32, i32, i32) #0`, 'declare void @um_window_close(i32, i32) #0');
      out.push(`declare i32 @um_window_poll(ptr, ${this.usz}, i32, i32) #0`, `declare void @um_window_present(i32, ptr, ${this.usz}, i32, i32, i32) #0`);
      out.push(`declare void @um_window_present_rect(i32, ptr, ${this.usz}, i32, i32, i32, i32, i32, i32, i32, i32) #0`);
      out.push(`declare i32 @um_window_set_icon(i32, ptr, ${this.usz}, i32, i32, i32) #0`);
      out.push(`declare i64 @um_window_ctl(i32, i32, ptr, ${this.usz}, i32) #0`, 'declare double @um_window_ctl_f(i32, i32, i64, i32) #0');
      out.push(`declare i64 @um_window_set_text(i32, i32, i64, ptr, ${this.usz}, i32) #0`, 'declare void @um_window_get_text(i32, i32, i64, ptr, ptr, i32) #0');
      out.push(`declare i32 @um_audio_write(i32, ptr, ${this.usz}, i64, i64, i32) #0`);
    }
    // WebAssembly（runtime/wasm.c）。使ったときだけ宣言する。ドライバはこの宣言を見て wasm.c と wasmtime をリンクする。
    if (this.usesWasm) {
      const u = this.usz;
      out.push(`declare i64 @um_wasm_compile(ptr, ${u}, i32) #0`, 'declare void @um_wasm_free_module(i64, i32) #0');
      out.push('declare void @um_wasm_exports(i64, ptr, ptr, i32) #0', 'declare i64 @um_wasm_instantiate(i64, i64, i64, i64, i32, i32) #0');
      out.push('declare void @um_wasm_free_instance(i64, i32) #0');
      out.push(`declare i32 @um_wasm_invoke(i64, ptr, ${u}, ptr, ${u}, ptr, ${u}, ptr, ${u}, i32, i32) #0`);
      out.push(`declare i32 @um_wasm_read(i64, i64, ptr, ${u}, ${u}, i64, i64, i32) #0`, `declare i32 @um_wasm_write(i64, i64, ptr, ${u}, ${u}, i64, i64, i32) #0`);
      out.push('declare i64 @um_wasm_memory_size(i64, i32) #0', `declare i32 @um_wasm_wat(ptr, ${u}, ptr, ptr) #0`, 'declare void @um_wasm_error(ptr, ptr) #0');
    }
    // 動画・音声（runtime/media.c）。使ったときだけ宣言する。ドライバはこの宣言を見て media.c と FFmpeg をリンクする。
    if (this.usesMedia) {
      const u = this.usz;
      out.push(`declare i64 @um_media_open(ptr, ${u}, i32, ptr, ${u}, i32) #0`, 'declare void @um_media_close(i64, i32) #0');
      out.push('declare void @um_media_info_numbers(i64, ptr, ptr, i32) #0', 'declare void @um_media_info_text(i64, ptr, ptr, i32) #0');
      out.push(`declare double @um_media_read_video(i64, double, ptr, ${u}, ${u}, i32, i32, i32, i32, i32, i32) #0`);
      out.push(`declare i64 @um_media_read_audio(i64, double, ptr, ${u}, ${u}, i32, i32, i32) #0`, 'declare double @um_media_last_time(i64, i32) #0');
      out.push(`declare i64 @um_media_create(ptr, ${u}, i32) #0`);
      out.push(`declare i32 @um_media_add_video(i64, ptr, ${u}, i32, i32, i32, i32, i64, ptr, ${u}, i32) #0`);
      out.push(`declare i32 @um_media_add_audio(i64, ptr, ${u}, i32, i32, i64, ptr, ${u}, i32) #0`);
      out.push(`declare i32 @um_media_write_video(i64, ptr, ${u}, ${u}, i32, i32, i32, i32, i32) #0`);
      out.push(`declare i32 @um_media_write_audio(i64, ptr, ${u}, ${u}, i32, i64, i32) #0`, 'declare i32 @um_media_finish(i64, i32) #0');
      out.push('declare void @um_media_writer_text(i64, ptr, ptr, i32) #0', 'declare void @um_media_encoders(i32, ptr, ptr) #0');
      out.push('declare void @um_media_error(ptr, ptr) #0', 'declare void @um_media_version(ptr, ptr) #0');
    }
    // wasm では LLVM が f16 を整数のビット列として変換関数に渡すので、ランタイム（wasm_rt.c）が直接その関数を持つ。
    out.push('declare zeroext i8 @um_f8_enc(double) #0', 'declare zeroext i8 @um_f4_enc(double) #0');
    if (this.usesF16()) out.push(...(this.target === 'native' ? F16_SHIMS : ['declare i16 @um_d2h(i64) #0']));
    if (this.target === 'native') out.push('declare void @um_parallel_for(ptr, ptr, i32) #0');
    for (const d of this.intrinsics) out.push(d);
    out.push('');
    out.push(...this.globals);
    if (this.globals.length > 0) out.push('');
    out.push(...this.dropFns());
    out.push(...bodies);
    out.push(...this.rangeFns);
    out.push(...this.elemDropFns);
    out.push(...this.threadFns);

    // ネイティブではCの main から呼び出す。wasmでは export 名で公開する。
    const main = this.prog.fns.get('main');
    if (this.target === 'native' && main && main.exported && main.params.length === 0) {
      // コマンドライン引数は args() のためにランタイムへ渡しておく。
      out.push('define i32 @main(i32 %argc, ptr %argv) #0 {');
      out.push('  call void @um_set_args(i32 %argc, ptr %argv)');
      if (main.ret.k === 'int' && main.ret.bits === 32) {
        out.push('  %r = call i32 @um.main()');
        out.push('  ret i32 %r');
      } else {
        out.push(`  call ${llTy(main.ret)} @um.main()`);
        out.push('  ret i32 0');
      }
      out.push('}');
      out.push('');
    }

    // 型にもとづく別名情報（TBAA）。配列の管理情報（要素の場所・長さ・容量）と、
    // 数値として読み書きする値は、決して同じ場所に重ならない。これを伝えると、
    // 要素への書き込みのたびに長さや場所を読み直さずに済み、ループの最適化が効く。
    out.push('!0 = !{!"um root"}');
    out.push('!1 = !{!"um header", !0}');
    out.push('!2 = !{!"um value", !0}');
    out.push(`${HDR} = !{!1, !1, i64 0}`);
    out.push(`${VAL} = !{!2, !2, i64 0}`);
    out.push('');
    out.push(`attributes #0 = { ${this.fnAttrs()} }`);
    // kernel の検査なし版は、必ず呼び出し側のループに埋め込む。
    out.push(`attributes #40 = { ${this.fnAttrs()} alwaysinline }`);
    // プラグインでは um_panic はランタイム（wasm_rt.c）の中にあり、トラップする（import しない）。
    out.push(this.plugin ? 'attributes #2 = { noreturn cold nounwind }' : 'attributes #2 = { noreturn cold nounwind "wasm-import-module"="env" "wasm-import-name"="um_panic" }');
    out.push(...importAttrs);
    out.push(...exportAttrs);
    return out.join('\n') + '\n';
  }
}

class FnGen {
  m: ModuleGen;
  d: FnDecl;
  lines: string[] = [];
  allocas: string[] = [];
  n = 0;
  cur = 'entry';
  terminated = false;
  vars = new Map<number, string>();
  loops: { brk: string; cont: string; depth: number }[] = [];
  /** 文の並びの中で、いま生成している文のすぐ次の文（stmts が設定し、stmt が受け取る）。 */
  nextStmt: Stmt | undefined;
  /** いま生成している dispatch 文が、GPU の完了待ちを次の dispatch 文に任せてよいか（gpuCanDefer を参照）。 */
  gpuDefer = false;
  /** 所有している変数の「生きている」フラグ。移動すると偽になる。 */
  flags = new Map<number, string>();
  /** スコープごとの、抜けるときに解放する変数。 */
  scopes: Droppable[][] = [[]];
  /** 文ごとの一時値。文の終わりで解放する。 */
  temps: Droppable[][] = [[]];
  /**
   * kernel 関数かどうか。kernel では停止できない（GPU に停止がない）ので、
   * 整数の除算・シフト・添字を、停止の代わりに定義済みの結果になるよう生成する。
   */
  kernel: boolean;
  /**
   * kernel の「検査なし版」を生成しているかどうか。添字が i ± 定数 の形の Buffer アクセスから、
   * 切り詰めを省く。dispatch が、その添字が必ず範囲内になる区間でだけ呼ぶ。
   */
  fast = false;
  /** 検査なし版で、添字として使ってよい変数（i ± 定数 しか代入されないもの）とその範囲。 */
  fastLocals = new Map<number, { min: number; max: number }>();
  /** 境界検査を省いてよい添字: 配列の変数 → その添字に使うループ変数。for ループの「検査なしの写し」を生成している間だけ入る。 */
  unchecked = new Map<number, number>();
  /** 0 より大きい間は、for ループを分けない（検査つきの写しの中で、さらに写しを増やさないため）。 */
  noSplit = 0;
  /** 値域の解析で「必ず範囲内」と分かった添字アクセス（範囲チェックを省く）。 */
  safeIndex: Set<Expr>;
  /** 値域の解析で「シフト量が必ずビット幅未満」と分かったシフト（シフト検査を省く）。 */
  safeShift: Set<Expr>;
  /**
   * グループをまとめて実行する関数（generateGroup）を生成している間の状態。
   *   lid: いまのスレッドのグループ内の番号（i32 の値）。スレッドを回すループの外では '0'
   *   next: いまのスレッドの残りを飛ばす先（return）
   *   perThread: スレッドごとに置き場所を持つ変数 → 1スレッドぶんの塊の中の番号（barrier を使う kernel だけ）
   */
  grp: { lid: string; next: string; perThread: Map<number, number> } | null = null;
  /** スレッドごとの変数を並べた1スレッドぶんの塊の型（LLVM）。barrier を使う kernel だけ。 */
  ptTy = '';
  /** simdgroup の集計の let → 引数を入れるスレッドごとの置き場所（塊の中の番号）。 */
  collSlots = new Map<Stmt, number[]>();

  constructor(m: ModuleGen, d: FnDecl) {
    this.m = m;
    this.d = d;
    this.kernel = !!d.isKernel;
    // kernel は停止しない（添字は切り詰め、シフトはマスク）ので解析は要らない。
    const b = this.kernel ? { safeIndex: new Set<Expr>(), safeShift: new Set<Expr>() } : analyzeBounds(d);
    this.safeIndex = b.safeIndex;
    this.safeShift = b.safeShift;
  }

  emit(s: string): void {
    this.lines.push('  ' + s);
  }

  tmp(): string {
    return `%t${++this.n}`;
  }

  newLabel(hint: string): string {
    return `${hint}${++this.n}`;
  }

  label(name: string): void {
    this.lines.push(`${name}:`);
    this.cur = name;
    this.terminated = false;
  }

  /**
   * 境界検査をループの外へ出した for ループ。本体を3つ生成する:
   *   head: 検査つき。添字が範囲内に入るまで。
   *   mid:  検査なし。ループ変数が [lo, hi) にある間。添字 i + c がすべての対象の配列で範囲内になる区間。
   *   tail: 検査つき。mid を出たあと。
   * 各反復で、条件式は1回、更新も1回だけ実行する。どの写しでも意味は同じで、範囲外の添字は
   * 検査つきの写しで実行されるので、必ず検出される。
   */
  forSplit(s: Stmt & { kind: 'for' }, plan: SplitPlan): void {
    const smin = (a: string, b: string) => {
      this.m.intrinsics.add('declare i64 @llvm.smin.i64(i64, i64)');
      const v = this.tmp();
      this.emit(`${v} = call i64 @llvm.smin.i64(i64 ${a}, i64 ${b})`);
      return v;
    };
    const lt = plan.loopVar.ty as Ty & { k: 'int' };

    // 区間 [lo, hi) を求める。配列の長さはループの中で変わらないので、ここで1回だけ読む。
    let lo = 0;
    let maxOff = 0;
    let hi = '4611686018427387904'; // 2^62。u64 のループ変数を符号つきで比べても正しくなる上限
    for (const a of plan.arrays.values()) {
      lo = Math.max(lo, -a.min);
      maxOff = Math.max(maxOff, a.max);
      const hdr = this.addr(a.base);
      const lp = this.tmp();
      const len = this.tmp();
      this.emit(`${lp} = getelementptr inbounds %um.Array, ptr ${hdr}, i32 0, i32 1`);
      this.emit(`${len} = load ${this.m.usz}, ptr ${lp}, !tbaa ${HDR}`);
      let len64 = len;
      if (this.m.usz !== 'i64') {
        len64 = this.tmp();
        this.emit(`${len64} = zext ${this.m.usz} ${len} to i64`);
      }
      const lim = this.tmp();
      this.emit(`${lim} = sub i64 ${len64}, ${a.max}`);
      hi = smin(hi, lim);
    }
    // 符号つきのループ変数では、i + c がその型の中で折り返さない範囲に限る。
    if (lt.signed && lt.bits === 32) hi = smin(hi, String(2147483648 - maxOff));

    const endL = this.newLabel('for.end');
    const inRange = (yes: string, no: string) => {
      const sym = plan.loopVar;
      let x = this.expr({ kind: 'ident', name: sym.name, sym, ty: sym.ty, pos: s.pos });
      if (lt.bits < 64) {
        const w = this.tmp();
        this.emit(`${w} = ${lt.signed ? 'sext' : 'zext'} i${lt.bits} ${x} to i64`);
        x = w;
      }
      const a = this.tmp();
      const b = this.tmp();
      const c = this.tmp();
      this.emit(`${a} = icmp sge i64 ${x}, ${lo}`);
      this.emit(`${b} = icmp slt i64 ${x}, ${hi}`);
      this.emit(`${c} = and i1 ${a}, ${b}`);
      this.emit(`br i1 ${c}, label %${yes}, label %${no}`);
      this.terminated = true;
    };
    const cond = (yes: string) => {
      if (s.cond) {
        const c = this.withTemps(() => this.expr(s.cond!));
        this.emit(`br i1 ${c}, label %${yes}, label %${endL}`);
        this.terminated = true;
      } else {
        this.br(yes);
      }
    };
    const body = (bodyL: string, stepL: string, checked: boolean) => {
      this.label(bodyL);
      this.loops.push({ brk: endL, cont: stepL, depth: this.scopes.length });
      const saved = this.unchecked;
      if (checked) {
        this.noSplit++;
      } else {
        this.unchecked = new Map(saved);
        for (const id of plan.arrays.keys()) this.unchecked.set(id, plan.loopVar.id);
      }
      this.scoped(s.body);
      if (checked) this.noSplit--;
      this.unchecked = saved;
      this.loops.pop();
      this.br(stepL);
      this.label(stepL);
      if (s.step) this.stmt(s.step);
    };

    const L = (n: string) => this.newLabel(`for.${n}`);
    const [headCond, headTest, headBody, headStep] = [L('head.cond'), L('head.test'), L('head.body'), L('head.step')];
    const [midCond, midTest, midBody, midStep] = [L('mid.cond'), L('mid.test'), L('mid.body'), L('mid.step')];
    const [tailCond, tailBody, tailStep] = [L('tail.cond'), L('tail.body'), L('tail.step')];

    this.br(headCond);
    this.label(headCond);
    cond(headTest);
    this.label(headTest);
    inRange(midBody, headBody);
    body(headBody, headStep, true);
    this.br(headCond);

    body(midBody, midStep, false);
    this.br(midCond);
    this.label(midCond);
    cond(midTest);
    this.label(midTest);
    inRange(midBody, tailBody);

    body(tailBody, tailStep, true);
    this.br(tailCond);
    this.label(tailCond);
    cond(tailBody);

    this.label(endL);
  }

  br(target: string): void {
    if (!this.terminated) this.emit(`br label %${target}`);
    this.terminated = true;
  }

  alloca(t: Ty): string {
    const name = `%a${++this.n}`;
    this.allocas.push(`  ${name} = alloca ${llTy(t)}`);
    return name;
  }

  // ---- 所有権: 解放 ----

  /** 関数の入口で偽に初期化されるフラグを作る。 */
  flagAlloca(): string {
    const f = `%f${++this.n}`;
    this.allocas.push(`  ${f} = alloca i1`, `  store i1 false, ptr ${f}`);
    return f;
  }

  dropAt(ptr: string, t: Ty): void {
    if (t.k === 'heap') {
      if (!this.m.isCopy(t.elem)) this.emit(`call void ${this.m.dropElems(t.elem)}(ptr ${ptr})`);
      const d = this.tmp();
      this.emit(`${d} = load ptr, ptr ${ptr}, !tbaa ${HDR}`);
      this.emit(`call void @um_free(ptr ${d})`);
    } else if (t.k === 'struct') {
      this.emit(`call void @um.drop.${t.name}(ptr ${ptr})`);
    }
  }

  /** フラグが真なら解放して、偽に戻す。 */
  condDrop(d: Droppable): void {
    const live = this.tmp();
    const dropL = this.newLabel('drop');
    const nextL = this.newLabel('kept');
    this.emit(`${live} = load i1, ptr ${d.flag}`);
    this.emit(`br i1 ${live}, label %${dropL}, label %${nextL}`);
    this.label(dropL);
    // スレッドの持ち手: 合流していなければ、ここで合流して戻り値を捨てる（自動の合流）。
    if (d.thread) this.joinThreads(d.ptr, d.thread, false);
    else this.dropAt(d.ptr, d.ty);
    this.emit(`store i1 false, ptr ${d.flag}`);
    this.br(nextL);
    this.label(nextL);
  }

  /** depth より内側のスコープの変数を、宣言と逆の順に解放する。 */
  dropScopes(depth: number): void {
    for (let i = this.scopes.length - 1; i >= depth; i--) {
      for (const d of [...this.scopes[i]].reverse()) this.condDrop(d);
    }
  }

  /** 所有する変数を登録する。コピーできる型は解放が要らないので何もしない。 */
  own(symId: number, ptr: string, t: Ty): void {
    if (this.m.isCopy(t)) return;
    const flag = this.flagAlloca();
    this.emit(`store i1 true, ptr ${flag}`);
    this.flags.set(symId, flag);
    this.scopes[this.scopes.length - 1].push({ ptr, flag, ty: t });
  }

  /** 値を一時領域に置く。ヒープを持つ値なら、文の終わりで解放されるように登録する。 */
  spill(v: string, t: Ty): string {
    const a = this.alloca(t);
    this.emit(`store ${llTy(t)} ${v}, ptr ${a}`);
    if (!this.m.isCopy(t)) {
      const flag = this.flagAlloca();
      this.emit(`store i1 true, ptr ${flag}`);
      this.temps[this.temps.length - 1].push({ ptr: a, flag, ty: t });
    }
    return a;
  }

  /** fn の中で作られた一時値を、fn の終わりで解放する。 */
  withTemps<T>(fn: () => T): T {
    this.temps.push([]);
    const r = fn();
    const list = this.temps.pop()!;
    if (!this.terminated) for (const d of list.reverse()) this.condDrop(d);
    return r;
  }

  /** 新しいスコープで文を実行し、抜けるときにそのスコープの変数を解放する。 */
  scoped(s: Stmt): void {
    this.scopes.push([]);
    this.stmt(s);
    const list = this.scopes.pop()!;
    if (!this.terminated) for (const d of list.reverse()) this.condDrop(d);
  }

  /** 値を取り出して消費する。変数から取り出した場合、その変数は移動済みになる。 */
  consume(e: Expr): string {
    const v = this.expr(e);
    if (e.kind === 'ident' && !this.m.isCopy(e.ty!)) {
      const flag = this.flags.get(e.sym!.id);
      if (flag) this.emit(`store i1 false, ptr ${flag}`);
    }
    return v;
  }

  generate(attr: string, suffix = ''): string {
    const d = this.d;
    const params = d.params.map((p, i) => `${this.m.paramSig(p.mode, p.ty!)} %p${i}`);
    const linkage = d.exported ? '' : 'internal ';
    const selfTy: Ty | null = d.cls ? { k: 'struct', name: d.cls } : null;
    let thisPtr = '';

    if (selfTy && d.isCtor) {
      // コンストラクタは値を組み立てて返す。まずフィールドの初期値を入れる。
      thisPtr = this.alloca(selfTy);
      this.vars.set(d.thisSym!.id, thisPtr);
      this.m.prog.structs.get(d.cls!)!.fields.forEach((f, i) => {
        if (!f.init) return;
        this.withTemps(() => {
          const v = this.expr(f.init!);
          const p = this.tmp();
          this.emit(`${p} = getelementptr inbounds ${llTy(selfTy)}, ptr ${thisPtr}, i32 0, i32 ${i}`);
          this.emit(`store ${llTy(f.ty)} ${v}, ptr ${p}`);
        });
      });
    } else if (selfTy) {
      // メソッドは受け手をポインタで受け取る。書き換えないメソッドは読み取り専用にできる。
      params.unshift(d.mutatesThis ? 'ptr noalias %this' : 'ptr noalias readonly %this');
      this.vars.set(d.thisSym!.id, '%this');
    }

    d.params.forEach((p, i) => {
      const ty = p.ty!;
      if (passByPtr(p.mode, ty) && p.mode !== 'own') {
        // 借用: 呼び出し側の置き場所をそのまま使う。
        this.vars.set(p.sym!.id, `%p${i}`);
      } else {
        const a = this.alloca(ty);
        if (passByPtr(p.mode, ty)) {
          const v = this.tmp();
          this.emit(`${v} = load ${llTy(ty)}, ptr %p${i}`);
          this.emit(`store ${llTy(ty)} ${v}, ptr ${a}`);
        } else {
          this.emit(`store ${llTy(ty)} %p${i}, ptr ${a}`);
        }
        this.vars.set(p.sym!.id, a);
        // own で受け取った値は、この関数が解放する責任を持つ。
        if (p.mode === 'own') this.own(p.sym!.id, a, ty);
      }
    });

    this.scopes.push([]);
    this.stmts(d.body);
    const retTy = selfTy && d.isCtor ? selfTy : d.ret!;
    if (!this.terminated) {
      this.dropScopes(0);
      if (selfTy && d.isCtor) {
        const v = this.tmp();
        this.emit(`${v} = load ${llTy(selfTy)}, ptr ${thisPtr}`);
        this.emit(`ret ${llTy(selfTy)} ${v}`);
      } else {
        // 型検査で全経路の return を確認済みなので、void 以外でここに来ることはない。
        this.emit(retTy.k === 'void' ? 'ret void' : 'unreachable');
      }
    }

    return [
      `define ${linkage}${llTy(retTy)} ${this.m.fnName(this.m.declKey(d))}${suffix}(${params.join(', ')}) ${attr} {`,
      'entry:',
      ...this.allocas,
      ...this.lines,
      '}',
    ].join('\n');
  }

  /**
   * group.* を使う kernel を、グループ1つをまとめて CPU で実行する関数にする（dispatchGroups の CPU の経路）。
   *   define void @um.K.group(i32 %gid, i32 %gsize, i32 %gcount, ptr %gshared, ptr %gscratch, 引数...)
   * 本体を barrier で区切り、区切りの間の文を「グループの全スレッドについて順に」実行する（スレッドを回すループ）。
   * barrier を含む文（if / while / for / { }）はグループで1回だけ実行し、条件はスレッド 0 の値で決める
   * （型検査の checkGroupBody が、その条件が全スレッドで同じ値になることを確かめている）。
   * barrier をまたいで生きうる変数は、スレッドごとの置き場所（%gscratch。1スレッドぶんの塊が gsize 個並ぶ）に置く。
   * barrier を使わない kernel は、本体全体を1つの区切りとして、スレッドを順に最後まで実行する。
   */
  generateGroup(): string {
    const d = this.d;
    const params = d.params.slice(1).map((p, i) => `${this.m.paramSig(p.mode, p.ty!)} %p${i + 1}`);
    params.unshift('i32 %gid', 'i32 %gsize', 'i32 %gcount', 'ptr %gshared', 'ptr %gscratch');

    // 共有配列は %gshared の中の決まった位置。
    for (const [id, off] of sharedLayout(d).offsets) {
      const p = `%sh${id}`;
      this.allocas.push(`  ${p} = getelementptr inbounds i8, ptr %gshared, i32 ${off}`);
      this.vars.set(id, p);
    }

    const perThread = new Map<number, number>();
    const ptTys: string[] = [];
    // 値の引数（添字を含む）は、スレッドごとに書き換えられうるので、スレッドごとの値として持つ。
    const valueParams = d.params.filter((p) => !(passByPtr(p.mode, p.ty!) && p.mode !== 'own'));
    if (d.usesBarrier) {
      const syms: Sym[] = valueParams.map((p) => p.sym!);
      groupLevelSyms(d.body, syms);
      for (const sym of syms) {
        perThread.set(sym.id, ptTys.length);
        ptTys.push(llTy(sym.ty));
      }
      // simdgroup の集計の引数も、スレッドごとの置き場所に入れてから集める（groupStmt の let）。
      const coll = (list: Stmt[]): void => {
        for (const x of list) {
          if (isCollectiveLet(x)) {
            const init = (x as Stmt & { kind: 'let' }).init as Expr & { kind: 'math' };
            this.collSlots.set(x, init.args.map((a) => {
              ptTys.push(llTy(a.ty!));
              return ptTys.length - 1;
            }));
            continue;
          }
          if (!hasBarrier(x)) continue;
          if (x.kind === 'block') coll(x.body);
          else if (x.kind === 'if') coll([x.then, ...(x.else ? [x.else] : [])]);
          else if (x.kind === 'while') coll([x.body]);
          else if (x.kind === 'for') coll([...(x.init ? [x.init] : []), x.body]);
        }
      };
      coll(d.body);
      this.ptTy = `{ ${ptTys.join(', ')} }`;
    }
    this.grp = { lid: '0', next: '', perThread };

    // 借用で受け取る Buffer などは、呼び出し側の置き場所をそのまま使う。値の引数は、スレッドの始めに入れる。
    d.params.forEach((p, i) => {
      if (i > 0 && passByPtr(p.mode, p.ty!) && p.mode !== 'own') this.vars.set(p.sym!.id, `%p${i}`);
    });
    const scalarSlots = new Map<number, string>();
    if (!d.usesBarrier) for (const p of valueParams) scalarSlots.set(p.sym!.id, this.alloca(p.ty!));
    const initParams = () => {
      d.params.forEach((p, i) => {
        if (!valueParams.includes(p)) return;
        const ty = p.ty!;
        const slot = this.vars.get(p.sym!.id)!;
        let v: string;
        if (i === 0) {
          // 添字 = グループの番号 × 大きさ + グループ内の番号（dispatch の側で u32 に収まることを確かめてある）。
          const base = this.tmp();
          v = this.tmp();
          this.emit(`${base} = mul i32 %gid, %gsize`);
          this.emit(`${v} = add i32 ${base}, ${this.grp!.lid}`);
        } else if (passByPtr(p.mode, ty)) {
          v = this.tmp();
          this.emit(`${v} = load ${llTy(ty)}, ptr %p${i}`);
        } else {
          v = `%p${i}`;
        }
        this.emit(`store ${llTy(ty)} ${v}, ptr ${slot}`);
      });
    };
    for (const [id, slot] of scalarSlots) this.vars.set(id, slot);
    this.atThread0();

    this.scopes.push([]);
    if (d.usesBarrier) {
      this.segment(initParams);
      this.groupStmts(d.body);
    } else {
      this.segment(() => {
        initParams();
        this.stmts(d.body);
      });
    }
    this.scopes.pop();
    this.emit('ret void');
    this.grp = null;

    return [
      `define internal void ${this.m.fnName(this.m.declKey(d))}.group(${params.join(', ')}) #0 {`,
      'entry:',
      ...this.allocas,
      ...this.lines,
      '}',
    ].join('\n');
  }

  /** スレッドごとの変数を、スレッド 0 の置き場所に向ける（グループで1回だけ評価する条件のため）。 */
  atThread0(): void {
    const g = this.grp!;
    g.lid = '0';
    for (const [id, k] of g.perThread) this.vars.set(id, this.perThreadPtr(k, '0', true));
  }

  /** スレッドごとの置き場所。entry: 入口に置く（どこからでも使える）。 */
  perThreadPtr(k: number, lid: string, entry = false): string {
    const p = entry ? `%pt${k}` : this.tmp();
    const line = `${p} = getelementptr inbounds ${this.ptTy}, ptr %gscratch, i32 ${lid}, i32 ${k}`;
    if (entry) {
      if (!this.allocas.includes(`  ${line}`)) this.allocas.push(`  ${line}`);
    } else {
      this.emit(line);
    }
    return p;
  }

  /** グループの全スレッドについて、body を順に実行するループ（スレッドを回すループ）。 */
  segment(body: () => void): void {
    const g = this.grp!;
    const slot = this.alloca({ k: 'int', bits: 32, signed: false });
    this.emit(`store i32 0, ptr ${slot}`);
    const condL = this.newLabel('thr.cond');
    const bodyL = this.newLabel('thr.body');
    const nextL = this.newLabel('thr.next');
    const endL = this.newLabel('thr.end');
    this.br(condL);
    this.label(condL);
    const lid = this.tmp();
    const more = this.tmp();
    this.emit(`${lid} = load i32, ptr ${slot}`);
    this.emit(`${more} = icmp ult i32 ${lid}, %gsize`);
    this.emit(`br i1 ${more}, label %${bodyL}, label %${endL}`);
    this.label(bodyL);
    g.lid = lid;
    g.next = nextL;
    for (const [id, k] of g.perThread) this.vars.set(id, this.perThreadPtr(k, lid));
    body();
    this.br(nextL);
    this.label(nextL);
    const lid2 = this.tmp();
    const inc = this.tmp();
    this.emit(`${lid2} = load i32, ptr ${slot}`);
    this.emit(`${inc} = add i32 ${lid2}, 1`);
    this.emit(`store i32 ${inc}, ptr ${slot}`);
    this.br(condL);
    this.label(endL);
    g.next = '';
    // ループの中で求めた置き場所は、ループの外では使えない。
    this.atThread0();
  }

  /** グループで実行する文の並び。barrier を含まない文が続く所は、まとめて1つのスレッドを回すループにする。 */
  groupStmts(list: Stmt[]): void {
    let run: Stmt[] = [];
    const flush = () => {
      if (run.length === 0) return;
      const r = run;
      run = [];
      this.segment(() => this.stmts(r));
    };
    for (const s of list) {
      if (!hasBarrier(s)) {
        run.push(s);
        continue;
      }
      flush();
      this.groupStmt(s);
    }
    flush();
  }

  /** barrier を含む文。グループで1回だけ実行する（条件はスレッド 0 の値）。 */
  groupStmt(s: Stmt): void {
    if (this.terminated) this.label(this.newLabel('dead'));
    const cond = (e: Expr) => {
      this.atThread0();
      return this.withTemps(() => this.expr(e));
    };
    switch (s.kind) {
      case 'expr':
        // group.barrier(): 前の区切りの全スレッドが終わってから次の区切りを始めるので、何もしなくてよい。
        return;
      case 'let':
        this.collective(s);
        return;
      case 'block':
        this.groupStmts(s.body);
        return;
      case 'if': {
        const c = cond(s.cond);
        const thenL = this.newLabel('gthen');
        const elseL = this.newLabel('gelse');
        const endL = this.newLabel('gendif');
        this.emit(`br i1 ${c}, label %${thenL}, label %${elseL}`);
        this.terminated = true;
        this.label(thenL);
        this.groupStmts(stmtList(s.then));
        this.br(endL);
        this.label(elseL);
        if (s.else) this.groupStmts(stmtList(s.else));
        this.br(endL);
        this.label(endL);
        return;
      }
      case 'while':
      case 'for': {
        if (s.kind === 'for' && s.init) {
          const init = s.init;
          this.segment(() => this.stmt(init));
        }
        const condL = this.newLabel('gloop.cond');
        const bodyL = this.newLabel('gloop.body');
        const endL = this.newLabel('gloop.end');
        this.br(condL);
        this.label(condL);
        const ce = s.cond;
        if (ce) {
          const c = cond(ce);
          this.emit(`br i1 ${c}, label %${bodyL}, label %${endL}`);
          this.terminated = true;
        } else {
          this.br(bodyL);
        }
        this.label(bodyL);
        this.groupStmts(stmtList(s.body));
        if (s.kind === 'for' && s.step) {
          const step = s.step;
          this.segment(() => this.stmt(step));
        }
        this.br(condL);
        this.label(endL);
        this.atThread0();
        return;
      }
      default:
        throw new Error(`internal: group statement ${s.kind}`);
    }
  }

  /**
   * simdgroup の集計 `const v = group.simdSum(x)` など。simdgroup はグループの中の連続した 32 スレッド
   * （最後のものは、グループの大きさが 32 で割り切れなければ短い）。Metal の simd_sum などと同じ。
   * 区切り 1: 全スレッドの引数をスレッドごとの置き場所に入れる。区切り 2: 各スレッドが、自分の simdgroup の値を集める。
   * 和は lane の順に足す（Metal の f32 の和は順番が決まっていないので、丸めが違うことがある）。
   */
  collective(s: Stmt & { kind: 'let' }): void {
    const init = s.init as Expr & { kind: 'math' };
    const slots = this.collSlots.get(s)!;
    const g = this.grp!;
    this.segment(() => this.withTemps(() => {
      init.args.forEach((a, k) => {
        const v = this.expr(a);
        this.emit(`store ${llTy(a.ty!)} ${v}, ptr ${this.perThreadPtr(slots[k], g.lid)}`);
      });
    }));
    const ty = s.sym!.ty as Ty & ({ k: 'int' } | { k: 'float' });
    const lt = llTy(ty);
    this.segment(() => {
      const lid = g.lid;
      const base = this.tmp();
      const rest = this.tmp();
      const short = this.tmp();
      const cnt = this.tmp();
      this.emit(`${base} = and i32 ${lid}, -32`);
      this.emit(`${rest} = sub i32 %gsize, ${base}`);
      this.emit(`${short} = icmp ult i32 ${rest}, 32`);
      this.emit(`${cnt} = select i1 ${short}, i32 ${rest}, i32 32`);
      let result: string;
      if (init.fn === 'group.simdShuffle') {
        // lane が simdgroup の外なら最後のスレッドに切り詰める。
        const lane = this.tmp();
        const lim = this.tmp();
        const inR = this.tmp();
        const l = this.tmp();
        const j = this.tmp();
        this.emit(`${lane} = load i32, ptr ${this.perThreadPtr(slots[1], lid)}`);
        this.emit(`${lim} = sub i32 ${cnt}, 1`);
        this.emit(`${inR} = icmp ult i32 ${lane}, ${cnt}`);
        this.emit(`${l} = select i1 ${inR}, i32 ${lane}, i32 ${lim}`);
        this.emit(`${j} = add i32 ${base}, ${l}`);
        result = this.tmp();
        this.emit(`${result} = load ${lt}, ptr ${this.perThreadPtr(slots[0], j)}`);
      } else {
        const acc = this.alloca(ty);
        const jslot = this.alloca({ k: 'int', bits: 32, signed: false });
        const first = this.tmp();
        const end = this.tmp();
        const start = this.tmp();
        this.emit(`${first} = load ${lt}, ptr ${this.perThreadPtr(slots[0], base)}`);
        this.emit(`store ${lt} ${first}, ptr ${acc}`);
        this.emit(`${end} = add i32 ${base}, ${cnt}`);
        this.emit(`${start} = add i32 ${base}, 1`);
        this.emit(`store i32 ${start}, ptr ${jslot}`);
        const condL = this.newLabel('simd.cond');
        const bodyL = this.newLabel('simd.body');
        const endL = this.newLabel('simd.end');
        this.br(condL);
        this.label(condL);
        const j = this.tmp();
        const more = this.tmp();
        this.emit(`${j} = load i32, ptr ${jslot}`);
        this.emit(`${more} = icmp ult i32 ${j}, ${end}`);
        this.emit(`br i1 ${more}, label %${bodyL}, label %${endL}`);
        this.terminated = true;
        this.label(bodyL);
        const x = this.tmp();
        const a = this.tmp();
        const c = this.tmp();
        this.emit(`${x} = load ${lt}, ptr ${this.perThreadPtr(slots[0], j)}`);
        this.emit(`${a} = load ${lt}, ptr ${acc}`);
        const isF = ty.k === 'float';
        const sign = ty.k === 'int' && ty.signed ? 's' : 'u';
        if (init.fn === 'group.simdSum') {
          this.emit(`${c} = ${isF ? 'fadd' : 'add'} ${lt} ${a}, ${x}`);
        } else {
          const op = init.fn === 'group.simdMin' ? 'min' : 'max';
          const name = isF ? `@llvm.${op}num.f32` : `@llvm.${sign}${op}.i32`;
          this.m.intrinsics.add(`declare ${lt} ${name}(${lt}, ${lt})`);
          this.emit(`${c} = call ${lt} ${name}(${lt} ${a}, ${lt} ${x})`);
        }
        this.emit(`store ${lt} ${c}, ptr ${acc}`);
        const j2 = this.tmp();
        this.emit(`${j2} = add i32 ${j}, 1`);
        this.emit(`store i32 ${j2}, ptr ${jslot}`);
        this.br(condL);
        this.label(endL);
        result = this.tmp();
        this.emit(`${result} = load ${lt}, ptr ${acc}`);
      }
      this.emit(`store ${lt} ${result}, ptr ${this.vars.get(s.sym!.id)!}`);
    });
  }

  /** group.local() など（generateGroup の中だけ）。 */
  groupMath(e: Expr & { kind: 'math' }): string {
    const g = this.grp;
    if (!g) throw new Error(`internal: ${e.fn} outside a group function`);
    switch (e.fn) {
      case 'group.simdLane': {
        const v = this.tmp();
        this.emit(`${v} = and i32 ${g.lid}, 31`);
        return v;
      }
      case 'group.local':
        return g.lid;
      case 'group.id':
        return '%gid';
      case 'group.size':
        return '%gsize';
      case 'group.count':
        return '%gcount';
      case 'group.barrier':
        return '';
    }
    throw new Error(`internal: ${e.fn}`);
  }

  // ---- 文 ----

  /** 文の並びを順に生成する。各文に「すぐ次の文」を知らせる（続けて書いた dispatch を1回の提出にまとめるため）。 */
  stmts(list: Stmt[]): void {
    list.forEach((s, i) => {
      this.nextStmt = list[i + 1];
      this.stmt(s);
    });
    this.nextStmt = undefined;
  }

  /**
   * dispatch 文 s を GPU に積んだあと、完了を待たずに次の文へ進んでよいか（Metal のみ）。
   * 次の文も GPU に渡しうる dispatch 文で、その間に Buffer の中身を読む式も、一時的な値の解放も
   * 挟まらないときだけ許す。待たずに進むと、ランタイムが次の dispatch と合わせて1回で GPU に提出する。
   * 次の dispatch が CPU で実行されることになった場合は、ランタイムが先に完了を待つ（um_gpu_pick）。
   */
  gpuCanDefer(s: Expr, next: Stmt | undefined): boolean {
    if ((this.m.gpu !== 'metal' && this.m.gpu !== 'wgpu') || !next || next.kind !== 'expr') return false;
    const simple = (e: Expr): boolean =>
      e.kind === 'int' || e.kind === 'float' || e.kind === 'bool' || (e.kind === 'ident' && !e.sym?.alias);
    const plain = (e: Expr): boolean =>
      e.kind === 'dispatch' && e.device !== 'cpu' && !e.async && simple(e.count) && (!e.group || simple(e.group)) &&
      e.args.every((a) => simple(a.expr));
    return plain(s) && plain(next.expr);
  }

  stmt(s: Stmt): void {
    const next = this.nextStmt;
    this.nextStmt = undefined;
    // return や break の後ろに書かれた文も、IRとして正しい形になるよう新しいブロックに置く。
    if (this.terminated) this.label(this.newLabel('dead'));

    switch (s.kind) {
      case 'block': {
        this.scopes.push([]);
        this.stmts(s.body);
        const list = this.scopes.pop()!;
        if (!this.terminated) for (const d of list.reverse()) this.condDrop(d);
        return;
      }

      case 'let':
        if (s.init?.kind === 'thread' && (s.init.op === 'spawn' || s.init.op === 'each')) {
          this.withTemps(() => this.spawn(s));
          return;
        }
        // 共有配列はグループの始めに 0 で埋めてある（groupRangeFn）。
        if (s.sym?.shared) return;
        this.withTemps(() => {
          const ty = s.sym!.ty;
          // 初期値のない宣言はゼロで初期化する。
          const v = s.init ? this.consume(s.init) : 'zeroinitializer';
          // barrier をまたぐ変数は、スレッドごとの置き場所（segment が向けてある）に入れる。
          const a = this.grp?.perThread.has(s.sym!.id) ? this.vars.get(s.sym!.id)! : this.alloca(ty);
          this.emit(`store ${llTy(ty)} ${v}, ptr ${a}${isScalar(ty) ? `, !tbaa ${VAL}` : ''}`);
          this.vars.set(s.sym!.id, a);
          this.own(s.sym!.id, a, ty);
        });
        return;

      case 'assign':
        this.withTemps(() => {
          const ty = s.target.ty!;
          // 右辺を先に評価する。右辺が配列を伸ばしても、代入先のアドレスはその後で求めるので安全。
          const v = this.consume(s.value);
          const p = this.addr(s.target);
          const flag = s.target.kind === 'ident' ? this.flags.get(s.target.sym!.id) : undefined;
          if (!this.m.isCopy(ty) && !s.initField) {
            // 古い値を解放してから上書きする。移動済みの変数（フラグが偽）には古い値がない。
            if (flag) this.condDrop({ ptr: p, flag, ty });
            else this.dropAt(p, ty);
          }
          if (isRacyElem(s.target)) {
            // RacyBuffer の要素: 他のスレッドと同時に読み書きされうるので、順序保証なしのアトミックな書き込みにする。
            // kernel の外では monotonic にする。unordered の読み込みは LLVM がループの外へ出してよいので、
            // 「別のスレッドが合図を立てるまで回る」ループが永久に回りうる（arm64 / x86-64 ではどちらも普通の命令）。
            this.emit(`store atomic ${llTy(ty)} ${v}, ptr ${p} ${this.kernel ? 'unordered' : 'monotonic'}, align ${racyAlign(ty)}`);
          } else {
            this.emit(`store ${llTy(ty)} ${v}, ptr ${p}${isScalar(ty) ? `, !tbaa ${VAL}` : ''}`);
          }
          if (flag) this.emit(`store i1 true, ptr ${flag}`);
        });
        return;

      case 'if': {
        const c = this.withTemps(() => this.expr(s.cond));
        const thenL = this.newLabel('then');
        const elseL = s.else ? this.newLabel('else') : '';
        const endL = this.newLabel('endif');
        this.emit(`br i1 ${c}, label %${thenL}, label %${s.else ? elseL : endL}`);
        this.label(thenL);
        this.scoped(s.then);
        this.br(endL);
        if (s.else) {
          this.label(elseL);
          this.scoped(s.else);
          this.br(endL);
        }
        this.label(endL);
        return;
      }

      case 'while': {
        const condL = this.newLabel('while.cond');
        const bodyL = this.newLabel('while.body');
        const endL = this.newLabel('while.end');
        this.br(condL);
        this.label(condL);
        const c = this.withTemps(() => this.expr(s.cond));
        this.emit(`br i1 ${c}, label %${bodyL}, label %${endL}`);
        this.label(bodyL);
        this.loops.push({ brk: endL, cont: condL, depth: this.scopes.length });
        this.scoped(s.body);
        this.loops.pop();
        this.br(condL);
        this.label(endL);
        return;
      }

      case 'for': {
        // 初期化で宣言した変数は、ループ全体を包むスコープに属する。
        this.scopes.push([]);
        if (s.init) this.stmt(s.init);
        const plan = !this.kernel && this.noSplit === 0 ? planSplit(s) : null;
        if (plan) {
          this.forSplit(s, plan);
          for (const d of this.scopes.pop()!.reverse()) this.condDrop(d);
          return;
        }
        const condL = this.newLabel('for.cond');
        const bodyL = this.newLabel('for.body');
        const stepL = this.newLabel('for.step');
        const endL = this.newLabel('for.end');
        this.br(condL);
        this.label(condL);
        if (s.cond) {
          const c = this.withTemps(() => this.expr(s.cond!));
          this.emit(`br i1 ${c}, label %${bodyL}, label %${endL}`);
          this.terminated = true;
        } else {
          this.br(bodyL);
        }
        this.label(bodyL);
        this.loops.push({ brk: endL, cont: stepL, depth: this.scopes.length });
        this.scoped(s.body);
        this.loops.pop();
        this.br(stepL);
        this.label(stepL);
        if (s.step) this.stmt(s.step);
        this.br(condL);
        this.label(endL);
        for (const d of this.scopes.pop()!.reverse()) this.condDrop(d);
        return;
      }

      case 'forOf':
        this.withTemps(() => this.forOf(s));
        return;

      case 'return': {
        if (this.grp) {
          // グループの関数では、いまのスレッドの残りを飛ばす（barrier を使う kernel では return は型検査が禁じている）。
          this.br(this.grp.next);
          return;
        }
        // 戻り値を取り出してから、一時値と、すべてのスコープの変数を解放する。
        const v = s.value ? this.consume(s.value) : '';
        for (const list of [...this.temps].reverse()) {
          for (const d of [...list].reverse()) this.condDrop(d);
        }
        this.dropScopes(0);
        this.emit(s.value ? `ret ${llTy(s.value.ty!)} ${v}` : 'ret void');
        this.terminated = true;
        return;
      }

      case 'break': {
        const loop = this.loops[this.loops.length - 1];
        this.dropScopes(loop.depth);
        this.br(loop.brk);
        return;
      }

      case 'continue': {
        const loop = this.loops[this.loops.length - 1];
        this.dropScopes(loop.depth);
        this.br(loop.cont);
        return;
      }

      case 'expr':
        this.withTemps(() => {
          const e = s.expr;
          const ty = e.ty!;
          if (ty.k === 'void' || this.m.isCopy(ty)) {
            this.gpuDefer = this.gpuCanDefer(e, next);
            this.expr(e);
            this.gpuDefer = false;
          } else if (isOwnedTemp(e)) {
            // 使われなかった、ヒープを持つ結果。一時値として解放する。
            this.spill(this.expr(e), ty);
          }
        });
        return;
    }
  }

  forOf(s: Stmt & { kind: 'forOf' }): void {
    // 反復中の配列は型検査が書き換えを禁じているので、base は反復の間ずっと有効。
    // 添字は 0 から長さ未満しか取らないので、境界検査は要らない。
    const at = s.iter.ty as Ty & ({ k: 'array' } | { k: 'heap' });
    const usz = this.m.usz;
    const base = this.addr(s.iter);
    const idx = this.alloca({ k: 'int', bits: this.m.prog.ptrBits, signed: false });
    const slot = this.alloca(at.elem);
    this.vars.set(s.sym!.id, slot);
    this.emit(`store ${usz} 0, ptr ${idx}`);
    const condL = this.newLabel('forof.cond');
    const bodyL = this.newLabel('forof.body');
    const stepL = this.newLabel('forof.step');
    const endL = this.newLabel('forof.end');
    this.br(condL);
    this.label(condL);
    const i = this.tmp();
    this.emit(`${i} = load ${usz}, ptr ${idx}`);
    let len: string;
    if (at.k === 'heap') {
      const lp = this.tmp();
      len = this.tmp();
      this.emit(`${lp} = getelementptr inbounds %um.Array, ptr ${base}, i32 0, i32 1`);
      this.emit(`${len} = load ${usz}, ptr ${lp}, !tbaa ${HDR}`);
    } else {
      len = String(at.len);
    }
    const more = this.tmp();
    this.emit(`${more} = icmp ult ${usz} ${i}, ${len}`);
    this.emit(`br i1 ${more}, label %${bodyL}, label %${endL}`);
    this.label(bodyL);
    const ep = this.tmp();
    if (at.k === 'heap') {
      const data = this.tmp();
      this.emit(`${data} = load ptr, ptr ${base}, !tbaa ${HDR}`);
      this.emit(`${ep} = getelementptr inbounds ${llTy(at.elem)}, ptr ${data}, ${usz} ${i}`);
    } else {
      this.emit(`${ep} = getelementptr inbounds ${llTy(at)}, ptr ${base}, ${usz} 0, ${usz} ${i}`);
    }
    const ev = this.tmp();
    if (at.k === 'heap' && at.racy) {
      this.emit(`${ev} = load atomic ${llTy(at.elem)}, ptr ${ep} ${this.kernel ? 'unordered' : 'monotonic'}, align ${racyAlign(at.elem)}`);
    } else {
      this.emit(`${ev} = load ${llTy(at.elem)}, ptr ${ep}`);
    }
    this.emit(`store ${llTy(at.elem)} ${ev}, ptr ${slot}`);
    this.loops.push({ brk: endL, cont: stepL, depth: this.scopes.length });
    this.scoped(s.body);
    this.loops.pop();
    this.br(stepL);
    this.label(stepL);
    const cur = this.tmp();
    const next = this.tmp();
    this.emit(`${cur} = load ${usz}, ptr ${idx}`);
    this.emit(`${next} = add ${usz} ${cur}, 1`);
    this.emit(`store ${usz} ${next}, ptr ${idx}`);
    this.br(condL);
    this.label(endL);
  }

  // ---- 検査コード ----

  /** bad が真なら停止する。未定義動作にしないための実行時検査。 */
  check(bad: string, code: number, line: number): void {
    const panicL = this.newLabel('panic');
    const okL = this.newLabel('ok');
    this.emit(`br i1 ${bad}, label %${panicL}, label %${okL}`);
    this.lines.push(`${panicL}:`);
    this.emit(`call void @um_panic(i32 ${code}, i32 ${line})`);
    this.emit('unreachable');
    this.label(okL);
  }

  // ---- 式 ----

  /** 式の置き場所（ポインタ）を返す。置き場所を持たない式は一時領域に退避する。 */
  addr(e: Expr): string {
    return this.prep(e)();
  }

  /**
   * 置き場所の計算を2段階に分ける。
   * この関数自体は、添字の式など利用者のコードが走る部分をすべて評価する。
   * 返す関数は、アドレスを求めるだけで利用者のコードを一切走らせない。
   * 呼び出しの引数は、全引数の評価を終えてからアドレスを求める。そうしないと、
   * 先に求めた要素のアドレスが、後の引数の評価で配列が伸びたときに無効になる。
   */
  prep(e: Expr): () => string {
    if (e.kind === 'ident') {
      // 最上位の定数の配列は、読み取り専用のグローバルな領域を直接指す。
      const p = e.sym!.global ? this.m.constArray(e.sym!) : this.vars.get(e.sym!.id)!;
      return () => p;
    }

    if (e.kind === 'field' && e.base.ty!.k === 'struct') {
      const base = this.prep(e.base);
      return () => {
        const p = this.tmp();
        this.emit(`${p} = getelementptr inbounds ${llTy(e.base.ty!)}, ptr ${base()}, i32 0, i32 ${e.index}`);
        return p;
      };
    }

    if (e.kind === 'field' && e.base.ty!.k === 'heap' && e.name === 'length') {
      const base = this.prep(e.base);
      return () => {
        const p = this.tmp();
        this.emit(`${p} = getelementptr inbounds %um.Array, ptr ${base()}, i32 0, i32 1`);
        return p;
      };
    }

    if (e.kind === 'index') {
      const at = e.base.ty as Ty & ({ k: 'array' } | { k: 'heap' });
      const base = this.prep(e.base);
      const it = e.index.ty as Ty & { k: 'int' };
      // 添字は 64 ビットに広げてから検査する（切り詰めてから検査すると範囲外を見逃す）。
      let idx = this.expr(e.index);
      if (it.bits < 64) {
        const w = this.tmp();
        this.emit(`${w} = ${it.signed ? 'sext' : 'zext'} i${it.bits} ${idx} to i64`);
        idx = w;
      }
      const line = e.pos.line;
      // for ループの検査なしの写しの中で、範囲内だと確認済みの添字か。
      const loopVar = e.base.kind === 'ident' ? this.unchecked.get(e.base.sym!.id) : undefined;
      const proven = at.k === 'heap' && loopVar !== undefined && affineOffset(e.index, loopVar) !== null;
      return () => {
        const b = base();
        const p = this.tmp();
        if (proven) {
          const data = this.tmp();
          this.emit(`${data} = load ptr, ptr ${b}, !tbaa ${HDR}`);
          this.emit(`${p} = getelementptr inbounds ${llTy((at as Ty & { k: 'heap' }).elem)}, ptr ${data}, i64 ${idx}`);
          return p;
        }
        if (at.k === 'heap') {
          const lp = this.tmp();
          const len = this.tmp();
          this.emit(`${lp} = getelementptr inbounds %um.Array, ptr ${b}, i32 0, i32 1`);
          this.emit(`${len} = load ${this.m.usz}, ptr ${lp}, !tbaa ${HDR}`);
          let len64 = len;
          if (this.m.usz !== 'i64') {
            len64 = this.tmp();
            this.emit(`${len64} = zext ${this.m.usz} ${len} to i64`);
          }
          if (this.fast && e.base.kind === 'ident' && indexRange(e.index, this.d.params[0].sym!.id, this.fastLocals) !== null) {
            // 検査なし版: この添字は、呼び出し側が範囲内であることを保証している。
            const data = this.tmp();
            this.emit(`${data} = load ptr, ptr ${b}, !tbaa ${HDR}`);
            this.emit(`${p} = getelementptr inbounds ${llTy(at.elem)}, ptr ${data}, i64 ${idx}`);
            return p;
          }
          if (this.kernel) {
            // 範囲外の添字は最後の要素に切り詰める。Buffer は空でも1要素ぶん確保してあるので、
            // 長さ 0 のときは添字 0 を指す（常に有効な場所）。
            const last = this.tmp();
            const empty = this.tmp();
            const top = this.tmp();
            const data = this.tmp();
            this.emit(`${last} = sub i64 ${len64}, 1`);
            this.emit(`${empty} = icmp eq i64 ${len64}, 0`);
            this.emit(`${top} = select i1 ${empty}, i64 0, i64 ${last}`);
            this.emit(`${data} = load ptr, ptr ${b}, !tbaa ${HDR}`);
            this.emit(`${p} = getelementptr inbounds ${llTy(at.elem)}, ptr ${data}, i64 ${this.umin(idx, top)}`);
            return p;
          }
          const bad = this.tmp();
          this.emit(`${bad} = icmp uge i64 ${idx}, ${len64}`);
          this.check(bad, PANIC_INDEX, line);
          const data = this.tmp();
          this.emit(`${data} = load ptr, ptr ${b}, !tbaa ${HDR}`);
          this.emit(`${p} = getelementptr inbounds ${llTy(at.elem)}, ptr ${data}, i64 ${idx}`);
          return p;
        }
        // 符号なし比較なので、負の添字も範囲外として検出される。定数添字は型検査で確認済み。
        // 値域の解析で「必ず範囲内」と分かっていれば、検査を省く。
        let at64 = idx;
        if (e.index.kind !== 'int' && !this.safeIndex.has(e)) {
          if (this.kernel) {
            at64 = this.umin(idx, String(at.len - 1));
          } else {
            const bad = this.tmp();
            this.emit(`${bad} = icmp uge i64 ${idx}, ${at.len}`);
            this.check(bad, PANIC_INDEX, line);
          }
        }
        this.emit(`${p} = getelementptr inbounds ${llTy(at)}, ptr ${b}, i64 0, i64 ${at64}`);
        return p;
      };
    }

    const a = this.spill(this.expr(e), e.ty!);
    return () => a;
  }

  /** 符号なしの小さい方（i64）。 */
  umin(a: string, b: string): string {
    this.m.intrinsics.add('declare i64 @llvm.umin.i64(i64, i64)');
    const v = this.tmp();
    this.emit(`${v} = call i64 @llvm.umin.i64(i64 ${a}, i64 ${b})`);
    return v;
  }

  expr(e: Expr): string {
    const ty = e.ty!;
    switch (e.kind) {
      case 'int':
        return ty.k === 'float' ? floatConst(Number(e.value), ty) : e.value.toString();

      case 'float':
        return floatConst(e.value, ty);

      case 'str':
        return this.strLit(e.value);

      case 'null':
        throw new Error('internal: null reached code generation');

      case 'bool':
        return e.value ? 'true' : 'false';

      case 'ident':
      case 'field':
      case 'index': {
        const p = this.addr(e);
        const v = this.tmp();
        if (e.kind === 'field' && e.base.ty!.k === 'heap' && e.name === 'length' && llTy(ty) !== this.m.usz) {
          // Buffer の長さは u32。内部では usize で持っているので切り詰める（確保時に 32 ビットに収まることを確認済み）。
          const full = this.tmp();
          this.emit(`${full} = load ${this.m.usz}, ptr ${p}, !tbaa ${HDR}`);
          this.emit(`${v} = trunc ${this.m.usz} ${full} to ${llTy(ty)}`);
          return v;
        }
        if (isRacyElem(e)) {
          this.emit(`${v} = load atomic ${llTy(ty)}, ptr ${p} ${this.kernel ? 'unordered' : 'monotonic'}, align ${racyAlign(ty)}`);
          return v;
        }
        const isLength = e.kind === 'field' && e.base.ty!.k === 'heap';
        const tag = isLength ? `, !tbaa ${HDR}` : isScalar(ty) ? `, !tbaa ${VAL}` : '';
        this.emit(`${v} = load ${llTy(ty)}, ptr ${p}${tag}`);
        return v;
      }

      case 'unary': {
        const x = this.expr(e.expr);
        const v = this.tmp();
        if (e.op === '!') this.emit(`${v} = xor i1 ${x}, true`);
        else if (e.op === '~') this.emit(`${v} = xor ${llTy(ty)} ${x}, -1`);
        else if (isSmall(ty)) {
          const w = this.tmp();
          this.emit(`${w} = fneg float ${this.cast(x, ty, F32T)}`);
          return this.cast(w, F32T, ty);
        } else if (ty.k === 'float') this.emit(`${v} = fneg ${llTy(ty)} ${x}`);
        else this.emit(`${v} = sub ${llTy(ty)} 0, ${x}`);
        return v;
      }

      case 'binary':
        return this.binary(e);

      case 'call':
        return this.call(e);

      case 'new': {
        const sig = this.m.prog.structs.get(e.cls)!.ctor!;
        return this.emitCall(sig, this.args(sig, e.args));
      }

      case 'mcall': {
        if (e.recv.ty!.k === 'heap') return this.arrayMethod(e);
        const info = this.m.prog.structs.get((e.recv.ty as Ty & { k: 'struct' }).name)!;
        const sig = info.methods.get(e.name)!;
        // 受け手に置き場所がなければ（一時値なら）prep() が一時領域に退避する。
        // 受け手のアドレスは、引数をすべて評価し終えてから求める。
        const recv = this.prep(e.recv);
        const args = this.prepArgs(sig, e.args);
        return this.emitCall(sig, [`ptr ${recv()}`, ...args.map((a) => a())]);
      }

      case 'newArray': {
        if (e.args.length === 0) return 'zeroinitializer';
        const ht = ty as Ty & { k: 'heap' };
        const n = this.toUsize(this.expr(e.args[0].expr), e.args[0].expr.ty as Ty & { k: 'int' }, e.pos.line);
        if (ht.buffer) {
          if (this.m.usz === 'i64') {
            // Buffer の長さは u32 で表せる範囲まで。
            const big = this.tmp();
            this.emit(`${big} = icmp ugt i64 ${n}, 4294967295`);
            this.check(big, PANIC_OOM, e.pos.line);
          }
          return this.arrayValue(this.allocBuffer(n, ht.elem), n, n);
        }
        const data = this.tmp();
        this.emit(`${data} = call ptr @um_alloc_zeroed(${this.m.usz} ${n}, ${this.m.usz} ${this.m.sizeOf(ht.elem)})`);
        return this.arrayValue(data, n, n);
      }

      case 'dispatch':
        return this.dispatch(e);

      case 'thread':
        return this.threadExpr(e);

      case 'math':
        return this.math(e);

      case 'object': {
        const info = this.m.prog.structs.get((ty as Ty & { k: 'struct' }).name)!;
        // 評価順はソースに書かれた順、格納先は宣言順。
        const vals = new Map<string, string>();
        for (const f of e.fields) vals.set(f.name, this.consume(f.value));
        let agg = 'undef';
        info.fields.forEach((f, i) => {
          const next = this.tmp();
          this.emit(`${next} = insertvalue ${llTy(ty)} ${agg}, ${llTy(f.ty)} ${vals.get(f.name)}, ${i}`);
          agg = next;
        });
        return agg;
      }

      case 'arrayLit': {
        if (ty.k === 'heap') {
          if (e.elems.length === 0) {
            return ty.buffer ? this.arrayValue(this.allocBuffer('0', ty.elem), '0', '0') : 'zeroinitializer';
          }
          const vals = e.elems.map((el) => this.consume(el));
          const n = String(vals.length);
          // Buffer は必ず allocBuffer で確保する（GPU と共有できる領域にするため）。
          let data = this.tmp();
          if (ty.buffer) data = this.allocBuffer(n, ty.elem);
          else this.emit(`${data} = call ptr @um_alloc(${this.m.usz} ${n}, ${this.m.usz} ${this.m.sizeOf(ty.elem)})`);
          vals.forEach((val, i) => {
            const p = this.tmp();
            this.emit(`${p} = getelementptr inbounds ${llTy(ty.elem)}, ptr ${data}, i32 ${i}`);
            this.emit(`store ${llTy(ty.elem)} ${val}, ptr ${p}`);
          });
          return this.arrayValue(data, n, n);
        }
        const elemTy = (ty as Ty & { k: 'array' }).elem;
        const vals = e.elems.map((el) => this.expr(el));
        let agg = 'undef';
        vals.forEach((val, i) => {
          const next = this.tmp();
          this.emit(`${next} = insertvalue ${llTy(ty)} ${agg}, ${llTy(elemTy)} ${val}, ${i}`);
          agg = next;
        });
        return agg;
      }

      case 'cast':
        return this.cast(this.expr(e.expr), e.expr.ty!, ty);

      case 'cond': {
        // 選ばれた側だけを評価する。
        const c = this.expr(e.cond);
        const thenL = this.newLabel('cond.then');
        const elseL = this.newLabel('cond.else');
        const endL = this.newLabel('cond.end');
        this.emit(`br i1 ${c}, label %${thenL}, label %${elseL}`);
        this.label(thenL);
        const a = this.expr(e.then);
        const aEnd = this.cur;
        this.br(endL);
        this.label(elseL);
        const b = this.expr(e.else);
        const bEnd = this.cur;
        this.br(endL);
        this.label(endL);
        const v = this.tmp();
        this.emit(`${v} = phi ${llTy(ty)} [ ${a}, %${aEnd} ], [ ${b}, %${bEnd} ]`);
        return v;
      }
    }
  }

  /** f8 / f4 のビット列を f32 の値にする（表を引くだけ。すべての値が f32 で正確に表せる）。 */
  decodeSmall(x: string, t: Ty & { k: 'float' }): string {
    const table = this.m.smallTable(t.bits as 4 | 8);
    const n = t.bits === 8 ? 256 : 16;
    let idx = this.tmp();
    this.emit(`${idx} = zext i8 ${x} to i32`);
    if (t.bits === 4) {
      const low = this.tmp();
      this.emit(`${low} = and i32 ${idx}, 15`);
      idx = low;
    }
    const p = this.tmp();
    const v = this.tmp();
    this.emit(`${p} = getelementptr inbounds [${n} x float], ptr ${table}, i32 0, i32 ${idx}`);
    this.emit(`${v} = load float, ptr ${p}`);
    return v;
  }

  cast(x: string, from: Ty, to: Ty): string {
    if (isSmall(from) || isSmall(to)) {
      if (from.k === 'float' && to.k === 'float' && from.bits === to.bits) return x;
      if (isSmall(from)) return this.cast(this.decodeSmall(x, from as Ty & { k: 'float' }), F32T, to);
      // 小さい型への変換: f64 にしてから、ランタイムが最も近い値に丸める（1回の丸めになる）。
      const wide = from.k === 'bool' ? this.cast(this.cast(x, from, { k: 'int', bits: 32, signed: false }), { k: 'int', bits: 32, signed: false }, { k: 'float', bits: 64 })
        : this.cast(x, from, { k: 'float', bits: 64 });
      const bits = (to as Ty & { k: 'float' }).bits;
      this.m.usesSmall = true;
      const r = this.tmp();
      this.emit(`${r} = call zeroext i8 @um_f${bits}_enc(double ${wide})`);
      return r;
    }
    const f = llTy(from);
    const t = llTy(to);
    if (f === t) return x;
    const v = this.tmp();

    if (from.k === 'bool') {
      this.emit(`${v} = zext i1 ${x} to ${t}`);
    } else if (from.k === 'int' && to.k === 'int') {
      if (to.bits < from.bits) this.emit(`${v} = trunc ${f} ${x} to ${t}`);
      else this.emit(`${v} = ${from.signed ? 'sext' : 'zext'} ${f} ${x} to ${t}`);
    } else if (from.k === 'int' && to.k === 'float' && to.bits === 16) {
      // f32 を経由する。f32 への変換で丸めが起きるほど大きい整数は、f16 ではどのみち無限大になる。
      const w = this.tmp();
      this.emit(`${w} = ${from.signed ? 'sitofp' : 'uitofp'} ${f} ${x} to float`);
      return this.cast(w, F32T, to);
    } else if (from.k === 'float' && to.k === 'int' && from.bits === 16) {
      return this.cast(this.cast(x, from, F32T), F32T, to);
    } else if (from.k === 'int' && to.k === 'float') {
      this.emit(`${v} = ${from.signed ? 'sitofp' : 'uitofp'} ${f} ${x} to ${t}`);
    } else if (from.k === 'float' && to.k === 'int') {
      // 範囲外やNaNで未定義動作にならないよう、飽和する変換を使う。
      const name = `llvm.${to.signed ? 'fptosi' : 'fptoui'}.sat.${t}.f${from.bits}`;
      this.m.intrinsics.add(`declare ${t} @${name}(${f})`);
      this.emit(`${v} = call ${t} @${name}(${f} ${x})`);
    } else if (from.k === 'float' && to.k === 'float' && to.bits === 16 && this.m.target === 'wasm32') {
      // wasm では LLVM が f16 の値を f32 のまま持ち回り、「f16 に丸めてすぐ広げる」変換を省いてしまう。
      // 丸めが必ず行われるよう、ランタイムの関数を直接呼ぶ。
      const d = this.cast(x, from, { k: 'float', bits: 64 });
      const bits = this.tmp();
      const h = this.tmp();
      this.emit(`${bits} = bitcast double ${d} to i64`);
      this.emit(`${h} = call i16 @um_d2h(i64 ${bits})`);
      this.emit(`${v} = bitcast i16 ${h} to half`);
    } else if (from.k === 'float' && to.k === 'float') {
      this.emit(`${v} = ${to.bits > from.bits ? 'fpext' : 'fptrunc'} ${f} ${x} to ${t}`);
    } else {
      throw new Error(`internal: cast ${tyStr(from)} -> ${tyStr(to)}`);
    }
    return v;
  }

  binary(e: Expr & { kind: 'binary' }): string {
    const op = e.op;

    if (op === '&&' || op === '||') {
      const l = this.expr(e.left);
      const lhsL = this.cur;
      const rhsL = this.newLabel('rhs');
      const endL = this.newLabel('sc.end');
      if (op === '&&') this.emit(`br i1 ${l}, label %${rhsL}, label %${endL}`);
      else this.emit(`br i1 ${l}, label %${endL}, label %${rhsL}`);
      this.label(rhsL);
      const r = this.expr(e.right);
      const rhsEndL = this.cur;
      this.br(endL);
      this.label(endL);
      const v = this.tmp();
      this.emit(`${v} = phi i1 [ ${op === '&&' ? 'false' : 'true'}, %${lhsL} ], [ ${r}, %${rhsEndL} ]`);
      return v;
    }

    const ot = e.left.ty!;
    const t = llTy(ot);
    if (ot.k === 'heap') {
      // 文字列の連結と比較。両辺は借りるだけで、移動しない。
      // 変数やその中身は、両辺の評価がすべて終わってから読む。先に読んでおくと、右辺の評価が
      // その変数を書き換えたとき（古い中身が解放される）、解放済みの場所を指してしまう。
      const usz = this.m.usz;
      const left = this.heapParts(e.left);
      const right = this.heapParts(e.right);
      const [ld, ll] = left();
      const [rd, rl] = right();
      const v = this.tmp();
      if (op === '+') {
        const total = this.tmp();
        this.emit(`${v} = call ptr @um_str_concat(ptr ${ld}, ${usz} ${ll}, ptr ${rd}, ${usz} ${rl})`);
        this.emit(`${total} = add ${usz} ${ll}, ${rl}`);
        return this.arrayValue(v, total, total);
      }
      const eq = this.tmp();
      this.emit(`${eq} = call i32 @um_str_eq(ptr ${ld}, ${usz} ${ll}, ptr ${rd}, ${usz} ${rl})`);
      this.emit(`${v} = icmp ${op === '==' ? 'ne' : 'eq'} i32 ${eq}, 0`);
      return v;
    }
    const l = this.expr(e.left);
    const r = this.expr(e.right);
    const v = this.tmp();

    if (op === '==' || op === '!=' || op === '<' || op === '<=' || op === '>' || op === '>=') {
      if (ot.k === 'float') {
        const cc = { '==': 'oeq', '!=': 'une', '<': 'olt', '<=': 'ole', '>': 'ogt', '>=': 'oge' }[op];
        // f8 / f4 はビット列なので、値に戻してから比べる。
        if (isSmall(ot)) this.emit(`${v} = fcmp ${cc} float ${this.cast(l, ot, F32T)}, ${this.cast(r, ot, F32T)}`);
        else this.emit(`${v} = fcmp ${cc} ${t} ${l}, ${r}`);
      } else {
        const s = ot.k === 'int' && ot.signed ? 's' : 'u';
        const cc = { '==': 'eq', '!=': 'ne', '<': `${s}lt`, '<=': `${s}le`, '>': `${s}gt`, '>=': `${s}ge` }[op];
        this.emit(`${v} = icmp ${cc} ${t} ${l}, ${r}`);
      }
      return v;
    }

    if (ot.k === 'float') {
      const ins = { '+': 'fadd', '-': 'fsub', '*': 'fmul', '/': 'fdiv', '%': 'frem' }[op as '+'];
      if (ot.bits <= 16) {
        // f16 の演算は「正確な結果を f16 に丸めたもの」。f32 に広げて計算し、丸め直す。
        // f32 は f16 の2倍より多い桁を持つので、2回丸めても1回で丸めた結果と一致する。
        // LLVM に half のまま演算させると、ターゲットによっては途中の丸めが省かれるので、変換を明示する。
        const w = this.tmp();
        this.emit(`${w} = ${ins} float ${this.cast(l, ot, F32T)}, ${this.cast(r, ot, F32T)}`);
        return this.cast(w, F32T, ot);
      }
      this.emit(`${v} = ${ins} ${t} ${l}, ${r}`);
      return v;
    }

    const it = ot as Ty & { k: 'int' };
    switch (op) {
      // 整数演算は既定でラップする（nsw/nuw を付けないので未定義動作にならない）。
      case '+':
        this.emit(`${v} = add ${t} ${l}, ${r}`);
        return v;
      case '-':
        this.emit(`${v} = sub ${t} ${l}, ${r}`);
        return v;
      case '*':
        this.emit(`${v} = mul ${t} ${l}, ${r}`);
        return v;
      case '&':
        this.emit(`${v} = and ${t} ${l}, ${r}`);
        return v;
      case '|':
        this.emit(`${v} = or ${t} ${l}, ${r}`);
        return v;
      case '^':
        this.emit(`${v} = xor ${t} ${l}, ${r}`);
        return v;

      case '/':
      case '%': {
        const zero = this.tmp();
        this.emit(`${zero} = icmp eq ${t} ${r}, 0`);
        if (this.kernel) {
          // kernel では停止しない。ゼロ除算は x / 0 = x、x % 0 = 0 と定義する（WGSL と同じ）。
          const one = this.tmp();
          const q = this.tmp();
          this.emit(`${one} = select i1 ${zero}, ${t} 1, ${t} ${r}`);
          if (it.signed) {
            // 最小値 / -1 は、-1 を 1 に置き換えてから符号を反転する（ラップ）。
            const neg1 = this.tmp();
            const safe = this.tmp();
            this.emit(`${neg1} = icmp eq ${t} ${r}, -1`);
            this.emit(`${safe} = select i1 ${neg1}, ${t} 1, ${t} ${one}`);
            this.emit(`${q} = ${op === '/' ? 'sdiv' : 'srem'} ${t} ${l}, ${safe}`);
            if (op === '/') {
              const neg = this.tmp();
              const signed = this.tmp();
              this.emit(`${neg} = sub ${t} 0, ${l}`);
              this.emit(`${signed} = select i1 ${neg1}, ${t} ${neg}, ${t} ${q}`);
              this.emit(`${v} = select i1 ${zero}, ${t} ${l}, ${t} ${signed}`);
            } else {
              const rem = this.tmp();
              this.emit(`${rem} = select i1 ${neg1}, ${t} 0, ${t} ${q}`);
              this.emit(`${v} = select i1 ${zero}, ${t} 0, ${t} ${rem}`);
            }
            return v;
          }
          this.emit(`${q} = ${op === '/' ? 'udiv' : 'urem'} ${t} ${l}, ${one}`);
          this.emit(`${v} = select i1 ${zero}, ${t} ${op === '/' ? l : '0'}, ${t} ${q}`);
          return v;
        }
        this.check(zero, PANIC_DIV_ZERO, e.pos.line);
        if (!it.signed) {
          this.emit(`${v} = ${op === '/' ? 'udiv' : 'urem'} ${t} ${l}, ${r}`);
          return v;
        }
        // 最小値 / -1 はLLVMでは未定義動作。-1 での除算は符号反転（ラップ）、剰余は 0 と定義する。
        const neg1 = this.tmp();
        const safe = this.tmp();
        const q = this.tmp();
        this.emit(`${neg1} = icmp eq ${t} ${r}, -1`);
        this.emit(`${safe} = select i1 ${neg1}, ${t} 1, ${t} ${r}`);
        this.emit(`${q} = ${op === '/' ? 'sdiv' : 'srem'} ${t} ${l}, ${safe}`);
        if (op === '/') {
          const neg = this.tmp();
          this.emit(`${neg} = sub ${t} 0, ${l}`);
          this.emit(`${v} = select i1 ${neg1}, ${t} ${neg}, ${t} ${q}`);
        } else {
          this.emit(`${v} = select i1 ${neg1}, ${t} 0, ${t} ${q}`);
        }
        return v;
      }

      case '<<':
      case '>>': {
        const ins = op === '<<' ? 'shl' : it.signed ? 'ashr' : 'lshr';
        if (this.kernel) {
          // kernel では停止しない。シフト量はビット幅で割った余りを使う（WGSL と同じ）。
          const amt = this.tmp();
          this.emit(`${amt} = and ${t} ${r}, ${it.bits - 1}`);
          this.emit(`${v} = ${ins} ${t} ${l}, ${amt}`);
          return v;
        }
        // 値域の解析で「シフト量が必ずビット幅未満」と分かっていれば、検査を省く。
        if (!this.safeShift.has(e)) {
          const bad = this.tmp();
          this.emit(`${bad} = icmp uge ${t} ${r}, ${it.bits}`);
          this.check(bad, PANIC_SHIFT, e.pos.line);
        }
        this.emit(`${v} = ${ins} ${t} ${l}, ${r}`);
        return v;
      }
    }
    throw new Error(`internal: binary ${op}`);
  }

  /**
   * 文字列や配列を借りて、データの場所と長さを読む。2段階に分ける（prep と同じ理由）:
   * この関数は式を評価し、返す関数が場所と長さを読む。借りる値が複数あるときは、すべて評価してから読むこと。
   * 一時値は、文の終わりで解放されるように登録する。
   */
  heapParts(x: Expr): () => [string, string] {
    const usz = this.m.usz;
    if (isOwnedTemp(x)) {
      const val = this.expr(x);
      this.spill(val, x.ty!);
      return () => {
        const data = this.tmp();
        const len = this.tmp();
        this.emit(`${data} = extractvalue %um.Array ${val}, 0`);
        this.emit(`${len} = extractvalue %um.Array ${val}, 1`);
        return [data, len];
      };
    }
    const place = this.prep(x);
    return () => {
      const p = place();
      const lp = this.tmp();
      const data = this.tmp();
      const len = this.tmp();
      this.emit(`${data} = load ptr, ptr ${p}, !tbaa ${HDR}`);
      this.emit(`${lp} = getelementptr inbounds %um.Array, ptr ${p}, i32 0, i32 1`);
      this.emit(`${len} = load ${usz}, ptr ${lp}, !tbaa ${HDR}`);
      return [data, len];
    };
  }

  /**
   * 入出力の組み込み（型検査が `__io_名前` に書き換えた呼び出し）。
   * 引数の文字列・配列は借りるだけ。結果（文字列、配列、それらの `| null`）は新しく確保した値で、
   * 呼び出した式が所有する一時値になる（変数に入れば移動し、使われなければ文の終わりで解放される）。
   */
  ioCall(e: Expr & { kind: 'call' }): string {
    const usz = this.m.usz;
    const name = e.callee.slice('__io_'.length);
    // ランタイムが結果（データの場所と長さ）を書く場所。失敗したときも必ず「空」が入っているようにする。
    const outSlots = (): [string, string] => {
      const d = `%g${++this.n}`;
      const l = `%g${++this.n}`;
      this.allocas.push(`  ${d} = alloca ptr`, `  ${l} = alloca ${usz}`);
      this.emit(`store ptr null, ptr ${d}`);
      this.emit(`store ${usz} 0, ptr ${l}`);
      return [d, l];
    };
    // `T | null` の値（{ __some, __value }）にまとめる。null のとき __value は空の配列（解放しても安全）。
    const optional = (ok: string, d: string, l: string): string => {
      const t = llTy(e.ty!);
      const data = this.tmp();
      const len = this.tmp();
      const some = this.tmp();
      this.emit(`${data} = load ptr, ptr ${d}`);
      this.emit(`${len} = load ${usz}, ptr ${l}`);
      this.emit(`${some} = icmp ne i32 ${ok}, 0`);
      const arr = this.arrayValue(data, len, len);
      const a = this.tmp();
      const b = this.tmp();
      this.emit(`${a} = insertvalue ${t} undef, i1 ${some}, 0`);
      this.emit(`${b} = insertvalue ${t} ${a}, %um.Array ${arr}, 1`);
      return b;
    };
    const flag = (call: string): string => {
      const r = this.tmp();
      const b = this.tmp();
      this.emit(`${r} = call i32 ${call}`);
      this.emit(`${b} = icmp ne i32 ${r}, 0`);
      return b;
    };
    switch (name) {
      case 'readLine': {
        this.mainOnly(e.pos.line);
        const [d, l] = outSlots();
        const ok = this.tmp();
        this.emit(`${ok} = call i32 @um_read_line(ptr ${d}, ptr ${l})`);
        return optional(ok, d, l);
      }
      case 'inputReady':
        this.mainOnly(e.pos.line);
        return flag('@um_input_ready()');
      case 'clock': {
        const v = this.tmp();
        this.emit(`${v} = call double @um_clock()`);
        return v;
      }
      case 'stringFromBytes': {
        const [data, len] = this.heapParts(e.args[0].expr)();
        const copy = this.tmp();
        this.emit(`${copy} = call ptr @um_str_concat(ptr ${data}, ${usz} ${len}, ptr null, ${usz} 0)`);
        return this.arrayValue(copy, len, len);
      }
      case 'readFile': {
        const [path, plen] = this.heapParts(e.args[0].expr)();
        const [d, l] = outSlots();
        const ok = this.tmp();
        this.emit(`${ok} = call i32 @um_read_file(ptr ${path}, ${usz} ${plen}, ptr ${d}, ptr ${l})`);
        return optional(ok, d, l);
      }
      case 'writeFile': {
        // 両方の引数を評価し終えてから、場所と長さを読む。
        const pathParts = this.heapParts(e.args[0].expr);
        const dataParts = this.heapParts(e.args[1].expr);
        const [path, plen] = pathParts();
        const [data, len] = dataParts();
        return flag(`@um_write_file(ptr ${path}, ${usz} ${plen}, ptr ${data}, ${usz} ${len})`);
      }
      case 'appendFile': {
        const pathParts = this.heapParts(e.args[0].expr);
        const dataParts = this.heapParts(e.args[1].expr);
        const [path, plen] = pathParts();
        const [data, len] = dataParts();
        return flag(`@um_append_file(ptr ${path}, ${usz} ${plen}, ptr ${data}, ${usz} ${len})`);
      }
      case 'fileSize': {
        const [path, plen] = this.heapParts(e.args[0].expr)();
        const slot = `%g${++this.n}`;
        this.allocas.push(`  ${slot} = alloca i64`);
        this.emit(`store i64 0, ptr ${slot}`);
        const ok = this.tmp();
        this.emit(`${ok} = call i32 @um_file_size(ptr ${path}, ${usz} ${plen}, ptr ${slot})`);
        const t = llTy(e.ty!);
        const v = this.tmp();
        const some = this.tmp();
        const a = this.tmp();
        const b = this.tmp();
        this.emit(`${v} = load i64, ptr ${slot}`);
        this.emit(`${some} = icmp ne i32 ${ok}, 0`);
        this.emit(`${a} = insertvalue ${t} undef, i1 ${some}, 0`);
        this.emit(`${b} = insertvalue ${t} ${a}, i64 ${v}, 1`);
        return b;
      }
      case 'readFileRange': {
        // 3つの引数を評価し終えてから、名前の場所と長さを読む。
        const pathParts = this.heapParts(e.args[0].expr);
        const off = this.expr(e.args[1].expr);
        const want = this.expr(e.args[2].expr);
        const [path, plen] = pathParts();
        const [d, l] = outSlots();
        const ok = this.tmp();
        this.emit(`${ok} = call i32 @um_read_file_range(ptr ${path}, ${usz} ${plen}, i64 ${off}, i64 ${want}, ptr ${d}, ptr ${l})`);
        return optional(ok, d, l);
      }
      case 'args': {
        const [d, l] = outSlots();
        this.emit(`call void @um_args(ptr ${d}, ptr ${l})`);
        const data = this.tmp();
        const len = this.tmp();
        this.emit(`${data} = load ptr, ptr ${d}`);
        this.emit(`${len} = load ${usz}, ptr ${l}`);
        return this.arrayValue(data, len, len);
      }
      case 'eprint': {
        const [data, len] = this.heapParts(e.args[0].expr)();
        this.emit(`call void @um_eprint(ptr ${data}, ${usz} ${len})`);
        return '';
      }
      // ファイルに直接読む・開いたままのファイル。番号と範囲の検査はランタイムが行い、違反なら行番号つきで停止する（16）。
      case 'readFileInto': {
        const pathParts = this.heapParts(e.args[0].expr);
        const off = this.expr(e.args[1].expr);
        const bufParts = this.heapParts(e.args[2].expr);
        const boff = this.expr(e.args[3].expr);
        const want = this.expr(e.args[4].expr);
        const [path, plen] = pathParts();
        const [data, len] = bufParts();
        const slot = this.i64Slot();
        const ok = this.tmp();
        this.emit(`${ok} = call i32 @um_read_file_into(ptr ${path}, ${usz} ${plen}, i64 ${off}, ptr ${data}, ${usz} ${len}, i64 ${boff}, i64 ${want}, ptr ${slot}, i32 ${e.pos.line})`);
        return this.optionalI64(e, ok, slot);
      }
      case 'fileOpen': {
        const pathParts = this.heapParts(e.args[0].expr);
        const modeParts = this.heapParts(e.args[1].expr);
        const [path, plen] = pathParts();
        const [mode, mlen] = modeParts();
        const id = this.tmp();
        this.emit(`${id} = call i32 @um_file_open(ptr ${path}, ${usz} ${plen}, ptr ${mode}, ${usz} ${mlen}, i32 ${e.pos.line})`);
        const t = llTy(e.ty!);
        const some = this.tmp();
        const a = this.tmp();
        const b = this.tmp();
        this.emit(`${some} = icmp ne i32 ${id}, 0`);
        this.emit(`${a} = insertvalue ${t} undef, i1 ${some}, 0`);
        this.emit(`${b} = insertvalue ${t} ${a}, i32 ${id}, 1`);
        return b;
      }
      case 'fileRead':
      case 'fileReadAt': {
        const at = name === 'fileReadAt' ? 2 : 1;
        const h = this.expr(e.args[0].expr);
        const off = at === 2 ? this.expr(e.args[1].expr) : '0';
        const bufParts = this.heapParts(e.args[at].expr);
        const start = e.args.length > at + 1 ? this.expr(e.args[at + 1].expr) : '0';
        const count = e.args.length > at + 1 ? this.expr(e.args[at + 2].expr) : '-1';
        const [data, len] = bufParts();
        const slot = this.i64Slot();
        const ok = this.tmp();
        this.emit(`${ok} = call i32 @um_file_read(i32 ${h}, i32 ${at === 2 ? 1 : 0}, i64 ${off}, ptr ${data}, ${usz} ${len}, i64 ${start}, i64 ${count}, ptr ${slot}, i32 ${e.pos.line})`);
        return this.optionalI64(e, ok, slot);
      }
      case 'fileWrite': {
        const h = this.expr(e.args[0].expr);
        const dataParts = this.heapParts(e.args[1].expr);
        const start = e.args.length > 2 ? this.expr(e.args[2].expr) : '0';
        const count = e.args.length > 2 ? this.expr(e.args[3].expr) : '-1';
        const [data, len] = dataParts();
        return flag(`@um_file_write(i32 ${h}, ptr ${data}, ${usz} ${len}, i64 ${start}, i64 ${count}, i32 ${e.pos.line})`);
      }
      case 'fileSeek': {
        const h = this.expr(e.args[0].expr);
        const off = this.expr(e.args[1].expr);
        return flag(`@um_file_seek(i32 ${h}, i64 ${off}, i32 ${e.pos.line})`);
      }
      case 'fileLength': {
        const h = this.expr(e.args[0].expr);
        const slot = this.i64Slot();
        const ok = this.tmp();
        this.emit(`${ok} = call i32 @um_file_length(i32 ${h}, ptr ${slot}, i32 ${e.pos.line})`);
        return this.optionalI64(e, ok, slot);
      }
      case 'fileClose': {
        const h = this.expr(e.args[0].expr);
        this.emit(`call void @um_file_close(i32 ${h}, i32 ${e.pos.line})`);
        return '';
      }
      // フォルダとパス。
      case 'listDir': {
        const [path, plen] = this.heapParts(e.args[0].expr)();
        const [d, l] = outSlots();
        const ok = this.tmp();
        this.emit(`${ok} = call i32 @um_list_dir(ptr ${path}, ${usz} ${plen}, ptr ${d}, ptr ${l})`);
        return optional(ok, d, l);
      }
      case 'makeDir':
      case 'isDir': {
        const [path, plen] = this.heapParts(e.args[0].expr)();
        return flag(`@um_${name === 'makeDir' ? 'make_dir' : 'is_dir'}(ptr ${path}, ${usz} ${plen})`);
      }
      case 'cwd':
      case 'exePath':
      case 'exeDir':
      case 'absPath':
      case 'envVar': {
        const kind = ['cwd', 'exePath', 'exeDir', 'absPath', 'envVar'].indexOf(name);
        const [path, plen] = name === 'absPath' || name === 'envVar' ? this.heapParts(e.args[0].expr)() : ['null', '0'];
        const [d, l] = outSlots();
        this.emit(`call void @um_path_query(i32 ${kind}, ptr ${path}, ${usz} ${plen}, ptr ${d}, ptr ${l})`);
        const data = this.tmp();
        const len = this.tmp();
        this.emit(`${data} = load ptr, ptr ${d}`);
        this.emit(`${len} = load ${usz}, ptr ${l}`);
        return this.arrayValue(data, len, len);
      }
      // ウィンドウ（runtime/window.c）。メインスレッドの確認と引数の検査はランタイムが行い、違反なら行番号つきで停止する。
      case 'windowOpen': {
        this.m.usesWindow = true;
        const titleParts = this.heapParts(e.args[0].expr);
        const w = this.expr(e.args[1].expr);
        const h = this.expr(e.args[2].expr);
        const [title, tlen] = titleParts();
        const id = this.tmp();
        this.emit(`${id} = call i32 @um_window_open(ptr ${title}, ${usz} ${tlen}, i32 ${w}, i32 ${h}, i32 ${e.pos.line})`);
        const t = llTy(e.ty!);
        const some = this.tmp();
        const a = this.tmp();
        const b = this.tmp();
        this.emit(`${some} = icmp ne i32 ${id}, 0`);
        this.emit(`${a} = insertvalue ${t} undef, i1 ${some}, 0`);
        this.emit(`${b} = insertvalue ${t} ${a}, i32 ${id}, 1`);
        return b;
      }
      case 'windowClose': {
        this.m.usesWindow = true;
        const id = this.expr(e.args[0].expr);
        this.emit(`call void @um_window_close(i32 ${id}, i32 ${e.pos.line})`);
        return '';
      }
      case 'windowPoll': {
        this.m.usesWindow = true;
        const evParts = this.heapParts(e.args[0].expr);
        const timeout = this.expr(e.args[1].expr);
        const [data, len] = evParts();
        return flag(`@um_window_poll(ptr ${data}, ${usz} ${len}, i32 ${timeout}, i32 ${e.pos.line})`);
      }
      case 'windowPresent': {
        this.m.usesWindow = true;
        const id = this.expr(e.args[0].expr);
        const pxParts = this.heapParts(e.args[1].expr);
        const w = this.expr(e.args[2].expr);
        const h = this.expr(e.args[3].expr);
        const [data, len] = pxParts();
        this.emit(`call void @um_window_present(i32 ${id}, ptr ${data}, ${usz} ${len}, i32 ${w}, i32 ${h}, i32 ${e.pos.line})`);
        return '';
      }
      case 'windowSetIcon': {
        this.m.usesWindow = true;
        const id = this.expr(e.args[0].expr);
        const pxParts = this.heapParts(e.args[1].expr);
        const w = this.expr(e.args[2].expr);
        const h = this.expr(e.args[3].expr);
        const [data, len] = pxParts();
        return flag(`@um_window_set_icon(i32 ${id}, ptr ${data}, ${usz} ${len}, i32 ${w}, i32 ${h}, i32 ${e.pos.line})`);
      }
      case 'windowPresentRect':
      case 'windowUpdateRect': {
        this.m.usesWindow = true;
        const show = name === 'windowPresentRect' ? 1 : 0;
        const id = this.expr(e.args[0].expr);
        const pxParts = this.heapParts(e.args[1].expr);
        const [w, h, x, y, rw, rh] = e.args.slice(2).map((a) => this.expr(a.expr));
        const [data, len] = pxParts();
        this.emit(`call void @um_window_present_rect(i32 ${id}, ptr ${data}, ${usz} ${len}, i32 ${w}, i32 ${h}, i32 ${x}, i32 ${y}, i32 ${rw}, i32 ${rh}, i32 ${show}, i32 ${e.pos.line})`);
        return '';
      }
      // 操作番号で呼び分ける組み込み（番号の意味は runtime/window.c と runtime/audio.c）。
      case 'windowCtl': {
        this.m.usesWindow = true;
        const op = this.expr(e.args[0].expr);
        const id = this.expr(e.args[1].expr);
        const ioParts = this.heapParts(e.args[2].expr);
        const [data, len] = ioParts();
        const r = this.tmp();
        this.emit(`${r} = call i64 @um_window_ctl(i32 ${op}, i32 ${id}, ptr ${data}, ${usz} ${len}, i32 ${e.pos.line})`);
        return r;
      }
      case 'windowCtlF': {
        this.m.usesWindow = true;
        const [op, id, a] = e.args.map((x) => this.expr(x.expr));
        const r = this.tmp();
        this.emit(`${r} = call double @um_window_ctl_f(i32 ${op}, i32 ${id}, i64 ${a}, i32 ${e.pos.line})`);
        return r;
      }
      case 'windowSetText': {
        this.m.usesWindow = true;
        const op = this.expr(e.args[0].expr);
        const id = this.expr(e.args[1].expr);
        const a = this.expr(e.args[2].expr);
        const textParts = this.heapParts(e.args[3].expr);
        const [text, tlen] = textParts();
        const r = this.tmp();
        this.emit(`${r} = call i64 @um_window_set_text(i32 ${op}, i32 ${id}, i64 ${a}, ptr ${text}, ${usz} ${tlen}, i32 ${e.pos.line})`);
        return r;
      }
      case 'windowGetText': {
        this.m.usesWindow = true;
        const [op, id, a] = e.args.map((x) => this.expr(x.expr));
        const [d, l] = outSlots();
        this.emit(`call void @um_window_get_text(i32 ${op}, i32 ${id}, i64 ${a}, ptr ${d}, ptr ${l}, i32 ${e.pos.line})`);
        const data = this.tmp();
        const len = this.tmp();
        this.emit(`${data} = load ptr, ptr ${d}`);
        this.emit(`${len} = load ${usz}, ptr ${l}`);
        return this.arrayValue(data, len, len);
      }
      case 'audioWrite': {
        this.m.usesWindow = true;
        const id = this.expr(e.args[0].expr);
        const sParts = this.heapParts(e.args[1].expr);
        const start = this.expr(e.args[2].expr);
        const count = this.expr(e.args[3].expr);
        const [data, len] = sParts();
        return flag(`@um_audio_write(i32 ${id}, ptr ${data}, ${usz} ${len}, i64 ${start}, i64 ${count}, i32 ${e.pos.line})`);
      }
    }
    if (name.startsWith('wasm')) {
      this.m.usesWasm = true;
      return this.wasmCall(e, name, outSlots, optional, flag);
    }
    if (name.startsWith('media')) {
      this.m.usesMedia = true;
      return this.mediaCall(e, name, outSlots, flag);
    }
    throw new Error(`internal: builtin ${name}`);
  }

  /**
   * 動画・音声の組み込み（runtime/media.c）。番号の検査はランタイムが行い、違反なら行番号つきで停止する。
   * 失敗（開けない、壊れたデータ、Buffer の型や長さが合わないなど）は停止せず、null / false / 負の値を返す（理由は mediaError）。
   * 画素・標本の配列は、場所・要素の数・要素のバイト数・要素の型の印（型検査の mediaArrayParam）で渡す。
   */
  mediaCall(e: Expr & { kind: 'call' }, name: string, outSlots: () => [string, string], flag: (call: string) => string): string {
    const usz = this.m.usz;
    const line = e.pos.line;
    // 番号（0 は失敗）を i64 | null にする。
    const handle = (call: string): string => {
      const id = this.tmp();
      this.emit(`${id} = call i64 ${call}`);
      const t = llTy(e.ty!);
      const some = this.tmp();
      const a = this.tmp();
      const b = this.tmp();
      this.emit(`${some} = icmp ne i64 ${id}, 0`);
      this.emit(`${a} = insertvalue ${t} undef, i1 ${some}, 0`);
      this.emit(`${b} = insertvalue ${t} ${a}, i64 ${id}, 1`);
      return b;
    };
    // ランタイムが書いた配列（場所と長さ）を受け取る。
    const array = (call: (d: string, l: string) => string): string => {
      const [d, l] = outSlots();
      this.emit(`call void ${call(d, l)}`);
      const data = this.tmp();
      const len = this.tmp();
      this.emit(`${data} = load ptr, ptr ${d}`);
      this.emit(`${len} = load ${usz}, ptr ${l}`);
      return this.arrayValue(data, len, len);
    };
    // 配列の要素のバイト数と、要素の型の印。
    const elemInfo = (x: Expr): [string | number, number] => {
      const t = x.ty!;
      return [t.k === 'heap' ? this.m.sizeOf(t.elem) : 1, e.elemScalar ?? 0];
    };
    switch (name) {
      case 'mediaOpen': {
        const pathParts = this.heapParts(e.args[0].expr);
        const kind = this.expr(e.args[1].expr);
        const hwParts = this.heapParts(e.args[2].expr);
        const [p, pl] = pathParts();
        const [h, hl] = hwParts();
        return handle(`@um_media_open(ptr ${p}, ${usz} ${pl}, i32 ${kind}, ptr ${h}, ${usz} ${hl}, i32 ${line})`);
      }
      case 'mediaClose': {
        const h = this.expr(e.args[0].expr);
        this.emit(`call void @um_media_close(i64 ${h}, i32 ${line})`);
        return '';
      }
      case 'mediaInfoNumbers':
      case 'mediaInfoText':
      case 'mediaWriterText': {
        const h = this.expr(e.args[0].expr);
        const fn = name === 'mediaInfoNumbers' ? 'um_media_info_numbers' : name === 'mediaInfoText' ? 'um_media_info_text' : 'um_media_writer_text';
        return array((d, l) => `@${fn}(i64 ${h}, ptr ${d}, ptr ${l}, i32 ${line})`);
      }
      case 'mediaReadVideo': {
        const h = this.expr(e.args[0].expr);
        const time = this.expr(e.args[1].expr);
        const parts = this.heapParts(e.args[2].expr);
        const [fmt, w, ht, filter] = e.args.slice(3).map((a) => this.expr(a.expr));
        const [data, len] = parts();
        const [size, scalar] = elemInfo(e.args[2].expr);
        const r = this.tmp();
        this.emit(`${r} = call double @um_media_read_video(i64 ${h}, double ${time}, ptr ${data}, ${usz} ${len}, ${usz} ${size}, i32 ${scalar}, ` +
          `i32 ${fmt}, i32 ${w}, i32 ${ht}, i32 ${filter}, i32 ${line})`);
        return r;
      }
      case 'mediaReadAudio': {
        const h = this.expr(e.args[0].expr);
        const time = this.expr(e.args[1].expr);
        const parts = this.heapParts(e.args[2].expr);
        const rate = this.expr(e.args[3].expr);
        const [data, len] = parts();
        const [size, scalar] = elemInfo(e.args[2].expr);
        const r = this.tmp();
        this.emit(`${r} = call i64 @um_media_read_audio(i64 ${h}, double ${time}, ptr ${data}, ${usz} ${len}, ${usz} ${size}, i32 ${scalar}, i32 ${rate}, i32 ${line})`);
        return r;
      }
      case 'mediaLastTime': {
        const h = this.expr(e.args[0].expr);
        const r = this.tmp();
        this.emit(`${r} = call double @um_media_last_time(i64 ${h}, i32 ${line})`);
        return r;
      }
      case 'mediaCreateWriter': {
        const [p, pl] = this.heapParts(e.args[0].expr)();
        return handle(`@um_media_create(ptr ${p}, ${usz} ${pl}, i32 ${line})`);
      }
      case 'mediaAddVideo': {
        const h = this.expr(e.args[0].expr);
        const encParts = this.heapParts(e.args[1].expr);
        const [w, ht, fn, fd, br] = e.args.slice(2, 7).map((a) => this.expr(a.expr));
        const optParts = this.heapParts(e.args[7].expr);
        const [ep, el] = encParts();
        const [op, ol] = optParts();
        return flag(`@um_media_add_video(i64 ${h}, ptr ${ep}, ${usz} ${el}, i32 ${w}, i32 ${ht}, i32 ${fn}, i32 ${fd}, i64 ${br}, ptr ${op}, ${usz} ${ol}, i32 ${line})`);
      }
      case 'mediaAddAudio': {
        const h = this.expr(e.args[0].expr);
        const encParts = this.heapParts(e.args[1].expr);
        const [rate, ch, br] = e.args.slice(2, 5).map((a) => this.expr(a.expr));
        const optParts = this.heapParts(e.args[5].expr);
        const [ep, el] = encParts();
        const [op, ol] = optParts();
        return flag(`@um_media_add_audio(i64 ${h}, ptr ${ep}, ${usz} ${el}, i32 ${rate}, i32 ${ch}, i64 ${br}, ptr ${op}, ${usz} ${ol}, i32 ${line})`);
      }
      case 'mediaWriteVideo': {
        const h = this.expr(e.args[0].expr);
        const parts = this.heapParts(e.args[1].expr);
        const [fmt, w, ht] = e.args.slice(2).map((a) => this.expr(a.expr));
        const [data, len] = parts();
        const [size, scalar] = elemInfo(e.args[1].expr);
        return flag(`@um_media_write_video(i64 ${h}, ptr ${data}, ${usz} ${len}, ${usz} ${size}, i32 ${scalar}, i32 ${fmt}, i32 ${w}, i32 ${ht}, i32 ${line})`);
      }
      case 'mediaWriteAudio': {
        const h = this.expr(e.args[0].expr);
        const parts = this.heapParts(e.args[1].expr);
        const count = this.expr(e.args[2].expr);
        const [data, len] = parts();
        const [size, scalar] = elemInfo(e.args[1].expr);
        return flag(`@um_media_write_audio(i64 ${h}, ptr ${data}, ${usz} ${len}, ${usz} ${size}, i32 ${scalar}, i64 ${count}, i32 ${line})`);
      }
      case 'mediaFinishWriter': {
        const h = this.expr(e.args[0].expr);
        return flag(`@um_media_finish(i64 ${h}, i32 ${line})`);
      }
      case 'mediaEncoderList': {
        const probe = this.expr(e.args[0].expr);
        const p32 = this.tmp();
        this.emit(`${p32} = zext i1 ${probe} to i32`);
        return array((d, l) => `@um_media_encoders(i32 ${p32}, ptr ${d}, ptr ${l})`);
      }
      case 'mediaError':
        return array((d, l) => `@um_media_error(ptr ${d}, ptr ${l})`);
      case 'mediaVersion':
        return array((d, l) => `@um_media_version(ptr ${d}, ptr ${l})`);
    }
    throw new Error(`internal: builtin ${name}`);
  }

  /**
   * WebAssembly の組み込み（runtime/wasm.c）。番号の検査と範囲の検査はランタイムが行い、違反なら行番号つきで停止する。
   * 失敗（読み込めない、トラップなど）は停止せず、null / false / -1 を返す（理由は wasmError）。
   */
  wasmCall(
    e: Expr & { kind: 'call' }, name: string, outSlots: () => [string, string],
    optional: (ok: string, d: string, l: string) => string, flag: (call: string) => string,
  ): string {
    const usz = this.m.usz;
    const line = e.pos.line;
    // 番号（0 は失敗）を i64 | null にする。
    const handle = (call: string): string => {
      const id = this.tmp();
      this.emit(`${id} = call i64 ${call}`);
      const t = llTy(e.ty!);
      const some = this.tmp();
      const a = this.tmp();
      const b = this.tmp();
      this.emit(`${some} = icmp ne i64 ${id}, 0`);
      this.emit(`${a} = insertvalue ${t} undef, i1 ${some}, 0`);
      this.emit(`${b} = insertvalue ${t} ${a}, i64 ${id}, 1`);
      return b;
    };
    switch (name) {
      case 'wasmCompile': {
        const [data, len] = this.heapParts(e.args[0].expr)();
        return handle(`@um_wasm_compile(ptr ${data}, ${usz} ${len}, i32 ${line})`);
      }
      case 'wasmFreeModule': {
        const h = this.expr(e.args[0].expr);
        this.emit(`call void @um_wasm_free_module(i64 ${h}, i32 ${line})`);
        return '';
      }
      case 'wasmExportList': {
        const h = this.expr(e.args[0].expr);
        const [d, l] = outSlots();
        this.emit(`call void @um_wasm_exports(i64 ${h}, ptr ${d}, ptr ${l}, i32 ${line})`);
        const data = this.tmp();
        const len = this.tmp();
        this.emit(`${data} = load ptr, ptr ${d}`);
        this.emit(`${len} = load ${usz}, ptr ${l}`);
        return this.arrayValue(data, len, len);
      }
      case 'wasmInstantiate': {
        const [m, fuel, timeout, mem, stub] = e.args.map((a) => this.expr(a.expr));
        const stub32 = this.tmp();
        this.emit(`${stub32} = zext i1 ${stub} to i32`);
        return handle(`@um_wasm_instantiate(i64 ${m}, i64 ${fuel}, i64 ${timeout}, i64 ${mem}, i32 ${stub32}, i32 ${line})`);
      }
      case 'wasmFreeInstance': {
        const h = this.expr(e.args[0].expr);
        this.emit(`call void @um_wasm_free_instance(i64 ${h}, i32 ${line})`);
        return '';
      }
      case 'wasmInvoke': {
        // すべての引数を評価し終えてから、場所と長さを読む。
        const h = this.expr(e.args[0].expr);
        const nameParts = this.heapParts(e.args[1].expr);
        const kindParts = this.heapParts(e.args[2].expr);
        const intParts = this.heapParts(e.args[3].expr);
        const floatParts = this.heapParts(e.args[4].expr);
        const nargs = this.expr(e.args[5].expr);
        const [np, nl] = nameParts();
        const [kp, kl] = kindParts();
        const [ip, il] = intParts();
        const [fp, fl] = floatParts();
        const r = this.tmp();
        this.emit(`${r} = call i32 @um_wasm_invoke(i64 ${h}, ptr ${np}, ${usz} ${nl}, ptr ${kp}, ${usz} ${kl}, ptr ${ip}, ${usz} ${il}, ptr ${fp}, ${usz} ${fl}, i32 ${nargs}, i32 ${line})`);
        return r;
      }
      case 'wasmRead':
      case 'wasmWrite': {
        const h = this.expr(e.args[0].expr);
        const off = this.expr(e.args[1].expr);
        const arrTy = e.args[2].expr.ty!;
        const parts = this.heapParts(e.args[2].expr);
        const start = e.args.length > 3 ? this.expr(e.args[3].expr) : '0';
        const count = e.args.length > 3 ? this.expr(e.args[4].expr) : '-1';
        const [data, len] = parts();
        const size = arrTy.k === 'heap' ? this.m.sizeOf(arrTy.elem) : 1;
        const fn = name === 'wasmRead' ? 'um_wasm_read' : 'um_wasm_write';
        return flag(`@${fn}(i64 ${h}, i64 ${off}, ptr ${data}, ${usz} ${len}, ${usz} ${size}, i64 ${start}, i64 ${count}, i32 ${line})`);
      }
      case 'wasmMemorySize': {
        const h = this.expr(e.args[0].expr);
        const r = this.tmp();
        this.emit(`${r} = call i64 @um_wasm_memory_size(i64 ${h}, i32 ${line})`);
        return r;
      }
      case 'wasmWat': {
        const [text, tlen] = this.heapParts(e.args[0].expr)();
        const [d, l] = outSlots();
        this.emit(`call i32 @um_wasm_wat(ptr ${text}, ${usz} ${tlen}, ptr ${d}, ptr ${l})`);
        const data = this.tmp();
        const len = this.tmp();
        this.emit(`${data} = load ptr, ptr ${d}`);
        this.emit(`${len} = load ${usz}, ptr ${l}`);
        return this.arrayValue(data, len, len);
      }
      case 'wasmError': {
        const [d, l] = outSlots();
        this.emit(`call void @um_wasm_error(ptr ${d}, ptr ${l})`);
        const data = this.tmp();
        const len = this.tmp();
        this.emit(`${data} = load ptr, ptr ${d}`);
        this.emit(`${len} = load ${usz}, ptr ${l}`);
        return this.arrayValue(data, len, len);
      }
    }
    throw new Error(`internal: builtin ${name}`);
  }

  /** ランタイムが i64 を書き込む場所（0 で初期化）。 */
  i64Slot(): string {
    const slot = `%g${++this.n}`;
    this.allocas.push(`  ${slot} = alloca i64`);
    this.emit(`store i64 0, ptr ${slot}`);
    return slot;
  }

  /** ok（i32、0 は失敗）と slot の i64 を `i64 | null` にする。 */
  optionalI64(e: Expr, ok: string, slot: string): string {
    const t = llTy(e.ty!);
    const v = this.tmp();
    const some = this.tmp();
    const a = this.tmp();
    const b = this.tmp();
    this.emit(`${v} = load i64, ptr ${slot}`);
    this.emit(`${some} = icmp ne i32 ${ok}, 0`);
    this.emit(`${a} = insertvalue ${t} undef, i1 ${some}, 0`);
    this.emit(`${b} = insertvalue ${t} ${a}, i64 ${v}, 1`);
    return b;
  }

  /** 文字列リテラル。内容を定数として持ち、新しく確保した領域へコピーする（文字列は所有される値なので）。 */
  strLit(text: string): string {
    const bytes = Buffer.from(text, 'utf8');
    if (bytes.length === 0) return 'zeroinitializer';
    const usz = this.m.usz;
    const g = this.m.bytesGlobal(bytes);
    const data = this.tmp();
    this.emit(`${data} = call ptr @um_alloc(${usz} ${bytes.length}, ${usz} 1)`);
    const name = `llvm.memcpy.p0.p0.${usz}`;
    this.m.intrinsics.add(`declare void @${name}(ptr, ptr, ${usz}, i1)`);
    this.emit(`call void @${name}(ptr ${data}, ptr ${g}, ${usz} ${bytes.length}, i1 false)`);
    return this.arrayValue(data, String(bytes.length), String(bytes.length));
  }

  /**
   * 文字列の値 v（式 e の結果）から、データの場所と長さを取り出す。
   * e が一時値なら、文の終わりで解放されるように登録する。
   */
  strParts(e: Expr, v: string): [string, string] {
    if (isOwnedTemp(e)) this.spill(v, e.ty!);
    const data = this.tmp();
    const len = this.tmp();
    this.emit(`${data} = extractvalue %um.Array ${v}, 0`);
    this.emit(`${len} = extractvalue %um.Array ${v}, 1`);
    return [data, len];
  }

  /** ランタイムが作った文字列（データの場所を返し、長さを outLen に書く）を値にまとめる。 */
  strFromRuntime(fn: string, arg: string): string {
    const usz = this.m.usz;
    const slot = this.alloca({ k: 'int', bits: this.m.usz === 'i64' ? 64 : 32, signed: false });
    const data = this.tmp();
    const len = this.tmp();
    this.emit(`${data} = call ptr @${fn}(${arg}, ptr ${slot})`);
    this.emit(`${len} = load ${usz}, ptr ${slot}`);
    return this.arrayValue(data, len, len);
  }

  /** String(x): 数・bool・文字列を新しい文字列にする。数の表記は print と同じ。 */
  toStr(a: Expr): string {
    const at = a.ty!;
    const x = this.expr(a);
    if (at.k === 'heap') {
      const [data, len] = this.strParts(a, x);
      const copy = this.tmp();
      this.emit(`${copy} = call ptr @um_str_concat(ptr ${data}, ${this.m.usz} ${len}, ptr null, ${this.m.usz} 0)`);
      return this.arrayValue(copy, len, len);
    }
    if (at.k === 'bool') {
      const t = this.strLit('true');
      const f = this.strLit('false');
      // 使わなかった方も一時値として解放する。
      const pick = this.tmp();
      const other = this.tmp();
      this.emit(`${pick} = select i1 ${x}, %um.Array ${t}, %um.Array ${f}`);
      this.emit(`${other} = select i1 ${x}, %um.Array ${f}, %um.Array ${t}`);
      this.spill(other, STR_TY);
      return pick;
    }
    if (at.k === 'float') return this.strFromRuntime('um_str_from_f64', `double ${this.cast(x, at, { k: 'float', bits: 64 })}`);
    const it = at as Ty & { k: 'int' };
    const w = this.cast(x, at, { k: 'int', bits: 64, signed: it.signed });
    return this.strFromRuntime(it.signed ? 'um_str_from_i64' : 'um_str_from_u64', `i64 ${w}`);
  }

  call(e: Expr & { kind: 'call' }): string {
    if (e.callee === 'String' && !this.m.prog.fns.has('String')) return this.toStr(e.args[0].expr);
    if (e.callee.startsWith('__io_')) return this.ioCall(e);
    if (e.callee === '__gpu_ready' || e.callee === '__gpu_wait') {
      // `gpu.ready(t)` / `gpu.wait(t)`。Metal 以外では、submit がその場で実行し終えているので、いつでも「終わっている」。
      const t = this.expr(e.args[0].expr);
      if (this.m.gpu !== 'metal' && this.m.gpu !== 'wgpu') return e.callee === '__gpu_ready' ? 'true' : '';
      if (this.m.gpu === 'wgpu') this.m.usesGpuRuntime = true;
      if (e.callee === '__gpu_wait') {
        this.emit(`call void @um_gpu_wait(i64 ${t})`);
        return '';
      }
      const r = this.tmp();
      const b = this.tmp();
      this.emit(`${r} = call i32 @um_gpu_ready(i64 ${t})`);
      this.emit(`${b} = icmp ne i32 ${r}, 0`);
      return b;
    }
    if (e.callee === '__gpu_flags') {
      // `gpu.flags(f)`。Metal 以外では印に意味がないので、引数の式だけ評価して 0 を返す。
      const f = this.expr(e.args[0].expr);
      // wgpu では印を覚えて前の値を返すだけ（効果はない）。
      if (this.m.gpu === 'wgpu') this.m.usesGpuRuntime = true;
      else if (this.m.gpu !== 'metal') return '0';
      const r = this.tmp();
      this.emit(`${r} = call i32 @um_gpu_set_flags(i32 ${f})`);
      return r;
    }
    if (e.callee === '__gpu_untracked') {
      // `gpu.untracked(b)`。Metal 以外では、引数の式を評価するだけ。
      const a = e.args[0].expr;
      const [data] = this.strParts(a, this.expr(a));
      if (this.m.gpu === 'metal') this.emit(`call void @um_gpu_untracked(ptr ${data})`);
      return '';
    }
    if (e.callee === '__gpu_available') {
      if (!this.m.gpu) return 'false';
      if (this.m.gpu === 'wgpu') this.m.usesGpuRuntime = true;
      const r = this.tmp();
      const b = this.tmp();
      this.emit(`${r} = call i32 @um_gpu_available()`);
      this.emit(`${b} = icmp ne i32 ${r}, 0`);
      return b;
    }
    if (e.callee === '__gpu_prefer') {
      const v = this.expr(e.args[0].expr);
      const w = this.tmp();
      const old = this.tmp();
      const b = this.tmp();
      this.emit(`${w} = zext i1 ${v} to i32`);
      this.emit(`${old} = load i32, ptr @um.gpu_prefer, align 4`);
      this.emit(`store i32 ${w}, ptr @um.gpu_prefer, align 4`);
      this.emit(`${b} = icmp ne i32 ${old}, 0`);
      return b;
    }
    if (e.callee === '__gpu_load' || e.callee === '__gpu_run') return this.gpuLoadCall(e);
    if (e.callee === '__gpu_commit') {
      if (this.m.gpu === 'metal') this.emit('call void @um_gpu_commit()');
      return '';
    }
    if (e.callee === 'print') {
      const a = e.args[0].expr;
      const at = a.ty!;
      const x = this.expr(a);
      if (at.k === 'bool') {
        const w = this.cast(x, at, { k: 'int', bits: 32, signed: false });
        this.emit(`call void @um_print_bool(i32 ${w})`);
      } else if (at.k === 'heap') {
        const [data, len] = this.strParts(a, x);
        this.emit(`call void @um_print_str(ptr ${data}, ${this.m.usz} ${len})`);
      } else if (at.k === 'float') {
        const w = this.cast(x, at, { k: 'float', bits: 64 });
        this.emit(`call void @um_print_f64(double ${w})`);
      } else if (at.k === 'int') {
        const w = this.cast(x, at, { k: 'int', bits: 64, signed: at.signed });
        this.emit(`call void @um_print_${at.signed ? 'i64' : 'u64'}(i64 ${w})`);
      }
      return '';
    }

    const sig = this.m.prog.fns.get(e.callee)!;
    return this.emitCall(sig, this.args(sig, e.args));
  }

  args(sig: FnSig, args: Arg[]): string[] {
    return this.prepArgs(sig, args).map((a) => a());
  }

  /** 引数を評価する。アドレスで渡す引数は、返した関数を呼んだ時点でアドレスを求める（prep を参照）。 */
  prepArgs(sig: FnSig, args: Arg[]): (() => string)[] {
    return sig.params.map((p, i) => {
      const a = args[i].expr;
      if (!passByPtr(p.mode, p.ty)) {
        const v = this.expr(a);
        return () => `${llTy(p.ty)} ${v}`;
      }
      if (p.mode === 'own') {
        // 所有権を渡す。受け取った側が解放するので、ここでは一時値として登録しない。
        const v = this.consume(a);
        const slot = this.alloca(p.ty);
        this.emit(`store ${llTy(p.ty)} ${v}, ptr ${slot}`);
        return () => `ptr ${slot}`;
      }
      // 借用。置き場所のある式はそのアドレスを貸し、そうでなければ一時領域に退避する。
      const pr = this.prep(a);
      return () => `ptr ${pr()}`;
    });
  }

  // ---- kernel の実行 ----

  /** kernel を添字 0 から count 未満について実行する（いまは CPU で順に実行）。 */
  dispatch(e: Expr & { kind: 'dispatch' }): string {
    const sig = this.m.prog.fns.get(e.kernel)!;
    const params = sig.params.slice(1);
    let n = this.expr(e.count);
    // dispatchGroups: count はグループの数。全体のスレッド数 n = グループの数 × 大きさ。
    let groups = '';
    let size = '';
    if (e.group) {
      groups = n;
      size = this.expr(e.group);
      const zero = this.tmp();
      const big = this.tmp();
      const badSize = this.tmp();
      this.emit(`${zero} = icmp eq i32 ${size}, 0`);
      this.emit(`${big} = icmp ugt i32 ${size}, ${MAX_GROUP_SIZE}`);
      this.emit(`${badSize} = or i1 ${zero}, ${big}`);
      this.check(badSize, PANIC_GROUP, e.pos.line);
      const g64 = this.tmp();
      const s64 = this.tmp();
      const t64 = this.tmp();
      const over = this.tmp();
      this.emit(`${g64} = zext i32 ${groups} to i64`);
      this.emit(`${s64} = zext i32 ${size} to i64`);
      this.emit(`${t64} = mul i64 ${g64}, ${s64}`);
      this.emit(`${over} = icmp ugt i64 ${t64}, 4294967295`);
      this.check(over, PANIC_GROUP, e.pos.line);
      n = this.tmp();
      this.emit(`${n} = trunc i64 ${t64} to i32`);
    }
    const args = this.prepArgs({ ...sig, params }, e.args).map((a) => a());

    // 書き込み先の Buffer は count 個以上の要素を持っていなければならない。
    // これにより、kernel の中の buf[i] が必ず範囲内になる。
    const usz = this.m.usz;
    let n64 = n;
    if (usz === 'i64') {
      n64 = this.tmp();
      this.emit(`${n64} = zext i32 ${n} to i64`);
    }
    params.forEach((p, i) => {
      if (p.mode !== 'mut') return;
      // RacyBuffer はどの要素にも書けるので、要素数と長さの関係を求めない（範囲外の添字は切り詰められる）。
      if (p.ty!.k === 'heap' && p.ty!.racy) return;
      const buf = args[i].slice('ptr '.length);
      const lp = this.tmp();
      const len = this.tmp();
      const short = this.tmp();
      this.emit(`${lp} = getelementptr inbounds %um.Array, ptr ${buf}, i32 0, i32 1`);
      this.emit(`${len} = load ${usz}, ptr ${lp}, !tbaa ${HDR}`);
      this.emit(`${short} = icmp ugt ${usz} ${n64}, ${len}`);
      this.check(short, PANIC_DISPATCH, e.pos.line);
    });

    // `gpu.submit`: Metal と wgpu では終わるのを待たずに戻り、券を返す。それ以外（GPU がない、WebGPU）では、
    // その場で最後まで実行して券 0 を返す（券 0 は「もう終わっている」）。
    const submit = e.async === true && (this.m.gpu === 'metal' || this.m.gpu === 'wgpu');
    const device = e.async && !submit ? 'auto' : e.device;
    let ticket = '';
    if (e.async) {
      ticket = this.alloca({ k: 'int', bits: 64, signed: false });
      this.emit(`store i64 0, ptr ${ticket}`);
    }
    const endL = this.newLabel('dispatch.end');
    // WebGPU（WGSL）には 64 ビットの整数がない。それを使う kernel は、自動の dispatch と gpu.submit では CPU で実行し、
    // GPU に固定した gpu.dispatch はエラーにする。
    const wgsl = this.m.gpu === 'webgpu' || this.m.gpu === 'wgpu';
    const no64 = wgsl && device !== 'cpu' && kernelUses64(this.m.prog, sig.decl);
    if (no64 && device === 'gpu' && !e.async) {
      throw new CompileError('kernel-type', `kernel '${e.kernel}' uses 64-bit integers, which WebGPU (WGSL) does not support`, e.pos,
        'WGSL has no 64-bit integers: 64-bit kernels run on the GPU only with Metal (UM_METAL=1 on macOS); use `dispatch` (runs on the CPU here) or `cpu.dispatch`');
    }
    // WGSL にできない kernel（simdgroup の集計を使う）は、wgpu では CPU の経路だけを作る。GPU を指定していればエラー。
    const k = this.m.gpu && device !== 'cpu' && !no64 ? this.m.gpuKernel(e.kernel, !!e.group) : null;
    if (this.m.gpu && device === 'gpu' && !e.async && !k && !no64) {
      throw new CompileError(WGSL_SIMDGROUP, `kernel '${e.kernel}' uses simdgroup functions, which WebGPU (WGSL) does not support`, e.pos,
        'simdgroup functions run on Metal and on the CPU; use `dispatch` / `dispatchGroups` (they run such a kernel on the CPU here) or `cpu.dispatchGroups`');
    }
    let pre: [string, number, string, number] | null = null;
    if (k) {
      // wasm32 では GPU（WebGPU）でも実行できる。auto は実行時にランタイムが選び、gpu は GPU がなければ停止する。
      const use = this.tmp();
      const yes = this.tmp();
      const gpuL = this.newLabel('dispatch.gpu');
      const cpuL = this.newLabel('dispatch.cpu');
      if (submit) {
        // GPU が使えなければ、下の CPU の経路でその場で実行する。
        this.emit(`${use} = call i32 @um_gpu_available()`);
        this.emit(`${yes} = icmp ne i32 ${use}, 0`);
        this.emit(`br i1 ${yes}, label %${gpuL}, label %${cpuL}`);
        this.terminated = true;
      } else if (device === 'gpu') {
        this.emit(`${use} = call i32 @um_gpu_available()`);
        const none = this.tmp();
        this.emit(`${none} = icmp eq i32 ${use}, 0`);
        this.check(none, PANIC_NO_GPU, e.pos.line);
        this.br(gpuL);
      } else {
        // 自動: `gpu.prefer(true)` なら GPU が使えるかぎり GPU。そうでなければランタイムが選ぶ。
        // wgpu では Buffer の表も渡し、Buffer の置き場所（GPU にあるか）を判断に入れる（runtime/wgpu.c の um_gpu_pick_bufs）。
        if (this.m.gpu === 'wgpu') pre = this.gpuArgs(params, args);
        const flag = this.tmp();
        const on = this.tmp();
        const prefL = this.newLabel('dispatch.prefer');
        const pickL = this.newLabel('dispatch.pick');
        this.emit(`${flag} = load i32, ptr @um.gpu_prefer, align 4`);
        this.emit(`${on} = icmp ne i32 ${flag}, 0`);
        this.emit(`br i1 ${on}, label %${prefL}, label %${pickL}`);
        this.terminated = true;
        this.label(prefL);
        const avail = this.tmp();
        this.emit(`${avail} = call i32 @um_gpu_available()`);
        const a = this.tmp();
        this.emit(`${a} = icmp ne i32 ${avail}, 0`);
        this.emit(`br i1 ${a}, label %${gpuL}, label %${cpuL}`);
        this.terminated = true;
        this.label(pickL);
        if (pre) this.emit(`${use} = call i32 @um_gpu_pick_bufs(i32 ${k.id}, i32 ${n}, ptr ${pre[2]}, i32 ${pre[3]})`);
        else this.emit(`${use} = call i32 @um_gpu_pick(i32 ${k.id}, i32 ${n})`);
        this.emit(`${yes} = icmp ne i32 ${use}, 0`);
        this.emit(`br i1 ${yes}, label %${gpuL}, label %${cpuL}`);
        this.terminated = true;
      }
      this.label(gpuL);
      const got = this.gpuDispatch(k, n, params, args, submit, size, pre);
      if (submit) this.emit(`store i64 ${got}, ptr ${ticket}`);
      // Metal は dispatch を積むだけなので、ここで提出して完了を待つ。次の文も dispatch なら待ちをそちらに任せる。
      // wgpu も同じ形で、um_gpu_flush は提出するだけで待たない（結果は CPU が読むときに読み戻す。runtime/wgpu.c）。
      else if ((this.m.gpu === 'metal' || this.m.gpu === 'wgpu') && !this.gpuDefer) this.emit('call void @um_gpu_flush()');
      this.br(endL);
      this.label(cpuL);
    }

    // CPU での実行。引数を1つの塊にまとめ、区間を実行する関数に渡す。
    // group.* を使う kernel は、グループを単位に実行する（区間はグループの番号。groupRangeFn）。
    const types = args.map((a) => a.slice(0, a.indexOf(' ')));
    const byGroup = !!e.group && sig.decl.usesGroup === true;
    const range = byGroup ? this.m.groupRangeFn(sig, types) : this.m.rangeFn(sig, types);
    const ctx = `%g${++this.n}`;
    this.allocas.push(`  ${ctx} = alloca ${range.ctxTy}`);
    const fields = byGroup ? [`i32 ${size}`, `i32 ${groups}`, ...args] : args;
    fields.forEach((a, i) => {
      const slot = this.tmp();
      this.emit(`${slot} = getelementptr inbounds ${range.ctxTy}, ptr ${ctx}, i32 0, i32 ${i}`);
      this.emit(`store ${a}, ptr ${slot}`);
    });
    const items = byGroup ? groups : n;
    if (this.m.target === 'native') {
      // ネイティブではランタイムが区間に分けて、複数のスレッドで実行する。
      this.emit(`call void @um_parallel_for(ptr ${range.fn}, ptr ${ctx}, i32 ${items})`);
    } else {
      this.emit(`call void ${range.fn}(i32 0, i32 ${items}, ptr ${ctx})`);
    }
    if ((this.m.gpu === 'metal' || (this.m.gpu === 'wgpu' && k)) && e.device === 'auto') {
      // 自動判断のために、CPU で実行し終えたことをランタイムに知らせる（かかった時間を測って次の判断に使う）。
      this.emit(`call void @um_gpu_cpu_done(i32 ${this.m.gpuKernel(e.kernel, !!e.group)!.id}, i32 ${n})`);
    }
    this.br(endL);
    this.label(endL);
    if (e.async) {
      const t = this.tmp();
      this.emit(`${t} = load i64, ptr ${ticket}`);
      return t;
    }
    return '';
  }

  // ---- SIMD（i8x16 / i16x8 / i32x4 / f32x4 と符号なしの版。値は固定長配列 [L x T]、演算は LLVM のベクトル <L x T>） ----

  simdTy(t: Ty): { vt: string; at: string; et: string; lanes: number; align: number; elem: Ty & { k: 'int' | 'float' } } {
    const info = simdInfo(t)!;
    const et = llTy(info.elem);
    return { vt: `<${info.lanes} x ${et}>`, at: `[${info.lanes} x ${et}]`, et, lanes: info.lanes, align: info.elem.bits / 8, elem: info.elem };
  }

  /**
   * 式を LLVM のベクトルの値として求める。SIMD の演算の結果はそのまま、変数などの置き場所はベクトルとして読み、
   * それ以外（配列リテラル、関数の戻り値）は一時領域を通して読み替える（最適化で消える）。
   */
  simdVec(e: Expr): string {
    if (e.kind === 'math' && e.fn.startsWith('simd.') && e.fn !== 'simd.sum' && e.fn !== 'simd.store' && e.fn !== 'simd.load') return this.simdOp(e);
    const { vt, at, align } = this.simdTy(e.ty!);
    const v = this.tmp();
    if (e.kind === 'math' && e.fn === 'simd.load') {
      this.emit(`${v} = load ${vt}, ptr ${this.simdAddr(e)}, align ${align}`);
      return v;
    }
    if (isPlace(e)) {
      this.emit(`${v} = load ${vt}, ptr ${this.addr(e)}, align ${align}`);
      return v;
    }
    const agg = this.expr(e);
    const slot = `%a${++this.n}`;
    this.allocas.push(`  ${slot} = alloca ${at}, align 16`);
    this.emit(`store ${at} ${agg}, ptr ${slot}`);
    this.emit(`${v} = load ${vt}, ptr ${slot}, align 16`);
    return v;
  }

  /** ベクトルの値を固定長配列の値（[L x T]）にする。 */
  simdAgg(v: string, t: Ty): string {
    const { vt, at } = this.simdTy(t);
    const slot = `%a${++this.n}`;
    this.allocas.push(`  ${slot} = alloca ${at}, align 16`);
    this.emit(`store ${vt} ${v}, ptr ${slot}, align 16`);
    const r = this.tmp();
    this.emit(`${r} = load ${at}, ptr ${slot}`);
    return r;
  }

  /** Simd.load / Simd.store の先頭の要素の番地。i から L 個が範囲内でなければ停止する。 */
  simdAddr(e: Expr & { kind: 'math' }): string {
    const arr = e.args[0];
    const at = arr.ty as Ty & ({ k: 'heap' } | { k: 'array' });
    const base = this.prep(arr);
    const it = e.args[1].ty as Ty & { k: 'int' };
    let idx = this.expr(e.args[1]);
    if (it.bits < 64) {
      const w = this.tmp();
      this.emit(`${w} = ${it.signed ? 'sext' : 'zext'} i${it.bits} ${idx} to i64`);
      idx = w;
    }
    const value = e.fn === 'simd.store' ? this.simdVec(e.args[2]) : '';
    const lanes = 128 / ((at.elem as { bits: number }).bits);
    const et = llTy(at.elem);
    const b = base();
    let len: string;
    if (at.k === 'heap') {
      const lp = this.tmp();
      const l = this.tmp();
      this.emit(`${lp} = getelementptr inbounds %um.Array, ptr ${b}, i32 0, i32 1`);
      this.emit(`${l} = load ${this.m.usz}, ptr ${lp}, !tbaa ${HDR}`);
      len = l;
      if (this.m.usz !== 'i64') {
        len = this.tmp();
        this.emit(`${len} = zext ${this.m.usz} ${l} to i64`);
      }
    } else len = String(at.len);
    // 負の添字は符号なしで見ると大きな値になるので、「長さ - L より大きい」に含まれる。
    const short = this.tmp();
    const room = this.tmp();
    const past = this.tmp();
    const bad = this.tmp();
    this.emit(`${short} = icmp ult i64 ${len}, ${lanes}`);
    this.emit(`${room} = sub i64 ${len}, ${lanes}`);
    this.emit(`${past} = icmp ugt i64 ${idx}, ${room}`);
    this.emit(`${bad} = or i1 ${short}, ${past}`);
    this.check(bad, PANIC_INDEX, e.pos.line);
    const p = this.tmp();
    if (at.k === 'heap') {
      const data = this.tmp();
      this.emit(`${data} = load ptr, ptr ${b}, !tbaa ${HDR}`);
      this.emit(`${p} = getelementptr inbounds ${et}, ptr ${data}, i64 ${idx}`);
    } else {
      this.emit(`${p} = getelementptr inbounds ${llTy(at)}, ptr ${b}, i64 0, i64 ${idx}`);
    }
    if (e.fn === 'simd.store') {
      this.emit(`store <${lanes} x ${et}> ${value}, ptr ${p}, align ${(at.elem as { bits: number }).bits / 8}`);
    }
    return p;
  }

  simd(e: Expr & { kind: 'math' }): string {
    if (e.fn === 'simd.store') {
      this.simdAddr(e);
      return '';
    }
    if (e.fn === 'simd.sum') {
      const { vt, et, lanes, elem } = this.simdTy(e.args[0].ty!);
      const v = this.simdVec(e.args[0]);
      const r = this.tmp();
      if (elem.k === 'float') {
        // 左から順に足す（-0.0 から始める順序つきの和。並べ替えないので、どのターゲットでも同じ値）。
        this.m.intrinsics.add(`declare float @llvm.vector.reduce.fadd.v${lanes}f32(float, ${vt})`);
        this.emit(`${r} = call float @llvm.vector.reduce.fadd.v${lanes}f32(float 0x8000000000000000, ${vt} ${v})`);
      } else {
        this.m.intrinsics.add(`declare ${et} @llvm.vector.reduce.add.v${lanes}${et}(${vt})`);
        this.emit(`${r} = call ${et} @llvm.vector.reduce.add.v${lanes}${et}(${vt} ${v})`);
      }
      return r;
    }
    return this.simdAgg(this.simdVec(e), e.ty!);
  }

  /** SIMD の演算（結果はベクトル）。 */
  simdOp(e: Expr & { kind: 'math' }): string {
    const op = e.fn.slice('simd.'.length);
    const r = this.tmp();
    if (op === 'splat') {
      const { vt, et, lanes } = this.simdTy(e.ty!);
      const x = this.expr(e.args[0]);
      const one = this.tmp();
      this.emit(`${one} = insertelement ${vt} poison, ${et} ${x}, i32 0`);
      this.emit(`${r} = shufflevector ${vt} ${one}, ${vt} poison, <${lanes} x i32> zeroinitializer`);
      return r;
    }
    if (op === 'dot' || op === 'mulAddWide') {
      const acc = this.simdVec(e.args[0]);
      const a = this.simdVec(e.args[1]);
      const b = this.simdVec(e.args[2]);
      if (op === 'dot' && this.m.appleArm) {
        // NEON の sdot（4 つずつの積の和を 32 ビットのレーンに足す）。下の汎用の式と同じ値。
        this.m.intrinsics.add('declare <4 x i32> @llvm.aarch64.neon.sdot.v4i32.v16i8(<4 x i32>, <16 x i8>, <16 x i8>)');
        this.emit(`${r} = call <4 x i32> @llvm.aarch64.neon.sdot.v4i32.v16i8(<4 x i32> ${acc}, <16 x i8> ${a}, <16 x i8> ${b})`);
        return r;
      }
      const n = op === 'dot' ? 16 : 8;
      const src = op === 'dot' ? 'i8' : 'i16';
      const group = n / 4;
      const sa = this.tmp();
      const sb = this.tmp();
      const m = this.tmp();
      this.emit(`${sa} = sext <${n} x ${src}> ${a} to <${n} x i32>`);
      this.emit(`${sb} = sext <${n} x ${src}> ${b} to <${n} x i32>`);
      // 積は i32 に収まる（i16 × i16 の最大 2^30）。和は i32 でラップする。
      this.emit(`${m} = mul <${n} x i32> ${sa}, ${sb}`);
      let sum = acc;
      for (let j = 0; j < group; j++) {
        const part = this.tmp();
        const lanes = [0, 1, 2, 3].map((k) => `i32 ${k * group + j}`).join(', ');
        this.emit(`${part} = shufflevector <${n} x i32> ${m}, <${n} x i32> poison, <4 x i32> <${lanes}>`);
        const next = this.tmp();
        this.emit(`${next} = add <4 x i32> ${sum}, ${part}`);
        sum = next;
      }
      return sum;
    }
    const { vt, elem, lanes } = this.simdTy(e.ty!);
    const a = this.simdVec(e.args[0]);
    const b = this.simdVec(e.args[1]);
    const isF = elem.k === 'float';
    const signed = elem.k === 'int' && elem.signed;
    switch (op) {
      case 'add':
      case 'sub':
      case 'mul':
        this.emit(`${r} = ${isF ? 'f' : ''}${op} ${vt} ${a}, ${b}`);
        return r;
      case 'min':
      case 'max': {
        // Math.min / Math.max と同じ「比較して選ぶ」定義（浮動小数点で NaN があれば 2 番目）。
        const c = this.tmp();
        const pred = isF ? (op === 'min' ? 'olt' : 'ogt') : `${signed ? 's' : 'u'}${op === 'min' ? 'lt' : 'gt'}`;
        this.emit(`${c} = ${isF ? 'fcmp' : 'icmp'} ${pred} ${vt} ${a}, ${b}`);
        this.emit(`${r} = select <${lanes} x i1> ${c}, ${vt} ${a}, ${vt} ${b}`);
        return r;
      }
      case 'shuffle': {
        const idx = (e as unknown as { lanes: number[] }).lanes.map((k) => `i32 ${k}`).join(', ');
        this.emit(`${r} = shufflevector ${vt} ${a}, ${vt} ${b}, <${lanes} x i32> <${idx}>`);
        return r;
      }
    }
    throw new Error(`internal: simd ${op}`);
  }

  /**
   * Math.prefetch(arr, i): arr[i] のあるメモリを先読みするよう頼む（llvm.prefetch。読むため、局所性は最大）。
   * 先読みはプログラムの動作を変えないので、範囲外でもよい（inbounds を付けない番地にする。検査もしない）。
   * wasm32 には先読みがないので、添字の式を評価するだけ。
   */
  prefetch(e: Expr & { kind: 'math' }): string {
    const at = e.args[0].ty as Ty & { k: 'heap' };
    const base = this.prep(e.args[0]);
    const it = e.args[1].ty as Ty & { k: 'int' };
    let idx = this.expr(e.args[1]);
    // 配列そのものの場所（`tables[k]` なら k の検査を含む）は、どのターゲットでも同じに求める。
    const b = base();
    if (this.m.target !== 'native') return '';
    if (it.bits < 64) {
      const w = this.tmp();
      this.emit(`${w} = ${it.signed ? 'sext' : 'zext'} i${it.bits} ${idx} to i64`);
      idx = w;
    }
    const data = this.tmp();
    const p = this.tmp();
    this.emit(`${data} = load ptr, ptr ${b}, !tbaa ${HDR}`);
    this.emit(`${p} = getelementptr ${llTy(at.elem)}, ptr ${data}, i64 ${idx}`);
    this.m.intrinsics.add('declare void @llvm.prefetch.p0(ptr, i32, i32, i32)');
    this.emit(`call void @llvm.prefetch.p0(ptr ${p}, i32 0, i32 3, i32 1)`);
    return '';
  }

  /**
   * `gpu.loadKernel(wgsl)` / `gpu.runKernel(k, count, ...)`（src/gpuload.ts）。wgpu の経路だけがランタイムを呼ぶ。
   * ほかのビルドでも引数は評価し、loadKernel は null、runKernel は false を返す。
   */
  gpuLoadCall(e: Expr & { kind: 'call' }): string {
    const wgpu = this.m.gpu === 'wgpu';
    if (wgpu) this.m.usesGpuRuntime = true;
    if (e.callee === '__gpu_load') {
      const [data, len] = this.heapParts(e.args[0].expr)();
      const id = this.tmp();
      if (wgpu) this.emit(`${id} = call i32 @um_gpu_load(ptr ${data}, ${this.m.usz} ${len})`);
      const v = wgpu ? id : '0';
      const t = llTy(e.ty!);
      const some = this.tmp();
      const a = this.tmp();
      const b = this.tmp();
      this.emit(`${some} = icmp ne i32 ${v}, 0`);
      this.emit(`${a} = insertvalue ${t} undef, i1 ${some}, 0`);
      this.emit(`${b} = insertvalue ${t} ${a}, i32 ${v}, 1`);
      return b;
    }
    const rest = e.args.slice(2);
    const params: FnSig['params'] = rest.map((a, i) => ({ name: `a${i}`, mode: a.mode, ty: a.expr.ty! }));
    const kernel = this.expr(e.args[0].expr);
    const count = this.expr(e.args[1].expr);
    const args = this.prepArgs({ key: '', name: '', params, ret: { k: 'void' }, exported: false, decl: this.d }, rest).map((a) => a());
    if (!wgpu) return 'false';
    // 値の塊と Buffer の表は dispatch と同じ形（gpuDispatch）。番号の代わりに読み込んだ kernel の番号を渡す。
    const [block, bytes, table, nbufs] = this.gpuArgs(params, args);
    const ok = this.tmp();
    const r = this.tmp();
    this.emit(`${ok} = call i32 @um_gpu_run(i32 ${kernel}, i32 ${count}, ptr ${block}, i32 ${bytes}, ptr ${table}, i32 ${nbufs})`);
    this.emit(`${r} = icmp ne i32 ${ok}, 0`);
    return r;
  }

  /**
   * GPU での実行をランタイムに頼む。値の引数は GPU 側の struct と同じ並び（gpuLayout）の塊にし、
   * Buffer は（データの場所、要素数、要素の大きさ、書き込みの有無）の表にして渡す。
   */
  gpuDispatch(k: { id: number; global: string; bytes: number }, n: string, params: FnSig['params'], args: string[], submit = false, group = '',
    pre: [string, number, string, number] | null = null): string {
    const [block, bytes, table, nbufs] = pre ?? this.gpuArgs(params, args);
    const call = `(i32 ${k.id}, ptr ${k.global}, i32 ${k.bytes}, i32 ${n}, ptr ${block}, i32 ${bytes}, ptr ${table}, i32 ${nbufs})`;
    // dispatchGroups: グループの大きさを、すぐ次の dispatch / submit に渡す（ランタイムがスレッドごとに覚えて、使ったら消す）。
    if (group) this.emit(`call void @um_gpu_group(i32 ${group})`);
    if (submit) {
      const t = this.tmp();
      this.emit(`${t} = call i64 @um_gpu_submit${call}`);
      return t;
    }
    this.emit(`call void @um_gpu_dispatch${call}`);
    return '';
  }

  /** GPU に渡す引数の塊（値の引数）と Buffer の表を作り、[塊, 塊のバイト数, 表, Buffer の数] を返す。 */
  gpuArgs(params: FnSig['params'], args: string[]): [string, number, string, number] {
    const prog = this.m.prog;
    const values = params.map((p, i) => ({ p, arg: args[i] })).filter((x) => x.p.ty.k !== 'heap');
    const buffers = params.map((p, i) => ({ p, arg: args[i] })).filter((x) => x.p.ty.k === 'heap');
    // 値の塊は、ランタイムが (2 + Buffer の数) 語の見出し（count、row、各 Buffer の長さ）のすぐ後ろに写す。
    // GPU 側の引数の struct（UmParams）と同じ並びにするため、各値の位置を見出しを含めた先頭からの境界に揃える
    // （64 ビットの整数は 8 バイト境界）。末尾も struct 全体の境界まで詰める。
    const head = (2 + buffers.length) * 4;
    let pos = head;
    let maxAlign = 4;
    const offsets = values.map(({ p }) => {
      const l = gpuLayout(prog, p.ty);
      pos = Math.ceil(pos / l.align) * l.align;
      const at = pos - head;
      pos += l.size;
      maxAlign = Math.max(maxAlign, l.align);
      return at;
    });
    pos = Math.ceil(pos / maxAlign) * maxAlign;
    const bytes = pos - head;

    const block = `%g${++this.n}`;
    this.allocas.push(`  ${block} = alloca [${Math.max(bytes / 4, 1)} x i32], align 8`);
    if (bytes > values.reduce((sum, x) => sum + gpuLayout(prog, x.p.ty).size, 0)) {
      // 詰め物の場所も決まった値にする（GPU には写すだけだが、未初期化のメモリを渡さない）。
      this.m.intrinsics.add('declare void @llvm.memset.p0.i32(ptr, i8, i32, i1)');
      this.emit(`call void @llvm.memset.p0.i32(ptr ${block}, i8 0, i32 ${bytes}, i1 false)`);
    }
    values.forEach(({ p, arg }, vi) => {
      const slot = this.tmp();
      this.emit(`${slot} = getelementptr inbounds i8, ptr ${block}, i32 ${offsets[vi]}`);
      const size = gpuLayout(prog, p.ty).size;
      if (arg.startsWith('ptr ')) {
        // struct と固定長配列は、CPU 側の並びのまま写す（GPU 側と同じ並び。gpuLayout）。
        this.m.intrinsics.add('declare void @llvm.memcpy.p0.p0.i32(ptr, ptr, i32, i1)');
        this.emit(`call void @llvm.memcpy.p0.p0.i32(ptr ${slot}, ${arg}, i32 ${size}, i1 false)`);
      } else if (p.ty.k === 'bool') {
        const w = this.tmp();
        this.emit(`${w} = zext ${arg} to i32`);
        this.emit(`store i32 ${w}, ptr ${slot}`);
      } else {
        this.emit(`store ${arg}, ptr ${slot}`);
      }
    });

    // Buffer の表: 1つにつき4語（データの場所、長さ、1要素のバイト数、書き込むかどうか）。
    // 1語はポインタの幅（wasm32 では 32 ビット、ネイティブでは 64 ビット）。
    const W = this.m.usz;
    const table = `%g${++this.n}`;
    this.allocas.push(`  ${table} = alloca [${Math.max(buffers.length, 1) * 4} x ${W}]`);
    buffers.forEach(({ p, arg }, i) => {
      const buf = arg.slice('ptr '.length);
      const data = this.tmp();
      const addr = this.tmp();
      const lp = this.tmp();
      const len = this.tmp();
      this.emit(`${data} = load ptr, ptr ${buf}, !tbaa ${HDR}`);
      this.emit(`${addr} = ptrtoint ptr ${data} to ${W}`);
      this.emit(`${lp} = getelementptr inbounds %um.Array, ptr ${buf}, i32 0, i32 1`);
      this.emit(`${len} = load ${W}, ptr ${lp}, !tbaa ${HDR}`);
      const et = (p.ty as Ty & { k: 'heap' }).elem;
      const elemBytes = et.k === 'int' && et.bits < 32 ? et.bits / 8 : gpuLayout(prog, et).size;
      // 4語目: 0 = 読むだけ、1 = 自分の添字の要素にだけ書く、2 = どの要素にも書きうる（RacyBuffer）。
      const racy = p.ty.k === 'heap' && p.ty.racy === true;
      [addr, len, String(elemBytes), p.mode === 'mut' ? (racy ? '2' : '1') : '0'].forEach((v, j) => {
        const slot = this.tmp();
        this.emit(`${slot} = getelementptr inbounds ${W}, ptr ${table}, i32 ${i * 4 + j}`);
        this.emit(`store ${W} ${v}, ptr ${slot}`);
      });
    });

    return [block, bytes, table, buffers.length];
  }

  math(e: Expr & { kind: 'math' }): string {
    const ty = e.ty!;
    if (e.fn.startsWith('group.')) return this.groupMath(e);
    if (e.fn === 'prefetch') return this.prefetch(e);
    if (e.fn.startsWith('simd.')) return this.simd(e);
    if (isSmall(ty) && (e.fn === 'min' || e.fn === 'max')) {
      const a = this.cast(this.expr(e.args[0]), ty, F32T);
      const b = this.cast(this.expr(e.args[1]), ty, F32T);
      const c = this.tmp();
      const w = this.tmp();
      this.emit(`${c} = fcmp ${e.fn === 'min' ? 'olt' : 'ogt'} float ${a}, ${b}`);
      this.emit(`${w} = select i1 ${c}, float ${a}, float ${b}`);
      return this.cast(w, F32T, ty);
    }
    if (ty.k === 'float' && ty.bits <= 16 && e.fn !== 'min' && e.fn !== 'max') {
      // f16 は f32 で計算してから丸め直す（binary の説明を参照）。
      const x = this.cast(this.expr(e.args[0]), ty, F32T);
      const w = this.tmp();
      const name = `llvm.${e.fn === 'abs' ? 'fabs' : e.fn}.f32`;
      this.m.intrinsics.add(`declare float @${name}(float)`);
      this.emit(`${w} = call float @${name}(float ${x})`);
      return this.cast(w, F32T, ty);
    }
    const t = llTy(ty);
    if ((e.fn === 'shl' || e.fn === 'shr') && ty.k === 'int') {
      // マスクつきシフト: シフト量を値のビット幅で割った余りにする（範囲検査なし）。kernel の `<<` / `>>` と同じ意味。
      const val = this.expr(e.args[0]);
      const amtRaw = this.expr(e.args[1]);
      const amt = this.cast(amtRaw, e.args[1].ty!, ty); // 量を値の型に合わせる（量が大きくても、次のマスクで範囲に収まる）
      const masked = this.tmp();
      this.emit(`${masked} = and ${t} ${amt}, ${ty.bits - 1}`);
      const v = this.tmp();
      const ins = e.fn === 'shl' ? 'shl' : ty.signed ? 'ashr' : 'lshr';
      this.emit(`${v} = ${ins} ${t} ${val}, ${masked}`);
      return v;
    }
    const xs = e.args.map((a) => this.expr(a));
    const v = this.tmp();
    const call = (name: string, extra = '') => {
      this.m.intrinsics.add(`declare ${t} @${name}(${xs.map(() => t).join(', ')}${extra ? ', i1' : ''})`);
      this.emit(`${v} = call ${t} @${name}(${xs.map((x) => `${t} ${x}`).join(', ')}${extra})`);
      return v;
    };

    if (ty.k === 'float') {
      const f = `f${ty.bits}`;
      switch (e.fn) {
        case 'sqrt':
          return call(`llvm.sqrt.${f}`);
        case 'floor':
          return call(`llvm.floor.${f}`);
        case 'ceil':
          return call(`llvm.ceil.${f}`);
        case 'abs':
          return call(`llvm.fabs.${f}`);
        case 'min':
        case 'max': {
          // 比較と選択で定義する。どちらかが NaN なら2番目の引数を返す。
          const c = this.tmp();
          this.emit(`${c} = fcmp ${e.fn === 'min' ? 'olt' : 'ogt'} ${t} ${xs[0]}, ${xs[1]}`);
          this.emit(`${v} = select i1 ${c}, ${t} ${xs[0]}, ${t} ${xs[1]}`);
          return v;
        }
      }
    } else if (ty.k === 'int') {
      const s = ty.signed ? 's' : 'u';
      switch (e.fn) {
        case 'abs':
          // 最小値の絶対値はラップして最小値のまま（未定義動作にしない）。符号なしはそのまま。
          return ty.signed ? call(`llvm.abs.${t}`, ', i1 false') : xs[0];
        case 'min':
          return call(`llvm.${s}min.${t}`);
        case 'max':
          return call(`llvm.${s}max.${t}`);
        case 'popcount':
          return call(`llvm.ctpop.${t}`);
        // 第2引数の false は「値が 0 でも結果を定義する（ビット幅を返す）」の指定。
        case 'clz':
          return call(`llvm.ctlz.${t}`, ', i1 false');
        case 'ctz':
          return call(`llvm.cttz.${t}`, ', i1 false');
      }
    }
    throw new Error(`internal: Math.${e.fn} on ${tyStr(ty)}`);
  }

  // ---- 可変長配列 ----

  /**
   * Buffer の要素を確保する（ゼロ埋め）。長さ 0 でも1要素ぶん確保する。
   * kernel の中の添字は停止できないので、切り詰めた先が常に有効な場所になるようにするため。
   */
  allocBuffer(n: string, elem: Ty): string {
    const usz = this.m.usz;
    const empty = this.tmp();
    const count = this.tmp();
    const data = this.tmp();
    this.emit(`${empty} = icmp eq ${usz} ${n}, 0`);
    this.emit(`${count} = select i1 ${empty}, ${usz} 1, ${usz} ${n}`);
    if (elem.k === 'int' && elem.bits < 32) {
      // 8 / 16 ビットの要素は、確保を 4 バイトの倍数に切り上げる。GPU は 4 バイト単位で読み書きする
      // （WGSL では 1 語に詰めて扱う）ので、最後の語が確保の外にはみ出さないようにするため。
      const per = 32 / elem.bits - 1;
      const up = this.tmp();
      const rounded = this.tmp();
      this.emit(`${up} = add ${usz} ${count}, ${per}`);
      this.emit(`${rounded} = and ${usz} ${up}, ${-(per + 1)}`);
      // 切り上げがあふれたら（wasm32 の 4 GiB 近く）、そのままの数で確保を頼む（確保の側が大きすぎると止める）。
      const wrap = this.tmp();
      const total = this.tmp();
      this.emit(`${wrap} = icmp ult ${usz} ${rounded}, ${count}`);
      this.emit(`${total} = select i1 ${wrap}, ${usz} ${count}, ${usz} ${rounded}`);
      this.emit(`${data} = call ptr @${this.bufferAlloc()}(${usz} ${total}, ${usz} ${this.m.sizeOf(elem)})`);
      return data;
    }
    const fn = this.bufferAlloc();
    this.emit(`${data} = call ptr @${fn}(${usz} ${count}, ${usz} ${this.m.sizeOf(elem)})`);
    return data;
  }

  /**
   * Buffer の領域を確保するランタイムの関数。Metal では、GPU とコピーなしで共有できるよう、ページ境界に揃えた領域を使う。
   * wgpu では、GPU 側の写しとの同期をページの保護で追うための領域（runtime/wgpu.c の um_wgpu_alloc）。
   */
  bufferAlloc(): string {
    return this.m.gpu === 'metal' ? 'um_alloc_shared' : this.m.gpu === 'wgpu' ? 'um_wgpu_alloc' : 'um_alloc_zeroed';
  }

  arrayValue(data: string, len: string, cap: string): string {
    const usz = this.m.usz;
    const a = this.tmp();
    const b = this.tmp();
    const c = this.tmp();
    this.emit(`${a} = insertvalue %um.Array undef, ptr ${data}, 0`);
    this.emit(`${b} = insertvalue %um.Array ${a}, ${usz} ${len}, 1`);
    this.emit(`${c} = insertvalue %um.Array ${b}, ${usz} ${cap}, 2`);
    return c;
  }

  /** 整数を usize に変換する。収まらない値（負の数を含む）は確保できないので停止する。 */
  toUsize(x: string, from: Ty & { k: 'int' }, line: number): string {
    const usz = this.m.usz;
    const bits = this.m.prog.ptrBits;
    if (from.signed) {
      const neg = this.tmp();
      this.emit(`${neg} = icmp slt i${from.bits} ${x}, 0`);
      this.check(neg, PANIC_OOM, line);
    }
    if (from.bits === bits) return x;
    const v = this.tmp();
    if (from.bits < bits) {
      this.emit(`${v} = zext i${from.bits} ${x} to ${usz}`);
      return v;
    }
    const big = this.tmp();
    this.emit(`${big} = icmp ugt i${from.bits} ${x}, ${(1n << BigInt(bits)) - 1n}`);
    this.check(big, PANIC_OOM, line);
    this.emit(`${v} = trunc i${from.bits} ${x} to ${usz}`);
    return v;
  }

  /**
   * xs.fill(v) / xs.fill(v, start, end): [start, end) の要素をすべて v にする。範囲は 0 <= start <= end <= 長さ
   * でなければ停止する。書き込みはランタイムの um_fill（ネイティブでは大きければ複数のスレッドで書く）。
   * RacyBuffer では、要素ごとに不可分な（relaxed の）書き込みにする（同時に読むスレッドに、ちぎれた値は見えない）。
   */
  fill(e: Expr & { kind: 'mcall' }, recv: () => string): string {
    const usz = this.m.usz;
    const ht = e.recv.ty as Ty & { k: 'heap' };
    const elem = ht.elem;
    const v = this.expr(e.args[0].expr);
    const wide = (x: string, t: Ty & { k: 'int' }): string => {
      if (t.bits === 64) return x;
      const w = this.tmp();
      this.emit(`${w} = ${t.signed ? 'sext' : 'zext'} i${t.bits} ${x} to i64`);
      return w;
    };
    const range = e.args.slice(1).map((a) => ({ x: wide(this.expr(a.expr), a.expr.ty as Ty & { k: 'int' }), t: a.expr.ty as Ty & { k: 'int' } }));
    const arr = recv();
    const lenP = this.tmp();
    const len = this.tmp();
    const data = this.tmp();
    this.emit(`${lenP} = getelementptr inbounds %um.Array, ptr ${arr}, i32 0, i32 1`);
    this.emit(`${len} = load ${usz}, ptr ${lenP}, !tbaa ${HDR}`);
    this.emit(`${data} = load ptr, ptr ${arr}, !tbaa ${HDR}`);
    let start = `${usz} 0`;
    let end = `${usz} ${len}`;
    if (range.length === 2) {
      // 64 ビットで比べる（wasm32 の長さは 32 ビット）。符号つきの負の値は、符号なしで見ると長さより大きい。
      let len64 = len;
      if (usz !== 'i64') {
        len64 = this.tmp();
        this.emit(`${len64} = zext ${usz} ${len} to i64`);
      }
      const neg = range.map((r) => {
        if (!r.t.signed) return 'false';
        const n = this.tmp();
        this.emit(`${n} = icmp slt i64 ${r.x}, 0`);
        return n;
      });
      const a = this.tmp();
      const b = this.tmp();
      const c = this.tmp();
      const d = this.tmp();
      const f = this.tmp();
      this.emit(`${a} = icmp ugt i64 ${range[0].x}, ${range[1].x}`);
      this.emit(`${b} = icmp ugt i64 ${range[1].x}, ${len64}`);
      this.emit(`${c} = or i1 ${a}, ${b}`);
      this.emit(`${d} = or i1 ${c}, ${neg[0]}`);
      this.emit(`${f} = or i1 ${d}, ${neg[1]}`);
      this.check(f, PANIC_INDEX, e.pos.line);
      const [s0, e0] = range.map((r) => {
        if (usz === 'i64') return r.x;
        const t = this.tmp();
        this.emit(`${t} = trunc i64 ${r.x} to ${usz}`);
        return t;
      });
      start = `${usz} ${s0}`;
      end = `${usz} ${e0}`;
    }
    // 値のビット列を i64 にする（下位の size バイトだけを使う）。
    const lt = llTy(elem);
    const bytes = elem.k === 'bool' ? 1 : elem.k === 'float' && elem.bits < 16 ? 1 : (elem as { bits: number }).bits / 8;
    let bits = v;
    if (elem.k === 'float' && elem.bits >= 16) {
      const ib = this.tmp();
      this.emit(`${ib} = bitcast ${lt} ${v} to i${elem.bits}`);
      bits = ib;
    }
    const intTy = elem.k === 'bool' ? 'i1' : elem.k === 'float' && elem.bits < 16 ? 'i8' : `i${(elem as { bits: number }).bits}`;
    if (intTy !== 'i64') {
      const z = this.tmp();
      this.emit(`${z} = zext ${intTy} ${bits} to i64`);
      bits = z;
    }
    this.emit(`call void @um_fill(ptr ${data}, ${start}, ${end}, i32 ${bytes}, i64 ${bits}, i32 ${ht.racy ? 1 : 0})`);
    return '';
  }

  arrayMethod(e: Expr & { kind: 'mcall' }): string {
    if (e.key?.startsWith('Racy.')) return this.racyAtomic(e);
    const usz = this.m.usz;
    const elem = (e.recv.ty as Ty & { k: 'heap' }).elem;
    const et = llTy(elem);
    const recv = this.prep(e.recv);
    // 引数を先に評価し、その後で配列のアドレスを求める。
    const arg = e.name === 'push' ? this.consume(e.args[0].expr) : '';
    if (e.name === 'fill') return this.fill(e, recv);
    const arr = recv();

    const lenP = this.tmp();
    const len = this.tmp();
    this.emit(`${lenP} = getelementptr inbounds %um.Array, ptr ${arr}, i32 0, i32 1`);
    this.emit(`${len} = load ${usz}, ptr ${lenP}, !tbaa ${HDR}`);

    switch (e.name) {
      case 'push': {
        const capP = this.tmp();
        const cap = this.tmp();
        const full = this.tmp();
        const growL = this.newLabel('grow');
        const roomL = this.newLabel('room');
        this.emit(`${capP} = getelementptr inbounds %um.Array, ptr ${arr}, i32 0, i32 2`);
        this.emit(`${cap} = load ${usz}, ptr ${capP}, !tbaa ${HDR}`);
        this.emit(`${full} = icmp eq ${usz} ${len}, ${cap}`);
        this.emit(`br i1 ${full}, label %${growL}, label %${roomL}`);
        this.label(growL);
        // 配列の構造体のアドレスはランタイムに渡さない（渡すと長さや容量をレジスタに置けなくなる）。
        const old = this.tmp();
        const newCap = this.tmp();
        const grown = this.tmp();
        this.emit(`${old} = load ptr, ptr ${arr}, !tbaa ${HDR}`);
        this.emit(`${newCap} = call ${usz} @um_grow_cap(${usz} ${cap})`);
        this.emit(`${grown} = call ptr @um_grow_data(ptr ${old}, ${usz} ${len}, ${usz} ${newCap}, ${usz} ${this.m.sizeOf(elem)})`);
        this.emit(`store ptr ${grown}, ptr ${arr}, !tbaa ${HDR}`);
        this.emit(`store ${usz} ${newCap}, ptr ${capP}, !tbaa ${HDR}`);
        this.br(roomL);
        this.label(roomL);
        const data = this.tmp();
        const p = this.tmp();
        const next = this.tmp();
        this.emit(`${data} = load ptr, ptr ${arr}, !tbaa ${HDR}`);
        this.emit(`${p} = getelementptr inbounds ${et}, ptr ${data}, ${usz} ${len}`);
        this.emit(`store ${et} ${arg}, ptr ${p}${isScalar(elem) ? `, !tbaa ${VAL}` : ''}`);
        this.emit(`${next} = add ${usz} ${len}, 1`);
        this.emit(`store ${usz} ${next}, ptr ${lenP}, !tbaa ${HDR}`);
        return '';
      }

      case 'pop': {
        const empty = this.tmp();
        this.emit(`${empty} = icmp eq ${usz} ${len}, 0`);
        this.check(empty, PANIC_POP_EMPTY, e.pos.line);
        const last = this.tmp();
        const data = this.tmp();
        const p = this.tmp();
        const v = this.tmp();
        this.emit(`${last} = sub ${usz} ${len}, 1`);
        this.emit(`store ${usz} ${last}, ptr ${lenP}, !tbaa ${HDR}`);
        this.emit(`${data} = load ptr, ptr ${arr}, !tbaa ${HDR}`);
        this.emit(`${p} = getelementptr inbounds ${et}, ptr ${data}, ${usz} ${last}`);
        this.emit(`${v} = load ${et}, ptr ${p}`);
        return v;
      }

      case 'clear':
        // 要素がヒープを持つなら、先にそれぞれを解放する。
        if (!this.m.isCopy(elem)) this.emit(`call void ${this.m.dropElems(elem)}(ptr ${arr})`);
        this.emit(`store ${usz} 0, ptr ${lenP}, !tbaa ${HDR}`);
        return '';

      case 'clone': {
        const size = this.m.sizeOf(elem);
        const data = this.tmp();
        let copy = this.tmp();
        const bytes = this.tmp();
        this.emit(`${data} = load ptr, ptr ${arr}, !tbaa ${HDR}`);
        // 確保の側が 要素数 × 要素サイズ のあふれを検査するので、下の掛け算はあふれない。
        if ((e.recv.ty as Ty & { k: 'heap' }).buffer) copy = this.allocBuffer(len, elem);
        else this.emit(`${copy} = call ptr @um_alloc(${usz} ${len}, ${usz} ${size})`);
        const rty = e.recv.ty as Ty & { k: 'heap' };
        if (rty.racy) {
          // RacyBuffer は他のスレッドが同時に要素を書くことがあるので、memcpy ではなく要素ごとに不可分に読む。
          // 写し先はまだ誰にも見えていないので、ふつうの書き込みでよい。
          const et = llTy(elem);
          const ord = this.kernel ? 'unordered' : 'monotonic';
          this.countLoop(len, (k) => {
            const sp = this.tmp();
            const dp = this.tmp();
            const v = this.tmp();
            this.emit(`${sp} = getelementptr inbounds ${et}, ptr ${data}, ${usz} ${k}`);
            this.emit(`${v} = load atomic ${et}, ptr ${sp} ${ord}, align ${racyAlign(elem)}`);
            this.emit(`${dp} = getelementptr inbounds ${et}, ptr ${copy}, ${usz} ${k}`);
            this.emit(`store ${et} ${v}, ptr ${dp}`);
          });
          return this.arrayValue(copy, len, len);
        }
        this.emit(`${bytes} = mul ${usz} ${len}, ${size}`);
        const name = `llvm.memcpy.p0.p0.${usz}`;
        this.m.intrinsics.add(`declare void @${name}(ptr, ptr, ${usz}, i1)`);
        this.emit(`call void @${name}(ptr ${copy}, ptr ${data}, ${usz} ${bytes}, i1 false)`);
        return this.arrayValue(copy, len, len);
      }
    }
    throw new Error(`internal: array method ${e.name}`);
  }

  // ---- スレッド ----

  /** 持ち手の変数 → 置き場所（{ 持ち手, 引数の塊, 本数 }）と塊の型。 */
  threadVars = new Map<number, { rec: string; info: ThreadInfo }>();

  /** 0 から n 未満の k について body(k) を生成するループ（k は usize）。 */
  countLoop(n: string, body: (k: string) => void): void {
    const usz = this.m.usz;
    const idx = this.alloca({ k: 'int', bits: this.m.prog.ptrBits, signed: false });
    this.emit(`store ${usz} 0, ptr ${idx}`);
    const condL = this.newLabel('each.cond');
    const bodyL = this.newLabel('each.body');
    const endL = this.newLabel('each.end');
    this.br(condL);
    this.label(condL);
    const k = this.tmp();
    const more = this.tmp();
    this.emit(`${k} = load ${usz}, ptr ${idx}`);
    this.emit(`${more} = icmp ult ${usz} ${k}, ${n}`);
    this.emit(`br i1 ${more}, label %${bodyL}, label %${endL}`);
    this.label(bodyL);
    body(k);
    const next = this.tmp();
    this.emit(`${next} = add ${usz} ${k}, 1`);
    this.emit(`store ${usz} ${next}, ptr ${idx}`);
    this.br(condL);
    this.label(endL);
  }

  /**
   * `const h = thread.spawn(f, ...)` / `thread.spawnEach(f, own items, ...)`。
   * 引数の塊（[0] が戻り値、[1..] が引数）を確保して埋め、ランタイムに起動を頼む。spawnEach は要素ごとに塊を1つ並べる。
   * 持ち手は { 持ち手, 塊, 本数 } として置き、スコープの解放の並びに入れる（抜けるときに合流する）。
   */
  spawn(s: Stmt & { kind: 'let' }): void {
    const e = s.init as Expr & { kind: 'thread' };
    const usz = this.m.usz;
    const sig = this.m.prog.fns.get(e.fn!)!;
    const each = e.op === 'each';
    const field = (p: { mode: string; ty: Ty }) => (p.mode === 'none' && !this.m.isCopy(p.ty) ? 'ptr' : llTy(p.ty));
    const blk = `{ ${[sig.ret.k === 'void' ? 'i8' : llTy(sig.ret), ...sig.params.map(field)].join(', ')} }`;
    const entry = this.m.threadEntry(sig, blk);
    const size = `ptrtoint (ptr getelementptr (${blk}, ptr null, i32 1) to ${usz})`;

    // 引数を順に評価する。借用の場所は、全部を評価し終えてから求める（prep を参照）。
    let items = '';
    const vals: (() => string)[] = [];
    sig.params.forEach((p, i) => {
      const a = e.args[i].expr;
      if (each && i === 0) {
        items = this.consume(a);
        return;
      }
      if (p.mode === 'own') {
        const v = this.consume(a);
        vals[i] = () => v;
      } else if (p.mode === 'none' && !this.m.isCopy(p.ty)) {
        const pr = this.prep(a);
        vals[i] = pr;
      } else {
        const v = this.expr(a);
        vals[i] = () => v;
      }
    });
    const got = vals.map((v) => (v ? v() : ''));
    const fill = (bp: string) => {
      sig.params.forEach((p, i) => {
        if (each && i === 0) return;
        const fp = this.tmp();
        this.emit(`${fp} = getelementptr inbounds ${blk}, ptr ${bp}, i32 0, i32 ${i + 1}`);
        this.emit(`store ${field(p)} ${got[i]}, ptr ${fp}`);
      });
    };
    let blocks = this.tmp();
    let n = '1';
    if (!each) {
      this.emit(`${blocks} = call ptr @um_alloc(${usz} 1, ${usz} ${size})`);
      fill(blocks);
    } else {
      // 要素を1つずつ塊へ移す（要素の所有は各スレッドへ）。配列の入れ物だけをここで解放する。
      const p0 = sig.params[0];
      const data = this.tmp();
      n = this.tmp();
      this.emit(`${data} = extractvalue %um.Array ${items}, 0`);
      this.emit(`${n} = extractvalue %um.Array ${items}, 1`);
      this.emit(`${blocks} = call ptr @um_alloc(${usz} ${n}, ${usz} ${size})`);
      const b = blocks;
      this.countLoop(n, (k) => {
        const bp = this.tmp();
        const ip = this.tmp();
        const iv = this.tmp();
        const fp = this.tmp();
        this.emit(`${bp} = getelementptr inbounds ${blk}, ptr ${b}, ${usz} ${k}`);
        this.emit(`${ip} = getelementptr inbounds ${llTy(p0.ty)}, ptr ${data}, ${usz} ${k}`);
        this.emit(`${iv} = load ${llTy(p0.ty)}, ptr ${ip}`);
        this.emit(`${fp} = getelementptr inbounds ${blk}, ptr ${bp}, i32 0, i32 1`);
        this.emit(`store ${llTy(p0.ty)} ${iv}, ptr ${fp}`);
        fill(bp);
      });
      this.emit(`call void @um_free(ptr ${data})`);
      blocks = b;
    }
    const h = this.tmp();
    this.emit(`${h} = call ptr @um_thread_start(ptr ${entry}, ptr ${blocks}, ${usz} ${size}, ${usz} ${n}, i32 ${s.pos.line})`);
    const recTy = `{ ptr, ptr, ${usz} }`;
    const rec = `%a${++this.n}`;
    this.allocas.push(`  ${rec} = alloca ${recTy}`);
    const at = (i: number) => {
      const p = this.tmp();
      this.emit(`${p} = getelementptr inbounds ${recTy}, ptr ${rec}, i32 0, i32 ${i}`);
      return p;
    };
    this.emit(`store ptr ${h}, ptr ${at(0)}`);
    this.emit(`store ptr ${blocks}, ptr ${at(1)}`);
    this.emit(`store ${usz} ${n}, ptr ${at(2)}`);
    const info: ThreadInfo = { blk, ret: sig.ret };
    const flag = this.flagAlloca();
    this.emit(`store i1 true, ptr ${flag}`);
    this.flags.set(s.sym!.id, flag);
    this.threadVars.set(s.sym!.id, { rec, info });
    this.scopes[this.scopes.length - 1].push({ ptr: rec, flag, ty: s.sym!.ty, thread: info });
  }

  /**
   * スレッドの終わりを待つ。keep なら戻り値を返す（spawn は R、spawnEach は R[]）。keep でなければ戻り値を解放する。
   * 引数の塊はここで解放する。
   */
  joinThreads(rec: string, info: ThreadInfo, keep: boolean, each = false): string {
    const usz = this.m.usz;
    const recTy = `{ ptr, ptr, ${usz} }`;
    const field = (i: number, t: string) => {
      const p = this.tmp();
      const v = this.tmp();
      this.emit(`${p} = getelementptr inbounds ${recTy}, ptr ${rec}, i32 0, i32 ${i}`);
      this.emit(`${v} = load ${t}, ptr ${p}`);
      return v;
    };
    const h = field(0, 'ptr');
    this.emit(`call void @um_thread_join(ptr ${h})`);
    const blocks = field(1, 'ptr');
    const n = field(2, usz);
    const ret = info.ret;
    const retAt = (k: string) => {
      const p = this.tmp();
      this.emit(`${p} = getelementptr inbounds ${info.blk}, ptr ${blocks}, ${usz} ${k}, i32 0`);
      return p;
    };
    let out = '';
    if (!keep) {
      if (ret.k !== 'void' && !this.m.isCopy(ret)) this.countLoop(n, (k) => this.dropAt(retAt(k), ret));
    } else if (ret.k !== 'void' && !each) {
      out = this.tmp();
      this.emit(`${out} = load ${llTy(ret)}, ptr ${retAt('0')}`);
    } else if (ret.k !== 'void') {
      const data = this.tmp();
      this.emit(`${data} = call ptr @um_alloc(${usz} ${n}, ${usz} ${this.m.sizeOf(ret)})`);
      this.countLoop(n, (k) => {
        const v = this.tmp();
        const dp = this.tmp();
        this.emit(`${v} = load ${llTy(ret)}, ptr ${retAt(k)}`);
        this.emit(`${dp} = getelementptr inbounds ${llTy(ret)}, ptr ${data}, ${usz} ${k}`);
        this.emit(`store ${llTy(ret)} ${v}, ptr ${dp}`);
      });
      out = this.arrayValue(data, n, n);
    }
    this.emit(`call void @um_free(ptr ${blocks})`);
    return out;
  }

  threadExpr(e: Expr & { kind: 'thread' }): string {
    if (e.op === 'cpus') {
      const v = this.tmp();
      this.emit(`${v} = call i32 @um_thread_cpus()`);
      return v;
    }
    if (e.op === 'yield') {
      this.emit('call void @um_thread_yield()');
      return '';
    }
    if (e.op !== 'join') throw new Error('internal: thread.spawn outside a const declaration');
    const sym = (e.args[0].expr as Expr & { kind: 'ident' }).sym!;
    const t = this.threadVars.get(sym.id)!;
    // 合流済みの印を先に付ける（自動の合流で二重に待たないように）。型検査が2回目の join を禁じている。
    this.emit(`store i1 false, ptr ${this.flags.get(sym.id)!}`);
    return this.joinThreads(t.rec, t.info, true, sym.thread!.each);
  }

  /** readLine / inputReady はメインスレッドだけ（標準入力のバッファは1つ）。ほかのスレッドなら停止する。 */
  mainOnly(line: number): void {
    const m = this.tmp();
    const bad = this.tmp();
    this.emit(`${m} = call i32 @um_is_main()`);
    this.emit(`${bad} = icmp eq i32 ${m}, 0`);
    this.check(bad, PANIC_THREAD_IO, line);
  }

  /** RacyBuffer のアトミック演算（load / store / fetchAdd / compareExchange、順序は seq_cst）。 */
  racyAtomic(e: Expr & { kind: 'mcall' }): string {
    const elem = (e.recv.ty as Ty & { k: 'heap' }).elem;
    const t = llTy(elem);
    const al = racyAlign(elem);
    // 添字の検査は b[i] と同じ。場所は全引数を評価し終えてから求める。
    const place = this.prep({ kind: 'index', base: e.recv, index: e.args[0].expr, ty: elem, pos: e.pos });
    const vals = e.args.slice(1).map((a) => this.expr(a.expr));
    const p = place();
    const v = this.tmp();
    switch (e.name) {
      case 'load':
        this.emit(`${v} = load atomic ${t}, ptr ${p} seq_cst, align ${al}`);
        return v;
      case 'store':
        this.emit(`store atomic ${t} ${vals[0]}, ptr ${p} seq_cst, align ${al}`);
        return '';
      case 'fetchAdd':
        this.emit(`${v} = atomicrmw add ptr ${p}, ${t} ${vals[0]} seq_cst, align ${al}`);
        return v;
      case 'compareExchange': {
        const r = this.tmp();
        this.emit(`${r} = cmpxchg ptr ${p}, ${t} ${vals[0]}, ${t} ${vals[1]} seq_cst seq_cst, align ${al}`);
        this.emit(`${v} = extractvalue { ${t}, i1 } ${r}, 0`);
        return v;
      }
    }
    throw new Error(`internal: RacyBuffer method ${e.name}`);
  }

  emitCall(sig: FnSig, args: string[]): string {
    const callee = `${this.m.fnName(sig.key)}(${args.join(', ')})`;
    if (sig.ret.k === 'void') {
      this.emit(`call void ${callee}`);
      return '';
    }
    const v = this.tmp();
    this.emit(`${v} = call ${llTy(sig.ret)} ${callee}`);
    return v;
  }
}

export function generate(prog: Program, target: TargetKind, sanitize: boolean | 'thread' = false, metal = false, appleArm = false, windows = false, plugin = false,
  wgpu = false): string {
  return new ModuleGen(prog, target, sanitize, metal, appleArm, windows, plugin, wgpu).generate();
}
