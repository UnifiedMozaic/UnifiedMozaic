import { inferGpuLoad, isGpuLoadCall } from './gpuload.ts';
import { CompileError, CompileErrors } from './diag.ts';
import type { Pos } from './diag.ts';
import { displayNames, GROUP_COLLECTIVE, isAgg, isCollectiveLet, isNumeric, tyEq, tyStr } from './ast.ts';
import type { Arg, ClassDecl, ConstDecl, Expr, FnDecl, InterfaceDecl, Mode, Module, Stmt, StructDecl, Sym, Ty, TypeNode } from './ast.ts';

export interface StructInfo {
  name: string;
  isClass: boolean;
  /** 代入でコピーしてよいか（ヒープを持たないか）。isCopyTy が最初に使うときに計算する。 */
  copy?: boolean;
  fields: { name: string; ty: Ty; isPrivate: boolean; init?: Expr }[];
  methods: Map<string, FnSig>;
  ctor?: FnSig;
  /** ジェネリックな struct を具体化したもののとき、雛形の名前と型引数。 */
  generic?: { name: string; args: Ty[] };
  /** class が満たしている interface の名前。 */
  implements?: string[];
  /** `T | null` を表す struct のとき、T。フィールドは `__some`（値があるか）と `__value`。 */
  opt?: Ty;
  /** enum のとき、その選択肢（値の名前つき）。 */
  variants?: { name: string; fields: string[] }[];
}

export interface FnSig {
  /** 生成コード上の名前。関数は `name`、メソッドは `Class.name`、コンストラクタは `Class.new`。 */
  key: string;
  name: string;
  params: { name: string; mode: Mode; ty: Ty }[];
  ret: Ty;
  exported: boolean;
  decl: FnDecl;
}

/** メソッド呼び出しの記録。this を書き換えるかどうかは全本体の検査後に決まるので、検査を遅らせる。 */
interface PendingCall {
  caller: FnDecl;
  recv: Expr;
  args: Arg[];
  sig: FnSig;
  pos: Pos;
  /** 呼び出し時点で for...of が反復中の置き場所。 */
  frozen: Frozen[];
}

/** for...of が反復中の置き場所。path は変数からのフィールド名の列（添字は '[]'）。 */
interface Frozen {
  sym: Sym;
  path: string[];
  text: string;
  /** 何が止めているか（エラーメッセージ用）。なければ for...of。thread はスレッドに読み取り借用で貸している間。 */
  by?: 'match' | 'thread';
}

/** 止めているものの説明（エラーメッセージ用）。 */
function frozenBy(f: Frozen): string {
  if (f.by === 'match') return `a \`match\` is looking at '${f.text}'`;
  if (f.by === 'thread') return `'${f.text}' is lent to a thread (until the end of the block that started it)`;
  return `a \`for...of\` is iterating over '${f.text}'`;
}

/** 検査中のループ。ループの外で宣言された変数が、周回の終わりに移動済みのまま残っていないかを調べる。 */
interface LoopCtx {
  depth: number;
  before: Set<Sym>;
  /** break で抜けた時点で移動済みだった変数。ループの後では「移動したかもしれない」として扱う。 */
  breakMoved: Set<Sym>;
}

/** 置き場所を、根の変数とそこからの経路に分ける。 */
function placePath(e: Expr): { sym: Sym; path: string[] } | null {
  if (e.kind === 'ident') return e.sym ? { sym: e.sym, path: [] } : null;
  if (e.kind === 'field' || e.kind === 'index') {
    const base = placePath(e.base);
    return base ? { sym: base.sym, path: [...base.path, e.kind === 'field' ? e.name : '[]'] } : null;
  }
  return null;
}

/** 2つの経路の一方がもう一方を含むか（同じ場所か、その内側・外側か）。 */
function overlaps(a: string[], b: string[]): boolean {
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) if (a[i] !== b[i]) return false;
  return true;
}

/** 最上位の定数の配列か、その要素（行）を指す式か: `T`、`T[i]`、`T[i][j]`。 */
export function isConstPlace(e: Expr): boolean {
  if (e.kind === 'ident') return !!e.sym?.global;
  return e.kind === 'index' && isConstPlace(e.base);
}

function placeText(e: Expr): string {
  if (e.kind === 'ident') return e.name;
  if (e.kind === 'field') return `${placeText(e.base)}.${e.name}`;
  if (e.kind === 'index') return `${placeText(e.base)}[...]`;
  return 'the value';
}

/**
 * 代入でコピーしてよい型かどうか。ヒープを持つ型（可変長配列と、それを含む構造体）はコピーできず、
 * 代入や own 渡しで所有権が移動する。
 */
export function isCopyTy(structs: Map<string, StructInfo>, t: Ty): boolean {
  if (t.k === 'heap') return false;
  if (t.k === 'array') return isCopyTy(structs, t.elem);
  if (t.k !== 'struct') return true;
  const info = structs.get(t.name)!;
  if (info.copy === undefined) {
    info.copy = true; // 自己参照は別の検査で弾く。ここでは無限再帰だけ避ける。
    info.copy = info.fields.every((f) => isCopyTy(structs, f.ty));
  }
  return info.copy;
}

export interface Program {
  structs: Map<string, StructInfo>;
  fns: Map<string, FnSig>;
  decls: FnDecl[];
  ptrBits: 32 | 64;
  /** f16 を使っているか。使うときだけ、変換用の関数を出力に含める。 */
  usesF16?: boolean;
}

const VOID: Ty = { k: 'void' };
const BOOL: Ty = { k: 'bool' };
const I32: Ty = { k: 'int', bits: 32, signed: true };
const I64: Ty = { k: 'int', bits: 64, signed: true };
const F64: Ty = { k: 'float', bits: 64 };
const F32: Ty = { k: 'float', bits: 32 };
/** 文字列。UTF-8 のバイト列を持つ、書き換えできない可変長配列として扱う。 */
export const STR: Ty = { k: 'heap', elem: { k: 'int', bits: 8, signed: false }, str: true };
const isStr = (t: Ty) => t.k === 'heap' && t.str === true;
const U32: Ty = { k: 'int', bits: 32, signed: false };
const U64: Ty = { k: 'int', bits: 64, signed: false };

const PRIMS: Record<string, Ty> = {
  i8: { k: 'int', bits: 8, signed: true },
  i16: { k: 'int', bits: 16, signed: true },
  i32: I32,
  i64: I64,
  u8: { k: 'int', bits: 8, signed: false },
  u16: { k: 'int', bits: 16, signed: false },
  u32: { k: 'int', bits: 32, signed: false },
  u64: { k: 'int', bits: 64, signed: false },
  f4: { k: 'float', bits: 4 },
  f8: { k: 'float', bits: 8 },
  f16: { k: 'float', bits: 16 },
  f32: { k: 'float', bits: 32 },
  f64: F64,
  bool: BOOL,
  void: VOID,
};

// TypeScriptの型名を書いたときの置き換え先。
const TYPE_HINTS: Record<string, [string, string]> = {
  number: ['ts-feature', 'there is no `number`; pick a sized type: `i32`, `i64`, `f32` or `f64`'],
  boolean: ['ts-feature', 'write `bool`'],
  bigint: ['ts-feature', 'write `i64` or `u64`'],
  any: ['ts-feature', 'there are no dynamic types; write the concrete type'],
  unknown: ['ts-feature', 'there are no dynamic types; write the concrete type'],
  object: ['ts-feature', 'declare a `struct` and use its name'],
  undefined: ['ts-feature', '`undefined` does not exist'],
};

const BUILTINS = new Set(['print', 'dispatch', 'dispatchGroups', 'String']);

// 128 ビットの SIMD のベクトルの型の名前。同じ要素の固定長配列の別名（i16x8 は i16[8] と同じ型）。
const SIMD_TYPES: Record<string, string> = {
  i8x16: 'i8', u8x16: 'u8', i16x8: 'i16', u16x8: 'u16', i32x4: 'i32', u32x4: 'u32', f32x4: 'f32',
};

/** t が 128 ビットの SIMD のベクトル（i8x16 / i16x8 / i32x4 / f32x4 と符号なしの版）なら、要素の型とレーンの数。 */
export function simdInfo(t: Ty | undefined): { elem: Ty & { k: 'int' | 'float' }; lanes: number } | undefined {
  if (t?.k !== 'array') return undefined;
  const e = t.elem;
  const ok = (e.k === 'int' && !e.usize && e.bits <= 32) || (e.k === 'float' && e.bits === 32);
  if (!ok || t.len * (e as { bits: number }).bits !== 128) return undefined;
  return { elem: e as Ty & { k: 'int' | 'float' }, lanes: t.len };
}

/**
 * 入出力の組み込み関数。同じ名前の関数をプログラムが定義していれば、そちらが優先される。
 * 型検査は呼び出しの名前を `__io_名前` に書き換え、コード生成がランタイムの呼び出しにする。kernel の中では使えない。
 */
const IO_BUILTINS = new Set(['readLine', 'inputReady', 'clock', 'stringFromBytes', 'readFile', 'writeFile', 'fileSize', 'readFileRange', 'appendFile', 'args', 'eprint',
  'windowOpen', 'windowClose', 'windowPoll', 'windowPresent',
  'wasmCompile', 'wasmFreeModule', 'wasmExportList', 'wasmInstantiate', 'wasmFreeInstance', 'wasmInvoke', 'wasmRead', 'wasmWrite',
  'wasmMemorySize', 'wasmWat', 'wasmError',
  'mediaOpen', 'mediaClose', 'mediaInfoNumbers', 'mediaInfoText', 'mediaReadVideo', 'mediaReadAudio', 'mediaLastTime',
  'mediaCreateWriter', 'mediaAddVideo', 'mediaAddAudio', 'mediaWriteVideo', 'mediaWriteAudio', 'mediaFinishWriter', 'mediaWriterText',
  'mediaEncoderList', 'mediaError', 'mediaVersion']);

/** 最上位の定数。値は最初に使われたとき（遅くとも宣言の収集の終わり）に求める。 */
interface ConstInfo {
  decl: ConstDecl;
  state: 'new' | 'busy' | 'done' | 'bad';
  ty?: Ty;
  value?: ConstValue;
}
/** 定数の値。固定長配列の定数（`const T: u32[8] = [...]`）は要素の値の配列（入れ子の配列は入れ子のまま）。 */
type ConstValue = bigint | number | boolean | string | ConstValue[];

/** すでに報告済みの定数の誤りを、使う側でもう一度報告しないための印。 */
const QUIET = 'const-already-reported';

/** 整数を f32 に、最も近い値へ1回で丸める（f64 を経由すると2回丸めになり、結果が変わることがある）。 */
function bigToF32(v: bigint): number {
  const neg = v < 0n;
  let a = neg ? -v : v;
  const bits = a.toString(2).length;
  let scale = 0;
  if (bits > 24) {
    scale = bits - 24;
    const rest = a & ((1n << BigInt(scale)) - 1n);
    const half = 1n << BigInt(scale - 1);
    a >>= BigInt(scale);
    if (rest > half || (rest === half && (a & 1n) === 1n)) a++;
  }
  const x = Math.fround(Number(a) * 2 ** scale);
  return neg ? -x : x;
}

/** 固定長配列と struct が持てる値（数値・bool）の個数の上限。 */
const MAX_VALUE_SCALARS = 4096;

/**
 * 最上位の定数の配列（`const T: u64[81] = [...]`）が持てる値の個数の上限。読み取り専用のグローバルな領域に置き、
 * 添字で読むだけならコピーしないので、値として扱う固定長配列より大きくてよい（コンパイルの時間とメモリの歯止め）。
 */
const MAX_CONST_SCALARS = 1 << 22;

/** Math.xxx として呼べる組み込み。float はすべての浮動小数点型、num は整数も受け付ける。 */
const MATH: Record<string, { argc: number; kind: 'float' | 'num' | 'int' }> = {
  sqrt: { argc: 1, kind: 'float' },
  floor: { argc: 1, kind: 'float' },
  ceil: { argc: 1, kind: 'float' },
  abs: { argc: 1, kind: 'num' },
  min: { argc: 2, kind: 'num' },
  max: { argc: 2, kind: 'num' },
  // ビット演算。立っているビットの数、上位・下位から続く 0 の数（値が 0 ならビット幅）。
  popcount: { argc: 1, kind: 'int' },
  clz: { argc: 1, kind: 'int' },
  ctz: { argc: 1, kind: 'int' },
  // マスクつきシフト。シフト量を値のビット幅で割った余りにする（範囲検査なし。kernel のシフトと同じ意味）。
  shl: { argc: 2, kind: 'int' },
  shr: { argc: 2, kind: 'int' },
  // 先読み Math.prefetch(arr, i)。型の検査は inferMath の専用の分岐で行う（kind は使わない）。
  prefetch: { argc: 2, kind: 'int' },
};

function intRange(t: Ty & { k: 'int' }): [bigint, bigint] {
  const b = BigInt(t.bits);
  return t.signed ? [-(1n << (b - 1n)), (1n << (b - 1n)) - 1n] : [0n, (1n << b) - 1n];
}

function isLiteral(e: Expr): boolean {
  // 最上位の定数を置き換えた値（fixed）は型が決まっているので、文脈に合わせるリテラルとしては扱わない。
  return (e.kind === 'int' || e.kind === 'float') && !e.fixed;
}

/** 代入先として書ける形かどうか。 */
export function isPlace(e: Expr): boolean {
  if (e.kind === 'ident') return true;
  if (e.kind === 'field') return isPlace(e.base);
  if (e.kind === 'index') return isPlace(e.base);
  return false;
}

function rootSym(e: Expr): Sym | undefined {
  if (e.kind === 'ident') return e.sym;
  if (e.kind === 'field' || e.kind === 'index') return rootSym(e.base);
  return undefined;
}

/** 引数をポインタで渡すかどうか。呼び出し側と定義側で共有する規約。 */
export function passByPtr(mode: Mode, ty: Ty): boolean {
  return mode === 'mut' || isAgg(ty);
}

/** 8 / 16 ビットの整数（i8 / u8 / i16 / u16）か。 */
export function isSmallInt(t: Ty | undefined): boolean {
  return t?.k === 'int' && t.bits <= 16 && !t.usize;
}

class Checker {
  usesF16 = false;
  structs = new Map<string, StructInfo>();
  fns = new Map<string, FnSig>();
  errors: CompileError[] = [];
  scopes: Map<string, Sym>[] = [];
  nextSym = 0;
  curRet: Ty = VOID;
  curFn!: FnDecl;
  loopDepth = 0;
  /** コンストラクタの検査中だけ使う。初期化済みのフィールド名。 */
  ctorInit: Set<string> | null = null;
  /** 文の入れ子の深さ。コンストラクタでは最上位の代入だけを初期化として数える。 */
  stmtDepth = 0;
  /** 代入先として検査中の式。初期化前の読み出し検査から除外する。 */
  assignTarget: Expr | null = null;
  pending: PendingCall[] = [];
  directMut = new Set<FnDecl>();
  /** 所有権が移動済み（または移動したかもしれない）変数。 */
  moved = new Set<Sym>();
  /** for...of が反復中の置き場所。反復が終わるまで、その場所は書き換えも移動もできない。 */
  frozen: Frozen[] = [];
  loops: LoopCtx[] = [];
  /** 変数が移動された位置（エラーの報告用）。 */
  movedAt = new Map<Sym, Pos>();
  /** 構造体のフィールドを集めている間は、要素型のコピー可否がまだ決まらないので検査を後回しにする。 */
  collecting = false;
  /** kernel 関数の本体を検査中かどうか。 */
  inKernel = false;
  /** 書き込み先の Buffer を持つ kernel の、添字の引数。 */
  idxParam: Sym | null = null;
  /** kernel から kernel への呼び出し。再帰の検出に使う。 */
  kernelCalls: { caller: FnDecl; callee: FnDecl; pos: Pos }[] = [];
  /** dispatch / dispatchGroups の呼び出し。group.* を使う kernel が dispatchGroups で実行されているかを最後に確かめる。 */
  groupDispatches: { kernel: FnDecl; groups: boolean; pos: Pos }[] = [];
  /** kernel を普通の関数として呼んでいる所（kernel の中からも外からも）。group.* を使う kernel は呼べない。 */
  kernelUses: { callee: FnDecl; pos: Pos }[] = [];
  /** 真の間は、共有配列の名前を読んでよい（添字の土台を検査している間）。 */
  sharedBase = false;
  ptrBits: 32 | 64;
  usize: Ty;
  /** このターゲットに GPU バックエンドがあるか。 */
  gpu: boolean;

  constructor(ptrBits: 32 | 64, gpu: boolean) {
    this.gpu = gpu;
    this.ptrBits = ptrBits;
    this.usize = { k: 'int', bits: ptrBits, signed: false, usize: true };
  }

  err(code: string, message: string, pos: Pos, hint?: string): never {
    throw new CompileError(code, message, pos, hint);
  }

  // ---- 型の解決 ----

  isCopy(t: Ty): boolean {
    return isCopyTy(this.structs, t);
  }

  /**
   * kernel の中で値として扱える型: bool / i32 / u32 / i64 / u64 / f32 と、それらの struct・固定長配列。
   * 64 ビットの整数は CPU と Metal（MSL の long / ulong）だけ。WebGPU（WGSL）には 64 ビットの整数がないので、
   * それを使う kernel は wasm32 では CPU で実行する（codegen.ts の kernelUses64）。
   */
  isKernelValue(t: Ty, nested = false): boolean {
    if (t.k === 'bool') return true;
    // 8 / 16 ビットの整数は、単独の値（変数、引数、Buffer の要素）としてだけ使える。
    // struct や固定長配列の中には置けない（WGSL の struct は 4 バイト単位で、並びが CPU と合わないため）。
    if (isSmallInt(t)) return !nested;
    if (t.k === 'int') return (t.bits === 32 || t.bits === 64) && !t.usize;
    if (t.k === 'float') return t.bits === 32;
    if (t.k === 'array') return this.isKernelValue(t.elem, true);
    if (t.k === 'struct') {
      const info = this.structs.get(t.name)!;
      return !info.isClass && info.fields.every((f) => this.isKernelValue(f.ty, true));
    }
    return false;
  }

  /** ゼロで初期化してよい型: 数値、bool、それらの固定長配列と struct。 */
  isZeroable(t: Ty): boolean {
    if (t.k === 'int' || t.k === 'float' || t.k === 'bool') return true;
    if (t.k === 'array') return this.isZeroable(t.elem);
    if (t.k === 'struct') {
      const info = this.structs.get(t.name)!;
      return !info.isClass && info.fields.every((f) => this.isZeroable(f.ty));
    }
    return false;
  }

  /** Buffer の要素にできる型: CPU と GPU でビット表現が同じもの（bool は含めない）。 */
  isShareable(t: Ty, nested = false): boolean {
    if (t.k === 'bool') return false;
    if (t.k === 'array') return this.isShareable(t.elem, true);
    if (t.k === 'struct') {
      const info = this.structs.get(t.name)!;
      return !info.isClass && info.fields.every((f) => this.isShareable(f.ty, true));
    }
    return this.isKernelValue(t, nested);
  }

  /** 型に含まれる数値・bool の個数（固定長配列は要素数ぶん）。 */
  scalarCount(t: Ty, seen: string[] = []): number {
    if (t.k === 'array') return t.len * this.scalarCount(t.elem, seen);
    if (t.k === 'struct') {
      if (seen.includes(t.name)) return 0; // 自己参照は別の検査で弾く
      return this.structs.get(t.name)!.fields.reduce((n, f) => n + this.scalarCount(f.ty, [...seen, t.name]), 0);
    }
    return 1;
  }

  /**
   * 固定長配列と struct は値として（スタックやレジスタで）扱うので、大きすぎるものは禁止する。
   * 巨大な値はコンパイルに極端な時間とメモリがかかり、実行時にはスタックをあふれさせる。
   */
  checkValueSize(t: Ty, pos: Pos): void {
    if (t.k !== 'array' && t.k !== 'struct') return;
    const n = this.scalarCount(t);
    if (this.constTy && t.k === 'array') {
      // 最上位の定数の配列の型: 値としてコピーしないので上限が大きい（値として使う所は checkExpr が別に確かめる）。
      if (n > MAX_CONST_SCALARS) {
        this.err('too-large', `type ${tyStr(t)} holds ${n} values, more than the limit of ${MAX_CONST_SCALARS} for a top-level constant array`, pos,
          'split the table into several constants, or load it at run time into a `Buffer<T>` / `T[]`');
      }
      return;
    }
    if (n > MAX_VALUE_SCALARS) {
      this.err('too-large', `type ${tyStr(t)} holds ${n} values, more than the limit of ${MAX_VALUE_SCALARS} for a fixed-length array or struct`, pos,
        'fixed-length arrays and structs are copied by value; for large data use a growable array (`T[]`, e.g. `new Array<T>(n)`) or a `Buffer<T>`');
    }
  }

  /** 配列の要素はコピーできる型に限る（ヒープを持つ要素の解放はまだ実装していない）。 */
  checkElem(t: Ty, pos: Pos): void {
    this.checkValueSize(t, pos);
    if (t.k === 'heap') this.checkValueSize(t.elem, pos);
    // RacyBuffer の要素は1命令でちぎれずに読み書きできる型だけ。
    const racyOk = (e: Ty) => (e.k === 'int' && !e.usize) || (e.k === 'float' && e.bits === 32);
    if (t.k === 'heap' && t.racy && !racyOk(t.elem)) {
      this.err('kernel-type', `'${tyStr(t.elem)}' cannot be an element of a RacyBuffer`, pos,
        'a RacyBuffer is read and written without synchronisation, so each element must be a single integer (i8 to i64, u8 to u64) or f32 (a struct could be seen half-updated)');
    }
    if (t.k === 'heap' && t.buffer && !t.racy && !this.isShareable(t.elem)) {
      this.err('kernel-type', `'${tyStr(t.elem)}' cannot be an element of a Buffer`, pos,
        'Buffer elements are shared with the GPU: use i8, u8, i16, u16, i32, u32, i64, u64, f32, or a struct / fixed-length array of i32, u32, i64, u64 and f32');
    }
    if (t.k === 'array' || t.k === 'heap') {
      // 可変長配列は、ヒープを持つ要素（文字列、配列、それらを持つ struct）を所有できる。固定長配列はまだできない。
      if (t.k === 'array' && !this.isCopy(t.elem)) {
        this.err('not-implemented', `fixed-length arrays of '${tyStr(t.elem)}' are not implemented yet`, pos,
          `elements that own heap memory need a growable array: write \`${tyStr(t.elem)}[]\``);
      }
      this.checkElem(t.elem, pos);
    }
  }

  /** null でないと分かっている変数（`if (x != null)` の中など）。const の変数だけが入る。 */
  narrowed = new Set<number>();
  optionals = new Map<string, string>();

  /** t が `T | null` なら T を返す。 */
  optInner(t: Ty | undefined): Ty | undefined {
    return t?.k === 'struct' ? this.structs.get(t.name)?.opt : undefined;
  }

  /** `inner | null` の型。内部では { __some: bool, __value: inner } の struct として持つ。 */
  optionalOf(inner: Ty, pos: Pos): Ty {
    if (inner.k === 'void') this.err('bad-type', '`void | null` is not a type', pos);
    if (this.optInner(inner)) this.err('bad-type', `${tyStr(inner)} is already nullable`, pos);
    if (inner.k === 'struct' && this.structs.get(inner.name)!.isClass) {
      this.err('not-implemented', `\`${tyStr(inner)} | null\` is not implemented yet for classes`, pos, 'use a struct, or keep a separate `bool` that says whether the object is present');
    }
    if (inner.k === 'heap' && inner.buffer) this.err('not-implemented', '`Buffer<T> | null` is not implemented yet', pos);
    const key = tyStr(inner);
    let name = this.optionals.get(key);
    if (!name) {
      name = `__opt_${key.replace(/[^A-Za-z0-9]+/g, '_')}`;
      while (this.structs.has(name)) name += '_';
      this.optionals.set(key, name);
      displayNames.set(name, `${key} | null`);
      this.structs.set(name, {
        name, isClass: false, methods: new Map(), opt: inner,
        fields: [{ name: '__some', ty: BOOL, isPrivate: false }, { name: '__value', ty: inner, isPrivate: false }],
      });
    }
    return { k: 'struct', name };
  }

  /** 型 t の「空の値」を表す式。null のときに __value へ入れる（読まれることはない）。 */
  zeroExpr(t: Ty, pos: Pos, inside: string[] = []): Expr {
    if (t.k === 'int') return { kind: 'int', value: 0n, pos };
    if (t.k === 'float') return { kind: 'float', value: 0, pos };
    if (t.k === 'bool') return { kind: 'bool', value: false, pos };
    if (t.k === 'heap' && t.str) return { kind: 'str', value: '', pos };
    if (t.k === 'heap' && !t.buffer) return { kind: 'arrayLit', elems: [], pos };
    if (t.k === 'array' && t.len <= 64) return { kind: 'arrayLit', elems: Array.from({ length: t.len }, () => this.zeroExpr(t.elem, pos, inside)), pos };
    if (t.k === 'struct' && !this.structs.get(t.name)!.isClass) {
      // 自分自身を値として含む型（別のエラーとして報告済み）で、止まらなくなるのを防ぐ。
      if (inside.includes(t.name)) this.err('recursive-struct', `'${tyStr(t)}' contains itself by value`, pos, 'hold it through an array (`T[]`) instead');
      const fields = this.structs.get(t.name)!.fields.map((f) => ({ name: f.name, value: this.zeroExpr(f.ty, pos, [...inside, t.name]), pos }));
      return { kind: 'object', fields, pos, synth: true };
    }
    this.err('not-implemented', `\`null\` for the type ${tyStr(t)} is not implemented yet`, pos);
  }

  /** e を別の式に書き換える（同じ場所のまま中身を入れ替える）。 */
  rewrite(e: Expr, to: Record<string, unknown>): void {
    const node = e as unknown as Record<string, unknown>;
    for (const key of Object.keys(node)) if (key !== 'pos') delete node[key];
    Object.assign(node, to);
  }

  /** `x != null` / `x == null` の形で、x が const の nullable な変数なら、その変数と向きを返す。 */
  nullTest(c: Expr): { id: number; nonNull: boolean } | null {
    if (c.kind !== 'binary' || (c.op !== '!=' && c.op !== '==')) return null;
    const v = c.left.kind === 'null' ? c.right : c.right.kind === 'null' ? c.left : null;
    if (!v || v.kind !== 'ident') return null;
    const sym = this.lookup(v.name);
    if (!sym || sym.mutable || !this.optInner(sym.ty) || this.narrowed.has(sym.id)) return null;
    return { id: sym.id, nonNull: c.op === '!=' };
  }

  resolve(n: TypeNode): Ty {
    if (n.kind === 'opt') return this.optionalOf(this.resolve(n.inner), n.pos);
    if (n.kind === 'generic') return this.genericStruct(n);
    if (n.kind === 'array' || n.kind === 'heap') {
      const elem = this.resolve(n.elem);
      if (elem.k === 'void') this.err('bad-type', 'array of `void` is not allowed', n.pos);
      if (n.kind === 'array' && n.lenName) n.len = this.constLength(n.lenName, n.pos);
      const t: Ty = n.kind === 'array'
        ? { k: 'array', elem, len: n.len }
        : n.racy ? { k: 'heap', elem, buffer: true, racy: true } : n.buffer ? { k: 'heap', elem, buffer: true } : { k: 'heap', elem };
      if (!this.collecting) this.checkElem(t, n.pos);
      return t;
    }
    if (n.name === 'usize') return this.usize;
    if (n.name === 'string') return STR;
    if (SIMD_TYPES[n.name]) {
      const elem = PRIMS[SIMD_TYPES[n.name]];
      return { k: 'array', elem, len: 128 / (elem as { bits: number }).bits };
    }
    const p = PRIMS[n.name];
    if (p) {
      if (n.name === 'f16') this.usesF16 = true;
      return p;
    }
    if (this.interfaces.has(n.name)) {
      this.err('not-implemented', `an interface cannot be used as the type of a value yet`, n.pos,
        `interfaces constrain type parameters: write \`function f<T extends ${n.name}>(x: T)\``);
    }
    if (this.structs.has(n.name)) {
      if (!n.free) this.visible(n.name, n.pos);
      return { k: 'struct', name: n.name };
    }
    const h = TYPE_HINTS[n.name];
    if (h) this.err(h[0], `type '${n.name}' is not available`, n.pos, h[1]);
    this.err('unknown-type', `unknown type '${n.name}'`, n.pos);
  }

  // ---- 宣言の収集 ----

  collect(mod: Module): FnDecl[] {
    this.importedNames = mod.importedNames ?? new Map();
    for (const d of mod.decls) if (!this.fileOf.has(d.name)) this.fileOf.set(d.name, d.file ?? 0);
    const typeDecls: (StructDecl | ClassDecl)[] = [];
    const fnDecls: FnDecl[] = [];
    for (const d of mod.decls) {
      try {
        if (d.kind === 'fn') {
          if (d.typeParams) {
            // ジェネリック関数は雛形として取っておき、呼び出されたときに具体的な型で複製する。
            if (d.exported) this.err('not-implemented', 'a generic function cannot be exported', d.pos, 'export a non-generic function that calls it');
            if (this.generics.has(d.name) || fnDecls.some((f) => f.name === d.name) || BUILTINS.has(d.name)) {
              this.err('duplicate', `function '${d.name}' is already defined`, d.pos);
            }
            this.generics.set(d.name, d);
          } else {
            if (this.generics.has(d.name)) this.err('duplicate', `function '${d.name}' is already defined`, d.pos);
            fnDecls.push(d);
          }
          continue;
        }
        if (d.kind === 'const') {
          if (this.consts.has(d.name)) this.err('duplicate', `constant '${d.name}' is already defined`, d.pos, 'names are shared by all files of the program; pick another name or remove one of the two');
          this.consts.set(d.name, { decl: d, state: 'new' });
          continue;
        }
        if (this.structs.has(d.name) || this.genericTypes.has(d.name) || this.interfaces.has(d.name) || PRIMS[d.name] || d.name === 'usize') {
          this.err('duplicate', `type '${d.name}' is already defined`, d.pos);
        }
        if (d.kind === 'interface') {
          this.interfaces.set(d.name, d);
          continue;
        }
        if (d.typeParams) {
          this.genericTypes.set(d.name, d);
          continue;
        }
        this.structs.set(d.name, { name: d.name, isClass: d.kind === 'class', fields: [], methods: new Map(), variants: d.kind === 'struct' ? d.variants : undefined });
        typeDecls.push(d);
      } catch (e) {
        this.note(e);
      }
    }

    // 定数の名前は、関数・型の名前と重なってはいけない（名前はすべてのファイルで1つの空間を共有する）。
    for (const c of [...this.consts.values()]) {
      const name = c.decl.name;
      const other = fnDecls.some((f) => f.name === name) || this.generics.has(name) || BUILTINS.has(name) ? 'a function'
        : this.structs.has(name) || this.genericTypes.has(name) || this.interfaces.has(name) || PRIMS[name] || name === 'usize' || name === 'string' ? 'a type' : null;
      if (other) {
        this.consts.delete(name);
        // 名前の持ち主は、残る方（関数・型）の宣言にする。定数のままだと、その関数・型を使っている
        // ほかのファイル（標準の関数を含む）で「import していない」という誤ったエラーが続く。
        const rest = mod.decls.find((x) => x !== c.decl && x.kind !== 'const' && x.name === name);
        if (rest) this.fileOf.set(name, rest.file ?? 0);
        this.errors.push(new CompileError('duplicate', `'${name}' is already defined as ${other}`, c.decl.pos, 'pick another name for the constant'));
      }
    }

    this.collecting = true;
    for (const d of typeDecls) {
      try {
        const info = this.structs.get(d.name)!;
        this.curFile = d.file ?? 0;
        const seen = new Set<string>();
        for (const f of d.fields) {
          if (seen.has(f.name)) this.err('duplicate', `field '${f.name}' is defined twice`, f.pos);
          seen.add(f.name);
          const ty = this.resolve(f.tyNode);
          if (ty.k === 'void') this.err('bad-type', 'a field cannot be `void`', f.pos);
          const cf = d.kind === 'class' ? (f as ClassDecl['fields'][number]) : null;
          info.fields.push({ name: f.name, ty, isPrivate: cf?.isPrivate ?? false, init: cf?.init });
        }
        if (info.fields.length === 0) this.err('bad-type', `'${d.name}' has no fields`, d.pos, 'add at least one field');
      } catch (e) {
        this.note(e);
      }
    }
    this.collecting = false;
    for (const p of this.pendingTypeChecks) {
      try {
        this.checkInstance(p.name, p.pos);
      } catch (e) {
        this.note(e);
      }
    }
    for (const d of typeDecls) {
      try {
        this.checkFiniteSize({ k: 'struct', name: d.name }, [], d.pos);
        const info = this.structs.get(d.name)!;
        d.fields.forEach((f, i) => {
          if (info.fields[i]) this.checkElem(info.fields[i].ty, f.pos);
        });
        this.checkValueSize({ k: 'struct', name: d.name }, d.pos);
      } catch (e) {
        this.note(e);
      }
    }

    // まだ値を求めていない定数（どこからも使われていないものを含む）を、ここですべて検査する。
    for (const name of this.consts.keys()) {
      try {
        this.constInfo(name);
      } catch (e) {
        this.note(e);
      }
    }

    const out: FnDecl[] = [];
    this.allDecls = out;
    for (const d of fnDecls) {
      try {
        if (this.fns.has(d.name) || BUILTINS.has(d.name)) {
          this.err('duplicate', `function '${d.name}' is already defined`, d.pos);
        }
        this.fns.set(d.name, this.signature(d, d.name));
        out.push(d);
      } catch (e) {
        this.note(e);
      }
    }

    for (const d of typeDecls) {
      if (d.kind === 'class') this.registerClass(d);
    }
    return out;
  }

  interfaces = new Map<string, InterfaceDecl>();

  // ---- 最上位の定数 ----

  consts = new Map<string, ConstInfo>();

  /**
   * 定数の型と値を求める（1回だけ）。初期化の式は、リテラル・他の定数・単項と二項の演算子・`as`・かっこに限る。
   * 式は `const x: T = 式` と同じ規則で型を検査し、実行時と同じ意味（整数はラップ、小数→整数は飽和）でここで計算する。
   * 実行時なら停止する計算（整数のゼロ除算、ビット幅以上のシフト）はコンパイルエラーにする。
   */
  constInfo(name: string): ConstInfo {
    const c = this.consts.get(name)!;
    if (c.state === 'done') return c;
    if (c.state === 'new' && !this.constOrdering) this.constDeps(name);
    const d = c.decl;
    if (c.state === 'bad') throw new CompileError(QUIET, '', d.pos);
    if (c.state === 'busy') {
      this.err('const-cycle', `constant '${name}' is defined in terms of itself`, d.pos, 'a constant may only use constants that do not depend on it');
    }
    c.state = 'busy';
    // 関数の検査の途中で呼ばれることがあるので、検査中の状態を退避する。
    const saved = { scopes: this.scopes, file: this.curFile, kernel: this.inKernel, ctor: this.ctorInit, target: this.assignTarget, narrowed: this.narrowed, init: this.constInit };
    this.scopes = [new Map()];
    this.curFile = d.file ?? 0;
    this.inKernel = false;
    this.ctorInit = null;
    this.assignTarget = null;
    this.narrowed = new Set();
    try {
      this.constTy = true;
      let ty: Ty;
      try {
        ty = this.resolve(d.tyNode);
      } finally {
        this.constTy = false;
      }
      let leaf = ty;
      while (leaf.k === 'array') leaf = leaf.elem;
      const scalarOk = leaf.k === 'int' || leaf.k === 'bool' || (leaf.k === 'float' && leaf.bits >= 32);
      const okTy = scalarOk || (isStr(ty) && leaf === ty);
      if (!okTy) {
        this.err('not-implemented', `a top-level constant of type ${tyStr(ty)} is not implemented yet`, d.pos,
          leaf.k === 'float' ? 'declare it as `f32` and convert where it is used (`X as f16`)'
            : leaf.k === 'heap' && !isStr(leaf) ? `a constant table must have a fixed length: write \`${tyStr(leaf.elem)}[N]\` (e.g. \`const T: i32[3] = [1, 2, 3];\`)`
            : 'top-level constants can be integers, `f32`, `f64`, `bool`, `string`, or fixed-length arrays (`T[N]`, `T[N][M]`) of integers, `f32`, `f64` and `bool`; build other values in a function that returns them');
      }
      this.constShape(d.init);
      this.constInit = true;
      this.expect(d.init, ty);
      c.value = this.constEval(d.init);
      c.ty = ty;
      c.state = 'done';
    } catch (e) {
      c.state = 'bad';
      throw e;
    } finally {
      this.scopes = saved.scopes;
      this.curFile = saved.file;
      this.inKernel = saved.kernel;
      this.ctorInit = saved.ctor;
      this.assignTarget = saved.target;
      this.narrowed = saved.narrowed;
      this.constInit = saved.init;
    }
    return c;
  }

  /** 定数の型を求めている途中か（配列の大きさの上限を、定数の配列のものにする）。 */
  constTy = false;
  /** 定数の初期化の式を検査している途中か（定数の配列を値として使っても大きさを問わない）。 */
  constInit = false;
  /** 添字の対象になっている式（`T[i]` の `T`）。定数の配列は、ここではコピーせずに読むので大きさを問わない。 */
  constIndexBases = new WeakSet<Expr>();
  /** 定数の配列を指す変数（定数ごとに1つ。コード生成は読み取り専用のグローバルな領域にする）。 */
  constSyms = new Map<string, Sym>();

  constSym(c: ConstInfo): Sym {
    let s = this.constSyms.get(c.decl.name);
    if (!s) {
      s = { id: this.nextSym++, name: c.decl.name, ty: c.ty!, mutable: false, owned: false, loopDepth: 0, global: { name: c.decl.name, value: c.value } };
      this.constSyms.set(c.decl.name, s);
    }
    return s;
  }

  constOrdering = false;

  /**
   * 定数 root が使っている定数を、使われる側から先に求めておく。
   * 後ろで宣言した定数を次々に使う長い並びでも、constInfo の再帰が深くならないようにするため
   * （再帰に任せると、数百個の並びでコンパイラのスタックがあふれる）。
   */
  constDeps(root: string): void {
    const deps = (name: string): string[] => {
      const out: string[] = [];
      const todo: Expr[] = [this.consts.get(name)!.decl.init];
      while (todo.length > 0) {
        const e = todo.pop()!;
        if (e.kind === 'ident') {
          if (this.consts.has(e.name)) out.push(e.name);
        } else if (e.kind === 'unary' || e.kind === 'cast') todo.push(e.expr);
        else if (e.kind === 'binary') todo.push(e.right, e.left);
        else if (e.kind === 'arrayLit') todo.push(...e.elems);
        else if (e.kind === 'index') todo.push(e.index, e.base);
      }
      return out;
    };
    const stack = [{ name: root, deps: deps(root), i: 0 }];
    const onPath = new Set([root]);
    while (stack.length > 0) {
      const top = stack[stack.length - 1];
      if (top.i < top.deps.length) {
        const dep = top.deps[top.i++];
        const dc = this.consts.get(dep)!;
        if (onPath.has(dep)) {
          // 循環。たどってきた定数はどれも値が決まらない。
          for (const f of stack) this.consts.get(f.name)!.state = 'bad';
          this.errors.push(new CompileError('const-cycle', `constant '${dep}' is defined in terms of itself`, dc.decl.pos,
            'a constant may only use constants that do not depend on it'));
          return;
        }
        if (dc.state !== 'new') continue;
        onPath.add(dep);
        stack.push({ name: dep, deps: deps(dep), i: 0 });
        continue;
      }
      stack.pop();
      onPath.delete(top.name);
      if (stack.length === 0) return; // root 自身は呼び出し側（constInfo）が求める
      this.constOrdering = true;
      try {
        this.constInfo(top.name);
      } catch (e) {
        this.note(e);
      } finally {
        this.constOrdering = false;
      }
    }
  }

  /** 定数の初期化に書ける形かどうかを確かめる。 */
  constShape(e: Expr): void {
    const hint = 'a top-level constant may use literals, array literals, other top-level constants (and their elements `T[i]`), operators, `as` and parentheses; compute anything else inside a function';
    switch (e.kind) {
      case 'int':
      case 'float':
      case 'bool':
      case 'str':
        return;
      case 'ident':
        if (!this.consts.has(e.name)) {
          this.err('const-expr', `'${e.name}' is not a top-level constant, so it cannot be used in a constant`, e.pos, hint);
        }
        return;
      case 'unary':
      case 'cast':
        return this.constShape(e.expr);
      case 'binary':
        if (e.op === '??') break;
        this.constShape(e.left);
        return this.constShape(e.right);
      case 'arrayLit':
        // 固定長配列の定数の要素（入れ子の配列も可）。可変長配列の定数はない（型の検査が弾く）。
        for (const x of e.elems) this.constShape(x);
        return;
      case 'index':
        // 定数の配列の要素: `T[3]`、`T[N - 1][0]`。
        this.constShape(e.base);
        return this.constShape(e.index);
    }
    const what = e.kind === 'call' || e.kind === 'mcall' || e.kind === 'new' || e.kind === 'newArray' ? 'a call'
      : e.kind === 'object' ? 'an object literal' : e.kind === 'cond' ? '`? :`'
      : e.kind === 'null' ? '`null`' : 'this expression';
    this.err('const-expr', `${what} cannot be used in a top-level constant`, e.pos, hint);
  }

  /** 型検査を終えた定数式の値を求める。意味は生成コード（kernel でない関数）と同じ。 */
  constEval(e: Expr): ConstValue {
    const ty = e.ty!;
    if (e.kind === 'arrayLit') return e.elems.map((x) => this.constEval(x));
    if (e.kind === 'ident') {
      // 定数の配列（ほかの定数はリテラルに置き換え済み）。値は共有せず、写しを返す。
      const g = e.sym?.global;
      if (!g) throw new Error(`internal: constant expression ident ${e.name}`);
      return structuredClone(g.value as ConstValue[]);
    }
    if (e.kind === 'index') {
      const base = this.constEval(e.base) as ConstValue[];
      const i = this.constEval(e.index) as bigint;
      if (i < 0n || i >= BigInt(base.length)) {
        this.err('index-range', `index ${i} is out of bounds for ${tyStr(e.base.ty!)} in a top-level constant`, e.index.pos);
      }
      return base[Number(i)];
    }
    if (ty.k === 'float' && ty.bits < 32) {
      this.err('not-implemented', `${tyStr(ty)} cannot be used in a top-level constant yet`, e.pos, 'compute with `f32` and convert where the constant is used');
    }
    const wrap = (v: bigint, t: Ty): bigint => {
      const it = t as Ty & { k: 'int' };
      return it.signed ? BigInt.asIntN(it.bits, v) : BigInt.asUintN(it.bits, v);
    };
    const fl = (v: number, t: Ty): number => (t.k === 'float' && t.bits === 32 ? Math.fround(v) : v);
    switch (e.kind) {
      case 'int':
        return ty.k === 'float' ? fl(Number(e.value), ty) : e.value;
      case 'float':
        return fl(e.value, ty);
      case 'bool':
        return e.value;
      case 'str':
        return e.value;
      case 'unary': {
        const x = this.constEval(e.expr);
        if (e.op === '!') return !x;
        if (e.op === '~') return wrap(~(x as bigint), ty);
        return typeof x === 'bigint' ? wrap(-x, ty) : -(x as number);
      }
      case 'cast': {
        const from = e.expr.ty!;
        const x = this.constEval(e.expr);
        if (tyEq(from, ty)) return x;
        if (from.k === 'bool') return x ? 1n : 0n;
        if (from.k === 'int' && ty.k === 'int') return wrap(x as bigint, ty);
        if (from.k === 'int') return ty.k === 'float' && ty.bits === 32 ? bigToF32(x as bigint) : Number(x as bigint);
        if (ty.k === 'float') return fl(x as number, ty);
        // 小数 → 整数: 0 の方向へ切り捨て、範囲外は飽和、NaN は 0（生成コードと同じ）。
        const [lo, hi] = intRange(ty as Ty & { k: 'int' });
        const v = x as number;
        if (Number.isNaN(v)) return 0n;
        if (v === Infinity) return hi;
        if (v === -Infinity) return lo;
        const n = BigInt(Math.trunc(v));
        return n < lo ? lo : n > hi ? hi : n;
      }
      case 'binary': {
        const op = e.op;
        const l = this.constEval(e.left);
        // 実行時と同じく、`&&` / `||` は左辺で結果が決まれば右辺を計算しない（右辺のゼロ除算などは起きない）。
        if (op === '&&' && l === false) return false;
        if (op === '||' && l === true) return true;
        const r = this.constEval(e.right);
        if (op === '&&' || op === '||') return r;
        switch (op) {
          case '==': return l === r;
          case '!=': return l !== r;
          case '<': return (l as number) < (r as number);
          case '<=': return (l as number) <= (r as number);
          case '>': return (l as number) > (r as number);
          case '>=': return (l as number) >= (r as number);
        }
        if (typeof l === 'string') return l + (r as string);
        if (typeof l === 'number') {
          const b = r as number;
          const v = op === '+' ? l + b : op === '-' ? l - b : op === '*' ? l * b : op === '/' ? l / b : l % b;
          return fl(v, ty);
        }
        const a = l as bigint;
        const b = r as bigint;
        const it = ty as Ty & { k: 'int' };
        switch (op) {
          case '+': return wrap(a + b, ty);
          case '-': return wrap(a - b, ty);
          case '*': return wrap(a * b, ty);
          case '&': return wrap(a & b, ty);
          case '|': return wrap(a | b, ty);
          case '^': return wrap(a ^ b, ty);
          case '/':
          case '%':
            if (b === 0n) this.err('const-expr', 'division by zero in a top-level constant', e.pos, 'at run time this would stop the program; fix the divisor');
            return wrap(op === '/' ? a / b : a % b, ty);
          case '<<':
          case '>>':
            if (b < 0n || b >= BigInt(it.bits)) {
              this.err('const-expr', `shift amount ${b} is out of range for ${tyStr(ty)} in a top-level constant`, e.pos,
                `the amount must be between 0 and ${it.bits - 1}; widen the value first, e.g. \`(1 as u64) << 40\``);
            }
            return wrap(op === '<<' ? a << b : a >> b, ty);
        }
        break;
      }
    }
    throw new Error(`internal: constant expression ${e.kind}`);
  }

  /** 定数を使う所に置く値（型の決まったリテラル）。 */
  constLiteral(c: ConstInfo, pos: Pos): Record<string, unknown> {
    const ty = c.ty!;
    const v = c.value!;
    if (ty.k === 'int') return { kind: 'int', value: v, fixed: ty };
    if (ty.k === 'float') {
      if (this.inKernel && !Number.isFinite(v as number)) {
        this.err('kernel-restriction', `constant '${c.decl.name}' is not a finite number, so it cannot be used in a kernel function`, pos,
          'GPU shader languages have no portable literal for infinity or NaN');
      }
      return { kind: 'float', value: v, fixed: ty };
    }
    if (ty.k === 'bool') return { kind: 'bool', value: v };
    return { kind: 'str', value: v };
  }

  /** `T[N]` の N。正の整数の定数でなければならない。 */
  constLength(name: string, pos: Pos): number {
    if (!this.consts.has(name)) {
      this.err('unknown-name', `unknown constant '${name}' used as an array length`, pos,
        `declare it at the top level: \`const ${name}: usize = 16;\` (a local variable cannot be a length; use \`T[]\` for a length known only at run time)`);
    }
    if (this.lookup(name)) {
      this.err('bad-type', `array length '${name}' is a local variable here, not the top-level constant of the same name`, pos,
        `a local variable cannot be a length; rename the local, or use \`T[]\` for a length known only at run time`);
    }
    this.visible(name, pos);
    const c = this.constInfo(name);
    const v = c.value;
    if (c.ty!.k !== 'int' || typeof v !== 'bigint') {
      this.err('bad-type', `array length '${name}' must be an integer constant but it is ${tyStr(c.ty!)}`, pos, `declare it as an integer, e.g. \`const ${name}: usize = 16;\``);
    }
    if (v <= 0n || v > BigInt(Number.MAX_SAFE_INTEGER)) this.err('bad-type', `array length must be a positive integer but '${name}' is ${v}`, pos, 'a fixed-length array has at least one element; use `T[]` for an array that can be empty');
    return Number(v);
  }

  /** class が、`implements` に書いた interface のメソッドをすべて同じ形で持っているかを確かめる。 */
  checkImplements(d: ClassDecl): void {
    const info = this.structs.get(d.name)!;
    info.implements = [];
    for (const imp of d.implements ?? []) {
      const iface = this.interfaces.get(imp.name);
      if (!iface) this.err('unknown-type', `unknown interface '${imp.name}'`, imp.pos);
      this.visible(imp.name, imp.pos);
      const file = this.curFile;
      this.curFile = iface.file ?? 0;
      const wanted = iface.methods.map((m) => ({
        m, params: m.params.map((p) => ({ mode: p.mode, ty: this.resolve(p.tyNode) })), ret: m.retNode ? this.resolve(m.retNode) : VOID,
      }));
      this.curFile = file;
      for (const w of wanted) {
        const shape = `${w.m.name}(${w.params.map((p) => `${p.mode === 'none' ? '' : p.mode + ' '}${tyStr(p.ty)}`).join(', ')}): ${tyStr(w.ret)}`;
        const got = info.methods.get(w.m.name);
        if (!got) {
          this.err('interface-mismatch', `class '${tyStr({ k: 'struct', name: d.name })}' says it implements '${imp.name}' but has no method '${w.m.name}'`, imp.pos,
            `add \`${shape}\``);
        }
        const same = got.params.length === w.params.length && tyEq(got.ret, w.ret)
          && got.params.every((p, i) => p.mode === w.params[i].mode && tyEq(p.ty, w.params[i].ty));
        if (!same) {
          this.err('interface-mismatch', `method '${w.m.name}' of '${tyStr({ k: 'struct', name: d.name })}' does not match interface '${imp.name}'`, got.decl.pos,
            `the interface asks for \`${shape}\``);
        }
      }
      info.implements.push(imp.name);
    }
  }

  /** class のフィールドの初期値を検査し、コンストラクタとメソッドを登録する。フィールドの型は登録済みであること。 */
  registerClass(d: ClassDecl): void {
    const out = this.allDecls;
    this.curFile = d.file ?? 0;
    const info = this.structs.get(d.name)!;
    const self: Ty = { k: 'struct', name: d.name };

    // フィールドの初期値。this や変数は参照できない。
    for (const f of info.fields) {
      if (!f.init) continue;
      try {
        this.scopes = [new Map()];
        this.expect(f.init, f.ty);
      } catch (e) {
        this.note(e);
      }
    }

    try {
      // コンストラクタを書かなかった場合は、初期値だけで作る暗黙のコンストラクタを用意する。
      const ctor: FnDecl = d.ctor ?? {
        kind: 'fn', name: 'constructor', exported: false, params: [], body: [], pos: d.pos, cls: d.name, isCtor: true, file: d.file,
      };
      info.ctor = this.signature(ctor, `${d.name}.new`);
      info.ctor.ret = self;
      out.push(ctor);
    } catch (e) {
      this.note(e);
    }

    for (const m of d.methods) {
      try {
        if (info.methods.has(m.name)) this.err('duplicate', `method '${m.name}' is defined twice`, m.pos);
        if (info.fields.some((f) => f.name === m.name)) {
          this.err('duplicate', `'${m.name}' is both a field and a method of '${d.name}'`, m.pos);
        }
        info.methods.set(m.name, this.signature(m, `${d.name}.${m.name}`));
        out.push(m);
      } catch (e) {
        this.note(e);
      }
    }
    try {
      this.checkImplements(d);
    } catch (e) {
      this.note(e);
    }
  }

  /** 名前 → それを宣言したファイルの番号。 */
  fileOf = new Map<string, number>();
  importedNames = new Map<number, Set<string>>();
  /** いま検査している宣言のファイル。 */
  curFile = 0;

  /**
   * 別の宣言（ジェネリックの雛形など）に書かれた型を、その宣言の場所のものとして読む。
   * 呼び出し側のファイル（import の見え方）とローカル変数は、その間は見えない。
   */
  atHome<R>(file: number, f: () => R): R {
    const saved = { file: this.curFile, scopes: this.scopes };
    this.curFile = file;
    this.scopes = [new Map()];
    try {
      return f();
    } finally {
      this.curFile = saved.file;
      this.scopes = saved.scopes;
    }
  }

  /** 他のファイルの名前は、import していなければ使えない。標準の関数（ファイル -1）はどこからでも使える。 */
  visible(name: string, pos: Pos): void {
    const owner = this.fileOf.get(name);
    if (owner === undefined || owner === -1 || owner === this.curFile) return;
    if (this.importedNames.get(this.curFile)?.has(name)) return;
    this.err('not-imported', `'${name}' is defined in another file and is not imported here`, pos,
      `add \`import { ${name} } from "./<file>.um";\` at the top (the other file must mark it with \`export\`)`);
  }

  /** ジェネリック関数の雛形（名前 → 宣言）。 */
  generics = new Map<string, FnDecl>();
  /** 具体化した関数（「名前<型, ...>」→ 生成した関数の名前）。 */
  instances = new Map<string, string>();
  /** 検査する関数の一覧。具体化した関数をここへ足すと、あとで本体が検査される。 */
  allDecls: FnDecl[] = [];

  /** 型を、ソースに書いたときの形（TypeNode）に戻す。雛形の型引数を置き換えるのに使う。 */
  tyNode(t: Ty, pos: Pos): TypeNode {
    if (t.k === 'array') return { kind: 'array', elem: this.tyNode(t.elem, pos), len: t.len, pos };
    if (t.k === 'heap' && !t.str) return { kind: 'heap', elem: this.tyNode(t.elem, pos), pos, buffer: t.buffer, racy: t.racy };
    // struct は内部の名前で指す（`T | null` やジェネリックな struct は、表示用の名前と内部の名前が違う）。
    return { kind: 'name', name: t.k === 'struct' ? t.name : tyStr(t), pos, free: true };
  }

  /**
   * ジェネリック関数の呼び出しを、具体的な型の関数の呼び出しに書き換える。
   * 型引数は実引数の型から決める。同じ型の組み合わせは1つの関数を共有する。
   */
  instantiate(e: Expr & { kind: 'call' }, expected?: Ty): void {
    const tmpl = this.generics.get(e.callee)!;
    const names = tmpl.typeParams!;
    if (e.args.length !== tmpl.params.length) {
      this.err('arg-count', `'${tmpl.name}' takes ${tmpl.params.length} argument(s) but ${e.args.length} were given`, e.pos);
    }
    const bound = new Map<string, Ty>();
    const unify = (n: TypeNode, t: Ty, pos: Pos): void => {
      if (n.kind === 'name') {
        if (!names.includes(n.name)) return;
        const prev = bound.get(n.name);
        if (!prev) bound.set(n.name, t);
        else if (!tyEq(prev, t)) {
          this.err('type-mismatch', `type parameter '${n.name}' of '${tmpl.name}' is ${tyStr(prev)} from an earlier argument but ${tyStr(t)} here`, pos,
            'convert one of the arguments with `as` so that both have the same type');
        }
      } else if (n.kind === 'array' && t.k === 'array') {
        unify(n.elem, t.elem, pos);
      } else if (n.kind === 'heap' && t.k === 'heap' && !t.str) {
        unify(n.elem, t.elem, pos);
      } else if (n.kind === 'generic' && t.k === 'struct') {
        const g = this.structs.get(t.name)!.generic;
        if (g && g.name === n.name && g.args.length === n.args.length) n.args.forEach((a, i) => unify(a, g.args[i], pos));
      } else if (n.kind === 'opt') {
        const inner = this.optInner(t);
        if (inner) unify(n.inner, inner, pos);
      }
    };
    // 型のはっきりした引数から先に決める。数のリテラルは、他で決まらなかったときだけ使う。
    const literal = (x: Expr) => isLiteral(x) || x.kind === 'arrayLit' || x.kind === 'object';
    for (const pass of [false, true]) {
      // リテラルだけでは決まらない型引数は、呼び出しの結果に求められている型から決める
      // （`const v: vec2<f32> = vec2(3, 4)` の 3 と 4 を f32 にする）。
      if (pass && expected && tmpl.retNode) {
        const before = new Map(bound);
        const soft = new Map<string, Ty>();
        const saved = this.errors.length;
        try {
          const all = new Map<string, Ty>();
          const collect = (n: TypeNode, t: Ty): void => {
            if (n.kind === 'name' && names.includes(n.name)) all.set(n.name, t);
            else if ((n.kind === 'array' && t.k === 'array') || (n.kind === 'heap' && t.k === 'heap' && !t.str)) collect(n.elem, t.elem);
            else if (n.kind === 'generic' && t.k === 'struct') {
              const g = this.structs.get(t.name)!.generic;
              if (g && g.name === n.name && g.args.length === n.args.length) n.args.forEach((a, i) => collect(a, g.args[i]));
            } else if (n.kind === 'opt') {
              const inner = this.optInner(t);
              if (inner) collect(n.inner, inner);
            }
          };
          collect(tmpl.retNode, expected);
          for (const [k, v] of all) if (!before.has(k)) soft.set(k, v);
        } catch {
          this.errors.length = saved;
        }
        for (const [k, v] of soft) bound.set(k, v);
      }
      tmpl.params.forEach((p, i) => {
        const a = e.args[i].expr;
        if (literal(a) !== pass) return;
        if (a.kind === 'object') return; // 型が分からないと検査できない。あとの通常の検査に任せる
        // 引数の型は雛形のファイルに書かれている（そこでだけ見える名前を使っていてよい）。
        const want = pass ? this.atHome(tmpl.file ?? 0, () => this.partial(p.tyNode, bound, names)) : undefined;
        if (pass && a.kind === 'arrayLit' && !want) return;
        unify(p.tyNode, this.checkExpr(a, want), a.pos);
      });
    }
    const missing = names.filter((n) => !bound.has(n));
    if (missing.length > 0) {
      this.err('needs-type', `cannot tell what '${missing.join("', '")}' is in this call to '${tmpl.name}'`, e.pos,
        'type parameters are taken from the argument types; give the argument a clear type (a typed variable or `as`)');
    }

    // `T extends Shape`: T に入る型は、その interface を implements した class でなければならない。
    for (const [param, ifaceName] of Object.entries(tmpl.typeBounds ?? {})) {
      const t = bound.get(param)!;
      const ok = t.k === 'struct' && this.structs.get(t.name)!.implements?.includes(ifaceName);
      if (!ok) {
        this.err('constraint', `'${tmpl.name}' needs a type that implements '${ifaceName}', but '${param}' is ${tyStr(t)} here`, e.pos,
          t.k === 'struct' ? `declare it as \`class ${tyStr(t)} implements ${ifaceName} { ... }\`` : `only a class can implement an interface`);
      }
    }
    const key = `${tmpl.name}<${names.map((n) => tyStr(bound.get(n)!)).join(', ')}>`;
    let inst = this.instances.get(key);
    if (!inst) {
      if (this.instances.size >= 500) {
        this.err('too-many-instances', `'${tmpl.name}' is being instantiated with ever more types`, e.pos, 'a generic function must not call itself with a larger type each time');
      }
      inst = `${tmpl.name}__${names.map((n) => tyStr(bound.get(n)!).replace(/[^A-Za-z0-9]+/g, '_')).join('_')}`;
      while (this.fns.has(inst) || this.generics.has(inst)) inst += '_';
      this.instances.set(key, inst);
      const copy = structuredClone(tmpl);
      copy.name = inst;
      delete copy.typeParams;
      this.substTypes(copy, bound);
      // kernel の雛形でも、kernel で使えない型（f64 など）で具体化したものは普通の関数にする。
      if (copy.isKernel && ![...bound.values()].every((t) => this.isKernelValue(t))) copy.isKernel = false;
      // signature() は検査中のファイルを切り替える。呼び出し側のローカル変数も見せない。
      this.fns.set(inst, this.atHome(this.curFile, () => this.signature(copy, inst)));
      this.allDecls.push(copy);
    }
    e.callee = inst;
  }

  /** 雛形の複製の中で、型引数の名前を具体的な型に置き換える。 */
  substTypes(node: unknown, bound: Map<string, Ty>): unknown {
    if (Array.isArray(node)) {
      node.forEach((x, i) => (node[i] = this.substTypes(x, bound)));
      return node;
    }
    if (node === null || typeof node !== 'object') return node;
    const o = node as Record<string, unknown>;
    if (o.kind === 'name' && typeof o.name === 'string' && bound.has(o.name)) return this.tyNode(bound.get(o.name)!, o.pos as Pos);
    for (const k of Object.keys(o)) o[k] = this.substTypes(o[k], bound);
    return o;
  }

  /** ジェネリックな struct の雛形（名前 → 宣言）と、具体化したもの（「名前<型, ...>」→ 内部の名前）。 */
  genericTypes = new Map<string, StructDecl | ClassDecl>();
  typeInstances = new Map<string, string>();
  /** 型の収集中に具体化した struct。大きさなどの検査は、全部の型が揃ってから行う。 */
  pendingTypeChecks: { name: string; pos: Pos }[] = [];

  /** `Pair<i32, f64>` のような型。型の組み合わせごとに struct を1つ作る。 */
  genericStruct(n: TypeNode & { kind: 'generic' }): Ty {
    const tmpl = this.genericTypes.get(n.name);
    if (!tmpl && (n.name === 'Thread' || n.name === 'Threads') && !this.structs.has(n.name)) {
      this.err('thread', `a \`${n.name}<...>\` handle cannot be stored, passed or returned`, n.pos,
        'the type can only annotate the `const` that starts the thread: `const h: Thread<i32> = thread.spawn(f, ...);`; to start many threads, use `thread.spawnEach`');
    }
    if (!tmpl) {
      const plain = this.structs.has(n.name) ? `'${n.name}' takes no type arguments` : `unknown generic type '${n.name}'`;
      this.err('unknown-type', plain, n.pos, 'declare it as `struct Name<T> { ... }`; classes and enums cannot be generic yet');
    }
    this.visible(n.name, n.pos);
    const names = tmpl.typeParams!;
    if (n.args.length !== names.length) {
      this.err('arg-count', `'${n.name}' takes ${names.length} type argument(s) but ${n.args.length} were given`, n.pos);
    }
    const args = n.args.map((a) => this.resolve(a));
    for (const a of args) if (a.k === 'void') this.err('bad-type', '`void` cannot be a type argument', n.pos);
    const key = `${n.name}<${args.map(tyStr).join(', ')}>`;
    let name = this.typeInstances.get(key);
    if (!name) {
      if (this.typeInstances.size >= 500) this.err('too-many-instances', `'${n.name}' is being instantiated with ever more types`, n.pos);
      name = `${n.name}__${args.map((a) => tyStr(a).replace(/[^A-Za-z0-9]+/g, '_')).join('_')}`;
      while (this.structs.has(name) || this.genericTypes.has(name)) name += '_';
      this.typeInstances.set(key, name);
      displayNames.set(name, key);
      const copy = structuredClone(tmpl);
      this.substTypes(copy, new Map(names.map((p, i) => [p, args[i]])));
      const isClass = copy.kind === 'class';
      if (copy.kind === 'class') {
        // class の複製は、名前と、メソッドが属する class の名前を付け替える。
        copy.name = name;
        delete copy.typeParams;
        if (copy.ctor) copy.ctor.cls = name;
        for (const m of copy.methods) m.cls = name;
      }
      const info: StructInfo = {
        name, isClass, fields: [], methods: new Map(), generic: { name: n.name, args },
        variants: tmpl.kind === 'struct' ? tmpl.variants : undefined,
      };
      this.structs.set(name, info); // 先に登録する（フィールドが自分自身の配列を持てるように）
      const file = this.curFile;
      const outer = this.scopes; // 型を使った側のローカル変数は、雛形の中からは見えない
      this.curFile = tmpl.file ?? 0;
      this.scopes = [new Map()];
      const seen = new Set<string>();
      for (const f of copy.fields) {
        if (seen.has(f.name)) this.err('duplicate', `field '${f.name}' is defined twice`, f.pos);
        seen.add(f.name);
        const ty = this.resolve(f.tyNode);
        if (ty.k === 'void') this.err('bad-type', 'a field cannot be `void`', f.pos);
        const cf = copy.kind === 'class' ? (f as ClassDecl['fields'][number]) : null;
        info.fields.push({ name: f.name, ty, isPrivate: cf?.isPrivate ?? false, init: cf?.init });
      }
      if (copy.kind === 'class') {
        // 関数の検査の途中で呼ばれることがあるので、検査中の状態を壊さないよう退避する。
        this.registerClass(copy);
      }
      this.scopes = outer;
      this.curFile = file;
      if (info.fields.length === 0) this.err('bad-type', `'${n.name}' has no fields`, tmpl.pos, 'add at least one field');
      if (this.collecting) this.pendingTypeChecks.push({ name, pos: n.pos });
      else this.checkInstance(name, n.pos);
    }
    return { k: 'struct', name };
  }

  checkInstance(name: string, pos: Pos): void {
    const t: Ty = { k: 'struct', name };
    this.checkFiniteSize(t, [], pos);
    for (const f of this.structs.get(name)!.fields) this.checkElem(f.ty, pos);
    this.checkValueSize(t, pos);
  }

  /** 型引数がすべて決まっていれば、その TypeNode の型を返す。決まっていなければ undefined。 */
  partial(n: TypeNode, bound: Map<string, Ty>, names: string[]): Ty | undefined {
    if (n.kind === 'name') return names.includes(n.name) ? bound.get(n.name) : this.resolve(n);
    if (n.kind === 'opt') {
      const inner = this.partial(n.inner, bound, names);
      return inner ? this.optionalOf(inner, n.pos) : undefined;
    }
    if (n.kind === 'generic') {
      const args = n.args.map((a) => this.partial(a, bound, names));
      if (args.some((a) => !a)) return undefined;
      return this.genericStruct({ ...n, args: args.map((a) => this.tyNode(a!, n.pos)) });
    }
    const elem = this.partial(n.elem, bound, names);
    if (!elem) return undefined;
    if (n.kind === 'array' && n.lenName) n.len = this.constLength(n.lenName, n.pos);
    return n.kind === 'array' ? { k: 'array', elem, len: n.len }
      : n.racy ? { k: 'heap', elem, buffer: true, racy: true } : n.buffer ? { k: 'heap', elem, buffer: true } : { k: 'heap', elem };
  }

  signature(d: FnDecl, key: string): FnSig {
    this.curFile = d.file ?? (d.cls ? this.fileOf.get(d.cls) ?? 0 : 0);
    const seen = new Set<string>();
    const params = d.params.map((p) => {
      if (seen.has(p.name)) this.err('duplicate', `parameter '${p.name}' is defined twice`, p.pos);
      seen.add(p.name);
      const ty = this.resolve(p.tyNode);
      if (ty.k === 'void') this.err('bad-type', 'a parameter cannot be `void`', p.pos);
      p.ty = ty;
      return { name: p.name, mode: p.mode, ty };
    });
    d.ret = d.retNode ? this.resolve(d.retNode) : VOID;
    if (d.isKernel) this.checkKernelSignature(d);
    return { key, name: d.name, params, ret: d.ret, exported: d.exported, decl: d };
  }

  checkKernelSignature(d: FnDecl): void {
    const hint = 'kernel functions run on the GPU too: use bool, i32, u32, i64, u64, f32, structs / fixed-length arrays of those, and Buffer<T>';
    let writes = false;
    for (const p of d.params) {
      const ty = p.ty!;
      const isBuffer = ty.k === 'heap' && !!ty.buffer;
      if (!isBuffer && !this.isKernelValue(ty)) {
        this.err('kernel-type', `parameter '${p.name}' has type ${tyStr(ty)}, which is not available in a kernel function`, p.pos, hint);
      }
      if (p.mode === 'own') this.err('kernel-restriction', 'a kernel function cannot take ownership (`own`)', p.pos);
      if (p.mode === 'mut' && !isBuffer) {
        this.err('kernel-restriction', `parameter '${p.name}' is \`mut\`, but a kernel can only write to a Buffer`, p.pos,
          'pass a `mut Buffer<T>` and write the result to `buf[i]`');
      }
      if (p.mode === 'mut') writes = true;
    }
    if (d.ret!.k !== 'void' && !this.isKernelValue(d.ret!)) {
      this.err('kernel-type', `return type ${tyStr(d.ret!)} is not available in a kernel function`, d.pos, hint);
    }
    const first = d.params[0]?.ty;
    if (writes && !(first && first.k === 'int' && first.bits === 32 && !first.signed)) {
      this.err('kernel-restriction', `kernel '${d.name}' writes to a Buffer, so its first parameter must be the element index`, d.pos,
        'declare it as `i: u32`; the kernel may only write to `buf[i]`');
    }
  }

  /** 書き込み先の Buffer を持つ kernel か。そうなら第1引数が要素の添字になる。 */
  kernelWrites(d: FnDecl): boolean {
    return !!d.isKernel && d.params.some((p) => p.mode === 'mut');
  }

  /** 構造体が自分自身を値で含むと大きさが無限になるので禁止する。 */
  checkFiniteSize(t: Ty, stack: string[], pos: Pos): void {
    if (t.k === 'array') return this.checkFiniteSize(t.elem, stack, pos);
    if (t.k === 'heap') return; // ヒープ越しなら大きさは有限
    if (t.k !== 'struct') return;
    if (stack.includes(t.name)) {
      this.err('recursive-struct', `struct '${tyStr(t)}' contains itself by value`, pos,
        'a struct cannot hold itself directly; indirection (Box/Rc) arrives in a later phase');
    }
    for (const f of this.structs.get(t.name)!.fields) this.checkFiniteSize(f.ty, [...stack, t.name], pos);
  }

  note(e: unknown): void {
    if (e instanceof CompileError) {
      if (e.code !== QUIET) this.errors.push(e);
    } else throw e;
  }

  // ---- スコープ ----

  declare(name: string, ty: Ty, mutable: boolean, pos: Pos, owned = true): Sym {
    const scope = this.scopes[this.scopes.length - 1];
    if (scope.has(name)) this.err('duplicate', `'${name}' is already declared in this scope`, pos);
    const sym: Sym = { id: this.nextSym++, name, ty, mutable, owned, loopDepth: this.loopDepth };
    scope.set(name, sym);
    return sym;
  }

  lookup(name: string): Sym | undefined {
    for (let i = this.scopes.length - 1; i >= 0; i--) {
      const s = this.scopes[i].get(name);
      if (s) return s;
    }
    return undefined;
  }

  // ---- 関数 ----

  checkFn(d: FnDecl): void {
    this.curFile = d.file ?? (d.cls ? this.fileOf.get(d.cls) ?? 0 : 0);
    this.scopes = [new Map()];
    this.curRet = d.ret!;
    this.curFn = d;
    this.loopDepth = 0;
    this.stmtDepth = 0;
    this.ctorInit = null;
    this.assignTarget = null;
    this.moved = new Set();
    this.narrowed = new Set();
    this.frozen = [];
    this.loops = [];
    this.movedAt = new Map();
    this.inKernel = !!d.isKernel;
    this.idxParam = null;
    if (d.cls) {
      // this は書き換え可能として検査し、実際に書き換えたかどうかを記録する（mutatesThis の推論）。
      d.thisSym = this.declare('this', { k: 'struct', name: d.cls }, true, d.pos, false);
      if (d.isCtor) {
        const info = this.structs.get(d.cls)!;
        this.ctorInit = new Set(info.fields.filter((f) => f.init).map((f) => f.name));
      }
    }
    for (const p of d.params) {
      // 修飾なしの集成体は読み取り借用なので書き換え不可。スカラはコピーなので書き換えてよい。
      const mutable = p.mode !== 'none' || !isAgg(p.ty!);
      p.sym = this.declare(p.name, p.ty!, mutable, p.pos, p.mode === 'own' || !isAgg(p.ty!));
      p.sym.param = true;
    }
    if (this.kernelWrites(d)) {
      // 添字の引数は書き換え不可。「書き込みは buf[i] だけ」を構文で判定できるようにするため。
      this.idxParam = d.params[0].sym!;
      this.idxParam.mutable = false;
    }
    this.scopes.push(new Map());
    for (const s of d.body) this.checkStmt(s);
    this.scopes.pop();
    if (d.isKernel && d.usesGroup) this.checkGroupBody(d);
    if (this.ctorInit) {
      const missing = this.structs.get(d.cls!)!.fields.filter((f) => !this.ctorInit!.has(f.name)).map((f) => f.name);
      this.ctorInit = null;
      if (missing.length > 0) {
        this.err('missing-field', `constructor of '${d.cls}' does not initialize: ${missing.join(', ')}`, d.pos,
          'assign each field at the top level of the constructor (`this.x = ...`), or give the field a default (`x: i32 = 0`)');
      }
      return;
    }
    if (d.ret!.k !== 'void' && !this.blockReturns(d.body)) {
      this.err('missing-return', `function '${d.name}' does not return a value on every path`, d.pos,
        `add a \`return\` of type ${tyStr(d.ret!)} at the end`);
    }
  }

  blockReturns(body: Stmt[]): boolean {
    return body.some((s) => this.stmtReturns(s));
  }

  /** この文の後ろに制御が続かないか（return / break / continue で抜けるか）。 */
  diverges(s: Stmt): boolean {
    if (s.kind === 'return' || s.kind === 'break' || s.kind === 'continue') return true;
    if (s.kind === 'block') return s.body.some((b) => this.diverges(b));
    if (s.kind === 'if') return !!s.else && this.diverges(s.then) && this.diverges(s.else);
    return false;
  }

  /**
   * 値を消費する場所（let の初期値、代入の右辺、return、own 引数、リテラルのフィールド）で呼ぶ。
   * コピーできない値を変数から取り出すと、その変数は移動済みになる。
   */
  consume(e: Expr): void {
    const ty = e.ty!;
    if (this.isCopy(ty)) return;
    // 一時値の一部（関数の結果のフィールドなど）も取り出せない。残りの部分と二重に解放されてしまう。
    if (!isPlace(e) && e.kind !== 'field' && e.kind !== 'index') return;
    const sym = rootSym(e);
    if (e.kind !== 'ident' || !sym) {
      this.err('move-out', `cannot move a value of type ${tyStr(ty)} out of a field or element`, e.pos,
        'it owns heap memory, so it is never copied implicitly; copy an array with `.clone()`, or lend it by passing it to a function parameter');
    }
    if (!sym.owned) {
      this.err('move-out', `cannot move out of '${sym.name}' because it is only borrowed here`, e.pos,
        `copy an array with \`.clone()\`, or take ownership by declaring the parameter \`${sym.name}: own ${tyStr(sym.ty)}\``);
    }
    const busy = this.frozen.find((f) => f.sym === sym);
    if (busy) {
      this.err('exclusive-borrow', `cannot move '${sym.name}' while ${frozenBy(busy)}`, e.pos,
        busy.by === 'thread' ? 'threads read it until they are joined; move it before starting the thread, or in a block after the one that started it' : undefined);
    }
    // ループの中で外側の変数を移動してもよいが、周回の終わりまでに代入し直す必要がある（leaveLoop で検査）。
    this.moved.add(sym);
    this.movedAt.set(sym, e.pos);
  }

  enterLoop(): void {
    this.loopDepth++;
    this.loops.push({ depth: this.loopDepth, before: new Set(this.moved), breakMoved: new Set() });
  }

  /** 周回の終わり（本体の末尾と continue）で、ループの外の変数が移動済みのまま残っていないことを確かめる。 */
  checkLoopMoves(): void {
    const ctx = this.loops[this.loops.length - 1];
    for (const sym of this.moved) {
      if (ctx.before.has(sym) || sym.loopDepth >= ctx.depth) continue;
      this.err('move-in-loop', `'${sym.name}' is moved inside the loop and would be used again on the next iteration`,
        this.movedAt.get(sym) ?? { line: 1, col: 1 },
        `assign a new value to '${sym.name}' before the iteration ends, move it before the loop, or pass a \`.clone()\``);
    }
  }

  leaveLoop(): void {
    this.checkLoopMoves();
    const ctx = this.loops.pop()!;
    this.loopDepth--;
    // ループが0回しか回らなかった場合と、break で抜けた場合を合わせる。
    this.moved = new Set([...this.moved, ...ctx.before, ...ctx.breakMoved]);
  }

  /**
   * place のアドレスは、同じ式の後ろの部分を評価し終えてから求められる。
   * その間に place の根の変数が移動されると、解放済みのメモリを指すことになるので禁止する。
   * before は、後ろの部分を検査する前の移動済み集合。
   */
  notMovedSince(place: Expr, before: Set<Sym>): void {
    const root = rootSym(place);
    if (root && this.moved.has(root) && !before.has(root)) {
      this.err('use-after-move', `'${root.name}' is moved by a later part of this expression while it is still in use`, place.pos,
        'do the move in a separate statement first, or pass `.clone()`');
    }
  }

  stmtReturns(s: Stmt): boolean {
    if (s.kind === 'return') return true;
    if (s.kind === 'block') return this.blockReturns(s.body);
    if (s.kind === 'if') return !!s.else && this.stmtReturns(s.then) && this.stmtReturns(s.else);
    return false;
  }

  // ---- 文 ----

  /**
   * enum の値を作る式を、内部の struct のオブジェクトリテラルに書き換える。
   * 選んだ選択肢の値だけを入れ、他の選択肢の値は 0 / false にする。
   */
  isEnumName(name: string): boolean {
    const tmpl = this.genericTypes.get(name);
    return !!(this.structs.get(name)?.variants || (tmpl?.kind === 'struct' && tmpl.variants));
  }

  /**
   * ジェネリックな enum の値を作る式で、型引数を決めて、具体化した enum の名前を返す。
   * 求められている型がその enum ならそれを使い、なければ値の型から決める。
   */
  enumInstance(e: Expr, enumName: string, variant: string, args: Arg[], want?: Ty): string {
    if (this.structs.has(enumName)) return enumName;
    const tmpl = this.genericTypes.get(enumName) as StructDecl;
    this.visible(enumName, e.pos);
    if (want?.k === 'struct' && this.structs.get(want.name)?.generic?.name === enumName) return want.name;
    const names = tmpl.typeParams!;
    const bound = new Map<string, Ty>();
    const v = tmpl.variants!.find((x) => x.name === variant);
    if (v && v.fields.length === args.length) {
      v.fields.forEach((fname, k) => {
        const node = tmpl.fields.find((f) => f.name === `${variant}__${fname}`)!.tyNode;
        if (node.kind === 'name' && names.includes(node.name) && !bound.has(node.name)) bound.set(node.name, this.checkExpr(args[k].expr));
      });
    }
    const missing = names.filter((n) => !bound.has(n));
    if (missing.length > 0) {
      this.err('needs-type', `cannot tell what '${missing.join("', '")}' is for this '${enumName}.${variant}'`, e.pos,
        `use it where the type is known, e.g. \`const r: ${enumName}<...> = ${enumName}.${variant}${args.length > 0 ? '(...)' : ''};\` or as the \`return\` of a function declared with that type`);
    }
    const t = this.genericStruct({ kind: 'generic', name: enumName, args: names.map((n) => this.tyNode(bound.get(n)!, e.pos)), pos: e.pos });
    return (t as Ty & { k: 'struct' }).name;
  }

  /**
   * enum の値を作る式を、内部の struct のオブジェクトリテラルに書き換える。
   * 選んだ選択肢の値だけを入れ、他の選択肢の値は空の値（0 / false / 空の文字列や配列）にする。
   */
  enumValue(e: Expr, enumName: string, variant: string, args: Arg[], want?: Ty): Ty {
    this.visible(enumName, e.pos);
    const instName = this.enumInstance(e, enumName, variant, args, want);
    const info = this.structs.get(instName)!;
    const k = info.variants!.findIndex((v) => v.name === variant);
    if (k < 0) {
      this.err('unknown-variant', `enum '${enumName}' has no variant '${variant}'`, e.pos, `variants: ${info.variants!.map((v) => v.name).join(', ')}`);
    }
    const v = info.variants![k];
    if (args.length !== v.fields.length) {
      const shape = v.fields.length > 0 ? `${enumName}.${variant}(${v.fields.join(', ')})` : `${enumName}.${variant}`;
      this.err('arg-count', `variant '${variant}' takes ${v.fields.length} value(s) but ${args.length} were given`, e.pos, `write \`${shape}\``);
    }
    for (const a of args) if (a.mode !== 'none') this.err('arg-mode', 'enum values are passed by value', a.expr.pos);
    const pos = e.pos;
    const fields = info.fields.map((f) => {
      let value: Expr;
      const own = f.name.startsWith(`${variant}__`) ? v.fields.indexOf(f.name.slice(variant.length + 2)) : -1;
      if (f.name === '__tag') value = { kind: 'int', value: BigInt(k), pos };
      else if (own >= 0) value = args[own].expr;
      else value = this.zeroExpr(f.ty, pos);
      return { name: f.name, value, pos };
    });
    this.rewrite(e, { kind: 'object', fields, synth: true });
    return this.infer(e, { k: 'struct', name: instName });
  }

  /** 別名（match の枝の名前）が指す変数。番号 → 変数。 */
  pinned = new Map<number, Sym>();

  matchCount = 0;

  /**
   * match を、if の連なりに書き換えて検査する。
   *   { const __matchN = 対象; if (__matchN.__tag == 0) { const r = __matchN.Circle__r; 本体 } else if ... else { 最後の枝 } }
   * すべての選択肢を扱っているかを確かめる（`_ =>` があればそれが残りを受ける）。
   */
  checkMatch(s: Stmt & { kind: 'match' }): void {
    const pos = s.pos;
    const tmp = `__match${this.matchCount++}`;
    this.scopes.push(new Map());
    this.stmtDepth++;
    let froze = false;
    try {
      // 対象が変数やフィールドで、ヒープを持つ enum のときは、コピーも移動もせず、その場所を直接見る。
      // それ以外（コピーできる enum、関数の結果など）は、一時変数に入れてから見る。
      // 枝の名前（別名）を対象にしている場合は、それが指す式に展開しておく。
      const expand = (x: Expr): void => {
        if (x.kind === 'field' || x.kind === 'index') return expand(x.base);
        if (x.kind !== 'ident' || x.pin !== undefined) return;
        const sym = this.lookup(x.name);
        if (sym?.alias) {
          this.rewrite(x, structuredClone(sym.alias) as unknown as Record<string, unknown>);
          expand(x);
        }
      };
      expand(s.scrut);
      // 添字の式は、先に1回だけ評価して変数に入れる（対象を何度も読むので、副作用が繰り返されないように。
      // また、枝の名前が添字の中の名前を隠さないように）。
      const hoisted: Stmt[] = [];
      const hoist = (x: Expr): void => {
        if (x.kind === 'field') return hoist(x.base);
        if (x.kind !== 'index') return;
        hoist(x.base);
        if (x.index.kind === 'int') return;
        const name = `__matchidx${this.matchCount++}`;
        const letIdx: Stmt = { kind: 'let', name, isConst: true, init: x.index, pos };
        this.checkStmt(letIdx);
        hoisted.push(letIdx);
        x.index = { kind: 'ident', name, pos: x.pos };
      };
      hoist(s.scrut);
      const pristine = structuredClone(s.scrut);
      const st = this.checkExpr(s.scrut);
      const inPlace = isPlace(s.scrut) && !this.isCopy(st);
      const root = inPlace ? rootSym(s.scrut) : undefined;
      if (root) this.pinned.set(root.id, root);
      const pin = (x: Expr): Expr => {
        if (x.kind === 'ident') x.pin = root!.id;
        else if (x.kind === 'field' || x.kind === 'index') pin(x.base);
        return x;
      };
      // 見ている間、対象は書き換えも移動もできない（枝の名前が中身を指しているため）。
      if (inPlace) {
        this.frozen.push({ ...placePath(s.scrut)!, text: placeText(s.scrut), by: 'match' });
        froze = true;
      }
      const letStmt: Stmt | null = inPlace ? null : { kind: 'let', name: tmp, isConst: true, init: s.scrut, pos };
      if (letStmt) this.checkStmt(letStmt);
      const t = s.scrut.ty!;
      const info = t.k === 'struct' ? this.structs.get(t.name)! : undefined;
      if (!info?.variants) {
        this.err('type-mismatch', `\`match\` needs an enum value but found ${tyStr(t)}`, s.scrut.pos, 'use `if` / `else if` for other types');
      }
      const variants = info.variants;
      const seen = new Set<string>();
      let wildcard = false;
      for (const a of s.arms) {
        if (wildcard) this.err('unreachable', 'this arm comes after `_`, so it can never run', a.pos, 'move `_ =>` to the end');
        if (a.wildcard) {
          wildcard = true;
          continue;
        }
        if (a.enumName !== (info.generic?.name ?? info.name)) {
          this.err('type-mismatch', `this pattern is for '${a.enumName}' but the value is a '${info.name}'`, a.pos, `write \`${info.name}.${a.variant}\``);
        }
        const v = variants.find((x) => x.name === a.variant);
        if (!v) this.err('unknown-variant', `enum '${info.name}' has no variant '${a.variant}'`, a.pos, `variants: ${variants.map((x) => x.name).join(', ')}`);
        if (seen.has(a.variant)) this.err('duplicate', `variant '${a.variant}' is matched twice`, a.pos);
        seen.add(a.variant);
        if (a.binds.length !== v.fields.length) {
          const shape = v.fields.length > 0 ? `${info.name}.${a.variant}(${v.fields.join(', ')})` : `${info.name}.${a.variant}`;
          this.err('arg-count', `variant '${a.variant}' has ${v.fields.length} value(s) but the pattern names ${a.binds.length}`, a.pos,
            `write \`${shape} => ...\` (use \`_\` for a value you do not need)`);
        }
      }
      const missing = variants.filter((x) => !seen.has(x.name)).map((x) => x.name);
      if (!wildcard && missing.length > 0) {
        this.err('non-exhaustive', `\`match\` does not handle: ${missing.join(', ')}`, pos,
          `add ${missing.map((m) => `\`${info.name}.${m} => ...\``).join(', ')} or a final \`_ => ...\``);
      }

      const value = (): Expr => (inPlace ? pin(structuredClone(pristine)) : { kind: 'ident', name: tmp, pos });
      let chain: Stmt | undefined;
      for (let i = s.arms.length - 1; i >= 0; i--) {
        const a = s.arms[i];
        const body: Stmt[] = [];
        if (!a.wildcard) {
          const v = variants.find((x) => x.name === a.variant)!;
          a.binds.forEach((name, k) => {
            if (name === '_') return;
            const fname = `${a.variant}__${v.fields[k]}`;
            const init: Expr = { kind: 'field', base: value(), name: fname, pos: a.pos, synth: true };
            // コピーできる値は変数に取り出す。ヒープを持つ値は取り出せないので、enum の中を指す別名にする。
            const field = info.fields.find((f) => f.name === fname);
            if (!field) return; // フィールドの型が解決できなかった（別のエラーとして報告済み）
            if (this.isCopy(field.ty)) body.push({ kind: 'let', name, isConst: true, init, pos: a.pos });
            else body.push({ kind: 'let', name, isConst: true, alias: init, pos: a.pos });
          });
        }
        body.push(a.body);
        const block: Stmt = { kind: 'block', body, pos: a.pos };
        if (chain === undefined) {
          // 最後の枝は条件なしで受ける。すべての選択肢を扱っていることは確認済み。
          chain = block;
        } else {
          const tag: Expr = { kind: 'field', base: value(), name: '__tag', pos: a.pos, synth: true };
          const k = variants.findIndex((x) => x.name === a.variant);
          const cond: Expr = { kind: 'binary', op: '==', left: tag, right: { kind: 'int', value: BigInt(k), pos: a.pos }, pos: a.pos };
          chain = { kind: 'if', cond, then: block, else: chain, pos: a.pos };
        }
      }
      const body: Stmt[] = [...hoisted, ...(letStmt ? [letStmt] : []), ...(chain ? [chain] : [])];
      if (chain) this.checkStmt(chain);
      const node = s as unknown as Record<string, unknown>;
      delete node.scrut;
      delete node.arms;
      Object.assign(node, { kind: 'block', body });
    } finally {
      if (froze) this.frozen.pop();
      this.stmtDepth--;
      this.scopes.pop();
    }
  }

  checkStmt(s: Stmt): void {
    // スレッドに読み取り借用で貸した場所の凍結（`const h = thread.spawn(...)` の let が積む）は、
    // その宣言を含むブロックの終わりまで続く。let 以外の文を出るときに、入る前の状態に戻す。
    const frozenLen = this.frozen.length;
    try {
      // 絞り込みは、それが分かったブロックの中だけで有効。ブロックやループを出たら元に戻す。
      if (s.kind === 'if' || s.kind === 'let' || s.kind === 'assign' || s.kind === 'expr' || s.kind === 'return') return this.checkStmtInner(s);
      const known = new Set(this.narrowed);
      try {
        this.checkStmtInner(s);
      } finally {
        this.narrowed = known;
      }
    } finally {
      if (s.kind !== 'let') this.frozen.length = frozenLen;
    }
  }

  checkStmtInner(s: Stmt): void {
    switch (s.kind) {
      case 'match':
        this.checkMatch(s);
        return;

      case 'block':
        this.scopes.push(new Map());
        this.stmtDepth++;
        for (const b of s.body) this.checkStmt(b);
        this.stmtDepth--;
        this.scopes.pop();
        return;

      case 'let': {
        if (s.alias) {
          // match の枝の名前。変数は作らず、読む所を「enum の中の値」を指す式に置き換える。
          const target = structuredClone(s.alias);
          const at = this.checkExpr(s.alias);
          const sym = this.declare(s.name, at, false, s.pos, false);
          sym.alias = target;
          const node = s as unknown as Record<string, unknown>;
          for (const key of Object.keys(node)) if (key !== 'pos') delete node[key];
          Object.assign(node, { kind: 'block', body: [] });
          return;
        }
        if (s.init && this.isThreadCall(s.init, 'spawn', 'spawnEach')) {
          this.checkSpawn(s);
          return;
        }
        if (s.init && this.isGroupCall(s.init, 'shared')) {
          this.checkShared(s);
          return;
        }
        // simdgroup の集計（group.simdSum など）は、let / const の初期化式としてだけ書ける。
        if (s.init && this.isGroupCall(s.init) && GROUP_COLLECTIVE.has(s.init.name)) this.collectiveInit = s.init;
        let ty: Ty;
        if (!s.init) {
          // `let x: T;` はゼロで初期化する。コンストラクタを飛ばすことになる class と、ヒープを持つ型は除く。
          ty = this.resolve(s.tyAnn!);
          if (!this.isZeroable(ty)) {
            const hint = ty.k === 'heap'
              ? (ty.buffer ? `allocate it: \`let ${s.name} = new Buffer<${tyStr(ty.elem)}>(n);\`` : `start empty: \`let ${s.name}: ${tyStr(ty)} = [];\``)
              : ty.k === 'struct' && this.structs.get(ty.name)!.isClass ? `construct it: \`let ${s.name} = new ${ty.name}(...);\``
              : 'give it an initial value';
            this.err('uninitialized', `a variable of type ${tyStr(ty)} must be initialized`, s.pos, hint);
          }
          if (this.inKernel && !this.isKernelValue(ty)) {
            this.err('kernel-type', `type ${tyStr(ty)} is not available in a kernel function`, s.pos);
          }
          s.sym = this.declare(s.name, ty, true, s.pos);
          return;
        }
        if (s.tyAnn) {
          ty = this.resolve(s.tyAnn);
          if (ty.k === 'void') this.err('bad-type', 'a variable cannot be `void`', s.pos);
          this.expect(s.init, ty);
        } else {
          ty = this.checkExpr(s.init);
          if (ty.k === 'void') this.err('bad-type', `'${s.name}' would have type void`, s.pos);
        }
        this.consume(s.init);
        s.sym = this.declare(s.name, ty, !s.isConst, s.pos);
        return;
      }

      case 'assign': {
        if (s.target.kind === 'ident' && !this.lookup(s.target.name) && this.consts.has(s.target.name)) {
          this.err('immutable', `cannot assign to '${s.target.name}' because it is a top-level constant`, s.target.pos,
            'a constant never changes; keep the changing value in a `let` variable inside a function and pass it to other functions as a `mut` parameter');
        }
        this.assignTarget = s.target;
        const ty = this.checkExpr(s.target);
        this.assignTarget = null;
        if (!isPlace(s.target)) this.err('not-assignable', 'this expression cannot be assigned to', s.target.pos);
        const t = s.target;
        if (t.kind === 'field' && t.name === 'length' && (t.base.ty!.k === 'heap' || t.base.ty!.k === 'array')) {
          this.err('not-assignable', '`length` cannot be assigned', t.pos, 'use `.push()`, `.pop()` or `.clear()`');
        }
        if (t.kind === 'index' && isStr(t.base.ty!)) {
          this.err('not-assignable', 'strings are immutable, so a byte cannot be assigned', t.pos, 'build a new string with `+` or a template string');
        }
        // RacyBuffer の要素への書き込みは、kernel の外では読み取り借用からでも、凍結中でもよい
        // （長さも領域も変わらないので、所有の規則は崩れない。要素の読み書きはアトミック）。
        if (!(this.isRacyElem(s.target) && !this.inKernel)) this.requireMutable(s.target, 'assign to');
        const beforeValue = new Set(this.moved);
        this.expect(s.value, ty);
        this.consume(s.value);
        // 代入先がフィールドや要素なら、右辺がその持ち主を移動してはいけない。
        if (t.kind !== 'ident') this.notMovedSince(t, beforeValue);
        // 変数全体への代入は再初期化。移動済みだった変数がまた使えるようになる。
        if (t.kind === 'ident' && t.sym) this.moved.delete(t.sym);
        // コンストラクタ最上位の `this.x = ...` をフィールドの初期化として数える。
        if (this.ctorInit && t.kind === 'field' && t.base.kind === 'ident' && t.base.name === 'this' && !this.ctorInit.has(t.name)) {
          if (this.stmtDepth > 0) {
            this.err('uninitialized', `field '${t.name}' must first be initialized at the top level of the constructor`, t.pos,
              'assign it once before the `if` / loop, then update it inside');
          }
          this.ctorInit.add(t.name);
          s.initField = true;
        }
        return;
      }

      case 'if': {
        // `if (x != null)` / `if (x == null)`: 分岐の中では x を「値がある」ものとして扱える。
        const test = this.nullTest(s.cond);
        this.expect(s.cond, BOOL);
        // 分岐ごとに移動を追い、合流後は「どちらかで移動した変数」を移動済みとして扱う。
        const before = new Set(this.moved);
        const known = new Set(this.narrowed);
        this.stmtDepth++;
        if (test?.nonNull) this.narrowed.add(test.id);
        this.checkStmt(s.then);
        this.narrowed = new Set(known);
        const afterThen = this.moved;
        this.moved = new Set(before);
        if (test && !test.nonNull) this.narrowed.add(test.id);
        if (s.else) this.checkStmt(s.else);
        this.narrowed = new Set(known);
        const afterElse = this.moved;
        this.stmtDepth--;
        const thenOut = this.diverges(s.then);
        const elseOut = !!s.else && this.diverges(s.else);
        // null の側が必ず抜ける（return など）なら、この先では値がある。
        if (test && ((!test.nonNull && thenOut) || (test.nonNull && elseOut))) this.narrowed.add(test.id);
        if (thenOut && !elseOut) this.moved = afterElse;
        else if (elseOut && !thenOut) this.moved = afterThen;
        else this.moved = new Set([...afterThen, ...afterElse]);
        return;
      }

      case 'while':
        // 条件は周回ごとに評価されるので、ループの中として検査する。
        this.enterLoop();
        this.expect(s.cond, BOOL);
        this.stmtDepth++;
        this.checkStmt(s.body);
        this.stmtDepth--;
        this.leaveLoop();
        return;

      case 'for':
        this.scopes.push(new Map());
        this.stmtDepth++;
        this.inferLoopVar(s);
        if (s.init) this.checkStmt(s.init);
        this.enterLoop();
        if (s.cond) this.expect(s.cond, BOOL);
        this.checkStmt(s.body);
        if (s.step) this.checkStmt(s.step);
        this.leaveLoop();
        this.stmtDepth--;
        this.scopes.pop();
        return;

      case 'forOf': {
        const it = this.checkExpr(s.iter);
        if (it.k !== 'array' && it.k !== 'heap') {
          this.err('type-mismatch', `\`for...of\` needs an array but found ${tyStr(it)}`, s.iter.pos);
        }
        if (this.inKernel && it.k === 'heap' && !it.racy && rootSym(s.iter)?.mutable) {
          this.err('kernel-restriction', 'a kernel cannot iterate over a Buffer it writes to', s.iter.pos,
            'a kernel may only touch element `[i]` of a `mut` Buffer; other invocations write the other elements at the same time');
        }
        this.scopes.push(new Map());
        // 反復中の配列は書き換えも移動もできない（要素の置き場所が動かないことを保証するため）。
        const place = isPlace(s.iter) ? placePath(s.iter) : null;
        if (place) this.frozen.push({ ...place, text: placeText(s.iter) });
        this.enterLoop();
        // 要素はコピーして渡す。ループ変数を書き換えても配列は変わらない。
        // ヒープを持つ要素はコピーできないので、借りるだけにする（移動も書き換えもできない）。
        const borrowed = !this.isCopy(it.elem);
        if (borrowed && !s.isConst) {
          this.err('not-assignable', `elements of type ${tyStr(it.elem)} are borrowed, not copied, so the loop variable must be \`const\``, s.pos,
            `write \`for (const ${s.name} of ...)\``);
        }
        s.sym = this.declare(s.name, it.elem, !s.isConst, s.pos, !borrowed);
        this.stmtDepth++;
        this.checkStmt(s.body);
        this.stmtDepth--;
        this.leaveLoop();
        if (place) this.frozen.pop();
        this.scopes.pop();
        return;
      }

      case 'return':
        if (this.curFn.isCtor) {
          this.err('syntax', '`return` is not allowed in a constructor', s.pos, 'let the constructor run to its end');
        }
        if (this.curRet.k === 'void') {
          if (s.value) {
            const vt = this.curFn.retNode ? undefined : (() => {
              try {
                return this.checkExpr(s.value!);
              } catch {
                return undefined;
              }
            })();
            const hint = this.curFn.retNode ? undefined
              : `return types are not inferred; declare it: \`function ${this.curFn.name}(...): ${vt && vt.k !== 'void' ? tyStr(vt) : 'T'}\``;
            this.err('type-mismatch', 'this function returns void but a value is returned', s.pos, hint);
          }
        } else {
          if (!s.value) this.err('type-mismatch', `this function must return ${tyStr(this.curRet)}`, s.pos);
          this.expect(s.value, this.curRet);
          this.consume(s.value);
        }
        return;

      case 'break':
      case 'continue': {
        if (this.loops.length === 0) this.err('syntax', `'${s.kind}' outside of a loop`, s.pos);
        const ctx = this.loops[this.loops.length - 1];
        if (s.kind === 'break') for (const sym of this.moved) ctx.breakMoved.add(sym);
        else this.checkLoopMoves();
        return;
      }

      case 'expr':
        this.checkExpr(s.expr);
        return;
    }
  }

  /**
   * `for (let i = 0; i < n; ...)` の i を n の型に合わせる。
   * 注釈なしの整数リテラルは i32 になるので、そのままでは usize の長さと比較できない。
   */
  inferLoopVar(s: Stmt & { kind: 'for' }): void {
    const init = s.init;
    const cond = s.cond;
    if (!init || init.kind !== 'let' || init.tyAnn || !init.init || init.init.kind !== 'int') return;
    if (!cond || cond.kind !== 'binary' || !['<', '<=', '>', '>=', '!='].includes(cond.op)) return;
    // 左辺がループ変数を含む式（`i` や `i + 2`）で、右辺が変数・フィールド・length のとき。
    const mentions = (e: Expr): boolean =>
      e.kind === 'ident' ? e.name === init.name : e.kind === 'binary' ? mentions(e.left) || mentions(e.right) : false;
    if (!mentions(cond.left)) return;
    // 右辺を先に検査するので、副作用のない形（変数、フィールド、length）に限る。
    if (!isPlace(cond.right)) return;
    // 右辺がループ変数を使うなら、ここでは検査しない。ループ変数はまだ見えないので、
    // 同じ名前の最上位の定数（や外側の変数）を指すものとして検査されてしまう。
    const uses = (x: unknown): boolean => {
      if (Array.isArray(x)) return x.some(uses);
      if (x === null || typeof x !== 'object') return false;
      const o = x as Record<string, unknown>;
      if (o.kind === 'ident' && o.name === init.name) return true;
      return Object.keys(o).some((k) => k !== 'ty' && k !== 'pos' && uses(o[k]));
    };
    if (uses(cond.right)) return;
    let bound: Ty;
    try {
      bound = this.checkExpr(cond.right);
    } catch {
      return; // 右辺がループ変数を使うなど。通常の検査に任せる。
    }
    // 右辺が最上位の定数なら、検査でリテラルに置き換わっている（fixed）。その型に合わせる。
    if (bound.k !== 'int' || (cond.right.kind === 'int' && !cond.right.fixed)) return;
    init.tyAnn = { kind: 'name', name: tyStr(bound), pos: init.pos };
  }

  requireMutable(place: Expr, verb: string): void {
    const sym = rootSym(place);
    if (sym && sym === this.curFn.thisSym) this.directMut.add(this.curFn);
    const at = placePath(place);
    const busy = at && this.frozen.find((f) => f.sym === at.sym && overlaps(f.path, at.path));
    if (busy) {
      this.err('exclusive-borrow', `cannot ${verb} '${placeText(place)}' while ${frozenBy(busy)}`, place.pos,
        busy.by === 'thread'
          ? 'threads read it until they are joined; start the threads in an inner block `{ ... }` and change it after that block (a `RacyBuffer` element can be written at any time)'
          : 'collect the changes in another variable and apply them after the loop, or use an index loop');
    }
    if (sym?.global) {
      this.err('immutable', `cannot ${verb} '${placeText(place)}' because '${sym.name}' is a top-level constant`, place.pos,
        `a constant array is read-only data; copy it into a variable first (\`let t: ${tyStr(sym.ty)} = ${sym.name};\`) or keep the table in a \`let\` / \`Buffer\``);
    }
    if (sym && !sym.mutable) {
      this.err('immutable', `cannot ${verb} '${sym.name}' because it is not mutable`, place.pos,
        sym.param ? `declare the parameter as \`${sym.name}: mut ${tyStr(sym.ty)}\` and pass \`mut ...\` at the call site`
          : `declare it with \`let\` instead of \`const\` (a \`for (const x of xs)\` variable is a copy; use \`for (let x of xs)\` to change the copy)`);
    }
  }

  // ---- 式 ----

  /** 式を検査し、期待する型と一致することを確かめる。 */
  expect(e: Expr, want: Ty): void {
    const inner = this.optInner(want);
    if (inner && !(e.kind === 'object' && e.synth)) {
      // `T | null` が求められている所: null はそのまま、T の値は自動で包む（唯一の暗黙の変換）。
      if (e.kind === 'null') {
        this.rewrite(e, { kind: 'object', synth: true, fields: [
          { name: '__some', value: { kind: 'bool', value: false, pos: e.pos }, pos: e.pos },
          { name: '__value', value: this.zeroExpr(inner, e.pos), pos: e.pos },
        ] });
      } else if (tyEq(this.checkExpr(e, inner), inner)) {
        const value = { ...e } as Expr;
        this.rewrite(e, { kind: 'object', synth: true, fields: [
          { name: '__some', value: { kind: 'bool', value: true, pos: e.pos }, pos: e.pos },
          { name: '__value', value, pos: e.pos },
        ] });
      }
    }
    const got = this.checkExpr(e, want);
    if (!tyEq(got, want)) {
      let hint: string | undefined;
      if (isNumeric(got) && isNumeric(want)) {
        hint = `there are no implicit conversions; write \`as ${tyStr(want)}\``;
      } else if (want.k === 'bool' && isNumeric(got)) {
        hint = 'numbers are not conditions; compare explicitly, e.g. `x != 0`';
      } else if (want.k === 'heap' && !want.buffer && got.k === 'array' && tyEq(got.elem, want.elem)) {
        hint = `an array literal without a type is fixed-length; declare the variable as \`${tyStr(want)}\` to get a growable array, e.g. \`const xs: ${tyStr(want)} = [...]\``;
      } else if (want.k === 'void' || got.k === 'void') {
        hint = got.k === 'void' ? 'the function called here returns nothing; give it a return type and a `return`' : undefined;
      }
      this.err('type-mismatch', `expected ${tyStr(want)} but found ${tyStr(got)}`, e.pos, hint);
    }
  }

  checkExpr(e: Expr, want?: Ty): Ty {
    const t = this.infer(e, want);
    e.ty = t;
    // 型注釈を通らずに推論された配列の型（配列リテラルなど）にも、要素と大きさの制限を適用する。
    // 最上位の定数の配列（とその行）は、添字を付けて読む所と定数の初期化の中では大きさを問わない
    // （グローバルな領域を直接読み、コピーしない）。値として使う所（変数に入れる、引数に渡す）だけ普通の上限を当てる。
    if (t.k === 'array' && (this.constInit || (this.constIndexBases.has(e) && isConstPlace(e)))) {
      // 要素の型は宣言のときに確かめてある。
    } else if (t.k === 'array' && isConstPlace(e) && this.scalarCount(t) > MAX_VALUE_SCALARS) {
      this.err('too-large', `constant array ${placeText(e)} holds ${this.scalarCount(t)} values, too many to copy as a value (the limit is ${MAX_VALUE_SCALARS})`, e.pos,
        `read its elements with an index instead, e.g. \`${placeText(e)}[i]\``);
    } else if (t.k === 'array' || t.k === 'heap') this.checkElem(t, e.pos);
    if (this.inKernel) this.checkSmallInKernel(e, t);
    if (this.inKernel && t.k !== 'void' && !this.isKernelValue(t)) {
      // Buffer そのものは引数の名前としてだけ現れてよい（添字や length の対象）。
      const bufferParam = t.k === 'heap' && !!t.buffer && e.kind === 'ident';
      if (!bufferParam) {
        this.err('kernel-type', `type ${tyStr(t)} is not available in a kernel function`, e.pos,
          'kernel functions run on the GPU too: use bool, i32, u32, i64, u64, f32 and structs / fixed-length arrays of those (convert with `as`)');
      }
    }
    return t;
  }

  /**
   * kernel の中の 8 / 16 ビットの整数は、読み書きと比較と `as` での変換だけにする。
   * GPU では 32 ビットのレジスタに入れて持つので、演算のたびに切り詰め直すことになる。それを避け、
   * 計算は `as i32` / `as u32` で広げてから行わせる（結果は CPU と同じ）。浮動小数点との直接の変換も、
   * 範囲外の飽和の仕方を GPU で揃えにくいので、32 ビットの整数を経由させる。
   */
  checkSmallInKernel(e: Expr, t: Ty): void {
    const wide = (x: Ty) => (x as Ty & { k: 'int' }).signed ? 'i32' : 'u32';
    if (e.kind === 'binary' && isSmallInt(e.left.ty) && !['==', '!=', '<', '<=', '>', '>='].includes(e.op)) {
      this.err('kernel-type', `'${e.op}' on ${tyStr(e.left.ty!)} is not available in a kernel function`, e.pos,
        `widen first, e.g. \`(a as ${wide(e.left.ty!)}) ${e.op} (b as ${wide(e.left.ty!)})\`, and convert back with \`as ${tyStr(e.left.ty!)}\``);
    } else if ((e.kind === 'unary' || e.kind === 'math') && isSmallInt(t)) {
      this.err('kernel-type', `arithmetic on ${tyStr(t)} is not available in a kernel function`, e.pos,
        `widen first with \`as ${wide(t)}\`, and convert back with \`as ${tyStr(t)}\``);
    } else if (e.kind === 'cast') {
      const from = e.expr.ty!;
      if ((isSmallInt(from) && t.k === 'float') || (from.k === 'float' && isSmallInt(t))) {
        this.err('kernel-type', `a kernel cannot convert ${tyStr(from)} to ${tyStr(t)} directly`, e.pos,
          `go through a 32-bit integer, e.g. \`(x as ${isSmallInt(t) ? wide(t) : wide(from)}) as ${tyStr(t)}\``);
      }
    }
  }

  /** kernel の中では使えない構文を弾く。 */
  noKernel(what: string, pos: Pos, hint?: string): void {
    if (this.inKernel) {
      this.err('kernel-restriction', `${what} is not available in a kernel function`, pos,
        hint ?? 'kernel functions run on the GPU too, so they cannot allocate, print, or call non-kernel functions');
    }
  }

  infer(e: Expr, want?: Ty): Ty {
    switch (e.kind) {
      case 'int': {
        if (e.fixed) return e.fixed;
        if (want?.k === 'float') return want;
        let t: Ty & { k: 'int' };
        if (want?.k === 'int') {
          t = want;
        } else {
          const [lo, hi] = intRange(I32 as Ty & { k: 'int' });
          t = (e.value >= lo && e.value <= hi ? I32 : I64) as Ty & { k: 'int' };
        }
        const [lo, hi] = intRange(t);
        if (e.value < lo || e.value > hi) {
          this.err('literal-range', `${e.value} does not fit in ${tyStr(t)}`, e.pos);
        }
        return t;
      }

      case 'float':
        // kernel には f64 がないので、型の決まらない小数リテラルは f32 にする。
        if (e.fixed) return e.fixed;
        return want?.k === 'float' ? want : this.inKernel ? F32 : F64;

      case 'bool':
        return BOOL;

      case 'null':
        this.err('needs-type', 'the type of this `null` is not known here', e.pos,
          '`null` can be used where a `T | null` is expected: a typed variable, a parameter, or a `return` of a function declared `: T | null`');

      case 'ident': {
        const sym = e.pin !== undefined ? this.pinned.get(e.pin) : this.lookup(e.name);
        if (sym?.alias) {
          this.rewrite(e, structuredClone(sym.alias) as unknown as Record<string, unknown>);
          return this.infer(e, want);
        }
        if (sym && this.narrowed.has(sym.id) && !e.raw) {
          // null でないと分かっている変数は、中の値として読む。
          this.rewrite(e, { kind: 'field', base: { kind: 'ident', name: sym.name, pos: e.pos, raw: true }, name: '__value', synth: true });
          return this.infer(e, want);
        }
        if (!sym && e.name === 'this') {
          this.err('unknown-name', '`this` is only available inside a class', e.pos);
        }
        if (!sym && this.consts.has(e.name)) {
          this.visible(e.name, e.pos);
          const c = this.constInfo(e.name);
          if (c.ty!.k === 'array') {
            // 定数の配列: 読み取り専用のグローバルな領域を指す名前として残す（使う所ごとに値を複製しない）。
            e.sym = this.constSym(c);
            return c.ty!;
          }
          // 最上位の定数: コンパイル時に求めた値（型の決まったリテラル）に置き換える。
          this.rewrite(e, this.constLiteral(c, e.pos));
          return this.infer(e, want);
        }
        if (sym && e.name === 'this' && this.ctorInit) {
          // フィールドがすべて初期化されるまで、this 全体は使えない（未初期化の値を読ませないため）。
          const missing = this.structs.get(this.curFn.cls!)!.fields.filter((f) => !this.ctorInit!.has(f.name));
          if (missing.length > 0) {
            this.err('uninitialized', `\`this\` is used before all fields are initialized (missing: ${missing.map((f) => f.name).join(', ')})`,
              e.pos, 'initialize every field first, at the top level of the constructor');
          }
        }
        if (!sym) {
          const cls = this.curFn?.cls ? this.structs.get(this.curFn.cls) : undefined;
          const member = cls && (cls.fields.some((f) => f.name === e.name) || cls.methods.has(e.name));
          const hint = member ? `members are accessed through \`this\`: write \`this.${e.name}\``
            : this.fns.has(e.name) ? `'${e.name}' is a function; call it with \`${e.name}(...)\`` : undefined;
          this.err('unknown-name', `unknown name '${e.name}'`, e.pos, hint);
        }
        if (sym?.shared && !this.sharedBase) {
          this.err('kernel-restriction', `'${e.name}' is a group shared array, so it can only be indexed (\`${e.name}[j]\`)`, e.pos,
            'read and write single elements; copy the elements you need into a local variable');
        }
        this.sharedBase = false;
        if (sym.thread && this.joinArg !== e) {
          this.err('thread', `'${e.name}' is a thread handle; it can only be passed to \`thread.join(${e.name})\``, e.pos,
            'a handle stays in the `const` that started the thread and is joined at the end of its block; to start many threads, use `thread.spawnEach(f, own items, ...)`');
        }
        if (this.moved.has(sym) && this.assignTarget !== e) {
          if (sym.thread) {
            this.err('use-after-move', `thread '${e.name}' was already joined`, e.pos, 'a handle can be joined only once');
          }
          this.err('use-after-move', `'${e.name}' was moved and can no longer be used`, e.pos,
            'a value that owns heap memory has one owner; pass `.clone()` where it was moved, or lend it through a function parameter instead of assigning it');
        }
        e.sym = sym;
        return sym.ty;
      }

      case 'unary': {
        const t = this.checkExpr(e.expr, want);
        const vec = e.op === '-' ? this.vecInfo(t) : undefined;
        if (vec) {
          this.rewrite(e, { kind: 'call', callee: `vec${vec.n}Neg`, args: [{ mode: 'none', expr: e.expr }] });
          return this.infer(e, want);
        }
        if (e.op === '!') {
          if (t.k !== 'bool') this.err('type-mismatch', `'!' needs bool but found ${tyStr(t)}`, e.pos, 'compare explicitly, e.g. `x == 0`');
        } else if (e.op === '~') {
          if (t.k !== 'int') this.err('type-mismatch', `'~' needs an integer but found ${tyStr(t)}`, e.pos);
        } else if (!isNumeric(t)) {
          this.err('type-mismatch', `'-' needs a number but found ${tyStr(t)}`, e.pos);
        }
        return t;
      }

      case 'str':
        this.noKernel('a string literal', e.pos);
        return STR;

      case 'binary':
        return this.inferBinary(e, want);

      case 'call':
        return this.inferCall(e, want);

      case 'mcall':
        return this.inferMethodCall(e, want);

      case 'cond': {
        this.expect(e.cond, BOOL);
        const t = this.checkPair(e.then, e.else, want);
        if (t.k === 'void') this.err('type-mismatch', 'both sides of `? :` must produce a value', e.pos);
        if (!this.isCopy(t)) {
          this.err('not-implemented', `\`? :\` cannot choose between values of type ${tyStr(t)} yet`, e.pos,
            'values that own heap memory are moved, not copied; use `if` / `else` and assign in each branch');
        }
        if (this.inKernel && t.k !== 'int' && t.k !== 'float' && t.k !== 'bool') {
          this.err('kernel-restriction', '`? :` in a kernel must choose between numbers or bools', e.pos, 'use `if` / `else` for structs and arrays');
        }
        return t;
      }

      case 'dispatch':
      case 'math':
      case 'thread':
        // これらは call / mcall を検査した時点で作られる。型は記録済み。
        return e.ty!;

      case 'new': {
        this.noKernel('`new`', e.pos);
        if (this.genericTypes.has(e.cls)) {
          // ジェネリックな class: `new Stack<i32>()`、または求められている型から決める。
          if (e.typeArgs) {
            e.cls = (this.genericStruct({ kind: 'generic', name: e.cls, args: e.typeArgs, pos: e.pos }) as Ty & { k: 'struct' }).name;
          } else if (want?.k === 'struct' && this.structs.get(want.name)?.generic?.name === e.cls) {
            this.visible(e.cls, e.pos);
            e.cls = want.name;
          } else {
            this.err('needs-type', `'${e.cls}' needs its type arguments here`, e.pos, `write \`new ${e.cls}<...>(...)\``);
          }
        } else if (e.typeArgs) {
          this.err('unknown-type', `'${e.cls}' takes no type arguments`, e.pos);
        }
        if (this.structs.has(e.cls)) this.visible(e.cls, e.pos);
        const info = this.structs.get(e.cls);
        if (!info) this.err('unknown-type', `unknown class '${e.cls}'`, e.pos);
        if (!info.isClass || !info.ctor) {
          this.err('ts-feature', `'${e.cls}' is a struct, not a class`, e.pos,
            `build it with an object literal: \`const v: ${e.cls} = { ... }\``);
        }
        this.checkArgs(info.ctor, e.args, e.pos);
        return { k: 'struct', name: e.cls };
      }

      case 'newArray': {
        this.noKernel('allocating an array', e.pos);
        const elem = this.resolve(e.elemNode);
        const t: Ty = e.racy ? { k: 'heap', elem, buffer: true, racy: true } : e.buffer ? { k: 'heap', elem, buffer: true } : { k: 'heap', elem };
        this.checkElem(t, e.pos);
        if (e.buffer && e.args.length !== 1) {
          this.err('arg-count', '`new Buffer<T>(n)` takes the number of elements', e.pos,
            'a Buffer has a fixed length and starts zero-filled, e.g. `new Buffer<f32>(1024)`');
        }
        if (e.args.length > 1) {
          this.err('arg-count', '`new Array<T>(...)` takes no argument (empty) or a length (zero-filled)', e.pos,
            'to list the elements, use an array literal: `const xs: i32[] = [1, 2, 3]`');
        }
        if (e.args.length === 1) {
          const a = e.args[0];
          if (a.mode !== 'none') this.err('arg-mode', 'the length is passed by value', a.expr.pos);
          const lt = this.checkExpr(a.expr, this.usize);
          if (lt.k !== 'int') this.err('type-mismatch', `the length must be an integer but found ${tyStr(lt)}`, a.expr.pos);
        }
        return t;
      }

      case 'field': {
        // `Shape.Empty`: 値を持たない選択肢。
        if (e.base.kind === 'ident' && !this.lookup(e.base.name) && this.isEnumName(e.base.name)) {
          return this.enumValue(e, e.base.name, e.name, [], want);
        }
        const onThis = e.base.kind === 'ident' && e.base.name === 'this';
        let bt: Ty;
        if (onThis && this.ctorInit) {
          // コンストラクタ内の this.x は、this 全体の初期化を待たずにフィールド単位で扱う。
          const sym = this.lookup('this')!;
          (e.base as Expr & { kind: 'ident' }).sym = sym;
          e.base.ty = sym.ty;
          bt = sym.ty;
        } else {
          bt = this.checkExpr(e.base);
        }
        if (bt.k === 'array' && e.name === 'length') {
          // 固定長配列の長さはコンパイル時定数。整数リテラルに置き換えて、文脈の型に合わせる。
          const lit = e as unknown as { kind: string; value: bigint };
          lit.kind = 'int';
          lit.value = BigInt(bt.len);
          return this.infer(e, want);
        }
        if (bt.k === 'heap' && e.name === 'length') return bt.buffer ? U32 : this.usize;
        if (this.optInner(bt) && !e.synth) {
          this.err('type-mismatch', `this value has type ${tyStr(bt)} and may be null, so '.${e.name}' cannot be read yet`, e.pos,
            'bind it to a `const`, then read the field inside `if (x != null) { ... }`');
        }
        if (bt.k !== 'struct') this.err('type-mismatch', `${tyStr(bt)} has no field '${e.name}'`, e.pos);
        const info = this.structs.get(bt.name)!;
        const idx = info.fields.findIndex((f) => f.name === e.name);
        if (idx < 0) {
          const hint = info.methods.has(e.name)
            ? `'${e.name}' is a method; call it with \`.${e.name}(...)\``
            : `fields: ${info.fields.map((f) => f.name).join(', ')}`;
          this.err('unknown-field', `'${bt.name}' has no field '${e.name}'`, e.pos, hint);
        }
        if (info.variants && !e.synth) {
          this.err('enum-access', `the values inside enum '${bt.name}' can only be read with \`match\``, e.pos,
            `write \`match (x) { ${bt.name}.${info.variants[0].name}${info.variants[0].fields.length > 0 ? `(${info.variants[0].fields.join(', ')})` : ''} => { ... } ... }\``);
        }
        if (info.fields[idx].isPrivate && this.curFn.cls !== bt.name) {
          this.err('private', `field '${e.name}' of '${bt.name}' is private`, e.pos, `add a method to '${bt.name}' that exposes what you need`);
        }
        if (onThis && this.ctorInit && !this.ctorInit.has(e.name) && this.assignTarget !== e) {
          this.err('uninitialized', `field '${e.name}' is read before it is initialized`, e.pos);
        }
        e.index = idx;
        return info.fields[idx].ty;
      }

      case 'index': {
        this.constIndexBases.add(e.base);
        this.sharedBase = e.base.kind === 'ident';
        const bt = this.checkExpr(e.base);
        this.sharedBase = false;
        if (bt.k !== 'array' && bt.k !== 'heap') this.err('type-mismatch', `${tyStr(bt)} cannot be indexed`, e.pos);
        const beforeIndex = new Set(this.moved);
        const it = this.checkExpr(e.index);
        this.notMovedSince(e.base, beforeIndex);
        if (this.inKernel && bt.k === 'heap' && !bt.racy && e.base.kind === 'ident' && e.base.sym!.mutable) {
          // 書き込み先の Buffer は、自分の担当する要素 [i] だけを読み書きできる。
          const own = e.index.kind === 'ident' && e.index.sym === this.idxParam;
          if (!own) {
            this.err('kernel-restriction', `a kernel may only access element \`[${this.idxParam!.name}]\` of the \`mut\` Buffer '${e.base.name}'`, e.index.pos,
              'other invocations write the other elements at the same time; read neighbours from a separate, non-`mut` Buffer');
          }
        }
        if (it.k !== 'int') this.err('type-mismatch', `array index must be an integer but found ${tyStr(it)}`, e.index.pos);
        if (e.index.kind === 'int' && e.index.value < 0n) {
          this.err('index-range', `index ${e.index.value} is negative`, e.index.pos);
        }
        if (bt.k === 'array' && e.index.kind === 'int' && e.index.value >= BigInt(bt.len)) {
          this.err('index-range', `index ${e.index.value} is out of bounds for ${tyStr(bt)}`, e.index.pos);
        }
        return bt.elem;
      }

      case 'object': {
        if (want?.k !== 'struct') {
          this.err('needs-type', 'the struct type of this object literal is not known here', e.pos,
            'add a type annotation, e.g. `const p: Point = { x: 1, y: 2 }`');
        }
        const info = this.structs.get(want.name)!;
        if (info.variants && !e.synth) {
          this.err('type-mismatch', `'${want.name}' is an enum and cannot be built from an object literal`, e.pos,
            `write a variant: \`${want.name}.${info.variants[0].name}${info.variants[0].fields.length > 0 ? '(...)' : ''}\``);
        }
        if (info.isClass) {
          this.err('type-mismatch', `'${want.name}' is a class and cannot be built from an object literal`, e.pos,
            `call the constructor: \`new ${want.name}(...)\``);
        }
        const seen = new Set<string>();
        for (const f of e.fields) {
          const def = info.fields.find((d) => d.name === f.name);
          if (!def) {
            this.err('unknown-field', `struct '${want.name}' has no field '${f.name}'`, f.pos,
              `fields: ${info.fields.map((d) => d.name).join(', ')}`);
          }
          if (seen.has(f.name)) this.err('duplicate', `field '${f.name}' is given twice`, f.pos);
          seen.add(f.name);
          this.expect(f.value, def.ty);
          this.consume(f.value);
        }
        const missing = info.fields.filter((d) => !seen.has(d.name)).map((d) => d.name);
        if (missing.length > 0) {
          this.err('missing-field', `missing field(s) for '${want.name}': ${missing.join(', ')}`, e.pos,
            'every field must be initialized');
        }
        return want;
      }

      case 'arrayLit': {
        if (want?.k === 'heap' && want.str) {
          this.err('type-mismatch', 'expected a string but found an array literal', e.pos, 'write a string literal such as "abc"');
        }
        if (want?.k === 'heap') {
          this.noKernel('allocating an array', e.pos);
          for (const el of e.elems) {
            this.expect(el, want.elem);
            this.consume(el); // 要素は配列に移動する
          }
          return want;
        }
        if (e.elems.length === 0) {
          this.err('needs-type', 'the element type of this empty array is not known here', e.pos,
            'add a type annotation, e.g. `let xs: i32[] = []`');
        }
        if (want?.k === 'array') {
          if (e.elems.length !== want.len) {
            this.err('type-mismatch', `expected ${want.len} element(s) for ${tyStr(want)} but found ${e.elems.length}`, e.pos);
          }
          for (const el of e.elems) this.expect(el, want.elem);
          return want;
        }
        const elem = this.checkExpr(e.elems[0]);
        if (elem.k === 'void') this.err('bad-type', 'array of `void` is not allowed', e.pos);
        for (const el of e.elems.slice(1)) this.expect(el, elem);
        return { k: 'array', elem, len: e.elems.length };
      }

      case 'cast': {
        const to = this.resolve(e.to);
        const from = this.checkExpr(e.expr);
        if (!this.isCopy(from)) this.err('bad-cast', `cannot convert ${tyStr(from)} to ${tyStr(to)}`, e.pos);
        if (tyEq(from, to)) return to;
        if (isNumeric(from) && isNumeric(to)) return to;
        if (from.k === 'bool' && to.k === 'int') return to;
        const hint = to.k === 'bool' && isNumeric(from) ? 'compare explicitly, e.g. `x != 0`' : undefined;
        this.err('bad-cast', `cannot convert ${tyStr(from)} to ${tyStr(to)}`, e.pos, hint);
      }
    }
  }

  /** 左右の型を揃えて検査する。片方がリテラルなら、もう片方の型に合わせる。 */
  /** checkPair が「ベクトルと数」の組を受け付けたとき、どちらが数だったか。 */
  pairScalar: 'left' | 'right' | null = null;

  /** t が標準の vec2 / vec3 / vec4 なら、成分の数と型を返す。 */
  vecInfo(t: Ty | undefined): { n: number; elem: Ty } | undefined {
    if (t?.k !== 'struct') return undefined;
    const g = this.structs.get(t.name)?.generic;
    if (!g || !/^vec[234]$/.test(g.name) || this.fileOf.get(g.name) !== -1) return undefined;
    return { n: Number(g.name[3]), elem: g.args[0] };
  }

  checkPair(left: Expr, right: Expr, want?: Ty): Ty {
    // 両辺を検査している間に、内側の式が pairScalar を書き換える。結果は最後にまとめて入れる。
    const done = (t: Ty, scalar: 'left' | 'right' | null): Ty => {
      this.pairScalar = scalar;
      return t;
    };
    let lt = this.checkExpr(left, want);
    if (isLiteral(left) && !isLiteral(right)) {
      const rt = this.checkExpr(right, want);
      const rv = this.vecInfo(rt);
      lt = this.checkExpr(left, rv ? rv.elem : rt);
      if (rv && tyEq(lt, rv.elem)) return done(rt, 'left');
      if (!tyEq(lt, rt)) this.mismatch(left, right, lt, rt);
      return done(rt, null);
    }
    const lv = this.vecInfo(lt);
    const rt = this.checkExpr(right, lv && isLiteral(right) ? lv.elem : lt);
    if (!tyEq(lt, rt)) {
      // ベクトルと数の組（`v * 2`、`2 * v`）。
      if (lv && tyEq(rt, lv.elem)) return done(lt, 'right');
      const rv = this.vecInfo(rt);
      if (rv && tyEq(lt, rv.elem)) return done(rt, 'left');
      this.mismatch(left, right, lt, rt);
    }
    return done(lt, null);
  }

  mismatch(left: Expr, right: Expr, lt: Ty, rt: Ty): never {
    const hint = isNumeric(lt) && isNumeric(rt)
      ? `there are no implicit conversions; convert one side, e.g. \`as ${tyStr(lt)}\``
      : undefined;
    const nullable = this.optInner(lt) || this.optInner(rt);
    this.err('type-mismatch', `operands have different types: ${tyStr(lt)} and ${tyStr(rt)}`, right.pos,
      nullable ? 'a value that may be null must be checked first: bind it to a `const`, then use it inside `if (x != null) { ... }`, or give a fallback with `x ?? 0`' : hint);
  }

  inferBinary(e: Expr & { kind: 'binary' }, want?: Ty): Ty {
    const beforeOperands = new Set(this.moved);
    const t = this.inferBinaryInner(e, want);
    // 左辺が文字列などヒープを持つ値のとき、右辺がそれを移動してはいけない（左辺は右辺のあとで読まれる）。
    if (e.kind === 'binary' && e.left.ty && !this.isCopy(e.left.ty)) this.notMovedSince(e.left, beforeOperands);
    return t;
  }

  inferBinaryInner(e: Expr & { kind: 'binary' }, want?: Ty): Ty {
    const op = e.op;
    if ((op === '==' || op === '!=') && (e.left.kind === 'null' || e.right.kind === 'null')) {
      // `x != null` は「値がある」、`x == null` は「値がない」。
      const v = e.left.kind === 'null' ? e.right : e.left;
      if (v.kind === 'null') this.err('type-mismatch', 'comparing `null` with `null`', e.pos);
      const t = this.checkExpr(v);
      if (!this.optInner(t)) {
        this.err('type-mismatch', `this value has type ${tyStr(t)}, which is never null`, e.pos,
          'only a `T | null` can be compared with `null`; inside `if (x != null) { ... }` the variable already counts as present');
      }
      const some: Expr = { kind: 'field', base: v, name: '__some', pos: e.pos, synth: true };
      if (op === '!=') this.rewrite(e, some as unknown as Record<string, unknown>);
      else this.rewrite(e, { kind: 'unary', op: '!', expr: some });
      return this.infer(e, BOOL);
    }
    if (op === '??') {
      // `a ?? b`: a に値があればそれ、なければ b。
      const t = this.checkExpr(e.left);
      const inner = this.optInner(t);
      if (!inner) this.err('type-mismatch', `the left side of \`??\` has type ${tyStr(t)}, which is never null`, e.pos);
      if (e.left.kind !== 'ident') {
        this.err('not-implemented', 'the left side of `??` must be a variable for now', e.left.pos, 'store the value in a `const` first');
      }
      const name = e.left.name;
      const use = (field: string): Expr => ({ kind: 'field', base: { kind: 'ident', name, pos: e.pos, raw: true }, name: field, pos: e.pos, synth: true });
      this.rewrite(e, { kind: 'cond', cond: use('__some'), then: use('__value'), else: e.right });
      return this.infer(e, inner);
    }
    if (op === '&&' || op === '||') {
      this.expect(e.left, BOOL);
      this.expect(e.right, BOOL);
      return BOOL;
    }
    if (op === '==' || op === '!=' || op === '<' || op === '<=' || op === '>' || op === '>=') {
      const t = this.checkPair(e.left, e.right);
      const ok = isNumeric(t) || ((t.k === 'bool' || isStr(t)) && (op === '==' || op === '!='));
      if (!ok) {
        this.err('type-mismatch', `'${op}' cannot compare ${tyStr(t)}`, e.pos,
          t.k === 'struct' || t.k === 'array' || t.k === 'heap' ? 'compare the fields or elements one by one' : undefined);
      }
      // 符号なし整数と 0 の比較で、結果が決まっているもの。逆順のループで無限に回る原因になる。
      if (t.k === 'int' && !t.signed && e.right.kind === 'int' && !e.right.fixed && e.right.value === 0n && !isLiteral(e.left) && (op === '>=' || op === '<')) {
        this.err('always-true', `\`${op} 0\` on the unsigned type ${tyStr(t)} is always ${op === '>=' ? 'true' : 'false'}`, e.pos,
          'an unsigned value never goes below 0 (it wraps around); for a reverse loop write `for (let i = xs.length; i > 0; i--) { const j = i - 1; ... }`');
      }
      return BOOL;
    }
    const t = this.checkPair(e.left, e.right, want && isNumeric(want) ? want : undefined);
    const vec = this.vecInfo(t);
    if (vec) {
      // ベクトルの演算は、標準の関数（runtime/prelude.um）の呼び出しに置き換える。
      const scalar = this.pairScalar;
      let fn: string | undefined;
      if (!scalar) fn = ({ '+': 'Add', '-': 'Sub', '*': 'Mul', '/': 'Div' } as Record<string, string>)[op];
      else if (op === '*') fn = 'Scale';
      else if (op === '/' && scalar === 'right') fn = 'DivBy';
      if (!fn) {
        this.err('type-mismatch', `'${op}' cannot be used with ${tyStr(t)}${scalar ? ' and a number here' : ''}`, e.pos,
          'vectors support `+ - * /` with another vector of the same type, `* k` and `/ k` with a number, and unary `-`');
      }
      const [first, second] = scalar === 'left' ? [e.right, e.left] : [e.left, e.right];
      this.rewrite(e, { kind: 'call', callee: `vec${vec.n}${fn}`, args: [{ mode: 'none', expr: first }, { mode: 'none', expr: second }] });
      return this.infer(e, want);
    }
    if (this.pairScalar) this.err('type-mismatch', `'${op}' cannot mix a vector and a number`, e.pos);
    const simd = simdInfo(t);
    if (simd && (op === '+' || op === '-' || op === '*')) {
      // SIMD のベクトル（i16x8 など）の + - * は、レーンごとの演算（整数はラップ）。Simd.add などと同じ。
      this.noKernel('SIMD vector arithmetic', e.pos, 'SIMD vectors are CPU-only; in a kernel, compute each element');
      const node = e as unknown as { kind: string; fn: string; args: Expr[] };
      const args = [e.left, e.right];
      for (const k of Object.keys(node)) if (k !== 'pos' && k !== 'ty') delete (node as unknown as Record<string, unknown>)[k];
      node.kind = 'math';
      node.fn = `simd.${({ '+': 'add', '-': 'sub', '*': 'mul' } as Record<string, string>)[op]}`;
      node.args = args;
      return t;
    }
    if (op === '+' && isStr(t)) {
      this.noKernel('string concatenation', e.pos);
      return STR;
    }
    if (op === '+' || op === '-' || op === '*' || op === '/' || op === '%') {
      if (!isNumeric(t)) this.err('type-mismatch', `'${op}' needs numbers but found ${tyStr(t)}`, e.pos);
      return t;
    }
    // ビット演算とシフト
    if (t.k !== 'int') this.err('type-mismatch', `'${op}' needs integers but found ${tyStr(t)}`, e.pos);
    return t;
  }

  inferCall(e: Expr & { kind: 'call' }, want?: Ty): Ty {
    if (e.callee === 'dispatch' && !this.fns.has('dispatch')) return this.inferDispatch(e, e.args, 'auto');
    if (e.callee === 'dispatchGroups' && !this.fns.has('dispatchGroups')) return this.inferDispatch(e, e.args, 'auto', false, true);
    if (e.callee === 'print') {
      this.noKernel('`print`', e.pos);
      if (e.args.length !== 1) this.err('arg-count', '`print` takes exactly one argument', e.pos);
      const t = this.checkExpr(e.args[0].expr);
      if (!isNumeric(t) && t.k !== 'bool' && !isStr(t)) {
        this.err('type-mismatch', `\`print\` accepts numbers, bool and string but found ${tyStr(t)}`, e.args[0].expr.pos,
          'print the fields one by one');
      }
      return VOID;
    }
    if (e.callee === 'String' && !this.fns.has('String')) {
      this.noKernel('`String`', e.pos);
      if (e.args.length !== 1 || e.args[0].mode !== 'none') this.err('arg-count', '`String` takes exactly one argument', e.pos);
      const t = this.checkExpr(e.args[0].expr);
      if (!isNumeric(t) && t.k !== 'bool' && !isStr(t)) {
        this.err('type-mismatch', `\`String\` converts numbers, bool and string but found ${tyStr(t)}`, e.args[0].expr.pos,
          'convert the fields one by one');
      }
      return STR;
    }

    // gpu.loadKernel / gpu.runKernel の検査済みの式（src/gpuload.ts）をもう一度検査するときは、前の結果の型。
    if ((e.callee === '__gpu_load' || e.callee === '__gpu_run') && e.ty) return e.ty;
    // 入出力の組み込み。検査済みの式（名前が `__io_...` になっている）をもう一度検査することもある。
    const io = e.callee.startsWith('__io_') ? e.callee.slice(5) : e.callee;
    if (IO_BUILTINS.has(io) && (io !== e.callee || (!this.fns.has(io) && !this.generics.has(io)))) return this.inferIo(e, io);

    if (this.fns.has(e.callee) || this.generics.has(e.callee)) this.visible(e.callee, e.pos);
    if (this.generics.has(e.callee) && !this.fns.has(e.callee)) this.instantiate(e, want);
    const sig = this.fns.get(e.callee);
    if (!sig) {
      const hint = this.structs.get(e.callee)?.isClass ? `write \`new ${e.callee}(...)\`` : undefined;
      this.err('unknown-name', `unknown function '${e.callee}'`, e.pos, hint);
    }
    if (this.inKernel) {
      if (!sig.decl.isKernel) {
        this.err('kernel-restriction', `'${sig.name}' is not a kernel function, so a kernel cannot call it`, e.pos,
          `declare it as \`kernel function ${sig.name}(...)\` if its body only uses GPU-compatible code`);
      }
      if (e.args.some((a) => a.mode === 'mut')) {
        this.err('kernel-restriction', 'a kernel cannot pass a `mut` Buffer on to another kernel', e.pos,
          'only the dispatched kernel writes; let the helper return a value and store it into `buf[i]` here');
      }
      this.kernelCalls.push({ caller: this.curFn, callee: sig.decl, pos: e.pos });
    }
    if (sig.decl.isKernel) this.kernelUses.push({ callee: sig.decl, pos: e.pos });
    // runtime/wasm.um の関数（wasmLoad など）。wasmCustomSection はバイト列を読むだけなので wasm32 でも使える。
    if (this.ptrBits === 32 && sig.decl.pos.file === 'wasm.um' && e.pos.file !== 'wasm.um' && sig.name !== 'wasmCustomSection') {
      this.wasmOnly(sig.name, e.pos);
    }
    // runtime/media.um の関数（mediaOpenVideo など）。
    if (this.ptrBits === 32 && sig.decl.pos.file === 'media.um' && e.pos.file !== 'media.um') this.mediaOnly(sig.name, e.pos);
    this.checkArgs(sig, e.args, e.pos);
    return sig.ret;
  }

  /**
   * 入出力の組み込み関数:
   *   readLine(): string | null            標準入力から1行（改行を除く）。入力の終わりで null。
   *   inputReady(): bool                   readLine() が待たずに戻るなら true。
   *   clock(): f64                         プログラムの開始からの秒数（逆戻りしない）。
   *   stringFromBytes(bytes: u8[]): string バイト列を写した新しい文字列。
   *   readFile(path: string): u8[] | null  ファイル全体。読めなければ null。
   *   writeFile(path: string, bytes: u8[]): bool
   *   fileSize(path: string): i64 | null  ふつうのファイルの大きさ。
   *   readFileRange(path: string, offset: i64, length: i64): u8[] | null  offset から最大 length バイト。
   *   appendFile(path: string, bytes: u8[]): bool  終わりに書き足す（なければ作る）。
   *   args(): string[]                     コマンドライン引数（プログラム名を除く）。
   *   eprint(text: string): void           標準エラー出力に1行。
   * ウィンドウ（ネイティブだけ。メインスレッドから呼ぶ。runtime/window.c が SDL3 で実装する）:
   *   windowOpen(title: string, width: i32, height: i32): i32 | null  ウィンドウを開き、その番号を返す。
   *   windowClose(window: i32): void       ウィンドウを閉じる。
   *   windowPoll(event: mut Buffer<i32>, timeoutMs: i32): bool  イベントを1つ取り出して event に書く。
   *       なければ timeoutMs ミリ秒まで待ち（0 は待たない、負は来るまで待つ）、来なければ false。
   *       ふつうは prelude.um の pollWindowEvent（WindowEvent の enum を返す）を使う。
   *   windowPresent(window: i32, pixels: Buffer<u32>, width: i32, height: i32): void
   *       0xAARRGGBB の画素を width × height 並べたものをウィンドウ全体に表示する。
   * WebAssembly（ネイティブだけ。どのスレッドからも呼べる。runtime/wasm.c が wasmtime で実装する。
   * ふつうは runtime/wasm.um の wasmLoad / wasmNew / wasmCall を使う。モジュールとインスタンスは i64 の番号）:
   *   wasmCompile(bytes: u8[]): i64 | null           wasm のバイナリをコンパイルする。失敗なら null（理由は wasmError）。
   *   wasmFreeModule(module: i64): void
   *   wasmExportList(module: i64): string[]          export の一覧（形は runtime/wasm.c の um_wasm_exports）。
   *   wasmInstantiate(module: i64, fuel: i64, timeoutMs: i64, memoryBytes: i64, stubImports: bool): i64 | null
   *   wasmFreeInstance(instance: i64): void
   *   wasmInvoke(instance: i64, name: string, kinds: mut i32[], ints: mut i64[], floats: mut f64[], nargs: i32): i32
   *       export された関数を呼び、戻り値の個数を返す（失敗なら -1）。引数と戻り値は kinds / ints / floats で受け渡す。
   *   wasmRead(instance: i64, offset: i64, dst: mut 数の配列か Buffer [, start: i64, count: i64]): bool
   *   wasmWrite(instance: i64, offset: i64, src: 数の配列か Buffer [, start: i64, count: i64]): bool
   *       インスタンスのメモリの offset バイト目と、配列の start 番目から count 個（省略は全体。count が負なら終わりまで）を写す。
   *   wasmMemorySize(instance: i64): i64             メモリのバイト数（メモリがなければ -1）。
   *   wasmWat(text: string): u8[]                    テキスト形式（.wat）をバイナリにする。失敗なら空。
   *   wasmError(): string                            このスレッドで最後に失敗した理由。
   * 動画・音声（ネイティブだけ。どのスレッドからも呼べる。runtime/media.c が FFmpeg で実装する。
   * ふつうは runtime/media.um の mediaOpenVideo / mediaFrame / mediaCreate などを使う。読み手と書き手は i64 の番号）:
   *   mediaOpen(path: string, kind: i32, hwaccel: string): i64 | null   kind は 0 情報だけ、1 映像、2 音。失敗なら null（理由は mediaError）。
   *   mediaClose(reader: i64): void
   *   mediaInfoNumbers(reader: i64): f64[] / mediaInfoText(reader: i64): string[]   情報（並びは runtime/media.c）。
   *   mediaReadVideo(reader: i64, time: f64, out: mut 画素の Buffer か配列, format: i32, width: i32, height: i32, filter: i32): f64
   *       time が負なら次のコマ。コマの時刻を返す（終わりなら -1、失敗なら -2）。
   *   mediaReadAudio(reader: i64, time: f64, out: mut f32 の Buffer か配列, sampleRate: i32): i64   標本の組の数（終わり 0、失敗 -1）。
   *   mediaLastTime(reader: i64): f64                最後に読んだコマ・音の先頭の時刻。
   *   mediaCreateWriter(path: string): i64 | null
   *   mediaAddVideo(writer: i64, encoder: string, width: i32, height: i32, fpsNum: i32, fpsDen: i32, bitRate: i64, options: string): bool
   *   mediaAddAudio(writer: i64, encoder: string, sampleRate: i32, channels: i32, bitRate: i64, options: string): bool
   *   mediaWriteVideo(writer: i64, pixels: 画素の Buffer か配列, format: i32, width: i32, height: i32): bool
   *   mediaWriteAudio(writer: i64, samples: f32 の Buffer か配列, count: i64): bool
   *   mediaFinishWriter(writer: i64): bool           書き手は閉じる。
   *   mediaWriterText(writer: i64): string[]         [映像の符号器, 音の符号器]
   *   mediaEncoderList(probe: bool): string[]        5 つずつ: 名前, "video" / "audio", 形式, 説明, "hw" / "sw"。
   *   mediaError(): string                           このスレッドで最後に失敗した理由。
   *   mediaVersion(): string                         FFmpeg のライブラリの版とライセンス。
   * 引数は借りるだけ。結果は呼び出した側が所有する新しい値。
   */
  inferIo(e: Expr & { kind: 'call' }, name: string): Ty {
    this.noKernel(`\`${name}\``, e.pos, 'kernel functions run on the GPU too, so they cannot do input or output; call it from a normal function and pass the result in');
    const bytes: Ty = { k: 'heap', elem: PRIMS.u8 };
    const shapes: Record<string, { params: [string, Ty][]; ret: Ty; mut?: number[] }> = {
      readLine: { params: [], ret: this.optionalOf(STR, e.pos) },
      inputReady: { params: [], ret: BOOL },
      clock: { params: [], ret: F64 },
      stringFromBytes: { params: [['bytes', bytes]], ret: STR },
      readFile: { params: [['path', STR]], ret: this.optionalOf(bytes, e.pos) },
      writeFile: { params: [['path', STR], ['bytes', bytes]], ret: BOOL },
      fileSize: { params: [['path', STR]], ret: this.optionalOf(I64, e.pos) },
      readFileRange: { params: [['path', STR], ['offset', I64], ['length', I64]], ret: this.optionalOf(bytes, e.pos) },
      appendFile: { params: [['path', STR], ['bytes', bytes]], ret: BOOL },
      args: { params: [], ret: { k: 'heap', elem: STR } },
      eprint: { params: [['text', STR]], ret: VOID },
      windowOpen: { params: [['title', STR], ['width', I32], ['height', I32]], ret: this.optionalOf(I32, e.pos) },
      windowClose: { params: [['window', I32]], ret: VOID },
      windowPoll: { params: [['event', { k: 'heap', elem: I32, buffer: true }], ['timeoutMs', I32]], ret: BOOL, mut: [0] },
      windowPresent: { params: [['window', I32], ['pixels', { k: 'heap', elem: PRIMS.u32, buffer: true }], ['width', I32], ['height', I32]], ret: VOID },
      wasmCompile: { params: [['bytes', bytes]], ret: this.optionalOf(I64, e.pos) },
      wasmFreeModule: { params: [['module', I64]], ret: VOID },
      wasmExportList: { params: [['module', I64]], ret: { k: 'heap', elem: STR } },
      wasmInstantiate: {
        params: [['module', I64], ['fuel', I64], ['timeoutMs', I64], ['memoryBytes', I64], ['stubImports', BOOL]], ret: this.optionalOf(I64, e.pos),
      },
      wasmFreeInstance: { params: [['instance', I64]], ret: VOID },
      wasmInvoke: {
        params: [['instance', I64], ['name', STR], ['kinds', { k: 'heap', elem: I32 }], ['ints', { k: 'heap', elem: I64 }], ['floats', { k: 'heap', elem: F64 }], ['nargs', I32]],
        ret: I32, mut: [2, 3, 4],
      },
      wasmRead: { params: name === 'wasmRead' ? this.wasmCopyParams(e, name, 'dst') : [], ret: BOOL, mut: [2] },
      wasmWrite: { params: name === 'wasmWrite' ? this.wasmCopyParams(e, name, 'src') : [], ret: BOOL },
      wasmMemorySize: { params: [['instance', I64]], ret: I64 },
      wasmWat: { params: [['text', STR]], ret: bytes },
      wasmError: { params: [], ret: STR },
      mediaOpen: { params: [['path', STR], ['kind', I32], ['hwaccel', STR]], ret: this.optionalOf(I64, e.pos) },
      mediaClose: { params: [['reader', I64]], ret: VOID },
      mediaInfoNumbers: { params: [['reader', I64]], ret: { k: 'heap', elem: F64 } },
      mediaInfoText: { params: [['reader', I64]], ret: { k: 'heap', elem: STR } },
      mediaReadVideo: {
        params: name === 'mediaReadVideo'
          ? [['reader', I64], ['time', F64], this.mediaArrayParam(e, 2, name, 'out', false), ['format', I32], ['width', I32], ['height', I32], ['filter', I32]] : [],
        ret: F64, mut: [2],
      },
      mediaReadAudio: {
        params: name === 'mediaReadAudio' ? [['reader', I64], ['time', F64], this.mediaArrayParam(e, 2, name, 'out', true), ['sampleRate', I32]] : [],
        ret: I64, mut: [2],
      },
      mediaLastTime: { params: [['reader', I64]], ret: F64 },
      mediaCreateWriter: { params: [['path', STR]], ret: this.optionalOf(I64, e.pos) },
      mediaAddVideo: {
        params: [['writer', I64], ['encoder', STR], ['width', I32], ['height', I32], ['fpsNum', I32], ['fpsDen', I32], ['bitRate', I64], ['options', STR]], ret: BOOL,
      },
      mediaAddAudio: { params: [['writer', I64], ['encoder', STR], ['sampleRate', I32], ['channels', I32], ['bitRate', I64], ['options', STR]], ret: BOOL },
      mediaWriteVideo: {
        params: name === 'mediaWriteVideo' ? [['writer', I64], this.mediaArrayParam(e, 1, name, 'pixels', false), ['format', I32], ['width', I32], ['height', I32]] : [],
        ret: BOOL,
      },
      mediaWriteAudio: { params: name === 'mediaWriteAudio' ? [['writer', I64], this.mediaArrayParam(e, 1, name, 'samples', true), ['count', I64]] : [], ret: BOOL },
      mediaFinishWriter: { params: [['writer', I64]], ret: BOOL },
      mediaWriterText: { params: [['writer', I64]], ret: { k: 'heap', elem: STR } },
      mediaEncoderList: { params: [['probe', BOOL]], ret: { k: 'heap', elem: STR } },
      mediaError: { params: [], ret: STR },
      mediaVersion: { params: [], ret: STR },
    };
    // wasm.um の中の呼び出しは報告しない（wasm.um の関数を呼んだ所で報告する。下の wasmOnly）。
    if (name.startsWith('wasm') && this.ptrBits === 32 && e.pos.file !== 'wasm.um') this.wasmOnly(name, e.pos);
    if (name.startsWith('media') && this.ptrBits === 32 && e.pos.file !== 'media.um') this.mediaOnly(name, e.pos);
    if (name.startsWith('window') && this.ptrBits === 32) {
      this.err('not-implemented', `\`${name}\` is only available in native programs`, e.pos,
        'windows need a desktop OS; build with `--target host` (the wasm32 target has no window)');
    }
    const shape = shapes[name];
    const decl: FnDecl = { kind: 'fn', name, exported: false, params: [], body: [], pos: e.pos };
    const sig: FnSig = { key: name, name, params: shape.params.map(([n, ty], i) => ({ name: n, mode: (shape.mut?.includes(i) ? 'mut' : 'none') as Mode, ty })), ret: shape.ret, exported: false, decl };
    this.checkArgs(sig, e.args, e.pos);
    e.callee = `__io_${name}`;
    return shape.ret;
  }

  /** WebAssembly を実行する関数は、wasm32 のプログラムでは使えない（wasmtime はネイティブのプログラムにリンクするため）。 */
  wasmOnly(name: string, pos: Pos): void {
    this.err('not-implemented', `\`${name}\` is only available in native programs`, pos,
      'running WebAssembly modules needs wasmtime, which is linked into native programs; build with `--target host`');
  }

  /** 動画・音声の関数は、wasm32 のプログラムでは使えない（FFmpeg はネイティブのプログラムにリンクするため）。 */
  mediaOnly(name: string, pos: Pos): void {
    this.err('not-implemented', `\`${name}\` is only available in native programs`, pos,
      'reading and writing video and audio needs FFmpeg, which is linked into native programs; build with `--target host`');
  }

  /**
   * mediaReadVideo / mediaWriteVideo / mediaReadAudio / mediaWriteAudio の画素・標本の引数。数（8〜64 ビットの整数、f32、f64）か
   * vec4<f32> の可変長配列か Buffer なら何でもよいので、実引数の型をそのまま仮引数の型にする（形式と合うかはランタイムが確かめる）。
   * 音（samplesOnly）は f32 だけ。要素の型の印（整数ならバイト数、浮動小数点なら 0x100 | バイト数）を呼び出しの式に覚えておく。
   */
  mediaArrayParam(e: Expr & { kind: 'call' }, index: number, name: string, what: string, samplesOnly: boolean): [string, Ty] {
    const arg = e.args[index];
    if (!arg) return [what, { k: 'heap', elem: PRIMS.u8, buffer: true }];
    if (!isPlace(arg.expr)) {
      this.err('type-mismatch', `the ${what} of \`${name}\` must be a variable (or a field / element of one)`, arg.expr.pos, 'store it in a variable first');
    }
    const t = this.checkExpr(arg.expr);
    let scalar = 0;
    if (t.k === 'heap' && !t.str && !t.racy) {
      const el = t.elem;
      const vec = this.vecInfo(el);
      if (el.k === 'int' && !el.usize) scalar = el.bits / 8;
      else if (el.k === 'float' && (el.bits === 32 || el.bits === 64)) scalar = 0x100 | (el.bits / 8);
      else if (vec && vec.n === 4 && vec.elem.k === 'float' && vec.elem.bits === 32) scalar = 0x104;
    }
    if (samplesOnly && (scalar !== 0x104 || (t.k === 'heap' && t.elem.k !== 'float'))) {
      this.err('type-mismatch', `\`${name}\` needs samples in a Buffer<f32> or f32[], but found ${tyStr(t)}`, arg.expr.pos,
        'audio is read and written as interleaved f32 samples');
    } else if (scalar === 0) {
      this.err('type-mismatch', `\`${name}\` needs pixels in a Buffer or array of numbers, but found ${tyStr(t)}`, arg.expr.pos,
        'use Buffer<u8> (Rgba8, Yuv420p8), Buffer<u32> (Argb32), Buffer<f32> or Buffer<vec4<f32>> (RgbaF32), or Buffer<u16> (Rgba16, Yuv420p10)');
    }
    e.elemScalar = scalar;
    return [what, t];
  }

  /**
   * wasmRead / wasmWrite の引数。写す配列は、数（8〜64 ビットの整数、f32、f64）の可変長配列か Buffer なら何でもよいので、
   * 実引数の型をそのまま仮引数の型にする。配列は変数（かそのフィールド・要素）に限る（もう一度検査しても同じになるように）。
   * 後ろの start と count は省略できる（省略は配列全体）。
   */
  wasmCopyParams(e: Expr & { kind: 'call' }, name: string, what: string): [string, Ty][] {
    const head: [string, Ty][] = [['instance', I64], ['offset', I64]];
    const tail: [string, Ty][] = e.args.length > 3 ? [['start', I64], ['count', I64]] : [];
    const arg = e.args[2];
    if (!arg) return [...head, [what, { k: 'heap', elem: PRIMS.u8 }], ...tail];
    if (!isPlace(arg.expr)) {
      this.err('type-mismatch', `the array of \`${name}\` must be a variable (or a field / element of one)`, arg.expr.pos,
        'store the array in a variable first');
    }
    const t = this.checkExpr(arg.expr);
    const okElem = (el: Ty) => (el.k === 'int' && !el.usize) || (el.k === 'float' && (el.bits === 32 || el.bits === 64));
    if (t.k !== 'heap' || t.str || t.racy || !okElem(t.elem)) {
      this.err('type-mismatch', `\`${name}\` copies between wasm memory and an array or Buffer of numbers, but found ${tyStr(t)}`, arg.expr.pos,
        'use u8[], Buffer<u8>, f32[], Buffer<f32>, or another array of fixed-size integers, f32 or f64');
    }
    return [...head, [what, t], ...tail];
  }

  /**
   * `dispatch(kernel, count, ...args)`: kernel を添字 0 から count 未満について実行する。
   * 式の節を dispatch に書き換える。
   */
  inferDispatch(e: Expr, args: Arg[], device: 'auto' | 'cpu' | 'gpu', submit = false, groups = false): Ty {
    this.noKernel(submit ? '`gpu.submit`' : '`dispatch`', e.pos);
    if (groups && args.length < 3) {
      this.err('arg-count', '`dispatchGroups` needs a kernel, the number of groups and the group size', e.pos,
        'e.g. `dispatchGroups(reduce, n / 256, 256, mut out, xs)` runs n / 256 groups of 256 threads');
    }
    if (device === 'gpu' && !this.gpu && !submit) {
      this.err('not-implemented', '`gpu.dispatch` is not implemented for this target yet', e.pos,
        'the GPU backends are Metal (build natively on a Mac with UM_METAL=1) and WebGPU (wasm32); otherwise use `dispatch(...)` (automatic) or `cpu.dispatch(...)`');
    }
    if (args.length < 2) {
      this.err('arg-count', '`dispatch` needs a kernel and an element count', e.pos, 'e.g. `dispatch(integrate, ps.length, mut ps, dt)`');
    }
    const k = args[0].expr;
    if (k.kind === 'ident' && this.fns.has(k.name)) this.visible(k.name, k.pos);
    const sig = k.kind === 'ident' ? this.fns.get(k.name) : undefined;
    if (args[0].mode !== 'none' || !sig || !sig.decl.isKernel) {
      this.err('type-mismatch', 'the first argument of `dispatch` must be the name of a kernel function', k.pos,
        'declare it as `kernel function name(i: u32, ...): void`');
    }
    const first = sig.params[0]?.ty;
    if (sig.ret.k !== 'void' || !(first && tyEq(first, U32))) {
      this.err('type-mismatch', `kernel '${sig.name}' cannot be dispatched`, k.pos,
        'a dispatched kernel takes the element index as its first parameter (`i: u32`) and returns void');
    }
    if (args[1].mode !== 'none') this.err('arg-mode', 'the element count is passed by value', args[1].expr.pos);
    this.expect(args[1].expr, U32);
    if (groups) {
      if (args[2].mode !== 'none') this.err('arg-mode', 'the group size is passed by value', args[2].expr.pos);
      this.expect(args[2].expr, U32);
      const g = args[2].expr;
      if (g.kind === 'int' && (g.value < 1n || g.value > 1024n)) {
        this.err('kernel-restriction', `the group size must be 1 to 1024, not ${g.value}`, g.pos,
          'a group runs on one GPU core; use 64 to 256 threads (WebGPU guarantees 256, Metal allows up to 1024)');
      }
    }
    const rest = args.slice(groups ? 3 : 2);
    this.groupDispatches.push({ kernel: sig.decl, groups, pos: k.pos });
    this.checkArgs({ ...sig, params: sig.params.slice(1) }, rest, e.pos, undefined, true);
    for (const p of sig.params.slice(1)) {
      // 値の引数は 4 バイト単位で GPU に渡すので、8 / 16 ビットの整数は直接は渡さない。
      if (isSmallInt(p.ty)) {
        this.err('kernel-type', `parameter '${p.name}' of kernel '${sig.name}' has type ${tyStr(p.ty)}, which cannot be passed to a dispatched kernel`, k.pos,
          `take it as \`${(p.ty as Ty & { k: 'int' }).signed ? 'i32' : 'u32'}\` and convert inside the kernel with \`as ${tyStr(p.ty)}\``);
      }
    }
    if (this.gpu && device !== 'cpu') {
      // GPU に値を渡すときは、CPU と同じビット表現でなければならない。
      for (const p of sig.params.slice(1)) {
        if (p.ty.k !== 'heap' && p.ty.k !== 'bool' && !this.isShareable(p.ty)) {
          this.err('kernel-type', `parameter '${p.name}' of kernel '${sig.name}' has type ${tyStr(p.ty)}, which cannot be sent to the GPU`, k.pos,
            'a struct or array passed to a dispatched kernel must not contain `bool`; use `u32` (0 or 1), or call `cpu.dispatch`');
        }
      }
    }

    if (submit) {
      // 終わる前に戻るので、CPU と GPU が同じ Buffer を同時に触りうる。同時に触ってよいと宣言された Buffer だけを受け付ける。
      for (const p of sig.params.slice(1)) {
        // 読むだけの普通の Buffer（重みなど）は渡せる。終わる前に CPU が書き換えると、kernel は古い値と新しい値の
        // 混ざったものを読むことがある（停止や範囲外の読み書きは起きない）。
        if (p.ty.k === 'heap' && !p.ty.racy && p.mode === 'mut') {
          this.err('type-mismatch', `kernel '${sig.name}' writes to '${p.name}' (\`mut ${tyStr(p.ty)}\`), which cannot be passed to \`gpu.submit\``, k.pos,
            '`gpu.submit` returns before the kernel has finished, so a buffer the kernel writes must be a `RacyBuffer` (a Buffer it only reads is fine); use `gpu.dispatch(...)` to run it and wait');
        }
      }
    }

    const node = e as unknown as { kind: string; kernel: string; count: Expr; args: Arg[]; device: string; async?: boolean };
    node.kind = 'dispatch';
    node.kernel = sig.name;
    node.count = args[1].expr;
    node.args = rest;
    node.device = device;
    if (submit) node.async = true;
    if (groups) (node as unknown as { group: Expr }).group = args[2].expr;
    return submit ? U64 : VOID;
  }

  // ---- スレッド ----

  /** `thread.join` に渡している持ち手の式（この式だけは持ち手を読んでよい）。 */
  joinArg: Expr | null = null;

  /** `thread.名前(...)` の呼び出しか（同じ名前の変数があれば違う）。names を渡せばその名前だけ。 */
  isThreadCall(e: Expr, ...names: string[]): e is Expr & { kind: 'mcall' } {
    return e.kind === 'mcall' && e.recv.kind === 'ident' && e.recv.name === 'thread' && !this.lookup('thread') &&
      (names.length === 0 || names.includes(e.name));
  }

  /** RacyBuffer の要素を指す式か。 */
  isRacyElem(e: Expr): boolean {
    return e.kind === 'index' && e.base.ty?.k === 'heap' && e.base.ty.racy === true;
  }

  /** `thread.join(h)`、`thread.cpus()`、`thread.yield()`。spawn / spawnEach は const の宣言の初期化でだけ書ける（checkSpawn）。 */
  inferThread(e: Expr & { kind: 'mcall' }): Ty {
    this.noKernel(`\`thread.${e.name}\``, e.pos, 'kernel functions run on the GPU too, so they cannot start or join threads');
    const node = e as unknown as Record<string, unknown>;
    if (e.name === 'spawn' || e.name === 'spawnEach') {
      this.err('thread', `\`thread.${e.name}\` can only be the whole initial value of a \`const\``, e.pos,
        `write \`const h = thread.${e.name}(...);\`; the thread is joined at the end of the block (or earlier with \`thread.join(h)\`)`);
    }
    if (e.name === 'cpus') {
      if (e.args.length !== 0) this.err('arg-count', '`thread.cpus()` takes no argument', e.pos);
      for (const k of Object.keys(node)) if (k !== 'pos') delete node[k];
      Object.assign(node, { kind: 'thread', op: 'cpus', args: [] });
      e.ty = U32;
      return U32;
    }
    if (e.name === 'yield') {
      if (e.args.length !== 0) this.err('arg-count', '`thread.yield()` takes no argument', e.pos);
      for (const k of Object.keys(node)) if (k !== 'pos') delete node[k];
      Object.assign(node, { kind: 'thread', op: 'yield', args: [] });
      e.ty = VOID;
      return VOID;
    }
    if (e.name !== 'join') {
      this.err('unknown-method', `\`thread.${e.name}\` is not available`, e.pos, 'available: `thread.spawn`, `thread.spawnEach`, `thread.join`, `thread.cpus`, `thread.yield`');
    }
    if (e.args.length !== 1) this.err('arg-count', '`thread.join` takes exactly one argument (the handle)', e.pos);
    const a = e.args[0];
    if (this.isThreadCall(a.expr)) this.checkExpr(a.expr); // `thread.join(thread.spawn(...))` は起動の側の誤りとして報告する
    const sym = a.expr.kind === 'ident' ? this.lookup(a.expr.name) : undefined;
    if (a.mode !== 'none' || !sym?.thread) {
      this.err('type-mismatch', 'the argument of `thread.join` must be a thread handle', a.expr.pos,
        'pass the `const` that holds it: `const h = thread.spawn(f, ...); ... thread.join(h)`');
    }
    this.joinArg = a.expr;
    try {
      this.checkExpr(a.expr);
    } finally {
      this.joinArg = null;
    }
    // 合流は持ち手を消費する（2回目は use-after-move）。合流しなかった持ち手は、ブロックの終わりで自動で合流する。
    this.moved.add(sym);
    this.movedAt.set(sym, a.expr.pos);
    const ret = sym.thread.ret;
    const t: Ty = sym.thread.each && ret.k !== 'void' ? { k: 'heap', elem: ret } : ret;
    if (t.k === 'heap') this.checkElem(t, e.pos);
    const args = e.args;
    for (const k of Object.keys(node)) if (k !== 'pos') delete node[k];
    Object.assign(node, { kind: 'thread', op: 'join', args });
    e.ty = t;
    return t;
  }

  /**
   * `const h = thread.spawn(f, 引数...)` / `const g = thread.spawnEach(f, own items, 引数...)`。
   * 持ち手 h は移動できず、`thread.join(h)` か、宣言を含むブロックの終わり（自動の合流）で合流する。
   * 読み取り借用で渡した変数は、ブロックの終わりまで凍結する（書き換え・移動・mut 渡しができない）。
   * 解放は宣言と逆の順なので、借りた変数は必ずスレッドの合流のあとで解放される。
   */
  checkSpawn(s: Stmt & { kind: 'let' }): void {
    const e = s.init as Expr & { kind: 'mcall' };
    const each = e.name === 'spawnEach';
    const what = `\`thread.${e.name}\``;
    this.noKernel(what, e.pos, 'kernel functions run on the GPU too, so they cannot start threads');
    if (!s.isConst) {
      this.err('thread', `a thread handle must be held in a \`const\`, not a \`let\``, s.pos, `write \`const ${s.name} = thread.${e.name}(...);\``);
    }
    if (e.args.length < (each ? 2 : 1)) {
      this.err('arg-count', `${what} needs a function${each ? ' and an array' : ''}`, e.pos,
        each ? 'e.g. `const g = thread.spawnEach(search, own workers, shared);`' : 'e.g. `const h = thread.spawn(search, a, b);`');
    }
    const k = e.args[0].expr;
    const builtin = k.kind === 'ident' && (BUILTINS.has(k.name) || IO_BUILTINS.has(k.name)) && !this.fns.has(k.name) && !this.generics.has(k.name);
    if (e.args[0].mode !== 'none' || k.kind !== 'ident' || this.lookup(k.name) || !(this.fns.has(k.name) || this.generics.has(k.name))) {
      this.err('type-mismatch', `the first argument of ${what} must be the name of a function`, k.pos,
        builtin ? `'${(k as Expr & { kind: 'ident' }).name}' is built in; wrap it in a function of your own and start that`
          : k.kind === 'mcall' || k.kind === 'field' ? 'methods cannot be started directly; write a function that takes the object and calls the method'
          : `e.g. \`const h = thread.${e.name}(search, ...);\``);
    }
    this.visible(k.name, k.pos);
    const rest = e.args.slice(1);
    let callee = k.name;
    let items: Expr | undefined;
    let elem: Ty | undefined;
    if (each) {
      const ia = rest[0];
      const it = this.checkExpr(ia.expr);
      if (ia.mode !== 'own' || it.k !== 'heap' || it.buffer || it.str || ia.expr.kind !== 'ident') {
        this.err('type-mismatch', `the second argument of ${what} must be an array variable passed with \`own\``, ia.expr.pos,
          'write `thread.spawnEach(f, own items, ...)`; each thread gets one element (moved), and `thread.join` returns the results in the same order');
      }
      items = ia.expr;
      elem = it.elem;
    }
    if (this.generics.has(callee) && !this.fns.has(callee)) {
      // ジェネリック関数: 呼び出しと同じ規則で型引数を決める（spawnEach の最初の引数は要素の型）。
      const args: Arg[] = each
        ? [{ mode: 'none', expr: { kind: 'index', base: { kind: 'ident', name: (items as Expr & { kind: 'ident' }).name, pos: items!.pos }, index: { kind: 'int', value: 0n, pos: items!.pos }, pos: items!.pos } }, ...rest.slice(1)]
        : rest;
      const call = { kind: 'call', callee, args, pos: e.pos } as Expr & { kind: 'call' };
      this.instantiate(call, undefined);
      callee = call.callee;
    }
    const sig = this.fns.get(callee)!;
    if (sig.decl.isKernel) {
      this.err('type-mismatch', `'${k.name}' is a kernel function, which ${what} cannot start`, k.pos,
        'run a kernel with `dispatch` / `gpu.submit`, or wrap that call in a normal function and start the function');
    }
    const mutParam = sig.params.find((p) => p.mode === 'mut');
    if (mutParam) {
      this.err('thread', `'${sig.name}' takes '${mutParam.name}' as \`mut\`, which a thread cannot borrow`, k.pos,
        'several threads could change the same value; pass it with `own` and return it from the function, or share a `RacyBuffer` (its elements can be written through a read-only borrow)');
    }
    const params = each ? sig.params.slice(1) : sig.params;
    const args = each ? rest.slice(1) : rest;
    if (each) {
      const p0 = sig.params[0];
      if (!p0) this.err('arg-count', `'${sig.name}' must take the element as its first parameter`, k.pos);
      if (!tyEq(p0.ty, elem!)) {
        this.err('type-mismatch', `'${sig.name}' takes ${tyStr(p0.ty)} first, but the elements are ${tyStr(elem!)}`, k.pos,
          `each thread calls \`${sig.name}(items[k], ...)\`, so its first parameter must be \`${p0.name}: ${this.isCopy(elem!) ? '' : 'own '}${tyStr(elem!)}\``);
      }
      if (!this.isCopy(elem!) && p0.mode !== 'own') {
        this.err('type-mismatch', `'${sig.name}' must take its first parameter with \`own\`: each thread receives its element by move`, k.pos,
          `declare it as \`${p0.name}: own ${tyStr(elem!)}\` and return it if the caller needs it back`);
      }
      const ownParam = params.find((p) => p.mode === 'own');
      if (ownParam) {
        this.err('thread', `'${sig.name}' takes '${ownParam.name}' with \`own\`, but ${what} shares the other arguments between all threads`, k.pos,
          'put the per-thread values into the array (`items`) so that each thread receives its own');
      }
    }
    if (args.length !== params.length) {
      this.err('arg-count', `'${sig.name}' takes ${sig.params.length} argument(s) but ${(each ? 1 : 0) + args.length} were given`, e.pos);
    }
    // 配列は先に移動する（同じ配列を共有の引数にも渡すと、移動済みの変数の使用としてエラーになる）。
    if (items) this.consume(items);
    this.checkArgs({ ...sig, params }, args, e.pos);
    // 読み取り借用で渡すヒープを持つ値は、スレッドが読んでいる間（合流まで）書き換えられないよう凍結する。
    params.forEach((p, i) => {
      if (p.mode !== 'none' || this.isCopy(p.ty)) return;
      const a = args[i].expr;
      if (!isPlace(a)) {
        this.err('thread', `a thread cannot borrow a temporary value`, a.pos,
          'the value would be freed at the end of this statement while the thread still reads it; store it in a `const` first and pass that');
      }
      const at = placePath(a);
      if (at) this.frozen.push({ ...at, text: placeText(a), by: 'thread' });
    });
    if (s.tyAnn) {
      // `const h: Thread<R> = thread.spawn(...)` の注釈だけは書ける。
      const n = s.tyAnn;
      const want = each ? 'Threads' : 'Thread';
      if (n.kind !== 'generic' || n.name !== want || n.args.length !== 1) {
        this.err('type-mismatch', `${what} returns a \`${want}<R>\` handle`, n.pos, `write \`const ${s.name}: ${want}<${tyStr(sig.ret)}> = ...\` or leave the type out`);
      }
      const r = this.resolve(n.args[0]);
      if (!tyEq(r, sig.ret)) {
        this.err('type-mismatch', `'${sig.name}' returns ${tyStr(sig.ret)}, not ${tyStr(r)}`, n.pos, `write \`${want}<${tyStr(sig.ret)}>\``);
      }
    }
    const node = e as unknown as Record<string, unknown>;
    const all = e.args.slice(1);
    for (const key of Object.keys(node)) if (key !== 'pos') delete node[key];
    Object.assign(node, { kind: 'thread', op: each ? 'each' : 'spawn', fn: callee, args: all, ty: U64 });
    s.sym = this.declare(s.name, U64, false, s.pos);
    s.sym.thread = { ret: sig.ret, each };
  }

  /** `Math.sqrt(x)` など。式の節を math に書き換える。 */
  inferMath(e: Expr & { kind: 'mcall' }): Ty {
    const spec = MATH[e.name];
    if (!spec) {
      this.err('unknown-method', `\`Math.${e.name}\` is not available`, e.pos, `available: ${Object.keys(MATH).map((n) => `Math.${n}`).join(', ')}`);
    }
    if (e.args.length !== spec.argc) {
      this.err('arg-count', `\`Math.${e.name}\` takes ${spec.argc} argument(s) but ${e.args.length} were given`, e.pos);
    }
    for (const a of e.args) if (a.mode !== 'none') this.err('arg-mode', 'numbers are passed by value', a.expr.pos);
    const exprs = e.args.map((a) => a.expr);
    if (e.name === 'prefetch') {
      // 先読み: 配列の要素 arr[i] のあるメモリを、使う前にキャッシュへ読み込むよう CPU に頼む。結果は返さない。
      // 範囲外の添字でも何も起きない（読み込みも停止もしない）。GPU にはないので kernel では使えない。
      this.noKernel('`Math.prefetch`', e.pos, 'the GPU has no prefetch; remove it from the kernel');
      const at = this.checkExpr(exprs[0]);
      const it = this.checkExpr(exprs[1]);
      if (at.k !== 'heap' || at.str) {
        this.err('type-mismatch', `\`Math.prefetch\` needs a growable array, Buffer or RacyBuffer but found ${tyStr(at)}`, exprs[0].pos,
          at.k === 'array' ? 'a fixed-length array lives in registers or on the stack, so there is nothing to prefetch' : 'pass the array and the index, e.g. `Math.prefetch(table, key & mask)`');
      } else if (!isPlace(exprs[0])) {
        this.err('type-mismatch', '`Math.prefetch` needs a variable (or a field / element of one) as the array', exprs[0].pos,
          'store the array in a variable first');
      }
      if (it.k !== 'int') this.err('type-mismatch', `the index of \`Math.prefetch\` must be an integer but found ${tyStr(it)}`, exprs[1].pos);
      const node = e as unknown as { kind: string; fn: string; args: Expr[] };
      node.kind = 'math';
      node.fn = e.name;
      node.args = exprs;
      return VOID;
    }
    // マスクつきシフトは、値とシフト量の型を揃えない（量はどの整数型でもよい。値のビット幅で割った余りにする）。
    if (e.name === 'shl' || e.name === 'shr') {
      const vt = this.checkExpr(exprs[0]);
      const st = this.checkExpr(exprs[1]);
      if (vt.k !== 'int') this.err('type-mismatch', `\`Math.${e.name}\` needs an integer value but found ${tyStr(vt)}`, exprs[0].pos, 'convert first, e.g. `x as u64`');
      if (st.k !== 'int') this.err('type-mismatch', `\`Math.${e.name}\` needs an integer shift amount but found ${tyStr(st)}`, exprs[1].pos, 'convert first, e.g. `k as u32`');
      const node = e as unknown as { kind: string; fn: string; args: Expr[] };
      node.kind = 'math';
      node.fn = e.name;
      node.args = exprs;
      return vt;
    }
    const t = exprs.length === 2 ? this.checkPair(exprs[0], exprs[1]) : this.checkExpr(exprs[0]);
    const ok = spec.kind === 'float' ? t.k === 'float' : spec.kind === 'int' ? t.k === 'int' : isNumeric(t);
    if (!ok) {
      const need = spec.kind === 'float' ? 'a floating-point number' : spec.kind === 'int' ? 'an integer' : 'numbers';
      const hint = spec.kind === 'int' ? 'convert first, e.g. `x as u32`'
        : spec.kind !== 'float' ? undefined
        : t.k === 'int' && (e.name === 'floor' || e.name === 'ceil')
          ? `the argument is already an integer (integer \`/\` truncates toward zero), so drop \`Math.${e.name}\``
          : 'convert first, e.g. `x as f32`';
      this.err('type-mismatch', `\`Math.${e.name}\` needs ${need} but found ${tyStr(t)}`, e.pos, hint);
    }
    if (e.name === 'clz32') {
      this.err('ts-feature', '`Math.clz32` is not available', e.pos, 'use `Math.clz(x)`; it works on any integer type');
    }
    const node = e as unknown as { kind: string; fn: string; args: Expr[] };
    node.kind = 'math';
    node.fn = e.name;
    node.args = exprs;
    return t;
  }

  /**
   * SIMD の組み込み `Simd.名前(...)`。ベクトルは 128 ビットの固定長配列（i8x16 / i16x8 / i32x4 / f32x4 と符号なしの版）。
   * CPU だけで使える（kernel ではエラー）。意味はどのターゲットでも同じ（docs/guide.ja.md「SIMD のベクトル」）。
   */
  inferSimd(e: Expr & { kind: 'mcall' }, want?: Ty): Ty {
    const names = ['add', 'sub', 'mul', 'min', 'max', 'splat', 'shuffle', 'dot', 'mulAddWide', 'sum', 'load', 'store'];
    if (!names.includes(e.name)) {
      this.err('unknown-method', `\`Simd.${e.name}\` is not available`, e.pos, `available: ${names.map((n) => `Simd.${n}`).join(', ')}`);
    }
    this.noKernel(`\`Simd.${e.name}\``, e.pos, 'SIMD vectors are CPU-only; in a kernel, compute each element');
    const argc: Record<string, number> = { splat: 1, sum: 1, load: 2, store: 3, shuffle: 3, dot: 3, mulAddWide: 3 };
    const n = argc[e.name] ?? 2;
    if (e.args.length !== n) this.err('arg-count', `\`Simd.${e.name}\` takes ${n} argument(s) but ${e.args.length} were given`, e.pos);
    e.args.forEach((a, i) => {
      const want = e.name === 'store' && i === 0 ? 'mut' : 'none';
      if (a.mode !== want) {
        this.err('arg-mode', want === 'mut' ? '`Simd.store` writes to the array: pass it as `mut`' : 'SIMD values are passed by value', a.expr.pos,
          want === 'mut' ? 'e.g. `Simd.store(mut xs, i, v)`' : undefined);
      }
    });
    const exprs = e.args.map((a) => a.expr);
    const vecOf = (t: Ty, pos: Pos, what = 'a SIMD vector'): { elem: Ty & { k: 'int' | 'float' }; lanes: number } => {
      const info = simdInfo(t);
      if (!info) this.err('type-mismatch', `\`Simd.${e.name}\` needs ${what} but found ${tyStr(t)}`, pos,
        'SIMD vectors are i8x16, u8x16, i16x8, u16x8, i32x4, u32x4 and f32x4 (the same as i8[16], ..., f32[4])');
      return info;
    };
    const named = (t: string): Ty => this.resolve({ kind: 'name', name: t, pos: e.pos });
    let ret: Ty;
    switch (e.name) {
      case 'add':
      case 'sub':
      case 'mul':
      case 'min':
      case 'max': {
        ret = this.checkPair(exprs[0], exprs[1], want && simdInfo(want) ? want : undefined);
        vecOf(ret, e.pos);
        break;
      }
      case 'splat': {
        if (want && simdInfo(want)) {
          this.expect(exprs[0], simdInfo(want)!.elem);
          ret = want;
        } else {
          const t = this.checkExpr(exprs[0]);
          const ok = (t.k === 'int' && !t.usize && t.bits <= 32) || (t.k === 'float' && t.bits === 32);
          if (!ok) this.err('type-mismatch', `\`Simd.splat\` needs an 8, 16 or 32-bit integer or an f32 but found ${tyStr(t)}`, exprs[0].pos,
            'give the vector type, e.g. `const v: i16x8 = Simd.splat(3)`, or convert the value, e.g. `Simd.splat(x as i16)`');
          ret = { k: 'array', elem: t, len: 128 / (t as { bits: number }).bits };
        }
        break;
      }
      case 'shuffle': {
        // Simd.shuffle(a, b, [i0, i1, ...]): 結果のレーン k は、i_k < レーン数なら a[i_k]、そうでなければ b[i_k - レーン数]。
        ret = this.checkPair(exprs[0], exprs[1]);
        const info = vecOf(ret, e.pos);
        const idx = exprs[2];
        const lanes: number[] = [];
        if (idx.kind === 'arrayLit' && idx.elems.length === info.lanes) {
          for (const x of idx.elems) {
            if (x.kind !== 'int' || x.value < 0n || x.value >= BigInt(2 * info.lanes)) break;
            lanes.push(Number(x.value));
          }
        }
        if (lanes.length !== info.lanes) {
          this.err('type-mismatch', `the lanes of \`Simd.shuffle\` must be an array literal of ${info.lanes} integer literals from 0 to ${2 * info.lanes - 1}`, idx.pos,
            `e.g. \`Simd.shuffle(a, b, [${Array.from({ length: info.lanes }, (_, k) => k % 2 === 0 ? k / 2 : info.lanes + (k - 1) / 2).join(', ')}])\` interleaves a and b`);
        }
        (e as unknown as { lanes: number[] }).lanes = lanes;
        exprs.pop();
        break;
      }
      case 'dot':
      case 'mulAddWide': {
        // dot: acc[k] + a[4k]*b[4k] + ... + a[4k+3]*b[4k+3]（i8x16）。mulAddWide: acc[k] + a[2k]*b[2k] + a[2k+1]*b[2k+1]（i16x8）。
        const src = e.name === 'dot' ? 'i8x16' : 'i16x8';
        ret = named('i32x4');
        this.expect(exprs[0], ret);
        this.expect(exprs[1], named(src));
        this.expect(exprs[2], named(src));
        break;
      }
      case 'sum': {
        const t = this.checkExpr(exprs[0]);
        ret = vecOf(t, exprs[0].pos).elem;
        break;
      }
      case 'load':
      case 'store': {
        // Simd.load(xs, i): xs[i] 〜 xs[i + レーン数 - 1] を1つのベクトルとして読む。Simd.store(mut xs, i, v) は書く。
        const at = this.checkExpr(exprs[0]);
        const elem = at.k === 'heap' && !at.str && !at.racy ? at.elem : at.k === 'array' ? at.elem : undefined;
        const lanes = elem && ((elem.k === 'int' && !elem.usize && elem.bits <= 32) || (elem.k === 'float' && elem.bits === 32))
          ? 128 / (elem as { bits: number }).bits : 0;
        if (!lanes) {
          this.err('type-mismatch', `\`Simd.${e.name}\` needs an array, Buffer or fixed-length array of 8, 16 or 32-bit integers or f32 but found ${tyStr(at)}`, exprs[0].pos,
            at.k === 'heap' && at.racy ? 'a RacyBuffer is read element by element; copy the elements you need first' : undefined);
        }
        if (!isPlace(exprs[0])) this.err('type-mismatch', `\`Simd.${e.name}\` needs a variable (or a field / element of one) as the array`, exprs[0].pos, 'store the array in a variable first');
        const it = this.checkExpr(exprs[1]);
        if (it.k !== 'int') this.err('type-mismatch', `the index of \`Simd.${e.name}\` must be an integer but found ${tyStr(it)}`, exprs[1].pos);
        const vt: Ty = { k: 'array', elem: elem!, len: lanes };
        if (e.name === 'store') {
          this.expect(exprs[2], vt);
          this.requireMutable(exprs[0], 'modify');
          ret = VOID;
        } else ret = vt;
        break;
      }
      default:
        ret = VOID;
    }
    const node = e as unknown as { kind: string; fn: string; args: Expr[] };
    node.kind = 'math';
    node.fn = `simd.${e.name}`;
    node.args = exprs;
    return ret;
  }

  inferMethodCall(e: Expr & { kind: 'mcall' }, want?: Ty): Ty {
    if (e.recv.kind === 'ident' && e.recv.name === 'console' && !this.lookup('console')) {
      this.err('ts-feature', '`console` is not available', e.pos, 'use the builtin `print(x)`');
    }
    if (e.recv.kind === 'ident' && e.recv.name === 'Math' && !this.lookup('Math')) return this.inferMath(e);
    if (e.recv.kind === 'ident' && e.recv.name === 'Simd' && !this.lookup('Simd')) return this.inferSimd(e, want);
    if (e.recv.kind === 'ident' && (e.recv.name === 'cpu' || e.recv.name === 'gpu') && e.name === 'dispatch' && !this.lookup(e.recv.name)) {
      return this.inferDispatch(e, e.args, e.recv.name);
    }
    if (e.recv.kind === 'ident' && (e.recv.name === 'cpu' || e.recv.name === 'gpu') && e.name === 'dispatchGroups' && !this.lookup(e.recv.name)) {
      return this.inferDispatch(e, e.args, e.recv.name, false, true);
    }
    if (e.recv.kind === 'ident' && e.recv.name === 'gpu' && e.name === 'submitGroups' && !this.lookup('gpu')) {
      return this.inferDispatch(e, e.args, 'gpu', true, true);
    }
    if (this.isGroupCall(e)) return this.inferGroup(e);
    // `gpu.submit(kernel, count, ...)`: 終わるのを待たずに実行を頼み、券を返す。
    // `gpu.ready(t)`: 券 t の仕事が終わって結果が読めるなら true。`gpu.wait(t)`: 終わるまで待つ。
    if (e.recv.kind === 'ident' && e.recv.name === 'gpu' && !this.lookup('gpu') && (e.name === 'submit' || e.name === 'ready' || e.name === 'wait')) {
      if (e.name === 'submit') return this.inferDispatch(e, e.args, 'gpu', true);
      this.noKernel(`\`gpu.${e.name}\``, e.pos);
      if (e.args.length !== 1) {
        this.err('arg-count', `\`gpu.${e.name}\` takes exactly one argument (the ticket returned by \`gpu.submit\`)`, e.pos);
      }
      if (e.args[0].mode !== 'none') this.err('arg-mode', 'the ticket is passed by value', e.args[0].expr.pos);
      this.expect(e.args[0].expr, U64);
      const node = e as unknown as { kind: string; callee: string; args: Arg[] };
      node.kind = 'call';
      node.callee = e.name === 'ready' ? '__gpu_ready' : '__gpu_wait';
      return e.name === 'ready' ? BOOL : VOID;
    }
    // `gpu.flags(f)`: このスレッドの、このあとの `gpu.submit` に付ける印を f にし、前の値を返す
    // （1 = 検算をしない、2 = 急ぎの列、4 = 提出を見送る。README の gpu.submit を参照）。
    // `gpu.commit()`: 見送ってためてある提出を、いま GPU に渡す。どちらも Metal 以外では何もしない。
    if (e.recv.kind === 'ident' && e.recv.name === 'gpu' && !this.lookup('gpu') && (e.name === 'flags' || e.name === 'commit')) {
      this.noKernel(`\`gpu.${e.name}\``, e.pos);
      const node = e as unknown as { kind: string; callee: string; args: Arg[] };
      if (e.name === 'commit') {
        if (e.args.length !== 0) this.err('arg-count', '`gpu.commit` takes no arguments', e.pos);
        node.kind = 'call';
        node.callee = '__gpu_commit';
        return VOID;
      }
      if (e.args.length !== 1) {
        this.err('arg-count', '`gpu.flags` takes exactly one argument (the flags for the following `gpu.submit` calls)', e.pos,
          'pass the flags as a u32: 1 = no checksum, 2 = urgent queue, 4 = hold the commit (0 = the default behaviour)');
      }
      if (e.args[0].mode !== 'none') this.err('arg-mode', 'the flags are passed by value', e.args[0].expr.pos);
      this.expect(e.args[0].expr, U32);
      node.kind = 'call';
      node.callee = '__gpu_flags';
      return U32;
    }
    // `gpu.untracked(b)`: Buffer b を、gpu.submit の経路で Metal の書き込みの追跡から外す（README「追跡しない Buffer」）。
    // Metal 以外では何もしない。
    if (e.recv.kind === 'ident' && e.recv.name === 'gpu' && !this.lookup('gpu') && e.name === 'untracked') {
      this.noKernel('`gpu.untracked`', e.pos);
      if (e.args.length !== 1) {
        this.err('arg-count', '`gpu.untracked` takes exactly one argument (a Buffer or RacyBuffer)', e.pos);
      } else {
        if (e.args[0].mode !== 'none') this.err('arg-mode', 'the buffer is only read; pass it without `mut`', e.args[0].expr.pos);
        const t = this.checkExpr(e.args[0].expr);
        if (t.k !== 'heap' || !(t.buffer || t.racy)) {
          this.err('type-mismatch', `\`gpu.untracked\` takes a Buffer or RacyBuffer, not ${tyStr(t)}`, e.args[0].expr.pos);
        }
      }
      const node = e as unknown as { kind: string; callee: string; args: Arg[] };
      node.kind = 'call';
      node.callee = '__gpu_untracked';
      return VOID;
    }
    // `gpu.loadKernel(wgsl)` / `gpu.runKernel(k, count, ...)`: 実行時に読み込む kernel（src/gpuload.ts）。
    if (isGpuLoadCall(e, (n) => this.lookup(n))) return inferGpuLoad(this, e);
    if (this.isThreadCall(e)) return this.inferThread(e);
    // `Shape.Circle(1.5)`: enum の値を作る。
    if (e.recv.kind === 'ident' && !this.lookup(e.recv.name) && this.isEnumName(e.recv.name)) {
      return this.enumValue(e, e.recv.name, e.name, e.args, want);
    }
    // kernel の中で使えるメソッドは、RacyBuffer のアトミック演算だけ。
    const rs = e.recv.kind === 'ident' ? this.lookup(e.recv.name) : undefined;
    const atomic = rs?.ty.k === 'heap' && rs.ty.racy === true && ['load', 'store', 'fetchAdd', 'compareExchange'].includes(e.name);
    if (!atomic) this.noKernel('a method call', e.pos);
    const rt = this.checkExpr(e.recv);
    if (rt.k === 'heap') return this.inferArrayMethod(e, rt);
    if (rt.k === 'array') {
      this.err('type-mismatch', `fixed-length array ${tyStr(rt)} has no method '${e.name}'`, e.pos,
        `only growable arrays have methods; declare it as \`${tyStr(rt.elem)}[]\` to use \`.push()\` and friends`);
    }
    if (rt.k !== 'struct') {
      this.err('type-mismatch', `${tyStr(rt)} has no method '${e.name}'`, e.pos);
    }
    const info = this.structs.get(rt.name)!;
    const sig = info.methods.get(e.name);
    if (!sig) {
      const names = [...info.methods.keys()];
      const hint = e.name === 'clone'
        ? `only arrays have a built-in \`.clone()\`; add a method to '${rt.name}' that builds a copy (clone each array field)`
        : names.length > 0 ? `methods: ${names.join(', ')}` : `'${rt.name}' has no methods`;
      this.err('unknown-method', `'${rt.name}' has no method '${e.name}'`, e.pos, hint);
    }
    if (sig.decl.isPrivate && this.curFn.cls !== rt.name) {
      this.err('private', `method '${e.name}' of '${rt.name}' is private`, e.pos);
    }
    e.key = sig.key;
    this.checkArgs(sig, e.args, e.pos, e.recv);
    this.pending.push({ caller: this.curFn, recv: e.recv, args: e.args, sig, pos: e.pos, frozen: [...this.frozen] });
    return sig.ret;
  }

  /** 可変長配列の組み込みメソッド。 */
  inferArrayMethod(e: Expr & { kind: 'mcall' }, rt: Ty & { k: 'heap' }): Ty {
    const argc = (n: number) => {
      if (e.args.length !== n) this.err('arg-count', `'${e.name}' takes ${n} argument(s) but ${e.args.length} were given`, e.pos);
      for (const a of e.args) if (a.mode !== 'none') this.err('arg-mode', 'array elements are passed by value', a.expr.pos);
    };
    const mutates = () => {
      if (isPlace(e.recv)) this.requireMutable(e.recv, 'modify');
    };
    e.key = `Array.${e.name}`;
    if (rt.str && e.name !== 'clone') {
      this.err('unknown-method', `a string has no method '${e.name}'`, e.pos,
        'strings are immutable; build a new one with `+` or a template string. Available: `.length`, `s[i]` (a byte), `.clone()`, `==`');
    }
    if (rt.buffer && (e.name === 'push' || e.name === 'pop' || e.name === 'clear')) {
      this.err('unknown-method', `a Buffer has a fixed length, so it has no method '${e.name}'`, e.pos,
        e.name === 'clear' ? 'to set every element to 0, write `b.fill(0)`'
          : 'use a growable array (`T[]`) to build the data, or allocate the final size with `new Buffer<T>(n)`');
    }
    if (rt.racy && ['load', 'store', 'fetchAdd', 'compareExchange'].includes(e.name)) {
      // RacyBuffer のアトミック演算（順序は seq_cst）。書き換えるものも、読み取り借用から呼べる。
      const n = ({ load: 1, store: 2, fetchAdd: 2, compareExchange: 3 } as Record<string, number>)[e.name];
      argc(n);
      const before = new Set(this.moved);
      const it = this.checkExpr(e.args[0].expr);
      if (it.k !== 'int') this.err('type-mismatch', `the index must be an integer but found ${tyStr(it)}`, e.args[0].expr.pos);
      const idx = e.args[0].expr;
      if (idx.kind === 'int' && idx.value < 0n) this.err('index-range', `index ${idx.value} is negative`, idx.pos);
      for (const a of e.args.slice(1)) this.expect(a.expr, rt.elem);
      this.notMovedSince(e.recv, before);
      if ((e.name === 'fetchAdd' || e.name === 'compareExchange') && rt.elem.k === 'float') {
        this.err('type-mismatch', `\`${e.name}\` needs integer elements, but this RacyBuffer holds ${tyStr(rt.elem)}`, e.pos,
          'for f32, keep the bits in a RacyBuffer<u32> (NaN and -0 make bit equality and `==` disagree)');
      }
      if (this.inKernel) {
        // GPU のアトミック演算は 32 ビットの整数だけ（MSL の atomic_int / atomic_uint、WGSL の atomic<i32> / atomic<u32>）。
        const int32 = rt.elem.k === 'int' && rt.elem.bits === 32;
        const word = int32 || (rt.elem.k === 'float' && rt.elem.bits === 32);
        if (!(e.name === 'load' || e.name === 'store' ? word : int32)) {
          this.err('kernel-type', `\`${e.name}\` on a RacyBuffer<${tyStr(rt.elem)}> is not available in a kernel function`, e.pos,
            'in a kernel, atomic operations need i32 or u32 elements (load / store also take f32)');
        }
        if (e.name !== 'load') this.requireMutable(e.recv, 'modify');
      }
      e.key = `Racy.${e.name}`;
      return e.name === 'store' ? VOID : rt.elem;
    }
    if (e.name === 'fill') {
      // 一括の書き込み: fill(v) は全体、fill(v, start, end) は [start, end) を v にする（長さは変わらない）。
      // RacyBuffer は要素の書き込みと同じく、読み取り借用からでも書ける。
      if (e.args.length !== 1 && e.args.length !== 3) {
        this.err('arg-count', `'fill' takes 1 or 3 arguments (value, or value, start, end) but ${e.args.length} were given`, e.pos,
          'e.g. `xs.fill(0)` or `xs.fill(0, start, end)` (end is exclusive)');
      }
      for (const a of e.args) if (a.mode !== 'none') this.err('arg-mode', 'array elements are passed by value', a.expr.pos);
      if (rt.str) this.err('unknown-method', 'a string has no method \'fill\'', e.pos);
      if (rt.elem.k !== 'int' && rt.elem.k !== 'float' && rt.elem.k !== 'bool') {
        this.err('not-implemented', `\`fill\` of an array of '${tyStr(rt.elem)}' is not implemented yet`, e.pos,
          '`fill` works on arrays of numbers and bools; for other elements, assign in a loop');
      }
      const before = new Set(this.moved);
      this.expect(e.args[0].expr, rt.elem);
      for (const a of e.args.slice(1)) {
        const it = this.checkExpr(a.expr);
        if (it.k !== 'int') this.err('type-mismatch', `the range of \`fill\` must be integers but found ${tyStr(it)}`, a.expr.pos);
      }
      this.notMovedSince(e.recv, before);
      if (!rt.racy) mutates();
      e.key = 'Array.fill';
      return VOID;
    }
    switch (e.name) {
      case 'push': {
        argc(1);
        mutates();
        const before = new Set(this.moved);
        this.expect(e.args[0].expr, rt.elem);
        this.consume(e.args[0].expr); // 要素は配列に移動する
        this.notMovedSince(e.recv, before);
        return VOID;
      }
      case 'pop':
        argc(0);
        mutates();
        return rt.elem;
      case 'clear':
        argc(0);
        mutates();
        return VOID;
      case 'clone':
        argc(0);
        if (!this.isCopy(rt.elem)) {
          this.err('not-implemented', `\`.clone()\` of an array of '${tyStr(rt.elem)}' is not implemented yet`, e.pos,
            'build the copy with a loop: push a `.clone()` of each element');
        }
        return rt;
    }
    const closures = ['map', 'filter', 'forEach', 'reduce', 'find', 'some', 'every', 'sort', 'findIndex', 'flatMap'];
    const hint = closures.includes(e.name)
      ? 'function values are not implemented yet; write the loop with `for (const x of xs)`'
      : rt.racy ? 'available: load, store, fetchAdd, compareExchange, fill, clone, and the `length` property'
      : rt.buffer ? 'available: fill, clone, and the `length` property'
      : 'available: push, pop, clear, fill, clone, and the `length` property';
    this.err('unknown-method', `arrays have no method '${e.name}'`, e.pos, hint);
  }

  /** 引数の個数・型・渡し方と、排他規則を検査する。recv はメソッド呼び出しの受け手。 */
  checkArgs(sig: FnSig, args: Arg[], pos: Pos, recv?: Expr, kernelCall = false): void {
    // kernelCall: dispatch / gpu.submit の kernel の引数。kernel は Buffer の領域を差し替えられないので、
    // `mut RacyBuffer` には読み取り借用（や凍結中）の RacyBuffer も渡せる（同じ呼び出しの他の引数と重なってもよい）。
    const racyMut = (p: { mode: Mode; ty: Ty }) => kernelCall && p.mode === 'mut' && p.ty.k === 'heap' && p.ty.racy === true;
    const what = sig.decl.isCtor ? `constructor of '${sig.decl.cls}'` : `'${sig.name}'`;
    if (args.length !== sig.params.length) {
      this.err('arg-count', `${what} takes ${sig.params.length} argument(s) but ${args.length} were given`, pos);
    }
    const beforeArgs = new Set(this.moved);

    for (let i = 0; i < sig.params.length; i++) {
      const p = sig.params[i];
      const a = args[i];
      this.expect(a.expr, p.ty);
      if (a.mode !== p.mode) {
        if (p.mode === 'none') {
          this.err('arg-mode', `parameter '${p.name}' is a read-only borrow; remove \`${a.mode}\``, a.expr.pos);
        }
        this.err('arg-mode', `parameter '${p.name}' is declared \`${p.mode}\``, a.expr.pos,
          `write \`${p.mode} ...\` at the call site so the ${p.mode === 'mut' ? 'mutation' : 'move'} is visible`);
      }
      if (p.mode === 'mut') {
        if (!isPlace(a.expr)) this.err('not-assignable', '`mut` needs a variable, field or element', a.expr.pos);
        if (!racyMut(p)) this.requireMutable(a.expr, 'mutably borrow');
      }
      if (p.mode === 'own') this.consume(a.expr);
    }

    // 借用で渡す引数と受け手は、同じ呼び出しの他の引数によって移動されてはいけない。
    sig.params.forEach((p, i) => {
      if (passByPtr(p.mode, p.ty) && p.mode !== 'own') this.notMovedSince(args[i].expr, beforeArgs);
    });
    if (recv) this.notMovedSince(recv, beforeArgs);

    // 排他規則: 可変借用している変数を、同じ呼び出しの他のポインタ渡し引数（受け手を含む）に使えない。
    const recvRoot = recv ? rootSym(recv) : undefined;
    for (let i = 0; i < sig.params.length; i++) {
      if (sig.params[i].mode !== 'mut' || racyMut(sig.params[i])) continue;
      const root = rootSym(args[i].expr);
      if (!root) continue;
      if (root === recvRoot) {
        this.err('exclusive-borrow', `'${root.name}' is mutably borrowed and is also the receiver of this call`,
          args[i].expr.pos, 'copy the value into a local variable before the call');
      }
      for (let j = 0; j < sig.params.length; j++) {
        if (j === i || !passByPtr(sig.params[j].mode, sig.params[j].ty) || racyMut(sig.params[j])) continue;
        if (rootSym(args[j].expr) === root) {
          this.err('exclusive-borrow', `'${root.name}' is mutably borrowed and used by another argument of the same call`,
            args[j].expr.pos, 'copy the other value into a `const` before the call');
        }
      }
    }
  }

  // ---- グループ（threadgroup / workgroup）----

  /** `group.名前(...)` の呼び出しか（group という名前の変数があれば違う）。 */
  isGroupCall(e: Expr, name?: string): e is Expr & { kind: 'mcall' } {
    return e.kind === 'mcall' && e.recv.kind === 'ident' && e.recv.name === 'group' && !this.lookup('group') && (!name || e.name === name);
  }

  /**
   * `group.local()` / `group.id()` / `group.size()` / `group.count()` / `group.barrier()`。
   * kernel の中でだけ使え、使った kernel は dispatchGroups でだけ実行できる。式は math の節に書き換える。
   */
  inferGroup(e: Expr & { kind: 'mcall' }): Ty {
    const names = ['local', 'id', 'size', 'count', 'barrier', 'simdLane', ...GROUP_COLLECTIVE];
    if (!this.inKernel) {
      this.err('kernel-restriction', `\`group.${e.name}\` can only be used in a kernel function`, e.pos,
        'group features describe the threads of a GPU group; run the kernel with `dispatchGroups(k, groups, size, ...)`');
    }
    if (e.name === 'shared') {
      this.err('kernel-restriction', '`group.shared()` can only initialize a `let` at the top level of a kernel', e.pos,
        'e.g. `let tile: f32[256] = group.shared();`');
    }
    if (!names.includes(e.name)) {
      this.err('unknown-method', `\`group\` has no function '${e.name}'`, e.pos, `available: ${names.join(', ')}, shared`);
    }
    this.curFn.usesGroup = true;
    if (GROUP_COLLECTIVE.has(e.name)) return this.inferCollective(e);
    if (e.args.length !== 0) this.err('arg-count', `\`group.${e.name}\` takes no arguments`, e.pos);
    if (e.name === 'barrier') this.curFn.usesBarrier = true;
    const node = e as unknown as { kind: string; fn: string; args: Expr[] };
    node.kind = 'math';
    node.fn = `group.${e.name}`;
    node.args = [];
    return e.name === 'barrier' ? VOID : U32;
  }

  /** let / const の初期化式として書かれた simdgroup の集計（checkStmt の let が印を付ける）。 */
  collectiveInit: Expr | null = null;

  /**
   * simdgroup（グループの中の連続した 32 スレッド）の集計: `group.simdSum(x)` / `simdMin(x)` / `simdMax(x)`（x は i32 / u32 / f32）と
   * `group.simdShuffle(x, lane)`（同じ simdgroup の lane 番目のスレッドの x）。Metal の simd_sum などになる（WebGPU は対応しない）。
   * simdgroup の全スレッドの値を使うので、barrier と同じく、グループの全スレッドが通る場所にだけ置ける（checkGroupBody）。
   * CPU では barrier と同じく区切りになる（codegen.ts の groupStmt）。
   */
  inferCollective(e: Expr & { kind: 'mcall' }): Ty {
    if (this.collectiveInit !== e) {
      this.err('kernel-restriction', `\`group.${e.name}\` can only be the initial value of a \`let\` or \`const\``, e.pos,
        `e.g. \`const s = group.${e.name}(${e.name === 'simdShuffle' ? 'x, 0' : 'x'});\``);
    }
    this.collectiveInit = null;
    this.curFn.usesBarrier = true;
    const want = e.name === 'simdShuffle' ? 2 : 1;
    if (e.args.length !== want) {
      this.err('arg-count', `\`group.${e.name}\` takes ${want === 2 ? 'two arguments (the value and the lane)' : 'one argument'}`, e.pos);
    }
    for (const a of e.args) if (a.mode !== 'none') this.err('arg-mode', 'the value is passed by value', a.expr.pos);
    const t = this.checkExpr(e.args[0].expr);
    const ok = (t.k === 'int' && t.bits === 32) || (t.k === 'float' && t.bits === 32);
    if (!ok) this.err('kernel-type', `\`group.${e.name}\` works on i32, u32 or f32, not ${tyStr(t)}`, e.args[0].expr.pos);
    if (want === 2) this.expect(e.args[1].expr, U32);
    const node = e as unknown as { kind: string; fn: string; args: Expr[] };
    node.kind = 'math';
    node.fn = `group.${e.name}`;
    node.args = e.args.map((a) => a.expr);
    return t;
  }

  /** `let tile: f32[256] = group.shared();`: グループの全スレッドが共有する配列。グループの始めに 0 になる。 */
  checkShared(s: Stmt & { kind: 'let' }): void {
    const init = s.init as Expr & { kind: 'mcall' };
    if (!this.inKernel) {
      this.err('kernel-restriction', '`group.shared()` can only be used in a kernel function', s.pos);
    }
    if (init.args.length !== 0) this.err('arg-count', '`group.shared` takes no arguments; the size comes from the type', init.pos);
    if (s.isConst) this.err('kernel-restriction', 'a shared array is written by the threads, so declare it with `let`', s.pos);
    if (!s.tyAnn) {
      this.err('needs-type', 'a shared array needs its type', s.pos, `e.g. \`let ${s.name}: f32[256] = group.shared();\``);
    }
    const ty = this.resolve(s.tyAnn);
    const okElem = ty.k === 'array' && ((ty.elem.k === 'int' && ty.elem.bits === 32) || (ty.elem.k === 'float' && ty.elem.bits === 32));
    if (!okElem) {
      this.err('kernel-type', `a shared array must be a fixed-length array of i32, u32 or f32, not ${tyStr(ty)}`, s.pos,
        `e.g. \`let ${s.name}: f32[256] = group.shared();\``);
    }
    const t = ty as Ty & { k: 'array' };
    // WebGPU が必ず用意する大きさ（maxComputeWorkgroupStorageSize の既定値）。
    this.sharedBytes = (this.sharedFn === this.curFn ? this.sharedBytes : 0) + t.len * 4;
    this.sharedFn = this.curFn;
    if (this.sharedBytes > 16384) {
      this.err('kernel-restriction', `the shared arrays of kernel '${this.curFn.name}' need ${this.sharedBytes} bytes, more than 16384`, s.pos,
        'WebGPU guarantees 16 KiB of group shared memory; use smaller arrays');
    }
    this.curFn.usesGroup = true;
    const node = init as unknown as { kind: string; fn: string; args: Expr[]; ty: Ty };
    node.kind = 'math';
    node.fn = 'group.shared';
    node.args = [];
    node.ty = ty;
    s.sym = this.declare(s.name, ty, true, s.pos);
    s.sym.shared = true;
  }

  sharedBytes = 0;
  sharedFn: FnDecl | null = null;

  /**
   * group.* を使う kernel の本体の形を確かめる（本体の型検査のあとで呼ぶ）。
   *   - 共有配列の宣言は本体の直下だけ（グループの始めに1回だけ 0 にする）。
   *   - group.barrier() は、グループの全スレッドが同じ回数だけ通る場所にだけ置ける。GPU では、通らない
   *     スレッドがあると止まりうる（WGSL はコンパイルを断る）。そこで「グループの全スレッドで同じ値になる式」
   *     （一様な式）を控えめに求め、barrier を囲む if / while / for の条件はそれに限る。
   *   - barrier を含むループの中の break / continue と、barrier を使う kernel の return は使えない。
   * CPU での実行（codegen.ts の groupFn）は、この形を前提に、barrier の間の文を全スレッドについて順に実行し、
   * barrier を含む文の条件はスレッド 0 の値で決める。
   */
  checkGroupBody(d: FnDecl): void {
    const err = (pos: Pos, msg: string, hint: string) => this.errors.push(new CompileError('kernel-restriction', msg, pos, hint));
    const isShared = (s: Stmt) => s.kind === 'let' && s.init?.kind === 'math' && s.init.fn === 'group.shared';
    const kids = (s: Stmt): (Stmt | undefined)[] => {
      switch (s.kind) {
        case 'block':
          return s.body;
        case 'if':
          return [s.then, s.else];
        case 'while':
          return [s.body];
        case 'for':
          return [s.init, s.step, s.body];
        case 'forOf':
          return [s.body];
        default:
          return [];
      }
    };

    // 共有配列は本体の直下でだけ宣言できる。
    const sharedAt = (s: Stmt | undefined, top: boolean): void => {
      if (!s) return;
      if (isShared(s) && !top) {
        err(s.pos, 'a shared array must be declared at the top level of the kernel body',
          'move `let ... = group.shared();` to the start of the kernel (it is zeroed once when the group starts)');
      }
      for (const k of kids(s)) sharedAt(k, false);
    };
    for (const s of d.body) sharedAt(s, true);
    if (!d.usesBarrier) return;

    // 一様でない変数を集める（増える一方なので、変わらなくなるまで繰り返す）。
    // 添字の引数・group.local()・Buffer と共有配列の要素は一様でない。値の引数・group.id() / size() / count()・
    // Buffer の長さ・定数は一様。一様でない条件の下での代入と、一様でない値の代入で、変数は一様でなくなる。
    const idx = d.params[0]?.sym;
    const nonUni = new Set<number>(idx ? [idx.id] : []);
    const uni = (e: Expr): boolean => {
      switch (e.kind) {
        case 'int':
        case 'float':
        case 'bool':
          return true;
        case 'ident':
          return !!e.sym && !nonUni.has(e.sym.id) && !e.sym.shared;
        case 'unary':
        case 'cast':
          return uni(e.expr);
        case 'binary':
          return uni(e.left) && uni(e.right);
        case 'cond':
          return uni(e.cond) && uni(e.then) && uni(e.else);
        case 'field':
          return e.base.ty!.k === 'heap' || uni(e.base);
        case 'index':
          return e.base.ty!.k === 'array' && uni(e.base) && uni(e.index);
        case 'call':
          return e.args.every((a) => a.expr.ty!.k !== 'heap' && uni(a.expr));
        case 'math':
          if (e.fn.startsWith('group.')) return e.fn === 'group.id' || e.fn === 'group.size' || e.fn === 'group.count';
          return e.args.every(uni);
        case 'object':
          return e.fields.every((f) => uni(f.value));
        case 'arrayLit':
          return e.elems.every(uni);
        default:
          return false;
      }
    };
    // 代入先の場所が一様か（`a[j] = v` の j が一様でなければ、a は一様でなくなる）。
    const placeUni = (e: Expr): boolean => {
      if (e.kind === 'index') return uni(e.index) && placeUni(e.base);
      if (e.kind === 'field') return placeUni(e.base);
      return true;
    };
    let changed = true;
    const mark = (sym: Sym | undefined) => {
      if (sym && !nonUni.has(sym.id)) {
        nonUni.add(sym.id);
        changed = true;
      }
    };
    // ループ: 条件が一様でも、一様でない条件の下の break / continue があれば、回る回数がスレッドごとに違う。
    const loop = (cond: () => boolean, parts: (Stmt | undefined)[], ctrl: boolean): void => {
      const c = ctrl && cond();
      const exits = parts.map((p) => walk(p, c)).some((x) => x);
      if (exits && c) for (const p of parts) walk(p, false);
    };
    // ctrl: ここまでの制御が一様か。戻り値: いまのループを、一様でない条件の下で抜けうる（break / continue）か。
    const walk = (s: Stmt | undefined, ctrl: boolean): boolean => {
      if (!s) return false;
      switch (s.kind) {
        case 'let':
          if (!isShared(s) && (!ctrl || (s.init !== undefined && !uni(s.init)))) mark(s.sym);
          return false;
        case 'assign': {
          if (s.target.kind === 'index' && s.target.base.ty!.k === 'heap') return false;
          const root = rootSym(s.target);
          if (root?.shared) return false;
          if (!ctrl || !uni(s.value) || !placeUni(s.target)) mark(root);
          return false;
        }
        case 'block':
          return s.body.map((x) => walk(x, ctrl)).some((x) => x);
        case 'if': {
          const c = ctrl && uni(s.cond);
          const a = walk(s.then, c);
          const b = walk(s.else, c);
          return a || b;
        }
        case 'while':
          loop(() => uni(s.cond), [s.body], ctrl);
          return false;
        case 'for':
          walk(s.init, ctrl);
          loop(() => !s.cond || uni(s.cond), [s.body, s.step], ctrl);
          return false;
        case 'forOf':
          // 回る回数は長さで決まる（Buffer の長さも一様）。要素は、一様な固定長配列からだけ一様。
          if (!ctrl || s.iter.ty!.k === 'heap' || !uni(s.iter)) mark(s.sym);
          loop(() => true, [s.body], ctrl);
          return false;
        case 'break':
        case 'continue':
        case 'return':
          return !ctrl;
        default:
          return false;
      }
    };
    while (changed) {
      changed = false;
      for (const s of d.body) walk(s, true);
    }

    // barrier と simdgroup の集計（どちらもグループの全スレッドがそろって通る）。
    const hasBarrier = (s: Stmt | undefined): boolean => {
      if (!s) return false;
      if (s.kind === 'expr') return s.expr.kind === 'math' && s.expr.fn === 'group.barrier';
      if (isCollectiveLet(s)) return true;
      return kids(s).some(hasBarrier);
    };
    const uniformHint = 'conditions around a barrier may only use constants, value parameters, `group.id()`, `group.size()`, `group.count()`, ' +
      'buffer lengths and variables computed only from those; move the barrier out of the `if` or loop';
    // inLoop: いちばん内側のループが barrier を含むか（その中の break / continue は使えない）。
    const place = (s: Stmt | undefined, ctrl: boolean, inLoop: boolean): void => {
      if (!s) return;
      switch (s.kind) {
        case 'expr':
          if (hasBarrier(s) && !ctrl) {
            err(s.pos, '`group.barrier()` must be reached by every thread of the group, but it is under a condition or loop that can differ between threads', uniformHint);
          }
          return;
        case 'let':
          if (isCollectiveLet(s) && !ctrl) {
            const fn = (s.init as Expr & { kind: 'math' }).fn;
            err(s.pos, `\`${fn}\` must be reached by every thread of the group, but it is under a condition or loop that can differ between threads`,
              uniformHint.replace('around a barrier', `around \`${fn}\``).replace('the barrier', 'it'));
          }
          return;
        case 'return':
          err(s.pos, '`return` cannot be used in a kernel that uses `group.barrier()` or `group.simd*`',
            'every thread must reach each barrier; put the rest of the body in an `if` instead');
          return;
        case 'break':
        case 'continue':
          if (inLoop) {
            err(s.pos, `\`${s.kind}\` cannot be used in a loop that contains \`group.barrier()\` or \`group.simd*\``,
              'every thread must run the loop the same number of times; express the end in the loop condition');
          }
          return;
        case 'block':
          for (const x of s.body) place(x, ctrl, inLoop);
          return;
        case 'if': {
          const c = ctrl && uni(s.cond);
          place(s.then, c, inLoop);
          place(s.else, c, inLoop);
          return;
        }
        case 'while':
          place(s.body, ctrl && uni(s.cond), hasBarrier(s.body));
          return;
        case 'for':
          place(s.init, ctrl, inLoop);
          place(s.body, ctrl && (!s.cond || uni(s.cond)), hasBarrier(s.body));
          return;
        case 'forOf':
          if (hasBarrier(s.body)) {
            err(s.pos, '`group.barrier()` and `group.simd*` cannot be inside a for-of loop',
              'every thread of the group must reach the barrier; use an index loop `for (let k = 0; k < n; k++)`');
            return;
          }
          place(s.body, ctrl, false);
          return;
        default:
          return;
      }
    };
    for (const s of d.body) place(s, true, false);
  }

  /** kernel どうしの再帰を禁止する（GPU のシェーダは再帰できない）。 */
  finishKernels(): void {
    for (const c of this.kernelUses) {
      if (!c.callee.usesGroup) continue;
      this.errors.push(new CompileError('kernel-restriction', `kernel '${c.callee.name}' uses \`group\`, so it cannot be called like a function`, c.pos,
        `group features (group.*, shared arrays) belong to the kernel that is run with \`dispatchGroups(${c.callee.name}, groups, size, ...)\`; move the shared code into a helper kernel without them`));
    }
    for (const g of this.groupDispatches) {
      if (g.groups || !g.kernel.usesGroup) continue;
      this.errors.push(new CompileError('kernel-restriction', `kernel '${g.kernel.name}' uses \`group\`, so it must be run in groups`, g.pos,
        `write \`dispatchGroups(${g.kernel.name}, groups, size, ...)\` (or \`gpu.dispatchGroups\` / \`cpu.dispatchGroups\` / \`gpu.submitGroups\`)`));
    }
    const edges = new Map<FnDecl, FnDecl[]>();
    for (const c of this.kernelCalls) edges.set(c.caller, [...(edges.get(c.caller) ?? []), c.callee]);
    const reaches = (from: FnDecl, target: FnDecl, seen: Set<FnDecl>): boolean => {
      for (const next of edges.get(from) ?? []) {
        if (next === target) return true;
        if (!seen.has(next)) {
          seen.add(next);
          if (reaches(next, target, seen)) return true;
        }
      }
      return false;
    };
    const reported = new Set<FnDecl>();
    for (const c of this.kernelCalls) {
      if (reported.has(c.caller) || !reaches(c.caller, c.caller, new Set())) continue;
      reported.add(c.caller);
      this.errors.push(new CompileError('kernel-restriction', `kernel '${c.caller.name}' is recursive`, c.pos,
        'GPU code cannot recurse; rewrite it as a loop'));
    }
  }

  /**
   * どのメソッドが this を書き換えるかを決め、書き換えるメソッドの呼び出しを検査する。
   * 「自分で書き換える」か「this に対して、書き換えるメソッドを呼ぶ」なら書き換えるメソッド。
   */
  finishMethods(decls: FnDecl[]): void {
    for (const d of decls) d.mutatesThis = this.directMut.has(d);
    for (let changed = true; changed; ) {
      changed = false;
      for (const c of this.pending) {
        if (c.caller.mutatesThis || !c.sig.decl.mutatesThis) continue;
        if (c.caller.thisSym && rootSym(c.recv) === c.caller.thisSym) {
          c.caller.mutatesThis = true;
          changed = true;
        }
      }
    }

    for (const c of this.pending) {
      if (!c.sig.decl.mutatesThis) continue;
      try {
        const root = rootSym(c.recv);
        const at = placePath(c.recv);
        const busy = at && c.frozen.find((f) => f.sym === at.sym && overlaps(f.path, at.path));
        if (busy) {
          this.err('exclusive-borrow', `method '${c.sig.name}' may modify '${busy.text}' while ${frozenBy(busy)}`, c.pos,
            'a method that modifies its object could change the array being iterated; update a local variable in the loop and store it after the loop, or use an index loop');
        }
        if (isPlace(c.recv) && root && !root.mutable) {
          this.err('immutable', `method '${c.sig.name}' modifies its object, but '${root.name}' is not mutable`, c.pos,
            root.param ? `declare the parameter as \`${root.name}: mut ${tyStr(root.ty)}\` and pass \`mut ...\` at the call site`
              : `declare '${root.name}' with \`let\` instead of \`const\``);
        }
        c.sig.params.forEach((p, j) => {
          if (root && passByPtr(p.mode, p.ty) && rootSym(c.args[j].expr) === root) {
            this.err('exclusive-borrow', `method '${c.sig.name}' modifies '${root.name}', which is also passed as an argument`,
              c.args[j].expr.pos, 'copy the argument into a `const` before the call');
          }
        });
      } catch (e) {
        this.note(e);
      }
    }
  }
}

export function check(mod: Module, ptrBits: 32 | 64, gpu = false): Program {
  const c = new Checker(ptrBits, gpu);
  const decls = c.collect(mod);
  for (const d of decls) {
    try {
      c.checkFn(d);
    } catch (e) {
      c.note(e);
    }
  }
  c.finishMethods(decls);
  c.finishKernels();
  if (c.errors.length > 0) throw new CompileErrors(c.errors);
  return { structs: c.structs, fns: c.fns, decls, ptrBits, usesF16: c.usesF16 };
}
