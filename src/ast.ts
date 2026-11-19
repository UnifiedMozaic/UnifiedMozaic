import type { Pos } from './diag.ts';

// ---- 型 ----

export type Ty =
  | { k: 'int'; bits: 8 | 16 | 32 | 64; signed: boolean; usize?: boolean }
  | { k: 'float'; bits: 4 | 8 | 16 | 32 | 64 }
  | { k: 'bool' }
  | { k: 'void' }
  | { k: 'struct'; name: string }
  | { k: 'array'; elem: Ty; len: number }
  // 可変長配列（ヒープに要素を持つ）。`T[]` と `Array<T>` は同じ型。
  // buffer が真なら `Buffer<T>`: 長さ固定で、CPU と GPU が共有できる要素だけを持つ。
  // racy: 同期なしで読み書きしてよい Buffer（RacyBuffer）。kernel はどの要素にも書ける。
  | { k: 'heap'; elem: Ty; buffer?: boolean; str?: boolean; racy?: boolean };

/** 内部の名前 → エラーメッセージで見せる名前（`T | null` を表す struct など）。 */
export const displayNames = new Map<string, string>();

export function tyStr(t: Ty): string {
  switch (t.k) {
    case 'int':
      return t.usize ? 'usize' : `${t.signed ? 'i' : 'u'}${t.bits}`;
    case 'float':
      return `f${t.bits}`;
    case 'bool':
      return 'bool';
    case 'void':
      return 'void';
    case 'struct':
      return displayNames.get(t.name) ?? t.name;
    case 'array':
      return `${tyStr(t.elem)}[${t.len}]`;
    case 'heap':
      return t.str ? 'string' : t.racy ? `RacyBuffer<${tyStr(t.elem)}>` : t.buffer ? `Buffer<${tyStr(t.elem)}>` : `${tyStr(t.elem)}[]`;
  }
}

export const tyEq = (a: Ty, b: Ty) => tyStr(a) === tyStr(b);
export const isAgg = (t: Ty) => t.k === 'struct' || t.k === 'array' || t.k === 'heap';
export const isNumeric = (t: Ty) => t.k === 'int' || t.k === 'float';

// ---- 構文木 ----

export type TypeNode =
  // free: 型引数の置き換えで入った名前。使う側のファイルで import されているかは問わない。
  | { kind: 'name'; name: string; pos: Pos; free?: boolean }
  // lenName: 長さを最上位の定数の名前で書いた場合（`T[N]`）。型検査が値を求めて len に入れる。
  | { kind: 'array'; elem: TypeNode; len: number; pos: Pos; lenName?: string }
  | { kind: 'heap'; elem: TypeNode; pos: Pos; buffer?: boolean; racy?: boolean }
  // `T | null`
  | { kind: 'opt'; inner: TypeNode; pos: Pos }
  // ジェネリックな struct の型引数つきの名前: `Pair<i32, f64>`
  | { kind: 'generic'; name: string; args: TypeNode[]; pos: Pos };

/** 引数の渡し方。none=読み取り借用、mut=可変借用、own=所有権の移動。 */
export type Mode = 'none' | 'mut' | 'own';

/** 型検査が束縛ごとに作る記号。コード生成は id で置き場所を引く。 */
export interface Sym {
  id: number;
  name: string;
  ty: Ty;
  mutable: boolean;
  /** 値を所有しているか（ローカル変数と own 引数）。借用（修飾なし・mut の引数、this）は false。 */
  owned: boolean;
  /** 関数の引数かどうか（エラーのヒントを出し分けるため）。 */
  param?: boolean;
  /** 宣言された場所のループの深さ。ループ内で外側の変数を移動させないための検査に使う。 */
  loopDepth: number;
  /** match の枝が付けた別名。この変数を読む所は、この式（enum の中の値を指す）に置き換わる。 */
  alias?: Expr;
  /**
   * スレッドの持ち手（`const h = thread.spawn(...)`）。ret は関数の戻り値の型、each は spawnEach か。
   * 持ち手は `thread.join(h)` に渡すことしかできない（型は内部の印で、値として扱わない）。
   */
  thread?: { ret: Ty; each: boolean };
}

export interface Arg {
  mode: Mode;
  expr: Expr;
}

export type Expr = { pos: Pos; ty?: Ty; /** コンパイラが enum / match から作った式。enum の中身のフィールドに触れてよい。 */ synth?: boolean; /** 絞り込み（null でないと分かった変数）の書き換えを、この式には行わない。 */ raw?: boolean; /** 名前ではなく、この番号の変数を直接指す（別名の中で、同じ名前の別の変数に取り違えないため）。 */ pin?: number } & (
  // fixed: 最上位の定数を置き換えた値。型はこれに決まっていて、文脈に合わせて変わらない。
  | { kind: 'int'; value: bigint; fixed?: Ty }
  | { kind: 'float'; value: number; fixed?: Ty }
  | { kind: 'str'; value: string }
  | { kind: 'null' }
  | { kind: 'bool'; value: boolean }
  | { kind: 'ident'; name: string; sym?: Sym }
  | { kind: 'unary'; op: string; expr: Expr }
  | { kind: 'binary'; op: string; left: Expr; right: Expr }
  | { kind: 'call'; callee: string; args: Arg[] }
  | { kind: 'mcall'; recv: Expr; name: string; args: Arg[]; key?: string }
  | { kind: 'new'; cls: string; args: Arg[]; typeArgs?: TypeNode[] }
  | { kind: 'newArray'; elemNode: TypeNode; args: Arg[]; buffer?: boolean; racy?: boolean }
  // kernel を 0 から count 未満の各添字について実行する。device は実行先の指定（auto は自動判断）。
  // async が付くのは `gpu.submit(...)`: 終わるのを待たずに戻り、券（u64）を返す。
  | { kind: 'dispatch'; kernel: string; count: Expr; args: Arg[]; device: 'auto' | 'cpu' | 'gpu'; async?: boolean }
  // スレッド。型検査が `thread.xxx(...)` をこれに書き換える。
  //   spawn: fn(args...) を新しいスレッドで実行する。each: args[0] の配列の要素ごとに1本（args[0] は移動）。
  //   join: args[0] は持ち手の変数。cpus: 論理 CPU 数。yield: CPU をほかのスレッドに譲る（値なし）。
  | { kind: 'thread'; op: 'spawn' | 'each' | 'join' | 'cpus' | 'yield'; fn?: string; args: Arg[] }
  | { kind: 'math'; fn: string; args: Expr[] }
  | { kind: 'field'; base: Expr; name: string; index?: number }
  | { kind: 'index'; base: Expr; index: Expr }
  | { kind: 'object'; fields: { name: string; value: Expr; pos: Pos }[] }
  | { kind: 'arrayLit'; elems: Expr[] }
  | { kind: 'cast'; expr: Expr; to: TypeNode }
  // 条件式 `cond ? then : else`。選ばれた側だけを評価する。
  | { kind: 'cond'; cond: Expr; then: Expr; else: Expr }
);

export type Stmt = { pos: Pos } & (
  // init がない `let x: T;` は、ゼロで初期化する（数値・bool・固定長配列・struct に限る）。
  // alias: match の枝で、ヒープを持つ値に付ける名前。変数は作らず、使う所を alias の式に置き換える。
  | { kind: 'let'; name: string; isConst: boolean; tyAnn?: TypeNode; init?: Expr; sym?: Sym; alias?: Expr }
  // initField: コンストラクタでの最初の代入。古い値がないので解放しない。
  | { kind: 'assign'; target: Expr; value: Expr; initField?: boolean }
  | { kind: 'if'; cond: Expr; then: Stmt; else?: Stmt }
  | { kind: 'while'; cond: Expr; body: Stmt }
  | { kind: 'for'; init?: Stmt; cond?: Expr; step?: Stmt; body: Stmt }
  | { kind: 'forOf'; name: string; isConst: boolean; iter: Expr; body: Stmt; sym?: Sym }
  | { kind: 'return'; value?: Expr }
  | { kind: 'break' }
  | { kind: 'continue' }
  | { kind: 'expr'; expr: Expr }
  | { kind: 'block'; body: Stmt[] }
  // 型検査が if の連なりに書き換える。
  | { kind: 'match'; scrut: Expr; arms: MatchArm[] }
);

/** match の1つの枝。wildcard は `_ =>`。 */
export interface MatchArm {
  enumName: string;
  variant: string;
  binds: string[];
  wildcard: boolean;
  body: Stmt;
  pos: Pos;
}

export interface Param {
  name: string;
  mode: Mode;
  tyNode: TypeNode;
  pos: Pos;
  ty?: Ty;
  sym?: Sym;
}

export interface FnDecl {
  kind: 'fn';
  name: string;
  /** どのファイルの宣言か（0 = 最初のファイル、-1 = 標準の関数）。 */
  file?: number;
  /** `export` が付いていて、他のファイルから import できる。 */
  importable?: boolean;
  /** ジェネリック関数の型引数の名前（`function f<T, U>`）。呼び出しごとに具体的な型で複製される。 */
  typeParams?: string[];
  /** 型引数への制約（`T extends Shape`）: 型引数の名前 → interface の名前。 */
  typeBounds?: Record<string, string>;
  exported: boolean;
  params: Param[];
  retNode?: TypeNode;
  ret?: Ty;
  body: Stmt[];
  pos: Pos;
  /** GPUでも実行できる関数。本体に制限があり、整数演算と添字はGPUと同じ意味になる。 */
  isKernel?: boolean;
  /** メソッドとコンストラクタだけが持つ。 */
  cls?: string;
  isCtor?: boolean;
  isPrivate?: boolean;
  thisSym?: Sym;
  /** this を書き換えるメソッドかどうか。型検査が本体から推論する。 */
  mutatesThis?: boolean;
}

export interface ClassDecl {
  kind: 'class';
  name: string;
  /** ジェネリックな class の型引数の名前（`class Stack<T>`）。 */
  typeParams?: string[];
  /** `implements` に書いた interface。 */
  implements?: { name: string; pos: Pos }[];
  file?: number;
  importable?: boolean;
  fields: { name: string; tyNode: TypeNode; init?: Expr; isPrivate: boolean; pos: Pos }[];
  ctor?: FnDecl;
  methods: FnDecl[];
  pos: Pos;
}

export interface StructDecl {
  kind: 'struct';
  name: string;
  /** ジェネリックな struct の型引数の名前（`struct Pair<A, B>`）。使われた型ごとに複製される。 */
  typeParams?: string[];
  file?: number;
  importable?: boolean;
  fields: { name: string; tyNode: TypeNode; pos: Pos; ty?: Ty }[];
  /** enum のとき、その選択肢。値は fields に `選択肢__名前` として並べ、どの選択肢かは `__tag` に持つ。 */
  variants?: { name: string; fields: string[]; pos: Pos }[];
  pos: Pos;
}

/** メソッドの約束。class が `implements` で満たし、ジェネリック関数が `T extends 名前` で求める。 */
export interface InterfaceDecl {
  kind: 'interface';
  name: string;
  file?: number;
  importable?: boolean;
  methods: { name: string; params: Param[]; retNode?: TypeNode; pos: Pos }[];
  pos: Pos;
}

/** 最上位の定数: `const NAME: T = 定数式;`。使う所は、コンパイル時に求めた値に置き換わる。 */
export interface ConstDecl {
  kind: 'const';
  name: string;
  file?: number;
  importable?: boolean;
  tyNode: TypeNode;
  init: Expr;
  pos: Pos;
}

export type Decl = FnDecl | StructDecl | ClassDecl | InterfaceDecl | ConstDecl;

export interface Import {
  names: { name: string; pos: Pos }[];
  path: string;
  pos: Pos;
}

export interface Module {
  decls: Decl[];
  imports?: Import[];
  /** 複数のファイルをまとめたとき: ファイル番号 → そのファイルが import した名前。 */
  importedNames?: Map<number, Set<string>>;
}
