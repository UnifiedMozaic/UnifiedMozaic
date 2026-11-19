// kernel 関数を WGSL（WebGPU のシェーダ言語）に変換する。
//
// CPU 用のコード生成（codegen.ts）と同じ結果になるように、次を明示的に生成する:
//   - 添字は最後の要素に切り詰める（um_clamp）。長さ 0 の Buffer は1要素ぶん確保されている
//   - シフト量はビット幅で割った余り
//   - 整数のゼロ除算・最小値 / -1 は WGSL の仕様で x / 0 = x、x % 0 = 0、最小値 / -1 = 最小値 と決まっており、
//     kernel の CPU 版と同じ
//
// レイアウト: kernel で使える型はすべて 4 バイト単位（i32 / u32 / f32 とその struct・固定長配列）なので、
// LLVM 側の構造体と WGSL の構造体は、同じ並びで隙間なく一致する。

import { tyStr } from './ast.ts';
import type { Expr, FnDecl, Stmt, Sym, Ty } from './ast.ts';
import type { FnSig, Program } from './checker.ts';

/** 1回の dispatch で1つのワークグループが担当する要素数。runtime/host.mjs の定数と揃えること。 */
export const WORKGROUP_SIZE = 64;

export interface WgslBuffer {
  name: string;
  writable: boolean;
  elem: Ty;
}

export interface WgslKernel {
  source: string;
  /** dispatch の引数の順（添字を除く）に並んだ Buffer。binding 番号は 1 から順。 */
  buffers: WgslBuffer[];
}

/** 型の大きさ（4 バイト単位）。 */
export function sizeWords(prog: Program, t: Ty): number {
  if (t.k === 'array') return t.len * sizeWords(prog, t.elem);
  if (t.k === 'struct') return prog.structs.get(t.name)!.fields.reduce((n, f) => n + sizeWords(prog, f.ty), 0);
  return 1;
}

/** Buffer の引数が、シェーダのどのグローバル変数と長さに対応するか。 */
interface BufferRef {
  global: string;
  len: string;
  /** 8 / 16 ビットの要素。WGSL には 8 / 16 ビットの整数がないので、u32 の語に詰めて持ち、読み書きの関数を通す。 */
  small?: { bits: number; signed: boolean };
}

class Emitter {
  prog: Program;
  out: string[] = [];
  structs = new Set<string>();
  /** 生成済みの補助関数（Buffer の対応ごとに特殊化した名前 → 済み）。 */
  fns = new Set<string>();
  fnText: string[] = [];
  names = new Map<number, string>();
  bufs = new Map<number, BufferRef>();
  indent = 1;
  lines: string[] = [];
  tmp = 0;

  constructor(prog: Program) {
    this.prog = prog;
  }

  // ---- 型 ----

  ty(t: Ty): string {
    switch (t.k) {
      case 'bool':
        return 'bool';
      case 'int':
        return t.signed ? 'i32' : 'u32';
      case 'float':
        return 'f32';
      case 'array':
        return `array<${this.ty(t.elem)}, ${t.len}>`;
      case 'struct':
        this.declareStruct(t.name);
        return `S_${t.name}`;
      default:
        throw new Error(`internal: wgsl type ${tyStr(t)}`);
    }
  }

  declareStruct(name: string): void {
    if (this.structs.has(name)) return;
    this.structs.add(name);
    const info = this.prog.structs.get(name)!;
    const fields = info.fields.map((f) => `  f_${f.name}: ${this.ty(f.ty)},`);
    this.out.push(`struct S_${name} {`, ...fields, '}', '');
  }

  // ---- 文 ----

  line(s: string): void {
    this.lines.push('  '.repeat(this.indent) + s);
  }

  name(sym: Sym): string {
    let n = this.names.get(sym.id);
    if (!n) {
      n = `u_${sym.name}_${sym.id}`;
      this.names.set(sym.id, n);
    }
    return n;
  }

  block(body: Stmt[]): void {
    this.indent++;
    for (const s of body) this.stmt(s);
    this.indent--;
  }

  /** 本体を { } で囲んで出す。 */
  body(s: Stmt): void {
    this.block(s.kind === 'block' ? s.body : [s]);
  }

  stmt(s: Stmt): void {
    switch (s.kind) {
      case 'block':
        this.line('{');
        this.block(s.body);
        this.line('}');
        return;

      case 'let':
        // 初期値のない宣言は、WGSL でもゼロで初期化される。
        this.line(s.init
          ? `var ${this.name(s.sym!)}: ${this.ty(s.sym!.ty)} = ${this.expr(s.init)};`
          : `var ${this.name(s.sym!)}: ${this.ty(s.sym!.ty)};`);
        return;

      case 'assign': {
        const t = s.target;
        if (t.kind === 'index' && t.base.ty!.k === 'heap') {
          const ref = this.bufferOf(t.base);
          if (ref.small) {
            this.line(`um_st_${ref.global}(${this.index(t.index)}, ${this.expr(s.value)});`);
            return;
          }
        }
        this.line(`${this.expr(s.target)} = ${this.expr(s.value)};`);
        return;
      }

      case 'if':
        this.line(`if (${this.expr(s.cond)}) {`);
        this.body(s.then);
        if (s.else) {
          this.line('} else {');
          this.body(s.else);
        }
        this.line('}');
        return;

      case 'while':
        this.line(`while (${this.expr(s.cond)}) {`);
        this.body(s.body);
        this.line('}');
        return;

      case 'for': {
        // WGSL の for は初期化と更新に文を1つずつ取れる。continue でも更新が実行される。
        this.line('{');
        this.indent++;
        if (s.init) this.stmt(s.init);
        const step = s.step ? this.inline(s.step) : '';
        this.line(`for (; ${s.cond ? this.expr(s.cond) : 'true'}; ${step}) {`);
        this.body(s.body);
        this.line('}');
        this.indent--;
        this.line('}');
        return;
      }

      case 'forOf': {
        const it = s.iter.ty as Ty & ({ k: 'array' } | { k: 'heap' });
        const k = `um_k${++this.tmp}`;
        this.line('{');
        this.indent++;
        let len: string;
        let elem: (i: string) => string;
        if (it.k === 'heap') {
          const ref = this.bufferOf(s.iter);
          len = ref.len;
          elem = (i) => (ref.small ? `um_ld_${ref.global}(${i})` : `${ref.global}[${i}]`);
        } else {
          // 配列の値を一度変数に置いてから回す（反復中に元の配列が変わっても影響しない）。
          const arr = `um_a${++this.tmp}`;
          this.line(`var ${arr}: ${this.ty(it)} = ${this.expr(s.iter)};`);
          len = `${it.len}u`;
          elem = (i) => `${arr}[${i}]`;
        }
        this.line(`for (var ${k}: u32 = 0u; ${k} < ${len}; ${k}++) {`);
        this.indent++;
        this.line(`var ${this.name(s.sym!)}: ${this.ty(s.sym!.ty)} = ${elem(k)};`);
        this.indent--;
        this.body(s.body);
        this.line('}');
        this.indent--;
        this.line('}');
        return;
      }

      case 'return':
        this.line(s.value ? `return ${this.expr(s.value)};` : 'return;');
        return;

      case 'break':
        this.line('break;');
        return;

      case 'continue':
        this.line('continue;');
        return;

      case 'expr':
        // 副作用があるのは関数呼び出しだけ。kernel の補助関数は値を返すだけなので、結果を捨てる。
        if (s.expr.kind === 'call') this.line(`_ = ${this.expr(s.expr)};`);
        return;
    }
  }

  /** for の更新部に置く文を1行の文字列にする（代入だけが来る）。 */
  inline(s: Stmt): string {
    if (s.kind !== 'assign') throw new Error('internal: wgsl for-step');
    return `${this.expr(s.target)} = ${this.expr(s.value)}`;
  }

  // ---- 式 ----

  bufferOf(e: Expr): BufferRef {
    if (e.kind !== 'ident') throw new Error('internal: wgsl buffer expression');
    return this.bufs.get(e.sym!.id)!;
  }

  float(v: number): string {
    const x = Math.fround(v);
    if (Object.is(x, -0)) return '(-0.0f)';
    let s = String(x);
    if (!/[.e]/.test(s)) s += '.0';
    // 負の値はかっこで囲む。最上位の定数を置き換えた負の値に単項の `-` が付くと、`--1.5f` と続いてしまう
    // （WGSL では `--` が1つの字句になる）。
    return x < 0 ? `(${s}f)` : `${s}f`;
  }

  /** 添字を u32 にする。負の i32 は大きな値になり、切り詰めで最後の要素を指す（CPU 版と同じ）。 */
  index(e: Expr): string {
    const t = e.ty as Ty & { k: 'int' };
    return t.signed ? `u32(${this.expr(e)})` : this.expr(e);
  }

  expr(e: Expr): string {
    const ty = e.ty!;
    switch (e.kind) {
      case 'int':
        if (ty.k === 'float') return this.float(Number(e.value));
        if (ty.k === 'int' && ty.signed) {
          // -2147483648i は「2147483648i の符号反転」と読まれて範囲外になるので、式で書く。
          return e.value === -2147483648n ? '(-2147483647i - 1i)' : e.value < 0n ? `(${e.value}i)` : `${e.value}i`;
        }
        return `${e.value}u`;

      case 'float':
        return this.float(e.value);

      case 'bool':
        return e.value ? 'true' : 'false';

      case 'ident':
        return this.name(e.sym!);

      case 'unary':
        // WGSL には符号なし整数の符号反転がないので、0 から引く（ラップする）。
        if (e.op === '-' && ty.k === 'int' && !ty.signed) return `(0u - ${this.expr(e.expr)})`;
        return `(${e.op}${this.expr(e.expr)})`;

      case 'binary': {
        const l = this.expr(e.left);
        const r = this.expr(e.right);
        if (e.op === '<<' || e.op === '>>') {
          const rt = e.right.ty as Ty & { k: 'int' };
          return `(${l} ${e.op} (${rt.signed ? `u32(${r})` : r} & 31u))`;
        }
        return `(${l} ${e.op} ${r})`;
      }

      case 'cast': {
        // 浮動小数点から整数への変換は、範囲外で飽和し、NaN は 0 になる（CPU と同じ）。
        // WGSL の i32() / u32() は、実機のGPUでは範囲外の値で飽和しないことがあるので、自前の関数を通す。
        const from = e.expr.ty!;
        if (from.k === 'float' && ty.k === 'int') return `${ty.signed ? 'um_f2i' : 'um_f2u'}(${this.expr(e.expr)})`;
        // 8 / 16 ビットへの変換: 下位のビットを取り出して、i8 / i16 なら符号を広げる（CPU の trunc と同じ値）。
        // 8 / 16 ビットの値は、広げた形の i32 / u32 で持つので、広げる側の変換は 32 ビットどうしと同じ。
        if (ty.k === 'int' && ty.bits < 32) return `extractBits(${this.ty(ty)}(${this.expr(e.expr)}), 0u, ${ty.bits}u)`;
        return `${this.ty(ty)}(${this.expr(e.expr)})`;
      }

      case 'cond':
        // select は両側を評価するが、kernel の式には副作用も停止もないので結果は同じ。
        return `select(${this.expr(e.else)}, ${this.expr(e.then)}, ${this.expr(e.cond)})`;

      case 'field': {
        if (e.base.ty!.k === 'heap' && e.name === 'length') return this.bufferOf(e.base).len;
        return `${this.expr(e.base)}.f_${e.name}`;
      }

      case 'index': {
        const bt = e.base.ty as Ty & ({ k: 'array' } | { k: 'heap' });
        if (bt.k === 'heap') {
          const ref = this.bufferOf(e.base);
          if (ref.small) return `um_ld_${ref.global}(${this.index(e.index)})`;
          return `${ref.global}[um_clamp(${this.index(e.index)}, ${ref.len})]`;
        }
        if (e.index.kind === 'int') return `${this.expr(e.base)}[${e.index.value}]`;
        return `${this.expr(e.base)}[min(${this.index(e.index)}, ${bt.len - 1}u)]`;
      }

      case 'object': {
        const info = this.prog.structs.get((ty as Ty & { k: 'struct' }).name)!;
        const vals = info.fields.map((f) => this.expr(e.fields.find((x) => x.name === f.name)!.value));
        return `${this.ty(ty)}(${vals.join(', ')})`;
      }

      case 'arrayLit':
        return `${this.ty(ty)}(${e.elems.map((x) => this.expr(x)).join(', ')})`;

      case 'call': {
        const sig = this.prog.fns.get(e.callee)!;
        const fn = this.helper(sig, e.args.map((a) => a.expr));
        const args = e.args.filter((a) => a.expr.ty!.k !== 'heap').map((a) => this.expr(a.expr));
        return `${fn}(${args.join(', ')})`;
      }

      case 'math': {
        const a = e.args.map((x) => this.expr(x));
        switch (e.fn) {
          case 'sqrt':
          case 'floor':
          case 'ceil':
          case 'abs':
            return `${e.fn}(${a[0]})`;
          case 'min':
          case 'max':
            // 浮動小数点は CPU 版と同じ「比較して選ぶ」定義にする（NaN の扱いを揃えるため）。
            if (ty.k === 'float') return `select(${a[1]}, ${a[0]}, ${a[0]} ${e.fn === 'min' ? '<' : '>'} ${a[1]})`;
            return `${e.fn}(${a[0]}, ${a[1]})`;
          case 'popcount':
            return `countOneBits(${a[0]})`;
          case 'clz':
            return `countLeadingZeros(${a[0]})`;
          case 'ctz':
            return `countTrailingZeros(${a[0]})`;
          case 'shl':
          case 'shr': {
            // マスクつきシフト。kernel は 32 ビットなのでシフト量を 31 で切る。
            const st = e.args[1].ty as Ty & { k: 'int' };
            return `(${a[0]} ${e.fn === 'shl' ? '<<' : '>>'} (${st.signed ? `u32(${a[1]})` : a[1]} & 31u))`;
          }
        }
        throw new Error(`internal: wgsl Math.${e.fn}`);
      }

      default:
        throw new Error(`internal: wgsl expression ${e.kind}`);
    }
  }

  /**
   * 補助の kernel 関数を出力し、その名前を返す。
   * WGSL では Buffer を関数の引数として渡せないので、Buffer の対応ごとに関数を特殊化する。
   */
  helper(sig: FnSig, args: Expr[]): string {
    const d = sig.decl;
    const refs: (BufferRef | null)[] = d.params.map((p, i) => (p.ty!.k === 'heap' ? this.bufferOf(args[i]) : null));
    const key = [d.name, ...refs.filter((r) => r).map((r) => r!.global)].join('__');
    const fnName = `k_${key}`;
    if (this.fns.has(fnName)) return fnName;
    this.fns.add(fnName);

    // 呼び出し元の生成状態を退避して、関数を別に組み立てる。
    const saved = { lines: this.lines, indent: this.indent, names: this.names, bufs: this.bufs };
    this.lines = [];
    this.indent = 1;
    this.names = new Map();
    this.bufs = new Map();

    const params: string[] = [];
    d.params.forEach((p, i) => {
      if (refs[i]) {
        this.bufs.set(p.sym!.id, refs[i]!);
        return;
      }
      // 引数は書き換えられることがあるので、変数に写してから使う。
      const arg = `a_${p.name}`;
      params.push(`${arg}: ${this.ty(p.ty!)}`);
      this.line(`var ${this.name(p.sym!)}: ${this.ty(p.ty!)} = ${arg};`);
    });
    for (const s of d.body) this.stmt(s);
    const ret = d.ret!.k === 'void' ? '' : ` -> ${this.ty(d.ret!)}`;
    this.fnText.push(`fn ${fnName}(${params.join(', ')})${ret} {`, ...this.lines, '}', '');

    this.lines = saved.lines;
    this.indent = saved.indent;
    this.names = saved.names;
    this.bufs = saved.bufs;
    return fnName;
  }

  /**
   * 8 / 16 ビットの要素の Buffer を読み書きする関数を出す。要素 j は語 j / (32 / bits) の
   * (j % (32 / bits)) * bits ビット目から入っている（リトルエンディアン。CPU のメモリの並びと同じ）。
   * 書き込みは、同じ語の別の要素を他の実行が同時に書くので、アトミックな演算で自分のビットだけを変える。
   * 普通の Buffer では要素 j を書くのは 1 つの実行だけなので、AND で消して OR で入れる 2 回でよい。
   * RacyBuffer では同じ要素を複数の実行が書きうるので、値が混ざらないように比較交換で入れ替える。
   */
  smallAccess(global: string, len: string, small: { bits: number; signed: boolean }, write: boolean, racy: boolean): void {
    const per = 32 / small.bits;
    const sh = per === 4 ? 2 : 1;
    const t = small.signed ? 'i32' : 'u32';
    const mask = small.bits === 8 ? '0xffu' : '0xffffu';
    const word = write ? `atomicLoad(&${global}[j >> ${sh}u])` : `${global}[j >> ${sh}u]`;
    const at = `(j & ${per - 1}u) * ${small.bits}u`;
    this.fnText.push(
      `fn um_ld_${global}(i: u32) -> ${t} {`,
      `  let j: u32 = um_clamp(i, ${len});`,
      `  return extractBits(${small.signed ? `bitcast<i32>(${word})` : word}, ${at}, ${small.bits}u);`,
      '}',
      '',
    );
    if (!write) return;
    const body = racy
      ? [
        `  loop {`,
        `    let old: u32 = atomicLoad(&${global}[w]);`,
        `    let r = atomicCompareExchangeWeak(&${global}[w], old, (old & ~m) | v);`,
        `    if (r.exchanged) { break; }`,
        `  }`,
      ]
      : [`  atomicAnd(&${global}[w], ~m);`, `  atomicOr(&${global}[w], v);`];
    this.fnText.push(
      `fn um_st_${global}(i: u32, x: ${t}) {`,
      `  let j: u32 = um_clamp(i, ${len});`,
      `  let w: u32 = j >> ${sh}u;`,
      `  let at: u32 = ${at};`,
      `  let m: u32 = ${mask} << at;`,
      `  let v: u32 = (bitcast<u32>(x) & ${mask}) << at;`,
      ...body,
      '}',
      '',
    );
  }

  /** dispatch される kernel 全体を出力する。 */
  kernel(d: FnDecl): WgslKernel {
    const buffers: WgslBuffer[] = [];
    const paramFields = ['  count: u32,', '  row: u32,'];
    const globals: string[] = [];
    const prelude: string[] = [];

    d.params.slice(1).forEach((p) => {
      const ty = p.ty!;
      if (ty.k === 'heap') {
        const global = `b_${p.name}`;
        const binding = buffers.length + 1;
        buffers.push({ name: p.name, writable: p.mode === 'mut', elem: ty.elem });
        paramFields.push(`  len_${p.name}: u32,`);
        const len = `um_params.len_${p.name}`;
        const el = ty.elem;
        if (el.k === 'int' && el.bits < 32) {
          const small = { bits: el.bits, signed: el.signed };
          const write = p.mode === 'mut';
          globals.push(`@group(0) @binding(${binding}) var<storage, ${write ? 'read_write' : 'read'}> ${global}: array<${write ? 'atomic<u32>' : 'u32'}>;`);
          this.smallAccess(global, len, small, write, !!ty.racy);
          this.bufs.set(p.sym!.id, { global, len, small });
          return;
        }
        globals.push(`@group(0) @binding(${binding}) var<storage, ${p.mode === 'mut' ? 'read_write' : 'read'}> ${global}: array<${this.ty(ty.elem)}>;`);
        this.bufs.set(p.sym!.id, { global, len });
      }
    });
    d.params.slice(1).forEach((p) => {
      const ty = p.ty!;
      if (ty.k === 'heap') return;
      // bool は GPU と共有できる表現がないので、u32 で渡して戻す。
      const shared = ty.k === 'bool' ? 'u32' : this.ty(ty);
      paramFields.push(`  p_${p.name}: ${shared},`);
      const value = ty.k === 'bool' ? `(um_params.p_${p.name} != 0u)` : `um_params.p_${p.name}`;
      prelude.push(`  var ${this.name(p.sym!)}: ${this.ty(ty)} = ${value};`);
    });

    this.names.set(d.params[0].sym!.id, 'um_i');
    for (const s of d.body) this.stmt(s);

    const source = [
      `// UnifiedMozaic が kernel '${d.name}' から生成した WGSL。手で編集しない。`,
      '',
      ...this.out,
      'struct UmParams {',
      ...paramFields,
      '}',
      '',
      '@group(0) @binding(0) var<storage, read> um_params: UmParams;',
      ...globals,
      '',
      '// 範囲外の添字を最後の要素に切り詰める。長さ 0 の Buffer は1要素ぶん確保されているので 0 を返す。',
      'fn um_clamp(i: u32, len: u32) -> u32 {',
      '  return min(i, max(len, 1u) - 1u);',
      '}',
      '',
      '// 浮動小数点から整数へ。範囲外は端の値に飽和し、NaN は 0。変換そのものは範囲内に収めてから行う。',
      'fn um_f2i(v: f32) -> i32 {',
      '  let r: i32 = i32(clamp(v, -2147483648.0f, 2147483520.0f));',
      '  let hi: i32 = select(r, 2147483647i, v >= 2147483648.0f);',
      '  let lo: i32 = select(hi, -2147483647i - 1i, v <= -2147483648.0f);',
      '  return select(lo, 0i, v != v);',
      '}',
      '',
      'fn um_f2u(v: f32) -> u32 {',
      '  let r: u32 = u32(clamp(v, 0.0f, 4294967040.0f));',
      '  let hi: u32 = select(r, 4294967295u, v >= 4294967296.0f);',
      '  let lo: u32 = select(hi, 0u, v <= 0.0f);',
      '  return select(lo, 0u, v != v);',
      '}',
      '',
      ...this.fnText,
      `@compute @workgroup_size(${WORKGROUP_SIZE})`,
      'fn um_main(@builtin(global_invocation_id) gid: vec3<u32>) {',
      '  let um_i: u32 = gid.y * um_params.row + gid.x;',
      '  if (um_i >= um_params.count) {',
      '    return;',
      '  }',
      ...prelude,
      ...this.lines,
      '}',
      '',
    ].join('\n');
    return { source, buffers };
  }
}

export function emitWgsl(prog: Program, kernel: FnDecl): WgslKernel {
  return new Emitter(prog).kernel(kernel);
}
