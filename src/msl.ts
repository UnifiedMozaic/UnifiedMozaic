// kernel 関数を MSL（Metal Shading Language）に変換する。
//
// CPU 用のコード生成（codegen.ts）や WGSL（wgsl.ts）と同じ結果になるように、次を明示的に生成する:
//   - 添字は最後の要素に切り詰める（um_clamp）。長さ 0 の Buffer は1要素ぶん確保されている
//   - シフト量はビット幅で割った余り
//   - 整数のゼロ除算は x / 0 = x、x % 0 = 0、最小値 / -1 = 最小値（x % -1 = 0）。MSL（C++）では未定義なので関数で包む
//   - 符号つき整数の + - * と符号反転は、符号なしにしてから計算して戻す（ラップ。C++ の符号つきオーバーフローは未定義）
//   - 浮動小数点から整数への変換は um_f2i / um_f2u / um_f2l / um_f2ul（範囲外は飽和、NaN は 0）
//   - 64 ビットの整数（i64 / u64）は MSL の long / ulong。シフト量は & 63、添字は uint の最大値で飽和させてから切り詰める
//
// WGSL との違い: MSL では Buffer を関数の引数（ポインタと長さ）として渡せるので、補助関数は特殊化せず1つだけ出す。
// 固定長配列は C++ の生の配列だとコピーできないので、配列を1つ包んだ struct にする。
// Metal での実行は runtime/metal.m（UM_METAL=1）。tests/metal の kernel が CPU と同じ結果になることを確かめている。

import { tyStr } from './ast.ts';
import type { Expr, FnDecl, Stmt, Sym, Ty } from './ast.ts';
import type { Program } from './checker.ts';

/** Buffer の引数が、MSL のどのポインタと長さに対応するか。 */
interface BufferRef {
  ptr: string;
  len: string;
  /**
   * アトミックに読み書きする RacyBuffer（kernel の中で load / store / fetchAdd / compareExchange を使うもの）。
   * ポインタは atomic_int（i32）か atomic_uint（u32、f32 はビット列）を指し、要素の読み書きもすべてアトミック（relaxed）にする。
   */
  atomic?: { elem: Ty };
}

class Emitter {
  prog: Program;
  out: string[] = [];
  structs = new Set<string>();
  arrays = new Set<string>();
  /** 生成済みの補助関数。 */
  fns = new Set<string>();
  fnText: string[] = [];
  names = new Map<number, string>();
  bufs = new Map<number, BufferRef>();
  indent = 1;
  lines: string[] = [];
  tmp = 0;
  /** dispatchGroups 用の形か（グループの番号・グループ内の番号から添字を作る）。 */
  groups = false;

  constructor(prog: Program) {
    this.prog = prog;
  }

  // ---- 型 ----

  ty(t: Ty): string {
    switch (t.k) {
      case 'bool':
        return 'bool';
      case 'int':
        // MSL にも C++ にもある名前（MSL の char / short の別名）。C++ の char は符号の有無が処理系で違うので使わない。
        if (t.bits === 8) return t.signed ? 'int8_t' : 'uint8_t';
        if (t.bits === 16) return t.signed ? 'int16_t' : 'uint16_t';
        // 64 ビットは MSL の long / ulong（Metal の long は 64 ビット）。
        if (t.bits === 64) return t.signed ? 'long' : 'ulong';
        return t.signed ? 'int' : 'uint';
      case 'float':
        return 'float';
      case 'array':
        return this.arrayTy(t);
      case 'struct':
        this.declareStruct(t.name);
        return `S_${t.name}`;
      default:
        throw new Error(`internal: msl type ${tyStr(t)}`);
    }
  }

  /** 配列を包む struct の名前。型の名前から作る（同じ型なら同じ名前）。 */
  arrayTy(t: Ty & { k: 'array' }): string {
    const elem = this.ty(t.elem);
    const name = `A_${elem.replace(/^S_/, 's_')}_${t.len}`;
    if (!this.arrays.has(name)) {
      this.arrays.add(name);
      this.out.push(`struct ${name} {`, `  ${elem} v[${t.len}];`, '};', '');
    }
    return name;
  }

  declareStruct(name: string): void {
    if (this.structs.has(name)) return;
    this.structs.add(name);
    const info = this.prog.structs.get(name)!;
    const fields = info.fields.map((f) => `  ${this.ty(f.ty)} f_${f.name};`);
    this.out.push(`struct S_${name} {`, ...fields, '};', '');
  }

  // ---- 文 ----

  line(s: string): void {
    this.lines.push('  '.repeat(this.indent) + s);
  }

  name(sym: Sym): string {
    if (sym.global) return this.constArray(sym);
    let n = this.names.get(sym.id);
    if (!n) {
      n = `u_${sym.name}_${sym.id}`;
      this.names.set(sym.id, n);
    }
    return n;
  }

  /** 出力した定数の配列。 */
  consts = new Set<string>();

  /**
   * 最上位の定数の配列を、プログラムの範囲の `constant` の変数として出力する（kernel の引数にはしない）。
   * 型は固定長配列と同じく配列を包んだ struct で、添字は同じく最後の要素に切り詰める。
   */
  constArray(sym: Sym): string {
    const name = `K_${sym.global!.name}`;
    if (!this.consts.has(name)) {
      this.consts.add(name);
      const lit = (v: unknown, t: Ty): string => {
        if (t.k === 'array') return `{{${(v as unknown[]).map((x) => lit(x, t.elem)).join(', ')}}}`;
        if (t.k === 'bool') return v ? 'true' : 'false';
        if (t.k === 'float') return this.float(v as number);
        const x = v as bigint;
        if (t.k === 'int' && t.signed) return x === -2147483648n ? '(-2147483647 - 1)' : x < 0n ? `(${x})` : `${x}`;
        return `${x}u`;
      };
      const ty = this.ty(sym.ty);
      this.out.push(`constant ${ty} ${name} = ${lit(sym.global!.value, sym.ty)};`, '');
    }
    return name;
  }

  block(body: Stmt[]): void {
    this.indent++;
    for (const s of body) this.stmt(s);
    this.indent--;
  }

  /** 本体を { } の中身として出す。 */
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
        // 共有配列は kernel の始めで宣言し、0 で埋めてある（kernel を参照）。
        if (s.sym!.shared) return;
        // 初期値のない宣言は、ゼロで初期化する（T{}）。
        this.line(`${this.ty(s.sym!.ty)} ${this.name(s.sym!)} = ${s.init ? this.expr(s.init) : `${this.ty(s.sym!.ty)}{}`};`);
        return;

      case 'assign': {
        const t = s.target;
        if (t.kind === 'index' && t.base.ty!.k === 'heap') {
          const ref = this.bufferOf(t.base);
          if (ref.atomic) {
            this.line(`${this.atomicStore(ref, this.index(t.index), this.expr(s.value))};`);
            return;
          }
        }
        this.line(`${this.expr(s.target)} = ${this.expr(s.value)};`);
        return;
      }

      case 'if':
        this.line(`if (${this.cond(s.cond)}) {`);
        this.body(s.then);
        if (s.else) {
          this.line('} else {');
          this.body(s.else);
        }
        this.line('}');
        return;

      case 'while':
        this.line(`while (${this.cond(s.cond)}) {`);
        this.body(s.body);
        this.line('}');
        return;

      case 'for': {
        // C++ の for は continue でも更新が実行される。初期化の変数は外側の { } に閉じ込める。
        this.line('{');
        this.indent++;
        if (s.init) this.stmt(s.init);
        const step = s.step ? this.inline(s.step) : '';
        this.line(`for (; ${s.cond ? this.cond(s.cond) : 'true'}; ${step}) {`);
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
          elem = (i) => (ref.atomic ? this.atomicLoad(ref, i, false) : `${ref.ptr}[${i}]`);
        } else {
          // 配列の値を一度変数に置いてから回す（反復中に元の配列が変わっても影響しない）。
          const arr = `um_a${++this.tmp}`;
          this.line(`${this.ty(it)} ${arr} = ${this.expr(s.iter)};`);
          len = `${it.len}u`;
          elem = (i) => `${arr}.v[${i}]`;
        }
        this.line(`for (uint ${k} = 0u; ${k} < ${len}; ${k}++) {`);
        this.indent++;
        this.line(`${this.ty(s.sym!.ty)} ${this.name(s.sym!)} = ${elem(k)};`);
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

      case 'expr': {
        // 副作用があるのは、関数呼び出し・RacyBuffer のアトミック演算・group.barrier()。
        const e = s.expr;
        if (e.kind === 'math' && e.fn === 'group.barrier') {
          // 共有配列の読み書きの順序を揃える（device のメモリは含めない）。
          this.line('threadgroup_barrier(mem_flags::mem_threadgroup);');
        } else if (e.kind === 'mcall' && e.ty!.k === 'void') {
          this.line(`${this.expr(e)};`);
        } else if (e.kind === 'call' || e.kind === 'mcall') {
          this.line(`(void)${this.expr(e)};`);
        }
        return;
      }
    }
  }

  /** for の更新部に置く文を1行の文字列にする（代入だけが来る）。 */
  inline(s: Stmt): string {
    if (s.kind !== 'assign') throw new Error('internal: msl for-step');
    return `${this.expr(s.target)} = ${this.expr(s.value)}`;
  }

  // ---- 式 ----

  /** 条件式の外側の ( ) を外す（clang の「余分な括弧」の警告を避ける）。 */
  cond(e: Expr): string {
    const s = this.expr(e);
    if (!s.startsWith('(') || !s.endsWith(')')) return s;
    let depth = 0;
    for (let i = 0; i < s.length - 1; i++) {
      if (s[i] === '(') depth++;
      else if (s[i] === ')' && --depth === 0) return s;
    }
    return s.slice(1, -1);
  }

  /** アトミックな RacyBuffer の要素 i（切り詰め前）の場所。 */
  atomicPlace(ref: BufferRef, i: string, clamp = true): string {
    return `&${ref.ptr}[${clamp ? `um_clamp(${i}, ${ref.len})` : i}]`;
  }

  atomicLoad(ref: BufferRef, i: string, clamp = true): string {
    const v = `atomic_load_explicit(${this.atomicPlace(ref, i, clamp)}, memory_order_relaxed)`;
    return ref.atomic!.elem.k === 'float' ? `as_type<float>(${v})` : v;
  }

  atomicStore(ref: BufferRef, i: string, v: string): string {
    const x = ref.atomic!.elem.k === 'float' ? `as_type<uint>(${v})` : v;
    return `atomic_store_explicit(${this.atomicPlace(ref, i)}, ${x}, memory_order_relaxed)`;
  }

  /** `r.load(i)` などの RacyBuffer のアトミック演算。アトミックでない Buffer（補助関数の引数）では普通に読む。 */
  racyCall(e: Expr & { kind: 'mcall' }): string {
    const ref = this.bufferOf(e.recv);
    const i = this.index(e.args[0].expr);
    const a = e.args.slice(1).map((x) => this.expr(x.expr));
    if (!ref.atomic) {
      if (e.name === 'load') return `${ref.ptr}[um_clamp(${i}, ${ref.len})]`;
      throw new Error(`internal: msl non-atomic ${e.name}`);
    }
    switch (e.name) {
      case 'load':
        return this.atomicLoad(ref, i);
      case 'store':
        return this.atomicStore(ref, i, a[0]);
      case 'fetchAdd':
        return `atomic_fetch_add_explicit(${this.atomicPlace(ref, i)}, ${a[0]}, memory_order_relaxed)`;
      case 'compareExchange': {
        const signed = (ref.atomic.elem as Ty & { k: 'int' }).signed;
        return `${signed ? 'um_cas_i' : 'um_cas_u'}(${this.atomicPlace(ref, i)}, ${a[0]}, ${a[1]})`;
      }
    }
    throw new Error(`internal: msl RacyBuffer.${e.name}`);
  }

  bufferOf(e: Expr): BufferRef {
    if (e.kind !== 'ident') throw new Error('internal: msl buffer expression');
    return this.bufs.get(e.sym!.id)!;
  }

  float(v: number): string {
    if (Number.isNaN(v)) return 'NAN';
    const x = Math.fround(v);
    if (x === Infinity) return 'INFINITY';
    if (x === -Infinity) return '(-INFINITY)';
    if (Object.is(x, -0)) return '(-0.0f)';
    let s = String(x);
    if (!/[.e]/.test(s)) s += '.0';
    // 負の値はかっこで囲む。最上位の定数を置き換えた負の値に単項の `-` が付くと、`--1.5f` と続いてしまう。
    return x < 0 ? `(${s}f)` : `${s}f`;
  }

  /**
   * 添字を uint にする。負の int は大きな値になり、切り詰めで最後の要素を指す（CPU 版と同じ）。
   * 64 ビットの添字は uint の最大値で飽和させる（下位 32 ビットに切ると範囲外が範囲内に見える）。
   * Buffer の長さは u32 なので、飽和した値は必ず最後の要素に切り詰められる（CPU は 64 ビットのまま切り詰める）。
   */
  index(e: Expr): string {
    const t = e.ty as Ty & { k: 'int' };
    if (t.bits === 64) return `uint(min(ulong(${this.expr(e)}), 4294967295ul))`;
    return t.signed ? `uint(${this.expr(e)})` : this.expr(e);
  }

  expr(e: Expr): string {
    const ty = e.ty!;
    switch (e.kind) {
      case 'int':
        if (ty.k === 'float') return this.float(Number(e.value));
        if (ty.k === 'int' && ty.bits === 64) {
          // 64 ビットのリテラルは long / ulong の型を付けて書く。最小値は 9223372036854775808 が範囲外なので式で書く。
          if (!ty.signed) return `${e.value}ul`;
          return e.value === -9223372036854775808n ? '(-9223372036854775807l - 1l)' : e.value < 0n ? `(${e.value}l)` : `${e.value}l`;
        }
        if (ty.k === 'int' && ty.signed) {
          // -2147483648 は「2147483648 の符号反転」と読まれて範囲外になるので、式で書く。
          return e.value === -2147483648n ? '(-2147483647 - 1)' : e.value < 0n ? `(${e.value})` : `${e.value}`;
        }
        return `${e.value}u`;

      case 'float':
        return this.float(e.value);

      case 'bool':
        return e.value ? 'true' : 'false';

      case 'ident':
        return this.name(e.sym!);

      case 'unary':
        // 符号つき整数の符号反転は、符号なしで 0 から引いて戻す（ラップする）。
        if (e.op === '-' && ty.k === 'int') {
          const [s, u, z] = ty.bits === 64 ? ['long', 'ulong', '0ul'] : ['int', 'uint', '0u'];
          return ty.signed ? `${s}(${z} - ${u}(${this.expr(e.expr)}))` : `(${z} - ${this.expr(e.expr)})`;
        }
        return `(${e.op}${this.expr(e.expr)})`;

      case 'binary': {
        const l = this.expr(e.left);
        const r = this.expr(e.right);
        const lt = e.left.ty!;
        const op = e.op;
        if (lt.k === 'int') {
          const signed = lt.signed;
          const wide = lt.bits === 64;
          const [st, ut] = wide ? ['long', 'ulong'] : ['int', 'uint'];
          switch (op) {
            case '+':
            case '-':
            case '*':
              return signed ? `${st}(${ut}(${l}) ${op} ${ut}(${r}))` : `(${l} ${op} ${r})`;
            case '/':
              return `${signed ? 'um_div_i' : 'um_div_u'}(${l}, ${r})`;
            case '%':
              return `${signed ? 'um_rem_i' : 'um_rem_u'}(${l}, ${r})`;
            case '<<':
            case '>>': {
              // シフト量はビット幅で割った余り（32 ビットは & 31、64 ビットは & 63）。
              const amt = this.shiftAmount(r, e.right.ty as Ty & { k: 'int' }, lt.bits);
              // 符号つきの左シフトは、符号なしでずらしてから戻す。
              if (op === '<<' && signed) return `${st}(${ut}(${l}) << ${amt})`;
              return `(${l} ${op} ${amt})`;
            }
          }
        }
        if (lt.k === 'float' && op === '%') return `fmod(${l}, ${r})`;
        if (lt.k === 'bool' && (op === '&' || op === '|' || op === '^')) return `bool(${l} ${op} ${r})`;
        return `(${l} ${op} ${r})`;
      }

      case 'cast': {
        // 浮動小数点から整数への変換は、範囲外で飽和し、NaN は 0 になる（CPU と同じ）。
        const from = e.expr.ty!;
        const x = this.expr(e.expr);
        if (from.k === 'float' && ty.k === 'int') {
          const f = ty.bits === 64 ? (ty.signed ? 'um_f2l' : 'um_f2ul') : ty.signed ? 'um_f2i' : 'um_f2u';
          return `${f}(${x})`;
        }
        if (ty.k === 'bool') return from.k === 'bool' ? x : `(${x} != 0)`;
        if (ty.k === 'int' && ty.bits < 32) {
          // 8 / 16 ビットへの変換は下位のビットを残す（CPU の trunc と同じ）。符号つきの型への範囲外の変換は
          // C++ では処理系定義なので、同じ幅の符号なしにしてから（2 の補数で）ビットを読み替える。
          const u = ty.bits === 8 ? 'uint8_t' : 'uint16_t';
          return ty.signed ? `as_type<${this.ty(ty)}>(${u}(${x}))` : `${u}(${x})`;
        }
        return `${this.ty(ty)}(${x})`;
      }

      case 'cond':
        // 選ばれた側だけが評価される（CPU 版と同じ）。
        return `(${this.expr(e.cond)} ? ${this.expr(e.then)} : ${this.expr(e.else)})`;

      case 'field': {
        if (e.base.ty!.k === 'heap' && e.name === 'length') return this.bufferOf(e.base).len;
        return `${this.expr(e.base)}.f_${e.name}`;
      }

      case 'index': {
        const bt = e.base.ty as Ty & ({ k: 'array' } | { k: 'heap' });
        if (bt.k === 'heap') {
          const ref = this.bufferOf(e.base);
          if (ref.atomic) return this.atomicLoad(ref, this.index(e.index));
          return `${ref.ptr}[um_clamp(${this.index(e.index)}, ${ref.len})]`;
        }
        if (e.index.kind === 'int') return `${this.expr(e.base)}.v[${e.index.value}]`;
        return `${this.expr(e.base)}.v[min(${this.index(e.index)}, ${bt.len - 1}u)]`;
      }

      case 'object': {
        const info = this.prog.structs.get((ty as Ty & { k: 'struct' }).name)!;
        const vals = info.fields.map((f) => this.expr(e.fields.find((x) => x.name === f.name)!.value));
        return `${this.ty(ty)}{${vals.join(', ')}}`;
      }

      case 'arrayLit':
        return `${this.ty(ty)}{{${e.elems.map((x) => this.expr(x)).join(', ')}}}`;

      case 'call': {
        const sig = this.prog.fns.get(e.callee)!;
        const fn = this.helper(sig.decl);
        // Buffer はポインタと長さの2つ、それ以外は値で渡す。
        const args = e.args.flatMap((a) => {
          if (a.expr.ty!.k !== 'heap') return [this.expr(a.expr)];
          const ref = this.bufferOf(a.expr);
          // 補助関数は読むだけなので、アトミックな Buffer は普通の要素のポインタとして渡す。
          if (ref.atomic) return [`reinterpret_cast<const device ${this.ty(ref.atomic.elem)}*>(${ref.ptr})`, ref.len];
          return [ref.ptr, ref.len];
        });
        return `${fn}(${args.join(', ')})`;
      }

      case 'mcall':
        return this.racyCall(e);

      case 'math': {
        const a = e.args.map((x) => this.expr(x));
        const cast = (s: string) => `${this.ty(ty)}(${s})`;
        switch (e.fn) {
          case 'group.local':
            return 'um_l';
          case 'group.id':
            return 'um_g';
          case 'group.size':
            return 'um_sz';
          case 'group.count':
            return 'um_params.count';
          // simdgroup（グループの中の連続した 32 スレッド。Apple の GPU の幅）。CPU（codegen.ts の collective）と同じ意味。
          case 'group.simdLane':
            return '(um_l & 31u)';
          case 'group.simdSum':
            return `simd_sum(${a[0]})`;
          case 'group.simdMin':
            return `simd_min(${a[0]})`;
          case 'group.simdMax':
            return `simd_max(${a[0]})`;
          case 'group.simdShuffle':
            // lane が simdgroup の外（最後の短い simdgroup を含む）なら、最後のスレッドに切り詰める。
            return `simd_shuffle(${a[0]}, ushort(min(${a[1]}, min(32u, um_sz - (um_l & ~31u)) - 1u)))`;
          case 'sqrt':
          case 'floor':
          case 'ceil':
            return `${e.fn}(${a[0]})`;
          case 'abs':
            if (ty.k === 'float') return `fabs(${a[0]})`;
            // 最小値の絶対値はラップして最小値のまま。符号なしはそのまま。
            return (ty as Ty & { k: 'int' }).signed ? `um_abs_i(${a[0]})` : a[0];
          case 'min':
          case 'max':
            // 浮動小数点は CPU 版と同じ「比較して選ぶ」定義にする（NaN の扱いを揃えるため）。
            if (ty.k === 'float') return `um_f${e.fn}(${a[0]}, ${a[1]})`;
            return `${e.fn}(${a[0]}, ${a[1]})`;
          case 'popcount':
            return cast(`popcount(${a[0]})`);
          case 'clz':
            return cast(`clz(${a[0]})`);
          case 'ctz':
            return cast(`ctz(${a[0]})`);
          case 'shl':
          case 'shr': {
            // マスクつきシフト。シフト量を値のビット幅で割った余りにする。
            const vt = e.args[0].ty as Ty & { k: 'int' };
            const amt = this.shiftAmount(a[1], e.args[1].ty as Ty & { k: 'int' }, vt.bits);
            if (e.fn === 'shl' && vt.signed) {
              return vt.bits === 64 ? cast(`long(ulong(${a[0]}) << ${amt})`) : cast(`int(uint(${a[0]}) << ${amt})`);
            }
            return cast(`(${a[0]} ${e.fn === 'shl' ? '<<' : '>>'} ${amt})`);
          }
        }
        throw new Error(`internal: msl Math.${e.fn}`);
      }

      default:
        throw new Error(`internal: msl expression ${e.kind}`);
    }
  }

  /**
   * シフト量を「値のビット幅で割った余り」にした式。量はどの整数型でもよく、符号なしにしてからマスクする
   * （CPU は量を値の型に変換してからマスクする。下位のビットは変換で変わらないので同じ結果）。
   */
  shiftAmount(r: string, rt: Ty & { k: 'int' }, bits: number): string {
    if (bits === 64) return `(${rt.bits === 64 && !rt.signed ? r : `ulong(${r})`} & 63ul)`;
    return `(${rt.bits === 32 && !rt.signed ? r : `uint(${r})`} & 31u)`;
  }

  /** 補助の kernel 関数を出力し、その名前を返す。Buffer の引数は「ポインタ, 長さ」の2つに展開する。 */
  helper(d: FnDecl): string {
    const fnName = `k_${d.name}`;
    if (this.fns.has(fnName)) return fnName;
    this.fns.add(fnName);

    // 呼び出し元の生成状態を退避して、関数を別に組み立てる。
    const saved = { lines: this.lines, indent: this.indent, names: this.names, bufs: this.bufs };
    this.lines = [];
    this.indent = 1;
    this.names = new Map();
    this.bufs = new Map();

    const params: string[] = [];
    for (const p of d.params) {
      if (p.ty!.k === 'heap') {
        // 補助関数に渡せるのは読み取りの Buffer だけ（mut は checker が弾く）。
        params.push(`const device ${this.ty(p.ty!.elem)}* b_${p.name}`, `uint l_${p.name}`);
        this.bufs.set(p.sym!.id, { ptr: `b_${p.name}`, len: `l_${p.name}` });
      } else {
        // 値渡しなので、書き換えられてもよい。
        params.push(`${this.ty(p.ty!)} ${this.name(p.sym!)}`);
      }
    }
    for (const s of d.body) this.stmt(s);
    const ret = d.ret!.k === 'void' ? 'void' : this.ty(d.ret!);
    this.fnText.push(`inline ${ret} ${fnName}(${params.join(', ')}) {`, ...this.lines, '}', '');

    this.lines = saved.lines;
    this.indent = saved.indent;
    this.names = saved.names;
    this.bufs = saved.bufs;
    return fnName;
  }

  /**
   * dispatch される kernel 全体を出力する。
   * groups: dispatchGroups 用。count はグループの数、row は1行のグループの数で、グループは
   * dispatchThreadgroups の格子に並ぶ（runtime/metal.m）。添字 = グループの番号 × 大きさ + グループ内の番号。
   */
  kernel(d: FnDecl, groups: boolean): string {
    this.groups = groups;
    const paramFields = ['  uint count;', '  uint row;'];
    const args: string[] = ['constant UmParams& um_params [[buffer(0)]]'];
    const prelude: string[] = [];
    // Buffer をポインタの表から受け取る入口（um_main_bl）のための、表のフィールドと、同じ名前の変数への取り出し。
    const bufFields: string[] = [];
    const bufLocals: string[] = [];
    let binding = 0;
    const atomics = atomicParams(d);

    d.params.slice(1).forEach((p) => {
      const ty = p.ty!;
      if (ty.k === 'heap') {
        binding++;
        paramFields.push(`  uint len_${p.name};`);
        const ref: BufferRef = { ptr: `b_${p.name}`, len: `um_params.len_${p.name}` };
        if (atomics.has(p.sym!.id)) {
          const signed = ty.elem.k === 'int' && ty.elem.signed;
          const pt = `device ${signed ? 'atomic_int' : 'atomic_uint'}*`;
          args.push(`${pt} b_${p.name} [[buffer(${binding})]]`);
          bufFields.push(`  ${pt} b_${p.name};`);
          bufLocals.push(`  ${pt} b_${p.name} = um_bufs.b_${p.name};`);
          ref.atomic = { elem: ty.elem };
        } else {
          const pt = `${p.mode === 'mut' ? '' : 'const '}device ${this.ty(ty.elem)}*`;
          args.push(`${pt} b_${p.name} [[buffer(${binding})]]`);
          bufFields.push(`  ${pt} b_${p.name};`);
          bufLocals.push(`  ${pt} b_${p.name} = um_bufs.b_${p.name};`);
        }
        this.bufs.set(p.sym!.id, ref);
      }
    });
    d.params.slice(1).forEach((p) => {
      const ty = p.ty!;
      if (ty.k === 'heap') return;
      // bool は CPU と共有できる表現がないので、uint で渡して戻す。
      const shared = ty.k === 'bool' ? 'uint' : this.ty(ty);
      paramFields.push(`  ${shared} p_${p.name};`);
      const value = ty.k === 'bool' ? `(um_params.p_${p.name} != 0u)` : `um_params.p_${p.name}`;
      prelude.push(`  ${this.ty(ty)} ${this.name(p.sym!)} = ${value};`);
    });
    if (groups) {
      args.push('uint3 um_tg [[threadgroup_position_in_grid]]', 'uint3 um_tl [[thread_position_in_threadgroup]]',
        'uint3 um_tn [[threads_per_threadgroup]]');
    } else {
      args.push('uint3 gid [[thread_position_in_grid]]');
    }

    // 共有配列: kernel の始めで宣言し、グループの全スレッドで手分けして 0 で埋め、そろうのを待つ。
    // （return より前に置くので、どのスレッドも必ずこの barrier を通る。）
    const shared: string[] = [];
    for (const s of d.body) {
      if (s.kind !== 'let' || !s.sym!.shared) continue;
      const t = s.sym!.ty as Ty & { k: 'array' };
      const n = this.name(s.sym!);
      shared.push(`  threadgroup ${this.ty(t)} ${n};`);
      shared.push(`  for (uint z = um_l; z < ${t.len}u; z += um_sz) {`, `    ${n}.v[z] = 0;`, '  }');
    }
    if (shared.length > 0) shared.push('  threadgroup_barrier(mem_flags::mem_threadgroup);');

    this.names.set(d.params[0].sym!.id, 'um_i');
    for (const s of d.body) this.stmt(s);

    const entry = groups
      ? [
        '  uint um_g = um_tg.y * um_params.row + um_tg.x;',
        '  if (um_g >= um_params.count) {',
        '    return;',
        '  }',
        '  uint um_l = um_tl.x;',
        '  uint um_sz = um_tn.x;',
        '  uint um_i = um_g * um_sz + um_l;',
        '  (void)um_l;',
        '  (void)um_sz;',
        '  (void)um_i;',
        ...shared,
      ]
      : [
        '  uint um_i = gid.y * um_params.row + gid.x;',
        '  if (um_i >= um_params.count) {',
        '    return;',
        '  }',
      ];

    return [
      `// UnifiedMozaic が kernel '${d.name}' から生成した MSL。手で編集しない。`,
      '// 浮動小数点は CPU と同じ結果にするため、高速化の近似と積和の融合を切る。',
      '// （MTLCompileOptions の fastMathEnabled = false か、コンパイラの -fno-fast-math も併せて指定すること）',
      '#include <metal_stdlib>',
      'using namespace metal;',
      '#pragma METAL fp math_mode(safe)',
      '#pragma METAL fp contract(off)',
      '',
      ...this.out,
      'struct UmParams {',
      ...paramFields,
      '};',
      '',
      '// 範囲外の添字を最後の要素に切り詰める。長さ 0 の Buffer は1要素ぶん確保されているので 0 を返す。',
      'inline uint um_clamp(uint i, uint len) {',
      '  return min(i, max(len, 1u) - 1u);',
      '}',
      '',
      '// 整数の除算と剰余。x / 0 = x、x % 0 = 0、最小値 / -1 = 最小値、x % -1 = 0。',
      'inline int um_div_i(int a, int b) {',
      '  if (b == 0) return a;',
      '  if (b == -1) return int(0u - uint(a));',
      '  return a / b;',
      '}',
      '',
      'inline int um_rem_i(int a, int b) {',
      '  if (b == 0 || b == -1) return 0;',
      '  return a % b;',
      '}',
      '',
      'inline uint um_div_u(uint a, uint b) {',
      '  return b == 0u ? a : a / b;',
      '}',
      '',
      'inline uint um_rem_u(uint a, uint b) {',
      '  return b == 0u ? 0u : a % b;',
      '}',
      '',
      '// 64 ビットの版（long / ulong の多重定義）。',
      'inline long um_div_i(long a, long b) {',
      '  if (b == 0) return a;',
      '  if (b == -1) return long(0ul - ulong(a));',
      '  return a / b;',
      '}',
      '',
      'inline long um_rem_i(long a, long b) {',
      '  if (b == 0 || b == -1) return 0;',
      '  return a % b;',
      '}',
      '',
      'inline ulong um_div_u(ulong a, ulong b) {',
      '  return b == 0ul ? a : a / b;',
      '}',
      '',
      'inline ulong um_rem_u(ulong a, ulong b) {',
      '  return b == 0ul ? 0ul : a % b;',
      '}',
      '',
      '// 最小値の絶対値は最小値のまま（ラップ）。',
      'inline int um_abs_i(int a) {',
      '  return a < 0 ? int(0u - uint(a)) : a;',
      '}',
      '',
      'inline long um_abs_i(long a) {',
      '  return a < 0 ? long(0ul - ulong(a)) : a;',
      '}',
      '',
      '// 浮動小数点の min / max は比較して選ぶ。どちらかが NaN なら2番目の引数。',
      'inline float um_fmin(float a, float b) {',
      '  return a < b ? a : b;',
      '}',
      '',
      'inline float um_fmax(float a, float b) {',
      '  return a > b ? a : b;',
      '}',
      '',
      '// 浮動小数点から整数へ。範囲外は端の値に飽和し、NaN は 0。変換そのものは範囲内でだけ行う。',
      'inline int um_f2i(float v) {',
      '  if (v != v) return 0;',
      '  if (v >= 2147483648.0f) return 2147483647;',
      '  if (v <= -2147483648.0f) return -2147483647 - 1;',
      '  return int(v);',
      '}',
      '',
      'inline uint um_f2u(float v) {',
      '  if (v != v) return 0u;',
      '  if (v >= 4294967296.0f) return 4294967295u;',
      '  if (v <= 0.0f) return 0u;',
      '  return uint(v);',
      '}',
      '',
      'inline long um_f2l(float v) {',
      '  if (v != v) return 0;',
      '  if (v >= 9223372036854775808.0f) return 9223372036854775807l;',
      '  if (v <= -9223372036854775808.0f) return -9223372036854775807l - 1l;',
      '  return long(v);',
      '}',
      '',
      'inline ulong um_f2ul(float v) {',
      '  if (v != v) return 0ul;',
      '  if (v >= 18446744073709551616.0f) return 18446744073709551615ul;',
      '  if (v <= 0.0f) return 0ul;',
      '  return ulong(v);',
      '}',
      '',
      '// 比較交換。前の値を返す（等しければ desired に入れ替わっている）。弱い版の空振りはやり直す。',
      'inline int um_cas_i(device atomic_int* p, int expected, int desired) {',
      '  int x = expected;',
      '  while (!atomic_compare_exchange_weak_explicit(p, &x, desired, memory_order_relaxed, memory_order_relaxed)) {',
      '    if (x != expected) return x;',
      '    x = expected;',
      '  }',
      '  return expected;',
      '}',
      '',
      'inline uint um_cas_u(device atomic_uint* p, uint expected, uint desired) {',
      '  uint x = expected;',
      '  while (!atomic_compare_exchange_weak_explicit(p, &x, desired, memory_order_relaxed, memory_order_relaxed)) {',
      '    if (x != expected) return x;',
      '    x = expected;',
      '  }',
      '  return expected;',
      '}',
      '',
      ...this.fnText,
      'kernel void um_main(',
      ...args.map((a, i) => `    ${a}${i < args.length - 1 ? ',' : ''}`),
      ') {',
      ...entry,
      ...prelude,
      ...this.lines,
      '}',
      '',
      ...(bufFields.length === 0 ? [] : [
        '// 同じ kernel の、Buffer をポインタの表から受け取る入口。表は引数の領域に置いた MTLBuffer の gpuAddress の並び',
        '// （Buffer の引数の順）。先に提出しておく命令が、使う Buffer を実行の直前に決められる（runtime/metal.m の「郵便受けの列」）。',
        'struct UmBufs {',
        ...bufFields,
        '};',
        '',
        'kernel void um_main_bl(',
        '    constant UmParams& um_params [[buffer(0)]],',
        '    constant UmBufs& um_bufs [[buffer(1)]],',
        ...args.slice(1 + bufFields.length).map((a, i, all) => `    ${a}${i < all.length - 1 ? ',' : ''}`),
        ') {',
        ...bufLocals,
        ...entry,
        ...prelude,
        ...this.lines,
        '}',
        '',
      ]),
    ].join('\n');
  }
}

/**
 * kernel の本体で、アトミック演算（load / store / fetchAdd / compareExchange）を使う `mut RacyBuffer` の引数。
 * これらは GPU 側でアトミックな型の配列として受け取り、要素の読み書きもすべてアトミックにする。
 */
export function atomicParams(d: FnDecl): Set<number> {
  const ids = new Set<number>();
  const cands = new Set(d.params.slice(1).filter((p) => p.mode === 'mut' && p.ty!.k === 'heap' && p.ty!.racy).map((p) => p.sym!.id));
  const ex = (e: Expr | undefined): void => {
    if (!e) return;
    switch (e.kind) {
      case 'mcall':
        if (e.key?.startsWith('Racy.') && e.recv.kind === 'ident' && cands.has(e.recv.sym!.id)) ids.add(e.recv.sym!.id);
        ex(e.recv);
        return e.args.forEach((a) => ex(a.expr));
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
      default:
        return;
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
      default:
        return;
    }
  };
  d.body.forEach(st);
  return ids;
}

export function emitMsl(prog: Program, kernel: FnDecl, groups = false): string {
  return new Emitter(prog).kernel(kernel, groups);
}
