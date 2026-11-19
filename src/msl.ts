// kernel 関数を MSL（Metal Shading Language）に変換する。
//
// CPU 用のコード生成（codegen.ts）や WGSL（wgsl.ts）と同じ結果になるように、次を明示的に生成する:
//   - 添字は最後の要素に切り詰める（um_clamp）。長さ 0 の Buffer は1要素ぶん確保されている
//   - シフト量はビット幅で割った余り
//   - 整数のゼロ除算は x / 0 = x、x % 0 = 0、最小値 / -1 = 最小値（x % -1 = 0）。MSL（C++）では未定義なので関数で包む
//   - 符号つき整数の + - * と符号反転は、符号なしにしてから計算して戻す（ラップ。C++ の符号つきオーバーフローは未定義）
//   - 浮動小数点から整数への変換は um_f2i / um_f2u（範囲外は飽和、NaN は 0）
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
        // 初期値のない宣言は、ゼロで初期化する（T{}）。
        this.line(`${this.ty(s.sym!.ty)} ${this.name(s.sym!)} = ${s.init ? this.expr(s.init) : `${this.ty(s.sym!.ty)}{}`};`);
        return;

      case 'assign':
        this.line(`${this.expr(s.target)} = ${this.expr(s.value)};`);
        return;

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
          elem = (i) => `${ref.ptr}[${i}]`;
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

      case 'expr':
        // 副作用があるのは関数呼び出しだけ。補助関数は値を返すだけなので、結果は捨てる。
        if (s.expr.kind === 'call') this.line(`(void)${this.expr(s.expr)};`);
        return;
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

  /** 添字を uint にする。負の int は大きな値になり、切り詰めで最後の要素を指す（CPU 版と同じ）。 */
  index(e: Expr): string {
    const t = e.ty as Ty & { k: 'int' };
    return t.signed ? `uint(${this.expr(e)})` : this.expr(e);
  }

  expr(e: Expr): string {
    const ty = e.ty!;
    switch (e.kind) {
      case 'int':
        if (ty.k === 'float') return this.float(Number(e.value));
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
          return ty.signed ? `int(0u - uint(${this.expr(e.expr)}))` : `(0u - ${this.expr(e.expr)})`;
        }
        return `(${e.op}${this.expr(e.expr)})`;

      case 'binary': {
        const l = this.expr(e.left);
        const r = this.expr(e.right);
        const lt = e.left.ty!;
        const op = e.op;
        if (lt.k === 'int') {
          const signed = lt.signed;
          switch (op) {
            case '+':
            case '-':
            case '*':
              return signed ? `int(uint(${l}) ${op} uint(${r}))` : `(${l} ${op} ${r})`;
            case '/':
              return `${signed ? 'um_div_i' : 'um_div_u'}(${l}, ${r})`;
            case '%':
              return `${signed ? 'um_rem_i' : 'um_rem_u'}(${l}, ${r})`;
            case '<<':
            case '>>': {
              const rt = e.right.ty as Ty & { k: 'int' };
              const amt = `(${rt.signed ? `uint(${r})` : r} & 31u)`;
              // 符号つきの左シフトは、符号なしでずらしてから戻す。
              if (op === '<<' && signed) return `int(uint(${l}) << ${amt})`;
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
        if (from.k === 'float' && ty.k === 'int') return `${ty.signed ? 'um_f2i' : 'um_f2u'}(${x})`;
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
          return [ref.ptr, ref.len];
        });
        return `${fn}(${args.join(', ')})`;
      }

      case 'math': {
        const a = e.args.map((x) => this.expr(x));
        const cast = (s: string) => `${this.ty(ty)}(${s})`;
        switch (e.fn) {
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
            // マスクつきシフト。kernel は 32 ビットなのでシフト量を 31 で切る。
            const st = e.args[1].ty as Ty & { k: 'int' };
            const amt = `(${st.signed ? `uint(${a[1]})` : a[1]} & 31u)`;
            const vt = e.args[0].ty as Ty & { k: 'int' };
            if (e.fn === 'shl' && vt.signed) return cast(`int(uint(${a[0]}) << ${amt})`);
            return cast(`(${a[0]} ${e.fn === 'shl' ? '<<' : '>>'} ${amt})`);
          }
        }
        throw new Error(`internal: msl Math.${e.fn}`);
      }

      default:
        throw new Error(`internal: msl expression ${e.kind}`);
    }
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

  /** dispatch される kernel 全体を出力する。 */
  kernel(d: FnDecl): string {
    const paramFields = ['  uint count;', '  uint row;'];
    const args: string[] = ['constant UmParams& um_params [[buffer(0)]]'];
    const prelude: string[] = [];
    let binding = 0;

    d.params.slice(1).forEach((p) => {
      const ty = p.ty!;
      if (ty.k === 'heap') {
        binding++;
        paramFields.push(`  uint len_${p.name};`);
        args.push(`${p.mode === 'mut' ? '' : 'const '}device ${this.ty(ty.elem)}* b_${p.name} [[buffer(${binding})]]`);
        this.bufs.set(p.sym!.id, { ptr: `b_${p.name}`, len: `um_params.len_${p.name}` });
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
    args.push('uint3 gid [[thread_position_in_grid]]');

    this.names.set(d.params[0].sym!.id, 'um_i');
    for (const s of d.body) this.stmt(s);

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
      '// 最小値の絶対値は最小値のまま（ラップ）。',
      'inline int um_abs_i(int a) {',
      '  return a < 0 ? int(0u - uint(a)) : a;',
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
      ...this.fnText,
      'kernel void um_main(',
      ...args.map((a, i) => `    ${a}${i < args.length - 1 ? ',' : ''}`),
      ') {',
      '  uint um_i = gid.y * um_params.row + gid.x;',
      '  if (um_i >= um_params.count) {',
      '    return;',
      '  }',
      ...prelude,
      ...this.lines,
      '}',
      '',
    ].join('\n');
  }
}

export function emitMsl(prog: Program, kernel: FnDecl): string {
  return new Emitter(prog).kernel(kernel);
}
