import { CompileError } from './diag.ts';
import type { Pos } from './diag.ts';
import { lex } from './lexer.ts';
import type { Token } from './lexer.ts';
import type { Arg, ClassDecl, ConstDecl, Decl, Expr, FnDecl, Import, InterfaceDecl, MatchArm, Mode, Module, Param, Stmt, StructDecl, TypeNode } from './ast.ts';

const KEYWORDS = new Set([
  'let', 'const', 'function', 'struct', 'if', 'else', 'while', 'for', 'return',
  'break', 'continue', 'true', 'false', 'mut', 'own', 'as', 'export',
  'class', 'constructor', 'new', 'this', 'static', 'private', 'public', 'readonly',
  'enum', 'match', 'null', 'interface', 'implements', 'extends', 'switch', 'case', 'default', 'do',
]);

// class の中で書かれがちだが、まだない（または持たない）もの。
const MEMBER_HINTS: Record<string, [string, string]> = {
  static: ['not-implemented', '`static` members are not implemented yet; use a top-level function'],
  readonly: ['not-implemented', '`readonly` is not implemented yet; remove it (bind the object with `const` to make it immutable)'],
  protected: ['ts-feature', 'there is no inheritance, so there is no `protected`; use `private`'],
  abstract: ['ts-feature', 'there is no inheritance, so there is no `abstract`'],
  get: ['not-implemented', 'getters are not implemented yet; write a normal method'],
  set: ['not-implemented', 'setters are not implemented yet; write a normal method'],
  async: ['not-implemented', '`async` arrives with the GPU phases'],
};

// TypeScriptの習慣で書かれがちだが、この言語にない（またはまだない）構文と、その置き換え先。
const STMT_HINTS: Record<string, [string, string]> = {
  var: ['ts-feature', 'use `let` or `const`'],
  try: ['ts-feature', 'there are no exceptions; errors will be returned as `Result<T, E>` (later phase)'],
  throw: ['ts-feature', 'there are no exceptions; errors will be returned as `Result<T, E>` (later phase)'],
};

const DECL_HINTS: Record<string, [string, string]> = {
  type: ['not-implemented', 'type aliases are not implemented yet; use `struct`'],
  let: ['not-implemented', 'global variables are not available; for a fixed value write a top-level constant (`const NAME: i32 = 64;`), otherwise declare the variable inside a function and pass it as a parameter'],
  var: ['ts-feature', 'global variables are not implemented yet; inside functions use `let` or `const`'],
  async: ['not-implemented', '`async` arrives with the GPU phases'],
};

// 二項演算子の優先順位（大きいほど強く結合）。
const BIN_PREC: Record<string, number> = {
  '??': 0.5, '||': 1, '&&': 2, '|': 3, '^': 4, '&': 5,
  '==': 6, '!=': 6, '===': 6, '!==': 6,
  '<': 7, '<=': 7, '>': 7, '>=': 7,
  '<<': 8, '>>': 8,
  '+': 9, '-': 9,
  '*': 10, '/': 10, '%': 10,
};

const COMPOUND: Record<string, string> = {
  '+=': '+', '-=': '-', '*=': '*', '/=': '/', '%=': '%',
  '&=': '&', '|=': '|', '^=': '^', '<<=': '<<', '>>=': '>>',
};

function containsCall(e: Expr): boolean {
  switch (e.kind) {
    case 'call':
    case 'mcall':
    case 'new':
    case 'newArray':
    case 'dispatch':
    case 'math':
      return true;
    case 'field':
      return containsCall(e.base);
    case 'index':
      return containsCall(e.base) || containsCall(e.index);
    case 'unary':
    case 'cast':
      return containsCall(e.expr);
    case 'cond':
      return containsCall(e.cond) || containsCall(e.then) || containsCall(e.else);
    case 'binary':
      return containsCall(e.left) || containsCall(e.right);
    default:
      return false;
  }
}

class Parser {
  toks: Token[];
  i = 0;

  file?: string;
  imports: Import[] = [];

  constructor(src: string, file?: string) {
    this.toks = lex(src);
    this.file = file;
  }

  get tok(): Token {
    return this.toks[this.i];
  }

  pos(): Pos {
    return this.file ? { line: this.tok.line, col: this.tok.col, file: this.file } : { line: this.tok.line, col: this.tok.col };
  }

  isP(text: string): boolean {
    return this.tok.kind === 'punct' && this.tok.text === text;
  }

  isK(text: string): boolean {
    return this.tok.kind === 'ident' && this.tok.text === text;
  }

  eatP(text: string): boolean {
    if (this.isP(text)) {
      this.i++;
      return true;
    }
    return false;
  }

  eatK(text: string): boolean {
    if (this.isK(text)) {
      this.i++;
      return true;
    }
    return false;
  }

  expectP(text: string): void {
    if (!this.eatP(text)) {
      const found = this.tok.text;
      const hint = found === '++' || found === '--'
        ? `\`${found}\` is a statement, not an expression; write the increment on its own line (\`i${found};\`)`
        : found === '=>' ? 'function values (arrow functions) are not implemented yet; declare a top-level `function`' : undefined;
      throw new CompileError('syntax', `expected '${text}' but found '${found}'`, this.pos(), hint);
    }
  }

  /** 開き括弧から対応する閉じ括弧までを読み飛ばした先が `=>`（または戻り値の型つきの `=>`）なら、アロー関数。 */
  looksLikeArrow(): boolean {
    let depth = 0;
    let j = this.i;
    for (; j < this.toks.length; j++) {
      const t = this.toks[j];
      if (t.kind === 'punct' && t.text === '(') depth++;
      else if (t.kind === 'punct' && t.text === ')' && --depth === 0) break;
      else if (t.kind === 'eof') return false;
    }
    for (let k = j + 1; k < Math.min(j + 8, this.toks.length); k++) {
      const t = this.toks[k].text;
      if (t === '=>') return true;
      if (t === '{' || t === ';' || t === ')' || t === ',') return false;
    }
    return false;
  }

  ident(what: string): string {
    const t = this.tok;
    if (t.kind !== 'ident') {
      throw new CompileError('syntax', `expected ${what} but found '${t.text}'`, this.pos());
    }
    if (KEYWORDS.has(t.text)) {
      throw new CompileError('syntax', `'${t.text}' is a keyword and cannot be used as ${what}`, this.pos());
    }
    if (t.text.startsWith('__')) {
      // コンパイラが enum と match のために作る名前と重ならないようにする。
      throw new CompileError('reserved-name', `names starting with '__' are reserved`, this.pos(), 'pick a name without the leading underscores');
    }
    this.i++;
    return t.text;
  }

  // ---- 宣言 ----

  module(): Module {
    const decls: Decl[] = [];
    while (this.tok.kind !== 'eof') {
      if (this.isK('import')) this.importDecl();
      else decls.push(this.decl());
    }
    return { decls, imports: this.imports };
  }

  /** interface Shape { area(): f64; scale(k: f64): void; } */
  interfaceDecl(pos: Pos): InterfaceDecl {
    const name = this.ident('an interface name');
    if (this.isP('<') || this.isK('extends')) {
      throw new CompileError('not-implemented', 'generic interfaces and interface inheritance are not implemented yet', this.pos());
    }
    this.expectP('{');
    const methods: InterfaceDecl['methods'] = [];
    while (!this.isP('}')) {
      const mpos = this.pos();
      const mname = this.ident('a method name');
      if (!this.isP('(')) {
        throw new CompileError('ts-feature', `an interface lists methods, but '${mname}' is a field`, mpos,
          'for plain data declare a `struct`; an interface here is a promise about methods, e.g. `area(): f64;`');
      }
      this.i++;
      const params: Param[] = [];
      while (!this.isP(')')) {
        const ppos = this.pos();
        const pname = this.ident('a parameter name');
        this.expectP(':');
        let mode: Mode = 'none';
        if (this.eatK('mut')) mode = 'mut';
        else if (this.eatK('own')) mode = 'own';
        params.push({ name: pname, mode, tyNode: this.type(), pos: ppos });
        if (!this.eatP(',')) break;
      }
      this.expectP(')');
      const retNode = this.eatP(':') ? this.type() : undefined;
      methods.push({ name: mname, params, retNode, pos: mpos });
      if (!this.eatP(';')) this.eatP(',');
    }
    this.expectP('}');
    return { kind: 'interface', name, methods, pos };
  }

  /** import { a, b } from "./other.um"; */
  importDecl(): void {
    const pos = this.pos();
    this.i++;
    if (!this.isP('{')) {
      throw new CompileError('ts-feature', 'only named imports are available', pos, 'write `import { name1, name2 } from "./file.um";`');
    }
    this.i++;
    const names: Import['names'] = [];
    while (!this.isP('}')) {
      const npos = this.pos();
      names.push({ name: this.ident('a name to import'), pos: npos });
      if (this.isK('as')) throw new CompileError('not-implemented', 'renaming an import with `as` is not implemented yet', this.pos());
      if (!this.eatP(',')) break;
    }
    this.expectP('}');
    if (!this.eatK('from')) throw new CompileError('syntax', `expected 'from' but found '${this.tok.text}'`, this.pos());
    if (this.tok.kind !== 'str') throw new CompileError('syntax', 'expected the path of a file in quotes', this.pos(), 'e.g. `from "./math.um"`');
    const path = this.tok.text;
    this.i++;
    this.eatP(';');
    this.imports.push({ names, path, pos });
  }

  decl(): Decl {
    const pos = this.pos();
    const exported = this.eatK('export');
    if (this.isK('kernel') && this.toks[this.i + 1]?.text === 'function') {
      this.i += 2;
      // kernel の export は「他のファイルから import できる」という意味だけを持つ。
      return { ...this.fnDecl(pos, false), isKernel: true, importable: exported };
    }
    if (this.eatK('function')) return { ...this.fnDecl(pos, exported), importable: exported };
    if (this.eatK('struct')) return { ...this.structDecl(pos), importable: exported };
    if (this.eatK('enum')) return { ...this.enumDecl(pos), importable: exported };
    if (this.eatK('class')) return { ...this.classDecl(pos), importable: exported };
    if (this.eatK('interface')) return { ...this.interfaceDecl(pos), importable: exported };
    if (this.eatK('const')) return { ...this.constDecl(pos), importable: exported };
    const h = this.tok.kind === 'ident' ? DECL_HINTS[this.tok.text] : undefined;
    if (h) throw new CompileError(h[0], `'${this.tok.text}' is not available here`, this.pos(), h[1]);
    throw new CompileError('syntax', `expected 'function', 'struct', 'class' or 'const' but found '${this.tok.text}'`, this.pos());
  }

  /** const NAME: T = 定数式; */
  constDecl(pos: Pos): ConstDecl {
    const name = this.ident('a constant name');
    if (!this.eatP(':')) {
      throw new CompileError('syntax', `top-level constant '${name}' needs a type annotation`, pos, `write \`const ${name}: i32 = ...;\` (or another type)`);
    }
    const tyNode = this.type();
    if (!this.eatP('=')) {
      throw new CompileError('uninitialized', `constant '${name}' must be initialized`, pos, `a \`const\` needs its value here, e.g. \`const ${name}: i32 = 0;\``);
    }
    const init = this.expr();
    this.eatP(';');
    return { kind: 'const', name, tyNode, init, pos };
  }

  fnDecl(pos: Pos, exported: boolean): FnDecl {
    const name = this.ident('a function name');
    return this.fnRest(pos, name, exported);
  }

  classDecl(pos: Pos): ClassDecl {
    const name = this.ident('a class name');
    const noExtends = () => {
      if (this.isK('extends')) {
        throw new CompileError('ts-feature', '`extends` is not available', this.pos(),
          'classes have no inheritance; put the shared data in a field, share behavior with top-level functions, or declare an `interface` and write `implements`');
      }
    };
    noExtends();
    let typeParams: string[] | undefined;
    if (this.eatP('<')) {
      typeParams = [];
      while (!this.isP('>')) {
        const tname = this.ident('a type parameter name');
        if (typeParams.includes(tname)) throw new CompileError('duplicate', `type parameter '${tname}' is given twice`, pos);
        typeParams.push(tname);
        if (!this.eatP(',')) break;
      }
      this.expectP('>');
      if (typeParams.length === 0) throw new CompileError('syntax', 'expected a type parameter name', pos);
    }
    noExtends();
    let impls: ClassDecl['implements'];
    if (this.eatK('implements')) {
      impls = [];
      for (;;) {
        const ipos = this.pos();
        impls.push({ name: this.ident('an interface name'), pos: ipos });
        if (!this.eatP(',')) break;
      }
    }
    this.expectP('{');
    const cls: ClassDecl = { kind: 'class', name, typeParams, implements: impls, fields: [], methods: [], pos };
    while (!this.isP('}')) {
      if (this.tok.kind === 'eof') throw new CompileError('syntax', "missing '}'", this.pos());
      const mpos = this.pos();
      let isPrivate = false;
      if (this.eatK('private')) isPrivate = true;
      else this.eatK('public');
      const next = this.toks[this.i + 1];
      const h = this.tok.kind === 'ident' && next?.kind === 'ident' ? MEMBER_HINTS[this.tok.text] : undefined;
      if (h) throw new CompileError(h[0], `'${this.tok.text}' is not available here`, mpos, h[1]);

      if (this.eatK('constructor')) {
        if (cls.ctor) throw new CompileError('duplicate', `class '${name}' already has a constructor`, mpos);
        const fn = this.fnRest(mpos, 'constructor', false);
        if (fn.retNode) throw new CompileError('syntax', 'a constructor has no return type', mpos);
        cls.ctor = { ...fn, cls: name, isCtor: true };
        continue;
      }

      const mname = this.ident('a member name');
      if (this.isP('<')) {
        throw new CompileError('not-implemented', 'generic methods are not implemented yet', mpos, 'write a generic top-level function that takes the object as a parameter');
      }
      if (this.isP('(')) {
        cls.methods.push({ ...this.fnRest(mpos, mname, false), cls: name, isPrivate });
        continue;
      }
      if (this.isP('?')) {
        throw new CompileError('ts-feature', 'optional fields are not available', mpos, 'every field holds a value; give it a default instead');
      }
      if (!this.eatP(':')) {
        throw new CompileError('syntax', `field '${mname}' needs a type annotation`, mpos, `write \`${mname}: i32\` (or another type)`);
      }
      const tyNode = this.type();
      const init = this.eatP('=') ? this.expr() : undefined;
      cls.fields.push({ name: mname, tyNode, init, isPrivate, pos: mpos });
      if (!this.eatP(';')) this.eatP(',');
    }
    this.expectP('}');
    return cls;
  }

  /** 関数名の後ろ（引数・戻り値の型・本体）を読む。関数とメソッドで共有する。 */
  fnRest(pos: Pos, name: string, exported: boolean): FnDecl {
    let typeParams: string[] | undefined;
    const typeBounds: Record<string, string> = {};
    if (this.isP('<')) {
      const gpos = this.pos();
      this.i++;
      typeParams = [];
      while (!this.isP('>')) {
        const tname = this.ident('a type parameter name');
        if (typeParams.includes(tname)) throw new CompileError('duplicate', `type parameter '${tname}' is given twice`, gpos);
        if (this.eatK('extends')) {
          typeBounds[tname] = this.ident('an interface name');
        }
        if (this.isP('=')) {
          throw new CompileError('not-implemented', 'defaults on type parameters are not implemented yet', this.pos());
        }
        typeParams.push(tname);
        if (!this.eatP(',')) break;
      }
      this.expectP('>');
      if (typeParams.length === 0) throw new CompileError('syntax', 'expected a type parameter name', gpos);
    }
    this.expectP('(');
    const params: Param[] = [];
    while (!this.isP(')')) {
      const ppos = this.pos();
      if (this.isK('public') || this.isK('private') || this.isK('readonly')) {
        throw new CompileError('ts-feature', 'parameter properties are not available', ppos,
          'declare the field in the class body and assign it in the constructor: `this.x = x;`');
      }
      const pname = this.ident('a parameter name');
      if (this.isP('?')) {
        throw new CompileError('ts-feature', 'optional parameters are not available', ppos, 'pass every argument explicitly');
      }
      if (!this.eatP(':')) {
        throw new CompileError('syntax', `parameter '${pname}' needs a type annotation`, ppos, `write \`${pname}: i32\` (or another type)`);
      }
      let mode: Mode = 'none';
      if (this.eatK('mut')) mode = 'mut';
      else if (this.eatK('own')) mode = 'own';
      params.push({ name: pname, mode, tyNode: this.type(), pos: ppos });
      if (this.isP('=')) {
        throw new CompileError('ts-feature', 'default parameter values are not available', ppos, 'pass every argument explicitly');
      }
      if (!this.eatP(',')) break;
    }
    this.expectP(')');
    const retNode = this.eatP(':') ? this.type() : undefined;
    const body = this.blockBody();
    const bounds = Object.keys(typeBounds).length > 0 ? typeBounds : undefined;
    return { kind: 'fn', name, typeParams, typeBounds: bounds, exported, params, retNode, body, pos };
  }

  /**
   * enum Shape { Circle(r: f32), Rect(w: f32, h: f32), Empty }
   * 内部では struct として持つ: `__tag`（どの選択肢か）と、全選択肢の値を並べたフィールド。
   */
  enumDecl(pos: Pos): StructDecl {
    const name = this.ident('an enum name');
    let typeParams: string[] | undefined;
    if (this.eatP('<')) {
      typeParams = [];
      while (!this.isP('>')) {
        const tname = this.ident('a type parameter name');
        if (typeParams.includes(tname)) throw new CompileError('duplicate', `type parameter '${tname}' is given twice`, pos);
        typeParams.push(tname);
        if (!this.eatP(',')) break;
      }
      this.expectP('>');
      if (typeParams.length === 0) throw new CompileError('syntax', 'expected a type parameter name', pos);
    }
    this.expectP('{');
    const fields: StructDecl['fields'] = [{ name: '__tag', tyNode: { kind: 'name', name: 'u32', pos }, pos }];
    const variants: NonNullable<StructDecl['variants']> = [];
    while (!this.isP('}')) {
      const vpos = this.pos();
      const vname = this.ident('a variant name');
      if (variants.some((x) => x.name === vname)) throw new CompileError('duplicate', `variant '${vname}' is defined twice`, vpos);
      if (this.isP('=')) {
        throw new CompileError('ts-feature', 'enum variants cannot be given numeric values', vpos,
          'variants are names, optionally with values: `enum Shape { Circle(r: f32), Empty }`');
      }
      const names: string[] = [];
      if (this.eatP('(')) {
        while (!this.isP(')')) {
          const fpos = this.pos();
          const fname = this.ident('a value name');
          if (!this.eatP(':')) {
            throw new CompileError('syntax', `value '${fname}' of variant '${vname}' needs a name and a type`, fpos, `write \`${vname}(${fname}: f32)\``);
          }
          names.push(fname);
          fields.push({ name: `${vname}__${fname}`, tyNode: this.type(), pos: fpos });
          if (!this.eatP(',')) break;
        }
        this.expectP(')');
      }
      variants.push({ name: vname, fields: names, pos: vpos });
      if (!this.eatP(',')) this.eatP(';');
    }
    this.expectP('}');
    if (variants.length === 0) throw new CompileError('bad-type', `enum '${name}' has no variants`, pos);
    return { kind: 'struct', name, typeParams, fields, variants, pos };
  }

  switchCount = 0;

  /**
   * switch (x) { case 1: ...; break; case 2: case 3: ...; break; default: ... }
   * if の連なりに置き換える。各 case は break / return / continue で終わること（次の case へ落ちる書き方はできない）。
   */
  switchStmt(pos: Pos): Stmt {
    this.expectP('(');
    const scrut = this.expr();
    this.expectP(')');
    this.expectP('{');
    interface Case { values: Expr[]; isDefault: boolean; body: Stmt[]; pos: Pos }
    const cases: Case[] = [];
    while (!this.isP('}')) {
      const cpos = this.pos();
      const c: Case = { values: [], isDefault: false, body: [], pos: cpos };
      // 続けて書いたラベル（case 1: case 2:）は、同じ本体を共有する。
      for (;;) {
        if (this.isK('case')) {
          this.i++;
          c.values.push(this.expr());
        } else if (this.isK('default')) {
          this.i++;
          c.isDefault = true;
        } else {
          break;
        }
        this.expectP(':');
      }
      if (c.values.length === 0 && !c.isDefault) {
        throw new CompileError('syntax', `expected 'case' or 'default' but found '${this.tok.text}'`, this.pos());
      }
      while (!this.isP('}') && !this.isK('case') && !this.isK('default')) c.body.push(this.stmt());
      // 末尾の break は「switch を抜ける」。if の連なりでは何もしなくてよいので取り除く。
      const last = c.body[c.body.length - 1];
      const ends = (x: Stmt | undefined): boolean =>
        !!x && (x.kind === 'break' || x.kind === 'return' || x.kind === 'continue' || (x.kind === 'block' && ends(x.body[x.body.length - 1])));
      if (!this.isP('}') && !ends(last)) {
        throw new CompileError('fallthrough', 'this `case` runs into the next one', cpos,
          'end it with `break;` (or `return`); to share one body, list the labels together: `case 1: case 2: ...`');
      }
      if (last?.kind === 'break') c.body.pop();
      else if (last?.kind === 'block' && last.body[last.body.length - 1]?.kind === 'break') last.body.pop();
      // 残った break は、外側のループではなく switch を抜けるつもりのもの。置き換えると意味が変わるので受け付けない。
      const stray = (x: Stmt | undefined): Stmt | undefined => {
        if (!x) return undefined;
        if (x.kind === 'break') return x;
        if (x.kind === 'block') return x.body.map(stray).find((y) => y);
        if (x.kind === 'if') return stray(x.then) ?? stray(x.else);
        if (x.kind === 'match') return x.arms.map((a) => stray(a.body)).find((y) => y);
        return undefined; // ループの中の break は、そのループのもの
      };
      const bad = c.body.map(stray).find((y) => y);
      if (bad) {
        throw new CompileError('not-implemented', '`break` in the middle of a `case` is not supported', bad.pos,
          'put the rest of the case in an `else`, so that the only `break` is the last statement of the case');
      }
      cases.push(c);
    }
    this.expectP('}');
    if (cases.filter((c) => c.isDefault).length > 1) throw new CompileError('duplicate', '`default` appears twice', pos);

    // 対象が変数やフィールドならそのまま毎回読む（文字列を移動させないため）。それ以外は一度だけ評価して変数に入れる。
    const pure = (e: Expr): boolean => e.kind === 'ident' || (e.kind === 'field' && pure(e.base));
    const tmp = `__switch${this.switchCount++}`;
    const subject = (): Expr => (pure(scrut) ? structuredClone(scrut) : { kind: 'ident', name: tmp, pos });
    const fallback = cases.find((c) => c.isDefault);
    let chain: Stmt | undefined = fallback ? { kind: 'block', body: fallback.body, pos: fallback.pos } : undefined;
    for (let i = cases.length - 1; i >= 0; i--) {
      const c = cases[i];
      if (c.values.length === 0) continue;
      let cond: Expr | undefined;
      for (const v of c.values) {
        const eq: Expr = { kind: 'binary', op: '==', left: subject(), right: v, pos: v.pos };
        cond = cond ? { kind: 'binary', op: '||', left: cond, right: eq, pos: v.pos } : eq;
      }
      // default のラベルも付いた case（case 1: default:）は、default の本体として既に入っている。
      if (c.isDefault) continue;
      chain = { kind: 'if', cond: cond!, then: { kind: 'block', body: c.body, pos: c.pos }, else: chain, pos: c.pos };
    }
    const body: Stmt[] = [];
    if (!pure(scrut)) body.push({ kind: 'let', name: tmp, isConst: true, init: scrut, pos });
    if (chain) body.push(chain);
    return { kind: 'block', body, pos };
  }

  /** match (x) { Shape.Circle(r) => ..., Shape.Empty => ..., _ => ... } */
  matchStmt(pos: Pos): Stmt {
    this.expectP('(');
    const scrut = this.expr();
    this.expectP(')');
    this.expectP('{');
    const arms: MatchArm[] = [];
    while (!this.isP('}')) {
      const apos = this.pos();
      let arm: MatchArm;
      const first = this.ident('a pattern such as `Shape.Circle(r)` or `_`');
      if (first === '_') {
        arm = { enumName: '', variant: '', binds: [], wildcard: true, body: { kind: 'block', body: [], pos: apos }, pos: apos };
      } else {
        this.expectP('.');
        const variant = this.ident('a variant name');
        const binds: string[] = [];
        if (this.eatP('(')) {
          while (!this.isP(')')) {
            binds.push(this.ident('a name for the value (or `_`)'));
            if (!this.eatP(',')) break;
          }
          this.expectP(')');
        }
        arm = { enumName: first, variant, binds, wildcard: false, body: { kind: 'block', body: [], pos: apos }, pos: apos };
      }
      this.expectP('=>');
      arm.body = this.stmt();
      this.eatP(',');
      arms.push(arm);
    }
    this.expectP('}');
    return { kind: 'match', scrut, arms, pos };
  }

  structDecl(pos: Pos): StructDecl {
    const name = this.ident('a struct name');
    let typeParams: string[] | undefined;
    if (this.eatP('<')) {
      typeParams = [];
      while (!this.isP('>')) {
        const tname = this.ident('a type parameter name');
        if (typeParams.includes(tname)) throw new CompileError('duplicate', `type parameter '${tname}' is given twice`, pos);
        typeParams.push(tname);
        if (!this.eatP(',')) break;
      }
      this.expectP('>');
      if (typeParams.length === 0) throw new CompileError('syntax', 'expected a type parameter name', pos);
    }
    this.expectP('{');
    const fields: StructDecl['fields'] = [];
    while (!this.isP('}')) {
      const fpos = this.pos();
      const fname = this.ident('a field name');
      this.expectP(':');
      fields.push({ name: fname, tyNode: this.type(), pos: fpos });
      if (!this.eatP(';')) this.eatP(',');
    }
    this.expectP('}');
    return { kind: 'struct', name, typeParams, fields, pos };
  }

  type(): TypeNode {
    const pos = this.pos();
    const t = this.typeNoNull();
    if (!this.isP('|')) return t;
    this.i++;
    if (!this.eatK('null')) {
      throw new CompileError('ts-feature', 'union types are not available', this.pos(),
        'only `T | null` is available; for a choice between several shapes, declare an `enum` with values');
    }
    return { kind: 'opt', inner: t, pos };
  }

  /** 型引数の終わりの `>` を読む。`>>` と続けて書かれた場合は、1つずつに分けて読む。 */
  closeAngle(): void {
    if (this.tok.kind === 'punct' && this.tok.text === '>>') {
      this.toks[this.i] = { ...this.tok, text: '>', col: this.tok.col + 1 };
      return;
    }
    if (!this.eatP('>')) {
      throw new CompileError('syntax', `expected '>' but found '${this.tok.text}'`, this.pos());
    }
  }

  typeNoNull(): TypeNode {
    const pos = this.pos();
    if (this.isK('null')) {
      throw new CompileError('syntax', 'write the type first: `T | null`', pos);
    }
    // かっこで囲んだ型: `(string | null)[]`
    const grouped = this.eatP('(') ? this.type() : null;
    if (grouped) this.expectP(')');
    const name = grouped ? '' : this.ident('a type name');
    let t: TypeNode = grouped ?? { kind: 'name', name, pos };
    if (grouped) {
      // 型引数は付かない。
    } else if (this.isP('<') && name !== 'Array' && name !== 'Buffer' && name !== 'RacyBuffer') {
      // ジェネリックな struct: Pair<i32, f64>
      this.i++;
      const args: TypeNode[] = [];
      for (;;) {
        args.push(this.type());
        if (!this.eatP(',')) break;
      }
      this.closeAngle();
      t = { kind: 'generic', name, args, pos };
    } else if (this.isP('<')) {
      this.i++;
      const elem = this.type();
      this.closeAngle();
      t = name === 'RacyBuffer' ? { kind: 'heap', elem, pos, buffer: true, racy: true }
        : name === 'Buffer' ? { kind: 'heap', elem, pos, buffer: true } : { kind: 'heap', elem, pos };
    }
    while (this.eatP('[')) {
      if (this.eatP(']')) {
        // TypeScriptと同じく、T[] は Array<T> の別の書き方。
        t = { kind: 'heap', elem: t, pos };
        continue;
      }
      const lt = this.tok;
      if (lt.kind === 'ident' && !KEYWORDS.has(lt.text) && this.toks[this.i + 1]?.text === ']') {
        // 長さを最上位の定数の名前で書く: `T[N]`。値は型検査が求める。
        this.i += 2;
        t = { kind: 'array', elem: t, len: 0, lenName: lt.text, pos };
        continue;
      }
      if (lt.kind !== 'int') {
        throw new CompileError('syntax', 'array length must be an integer literal or the name of a top-level constant', this.pos(),
          'write `T[16]` or `T[N]` with `const N: usize = 16;` at the top level; for a length known only at run time use `T[]`');
      }
      this.i++;
      const len = Number(lt.text);
      if (!Number.isSafeInteger(len) || len <= 0) {
        throw new CompileError('bad-type', 'array length must be a positive integer', pos);
      }
      this.expectP(']');
      t = { kind: 'array', elem: t, len, pos };
    }
    return t;
  }

  // ---- 文 ----

  blockBody(): Stmt[] {
    this.expectP('{');
    const body: Stmt[] = [];
    while (!this.isP('}')) {
      if (this.tok.kind === 'eof') throw new CompileError('syntax', "missing '}'", this.pos());
      body.push(this.stmt());
    }
    this.expectP('}');
    return body;
  }

  stmt(): Stmt {
    const pos = this.pos();
    if (this.isP('{')) return { kind: 'block', body: this.blockBody(), pos };
    if (this.eatP(';')) return { kind: 'block', body: [], pos };

    if (this.isK('let') || this.isK('const')) {
      const s = this.letStmt();
      this.eatP(';');
      return s;
    }
    if (this.eatK('match')) return this.matchStmt(pos);
    if (this.isK('do')) {
      // do { ... } while (c);  →  { let first = true; while (first || c) { first = false; ... } }
      // 条件は本体のあとで評価され、continue も条件へ進む。
      this.i++;
      const body = this.stmt();
      if (!this.eatK('while')) throw new CompileError('syntax', `expected 'while' after the body of 'do' but found '${this.tok.text}'`, this.pos());
      this.expectP('(');
      const cond = this.expr();
      this.expectP(')');
      this.eatP(';');
      const flag = `__do${this.switchCount++}`;
      const first = (): Expr => ({ kind: 'ident', name: flag, pos });
      const clear: Stmt = { kind: 'assign', target: first(), value: { kind: 'bool', value: false, pos }, pos };
      return {
        kind: 'block', pos, body: [
          { kind: 'let', name: flag, isConst: false, init: { kind: 'bool', value: true, pos }, pos },
          { kind: 'while', cond: { kind: 'binary', op: '||', left: first(), right: cond, pos }, body: { kind: 'block', body: [clear, body], pos }, pos },
        ],
      };
    }
    if (this.isK('switch')) {
      this.i++;
      return this.switchStmt(pos);
    }
    if (this.eatK('if')) {
      this.expectP('(');
      const cond = this.expr();
      this.expectP(')');
      const then = this.stmt();
      const els = this.eatK('else') ? this.stmt() : undefined;
      return { kind: 'if', cond, then, else: els, pos };
    }
    if (this.eatK('while')) {
      this.expectP('(');
      const cond = this.expr();
      this.expectP(')');
      return { kind: 'while', cond, body: this.stmt(), pos };
    }
    if (this.eatK('for')) return this.forStmt(pos);
    if (this.eatK('return')) {
      const value = this.isP(';') || this.isP('}') ? undefined : this.expr();
      this.eatP(';');
      return { kind: 'return', value, pos };
    }
    if (this.eatK('break')) {
      this.eatP(';');
      return { kind: 'break', pos };
    }
    if (this.eatK('continue')) {
      this.eatP(';');
      return { kind: 'continue', pos };
    }
    const h = this.tok.kind === 'ident' ? STMT_HINTS[this.tok.text] : undefined;
    if (h) throw new CompileError(h[0], `'${this.tok.text}' is not available`, pos, h[1]);
    if (this.isK('function')) {
      throw new CompileError('not-implemented', 'nested functions are not implemented yet', pos, 'declare the function at the top level');
    }

    const s = this.simpleStmt();
    this.eatP(';');
    return s;
  }

  letStmt(): Stmt {
    const pos = this.pos();
    const isConst = this.tok.text === 'const';
    this.i++;
    const name = this.ident('a variable name');
    const tyAnn = this.eatP(':') ? this.type() : undefined;
    if (!this.eatP('=')) {
      // `let x: T;` はゼロで初期化する（使える型かどうかは型検査で確かめる）。
      if (tyAnn && !isConst) return { kind: 'let', name, isConst, tyAnn, pos };
      throw new CompileError('uninitialized', `variable '${name}' must be initialized`, pos,
        isConst ? 'a `const` needs its value here, e.g. `const x: i32 = 0`' : 'give it a value (`let x = 0`) or a type (`let x: i32;` starts at zero)');
    }
    return { kind: 'let', name, isConst, tyAnn, init: this.expr(), pos };
  }

  /** 代入・複合代入・++/--・式文。 */
  simpleStmt(): Stmt {
    const pos = this.pos();
    const target = this.expr();
    if (this.eatP('=')) {
      return { kind: 'assign', target, value: this.expr(), pos };
    }
    const op = this.tok.kind === 'punct' ? this.tok.text : '';
    if (op in COMPOUND || op === '++' || op === '--') {
      if (containsCall(target)) {
        throw new CompileError('compound-call', `'${op}' on a target that contains a function call is not allowed`, pos,
          'store the call result in a `const` first, then update through it');
      }
      this.i++;
      const rhs: Expr = op in COMPOUND ? this.expr() : { kind: 'int', value: 1n, pos };
      const bop = op in COMPOUND ? COMPOUND[op] : op === '++' ? '+' : '-';
      return { kind: 'assign', target, value: { kind: 'binary', op: bop, left: target, right: rhs, pos }, pos };
    }
    return { kind: 'expr', expr: target, pos };
  }

  forStmt(pos: Pos): Stmt {
    this.expectP('(');
    let init: Stmt | undefined;
    if (!this.isP(';')) {
      if (this.isK('let') || this.isK('const')) {
        const save = this.i;
        this.i++;
        const isConst = this.toks[save].text === 'const';
        if (this.tok.kind === 'ident' && this.toks[this.i + 1]?.text === 'in') {
          throw new CompileError('ts-feature', '`for...in` is not available', pos,
            'use `for (const x of xs)` for elements, or an index loop: `for (let i = 0; i < xs.length; i++)`');
        }
        if (this.tok.kind === 'ident' && this.toks[this.i + 1]?.text === 'of') {
          const name = this.ident('a variable name');
          this.i++;
          const iter = this.expr();
          this.expectP(')');
          return { kind: 'forOf', name, isConst, iter, body: this.stmt(), pos };
        }
        this.i = save;
        init = this.letStmt();
      } else {
        init = this.simpleStmt();
      }
    }
    this.expectP(';');
    const cond = this.isP(';') ? undefined : this.expr();
    this.expectP(';');
    const step = this.isP(')') ? undefined : this.simpleStmt();
    this.expectP(')');
    return { kind: 'for', init, cond, step, body: this.stmt(), pos };
  }

  // ---- 式 ----

  expr(): Expr {
    const pos = this.pos();
    const cond = this.binary(0);
    if (!this.eatP('?')) return cond;
    // 条件式。then の側は ':' まで、else の側は右結合で読む。
    const then = this.expr();
    this.expectP(':');
    return { kind: 'cond', cond, then, else: this.expr(), pos };
  }

  binary(minPrec: number): Expr {
    let left = this.unary();
    for (;;) {
      const t = this.tok;
      const prec = t.kind === 'punct' ? BIN_PREC[t.text] : undefined;
      if (prec === undefined || prec <= minPrec) return left;
      this.i++;
      const right = this.binary(prec);
      // TypeScriptの === / !== はそのまま == / != として受け付ける。
      const op = t.text === '===' ? '==' : t.text === '!==' ? '!=' : t.text;
      left = { kind: 'binary', op, left, right, pos: { line: t.line, col: t.col } };
    }
  }

  /** 前置演算子つきの項に `as` を後置する。`-x as T` は `(-x) as T`。 */
  unary(): Expr {
    const pos = this.pos();
    let e = this.prefix();
    while (this.eatK('as')) {
      e = { kind: 'cast', expr: e, to: this.type(), pos };
    }
    return e;
  }

  prefix(): Expr {
    const pos = this.pos();
    if (this.eatP('-')) {
      const e = this.prefix();
      // リテラルの符号は畳み込む（-128 を i8 に収めるため）。
      if (e.kind === 'int') return { kind: 'int', value: -e.value, pos };
      if (e.kind === 'float') return { kind: 'float', value: -e.value, pos };
      return { kind: 'unary', op: '-', expr: e, pos };
    }
    if (this.eatP('!')) return { kind: 'unary', op: '!', expr: this.prefix(), pos };
    if (this.eatP('~')) return { kind: 'unary', op: '~', expr: this.prefix(), pos };
    if (this.eatP('+')) return this.prefix();
    return this.postfix();
  }

  postfix(): Expr {
    let e = this.primary();
    for (;;) {
      const pos = this.pos();
      if (this.eatP('.')) {
        const name = this.ident('a field name');
        if (this.eatP('(')) {
          e = { kind: 'mcall', recv: e, name, args: this.args(), pos };
        } else {
          e = { kind: 'field', base: e, name, pos };
        }
      } else if (this.eatP('[')) {
        const index = this.expr();
        this.expectP(']');
        e = { kind: 'index', base: e, index, pos };
      } else {
        return e;
      }
    }
  }

  /** 呼び出しの引数列。開き括弧は読み終えている前提。 */
  args(): Arg[] {
    const args: Arg[] = [];
    while (!this.isP(')')) {
      let mode: Mode = 'none';
      if (this.eatK('mut')) mode = 'mut';
      else if (this.eatK('own')) mode = 'own';
      args.push({ mode, expr: this.expr() });
      if (!this.eatP(',')) break;
    }
    this.expectP(')');
    return args;
  }

  primary(): Expr {
    const t = this.tok;
    const pos = this.pos();

    if (t.kind === 'int') {
      this.i++;
      return { kind: 'int', value: BigInt(t.text), pos };
    }
    if (t.kind === 'float') {
      this.i++;
      return { kind: 'float', value: Number(t.text), pos };
    }
    if (t.kind === 'str') {
      this.i++;
      return { kind: 'str', value: t.text, pos };
    }
    if (this.eatK('null')) return { kind: 'null', pos };
    if (this.eatK('true')) return { kind: 'bool', value: true, pos };
    if (this.eatK('false')) return { kind: 'bool', value: false, pos };

    if (this.isP('(') && this.looksLikeArrow()) {
      throw new CompileError('not-implemented', 'function values (arrow functions) are not implemented yet', pos,
        'declare a top-level `function` and call it by name; for arrays, write the loop with `for (const x of xs)`');
    }
    if (this.eatP('(')) {
      const e = this.expr();
      this.expectP(')');
      return e;
    }

    if (this.eatP('[')) {
      const elems: Expr[] = [];
      while (!this.isP(']')) {
        elems.push(this.expr());
        if (!this.eatP(',')) break;
      }
      this.expectP(']');
      return { kind: 'arrayLit', elems, pos };
    }

    if (this.eatP('{')) {
      const fields: { name: string; value: Expr; pos: Pos }[] = [];
      while (!this.isP('}')) {
        const fpos = this.pos();
        const name = this.ident('a field name');
        const value: Expr = this.eatP(':') ? this.expr() : { kind: 'ident', name, pos: fpos };
        fields.push({ name, value, pos: fpos });
        if (!this.eatP(',')) break;
      }
      this.expectP('}');
      return { kind: 'object', fields, pos };
    }

    if (t.kind === 'ident') {
      if (t.text === 'null' || t.text === 'undefined') {
        throw new CompileError('ts-feature', `'${t.text}' is not available`, pos,
          '`undefined` does not exist and `T | null` is not implemented yet; give every variable a real value');
      }
      if (this.eatK('this')) return { kind: 'ident', name: 'this', pos };
      if (this.eatK('new')) {
        if ((this.isK('Array') || this.isK('Buffer') || this.isK('RacyBuffer')) && this.toks[this.i + 1]?.text === '<') {
          const tn = this.type();
          if (tn.kind !== 'heap') throw new CompileError('syntax', 'expected `new Array<T>(...)` or `new Buffer<T>(n)`', pos);
          this.expectP('(');
          return { kind: 'newArray', elemNode: tn.elem, args: this.args(), buffer: tn.buffer, racy: tn.racy, pos };
        }
        const cls = this.ident('a class name');
        let typeArgs: TypeNode[] | undefined;
        if (this.eatP('<')) {
          // new Stack<i32>()
          typeArgs = [];
          for (;;) {
            typeArgs.push(this.type());
            if (!this.eatP(',')) break;
          }
          this.closeAngle();
        }
        this.expectP('(');
        return { kind: 'new', cls, typeArgs, args: this.args(), pos };
      }
      const name = this.ident('an expression');
      if (this.eatP('(')) return { kind: 'call', callee: name, args: this.args(), pos };
      if (this.isP('=>')) {
        throw new CompileError('not-implemented', 'function values (arrow functions) are not implemented yet', pos,
          'declare a top-level `function` and call it by name; for arrays, write the loop with `for (const x of xs)`');
      }
      return { kind: 'ident', name, pos };
    }

    throw new CompileError('syntax', `unexpected '${t.text}'`, pos);
  }
}

export function parse(src: string, file?: string): Module {
  return new Parser(src, file).module();
}
