import { CompileError } from './diag.ts';

export type TokKind = 'ident' | 'int' | 'float' | 'str' | 'punct' | 'eof';

export interface Token {
  kind: TokKind;
  text: string;
  line: number;
  col: number;
}

// 長いものから順に照合する。
const PUNCTS = [
  '===', '!==', '<<=', '>>=', '...',
  '=>', '==', '!=', '??', '<=', '>=', '&&', '||', '<<', '>>',
  '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '++', '--',
  '{', '}', '(', ')', '[', ']', ';', ':', ',', '.', '=',
  '+', '-', '*', '/', '%', '<', '>', '!', '&', '|', '^', '~', '?',
];

const isDigit = (c: string) => c >= '0' && c <= '9';
const isIdentStart = (c: string) =>
  (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c === '_' || c === '$';
const isIdentPart = (c: string) => isIdentStart(c) || isDigit(c);

export function lex(src: string, startLine = 1, startCol = 1): Token[] {
  const toks: Token[] = [];
  let i = 0;
  let line = startLine;
  let col = startCol;

  // 文字列の中の \ で始まる並びを1文字ぶん読み、表す文字を返す。
  const escape = (pos: { line: number; col: number }): string => {
    advance(1); // バックスラッシュ
    const c = src[i];
    const simple: Record<string, string> = { n: '\n', t: '\t', r: '\r', '0': '\0', '\\': '\\', "'": "'", '"': '"', '`': '`', $: '$' };
    if (c in simple) {
      advance(1);
      return simple[c];
    }
    const hex = (digits: string): string => {
      const v = /^[0-9a-fA-F]+$/.test(digits) ? parseInt(digits, 16) : NaN;
      if (Number.isNaN(v) || v > 0x10ffff || (v >= 0xd800 && v <= 0xdfff)) {
        throw new CompileError('bad-string', 'invalid escape sequence in string', pos, 'use \\u{...} with a Unicode code point, e.g. \\u{3042}');
      }
      return String.fromCodePoint(v);
    };
    if (c === 'x') {
      const d = src.slice(i + 1, i + 3);
      advance(3);
      return hex(d.length === 2 ? d : '');
    }
    if (c === 'u' && src[i + 1] === '{') {
      const end = src.indexOf('}', i);
      const d = end < 0 ? '' : src.slice(i + 2, end);
      advance(end < 0 ? 2 : end - i + 1);
      return hex(d.length <= 6 ? d : '');
    }
    if (c === 'u') {
      const d = src.slice(i + 1, i + 5);
      advance(5);
      return hex(d.length === 4 ? d : '');
    }
    throw new CompileError('bad-string', `unknown escape sequence '\\${c ?? ''}'`, pos);
  };

  const advance = (n: number) => {
    for (let k = 0; k < n; k++) {
      if (src[i] === '\n') {
        line++;
        col = 1;
      } else {
        col++;
      }
      i++;
    }
  };

  while (i < src.length) {
    const c = src[i];

    if (c === ' ' || c === '\t' || c === '\r' || c === '\n') {
      advance(1);
      continue;
    }
    if (c === '/' && src[i + 1] === '/') {
      while (i < src.length && src[i] !== '\n') advance(1);
      continue;
    }
    if (c === '/' && src[i + 1] === '*') {
      const start = { line, col };
      advance(2);
      while (i < src.length && !(src[i] === '*' && src[i + 1] === '/')) advance(1);
      if (i >= src.length) throw new CompileError('unterminated-comment', 'unterminated block comment', start);
      advance(2);
      continue;
    }

    const pos = { line, col };

    if (isIdentStart(c)) {
      let j = i;
      while (j < src.length && isIdentPart(src[j])) j++;
      toks.push({ kind: 'ident', text: src.slice(i, j), ...pos });
      advance(j - i);
      continue;
    }

    if (isDigit(c)) {
      let j = i;
      let isFloat = false;
      if (c === '0' && (src[j + 1] === 'x' || src[j + 1] === 'X')) {
        j += 2;
        while (j < src.length && /[0-9a-fA-F_]/.test(src[j])) j++;
      } else if (c === '0' && (src[j + 1] === 'b' || src[j + 1] === 'B')) {
        j += 2;
        while (j < src.length && /[01_]/.test(src[j])) j++;
      } else {
        while (j < src.length && (isDigit(src[j]) || src[j] === '_')) j++;
        if (src[j] === '.' && isDigit(src[j + 1] ?? '')) {
          isFloat = true;
          j++;
          while (j < src.length && (isDigit(src[j]) || src[j] === '_')) j++;
        }
        if (src[j] === 'e' || src[j] === 'E') {
          let k = j + 1;
          if (src[k] === '+' || src[k] === '-') k++;
          if (isDigit(src[k] ?? '')) {
            isFloat = true;
            j = k;
            while (j < src.length && isDigit(src[j])) j++;
          }
        }
      }
      if ((src[i + 1] === 'x' || src[i + 1] === 'X' || src[i + 1] === 'b' || src[i + 1] === 'B') && c === '0'
        && src.slice(i + 2, j).replace(/_/g, '') === '') {
        throw new CompileError('bad-number', `invalid number literal '${src.slice(i, j)}'`, pos, 'a hexadecimal or binary literal needs digits after the prefix, e.g. 0xff or 0b1010');
      }
      if (j < src.length && isIdentStart(src[j])) {
        throw new CompileError('bad-number', `invalid number literal '${src.slice(i, j + 1)}'`, pos,
          'numeric literals take no suffix; write the type with an annotation or `as`, e.g. `1 as u8`');
      }
      toks.push({ kind: isFloat ? 'float' : 'int', text: src.slice(i, j).replace(/_/g, ''), ...pos });
      advance(j - i);
      continue;
    }

    if (c === '"' || c === "'") {
      advance(1);
      let text = '';
      for (;;) {
        if (i >= src.length || src[i] === '\n') throw new CompileError('bad-string', 'unterminated string literal', pos);
        if (src[i] === c) break;
        if (src[i] === '\\') {
          text += escape(pos);
        } else {
          text += src[i];
          advance(1);
        }
      }
      advance(1);
      toks.push({ kind: 'str', text, ...pos });
      continue;
    }

    if (c === '`') {
      // テンプレート文字列 `a${x}b` は ("a" + String(x) + "b") として読む。
      const punct = (text: string) => toks.push({ kind: 'punct', text, ...pos });
      advance(1);
      punct('(');
      let text = '';
      for (;;) {
        if (i >= src.length) throw new CompileError('bad-string', 'unterminated template string', pos);
        if (src[i] === '`') break;
        if (src[i] === '\\') {
          text += escape(pos);
        } else if (src[i] === '$' && src[i + 1] === '{') {
          toks.push({ kind: 'str', text, ...pos });
          text = '';
          advance(2);
          const at = { line, col };
          const from = i;
          let depth = 1;
          while (i < src.length && depth > 0) {
            const ch = src[i];
            if (ch === '`') {
              throw new CompileError('not-implemented', 'a template string inside `${...}` is not supported', at, 'compute the inner string in a variable first');
            }
            if (ch === '"' || ch === "'") {
              advance(1);
              while (i < src.length && src[i] !== ch && src[i] !== '\n') advance(src[i] === '\\' ? 2 : 1);
            } else if (ch === '{') {
              depth++;
            } else if (ch === '}') {
              depth--;
              if (depth === 0) break;
            }
            advance(1);
          }
          if (depth !== 0) throw new CompileError('bad-string', 'unterminated `${` in template string', at);
          const inner = lex(src.slice(from, i), at.line, at.col);
          inner.pop(); // eof
          if (inner.length === 0) throw new CompileError('bad-string', 'empty `${}` in template string', at);
          advance(1); // }
          punct('+');
          toks.push({ kind: 'ident', text: 'String', ...pos });
          punct('(');
          toks.push(...inner);
          punct(')');
          punct('+');
        } else {
          text += src[i];
          advance(1);
        }
      }
      advance(1);
      toks.push({ kind: 'str', text, ...pos });
      punct(')');
      continue;
    }

    const p = PUNCTS.find((s) => src.startsWith(s, i));
    if (!p) throw new CompileError('bad-char', `unexpected character '${c}'`, pos);
    toks.push({ kind: 'punct', text: p, ...pos });
    advance(p.length);
  }

  toks.push({ kind: 'eof', text: '<eof>', line, col });
  return toks;
}
