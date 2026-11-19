// 診断（エラー）の共通定義。
// AIが自分で直せるよう、位置・原因コード・修正のヒントを必ず構造化して持つ。

export interface Pos {
  line: number;
  col: number;
  /** import したファイルの中の位置なら、そのファイルの名前。なければ最初のファイル。 */
  file?: string;
}

export interface Diagnostic {
  code: string;
  message: string;
  file: string;
  line: number;
  col: number;
  hint?: string;
}

export class CompileError extends Error {
  code: string;
  pos: Pos;
  hint?: string;

  constructor(code: string, message: string, pos: Pos, hint?: string) {
    super(message);
    this.code = code;
    this.pos = pos;
    this.hint = hint;
  }

  toDiagnostic(file: string): Diagnostic {
    const d: Diagnostic = {
      code: this.code,
      message: this.message,
      file: this.pos.file ?? file,
      line: this.pos.line,
      col: this.pos.col,
    };
    if (this.hint) d.hint = this.hint;
    return d;
  }
}

/** 複数の診断をまとめて投げるための例外。 */
export class CompileErrors extends Error {
  errors: CompileError[];

  constructor(errors: CompileError[]) {
    super(errors.map((e) => e.message).join('\n'));
    this.errors = errors;
  }
}

export function formatDiagnostic(d: Diagnostic): string {
  let s = `${d.file}:${d.line}:${d.col}: error[${d.code}]: ${d.message}`;
  if (d.hint) s += `\n  hint: ${d.hint}`;
  return s;
}
