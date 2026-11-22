#!/usr/bin/env node
// wasm を Node で実行する: node runtime/run-node.mjs out.wasm [関数名] [-- 引数...]
// `--` のあとの引数は、プログラムの args() に渡す。
// 標準入力の行は readLine() に、ファイルの読み書きは readFile() / writeFile() にそのままつなぐ
// （ネイティブのバイナリと同じく、プログラムが指定したファイルを読み書きできる）。
// 開いたままのファイル・フォルダ・cwd / exePath（exePath は .wasm の場所）・absPath・環境変数（envVar）もつなぐ。
import {
  appendFileSync, closeSync, fstatSync, mkdirSync, openSync, readdirSync, readFileSync, readSync, statSync, writeFileSync, writeSync,
} from 'node:fs';
import { resolve } from 'node:path';
import { instantiate, UmPanic } from './host.mjs';

const argv = process.argv.slice(2);
const dash = argv.indexOf('--');
const [file, fn = 'main'] = dash >= 0 ? argv.slice(0, dash) : argv;
const programArgs = dash >= 0 ? argv.slice(dash + 1) : [];
if (!file) {
  console.error('usage: node runtime/run-node.mjs <file.wasm> [function] [-- args...]');
  process.exit(2);
}

// 標準入力を同期的に読む。読んだ分はここにためて、1行ずつ渡す。
let input = Buffer.alloc(0);
let inputEnded = false;
function fillInput() {
  if (inputEnded) return false;
  const chunk = Buffer.alloc(65536);
  let n = 0;
  try {
    n = readSync(0, chunk, 0, chunk.length, null);
  } catch (e) {
    // 端末やパイプが「いまは読めない」と答えたときは、もう一度試す。それ以外は入力の終わりとして扱う。
    if (e.code === 'EAGAIN' || e.code === 'EINTR') return true;
    n = 0;
  }
  if (n <= 0) {
    inputEnded = true;
    return false;
  }
  input = Buffer.concat([input, chunk.subarray(0, n)]);
  return true;
}
function readLine() {
  let from = 0;
  for (;;) {
    const nl = input.indexOf(10, from);
    if (nl >= 0) {
      const line = input.subarray(0, nl);
      input = input.subarray(nl + 1);
      return new Uint8Array(line);
    }
    from = input.length;
    if (!fillInput()) {
      if (input.length === 0) return null;
      const line = input;
      input = Buffer.alloc(0);
      return new Uint8Array(line);
    }
  }
}

const exports = await instantiate(readFileSync(file), {
  readLine,
  // Node では「待たずに読めるか」を同期的に確かめる方法がないので、すでに読み込んである分だけで答える
  // （1行ぶんたまっているか、入力が終わっていれば true）。
  inputReady: () => inputEnded || input.includes(10),
  readFile: (path) => {
    try {
      return new Uint8Array(readFileSync(path));
    } catch {
      return null;
    }
  },
  writeFile: (path, bytes) => {
    try {
      writeFileSync(path, bytes);
      return true;
    } catch {
      return false;
    }
  },
  fileSize: (path) => {
    try {
      const st = statSync(path, { bigint: true });
      return st.isFile() ? st.size : null;
    } catch {
      return null;
    }
  },
  readFileRange: (path, offset, length) => {
    let fd = -1;
    try {
      fd = openSync(path, 'r');
      const st = fstatSync(fd);
      if (st.isDirectory()) return null;
      // ふつうのファイルなら、残っている分より多くは確保しない。
      if (st.isFile()) length = Math.max(0, Math.min(length, st.size - offset));
      else if (offset !== 0) return null;
      const buf = new Uint8Array(length);
      let n = 0;
      while (n < length) {
        const r = readSync(fd, buf, n, length - n, st.isFile() ? offset + n : null);
        if (r <= 0) break;
        n += r;
      }
      return buf.subarray(0, n);
    } catch {
      return null;
    } finally {
      if (fd >= 0) closeSync(fd);
    }
  },
  appendFile: (path, bytes) => {
    try {
      appendFileSync(path, bytes);
      return true;
    } catch {
      return false;
    }
  },
  // Node には位置を動かす関数がないので、位置はここで覚えておき、読み書きのたびに渡す（"a" は常に終わりへ）。
  fileOpen: (path, mode) => {
    try {
      const fd = openSync(path, mode);
      if (fstatSync(fd).isDirectory()) {
        closeSync(fd);
        return null;
      }
      return { fd, pos: 0, append: mode === 'a' };
    } catch {
      return null;
    }
  },
  fileRead: (f, length, offset) => {
    const buf = new Uint8Array(length);
    let n = 0;
    const at = offset ?? f.pos;
    while (n < length) {
      const r = readSync(f.fd, buf, n, length - n, at + n);
      if (r <= 0) break;
      n += r;
    }
    if (offset === null) f.pos += n;
    return buf.subarray(0, n);
  },
  fileWrite: (f, bytes) => {
    let n = 0;
    while (n < bytes.length) {
      n += writeSync(f.fd, bytes, n, bytes.length - n, f.append ? null : f.pos + n);
    }
    if (!f.append) f.pos += n;
    return true;
  },
  fileSeek: (f, offset) => {
    if (f.append) return false;
    f.pos = offset;
    return true;
  },
  fileLength: (f) => {
    const st = fstatSync(f.fd, { bigint: true });
    return st.isFile() ? st.size : null;
  },
  fileClose: (f) => closeSync(f.fd),
  listDir: (path) => {
    try {
      return readdirSync(path);
    } catch {
      return null;
    }
  },
  makeDir: (path) => {
    try {
      mkdirSync(path, { recursive: true });
    } catch {
      // すでにあるときも含め、下で確かめる。
    }
    try {
      return statSync(path).isDirectory();
    } catch {
      return false;
    }
  },
  isDir: (path) => {
    try {
      return statSync(path).isDirectory();
    } catch {
      return false;
    }
  },
  cwd: () => process.cwd(),
  exePath: resolve(file),
  absPath: (path) => resolve(path),
  env: process.env,
  args: programArgs,
  eprint: (line) => process.stderr.write(line + '\n'),
});
if (typeof exports[fn] !== 'function') {
  console.error(`'${fn}' is not exported; write \`export function ${fn}()\``);
  process.exit(2);
}

try {
  // GPU が使える環境では Promise が返るので、常に await する。
  const r = await exports[fn]();
  if (typeof r === 'number' && fn === 'main') process.exitCode = r & 0xff;
} catch (e) {
  if (e instanceof UmPanic) {
    console.error(e.message);
    process.exit(101);
  }
  throw e;
}
