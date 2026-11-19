#!/usr/bin/env node
// wasm を Node で実行する: node runtime/run-node.mjs out.wasm [関数名]
// 標準入力の行は readLine() に、ファイルの読み書きは readFile() / writeFile() にそのままつなぐ
// （ネイティブのバイナリと同じく、プログラムが指定したファイルを読み書きできる）。
import { readFileSync, readSync, writeFileSync } from 'node:fs';
import { instantiate, UmPanic } from './host.mjs';

const [file, fn = 'main'] = process.argv.slice(2);
if (!file) {
  console.error('usage: node runtime/run-node.mjs <file.wasm> [function]');
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
