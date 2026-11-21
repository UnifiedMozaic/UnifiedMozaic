# UnifiedMozaic 言語ガイド

UnifiedMozaic は、TypeScript 風の文法で書き、GC なしでネイティブバイナリと WebAssembly に落とす言語です。CPU と GPU のコードを同じソース（`.um`）で書けます。このガイドでは、いま書けるものと、その決まりごとを説明します。

```ts
struct Vec3 {
  x: f32;
  y: f32;
  z: f32;
}

function add(a: Vec3, b: Vec3): Vec3 {
  return { x: a.x + b.x, y: a.y + b.y, z: a.z + b.z };
}

// mut は可変借用。呼び出し側にも mut と書く。
function scaleInPlace(v: mut Vec3, s: f32): void {
  v.x *= s;
  v.y *= s;
  v.z *= s;
}

export function main(): i32 {
  let v: Vec3 = add({ x: 1, y: 2, z: 3 }, { x: 4, y: 5, z: 6 });
  scaleInPlace(mut v, 0.5);
  print(v.x);
  return 0;
}
```

**いまの状態:** 言語とランタイムは試験段階で、よく変わります。`kernel` 関数は、ネイティブでは CPU の複数のスレッドで、さらに GPU でも実行できます。ネイティブの GPU の経路は2つです: どの OS でも動く wgpu（`UM_WGPU=1`。kernel を WGSL にして wgpu-native で実行する。[wgpu での GPU 実行](wgpu.ja.md)）と、Mac（Apple Silicon）の Metal（`UM_METAL=1`。下の「Metal（Mac）での GPU 実行」）です。wasm32 では、同じ WGSL をブラウザの WebGPU で実行できます（下の「ブラウザでの GPU 実行（WebGPU）」）。

## 必要なもの・使い方

入れ方は [環境構築](setup.ja.md)、最初の一歩は [README.ja.md](../README.ja.md) にあります。要点だけ書きます。

- Node.js 22.18 以上（TypeScript をそのまま実行します。npm パッケージへの依存はありません）
- LLVM 18 以上の `clang`。wasm32 を出すには `wasm-ld` も必要です
  - macOS: Xcode の clang は wasm に対応していないため、wasm32 を使う場合は `brew install llvm lld` で入れてください。入っていれば、wasm32 のビルドには自動でそちらを使います（`UM_CLANG` で clang の場所を指定することもできます）

```sh
# wasm32（既定）
node src/cli.ts build examples/structs.um -o structs.wasm
node runtime/run-node.mjs structs.wasm

# このマシン向けのネイティブバイナリ
node src/cli.ts build examples/structs.um --target host -o structs
./structs

# kernel を GPU でも実行するネイティブバイナリ（Windows・Linux・macOS。wgpu-native が要る。docs/wgpu.ja.md）
UM_WGPU=1 node src/cli.ts build examples/kernels.um --target host -o kernels

# Mac で kernel を GPU（Metal）でも実行するネイティブバイナリ
UM_METAL=1 node src/cli.ts build examples/kernels.um --target host -o kernels

# ホストのプログラムに組み込む wasm（プラグイン。import なし。docs/wasm.ja.md の「UM で書いたプラグイン」）
node src/cli.ts build examples/wasm/plugin/brighten.um --target wasm32-plugin -o brighten.wasm

# Apple Silicon の macOS 向け（Mac 以外ではオブジェクトファイルまで）
node src/cli.ts build examples/structs.um --target arm64-macos -o structs

# 型検査だけ。--json で診断を機械可読な形で出す
node src/cli.ts check examples/structs.um --json

# 生成される LLVM IR を見る（--emit-wgsl / --emit-msl で kernel のシェーダも書き出す）
node src/cli.ts build examples/structs.um --emit-llvm -o structs.ll
```

ブラウザでは `runtime/host.mjs` の `instantiate(bytes)` で読み込み、`export` した関数を呼びます。

## いま書けるもの

| 分類 | 内容 |
| --- | --- |
| 型 | `i8`〜`i64`、`u8`〜`u64`、`usize`、`f4`、`f8`、`f16`、`f32`、`f64`、`bool`、`string`、`struct`、`enum`（値つき。`match` で分岐）、`class`、固定長配列 `T[N]`、可変長配列 `T[]`（`Array<T>` と同じ）、`Buffer<T>`、`RacyBuffer<T>`、`T \| null`、標準の `Result<T, E>`、ベクトルと行列 `vec2<T>`〜`vec4<T>` / `mat2<T>`〜`mat4<T>`、SIMD の `i8x16` など |
| 宣言 | `function`、`export function`、`kernel function`、`struct`、`enum`、`class`（フィールド、初期値、`constructor`、メソッド、`private`、`implements`）、`interface`、ジェネリックな関数・struct・class・enum・kernel、最上位の定数 `const NAME: T = 定数式;`（`export const` も可）、`import { a, b } from "./x.um";` |
| 文 | `let` / `const`（`let x: T;` はゼロで初期化）、代入と複合代入、`++` / `--`、`if` / `else`、`while`、`do ... while`、`for (;;)`、`for...of`、`switch`、`match`、`break`、`continue`、`return` |
| 式 | 四則演算と剰余、比較、`&&` / `\|\|` / `!`、条件式 `a ? b : c`、`x ?? 既定値`、ビット演算とシフト、`as` による変換、関数とメソッドの呼び出し、`new`、`this`、フィールドと添字、オブジェクトリテラル、配列リテラル、`new Array<T>(n)`、`xs.length`、テンプレート文字列 `` `x = ${x}` ``、`String(x)` |
| 配列のメソッド | `push`、`pop`、`clear`、`clone`、一括の書き込み `fill(v)` / `fill(v, start, end)`（Buffer / RacyBuffer にも。下の「一括の書き込み」） |
| 組み込み | `print(x)`（数、`bool`、文字列）、`dispatch` / `cpu.dispatch` / `gpu.dispatch`、グループで実行する `dispatchGroups`（下の「グループ（共有配列・barrier）と kernel の中のアトミック演算」）、`gpu.submit` / `gpu.ready` / `gpu.wait`、`gpu.flags` / `gpu.commit` / `gpu.untracked`（下の「Metal（Mac）での GPU 実行」）、実行時に WGSL の kernel を読み込む `gpu.loadKernel` / `gpu.runKernel`（[wgpu での GPU 実行](wgpu.ja.md)）、`thread.*`（下の「スレッド」）、`Math.sqrt` / `floor` / `ceil` / `abs` / `min` / `max`、ビット演算の `Math.popcount` / `clz` / `ctz`、マスクつきシフト `Math.shl` / `Math.shr`（下の「マスクつきシフトと範囲チェックの除去」）、先読み `Math.prefetch(arr, i)`（下の「先読み」）、SIMD の `Simd.*`（下の「SIMD のベクトル」） |
| 入出力の組み込み | `readLine()`、`inputReady()`、`clock()`、`stringFromBytes(bytes)`、`readFile(path)`、`writeFile(path, bytes)`、`fileSize(path)`、`readFileRange(path, offset, length)`、`appendFile(path, bytes)`、`args()`、`eprint(text)`（下の「入出力」） |
| ネイティブだけの組み込み | ウィンドウ（`windowOpen` など。[window.ja.md](window.ja.md)）、WebAssembly のモジュールの実行（`wasmLoad` / `wasmNew` / `wasmCall` など。[wasm.ja.md](wasm.ja.md)）、動画・音声の読み書き（`mediaOpenVideo` / `mediaFrame` / `mediaSamples` / `mediaCreate` など。FFmpeg を使う。[media.ja.md](media.ja.md)） |

それぞれの書き方は、下の各節と「そのほかの機能」の例にあります。

決まりごと:

- **暗黙の型変換はありません。** `i32` と `f64` を混ぜるときは `as` で明示します。数値リテラルは文脈の型に合わせます。
- **引数の渡し方は3種類です。** 修飾なしは読み取り借用、`mut` は可変借用、`own` は所有権の移動です。`mut` と `own` は呼び出し側にも書きます。
- **構造体・`class`・配列は値型です。** 代入すると、ヒープを持たない値はコピーされ、ヒープを持つ値は移動します（下の「所有権」）。
- **`class` に継承はありません。** `struct` はオブジェクトリテラルで、`class` は `new` で作ります。共通のメソッドの約束には `interface` を使います。
- **メソッドが `this` を書き換えるかどうかは、コンパイラが本体から判断します。** 書き換えるメソッドは、`let` で宣言した変数か `mut` の引数に対してだけ呼べます。
- **コンストラクタは全フィールドを初期化します。** 最上位の `this.x = ...` か、フィールドの初期値（`x: i32 = 0`）で与えます。初期化が済むまで `this` 全体は使えません。
- **`for...of` の要素は、数などはコピー、ヒープを持つ要素（文字列や配列）は借用です。** 借用するときは `for (const x of xs)` と書きます。境界検査は入りません。反復中の配列は書き換えられません。
- **`for (let i = 0; i < xs.length; i++)` の `i` は、比較相手の型（ここでは `usize`）になります。** それ以外の場所では `xs.length` は `usize` なので、`i32` と混ぜるときは `as` が要ります。
- **`usize` の幅はターゲットで違います（wasm32 は32ビット、ネイティブは64ビット）。** 幅に依存する計算には `u32` や `u64` を明示してください。
- **符号なし整数に `>= 0` と書くとコンパイルエラーです。** 常に真で、逆順のループが止まらなくなるためです。
- **代入は右辺を先に評価します。** `a[f()] = g()` は `g()`、`f()` の順です（JavaScript とは逆）。右辺が配列を伸ばしても、代入先を安全に求められるようにするためです。
- **`print` の浮動小数点の表記は、どのターゲットでも JavaScript と同じです。**
- **`Math.min` / `Math.max` は「比較して選ぶ」定義です。** どちらかが NaN のときは2番目の引数を返します（JavaScript とは違います）。
- **ネイティブのプログラムには `export function main(): i32`（引数なし）が要ります。**

## そのほかの機能

短い説明と例です。例はどれもテストを兼ねていて、そのまま動きます。

| 機能 | 要点 | 例 |
| --- | --- | --- |
| `enum` と `match` | 選択肢ごとに値を持てます（`Shape.Circle(1.5)`）。`match` は文で、すべての選択肢を扱わないとコンパイルエラーです。値が `i32` / `u32` / `f32` / `bool` だけの enum は kernel でも使えます。 | [enums.um](../examples/enums.um) |
| `Result<T, E>` | 標準の enum（`Result.Ok(v)` / `Result.Err(e)`）。enum の値には文字列や配列も持てます。例外（`try` / `throw`）はありません。 | [results.um](../examples/results.um) |
| `T \| null` | 使う前に `if (x != null)` などで確かめます（`const` の変数だけが絞り込まれます）。`x ?? 既定値` も使えます。配列の要素にするときは `(string \| null)[]` とかっこで囲みます。ほかの union 型はありません。 | [nullable.um](../examples/nullable.um)、[optional_arrays.um](../examples/optional_arrays.um) |
| ジェネリクス | `function f<T>(...)`、`struct Pair<A, B>`、`class Stack<T>`、`enum Maybe<T>`。型引数は実引数（または求められている型）から決まり、使われた型ごとに複製されます。ジェネリックなメソッドとジェネリック関数の export はまだありません。 | [generics.um](../examples/generics.um)、[generic_structs.um](../examples/generic_structs.um)、[generic_classes.um](../examples/generic_classes.um) |
| `interface` | 「このメソッドを持つ」という約束です。class が `implements` で満たし、ジェネリック関数が `T extends 名前` で求めます。呼び出しはコンパイル時に決まります。interface を値の型として使うことはできません。 | [interfaces.um](../examples/interfaces.um) |
| `switch` / `do ... while` | `switch` は整数・`bool`・文字列で分岐します。各 `case` は `break` / `return` / `continue` で終えます（次の case へ落ちる書き方はエラー。ラベルを並べて本体を共有するのは可）。 | [switches.um](../examples/switches.um)、[do_while.um](../examples/do_while.um) |
| 文字列 | UTF-8 のバイト列を持つ、書き換えできない値です。`+` で連結、`==` で比較、`.length` と `s[i]` はバイト単位です。 | [strings.um](../examples/strings.um) |
| ヒープを持つ要素の配列 | `string[]`、`i32[][]`、配列を持つ struct の配列、自分自身の配列を持つ struct（木）が書けます（下の「所有権」）。 | [nested_arrays.um](../examples/nested_arrays.um) |
| ベクトルと行列 | `vec2` / `vec3` / `vec4` は演算子で成分ごとに計算できます。`mat2` / `mat3` / `mat4` は関数（`mat3Mul` など）で計算します。`f32` / `i32` / `u32` のベクトルは kernel と `Buffer` で、行列は kernel の中でも使えます。 | [vectors.um](../examples/vectors.um)、[matrices.um](../examples/matrices.um) |
| 小さい浮動小数点 | `f16`（IEEE 754 半精度）、`f8`（FP8 E4M3）、`f4`（FP4 E2M1）。演算の結果は、正確な値をその型に丸めたものです。kernel と `Buffer` ではまだ使えません。 | [half_floats.um](../examples/half_floats.um)、[small_floats.um](../examples/small_floats.um) |
| 複数のファイル | `import { a, b } from "./lib/math.um";`。相手のファイルで `export` を付けたものだけ import できます。名前は全ファイルで1つの空間です。 | [tests/modules/basic](../tests/modules/basic/main.um) |

## 最上位の定数

関数の外に、名前つきの定数を書けます。値はコンパイル時に求まります。

```ts
const FILES: i32 = 9;
const SQUARES: i32 = FILES * FILES;        // 他の定数を使える
const BIG: u64 = (1 as u64) << 40;
export const NAME: string = "um-engine";   // 他のファイルから import できる

function sum(xs: i32[SQUARES]): i32 { ... } // 固定長配列の長さにも使える
```

- **型は必ず書きます。** 使える型は整数、`f32`、`f64`、`bool`、`string` と、整数・`f32`・`f64`・`bool` の固定長配列（下の「定数の配列」）です（`struct` の定数、`f4` / `f8` / `f16` の定数、可変長配列の定数はまだありません）。
- **初期化に書けるのは、リテラル・配列リテラル・他の定数（定数の配列の要素 `T[i]` も）・単項と二項の演算子・`as`・かっこだけです。** 関数の呼び出しなどはコンパイルエラーです。定数どうしは書く順番によらず使えます（互いに頼り合うとエラー）。
- **値の計算は実行時と同じ意味です。** 式は `const x: T = 式` と同じ規則で型を検査します（リテラルは定数の型に合わせます）。整数はラップし、小数から整数への変換は飽和します。`&&` / `||` は、左辺で結果が決まれば右辺を計算しません。実行時なら停止する計算（整数のゼロ除算、ビット幅以上のシフト）はコンパイルエラーです。
- **使う所では、その型の値として扱われます。** 暗黙の型変換はないので、`i32` の定数を `usize` の所で使うには `as` が要ります。`for (let i = 0; i < N; i++)` の `i` は `N` の型になります。
- **kernel の中でも使えます**（kernel で使える型の定数に限ります。GPU 向けのコードにも同じ値が入ります）。
- **同じ名前の引数やローカル変数があれば、そちらが優先されます。** 定数への代入はコンパイルエラーです。関数・型と同じ名前の定数は作れません（名前は全ファイルで1つの空間です。`vec3` や `vec3Add` のような標準の関数・型の名前も使えません）。配列の長さ `T[N]` の `N` は最上位の定数だけを指すので、その場所に同じ名前のローカル変数があるとコンパイルエラーです。
- **`switch` の `case` にも書けます**（`case` には式を書けるため）。
- **グローバル変数（最上位の `let`）はありません。** 変わる値は関数の中に置き、`mut` 引数で渡します。

例: [examples/global_consts.um](../examples/global_consts.um)。

### 定数の配列

固定長配列の定数も書けます。盤面の利きのような表を、コンパイル時に決まる読み取り専用のデータとして持つためのものです。

```ts
const BIT: u64[4] = [1, (1 as u64) << 13, (1 as u64) << 27, (1 as u64) << 40];
const STEP: i32[3][4] = [[1, 2, 3], [-1, -2, -3], [10, 20, 30], [0, 0, 0]];  // i32[3] が 4 個
const LAST: u64 = BIT[3];                          // 要素を別の定数に使える
const ROWS: i32[3][2] = [STEP[1], STEP[0]];        // 行を別の定数の配列に使える

bits |= BIT[sq & 3];      // 範囲内と分かる添字なので検査なし
const d = STEP[dir][k];
```

- **入れ子の順番は TypeScript と同じです。** `T[N][M]` は「`T[N]` が `M` 個」で、`STEP[i]` の `i` は `0`〜`M-1`、`STEP[i][j]` の `j` は `0`〜`N-1` です（C とは逆）。
- **要素はコンパイル時に求めます。** 規則は普通の定数と同じです（整数はラップ、ゼロ除算などはコンパイルエラー）。要素の個数が型の長さと違うとエラーです。
- **置き場所はプログラムの読み取り専用のデータです。** ヒープを使わず、使う所ごとのコピーもしません。添字は固定長配列と同じく検査され、`x & 63`、`x % 81`、`if (i < N)` の中のように範囲内と分かる添字では検査が省かれます（下の「マスクつきシフトと範囲チェックの除去」）。
- **書き換えられません。** 要素への代入と `mut` での貸し出しはコンパイルエラーです。`let t = STEP[0];` のように変数に写せば、その写しは書き換えられます。関数の引数（修飾なし）にはコピーせずに貸します。
- **大きさ。** 定数の配列は 4,194,304 個まで持てます。ただし、値としてコピーする使い方（変数に入れる、`own` で渡す、kernel の値の引数にする）ができるのは、普通の固定長配列と同じ 4096 個までです。大きな表は添字で読みます。
- **kernel の中でも読めます**（kernel で使える型の表に限ります）。CPU では同じ読み取り専用のデータを読み、Metal では MSL のプログラムの範囲の `constant` の変数に、WGSL（wgpu と WebGPU）ではモジュールの範囲の `const` になります（kernel の引数は増えません）。範囲外の添字は固定長配列と同じく最後の要素に切り詰めます。
- **まだできないこと:** 関数の呼び出しで表を作ること（`const T: u64[81] = makeTable();` のようなコンパイル時の関数の実行）。大きな表は、生成するスクリプトで `.um` の配列リテラルを書き出してください。`struct` の要素の表もまだです。

例: [examples/const_arrays.um](../examples/const_arrays.um)、[tests/metal/const_arrays.um](../tests/metal/const_arrays.um)・[tests/metal/const_arrays_i64.um](../tests/metal/const_arrays_i64.um)（Metal。64 ビットの整数の表も CPU と同じ値になる）。

## 入出力

```ts
while (true) {
  const line = readLine();          // string | null
  if (line == null) { break; }      // 入力の終わり
  print("got " + line);
}
```

| 関数 | 動作 |
| --- | --- |
| `readLine(): string \| null` | 標準入力から1行読みます。末尾の改行（とその前の `\r` 1つ）は含みません。入力の終わりでは `null`。行の長さに上限はありません。待つ前に標準出力を流します。 |
| `inputReady(): bool` | `readLine()` が待たずに戻るなら `true`（改行までの1行が届いているか、入力が終わっているとき。行の途中までしか届いていなければ `false`）。標準出力も流します。軽いので、探索の途中で何度も呼べます。 |
| `clock(): f64` | プログラムの開始からの秒数。逆戻りしません。 |
| `stringFromBytes(bytes: u8[]): string` | バイト列を写した新しい文字列を作ります（UTF-8 かどうかは確かめません）。 |
| `readFile(path: string): u8[] \| null` | ファイル全体をバイト列として読みます。読めなければ `null`。 |
| `writeFile(path: string, bytes: u8[]): bool` | ファイルを作って（あれば中身を置き換えて）書きます。すべて書けたら `true`。 |
| `fileSize(path: string): i64 \| null` | ふつうのファイルの大きさ（バイト数）。ないファイルやフォルダなどは `null`。 |
| `readFileRange(path: string, offset: i64, length: i64): u8[] \| null` | `offset` バイト目から最大 `length` バイトを読みます。ファイルの終わりで止まるので短く（終わりより後ろなら長さ 0 に）なります。負の `offset` / `length` や読めないファイルは `null`。大きなファイルの中を二分探索するときなどに、全体を読まずに済みます。ふつうのファイルでないもの（`/dev/urandom` など）は `offset` が 0 のときだけ読めます。 |
| `appendFile(path: string, bytes: u8[]): bool` | ファイルの終わりに書き足します（なければ作ります）。すべて書けたら `true`。ログを少しずつ書くのに使います。 |
| `args(): string[]` | コマンドライン引数。プログラム名（`argv[0]`）は含みません。引数がなければ長さ 0。 |
| `eprint(text: string)` | 標準エラー出力に1行書きます（改行を付けます）。`print` と違って、まとめずにすぐ書きます。 |

- **引数は借りるだけで、結果は新しい値です。** `string | null` と `u8[] | null` は、`const` に入れて `if (x != null)` で確かめてから使います。
- **同じ名前の関数をプログラムが定義していれば、そちらが優先されます。** kernel の中では使えません。
- **`print` の出力はまとめて書き出されます。** 相手のプログラムとパイプで話すときは、`readLine()` か `inputReady()` を呼んだ時点（と終了時）に相手へ届きます。
- **名前に 0 のバイトを含むファイルは読めず、書けません**（`null` / `false`）。
- **標準入力が「待たない」設定（O_NONBLOCK）でも、`readLine()` は行が届くまで待ちます**（まだ届いていないことを、入力の終わりとは扱いません）。
- **2 GiB 以上のファイルや行も読み書きできます**（OS への依頼を 1 GiB ずつに分けます）。
- **wasm32 では、入出力は読み込む側が渡した口だけを通ります。** `instantiate(bytes, { readLine, inputReady, readFile, writeFile, fileSize, readFileRange, appendFile, args, eprint })` で渡します。渡さなければ `readLine()` は `null`、`inputReady()` は `true`（入力は終わっている）、`readFile()` は `null`、`writeFile()` は `false`、`args()` は空、`eprint()` は `console.error` です。`fileSize` / `readFileRange` / `appendFile` の口がなければ、`readFile` / `writeFile` の口で代わりに行います（ファイル全体を読むので遅くなります）。wasm32 の `readFileRange` が一度に返せるのは 2 GiB 未満までです。ファイルの名前は UTF-8 の文字列として口に渡すので、UTF-8 として読めない名前は wasm32 では失敗（`null` / `false`）になります。`runtime/run-node.mjs` は標準入力とファイルをそのままつなぎ、`--` のあとの引数を `args()` に渡します（Node には待たずに読めるかを確かめる方法がないので、`inputReady()` は読み込み済みの分だけで答えます）。wasm32 の入出力は、`node tests/run.ts` が `runtime/run-node.mjs` で実行して、ネイティブと同じ結果になることを確かめています。

例: [examples/input_lines.um](../examples/input_lines.um)、[examples/bytes_and_files.um](../examples/bytes_and_files.um)、[examples/file_ranges.um](../examples/file_ranges.um)。

## 所有権

ヒープを持つ値（文字列、可変長配列、`Buffer`、それらをフィールドに持つ `struct` / `class`）は、所有者が常に1つです。GC も参照カウントも使いません。

```ts
function total(xs: i32[]): i32 { ... }          // 借用（読むだけ）
function fill(xs: mut i32[]): void { ... }      // 可変借用（書き換える）
function consume(xs: own i32[]): void { ... }   // 所有権を受け取る

let a: i32[] = [1, 2, 3];
total(a);          // 貸すだけ。a はそのまま使える
fill(mut a);
const b = a;       // 移動。以後 a は使えない（コンパイルエラー）
const c = b.clone(); // 独立したコピーが欲しいときは clone
consume(own b);    // 所有権ごと渡す
// c はスコープを抜けるときに自動で解放される
```

- **移動するのは** 代入、`return`、`own` 引数、オブジェクトリテラルのフィールドです。
- **移動できるのは変数全体だけです。** フィールドや要素から取り出すこと、借用している値を移動することはできません（`clone()` を使います）。
- **数値や、ヒープを持たない `struct` / `class` / 固定長配列は、これまでどおりコピーです。**
- **可変長配列は、ヒープを持つ要素も所有できます**（`string[]`、`i32[][]`、文字列や配列を持つ struct の配列、自分自身の配列を持つ struct = 木）。`push` と配列リテラルは要素を移動し、`pop()` は所有権ごと返します。`xs[i]` は借りるだけで、取り出して移動はできません。配列の配列のリテラルは、型を書いて作ります（`let g: i32[][] = [a, b];`。型がないと固定長配列と見なされ、エラーになります）。
- **ヒープを持つ要素の配列の `.clone()` と、固定長配列 `T[N]` にヒープを持つ要素を入れることは、まだできません。** 複製はループで要素ごとに `.clone()` して `push` します。
- **固定長配列と `struct` が持てる値は4096個までです。** これらは値としてコピーされるので、大きなデータには `T[]` か `Buffer<T>` を使います。
- **`===` / `!==` は `==` / `!=` と同じ意味で受け付けます。**

例: [examples/ownership.um](../examples/ownership.um)、[examples/nested_arrays.um](../examples/nested_arrays.um)。

## kernel と Buffer

`kernel function` は、GPU でも実行できる関数です。`dispatch(kernel, 個数, 引数...)` で、0 から個数未満の各添字について実行します。CPU で実行しても GPU で実行しても同じ結果になるように意味を決めてあります。ネイティブの CPU では、大きな dispatch を複数のスレッドに分けて実行します。

```ts
struct Particle {
  pos: f32;
  vel: f32;
}

kernel function integrate(i: u32, ps: mut Buffer<Particle>, dt: f32): void {
  ps[i].pos = ps[i].pos + ps[i].vel * dt;
}

export function main(): i32 {
  let ps = new Buffer<Particle>(1000);   // 長さ固定、ゼロ埋め
  dispatch(integrate, ps.length, mut ps, 0.016);
  return 0;
}
```

- **`Buffer<T>` は CPU と GPU が共有するデータの置き場です。** 長さは固定で、要素は `i32` / `u32` / `i64` / `u64` / `f32` と、それらの `struct`・固定長配列、`f32` / `i32` / `u32` のベクトル、それに 8 / 16 ビットの整数 `i8` / `u8` / `i16` / `u16`（要素そのものとしてだけ。struct のフィールドや固定長配列の要素にはできない）に限ります。所有権の扱いは可変長配列と同じです。下の「8 / 16 ビットの整数の Buffer」を参照。
- **kernel の中で使える型は `bool` / `i32` / `u32` / `i64` / `u64` / `f32` と、それらの `struct`・固定長配列、`f32` / `i32` / `u32` のベクトルと行列、値が `i32` / `u32` / `f32` / `bool` だけの enum です。** 8 / 16 ビットの整数は、単独の値として読み書きと変換だけに使えます（下の「8 / 16 ビットの整数の Buffer」）。64 ビットの整数を GPU で実行できるのは Metal だけです（下の「kernel の中の 64 ビットの整数」）。小数リテラルは `f32` になります。
- **kernel が書き込めるのは、`mut` の Buffer の `[i]` だけです。** `i` は第1引数（`u32`）で、各要素を別々の実行が担当するので競合しません。隣の要素は、別の（`mut` でない）Buffer から読みます。どの要素にも書きたいときは `RacyBuffer`（下の「スレッド」）を使います。
- **kernel は確保・`print`・通常の関数の呼び出し・再帰ができません。** 補助関数も `kernel function` として書きます。
- **kernel は停止しません（GPU に停止がないため）。** 範囲外の添字は最後の要素に切り詰めます。整数のゼロ除算は `x / 0 = x`、`x % 0 = 0`、シフト量はビット幅で割った余りです。
- **`dispatch` の個数は、書き込み先の Buffer の長さ以下でなければなりません。** 超えると停止します。
- **実行先は `dispatch` が自動で選びます。** 明示するときは `cpu.dispatch` / `gpu.dispatch` と書きます。GPU で実行できるビルドは、ネイティブの wgpu（`UM_WGPU=1`。Windows・Linux・macOS）と Metal（`UM_METAL=1`。Mac）、wasm32（WebGPU）です。それ以外のネイティブのビルドでは `dispatch` は常に CPU で、`gpu.dispatch` はコンパイルエラーです。`gpu.dispatch` は、GPU が使えない環境で実行すると停止します。
- **GPU で実行できるビルドでは、dispatch する kernel に値として渡す struct や配列に `bool` を含められません**（CPU と GPU でビット表現が違うため。`u32` の 0 / 1 を使うか、`cpu.dispatch` で呼びます）。
- **CPU と GPU を同時に走らせるには `gpu.submit` を使います**（下の「Metal（Mac）での GPU 実行」）。

例: [examples/kernels.um](../examples/kernels.um)、[examples/particles_kernel.um](../examples/particles_kernel.um)、[examples/parallel_kernels.um](../examples/parallel_kernels.um)。

### kernel の中の 64 ビットの整数

kernel の中で `i64` / `u64` が使えます（変数、すべての演算、比較、`as`、`Math.popcount` / `clz` / `ctz` / `shl` / `shr` / `min` / `max` / `abs`、補助の kernel の引数と戻り値、struct のフィールドと固定長配列、dispatch する kernel の値の引数、`Buffer<i64>` / `Buffer<u64>` / `RacyBuffer<i64>` / `RacyBuffer<u64>` の要素の読み書き）。81 マスの盤面を `u64` 2 つで持つようなビットボードを、GPU で扱うためのものです。

```ts
struct Bb { lo: u64; hi: u64; }

kernel function count(i: u32, out: mut Buffer<u32>, boards: Buffer<Bb>): void {
  let lo = boards[i].lo;
  let n: u32 = 0;
  while (lo != 0) {               // 1 のビットを ctz で 1 つずつ取り出す
    n += Math.ctz(lo) as u32;
    lo = lo & (lo - 1);
  }
  out[i] = n + (Math.popcount(boards[i].hi) as u32);
}
```

- **意味は 32 ビットと同じ規則です。** 演算はラップし、シフト量は 64 で割った余り、ゼロ除算は `x / 0 = x`、`x % 0 = 0`、最小値 / -1 = 最小値です。`f32` からの変換は範囲外で飽和し、NaN は 0 です。64 ビットの添字は、範囲外なら最後の要素に切り詰めます。
- **GPU で実行できるのは Metal だけです。** MSL の `long` / `ulong` をそのまま使います（Apple M4 で `tests/metal/i64.um` が CPU と全要素一致）。WGSL には 64 ビットの整数がないので、64 ビットの整数を使う kernel（呼ぶ補助の kernel を含む）は、wgpu（`UM_WGPU=1`）と wasm32 では `dispatch` / `gpu.submit` が CPU で実行し、`gpu.dispatch` はコンパイルエラー（`kernel-type`）になります。
- **64 ビットのアトミック演算（`fetchAdd` / `compareExchange` など）は kernel の外だけです**（Apple の GPU にも 64 ビットのアトミックな加算・比較交換はありません）。kernel の中の `RacyBuffer<u64>` の `b[i]` は普通の 8 バイトの読み書きです。CPU ではちぎれませんが、Metal が 8 バイトの読み書きを 1 回で行うことは仕様では保証されていません（別の実行が同じ要素を同時に書くと、上下 32 ビットが別々の値になりうると考えてください）。
- **並び**: `i64` / `u64` は 8 バイトで 8 バイト境界です（struct の中では前に詰め物が入ります。CPU と MSL で同じ並び）。dispatch の値の引数も、GPU 側の引数の struct と同じ境界に揃えて渡します（`src/codegen.ts` の `gpuDispatch`）。
- 例: [tests/metal/i64.um](../tests/metal/i64.um)（Metal）。

### 8 / 16 ビットの整数の Buffer

`Buffer<i8>` / `Buffer<u8>` / `Buffer<i16>` / `Buffer<u16>`（と同じ要素の `RacyBuffer`）が使えます。量子化したニューラルネットの重み（i8 / i16）のような小さい整数を、詰め直さずにそのまま GPU と共有できます。

```ts
kernel function affine(i: u32, out: mut RacyBuffer<i32>, w: Buffer<i8>, x: Buffer<u8>, n: u32): void {
  let acc: i32 = 0;
  for (let k: u32 = 0; k < n; k++) { acc = acc + (w[i * n + k] as i32) * (x[k] as i32); }
  out[i] = acc;
}
const t = gpu.submit(affine, rows, mut out, w, x, n);  // w と x は読むだけなので普通の Buffer でよい
const mine = affineCpu(w, x, 2, n);                     // そのあいだに CPU も同じ重みを読める
gpu.wait(t);
```

- **kernel の中の 8 / 16 ビットの値は、読み書き・比較・`as` での変換だけです。** 計算は `as i32` / `as u32` で広げてから行い、書くときに `as i8` などで戻します（`as` は下位のビットを残す。CPU と GPU で同じ値）。GPU では 32 ビットのレジスタに入れて持つので、演算ごとに切り詰め直す手間を避けるためです。浮動小数点との直接の変換もできません（32 ビットの整数を経由する）。kernel の外では、ほかの整数と同じにすべての演算が使えます。
- **dispatch する kernel の値の引数には使えません**（`i32` で受けて kernel の中で変換）。kernel の補助関数の引数と戻り値には使えます。
- **確保は 4 バイトの倍数に切り上げます**（長さは頼んだ数のまま）。GPU が 4 バイト単位で読み書きするためです。
- **GPU での実装**: Metal（MSL）では `char` / `uchar` / `short` / `ushort` をそのまま読み書きします。WebGPU（WGSL）には 8 / 16 ビットの整数がないので、`u32` の配列として渡し、要素を取り出す関数を通します。書き込みは、同じ語の隣の要素を別の実行が同時に書くので、アトミックな演算（`atomicAnd` + `atomicOr`、RacyBuffer では比較交換）で自分のビットだけを変えます。WGSL の版は Deno の WebGPU（wgpu）で、CPU と同じ結果になることを確かめました（`tests/metal/small_ints.um` の kernel）。
- **書く Buffer が 8 / 16 ビットの要素だと、Metal の検算による早い完了の確認は使いません**（Metal の完了の知らせを待つ）。
- `f16` の Buffer はまだありません（WGSL の f16 は拡張機能で、丸めの決まりを CPU と揃えられないため）。
- 例: [examples/small_ints.um](../examples/small_ints.um)。

### グループ（共有配列・barrier）と kernel の中のアトミック演算

`dispatchGroups(k, groups, size, 引数...)` は、kernel を「大きさ `size` のグループ」を `groups` 個並べて実行します（全体で `groups × size` 個。`cpu.dispatchGroups` / `gpu.dispatchGroups` と、待たない `gpu.submitGroups` もあります）。GPU では1つのグループが1つの threadgroup（Metal）/ workgroup（WebGPU）になり、グループの中のスレッドは共有配列を一緒に使えます。ニューラルネットの層のように同じ入力を何度も読む計算や、和を求める計算を速くするためのものです。

```ts
kernel function groupSum(i: u32, out: mut RacyBuffer<u32>, xs: Buffer<u32>): void {
  let t: u32[256] = group.shared();   // グループで共有する配列（グループの始めに 0）
  const l = group.local();            // グループの中の番号
  t[l] = xs[i];                       // i = group.id() * group.size() + group.local()
  group.barrier();                    // グループの全スレッドがここまで来るのを待つ
  for (let s = group.size() / 2; s > 0; s = s / 2) {
    if (l < s) {
      t[l] = t[l] + t[l + s];
    }
    group.barrier();
  }
  if (l == 0) {
    out[group.id()] = t[0];
  }
}
dispatchGroups(groupSum, n / 256, 256, mut sums, xs);
```

- **kernel の中で使えるもの**: `group.local()` / `group.id()` / `group.size()` / `group.count()`（いずれも `u32`）、`group.barrier()`、`let t: T[N] = group.shared();`（T は `i32` / `u32` / `f32`）。これらを使う kernel は `dispatchGroups` でしか実行できず、関数として呼ぶこともできません。
- **対応する実行先**: グループの機能は CPU と Metal で揃えています。WGSL（wgpu と wasm32 の WebGPU）で使えるのは共有配列・barrier・kernel の中のアトミック演算までです。simdgroup の集計を使う kernel は、wgpu では `dispatchGroups` が CPU で実行し `gpu.dispatchGroups` はコンパイルエラー（`webgpu-simdgroup`）、wasm32 ではコンパイルエラー（`kernel-restriction`）になります。
- **大きさ**: グループの大きさは 1〜1024 です（それ以外と、`groups × size` が `u32` に収まらないときは停止）。WebGPU が必ず許すのは 256 まで、Metal は kernel ごとに上限があり（ふつう 1024）、超えると GPU の実行がエラーで止まります。共有配列は kernel ごとに合わせて 16 KiB まで（WebGPU が必ず用意する大きさ）です。
- **共有配列は本体の直下で宣言し、添字で1要素ずつ読み書きします**（配列ごと写したり、関数に渡したりはできません）。範囲外の添字は最後の要素に切り詰めます。
- **`group.barrier()` はグループの全スレッドが通る場所にだけ置けます**（GPU では、通らないスレッドがあると止まりうるため）。barrier を囲む `if` / `while` / `for` の条件は、グループの全スレッドで同じ値になる式（定数、値の引数、`group.id()` / `size()` / `count()`、Buffer の長さと、それらだけから計算した変数）に限ります。型検査が控えめに確かめ、満たさなければコンパイルエラーです。barrier を含むループの中の `break` / `continue`、barrier を使う kernel の `return`、`for...of` の中の barrier も使えません。
- **barrier は共有配列の読み書きをそろえます。** barrier なしで、ほかのスレッドが書いた共有配列の要素を読んだときの値は決まりません（メモリは壊れません）。
- **CPU では、グループを1つずつ実行します**（グループどうしはスレッドプールで並列）。barrier の間の文を、グループの全スレッドについて順に実行し、barrier を含む文の条件はスレッド 0 の値で決めます（上の規則により全スレッドで同じ）。barrier をまたいで生きる変数はスレッドごとの置き場所に置きます。barrier を使わない kernel は、スレッドを1つずつ最後まで実行します。GPU と CPU で、足す順番が同じ計算は同じ結果になります（`tests/metal/groups.um`）。
- **kernel の中でも `RacyBuffer` のアトミック演算 `load` / `store` / `fetchAdd` / `compareExchange` が使えます**（要素は `i32` / `u32`、`load` / `store` は `f32` も）。書き換える演算は `mut` の RacyBuffer だけです。順序は CPU では seq_cst、GPU では relaxed（Metal の `atomic_*_explicit`、WebGPU の `atomic*`）です。kernel の中でアトミック演算を使う RacyBuffer は、GPU ではアトミックな型の配列として渡し、`b[j]` の読み書きもアトミック（relaxed）になります。
- **simdgroup の集計**（CPU と Metal だけ。WGSL は対応しません）: simdgroup は、グループの中の連続した 32 スレッドです（グループの大きさが 32 で割り切れなければ、最後のものは短くなります）。
  - `const s = group.simdSum(x);`（和）、`group.simdMin(x)`、`group.simdMax(x)`: 同じ simdgroup の全スレッドの x（`i32` / `u32` / `f32`）を集めた値。
  - `const y = group.simdShuffle(x, lane);`: 同じ simdgroup の lane 番目のスレッドの x（lane が simdgroup の外なら最後のスレッド）。
  - `group.simdLane()`（`u32`）: simdgroup の中の番号（`group.local() % 32`）。
  - 集計は `let` / `const` の初期化式としてだけ書けます。barrier と同じく、グループの全スレッドが通る場所にだけ置け（囲む条件は一様な式だけ）、集計を使う kernel では `return` を使えません。CPU では barrier と同じく区切りになります。
  - Metal の `f32` の和は足す順番が決まっていないので、CPU と丸めが違うことがあります（整数は一致）。Metal では `simd_sum` などの命令になり、共有配列と barrier を使う木の形の和より速くなります。
  - 例（グループ全体の和）:
    ```ts
    let part: i32[32] = group.shared();
    const s = group.simdSum(xs[i]);
    if (group.simdLane() == 0) { part[group.local() / 32] = s; }
    group.barrier();
    let v: i32 = 0;
    if (group.local() < (group.size() + 31) / 32) { v = part[group.local()]; }
    const total = group.simdSum(v);   // どのスレッドでも、グループ全体の和
    ```
- 例: [examples/groups.um](../examples/groups.um)、[tests/metal/groups.um](../tests/metal/groups.um)（Metal）。

### 標準の関数（詰めた小さい整数）

`runtime/prelude.um` の関数は、どのプログラムからもそのまま呼べます。BitNet のような3値の重みを、1個2ビットで `Buffer<u32>` に詰めて扱うためのものです。

```ts
let weights = new Buffer<u32>(packed2Words(count));   // 必要なワード数
packed2Set(mut weights, k, -1);                       // 書く（CPU専用。範囲外は停止）
const w = packed2Get(weights, k);                     // 読む（-2 〜 1。kernel の中でも使える）
out[i] = ternaryDot(weights, i * inputs, x, 0, inputs); // 3値の重みと入力の積和（掛け算なし）
```

`w[i]` と書ける専用の型ではなく、関数です。3値の重みを「2ビットずつ詰める」形に決め打ちしています（5個を1バイトに詰める形は未対応）。1ビットの `packed1Words` / `packed1Get` / `packed1Set` もあります。例: [examples/packed_weights.um](../examples/packed_weights.um)。[examples/bitnet.um](../examples/bitnet.um) には、重みが -1 / 0 / +1 の3値の線形層（BitNet 風）を kernel で書いた例があります。

## Metal（Mac）での GPU 実行

macOS では、`UM_METAL=1` を付けてビルドすると、kernel を Metal で GPU でも実行するネイティブバイナリになります（ランタイムは `runtime/metal.m`）。Apple Silicon（Apple M4）で確かめています。

```sh
UM_METAL=1 node src/cli.ts build x.um --target host -o x    # --target arm64-macos でもよい
./x
```

- **`UM_METAL=1` はコンパイル時の指定です。** 付けないと、ネイティブのビルドは GPU を使いません（`dispatch` は CPU、`gpu.dispatch` はコンパイルエラー。`UM_WGPU=1` のときを除く）。Mac 以外では `UM_METAL=1` は効きません。どの OS でも動く GPU の経路は [wgpu](wgpu.ja.md)（`UM_WGPU=1`）です。Mac では Metal の経路を既定とします（Buffer をコピーせずに GPU に渡せるため。理由は [wgpu での GPU 実行](wgpu.ja.md)の「バックエンドの選び方」）。
- **kernel は MSL（Metal Shading Language）に変換されてバイナリに入り、初めて GPU で実行するときに Metal がコンパイルします。** 浮動小数点の近似は切ってあり（`MTLMathModeSafe` と精度の高い数学関数）、`tests/metal/parity.um` の kernel は CPU と同じ結果になります。macOS 15 より前では精度の指定の一部が使えず、`sqrt` の丸めが CPU と違う可能性があります（未確認）。
- **Buffer はコピーせずに GPU と共有します**（ユニファイドメモリ）。`UM_METAL=1` のビルドでは Buffer をページ境界に確保し、そのまま GPU に渡します。
- **GPU の準備（数十ミリ秒）は、初めて GPU を使うときまでしません。** CPU だけで済むプログラムには、この手間がかかりません。

### 実行先の選び方（`dispatch` / `cpu.dispatch` / `gpu.dispatch`）

| 書き方 | 実行先 |
| --- | --- |
| `dispatch(k, n, ...)` | 実測で速いほうを選びます（下の説明） |
| `cpu.dispatch(k, n, ...)` | 常に CPU（複数のスレッド） |
| `gpu.dispatch(k, n, ...)` | 常に GPU。GPU が使えなければ停止（`panic: the GPU is not available`） |

`dispatch` の自動判断は、kernel ごとに CPU と GPU の時間を測って決めます。

- **最初の2回は CPU で実行して時間を測ります。** CPU での1回が 50 マイクロ秒未満の仕事は、GPU を試しません（GPU に1回渡すだけでそれ以上かかるため）。
- **それより重い仕事は、GPU でも2回試し、以後は見込みの速いほうを選びます。** 見込みは「GPU の固定の手間 + 1要素あたりの時間 × 個数」と「CPU の1要素あたりの時間 × 個数」の比較です。
- **選ばれなかった側も、ときどき測り直します**（およそ 64 回に1回。差が大きいほど、勝ち負けが変わらないほど間隔を広げます）。
- **途中で実行先が変わっても、結果は同じです**（kernel の意味を CPU と GPU で揃えてあるため。RacyBuffer で競合する書き方は除きます）。
- `UM_GPU_THRESHOLD=N` を付けると実測をやめ、「要素数が N 以上なら GPU」という固定の規則になります。

### 同期の dispatch を速くする仕組み

`dispatch` / `gpu.dispatch` は、GPU の仕事が終わるまで戻りません。そのうえで、往復の手間を減らす仕組みが自動で働きます。

- **続けて書いた dispatch 文は、1回の提出にまとめます。** 次の文も dispatch 文で、どちらの個数と引数も変数かリテラルだけのとき、完了を待たずに次と一緒に提出します（前の kernel の書き込みは、次の kernel から見えます）。
- **同じ形の dispatch を何度も呼ぶと、常駐の経路に乗ります。** 「形」は、1回の提出にまとまる dispatch の並び（1つでもよい。最大 8 個）と、それぞれの kernel・Buffer の組です。同じ形を 8 回見ると、その形の命令を先に GPU に提出して合図を待たせておき、以後は引数と個数を書いて合図を出すだけで走らせます。同時に待機させておける形は 4 つまでです。しばらく（約 1 秒）使われない待機と、関係する Buffer が解放されるときの待機は片付けます。`UM_GPU_RESIDENT=0` で使いません。
- **完了の知らせを待たずに、検算で結果を取り込みます。** GPU が書いた範囲の検算値を、依頼の番号を混ぜて GPU と CPU の両方で計算し、一致したら結果が読めると判断します（Metal の完了の知らせは、GPU が終わってから数十マイクロ秒遅れて届くため）。使うのは、書く Buffer が 4 つ以下、要素の大きさが 4 バイトの倍数、それぞれ 65536 語以下のときです。外れると完了の知らせを待ちます（結果は同じ）。`UM_GPU_RESIDENT=status` で常に完了の知らせを待ちます。

目安（Apple M4、小さな評価を1回ずつ GPU に渡して毎回答えを読む計測）: 1回の往復は常駐と検算を使って約 0.1 ms です。同じ仕事が CPU で数十マイクロ秒で終わるなら、CPU のほうが速くなります。1回に 1024 件 × 2048 回の積和をする重い評価では、GPU（約 0.2 ms）が CPU の全コア（約 0.45 ms）より速くなりました。

### CPU と GPU を同時に走らせる（`gpu.submit` / `gpu.ready` / `gpu.wait`）

```ts
const t = gpu.submit(k, n, mut out, weights);   // 待たずに戻る。券（u64）を返す
doOtherWork();                                   // そのあいだ CPU は別の仕事
if (gpu.ready(t)) { ... }                        // 終わっていれば true（待たない）
gpu.wait(t);                                     // 終わるまで待つ
print(out[0]);                                   // ready / wait のあとに読めば新しい結果
```

- **`gpu.submit` に渡す Buffer のうち、kernel が書くものは `RacyBuffer` でなければなりません**（コンパイルエラー）。待たずに戻るので、CPU と GPU が同じ領域を同時に触りうるためです。読み取り借用の `RacyBuffer` も `mut` として渡せます。
- **kernel が読むだけの普通の `Buffer` も渡せます。** 重みのように一度作って読むだけのデータを `Buffer` に置けば、CPU は普通の（アトミックでない）読み込みで同じ領域を読めます。kernel が終わる前に CPU がその Buffer を書き換えると、kernel は古い値と新しい値の混ざったものを読むことがあります（停止や範囲外の読み書きは起きません）。
- **`gpu.ready` / `gpu.wait` の前に読んだ値は古いことがあります。** 同じ Buffer に続けて頼んだ仕事は、頼んだ順に実行されます。
- **終わった券と券 0 は、いつ確かめても「終わっている」です。** 券はどのスレッドで待ってもかまいません。
- **Buffer が解放されるときは、それを使っている仕事が終わるのを待ちます。**
- **預けたままの券が 1024 件を超えると、いちばん古い仕事の終わりをその場で待ちます**（結果は変わりません）。
- **実際に待たずに戻るのは Metal だけです。** GPU のないビルド・環境（`UM_METAL` なし、`UM_GPU=0`）と wasm32 では、その場で最後まで実行して券 0 を返します（結果は同じ）。`UM_GPU_COPY=1` のときも、その場で実行します。
- 書く Buffer の検算による早い完了の確認（上と同じ条件）は、`gpu.ready` / `gpu.wait` でも使います。
- 目安（Apple M4）: `gpu.submit` 1回の呼び出しは数マイクロ秒です。小さな評価を 8 件ほど預けたまま回すと、1件ずつ待つ `gpu.dispatch` の 3〜4 倍の処理量になりました。

例: [examples/gpu_submit.um](../examples/gpu_submit.um)（GPU なしでも動きます）。

### gpu.submit の印（`gpu.flags` / `gpu.commit`）

`gpu.submit` は、既定では「書く Buffer ごとに検算の kernel を足して、完了を早く知る」「1 本の列に積み、GPU が忙しければ（提出して終わっていない命令が 4 つ）ためて、まとめて提出する」という動きをします。探索の評価のように小さい仕事を何度も往復させる使い方のために、この動きを呼ぶ側で変えられます（Metal だけ。ほかの環境では何もしません）。

```ts
const old = gpu.flags(1 | 2 | 4);   // 検算なし、急ぎの列、提出を見送る
gpu.submit(k1, n, mut a1, x);
gpu.submit(k2, n, mut a2, a1);
gpu.flags(1 | 2);                   // 最後は見送らない（ここで 3 つまとめて提出される）
gpu.submit(k3, n, mut out, a2, tag);
gpu.flags(old);
```

- **`gpu.flags(f: u32): u32`** は、呼んだスレッドで、このあとの `gpu.submit` に付ける印を f にして、前の値を返します。印は 0（既定）か、次のビットの組み合わせです。
  - **1（検算なし）**: 検算の kernel を積みません。完了は Metal の完了の知らせ（GPU が終わってから数十マイクロ秒遅れて届く）と、完了の印（命令の完了の処理が券ごとに書く。`gpu.ready` / `gpu.wait` が最初に見る）で知ります。検算を外すだけだと完了を知るのが遅くなるので、下の「札で完了を判断する」と組にして使います。
  - **2（急ぎの列）**: 2 本目の Metal の列（最初に使ったときに作る）に積み、ためずにすぐ提出します。今までの列にたまっている大きい仕事の後ろに並びません。
  - **4（提出を見送る）**: 積むだけで提出しません。同じ列への次の「見送らない」submit、`gpu.commit()`、その券の `gpu.ready` / `gpu.wait`、ためた数が 32 に達したとき、のどれかで提出されます。いくつかの kernel を 1 回の提出（約 0.1 ms の固定の手間）にまとめるためのものです。最後の submit は見送らないようにします。
  - **8（郵便受け）**: 検算なし（1）の submit を待機命令（常駐）の経路に乗せ（急ぎの列でなくても）、合図で待たせる代わりに「郵便受けの列」で待たせます（下の「郵便受けの列」）。並びのどれか 1 つの submit に付いていれば、その並びが郵便受けの列の形になります。検算ありの submit には効きません。
- **`gpu.commit()`** は、見送ってためてある提出を（どちらの列も）いま提出します。
- どちらも kernel の中では使えません。印はスレッドごとです（別のスレッドの submit には効きません）。
- 印は環境変数で上書きできます（プログラムを作り直さずに比べるため。下の「環境変数」の `UM_GPU_CHECK` / `UM_GPU_URGENT` / `UM_GPU_HOLD` / `UM_GPU_MAILBOX`）。`UM_GPU_STATS=1` で、終わるときに列ごとの submit の数、完了を何で知ったか（検算・status・完了の印）、`waitUntilCompleted` まで待った数と時間、券の表の待ちなどを標準エラー出力に 1 行出します。
- **列をまたぐ注意**: 2 本の列のあいだでは、Metal は前の命令の書き込みを待ちません。ランタイムは、別の列のまだ終わっていない仕事が書いた領域を使う submit を、その仕事が終わるまで待たせます（安全網）。ただし「別の列で読んでいる最中の領域に書く」場合は見ないので、1 つの領域はいつも同じ列で使ってください。
- 例: [tests/metal/submit_flags.um](../tests/metal/submit_flags.um)。

**待機命令（常駐）の経路**: 検算なし（1）で急ぎの列（2）の submit は、同じ形（見送り（4）でつないだ kernel の並びと Buffer の組）が 8 回続くと、その形の命令を先に提出して合図を待たせておき、以後は引数を共有メモリに書いて合図を出すだけで走らせます（形ごとに別の Metal の列）。同期の dispatch の常駐の経路（上の「同期の dispatch を速くする仕組み」）の、`gpu.submit` 版です。

- 見送りの submit はそのスレッドの中にためておき、並びの最後（見送らない submit）でまとめて走らせます。そのため、急ぎ・検算なしの見送りの submit は、別のスレッドの submit では提出されません（そのスレッドの次の見送らない submit、`gpu.commit()`、その券の `gpu.ready` / `gpu.wait` で提出されます）。
- `UM_GPU_ASYNC_RESIDENT=0` で使いません。`=all` で、急ぎでない検算なしの submit にも使います。
- グループの submit（`gpu.submitGroups`）も同じく対象です（形にグループの大きさを含みます）。同期の `gpu.dispatchGroups` も、普通の dispatch と同じく常駐の経路に乗ります。
- `UM_GPU_STATS=1` の `resident submits` が、この経路で走らせた submit の数です。

**郵便受けの列（印 8）**: 合図で待たせた命令は、合図を出してから GPU が動き出すまでに 60〜100 マイクロ秒かかります。郵便受けの列では、待機命令を先に走らせておきます。命令の先頭の「見張り」の kernel が共有メモリの郵便受けを見ていて、依頼の番号が書かれたら抜け、続きの kernel が走ります（依頼から kernel の終わりまで、kernel の実行時間 + 数マイクロ秒）。

- 形は kernel の並びだけで決まります（Buffer の組は含めません）。待機命令の kernel は、Buffer を依頼のときに渡されるポインタの表から読むので、同じ kernel を別々の Buffer の組で（置き場ごと・スレッドごとに）どんな順に投げても、同じ待機命令で受けられます。並びが短い仕事（長い並びの一部になっているもの）は、長い並びの待機命令に相乗りします。
- 結果が CPU から見えるのは、今までどおり命令の区切りです（GPU が終えてから 20〜85 マイクロ秒。続けて投げているほど早い）。
- 効果（Apple M4、3 つの kernel の並びを繰り返し投げる小さな計測。往復の時間 / 毎秒の件数）: 1 スレッド 113 µs / 8.9 千 → 75 µs / 13.2 千、4 スレッド 150 µs / 24.1 千 → 129 µs / 30.9 千、10 スレッド 298 µs / 30.8 千 → 268 µs / 37.1 千。
- 見張りが回っているあいだ、同じプロセスのほかの Metal の列の命令は走れません（GPU は列を命令の切れ目でしか切り替えません）。ランタイムは、ほかの経路（普通の dispatch、検算ありの submit など）で命令を出すたびに待機分を空振りさせて見張りをどかします。郵便受けの列の仕事とほかの経路の GPU の仕事を細かく交互に使うプログラムでは効果がありません。
- 違う kernel の並びをたくさん、入り混じった順で投げて空振りが多くなると、自動でやめて合図の経路に戻ります（その実行のあいだ）。`UM_GPU_STATS=1` の `gpu mailbox` の行に、空振りの数などが出ます。
- 例: [tests/metal/mailbox.um](../tests/metal/mailbox.um)。

### 追跡しない Buffer（`gpu.untracked`）

Metal は、同じ Buffer に書く仕事どうしを、提出（命令）をまたいで順に実行します（書き込みの追跡）。いくつものスレッドが 1 つの作業用の Buffer の別々の部分を使って仕事を投げると、この追跡のせいで仕事が 1 つずつしか走りません。`gpu.untracked(b)` は、このあとの `gpu.submit` で Buffer（または RacyBuffer）b を追跡しないように Metal に伝えます（Metal だけ。CPU と wasm32 では何もしません。Buffer を解放すると元に戻ります）。

```ts
let scratch = new RacyBuffer<i32>(threads * PER_JOB);
gpu.untracked(scratch);
// スレッドごとに scratch の別々の部分を使う。並びは見送り（4）でつないで 1 つの提出にまとめる。
gpu.flags(1 | 2 | 4);
gpu.submit(layer1, n, mut scratch, me * PER_JOB, x);
gpu.flags(1 | 2);
const t = gpu.submit(layer2, n, mut out, scratch, me * PER_JOB);
```

- **安全の決まり（呼ぶ側が守る）**: 同時に走りうる GPU の仕事どうしで、追跡しない Buffer の同じ要素を、一方が書き、もう一方が読む・書くことがあってはいけません。破ると、古い値や混ざった値を読みます（メモリは壊れず、停止もしません）。
- **別々に提出した仕事どうしの順番は守られません。** 前の submit が書いた値を読む submit は、次のどれかにします。
  - 前の券を `gpu.wait` / `gpu.ready`（または札）で終わったと確かめてから投げる。
  - 見送り（4）でつないで 1 つの提出にまとめる。急ぎ・検算なしの並び（`gpu.flags(1 | 2 | 4)` … 最後を `gpu.flags(1 | 2)`、8 個まで）は、ランタイムが必ず 1 つの命令に入れます。それ以外の見送りは、ほかのスレッドが同じ列に submit すると途中で提出されることがあるので、1 つのスレッドだけが投げる場合に限ります。
  - 1 つの命令の中の順番（並びの前の kernel の書き込みと、検算の kernel）は、ランタイムが守ります。
- 同期の dispatch 文（`gpu.dispatch` など）は完了を待つので、この決まりには関係しません。
- 例: [tests/metal/untracked.um](../tests/metal/untracked.um)。

### 札で完了を判断する

kernel が結果の語に世代の番号（札）を一緒に書けば、`gpu.ready` / `gpu.wait` を呼ばずに、CPU が札を見て完了を知れます（Metal の完了の知らせを待ちません）。

```ts
kernel function eval(i: u32, out: mut RacyBuffer<u32>, x: Buffer<i32>, tag: u32): void {
  out[i] = ((score(i, x) as u32) & 65535) | (tag << 16);  // 値と札を同じ 32 ビットの 1 語に
}
gpu.flags(1);
gpu.submit(eval, n, mut out, x, tag);
while ((out[0] >> 16) != tag) { thread.yield(); }   // 札が見えたら、その語の値は正しい
```

- **値と札は、同じ 32 ビットの 1 語に入れます。** GPU の書き込みは語ごとに別々のタイミングで CPU から見えるので、「別の語に終わりの印を書いて、それが見えたら全部読める」という方法は使えません（印が見えても、ほかの語はまだ古い値のことがあります。`runtime/metal.m` の `um_park_wait` の前のコメント）。1 語の中なら、札が新しければ値も新しい値です。
- 札が 1 周して古い札と同じになるのを避けるため、札は 1 から増やし、一周するときは結果の Buffer を 0（「まだ何も書いていない」）で埋めてから使います。
- **すべての語の札が見えたら、その仕事の入力の Buffer は使い回してかまいません**（最後の kernel が前の kernel の結果を読んで全部の語を書いた、つまり前の kernel も終わっている、という形のとき）。
- 札で判断する呼び手は、補助の kernel の券に `gpu.ready` / `gpu.wait` を呼ばなくてかまいません。券の表（1024 枠）が満杯になると、ランタイムは完了の印が付いた枠をまとめて回収します。見続けても札が見えない（GPU が失敗した、など）ときのために、時間を決めて `gpu.wait` に切り替える逃げ道を用意してください。
- 例: [tests/metal/submit_flags.um](../tests/metal/submit_flags.um)。

### 環境変数

実行時に読むもの（Metal のランタイム）:

| 変数 | 意味 |
| --- | --- |
| `UM_GPU=0` | GPU を使いません。`dispatch` はすべて CPU、`gpu.dispatch` は停止、`gpu.submit` はその場で CPU で実行します |
| `UM_GPU_THRESHOLD=N` | 自動判断の実測をやめ、要素数が N 以上なら GPU にします |
| `UM_GPU_RESIDENT=0` | 常駐の経路を使いません |
| `UM_GPU_RESIDENT=status` | 検算による早い完了の確認を使わず、常に Metal の完了の知らせを待ちます |
| `UM_GPU_TRACE=1` | 実行先・共有した Buffer の数・時間・常駐の開始と停止などを標準エラー出力に出します |
| `UM_GPU_COPY=1` | 比較用。Buffer を共有せず、毎回コピーして渡します（遅くなります） |
| `UM_GPU_CHECK=0` / `1` | `gpu.submit` の検算を、印によらず常にしない / 常にします（付けなければ `gpu.flags` の印に従います） |
| `UM_GPU_URGENT=0` | `gpu.flags` の印 2 があっても、急ぎの列を使いません |
| `UM_GPU_HOLD=0` | `gpu.flags` の印 4 があっても、提出を見送りません |
| `UM_GPU_MAILBOX=1` / `0` | 印 8 がなくても待機命令の経路の全部に郵便受けの列を使う / 印 8 があっても使いません |
| `UM_GPU_ASYNC_RESIDENT=0` / `all` | `gpu.submit` の待機命令の経路を使わない / 急ぎでない検算なしの submit にも使います |
| `UM_GPU_STATS=1` | 終わるときに、列ごとの submit の数、完了を何で知ったか、待った数と時間、待機命令の経路と郵便受けの列の数などを標準エラー出力に出します |

CPU の並列実行とスレッドについては、下の「スレッド」の `UM_THREADS`、`UM_POOL_SPIN_US`、`UM_THREAD_STACK_MB` を見てください。

### 制限

- **1つの dispatch に渡せる Buffer は 64 個までです**（超えると停止）。Buffer が 17 個以上か、Buffer 以外の引数が 512 バイトを超える dispatch は、まとめずに1つだけで提出します。
- **GPU の個数と添字は 32 ビットです**（Buffer の長さは `u32`）。
- `f16` / `f8` / `f4` / `f64`、SIMD のベクトル、`Math.prefetch` は kernel では使えません（64 ビットの整数は使えます。上の「kernel の中の 64 ビットの整数」）。
- 64 ビットのアトミック演算は kernel の中では使えません（kernel の外だけ）。
- GPU の仕事がシェーダのコンパイルや実行に失敗すると、プログラムは停止します（`panic: the GPU could not run a kernel (...)`）。

## ブラウザでの GPU 実行（WebGPU）

kernel を WGSL（WebGPU のシェーダ言語）にする変換（`src/wgsl.ts`）は、ネイティブの wgpu の経路（`UM_WGPU=1`。[wgpu での GPU 実行](wgpu.ja.md)）と共通の、主な GPU の経路の一つです。wgpu の経路で確かめた WGSL の修正（割り算と平方根の正しい丸め、FMA にまとめない掛け算、定数の配列など）は、ブラウザでもそのまま使われます。ブラウザの側のランタイム（`runtime/host.mjs`）は、wgpu の経路にある「Buffer を GPU に置いたままにする」仕組みや実測による自動判断を持たない、簡単なものです。

wasm32 向けにビルドすると、dispatch している kernel が WGSL に変換されて wasm に埋め込まれます。`runtime/host.mjs` の `instantiate` が、WebGPU でその kernel を実行します。

```js
import { instantiate } from './runtime/host.mjs';
const exports = await instantiate(wasmBytes, { print: console.log });
await exports.main();   // GPU が使える環境では Promise が返るので、常に await する
```

- **自動判断は単純です。** GPU が使えて、要素数がしきい値（既定 4096、`gpuThreshold` で変更可）以上なら GPU で実行します。実測による判断はありません。
- **dispatch のたびに Buffer を送り、書き込み先を読み戻します。** WebGPU では CPU と GPU のメモリが別だからです。小さい仕事や、CPU と頻繁に行き来する処理では、CPU の方が速くなります。
- **`gpu.submit` は、その場で最後まで実行して券 0 を返します。** `gpu.flags` / `gpu.commit` / `gpu.untracked` は何もしません。
- **WGSL にできない kernel があります。** 64 ビットの整数を使う kernel は、`dispatch` / `gpu.submit` では CPU で実行し、`gpu.dispatch` ではコンパイルエラーになります。simdgroup の集計（`group.simdSum` など）を使う kernel はコンパイルエラーです。
- **JSPI（WebAssembly の実行を一時停止して Promise を待つ機能）が必要です。** WebGPU の読み戻しは非同期でしか行えないためです。JSPI か WebGPU がない環境では GPU なしとして動き、`dispatch` は CPU で実行されます。デスクトップ版の Chrome / Edge 137 以降で使えます。
- **確かめた範囲**: 生成する WGSL は、ネイティブの wgpu（Windows の NVIDIA、Vulkan と Direct3D 12）で `tests/metal` と `tests/wgpu` の kernel が CPU と一致することを確かめています。ブラウザでの実行は、以前の版の WGSL を Android の Chrome 系ブラウザ（Adreno GPU）で一度試しただけで、今の版では確かめていません。
- **生成される WGSL を見るには** `node src/cli.ts build x.um --emit-wgsl` とします。

## MSL（Metal Shading Language）の出力

`node src/cli.ts build x.um --emit-msl` で、kernel ごとの MSL が `<出力>.<kernel>.metal` に書き出されます（`UM_METAL=1` のビルドに入るものと同じ変換です）。

- **Metal での確認**: Metal コンパイラ（Apple M4）で、テストと例の 51 個の kernel の MSL がすべてコンパイルできることを確かめました（2026-11-15）。GPU での実行は、`tests/metal` の kernel が CPU と同じ結果になることを Mac で確かめています（`node tests/run.ts` が Mac で実行します）。
- **Metal のない環境での確認**: 生成した MSL を、スタブヘッダ（`tests/msl/metal_stub.h`）つきで C++14 として `clang++` でコンパイルし（`tests/gpu` と `examples` の kernel すべて）、`tests/msl/kernels.um` の kernel は C++ として全添字で実行して、CPU の結果とチェックサムを比べます（`node tests/run.ts` が `clang++` があれば行います）。C++ として通ることは Metal の仕様に合っていることを意味しないので、Metal 固有の規則は Mac での実行で確かめます。

## スレッド

`thread.spawn(f, 引数...)` で関数を新しいスレッドで実行し、`thread.join(h)` で終わるのを待って戻り値を受け取ります。`thread.spawnEach(f, own items, 引数...)` は配列の要素ごとに1本ずつ起動し（k 番目は `f(items[k], 引数...)`、要素は移動）、`thread.join` は戻り値の配列を items と同じ順で返します。`thread.cpus()` は論理 CPU 数、`thread.yield()` は CPU をほかの動けるスレッドに譲ります（RacyBuffer の印を見ながら待つループ用。ほかに動けるスレッドがなければすぐ戻る。wasm32 では何もしない）。

```ts
function search(w: own Worker, sh: Shared, root: Position): Worker { ... }

const sh: Shared = { tt: new RacyBuffer<u64>(n), stop: new RacyBuffer<u32>(1) };
{
  const g = thread.spawnEach(search, own workers, sh, root);  // 補助スレッド
  ...                                                          // メインはメインの仕事（入出力、時間の管理）
  sh.stop.store(0, 1);
  workers = thread.join(g);                                    // Worker ごと返ってくる
}
```

- **持ち手（`thread.spawn` の結果）は、`const` の宣言の初期化式としてだけ書けます。** 移動・代入・`return`・引数・フィールド・配列の要素にはできません（型 `Thread<R>` / `Threads<R>` を書けるのは、その `const` の注釈だけ）。`thread.join` は持ち手を消費し、合流しなかった持ち手は、宣言を含むブロックの終わり（`return` / `break` / `continue` で抜けるときも）で自動で合流します（戻り値は捨てる）。
- **スレッドへの渡し方**: コピーされる型は写し、`own` は移動（スレッドが所有して返すか解放する）、修飾なしのヒープを持つ値は読み取り借用で共有します。`mut` の引数を持つ関数は起動できません。読み取り借用で貸した変数は、`const` の宣言からブロックの終わりまで（`join` のあとも）書き換え・移動・`mut` 渡しができません。合流後に書き換えるなら、起動を内側のブロック `{ ... }` に入れます。一時値は貸せません（`const` に入れてから）。
- **複数のスレッドが書き換えられるのは `RacyBuffer` の要素だけです。** kernel の外では、`RacyBuffer` の要素は読み取り借用からでも書けます（長さと領域は変わらない）。要素の読み書き `b[i]` は順序の保証がないアトミック命令（kernel の外は relaxed）で、値は実行ごとに変わりえますが、メモリは壊れません。`for...of` と `clone()` も要素を1つずつアトミックに読みます。要素の型は `i8` / `u8` / `i16` / `u16` / `i32` / `u32` / `f32` / `i64` / `u64` です（kernel の中の `i64` / `u64` は CPU と Metal だけ。上の「kernel の中の 64 ビットの整数」）。8 / 16 ビットの要素も、隣の要素を別のスレッドが同時に書いて混ざることはありません。kernel は `mut RacyBuffer` のどの要素にも読み書きできます。
- **アトミック演算**: `b.load(i)`、`b.store(i, v)`、`b.fetchAdd(i, v)`（前の値を返す）、`b.compareExchange(i, expected, desired)`（前の値を返す）。順序は seq_cst で、`store` より前の書き込み（`b[i]` を含む）は、その値を `load` で見たスレッドから見えます。「データを書いてから印を立てる」受け渡しにはこれを使います（`b[i]` どうしには順序がない）。`fetchAdd` / `compareExchange` は整数だけです。kernel の中でも使えます（32 ビットの要素だけ。GPU では順序は relaxed。上の「グループ（共有配列・barrier）と kernel の中のアトミック演算」）。
- **kernel の `mut RacyBuffer` には、読み取り借用の `RacyBuffer` もそのまま渡せます**（`gpu.submit(k, n, mut sh.out, sh.inp)`。kernel は領域を差し替えられないため）。`dispatch` / `gpu.submit` / `gpu.ready` / `gpu.wait` / `gpu.flags` / `gpu.commit` はどのスレッドからでも呼べ（`gpu.flags` の印はスレッドごと）、券はどのスレッドで待ってもかまいません。
- **`print` 1回の出力（1行）はほかのスレッドと混ざりません。** `readLine()` / `inputReady()` はメインスレッドだけで使えます（ほかで呼ぶと停止）。どのスレッドで停止しても、プログラム全体が止まります（メッセージは1回）。
- 同時に存在できるスレッドは 256 本まで（超えると停止）。スタックは 64 MiB（`UM_THREAD_STACK_MB` で変更）。起動と合流の手間は1本 10〜15 マイクロ秒（Apple M4）。
- **kernel を CPU で並列に実行するスレッドプール**は、既定で論理 CPU 数のスレッド（呼んだスレッドを含む。最大 64）を使います。`UM_THREADS` で数を変えられます。プールのスレッドは仕事のあと 100 マイクロ秒は寝ずに次を待ちます（`UM_POOL_SPIN_US` で変更。0 ならすぐ寝る）。プールは同時に1つの仕事だけなので、ほかのスレッドが使っている最中の `cpu.dispatch` は、呼んだスレッドだけで実行します。
- **wasm32 にはスレッドがありません。** 書き方はすべて受け付け、合流の時点（`thread.join` か自動の合流）で関数をその場で最後まで実行します（`spawnEach` は要素の順、`thread.cpus()` は 1）。ほかのスレッドの結果を待って回るプログラム（集めて GPU に投げる係など）は wasm32 では終わりません。

例: [examples/threads.um](../examples/threads.um)、[examples/racy_shared.um](../examples/racy_shared.um)、[examples/racy_buffer.um](../examples/racy_buffer.um)。

## 安全性

`unsafe` はまだないので、書けるコードはすべて次の保証の中にあります。

| 起こりうること | 動作 |
| --- | --- |
| 配列の範囲外アクセス | 停止する（定数の添字はコンパイルエラー） |
| 整数オーバーフロー | ラップする。未定義動作にはならない |
| ゼロ除算（整数） | 停止する |
| 最小値 ÷ -1 | ラップする |
| ビット幅以上のシフト | 停止する |
| 浮動小数点から整数への変換で範囲外 | 飽和する。NaN は 0 |
| 未初期化の変数やフィールド | 書けない（初期化が必須。コンストラクタは全フィールドの初期化を検査） |
| 解放後の使用・二重解放 | 起きない（移動済みの変数を使うとコンパイルエラー） |
| 解放漏れ | 起きない（所有者がスコープを抜けると解放。停止したときを除く） |
| 反復中・借用中の配列を伸ばして要素の置き場所を無効にする | コンパイルエラー |
| 空の配列からの `pop` | 停止する |
| メモリ不足、確保サイズのあふれ | 停止する |
| kernel の中の範囲外アクセス・ゼロ除算・過大なシフト | 停止せず、決まった結果になる（上の「kernel と Buffer」） |
| 複数の実行が同じ要素に書く（データ競合） | 起きない（kernel が書けるのは自分の `[i]` だけ。`RacyBuffer` はアトミック命令） |
| スレッドどうしのデータ競合 | 起きない（共有して書き換えられるのは `RacyBuffer` の要素だけで、読み書きはアトミック命令） |
| スレッドが読んでいる値の解放・書き換え | コンパイルエラー（貸した変数はブロックの終わりまで凍結され、合流は解放より先） |
| GPU が使っている Buffer の解放 | 解放の前に、その Buffer を使う GPU の仕事の終わりを待つ |
| 可変借用中の変数を同じ呼び出しの別の引数や受け手に渡す | コンパイルエラー |
| スタックあふれ | 停止する（wasm は実行環境の検査、ネイティブはガードページとスタックプローブ） |

停止は、ネイティブでは終了コード101と `panic: 理由 (line N)`、wasm では `UmPanic` 例外になります。ほかのスレッドが動いているときの停止は、`atexit` の後始末をせずに終わります（`_exit`）。

## エラーメッセージ

この言語のコードは主に AI が書く前提なので、診断には原因コード・位置・直し方のヒントを付けます。TypeScript の習慣で書かれがちなもの（`number`、`any`、`try`、`extends`、アロー関数など）には、置き換え先を示します。

```
$ node src/cli.ts check bad.um
bad.um:1:15: error[ts-feature]: type 'number' is not available
  hint: there is no `number`; pick a sized type: `i32`, `i64`, `f32` or `f64`
```

コンパイルエラーになるべきコードの例は `tests/errors/` にあります。

## まだないもの

- **言語**: 関数値（アロー関数、`map` / `filter` などのコールバック）、グローバル変数（最上位の `let`。値の変わらない最上位の `const` と固定長配列の定数は使えます）、`struct` の定数と可変長配列の定数、定数を関数の呼び出しで作ること（コンパイル時の関数の実行）、型の別名（`type`）、`static`、`readonly`、getter / setter、`async` / `await`、`unsafe`、interface を値の型として使うこと、ジェネリックなメソッド、ジェネリック関数の export、`switch` の case の途中の `break`、行列の演算子。
- **配列**: ヒープを持つ要素の配列の `.clone()`、固定長配列にヒープを持つ要素を入れること、配列の一部を指すビュー、`f4` を1バイトに2個詰めること、2^32 要素以上の Buffer。
- **kernel と GPU**: `f16` / `f8` / `f4` の Buffer と kernel、ループをまたいで dispatch をまとめること（Metal。いまは隣り合った dispatch 文だけ）。
- **入出力**: 上の「入出力」にある関数だけです。ファイルの一部だけの書き換え、フォルダの操作、環境変数の読み取り、ネットワークはありません。
- **wasm32**: スレッド（合流の時点でその場で実行します）、simd128。

次のものは、TypeScript にはありますが、この言語では持たない方針です（書くとエラーで置き換え先を示します）: 継承（`extends`、`protected`、`abstract`）、例外（`try` / `throw`。失敗は `Result<T, E>` で返します）、`var`、`for...in`、`T | null` 以外の union 型、省略できるフィールドや引数、引数の既定値、`number` / `any`。

## マスクつきシフトと範囲チェックの除去

変数ぶんのシフトと配列の添字には、ふつう実行時の検査が入ります（シフト量がビット幅以上なら停止、範囲外の添字でも停止）。速さが要る所では、次の2つでこの検査を省けます。

**マスクつきシフト `Math.shl(x, k)` / `Math.shr(x, k)`。** シフト量 `k` を値のビット幅で割った余り（`k & (ビット幅 - 1)`）にしてからずらします。範囲検査を付けないので分岐が消えます。意味は kernel の `<<` / `>>` と同じで、CPU・GPU（Metal / WebGPU）・wasm で揃います。`k` は負でも大きくてもよく、値の型とちがう整数型でもかまいません（`Math.shl(1 as u64, sq)` のように `sq` は `i32` のままでよい）。右シフトは値の符号で決まります（符号つきは算術、符号なしは論理）。ビットボードのように「シフト量が必ず収まる」と書き手が分かっている所に使います。

```ts
const bit = Math.shl(1 as u64, sq);   // sq が 64 以上でも停止しない（64 で割った余り）
const hi  = Math.shr(x, 48);
```

**範囲内と証明できる添字の検査を省く。** コンパイラが「添字が必ず範囲内」と分かる所では、添字の範囲検査を自動で省きます（結果は検査ありと同じ。危ないので、確実なときだけ省きます）。分かる形:

- **固定長配列 `T[N]`** への添字で、値が `[0, N)` に必ず収まるとき。たとえば `a[x & 15]`（`T[16]` 以上）、`a[Math.ctz(w)]`（`T[65]` 以上。`ctz` / `clz` は 0〜ビット幅）、`a[x % N]`。
- **前の比較で分かるとき。** `if (i < N) { a[i] }` や `while (i < n) { ... }`、早い `return` / `break` のあと（`if (i >= N) return;` の後は `i < N`）。符号つきの変数は、下限も分かるときだけ（符号なしは常に 0 以上）。
- 表を添字の検査なしで読みたいときは、最上位の定数の配列（上の「定数の配列」）にします。長さがコンパイル時に分かるので、固定長配列と同じく検査を省けます。
- ヒープ配列（`T[]` / `Buffer`）は長さが実行時に決まるので、この自動除去の対象にしません（`for (let i = 0; i < a.length; i++)` のループは別の仕組みで検査を外へ出します。下の「速度」）。

危険な「全部の検査を外す」一括の設定は入れていません。

例: [examples/masked_shift.um](../examples/masked_shift.um)。

## 一括の書き込み（fill）

`xs.fill(v)` は全要素を、`xs.fill(v, start, end)` は `[start, end)`（end は含まない）の要素を `v` にします。可変長配列・`Buffer`・`RacyBuffer` で使え、長さは変わりません。

```ts
tt.fill(0);              // 置換表のような大きな表を消す（RacyBuffer<u64>）
xs.fill(-1, 10, 20);     // xs[10] 〜 xs[19] を -1 に
```

- **要素は数と `bool` だけ**（struct などはループで書く）。範囲が `0 <= start <= end <= 長さ` でなければ停止します（添字の範囲外と同じ）。`start` / `end` はどの整数型でもかまいません。
- **`RacyBuffer` は、要素の書き込みと同じく読み取り借用からでも書けます。** 要素ごとに不可分な（relaxed の）書き込みなので、同時に読むほかのスレッドに見えるのは前の値か新しい値だけです（ちぎれた値は見えない）。普通の配列と Buffer は `mut` が要ります。
- **16 MiB 以上は、ネイティブでは複数のスレッドで書きます**（kernel を CPU で並列に実行するのと同じスレッドプール。ほかのスレッドがプールを使っている最中なら、呼んだスレッドだけで書く）。全バイトが同じ値（0 など）の普通の配列は `memset` です。wasm32 では1つのスレッドで書きます。
- Buffer に `clear()` はありません（長さが固定のため）。全部を 0 にするには `b.fill(0)`。
- 速さ（Apple M4、`RacyBuffer<u64>` を 0 にする。要素ごとのループとの比較）: 256 MiB 8.6 → 3.3 ms、1 GiB 35 → 14 ms、2 GiB 70 → 27 ms（約 2.6 倍）。
- **Buffer / RacyBuffer の長さ（要素数）は `u32` のままです**（`b.length` の型。GPU の dispatch の個数と添字が 32 ビットのため）。要素の大きさを掛けたバイト数は 64 ビットで扱うので、`RacyBuffer<u64>` なら 2^32 - 1 要素（32 GiB 近く）まで作れます（4 GiB の `RacyBuffer<u64>` の確保と fill を確かめた）。2^32 要素以上の Buffer はまだありません。可変長配列 `T[]` の長さは `usize`（ネイティブでは 64 ビット）です。
- 例: [examples/fill.um](../examples/fill.um)。

## SIMD のベクトル

128 ビットのベクトルの型 `i8x16` / `u8x16` / `i16x8` / `u16x8` / `i32x4` / `u32x4` / `f32x4` と、その演算 `Simd.*` があります。LLVM のベクトル命令になり、arm64 では NEON（`sdot`、`smull` など）、wasm32 などでは同じ意味の命令（またはレーンごとの計算）になります。**CPU だけで使えます**（kernel ではエラー。GPU の kernel はもともと要素ごとに並列）。

```ts
// ニューラルネットの差分更新（i16）とアフィン層の内積（i8）
for (let j: u32 = 0; j < n; j += 8) {
  Simd.store(mut acc, j, Simd.load(acc, j) + Simd.load(w, add * n + j) - Simd.load(w, sub * n + j));
}
let s: i32x4;
for (let j: u32 = 0; j < n; j += 16) { s = Simd.dot(s, Simd.load(x, j), Simd.load(w, j)); }
const y = Simd.sum(s);
```

- **型は同じ要素の固定長配列の別名です**（`i16x8` は `i16[8]` と同じ型）。`v[k]` でレーンを読み書きでき、配列リテラルで作れ、struct のフィールドや配列の要素（`i16x8[]`）にもできます。`vec2` / `vec3` / `vec4` と同じく値型で、`+ - *` が使えます。
- **演算**（`a`、`b` は同じ型のベクトル。整数はラップ、`f32` は IEEE の1回の丸め）:

| 書き方 | 意味 |
| --- | --- |
| `a + b`、`a - b`、`a * b`（`Simd.add` / `sub` / `mul`） | レーンごと |
| `Simd.min(a, b)` / `Simd.max(a, b)` | レーンごと。`Math.min` / `Math.max` と同じ「比較して選ぶ」定義（符号なしの型は符号なしで比べる。NaN があれば `b`） |
| `Simd.splat(x)` | 全レーンが x。型は求められている型か、x の型から決まる（`const v: i16x8 = Simd.splat(3)`） |
| `Simd.shuffle(a, b, [i0, ..., iL-1])` | レーン k は `ik < L` なら `a[ik]`、そうでなければ `b[ik - L]`。番号は整数のリテラル |
| `Simd.dot(acc: i32x4, a: i8x16, b: i8x16)` | `acc[k] + a[4k]*b[4k] + … + a[4k+3]*b[4k+3]`（NEON の `sdot`） |
| `Simd.mulAddWide(acc: i32x4, a: i16x8, b: i16x8)` | `acc[k] + a[2k]*b[2k] + a[2k+1]*b[2k+1]`（i32 でラップ） |
| `Simd.sum(v)` | 全レーンの和（整数は要素の型でラップ。`f32` は lane 0 から順に足す） |
| `Simd.load(xs, i)` / `Simd.store(mut xs, i, v)` | `xs[i]` から L 個をまとめて読む / 書く。`xs` は可変長配列・`Buffer`・固定長配列（`RacyBuffer` は不可）。`i + L` が長さを超えると停止 |

- **どのターゲットでも同じ値です。** macOS の arm64 では `Simd.dot` に `sdot` を直接使い、それ以外では同じ値になる汎用のベクトル命令にします（`tests/simd/lanes.um` が両方の経路をスカラーの計算と突き合わせる。コンパイル時に `UM_GENERIC_SIMD=1` で汎用の経路を選べる）。wasm32 では LLVM がレーンごとの計算にします（simd128 はまだ使っていない）。
- **速さ**（Apple M4、小さな計測）: Apple clang 21 は、範囲検査が外に出せる素直なループ（`s += (w[j] as i32) * (x[j] as i32)`）を自分で `sdot` にベクトル化するので、SIMD で書いても同じくらいです（1024 × 16 の内積: ループ 190〜340 ns、`Simd.dot` で和を 2 本に分けて 200〜230 ns。1 本だけだと 400 ns と遅い。i16 1024 個の差分更新: ループ 38〜55 ns、SIMD 44〜56 ns）。`Simd.load` の範囲検査もループの外に出ます。自動のベクトル化に頼らずに書きたいとき（シャッフル、横方向の和、古い clang）に使います。
- 例: [examples/simd.um](../examples/simd.um)。

## 先読み

`Math.prefetch(arr, i)` は、`arr[i]` のあるメモリを、使う前にキャッシュへ読み込むよう CPU に頼みます（LLVM の `llvm.prefetch`。読むための先読みで、局所性は最大。arm64 では `prfm pldl1keep`）。置換表のような大きな表で、次に読む場所が先に分かるときに使います。

```ts
Math.prefetch(table, slot * 4);   // 次に読む場所を先に頼んでおく
doOtherWork();                    // そのあいだに別の仕事
const e = table[slot * 4];        // 読むころには届いている
```

- **結果は返さず、プログラムの動作は変わりません。** 範囲外の添字でも何も起きません（検査も停止もしない）。ただし配列そのものの場所（`tables[k]` の `k` など）はふつうに検査します。
- `arr` は可変長配列・`Buffer`・`RacyBuffer` の変数（またはそのフィールドや要素）。固定長配列と一時値は使えません。添字はどの整数型でもかまいません。
- **wasm32 では何もしません**（添字の式は評価します）。**kernel の中では使えません**（GPU には先読みがない）。
- 効果（Apple M4、1 GiB の表をでたらめな順に読む小さな計測）: 読むあいだに 20〜40 回の掛け算の仕事があると、1回の読み出しが 約 33 → 27 ns、55 → 45 ns（約 -18%）。仕事がないと速くならない（CPU が先の読み出しを自分で重ねるため）。
- 例: [examples/prefetch.um](../examples/prefetch.um)。

## 速度

同じ計算を C で書いたものと比べると（Linux x86_64 での計測）、多くの項目が C の `-O2` と同等です。添字アクセスの境界検査が残るループ（隣の要素を読むステンシル計算など）は、C より2〜6割ほど遅くなります。`for...of` と kernel には境界検査の分岐が入りません。

## テスト

```sh
node tests/run.ts
```

`examples/` の各プログラムを wasm32 とこのマシンのネイティブで実行して出力を照合し（`// expect:` 行と比べます。`// stdin: 行` と書いた行は標準入力として与えます）、arm64 macOS 向けのオブジェクト生成を確認します。AddressSanitizer か valgrind がある環境では（Mac では AddressSanitizer と `leaks`）、メモリ検査つきでも実行して、解放後使用・範囲外アクセス・二重解放・解放漏れがないことを確かめます。あわせて、エラーになるべきコード（`tests/errors/`）と停止するべきコード（`tests/panics/`）を確認します。スレッドのテスト（`tests/threads/`、ネイティブだけ）は、ThreadSanitizer つきでも実行してデータ競合がないことを確かめます。`tests/metal/` は、Mac では Metal で実行して CPU の結果と比べ、それ以外では arm64 向けに作れることだけを確かめます。

- 個別に実行するオプションはありません。1つだけ確かめるときは、そのファイルをビルドして実行し、`// expect:` 行と出力を見比べます。
- wasm32 のテストは、wasm32 を出せる clang がない環境（Mac の Xcode の clang など）では飛ばされます。

## 構成

```
src/lexer.ts     字句解析
src/parser.ts    構文解析
src/checker.ts   型検査と借用の検査、kernel の制約、定数の評価
src/codegen.ts   LLVM IR（テキスト）の生成
src/msl.ts       kernel から MSL（Metal）への変換
src/wgsl.ts      kernel から WGSL（WebGPU。実験的）への変換
src/driver.ts    clang の呼び出し
src/cli.ts       コマンド
runtime/         ネイティブ（native.c）、Metal（metal.m）、wasm（wasm_rt.c と host.mjs。WebGPU での実行を含む）のランタイム、標準の関数（prelude.um）
assets/logo/     ロゴ（SVG と PNG）
examples/        動くサンプル（テストを兼ねる）
tests/           テストランナーと、エラー系・停止系・スレッド・Metal・MSL などのケース
docs/            このガイド
```

プロジェクト自体のセキュリティ方針は [SECURITY.md](../SECURITY.md) にあります。
