# WebAssembly の実行

[English](wasm.md)

ネイティブのプログラムは、WebAssembly のモジュール（`.wasm`）を読み込んで、その関数を呼べます。
中身は [wasmtime](https://wasmtime.dev/) の C API（JIT でネイティブのコードにしてから実行する）で、
ネイティブのコードに近い速さで動きます。プラグインを OS に依存しない形で配り、本体から隔離して動かすのに使えます。

`import { wasmLoad, wasmNew, wasmCall } from "um:wasm";` のように、ライブラリ `um:wasm` から読み込んで使います（[ライブラリ](libraries.ja.md)）。

- モジュールはインスタンスの中だけで動きます。既定では、ホストの関数（import）も WASI も渡さないので、
  ファイル・ネットワーク・本体のメモリには触れられません。
- wasm の中の異常（トラップ）、燃料切れ、時間切れ、読み込みの失敗では、プログラムは止まらず、`Result.Err(理由)` が返ります。
- 複数のスレッドから、それぞれ別のインスタンスを同時に使えます。
- wasm32 のプログラムでは使えません（下の「wasm32 では」）。

## 準備

wasmtime の C API が必要です。WebAssembly の組み込みを使うプログラムをビルドしたときだけリンクします。
[wasmtime のリリース](https://github.com/bytecodealliance/wasmtime/releases) から、使う OS の C API を取ってきて展開し、
環境変数 `UM_WASMTIME` にその場所（`include/wasmtime.h` と `lib` のある場所）を設定します。

| OS | ファイル |
| --- | --- |
| Windows（x64） | `wasmtime-v*-x86_64-windows-c-api.zip` |
| Windows（ARM64） | `wasmtime-v*-aarch64-windows-c-api.zip` |
| macOS（Apple Silicon） | `wasmtime-v*-aarch64-macos-c-api.tar.xz` |
| macOS（Intel） | `wasmtime-v*-x86_64-macos-c-api.tar.xz` |
| Linux（x64） | `wasmtime-v*-x86_64-linux-c-api.tar.xz` |
| Linux（ARM64） | `wasmtime-v*-aarch64-linux-c-api.tar.xz` |

```powershell
# Windows
$env:UM_WASMTIME = "C:\wasmtime-v51.0.0-x86_64-windows-c-api"
```

```sh
# macOS・Linux
export UM_WASMTIME=$HOME/wasmtime-v51.0.0-aarch64-macos-c-api
```

探す順番は、`UM_WASMTIME` → `pkg-config wasmtime`（ソースから cmake でビルドして入れた場合）→ macOS の Homebrew の決まった場所
（`/opt/homebrew/opt/wasmtime`）です。

- Windows では `lib/wasmtime.dll.lib` でリンクし、`lib/wasmtime.dll` を実行ファイルの隣に自動で写します。配るときは、実行ファイルと一緒に `wasmtime.dll` を置いてください。
- macOS・Linux では `lib` の共有ライブラリ（`libwasmtime.dylib` / `libwasmtime.so`）をリンクし、その場所を rpath に入れます。配るときは、ライブラリを実行ファイルと一緒に置いて rpath を書き換えるか（`install_name_tool` / `patchelf`）、`lib` の静的ライブラリ（`libwasmtime.a`）でリンクしてください（静的リンクはまだドライバの選択肢にありません）。
- 確かめた組み合わせは、Windows 11（x64、clang の MSVC ターゲット）と Linux（WSL の Ubuntu 24.04、x64、clang 18）、どちらも wasmtime 51.0.0 と `UM_WASMTIME` です
  （Windows では `node tests/run.ts` のすべて、Linux では examples/wasm と tests/wasm）。macOS は、同じ探し方でリンクできるはずですが、まだ実機で確かめていません。`pkg-config` での探し方もまだ確かめていません。

## 例

```sh
node src/cli.ts build examples/wasm/add.um --target host -o add
./add
```

| 例 | 内容 |
| --- | --- |
| [examples/wasm/add.um](../examples/wasm/add.um) | 関数を呼ぶ（i32 / i64 / f64）、export の一覧、型の違う引数・ない関数の失敗 |
| [examples/wasm/pixels.um](../examples/wasm/pixels.um) | 画素を wasm のメモリへ写して書き換え、写し戻す（`u8[]`・`Buffer<u8>`・`Buffer<f32>`・配列の一部） |
| [examples/wasm/limits.um](../examples/wasm/limits.um) | トラップ（ゼロ除算・unreachable・範囲外）、燃料・時間・メモリの上限、import を持つモジュール |
| [examples/wasm/threads.um](../examples/wasm/threads.um) | 画面を帯に分け、スレッドごとのインスタンスで同時に処理する |
| [examples/wasm/custom_section.um](../examples/wasm/custom_section.um) | カスタムセクションを名前で読む、ファイルから読み込む |
| [examples/wasm/um_plugin.um](../examples/wasm/um_plugin.um) | UM で書いた効果（[plugin/brighten.um](../examples/wasm/plugin/brighten.um)）を `--target wasm32-plugin` で wasm にして読み込み、ネイティブで直接実行した結果と比べる（下の「UM で書いたプラグイン」） |
| [examples/wasm/bench/](../examples/wasm/bench/bench.um) | 速さの測定（下の「速さ」） |

例のモジュールは、テキスト形式（`.wat`）から `wasmFromWat` で作っています。

## 使い方

```ts
import { wasmLoadFile } from "um:wasm";
export function main(): i32 {
  match (wasmLoadFile("effect.wasm")) {             // Result<i64, string>: モジュールの番号
    Result.Err(why) => {
      eprint(why);
      return 1;
    }
    Result.Ok(module) => {
      let limits = new WasmLimits();
      limits.fuel = 1000000000;                     // 呼び出し1回あたりの燃料（0 は上限なし）
      limits.memoryBytes = 256 * 1024 * 1024;       // メモリの上限
      match (wasmNew(module, limits)) {             // Result<i64, string>: インスタンスの番号
        Result.Err(why) => eprint(why),
        Result.Ok(inst) => {
          let px = new Buffer<u8>(1280 * 720 * 4);
          wasmWrite(inst, 65536, px);               // UM → wasm のメモリの 65536 バイト目
          match (wasmCall(inst, "invert", [WasmValue.I32(65536), WasmValue.I32(1280 * 720)])) {
            Result.Err(why) => eprint(why),         // トラップなど。プログラムは止まらない
            Result.Ok(results) => {}
          }
          wasmRead(inst, 65536, mut px);            // wasm のメモリ → UM
          wasmFreeInstance(inst);
        }
      }
      wasmFreeModule(module);
    }
  }
  return 0;
}
```

モジュールとインスタンスは `i64` の番号で表します。番号には種類（モジュールかインスタンスか）と世代が入っていて、
解放した番号や種類の違う番号を渡すと停止します（使い方の誤りなので）。使い終わったら `wasmFreeInstance` / `wasmFreeModule` で解放します
（解放しなくても、プログラムの終わりに OS が回収します）。

## 関数

`lib/wasm.um`（UM で書いたライブラリ `um:wasm`）と、組み込みの関数（`runtime/wasm.c`）です。どれもネイティブだけで、どのスレッドからも呼べます。

### 読み込みとインスタンス

| 関数 | 説明 |
| --- | --- |
| `wasmLoad(bytes: u8[]): Result<i64, string>` | wasm のバイナリをコンパイルし、モジュールの番号を返す。コンパイル結果はディスクにキャッシュする（下の「キャッシュ」） |
| `wasmLoadFile(path: string): Result<i64, string>` | ファイルから読み込む |
| `wasmFromWat(text: string): Result<u8[], string>` | テキスト形式（`.wat`）をバイナリにする（テストや小さなモジュールに） |
| `wasmFreeModule(module: i64): void` | モジュールを解放する。作ったインスタンスはそのまま使える |
| `wasmNew(module: i64, limits: WasmLimits): Result<i64, string>` | インスタンスを作る。インスタンスごとに、別のメモリ・別のグローバル変数を持つ。start 関数があれば、ここで実行する |
| `wasmFreeInstance(instance: i64): void` | インスタンスを解放する（メモリも消える） |
| `wasmExports(module: i64): WasmExport[]` | export の一覧（宣言の順） |
| `wasmCustomSection(bytes: u8[], name: string): u8[] \| null` | バイナリから、名前が `name` のカスタムセクションの中身を取り出す（最初に見つかったもの）。なければ、または壊れていれば `null`。コンパイルしないので、プラグインの宣言を読むのに向く。wasm32 でも使える |

```ts
class WasmLimits {
  fuel: i64 = 0;          // 燃料。0 は上限なし
  timeoutMs: i64 = 0;     // 時間（ミリ秒）。0 は上限なし
  memoryBytes: i64 = 0;   // 線形メモリの上限（バイト）。0 は上限なし
  stubImports: bool = false;  // import を「呼ぶとトラップする関数」で埋める
}

struct WasmExport {
  name: string;
  kind: WasmExportKind;   // Function / Memory / Global / Table / Other
  params: WasmType[];     // 関数のときだけ。I32 / I64 / F32 / F64 / V128 / Ref
  results: WasmType[];
}
```

### 呼び出し

| 関数 | 説明 |
| --- | --- |
| `wasmCall(instance: i64, name: string, args: WasmValue[]): Result<WasmValue[], string>` | export された関数を呼ぶ。戻り値は複数のこともある（multi-value） |

```ts
enum WasmValue { I32(value: i32), I64(value: i64), F32(value: f32), F64(value: f64) }
```

- 引数の型と数は、関数の宣言と同じでなければなりません（`i32` の引数に `WasmValue.I64` は渡せない）。違えば `Result.Err` です。
- 引数・戻り値に `v128` や参照（`funcref` / `externref`）を持つ関数は呼べません（`Result.Err`）。引数と戻り値は、それぞれ 64 個までです。
- トラップ・燃料切れ・時間切れのあとも、同じインスタンスで次の呼び出しができます。ただし、途中で止まった関数が書きかけたメモリやグローバル変数はそのままなので、プラグインの状態が壊れている可能性があります。作り直すのが安全です。

### メモリ

| 関数 | 説明 |
| --- | --- |
| `wasmWrite(instance: i64, offset: i64, src: 配列): bool` | UM の配列をまとめて、インスタンスのメモリの `offset` バイト目へ写す |
| `wasmWrite(instance: i64, offset: i64, src: 配列, start: i64, count: i64): bool` | 配列の `start` 番目から `count` 個だけ写す（`count` が負なら終わりまで） |
| `wasmRead(instance: i64, offset: i64, dst: mut 配列): bool` | インスタンスのメモリの `offset` バイト目から、配列の長さぶんを写す |
| `wasmRead(instance: i64, offset: i64, dst: mut 配列, start: i64, count: i64): bool` | 配列の `start` 番目から `count` 個へ写す |
| `wasmMemorySize(instance: i64): i64` | メモリの大きさ（バイト）。メモリを export していなければ `-1` |

- 写す配列は、数の可変長配列か `Buffer` です（要素は `i8`〜`i64`、`u8`〜`u64`、`f32`、`f64`）。`u8[]`・`Buffer<u8>`・`Buffer<f32>`・`Buffer<u32>` など。配列は変数（かそのフィールド・要素）に入れて渡します。
- 写すのは、`memory` という名前で export されたメモリ（なければ、最初に見つかった export のメモリ）です。
- バイトの並びはそのまま写します（wasm もこのランタイムが動く CPU もリトルエンディアン）。
- wasm のメモリの外にはみ出す範囲は `false`（理由は `wasmError()`）、`start` / `count` が配列の外なら停止です。

### 低い層の組み込み

`lib/wasm.um` の関数は、次の組み込みで書いてあります。失敗は `null` / `false` / `-1` で返り、理由は `wasmError()` で読みます
（理由はスレッドごとに覚えていて、次に失敗するまで変わりません）。

| 関数 | 説明 |
| --- | --- |
| `wasmCompile(bytes: u8[]): i64 \| null` | `wasmLoad` の中身 |
| `wasmInstantiate(module: i64, fuel: i64, timeoutMs: i64, memoryBytes: i64, stubImports: bool): i64 \| null` | `wasmNew` の中身 |
| `wasmInvoke(instance: i64, name: string, kinds: mut i32[], ints: mut i64[], floats: mut f64[], nargs: i32): i32` | `wasmCall` の中身。引数 k は `kinds[k]`（0 = i32、1 = i64、2 = f32、3 = f64）の種類で、整数は `ints[k]`、小数は `floats[k]` から取る。戻り値を同じ形で先頭から書き、個数を返す（失敗は -1）。3つの配列は、引数と戻り値の数より長くしておく |
| `wasmExportList(module: i64): string[]` | `wasmExports` の中身（形は `runtime/wasm.c` を見てください） |
| `wasmWat(text: string): u8[]` | `wasmFromWat` の中身（失敗なら空） |
| `wasmError(): string` | このスレッドで最後に失敗した理由 |

## 安全

| 起こりうること | 動作 |
| --- | --- |
| wasm の中の異常（ゼロ除算、unreachable、メモリの範囲外、スタックあふれなど） | 呼び出しが `Result.Err("trap in '関数': 理由")` になる |
| 無限ループ | 燃料（`fuel`）か時間（`timeoutMs`）の上限で `Result.Err` になる |
| メモリを使いすぎる | `memoryBytes` を超える `memory.grow` は失敗（wasm に -1 が返る）。初期の大きさが上限を超えるモジュールはインスタンスを作れない |
| ファイル・ネットワーク・本体のメモリへのアクセス | できない（import も WASI も渡さない。wasm のメモリはインスタンスごとに別） |
| import を持つモジュール | 既定ではインスタンスを作れない（`Result.Err("... unknown import: ...")`）。`stubImports` なら作れるが、import を呼ぶとトラップ |
| 壊れたバイナリ | `wasmLoad` が `Result.Err` |
| 解放した番号・種類の違う番号 | 停止（使い方の誤り） |
| 同じインスタンスを2つのスレッドが同時に使う | 後から来たほうが失敗（`Result.Err` / `false`）。使っている最中のインスタンスの解放は停止 |

- **燃料**は、だいたい wasm の命令の数です。呼び出しのたびに `fuel` まで満たし直します（インスタンスを作るときの start 関数にも同じ上限）。結果が実行環境に左右されないので、テストや「止まらないプラグイン」の検出に向きます。
- **時間**は、呼び出し1回あたりのミリ秒です。wasmtime の epoch（別のスレッドが 1 ミリ秒ごとに進める時計）で測るので、精度は数ミリ秒です（Windows では OS の時計の精度により 15 ミリ秒ほど遅れることがある）。時計のスレッドは、時間の上限を使うインスタンスを初めて作ったときに起こします。
- 燃料と epoch の検査は、生成されるコードを遅くします（測ったところ、燃料と epoch の両方で 15〜35%、epoch だけで 10% ほど）。
  そこで、インスタンスの上限に合わせてコードを使い分けます: 上限なし → 検査なし、`timeoutMs` だけ → epoch の検査だけ、`fuel` あり → 両方。
  モジュールは読み込むときに「検査なし」でコンパイルし、ほかは初めて使うときにコンパイルします（ディスクのキャッシュが効きます）。
- 停止した理由の文は英語です（wasmtime の文を1行にまとめたもの）。版によって細かい言い回しが変わることがあります。

## スレッド

- エンジン（コンパイラ）とモジュールはスレッドの間で共有します。1つのモジュールから、スレッドごとにインスタンスを作ってください。
- インスタンスごとに wasmtime の Store を分けているので、別々のインスタンスは本当に同時に動きます（ロックなし）。
- 同じインスタンスを2つのスレッドで同時に使うことはできません。後から来たほうは失敗します（理由: `the instance is being used by another thread`）。
- 番号は `i64` なので、`thread.spawn(f, module, ...)` でそのまま渡せます。

```ts
function band(module: i64, y0: i32, y1: i32): u64 {
  match (wasmNew(module, new WasmLimits())) {   // スレッドごとに自分のインスタンス
    ...
  }
}

const t0 = thread.spawn(band, m, 0, 360);
const t1 = thread.spawn(band, m, 360, 720);
```

## UM で書いたプラグイン（`--target wasm32-plugin`）

UM のプログラムを、ホストに組み込んで使う wasm（プラグイン）にできます。同じソースを、ネイティブのプログラムに `import` して直接実行することもできます。

```sh
node src/cli.ts build examples/wasm/plugin/brighten.um --target wasm32-plugin -o brighten.wasm
```

```ts
// examples/wasm/plugin/brighten.um
kernel function brightenPixel(i: u32, dst: mut Buffer<u8>, src: Buffer<u8>, amount: i32): void { ... }

export function brighten(dst: mut Buffer<u8>, src: Buffer<u8>, amount: i32): void {
  dispatch(brightenPixel, dst.length, mut dst, src, amount);
}
```

ふつうの wasm32（ブラウザや `runtime/host.mjs` で動かすもの）との違いです。

- **import を一切持ちません。** WASI もホストの関数も渡さない wasmtime で、そのままインスタンスを作れます。
  ホストの関数が要るもの（`print`、`readLine` / `readFile` などの入出力、`String(x)` やテンプレート文字列での f64 の文字列化）を使うと、
  ビルドの時点でエラー（`plugin-import`）になります。
- **kernel の `dispatch` は CPU の1スレッドで実行します**（WebGPU の経路を使わない。`gpu.dispatch` はコンパイルエラー）。
  並列にしたいときは、ホストが画面を帯に分けて、帯ごとのインスタンスで呼びます。
- **export した関数は、Buffer を (要素の位置, 要素の数) の2つの i32 で受け取ります。** 引数に書けるのは、数（整数・f32・f64）・`bool`・
  `Buffer<T>`（借用か `mut`）で、戻り値は数・`bool`・`void` です。ほかの型（文字列、配列、`own` の Buffer）はエラー（`plugin-export`）です。
  上の `brighten` は、wasm からは `brighten(dstAt: i32, dstLen: i32, srcAt: i32, srcLen: i32, amount: i32)` に見えます。
- **領域はホストが確保します。** ランタイムが export する `umAlloc(bytes: i32): i32`（16 バイト境界）と `umFree(at: i32)` を使います。
  UM の関数はその領域を借りるだけで、解放も差し替えもしません（`mut` の Buffer に新しい Buffer を代入したら停止します）。
- **停止（範囲外の添字など）は `unreachable` のトラップになります。** ホストの `wasmCall` は `Result.Err("trap in '関数': unreachable executed")` を受け取り、
  そのあと `umPanicCode()` / `umPanicLine()` を呼ぶと、原因コードと行番号が分かります（原因コードは [言語ガイドの「安全性」](guide.ja.md#安全性) の停止と同じ。
  1 = 範囲外の添字、2 = ゼロ除算、5 = メモリ不足、15 = 借りた Buffer を差し替えた、など）。
- **SIMD（simd128）を使います。** ループがベクトル化されます（wasmtime は x86_64 と aarch64 で SIMD に対応しています）。

ホストでの使い方（[examples/wasm/um_plugin.um](../examples/wasm/um_plugin.um)）:

```ts
const srcAt = call(inst, "umAlloc", [WasmValue.I32(n)]);   // 画素の置き場所をプラグインの中に確保する
const dstAt = call(inst, "umAlloc", [WasmValue.I32(n)]);
wasmWrite(inst, srcAt as i64, src);                         // 画素を写す
wasmCall(inst, "brighten", [WasmValue.I32(dstAt), WasmValue.I32(n), WasmValue.I32(srcAt), WasmValue.I32(n), WasmValue.I32(40)]);
wasmRead(inst, dstAt as i64, mut dst);                      // 結果を写し戻す
```

確保した領域は、インスタンスを作り直すまで何度でも使えます（コマごとに確保し直す必要はありません）。

## キャッシュ

`wasmLoad` は、コンパイルした結果をディスクにキャッシュします（wasmtime のキャッシュ。同じバイト列なら、2回目からはコンパイルせずに読み込む）。
場所と大きさの上限は wasmtime の設定に従います（既定の場所は、Windows では `%LOCALAPPDATA%\BytecodeAlliance\wasmtime`、
macOS では `~/Library/Caches/BytecodeAlliance.wasmtime`、Linux では `~/.cache/wasmtime`）。`UM_WASM_CACHE=0` で切れます。

## 速さ

[examples/wasm/bench/](../examples/wasm/bench/) で測った、1280x720 の RGBA の1コマあたりの時間です。同じ C のソース（`effect.c`）を
clang で wasm32 とネイティブにコンパイルして比べました。wasm の時間には、画素を wasm のメモリへ写して戻す時間（1コマ 3.7 MB の往復）も含みます。

測った環境: AMD Ryzen 9 3900X（12 コア 24 スレッド）、Windows 11、clang 23、wasmtime 51.0.0。3 回測って一番速い値（ミリ秒 / コマ）。
wasm は上限なし（検査なしのコード）です。

| 効果 | ネイティブ C（-O2） | ネイティブ C（ベクトル化なし） | UM（ネイティブ、1 スレッド） | wasm、1 インスタンス | wasm + simd128、1 インスタンス | wasm、8 スレッド × 8 インスタンス |
| --- | --- | --- | --- | --- | --- | --- |
| invert（C） | 0.07 | 0.25 | 0.78 | 0.73 | 0.67 | 0.06〜0.08 |
| saturate（C） | 2.10 | 4.26 | 3.25 | 6.00 | 6.19 | 0.86 |

| 効果 | UM（ネイティブ、1 スレッド） | UM（ネイティブ、dispatch で全コア） | UM の wasm32-plugin、1 インスタンス |
| --- | --- | --- | --- |
| brighten（[plugin/brighten.um](../examples/wasm/plugin/brighten.um)） | 0.44 | 0.38 | 1.98 |

読み方:

- **写す時間が効きます。** 1 コマ（3.7 MB）を wasm のメモリへ写して戻すのに 0.6 ms ほどかかります。invert のような軽い効果では、時間のほとんどがこれです。
- **計算そのものは、ネイティブの 1.5〜3 倍ほどです**（saturate: ベクトル化なしのネイティブ 4.26 に対して 6.0）。ネイティブの -O2 はループをベクトル化（SSE）するので、さらに差が開きます。
- **帯に分けてスレッドごとのインスタンスで並べると、ほぼコアの数だけ速くなります**（saturate: 6.0 → 0.86、8 スレッド）。
- **燃料や時間の上限を付けると遅くなります**（燃料と時間で 15〜35%、時間だけで 10% ほど。上の「安全」）。
- **コンパイル**: 815 KB のモジュール（関数 3000 個）の `wasmLoad` は、初回 178 ms、キャッシュがあれば 12〜19 ms でした（`UM_WASM_CACHE=0` では毎回 144 ms）。

```sh
clang --target=wasm32 -O2 -nostdlib -Wl,--no-entry -Wl,--initial-memory=8388608 -o effect.wasm examples/wasm/bench/effect.c
clang --target=wasm32 -O2 -msimd128 -nostdlib -Wl,--no-entry -Wl,--initial-memory=8388608 -o effect_simd.wasm examples/wasm/bench/effect.c
clang -O2 -o native examples/wasm/bench/native.c examples/wasm/bench/effect.c && ./native
node src/cli.ts build examples/wasm/bench/bench.um --target host -o bench && ./bench effect.wasm 8
node src/cli.ts build examples/wasm/plugin/brighten.um --target wasm32-plugin -o brighten.wasm
node src/cli.ts build examples/wasm/um_plugin.um --target host -o um_plugin && ./um_plugin bench
```

## wasm32 では

wasm32 のプログラムでは、WebAssembly を実行する関数（`wasmLoad` など）はコンパイルエラー（`not-implemented`）です。
`wasmCustomSection` だけは、バイト列を読むだけなので使えます。

JS の `WebAssembly` で実装することもできますが、次の理由で見送りました。

- ブラウザの `WebAssembly.compile` / `instantiate` は大きなモジュールでは非同期でしか使えず、UM の同期的な呼び出しに合わせるには JSPI が要る。
- 燃料による上限がなく、無限ループを止めるには Worker ごと止めるしかない。
- 本体（wasm32）とプラグインのメモリが別なので、画素の受け渡しは JS を通した写しになり、ネイティブより遅くなる。
- いまの使い道（MozaicCut のプラグイン）はネイティブのプログラム。

## しくみ

- 組み込みの部分は `runtime/wasm.c`（C と wasmtime の C API）。`WasmValue`・`WasmLimits`・`wasmLoad`・`wasmCall` などは `lib/wasm.um`（UM）で、
  ライブラリ `um:wasm` として `import` で読み込みます。WebAssembly の組み込みを使わないプログラムは wasmtime をリンクしません。
- エンジンは1つで、最初に使ったときに作ります（燃料・epoch による中断・ディスクのキャッシュを有効にする）。
- 番号の表はモジュールとインスタンスでそれぞれ 4096 個までです（超えると `Result.Err`）。
- 停止の原因コードは 13（番号か配列の範囲が正しくない）と 14（使っている最中のインスタンスを解放した）です。
