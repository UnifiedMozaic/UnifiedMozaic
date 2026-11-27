# ライブラリ（`um:`）

UM の標準は、どの環境（ネイティブ・wasm32）でも使える最低限のものだけです。それ以外の部品は、UnifiedMozaic に付いてくる
ライブラリ（`lib/` のフォルダ）にあり、使うプログラムが名前を挙げて読み込みます。

```ts
import * from "um:window";                         // export されたすべての名前
import { drawText, textWidth } from "um:text";     // 挙げた名前だけ
import { mat4Perspective, mat4LookAt } from "um:transform";
```

`um:名前` は `lib/名前.um` を指します。相対パスの import と違って、どのフォルダのプログラムからも読めます。
`import { … }` は挙げた名前だけを、`import * from` は export されたすべての名前（組み込みの関数も）を使えるようにします。
読み込んでいないライブラリの名前を使うと、エラーのヒントに足すべき import の行が出ます。

## 標準とライブラリの線引き

| | 中身 |
| --- | --- |
| 標準（import なし） | 言語の機能（型・演算・三角関数・`vec` と `mat` の型と基本の演算・`Result`・スレッド）、GPU（`gpu.dispatch`・`gpu.draw` と描き方、表示の面に描く `gpu.beginFrame`）、どこでも使える基本の入出力（ファイル・標準入出力・`args`） |
| ライブラリ（`um:`） | UM で書ける部品と、外部のもの（SDL3・ブラウザの canvas・FFmpeg・wasmtime など）に頼る機能 |

## 一覧

| ライブラリ | 中身 | 使える所 | 説明 |
| --- | --- | --- | --- |
| `um:window` | ウィンドウ・入力・音・時計（`windowOpen`・`pollWindowEvent`・`audioOpen`・`ticksNs` など。`windowSurface` で `gpu.beginFrame` の表示の面を作る） | ネイティブ（SDL3）とブラウザ（canvas・Web Audio） | [ウィンドウと音](window.ja.md) |
| `um:transform` | 3D の変換の行列（`mat4Translate`・`mat4Scale`・`mat4RotateX/Y/Z`・`mat4Perspective`・`mat4Ortho`・`mat4LookAt`） | どこでも（kernel・シェーダでも） | [描画](render.ja.md) |
| `um:texture` | テクスチャを読む（`textureSample`・`textureFetch`・`colorFromArgb`） | どこでも（kernel・シェーダでも） | [描画](render.ja.md) |
| `um:text` | 5×7 の点の文字を描く（`drawText`・`textWidth`） | どこでも | [描画](render.ja.md) |
| `um:png` | PNG の読み込み（`Bitmap`・`imageLoad`・`imageDecodePng`） | どこでも | [画像ファイルの読み込み](image.ja.md) |
| `um:http` | HTTP(S) の要求（`httpRequest`・`httpPoll` と組み込みの `httpStart`・`httpReadLine` など） | ネイティブ | [HTTP(S)](http.ja.md) |
| `um:media` | 動画・音声の読み書き（FFmpeg） | ネイティブ | [動画・音声](media.ja.md) |
| `um:wasm` | WebAssembly のモジュールの実行（wasmtime） | ネイティブ | [WebAssembly の実行](wasm.ja.md) |

`um:http`・`um:media`・`um:wasm` の組み込みの関数（`httpReadLine`・`mediaClose`・`wasmWrite` など、ランタイムが直接持つもの）も、
同じように import します（`import { httpReadLine } from "um:http";`）。

## 決まり

- ライブラリの中の名前も、プログラムの名前と同じ 1 つの空間にあります。ライブラリの中だけで使う名前（`pngBits` など）と
  同じ名前をプログラムが定義すると、`duplicate` のエラーになります（黙ってどちらかが使われることはありません）。
- ライブラリは、ほかのライブラリを `um:` で読めます（`um:text` は `um:texture` を使う）。相対パスの import はできません。
