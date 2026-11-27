# 画像ファイルの読み込み

PNG のファイルを読んで、画素の Buffer にします。読んだ画像は、そのままテクスチャ（[描画](render.ja.md)の `textureSample`）に使えます。
ライブラリ `um:png` にあります（[ライブラリ](libraries.ja.md)）。展開（zlib の inflate）も含めて UM で書いてあるので（`lib/png.um`）、ネイティブでも wasm32（ブラウザ）でも同じに動き、
同じ画素になります。

```ts
import { imageLoad } from "um:png";

const bricks = imageLoad("bricks.png");
if (bricks.error != "") {
  eprint(bricks.error);
  return 1;
}
// fragment function の中で: textureSample(tex, w, h, v.uv)
gpu.draw(mut frame, 36, vs(camera), fs(bricks.pixels, bricks.width as u32, bricks.height as u32));
```

| 関数 | 内容 |
| --- | --- |
| `imageLoad(path: string): Bitmap` | ファイルを読んで画像にする |
| `imageDecodePng(data: u8[]): Bitmap` | PNG のバイト列を画像にする（埋め込んだデータや、ネットから受け取ったデータに） |

```ts
struct Bitmap {
  width: i32;
  height: i32;
  pixels: Buffer<u32>;  // 0xAARRGGBB（不透明度は乗じていない）。上の行から、左から右に
  error: string;        // 読めなかった理由（読めたら空）。読めなかったときは幅と高さが 0
}
```

読めなかったとき、`null` ではなく `error` に理由の入った `Bitmap` を返します（`Buffer` を持つ struct は、まだ `null` にできないため）。

## 読める PNG

- 色の形式: グレー、RGB、パレット、グレーと不透明度、RGBA
- ビットの深さ: 1・2・4・8・16（16 ビットは上の 8 ビットにする。1〜4 ビットのグレーは 0〜255 に広げる）
- インターレース（Adam7）
- 透明の色（tRNS。パレットの不透明度、グレーと RGB の透明にする色）
- 大きさは幅・高さとも 32768 まで、画素の数は 2^26（約 6700 万）まで

色の補正（gAMA・cHRM・sRGB・iCCP）はしません（画素の値をそのまま使います）。CRC と Adler-32 の検査はしません。
JPEG などほかの形式はまだ読めません。

## ブラウザ

`--web` で作ったページ（`runtime/web/index.html`）では、`imageLoad("bricks.png")` は、ページと同じ場所のファイルを
`fetch` で読みます（JSPI で wasm を一時停止して待ちます）。画像のファイルを `index.html` と同じフォルダに置いてください。
自分のページに組み込むときは、`instantiate(bytes, { readFileAsync: async (path) => ... })` で読み方を渡します。

## 確かめたこと

- `tests/image/png.um`: 色の形式・ビットの深さ・インターレース・透明の色・圧縮の方式（圧縮しない・決まった符号・ブロックの頭の符号）・
  行ごとのフィルタ（5 種類）の組み合わせ 17 個を、Python の zlib で作った PNG で、ネイティブと wasm32 で確かめる
  （`tests/image/make_png.py` が作る）。壊れたファイルと PNG でないファイルは `error` になる。
- ほかのソフトが書いた PNG（512×512 の RGBA、3000×100 の RGB、16 ビットの RGB など）を読み、Chromium が読んだ画素と
  一致した（不透明度のある画素は、Chromium の canvas が不透明度を乗じて持つため、不透明な画素で比べた）。
- 速さ: ネイティブで、512×512 の RGBA が約 18 ミリ秒、3000×200 の RGBA が約 50 ミリ秒（プロセスの起動を含む）。
