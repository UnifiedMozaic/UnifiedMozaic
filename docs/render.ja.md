# 描画（GPU で三角形を塗る）

三角形を塗る描画です。シェーダは UM で書きます。GPU が使えるときは GPU の三角形のラスタライザ（WebGPU の render pipeline）で、使えないときは CPU のラスタライザで、同じプログラムが同じ絵を描きます（kernel の `dispatch` と同じ考え方）。

**いまできること:** 画面なしの画像に描いて画素を `Buffer<u32>` に読み戻す（`gpu.beginImage`）。ウィンドウに描く（`gpu.beginFrame`。ネイティブとブラウザ）。

| ビルド | 描く場所 |
| --- | --- |
| ネイティブ（`UM_WGPU=1`。[wgpu での GPU 実行](wgpu.ja.md)） | GPU。GPU が使えなければ CPU |
| ブラウザ（wasm32。[ブラウザで動かす](web.ja.md)） | WebGPU。WebGPU がなければ CPU |
| ネイティブ（`UM_WGPU=1` なし）、Metal（`UM_METAL=1`）、Node で動かす wasm32 | CPU |

**これから:** 裏面を描かない設定・半透明の合成、画像ファイルの読み込み、日本語の文字、GPU のテクスチャ（ハードウェアの補間とミップマップ）など。

## 例

```ts
struct Vertex {
  x: f32;
  y: f32;
  r: f32;
  g: f32;
  b: f32;
}

// vertex function の戻り値（fragment function に渡す値）。position は必ず要る。
struct VOut {
  position: vec4<f32>;
  color: vec3<f32>;
}

vertex function vs(i: u32, verts: Buffer<Vertex>, scale: f32): VOut {
  const v = verts[i];
  return { position: vec4(v.x * scale, v.y * scale, 0.5, 1), color: vec3(v.r, v.g, v.b) };
}

fragment function fs(v: VOut, bright: f32): vec4<f32> {
  return vec4(v.color.x * bright, v.color.y * bright, v.color.z * bright, 1);
}

export function main(): i32 {
  let verts = new Buffer<Vertex>(3);
  verts[0] = { x: -0.9, y: -0.9, r: 1, g: 0, b: 0 };
  verts[1] = { x: 0.9, y: -0.9, r: 0, g: 1, b: 0 };
  verts[2] = { x: 0, y: 0.9, r: 0, g: 0, b: 1 };
  let img = gpu.beginImage(320, 240, 0xFF000000);
  gpu.draw(mut img, 3, vs(verts, 1.0 as f32), fs(1.0 as f32));
  let px = new Buffer<u32>(320 * 240);
  gpu.endImage(mut img, mut px);   // px は 0xAARRGGBB の画素。上の行から
  return 0;
}
```

```sh
UM_WGPU=1 node src/cli.ts build tri.um --target host -o tri && ./tri   # GPU で描く
node src/cli.ts build tri.um --target host -o tri && ./tri             # CPU で描く
```

## シェーダの関数

| 種類 | 第1引数 | 残りの引数 | 戻り値 |
| --- | --- | --- | --- |
| `vertex function` | 頂点の番号（`u32`。0 から頂点の数未満） | Buffer（読むだけ）と値 | `position: vec4<f32>` を持つ struct |
| `fragment function` | vertex function の戻り値と同じ struct（三角形の中で補間された値） | Buffer（読むだけ）と値 | 色の `vec4<f32>`（赤・緑・青・不透明度。0〜1） |

- **頂点は vertex function が自分で読みます。** 頂点の形式を宣言する代わりに、`verts[i]` のように Buffer から読みます。kernel が書いた Buffer（パーティクルなど）も、CPU に戻さずにそのまま描けます。
- **position はクリップ空間の位置です。** `x / w` と `y / w` が -1〜1 の範囲が画像に入り（y は上が正）、`z / w` が 0〜1 の範囲が深度です（0 が近い）。`mat4Perspective` と `mat4LookAt`（ライブラリ `um:transform`。[ライブラリ](libraries.ja.md)）の結果をそのまま使えます。
- **受け渡す struct のフィールド**は `f32` / `i32` / `u32` か、それらの `vec2`〜`vec4` です。浮動小数点は三角形の中で補間し、整数は補間しません（最初の頂点の値）。fragment function の `position` は、画素の座標（左上が (0, 0)、画素の中心は 0.5）と深度です。
- 本体の決まりは kernel と同じです（GPU で使える型だけ、`kernel function` を呼べる）。Buffer の要素と値は 32 ビットの型（`i32` / `u32` / `f32` と、その struct・固定長配列）だけで、Buffer には書けません。
- シェーダの関数は `gpu.draw` の中でだけ使えます（普通の関数として呼ぶことも、`dispatch` することもできません）。共通の計算は `kernel function` にして、両方から呼びます。
- シェーダの関数は CPU でも動きます（GPU のないときの描画。下の「CPU のラスタライザ」）。

## 組み込みの関数

| 関数 | 内容 |
| --- | --- |
| `gpu.beginImage(width: i32, height: i32, clear: u32): GpuImage` | 画像（色と深度）を用意し、色を `clear`（0xAARRGGBB）で、深度を 1 で埋める。GPU が使えれば GPU に、使えなければ CPU に用意する |
| `gpu.draw(mut image: GpuImage, vertexCount: u32, vs(引数...), fs(引数...))` | vertex function を頂点の番号 0 から `vertexCount` 未満について実行し、3 つずつを三角形として塗る。`vs(...)` と `fs(...)` は呼び出しではなく、引数の指定（vs は頂点の番号、fs は受け取る値を除いた引数） |
| `gpu.endImage(mut image: GpuImage, mut pixels: Buffer<u32>)` | 色を `pixels` に写す（0xAARRGGBB、上の行から）。`pixels` は `width × height` 以上の長さが要る。GPU の画像はここで片づける |

- `GpuImage` は標準の class です（`runtime/render.um`）。`handle` が 0 でなければ GPU の画像、0 なら CPU の画像で、色と深度を `color` / `depth` の Buffer に持ちます。`new GpuImage(width, height, 0, clear)` と書くと、GPU があっても CPU で描く画像を作れます（GPU と CPU の結果を比べるときなど）。

- **番号で描く**: `gpu.drawIndexed(mut image, indices: Buffer<u32>, vs(...), fs(...))` は、`indices` の値を頂点の番号として vertex function に渡し、3 つずつを三角形にする。四角なら 4 つの頂点と 6 つの番号で描ける。
- **インスタンス**: `gpu.draw` と `gpu.drawIndexed` に、図形の前にインスタンスの数を足すと（`gpu.draw(mut image, vertexCount, instances, vs(...), fs(...))`）、同じ形をその数だけ描く。vertex function は 2 番目の引数でインスタンスの番号を受け取る（`vertex function vs(i: u32, inst: u32, ...)`）ので、位置や色はインスタンスごとの Buffer から `places[inst]` のように読む。
- **深度テスト**があります。深度の小さい（近い）ものが残り、同じか遠いものは塗りません。同じ画像への `gpu.draw` は、前の結果に重ねて描きます。
- 描き方（裏面を描くか・色の混ぜ方・深度を書くか）は、画像のフィールドで決めます（下の「描き方」）。既定は、裏表とも描き、色を上書きし、深度を書きます。
- `gpu.draw` は dispatch と同じ順に GPU に積まれます。前に dispatch した kernel が書いた Buffer を、続けて描けます。
- GPU の画像の大きさが GPU の上限を超えると停止します。

## 描き方（カリング・半透明）

`gpu.draw` の前に、画像のフィールドを変えます。`gpu.draw` のたびに、そのときの値で描きます（変えた値は、戻すまで続きます）。

| フィールド | 値 | 内容 |
| --- | --- | --- |
| `cull: CullMode` | `None`（既定）・`Back`・`Front` | 面の向きで捨てる。表は、**画面で反時計回り**に頂点が並ぶ三角形（WebGPU の既定と同じ）。閉じた立体は `Back` にすると、見えない裏の面を塗らないぶん速くなる |
| `blend: BlendMode` | `Opaque`（既定）・`Alpha`・`Add` | 色の混ぜ方。`Alpha` は 色 = src × a + dst × (1 − a)、`Add` は 色 = src × a + dst（光・炎など）。不透明度はどちらも src.a + dst.a × (1 − src.a)。src は fragment function の色を 0〜1 に収めたもの |
| `depthWrite: bool` | `true`（既定）・`false` | 深度を書くか。`false` でも深度は比べる（奥にあるものは塗らない） |

```ts
// 不透明なものを先に描き、半透明のものを奥から順に、深度を書かずに重ねる。
let img = gpu.beginFrame(windowSurface(window), 0xFF000000);
img.cull = CullMode.Back;
gpu.drawIndexed(mut img, solidIndices, vs(scene), fs(light));
img.blend = BlendMode.Alpha;
img.depthWrite = false;
gpu.drawIndexed(mut img, glassIndices, vs(scene), glassFs(light));
gpu.endFrame(mut img);
```

- 半透明のものどうしは、描いた順に重なります（奥から順に描くのはプログラムの役目）。
- GPU と CPU のラスタライザは同じ式で混ぜます（`tests/wgpu/draw_state.um`）。ただし GPU によっては混ぜる計算の丸めが違い、色が 1 ずれることがあります。

## テクスチャ

画像の色を読む `kernel function` です。ライブラリ `um:texture` にあります（`import { textureSample } from "um:texture";`。[ライブラリ](libraries.ja.md)）。画像は `Buffer<u32>` の画素（0xAARRGGBB、上の行から）と幅・高さで渡します。UM で書いた関数なので、シェーダでも kernel でも CPU の描画でも使え、どこでも同じ値になります。

| 関数 | 内容 |
| --- | --- |
| `textureSample(pixels, width, height, uv: vec2<f32>): vec4<f32>` | 座標 `uv` の色。隣り合う 4 つの画素を近さで混ぜる（バイリニア）。`uv` は左上が (0, 0)、右下が (1, 1)。範囲の外は画像を繰り返す |
| `textureFetch(pixels, width, height, x: i32, y: i32): vec4<f32>` | 画素 (x, y) の色（混ぜない）。範囲の外は繰り返す |
| `colorFromArgb(c: u32): vec4<f32>` | 0xAARRGGBB を 0〜1 の色（x = 赤、y = 緑、z = 青、w = 不透明度）にする |

```ts
import { textureSample } from "um:texture";
fragment function fs(v: VOut, bricks: Buffer<u32>): vec4<f32> {
  return textureSample(bricks, 32, 32, v.uv);
}
```

画像ファイル（PNG）は `imageLoad(path)` で読めます（[画像ファイルの読み込み](image.ja.md)）。読んだ `Bitmap` の `pixels` / `width` / `height` をそのまま渡します。

GPU の専用のテクスチャ（ハードウェアの補間、ミップマップ）ではなく、Buffer を読んで計算します。そのぶん遅くなりますが、CPU と GPU で同じ絵になります。例は [examples/window/demo.um](../examples/window/demo.um)（壁と柱のレンガ）と [tests/wgpu/draw_instanced.um](../tests/wgpu/draw_instanced.um)。

## 文字

画像やフレームに ASCII の文字を描く関数です。ライブラリ `um:text` にあります（`import { drawText, textWidth } from "um:text";`。[ライブラリ](libraries.ja.md)）。

| 関数 | 内容 |
| --- | --- |
| `drawText(mut frame: GpuImage, text: string, x: f32, y: f32, scale: f32, color: u32)` | (x, y)（左上。画素）から `text` を描く。1 文字は 5 × 7 の点で、点 1 つが `scale` 画素の正方形。`color` は 0xAARRGGBB |
| `textWidth(text: string, scale: f32): f32` | 描いたときの幅（画素）。1 文字は `6 × scale`（5 つの点と 1 つの隙間）。高さは `7 × scale` |

```ts
import { drawText, textWidth } from "um:text";
drawText(mut frame, `GEMS ${score} / 8`, 10, 10, 3, 0xFFFFFFFF);
const w = textWidth("CLEAR!", 8);
drawText(mut frame, "CLEAR!", (frame.width as f32 - w) / 2, 200, 8, 0xFFFFD54A);
```

- フォントは UnifiedMozaic のために描いた 5 × 7 の点のフォントで、ASCII の印字できる文字（32〜126）があります。それ以外の文字は `?` になります（日本語はまだ描けません）。
- 点の 1 つずつを小さな四角として `gpu.draw` で描くので、GPU でも CPU でも、ネイティブでもブラウザでも同じ絵になります。
- 深度は 0（いちばん手前）なので、3D の場面の上に重なります。場面を描いたあとに呼びます。
- 影を付けたいときは、少しずらして暗い色で先に描きます（[examples/window/demo.um](../examples/window/demo.um) の `shadowText`）。

## ウィンドウに描く

```ts
const win = windowOpen("game", 1280, 720) ?? 0;
while (running) {
  // ...入力（pollWindowEvent）...
  let frame = gpu.beginFrame(windowSurface(win), 0xFF202028);
  gpu.draw(mut frame, 36, vs(verts, camera), fs(light));
  gpu.endFrame(mut frame);   // 表示する（画面の更新に合わせて待つ）
}
```

| 関数 | 内容 |
| --- | --- |
| `gpu.beginFrame(surface: GpuSurface, clear: u32): GpuImage` | 表示の面に描くフレームを始める。面は表示する側のライブラリが作る（ウィンドウなら `um:window` の `windowSurface(window)`）。大きさは面の画素の大きさ（ウィンドウなら `windowPixelSize`）で、`frame.width` / `frame.height` で分かる。色を `clear` で、深度を 1 で埋める |
| `gpu.endFrame(mut frame: GpuImage)` | 描いたフレームをウィンドウに表示する |

- **GPU が使えれば**（`UM_WGPU=1` のビルドで GPU があれば）、ウィンドウの GPU の面（surface）に直接描きます。初めてのフレームで、ウィンドウは SDL の描画の装置を手放すので、**以後そのウィンドウに `windowPresent` は使えません**（使うと停止）。表示は画面の更新（vsync）に合わせて待ちます。
- **GPU が使えなければ**、CPU の画像に描き、`gpu.endFrame` が `windowPresent` で表示します。
- ウィンドウの大きさが変わっても、次の `gpu.beginFrame` から新しい大きさになります。縦横比は `frame.width / frame.height` で求めます（[examples/window/cube.um](../examples/window/cube.um)）。
- 最小化の途中などで GPU の面を得られないときは、そのフレームは描かずに捨てます。
- ウィンドウの関数と同じく、メインスレッドから呼びます。

**確かめた環境:** Linux（Xvfb の X11 と lavapipe の Vulkan）で、GPU と CPU のどちらでも回る立方体が描けることを画面の写しで確かめました。Windows と macOS のウィンドウ（HWND と CAMetalLayer の面）は、コードはありますが未確認です。

## CPU のラスタライザ

GPU のない環境でも同じプログラムが動くよう、コンパイラは `gpu.draw` の vertex function と fragment function の組ごとに、描画の関数（`umRenderDraw__vs__fs`。`src/render.ts` の `drawSource`）を UM で作ります。その関数は、画像が GPU にあれば GPU に頼み、CPU にあれば次のように描きます。

1. vertex function を CPU で呼んで、3 つずつの頂点を作る（番号で描くときは番号の Buffer の値、インスタンスのあるときはインスタンスごとに繰り返す）
2. クリップ空間で、近い面（z ≥ 0）と遠い面（z ≤ w）で切る（GPU と同じ）
3. 画面の座標にし、画素の中心が三角形の中にある画素を塗る。辺の上の画素は top-left rule（GPU と同じ決まり）
4. 深度を比べ、近ければ、遠近を考えた補間で受け渡す値を作り、fragment function を呼んで色を書く

**GPU と同じ絵になります。** 透視投影の立方体と近い面で切られる三角形の場面（`tests/wgpu/draw_compare.um`）で、塗られた画素も補間した色も GPU（lavapipe）と一致しました。ただし GPU は頂点の位置を画素の細かい格子（ふつう 1/256 画素）に丸めるので、辺のすぐ近くの画素は、GPU によって分かれることがあります。

CPU の描画は画素ごとに fragment function を呼ぶので、GPU よりずっと遅くなります（試作・テスト・GPU のない環境のためのもの）。

## 速いシェーダの計算（UM_SHADER_FAST_MATH=1）

ふつう、シェーダ（vertex function と fragment function、そこから呼ぶ kernel function）の浮動小数点の計算は、kernel と同じく CPU と同じ値になるように WGSL にします（割り算と平方根を整数の演算で正しく丸め、掛け算と足し算を FMA にまとめさせず、組み替えさせない）。そのぶん、割り算や平方根の多いシェーダは重くなります。

ビルドのときに `UM_SHADER_FAST_MATH=1` を付けると、**描画のシェーダだけ** GPU のふつうの計算を使います。

```sh
UM_SHADER_FAST_MATH=1 UM_WGPU=1 node src/cli.ts build game.um --target host -o game
UM_SHADER_FAST_MATH=1 node src/cli.ts build game.um --web -o game-web
```

- 割り算と平方根は GPU の命令のまま（WebGPU は数 ULP の誤差を許す）。掛け算と足し算は FMA にまとめたり、順を組み替えたりしてよい。
- 速くなるが、GPU の絵と CPU の絵が少し変わる（色が 1 / 255 ほど違う画素が出る）。どの画素を塗るかは、もともと GPU が決めるので、描画の結果はこの指定がなくてもビットまで同じにはならない。
- **kernel（`dispatch`）は変わらない。** 正確な計算のまま、CPU と GPU で同じ値になる。
- CPU で描くとき（GPU がないとき）の計算は変わらない。

## 仕組み（GPU）

vertex function と fragment function の組を、1 つの WGSL のモジュール（`@vertex fn um_vs` と `@fragment fn um_fs`）にします（`src/wgsl.ts` の `render`）。束縛の並びは kernel と同じで、binding 0 が `UmParams`（頂点の数、0、Buffer ごとの長さ、値の引数）、binding 1 からが Buffer です。引数は vertex の引数、fragment の引数の順に並びます。画像の色は `BGRA8Unorm`（読み戻すと 0xAARRGGBB の `u32` と同じバイトの並び）、深度は `Depth32Float` のテクスチャです（`runtime/wgpu.c` の `um_gpu_image_begin` など）。

浮動小数点の計算は kernel と同じ WGSL（割り算と平方根を正しく丸め、掛け算と足し算を組み替えさせない）なので、シェーダの中の計算は CPU と同じ値になります。ただし、どの画素を塗るか（ラスタライズ）と、補間の精度は GPU が決めます。
