# 描画（GPU で三角形を塗る）

GPU の三角形のラスタライザ（WebGPU の render pipeline）を UM から使います。シェーダは UM で書き、kernel と同じく WGSL に変換して GPU で実行します。

**いまできること（段階 1）:** 画面なしの画像に描いて、画素を `Buffer<u32>` に読み戻す。ネイティブの wgpu の経路（`UM_WGPU=1`。[wgpu での GPU 実行](wgpu.ja.md)）だけです。

**これから:** ウィンドウに描く、ブラウザ（wasm32 と WebGPU）で描く、インデックスとインスタンス、テクスチャ、裏面を描かない設定・半透明の合成など。

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
  const img = gpu.beginImage(320, 240, 0xFF000000);
  gpu.draw(img, 3, vs(verts, 1.0 as f32), fs(1.0 as f32));
  let px = new Buffer<u32>(320 * 240);
  gpu.endImage(img, mut px);   // px は 0xAARRGGBB の画素。上の行から
  return 0;
}
```

```sh
UM_WGPU=1 node src/cli.ts build tri.um --target host -o tri && ./tri
```

## シェーダの関数

| 種類 | 第1引数 | 残りの引数 | 戻り値 |
| --- | --- | --- | --- |
| `vertex function` | 頂点の番号（`u32`。0 から頂点の数未満） | Buffer（読むだけ）と値 | `position: vec4<f32>` を持つ struct |
| `fragment function` | vertex function の戻り値と同じ struct（三角形の中で補間された値） | Buffer（読むだけ）と値 | 色の `vec4<f32>`（赤・緑・青・不透明度。0〜1） |

- **頂点は vertex function が自分で読みます。** 頂点の形式を宣言する代わりに、`verts[i]` のように Buffer から読みます。kernel が書いた Buffer（パーティクルなど）も、CPU に戻さずにそのまま描けます。
- **position はクリップ空間の位置です。** `x / w` と `y / w` が -1〜1 の範囲が画像に入り（y は上が正）、`z / w` が 0〜1 の範囲が深度です（0 が近い）。`mat4Perspective` と `mat4LookAt`（[言語ガイド](guide.ja.md)の「三角関数」）の結果をそのまま使えます。
- **受け渡す struct のフィールド**は `f32` / `i32` / `u32` か、それらの `vec2`〜`vec4` です。浮動小数点は三角形の中で補間し、整数は補間しません（最初の頂点の値）。fragment function の `position` は、画素の座標（左上が (0, 0)、画素の中心は 0.5）と深度です。
- 本体の決まりは kernel と同じです（GPU で使える型だけ、`kernel function` を呼べる）。Buffer の要素と値は 32 ビットの型（`i32` / `u32` / `f32` と、その struct・固定長配列）だけで、Buffer には書けません。
- シェーダの関数は `gpu.draw` の中でだけ使えます（普通の関数として呼ぶことも、`dispatch` することもできません）。共通の計算は `kernel function` にして、両方から呼びます。

## 組み込みの関数

| 関数 | 内容 |
| --- | --- |
| `gpu.beginImage(width: i32, height: i32, clear: u32): i32` | 画像（色と深度）を GPU に用意し、色を `clear`（0xAARRGGBB）で、深度を 1 で埋める。番号を返す |
| `gpu.draw(image: i32, vertexCount: u32, vs(引数...), fs(引数...))` | vertex function を頂点の番号 0 から `vertexCount` 未満について実行し、3 つずつを三角形として塗る。`vs(...)` と `fs(...)` は呼び出しではなく、引数の指定（vs は頂点の番号、fs は受け取る値を除いた引数） |
| `gpu.endImage(image: i32, mut pixels: Buffer<u32>)` | 色を `pixels` に写し（0xAARRGGBB、上の行から）、画像を片づける。`pixels` は `width × height` 以上の長さが要る |

- **深度テスト**があります。深度の小さい（近い）ものが残り、同じか遠いものは塗りません。同じ画像への `gpu.draw` は、前の結果に重ねて描きます。
- いまは、裏面も描きます（カリングなし）。色はそのまま書きます（半透明の合成なし）。
- `gpu.draw` は dispatch と同じ順に GPU に積まれます。前に dispatch した kernel が書いた Buffer を、続けて描けます。
- `UM_WGPU=1` でないビルドや wasm32 では、コンパイルエラーです。GPU が使えない環境では、`gpu.beginImage` で停止します（GPU のない Linux では、lavapipe を入れると試せます。[wgpu での GPU 実行](wgpu.ja.md)）。

## 仕組み

vertex function と fragment function の組を、1 つの WGSL のモジュール（`@vertex fn um_vs` と `@fragment fn um_fs`）にします（`src/wgsl.ts` の `render`）。束縛の並びは kernel と同じで、binding 0 が `UmParams`（頂点の数、0、Buffer ごとの長さ、値の引数）、binding 1 からが Buffer です。引数は vertex の引数、fragment の引数の順に並びます。画像の色は `BGRA8Unorm`（読み戻すと 0xAARRGGBB の `u32` と同じバイトの並び）、深度は `Depth32Float` のテクスチャです（`runtime/wgpu.c` の `um_gpu_image_begin` など）。

浮動小数点の計算は kernel と同じ WGSL（割り算と平方根を正しく丸め、掛け算と足し算を組み替えさせない）なので、シェーダの中の計算は CPU と同じ値になります。ただし、どの画素を塗るか（ラスタライズ）と、補間の精度は GPU が決めます。
