<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo/logo-white.svg">
    <img src="assets/logo/logo-black.svg" alt="UnifiedMozaic のロゴ" width="120">
  </picture>
</p>

# UnifiedMozaic

**One language. Every core.**

[English](README.md)

TypeScript 風の文法で書く、静的型の小さな言語です。LLVM を通して、GC なしのネイティブバイナリと WebAssembly にコンパイルします。CPU と GPU のコードを同じソース（`.um`）に書けます。

> **実験的なプロジェクトです。** 1 人で開発していて、言語もランタイムもよく変わります。

```ts
struct Particle {
  pos: f32;
  vel: f32;
}

// kernel は添字ごとに 1 回ずつ、CPU（全コア）か GPU で実行される。
kernel function integrate(i: u32, ps: mut Buffer<Particle>, dt: f32): void {
  ps[i].pos = ps[i].pos + ps[i].vel * dt;
}

export function main(): i32 {
  let ps = new Buffer<Particle>(1000);        // 長さ固定、ゼロ埋め
  ps[0].vel = 2;
  dispatch(integrate, ps.length, mut ps, 0.5);
  print(ps[0].pos);                           // 1
  return 0;
}
```

## 特徴

- **TypeScript に近い文法と、明示的な型。** 幅つきの数値型（`i8`〜`i64`、`u8`〜`u64`、`f16` / `f32` / `f64`、小さい小数 `f4` / `f8`）、`struct`、`class`（継承なし）、`enum` と `match`、interface、ジェネリクス、文字列、null になりうる型。暗黙の型変換はありません。
- **GC の代わりに所有権。** 配列などヒープを持つ値の所有者は常に 1 つです。引数は読み取り借用（既定）・可変借用（`mut`）・移動（`own`）で、コンパイル時に検査します。
- **既定で安全。** 範囲外アクセス・整数のゼロ除算・ビット幅以上のシフトは、理由を示して停止します。整数のオーバーフローはラップします。未定義動作も `unsafe` もありません。
- **CPU と GPU を 1 つのソースで。** `kernel` 関数はどの実行先でも同じ意味です。CPU のスレッドプール、wgpu（WGSL。Windows・Linux・macOS の GPU）、Metal（MSL）による Apple の GPU、ブラウザの WebGPU で動きます。`dispatch` は CPU と GPU を実測で選び、`gpu.submit` / `gpu.ready` / `gpu.wait` で GPU の仕事を待たずに進められます。
- **スレッド。** `thread.spawn` / `thread.join`。複数のスレッドで書き換えられるのは `RacyBuffer` だけ（要素の読み書きはアトミック）なので、データ競合は起きません。
- **速さが要る所のための機能。** 128 ビットの SIMD の型（`i8x16`、`i16x8`、`f32x4` など）、`Math.prefetch`、マスクつきシフト、範囲チェックの除去、一括の書き込み `fill`。
- **人にも AI にも読みやすい診断。** エラーには原因コード・位置・直し方のヒントが付き、`check --json` で機械可読な形でも出せます。
- **npm の依存なし。** コンパイラは Node.js がそのまま実行する TypeScript で、外部のツールは `clang` だけです。

## 必要なもの

- Node.js 22.18 以上（TypeScript のソースをそのまま実行します）
- LLVM 18 以上の `clang`。wasm32 を出すには `wasm-ld` も必要です
  - macOS の Xcode の clang は wasm32 を出せません。`brew install llvm lld` で入れておけば、wasm32 のビルドには自動でそちらを使います
  - Windows では LLVM（`winget install LLVM.LLVM`）と、Visual Studio か Build Tools の MSVC（「C++ によるデスクトップ開発」）を入れます
- どの OS でも GPU で実行するには: wgpu-native（[wgpu での GPU 実行](docs/wgpu.ja.md)）
- Mac の GPU を Metal で使うには: Apple Silicon と、Xcode か Command Line Tools（Metal）

入れ方の手順（`um` コマンドを使えるようにする `npm link` や、うまくいかないときの対処を含む）は [docs/setup.ja.md](docs/setup.ja.md) にあります。

## 使い方

```sh
git clone https://github.com/UnifiedMozaic/UnifiedMozaic.git
cd UnifiedMozaic

# このマシン向けのネイティブバイナリ
node src/cli.ts build examples/structs.um --target host -o structs
./structs

# kernel を GPU でも実行するネイティブバイナリ（Windows・Linux・macOS。UM_WGPU_HOME に wgpu-native）
UM_WGPU=1 node src/cli.ts build examples/kernels.um --target host -o kernels

# kernel を Metal で GPU 実行するネイティブバイナリ（Apple Silicon）
UM_METAL=1 node src/cli.ts build examples/kernels.um --target host -o kernels
./kernels

# WebAssembly（既定のターゲット）
node src/cli.ts build examples/structs.um -o structs.wasm
node runtime/run-node.mjs structs.wasm

# 型検査だけ（--json で機械可読な診断）
node src/cli.ts check examples/structs.um

# 生成されるコードを見る
node src/cli.ts build examples/kernels.um --emit-llvm -o kernels.ll
node src/cli.ts build examples/kernels.um --target host --emit-msl -o kernels   # kernels.<名前>.metal
```

## 説明書

- [環境構築](docs/setup.ja.md) — Node.js と clang の入れ方、`um` コマンド、テストの実行
- [言語ガイド](docs/guide.ja.md) — 文法、所有権、kernel と Buffer、GPU での実行、スレッド、安全性の保証
- [wgpu での GPU 実行](docs/wgpu.ja.md) — `UM_WGPU=1`、Buffer の写し、実行時に kernel を読み込む `gpu.loadKernel`、速さ
- [ウィンドウ](docs/window.ja.md) — ウィンドウを開いて画素を表示し、マウスとキーボードの入力を受け取る（ネイティブ、SDL3）
- [WebAssembly の実行](docs/wasm.ja.md) — wasm のモジュールを隔離して読み込み、呼び出す。UM で書いたプラグイン（ネイティブ、wasmtime）
- [動画・音声](docs/media.ja.md) — 動画・音声のファイルの情報、コマ単位で正確なシーク、展開（ハードウェアの展開も）、H.264 / AAC などでの書き出し（ネイティブ、FFmpeg）
- [examples/](examples) — 動くサンプル。どれもテストを兼ねます（`// expect:` の行）
- [vscode-extension](https://github.com/UnifiedMozaic/vscode-extension) — VSCode 拡張（色分け、保存時の型検査、ビルドと実行）
- [SECURITY.md](SECURITY.md) — セキュリティ方針と、脆弱性の報告のしかた

## テスト

```sh
node tests/run.ts
```

各サンプルを wasm32 とネイティブで実行して出力を照合し、コンパイルエラーになるべきコード（`tests/errors/`）と停止するべきコード（`tests/panics/`）を確かめます。ThreadSanitizer・AddressSanitizer・valgrind があれば、それらをつけても実行します。GPU のテスト（`tests/metal/`・`tests/wgpu/`）は、Metal のある Mac と、wgpu-native が見つかる（`UM_WGPU_HOME`）環境で行います。wasm32 を出せる `clang` がなければ、wasm32 のテストは飛ばします。

## 構成

```
src/        コンパイラ: lexer → parser → checker → codegen（LLVM IR）→ driver（clang）
            と、kernel の変換 msl.ts（Metal）・wgsl.ts（wgpu と WebGPU）、実行時に読み込む kernel gpuload.ts
runtime/    ネイティブ（C）・wgpu（C）・Metal（Objective-C）・wasm のランタイムと JS のホスト
examples/   サンプル（テストを兼ねる）
tests/      テストランナーとテストケース
assets/     ロゴ
docs/       説明書
```

## ライセンス

次のどちらかを選んで使えます。

- Apache License, Version 2.0（[LICENSE-APACHE](LICENSE-APACHE)）
- MIT license（[LICENSE-MIT](LICENSE-MIT)）

明示しない限り、このプロジェクトに取り込まれることを意図して送られた貢献は、Apache-2.0 の定義に従い、追加の条件なしに上の 2 つのライセンスで提供されるものとします。
