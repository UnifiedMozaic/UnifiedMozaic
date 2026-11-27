# wgpu での GPU 実行（Windows・Linux・macOS）

`UM_WGPU=1` を付けてネイティブにビルドすると、kernel を **wgpu-native**（WebGPU の C API。github.com/gfx-rs/wgpu-native）で GPU でも実行するバイナリになります。kernel は WGSL（WebGPU のシェーダ言語）に変換してバイナリに入れ、実行時に wgpu が各 OS の API で動かします。

| OS | wgpu が使う API |
| --- | --- |
| Windows | Vulkan（既定）か Direct3D 12 |
| Linux | Vulkan |
| macOS | Metal |

この経路は、Metal の経路（`UM_METAL=1`、Mac だけ。[言語ガイド](guide.ja.md)の「Metal（Mac）での GPU 実行」）と並ぶ、ネイティブの主な GPU の経路です。kernel の意味（範囲外の添字の切り詰め、ゼロ除算の決まった結果、浮動小数点の丸めなど）は CPU・Metal と同じで、同じ kernel は CPU と GPU で同じ結果になります（下の「CPU と GPU で同じ結果にするために」）。

**確かめた環境:** Windows 11（NVIDIA GeForce RTX 5060 Ti、Vulkan と Direct3D 12）と、Linux（Ubuntu 24.04、ソフトウェアの Vulkan の lavapipe（Mesa 25.2、llvmpipe）。GPU のない環境で試すとき）、wgpu-native v29.0.1.1。`tests/metal` と `tests/wgpu` の全部が CPU と同じ結果になります。**macOS では未確認です**（コードはありますが、実行していません）。

GPU のない Linux では、`apt-get install mesa-vulkan-drivers` で lavapipe を入れると、wgpu がそれを使います（遅いですが、結果を確かめるには足ります）。

## 準備: wgpu-native を入れる

wgpu-native のリリース（https://github.com/gfx-rs/wgpu-native/releases ）から、OS に合う `wgpu-<OS>-<CPU>-release.zip` を取って展開し、その場所を環境変数 `UM_WGPU_HOME` に入れます（`include/webgpu/webgpu.h` と `lib/` がある場所）。v29.0.1.1 で確かめています。

```powershell
# Windows（PowerShell）。zip を展開した場所を指す
$env:UM_WGPU_HOME = "C:\Users\you\wgpu-native\v29.0.1.1"
```

```sh
# Linux / macOS
export UM_WGPU_HOME=$HOME/wgpu-native/v29.0.1.1
```

`UM_WGPU_HOME` がなければ、Linux と macOS では pkg-config の `wgpu-native` を、macOS ではさらに Homebrew（`brew install wgpu-native`）の場所を探します。

## ビルド

```sh
UM_WGPU=1 node src/cli.ts build x.um --target host -o x
./x
```

- **`UM_WGPU=1` はコンパイル時の指定です。** 付けないと、ネイティブのビルドは GPU を使いません（Mac で `UM_METAL=1` のときを除く）。`UM_METAL=1` と同時には指定できません。
- **GPU を使う呼び出しがないプログラムには、wgpu をリンクしません。** GPU で実行しうる dispatch（`cpu.dispatch` 以外）と `gpu.loadKernel` などがないプログラムは、`UM_WGPU=1` でも CPU だけのバイナリになります。
- **Windows では `wgpu_native.dll` を実行ファイルの隣に写します**（DLL は実行ファイルと同じ場所にないと起動できないため）。配布するときは一緒に置いてください。Linux と macOS では、ライブラリの場所を実行ファイルに書き込みます（rpath）。
- **kernel の WGSL は `--emit-wgsl` で見られます**（`<出力>.<kernel>.wgsl`。どのターゲットでも書き出せます）。バイナリに入るものと同じです。
- **GPU の準備（数十ミリ秒）は、初めて GPU を使うときまでしません。**

## 意味は Metal と同じ

| 書き方 | wgpu の経路での動き |
| --- | --- |
| `dispatch(k, n, ...)` | 実測と Buffer の置き場所から、CPU と GPU の速いほうを選ぶ（下の「自動判断」）。`gpu.prefer(true)` のあいだは GPU を優先する |
| `cpu.dispatch(k, n, ...)` | 常に CPU（複数のスレッド） |
| `gpu.dispatch(k, n, ...)` | 常に GPU。GPU が使えなければ停止（`panic: the GPU is not available`） |
| `gpu.submit(k, n, ...)` | 待たずに戻り、券を返す。`gpu.ready(t)` / `gpu.wait(t)` で完了を知る |
| `dispatchGroups` / `gpu.submitGroups` | 共有配列・barrier・kernel の中のアトミック演算が使える（simdgroup の集計は CPU で実行。下） |
| `gpu.flags(f)` | 印を覚えて前の値を返すだけで、効果はない（Metal だけの仕組み） |
| `gpu.commit()` / `gpu.untracked(b)` | 何もしない |

- **dispatch のあとは、すぐに結果を CPU で読めます**（同期の意味）。実際には、wgpu の経路の dispatch は GPU に提出するだけで完了を待たず、CPU が結果に触ったときに完了を待って読み戻します（下の「Buffer の写し」）。どちらでも、プログラムから見える結果は同じです。
- **`gpu.submit` のあと、終わる前に CPU が書く Buffer を読むと、終わるまで待ってから読みます**（Metal では古い値と新しい値が混ざりえます）。どちらも許された動きです。
- 実行時の失敗（シェーダの生成の誤りなど、ふつうは起きない）は停止します（`panic: the GPU could not run a kernel (...)`）。

## CPU で実行する kernel（判断の規則）

WGSL にない機能を使う kernel は、GPU では実行せず CPU で実行します。

| kernel が使うもの | `dispatch` / `dispatchGroups` / `gpu.submit` | `gpu.dispatch` / `gpu.dispatchGroups` |
| --- | --- | --- |
| 64 ビットの整数（`i64` / `u64`。値・Buffer の要素・呼ぶ補助の kernel の中を含む） | CPU で実行（`gpu.submit` は券 0） | コンパイルエラー `kernel-type` |
| simdgroup の集計（`group.simdSum` / `simdMin` / `simdMax` / `simdShuffle` / `simdLane`） | CPU で実行 | コンパイルエラー `webgpu-simdgroup` |

- 判断はコンパイル時に kernel ごとに行います（kernel の本体と、そこから呼ぶ補助の kernel を調べます）。同じプログラムのほかの kernel は GPU で実行できます。
- **最上位の定数の配列は使えます**（WGSL のモジュールの範囲の `const` にします。添字は固定長配列と同じく最後の要素に切り詰めます）。以前の `webgpu-const-array` のエラーはなくなりました（wasm32 の WebGPU でも同じです）。
- 64 ビットの整数と simdgroup の集計を GPU で実行できるのは Metal だけです。

例: [tests/wgpu/fallback.um](../tests/wgpu/fallback.um)。

## CPU と GPU で同じ結果にするために

WebGPU（Vulkan・Direct3D 12・Metal）は、浮動小数点の割り算と平方根に数 ULP の誤差を許し、掛け算と足し算を 1 つの命令（FMA）にまとめることも許します。ドライバによっては、足し算・引き算の順を組み替えることもあります。どちらも CPU と結果が変わる原因なので、生成する WGSL（`src/wgsl.ts`）で次のようにしています。

- **割り算と平方根は、整数の演算で正しく丸めます**（`um_fdiv` / `um_fsqrt`。IEEE 754 の最近接偶数への丸め。非正規化数・無限大・NaN も CPU と同じ）。GPU の命令より遅くなりますが、GPU の計算の速さに比べると小さな手間です。
- **掛け算の結果は、FMA にまとめられないようにします**（`um_fmul`。結果を、実行時にしか分からない 0 と OR する）。
- **足し算・引き算の結果も、組み替えられないようにします**（`um_fadd` / `um_fsub`。同じく 0 と OR する）。ドライバのコンパイラは `(a - b) + 1.0` を `(a + 1.0) - b` のように並べ替えることがあり（Linux の Mesa の NIR などで確認。`f32` の多項式で 1 ULP ずれた）、丸めが CPU と変わるためです。
- **整数の余り（`%`）は、割り算から求めます**（`um_rem_i` / `um_rem_u`。NVIDIA の Vulkan のドライバで、WGSL の `%` が負の数で誤った値を返すことがあったため）。
- **共有配列は、グループの始めに WGSL の中で 0 で埋めます**（WebGPU の仕様でも 0 ですが、NVIDIA の Vulkan で 0 にならないことがあったため）。
- 範囲外の添字の切り詰め、ゼロ除算（`x / 0 = x`、`x % 0 = 0`）、シフト量の剰余、浮動小数点から整数への飽和する変換は、wasm32 の WebGPU と同じ WGSL です。

**残る違い:** GPU によっては、非正規化数（絶対値が 2^-126 未満の `f32`）を足し算・掛け算の入力や結果で 0 にすることがあります（Vulkan・Direct3D の仕様で許されています）。非正規化数の値を CPU と GPU で揃える必要がある計算では、`cpu.dispatch` を使ってください。NaN のビットの並び（符号や payload）も揃うとは限りません（NaN であることは同じです）。

## Buffer の写し

NVIDIA や AMD の単体の GPU は、CPU とメモリが分かれているので、Buffer の中身を写す必要があります。wgpu の経路は、写しを必要なときだけにします。

- **Buffer を GPU に置いたままにします。** 一度 GPU に送った Buffer は、CPU が書き換えない限り送り直しません。
- **GPU が書いた Buffer は、CPU が触ったときに読み戻します。** GPU で続けて実行する dispatch のあいだでは、読み戻しは起きません。
- **dispatch は提出するだけで、完了を待ちません。** 結果を CPU が読むときに、読み戻しが完了を待ちます。小さな仕事の dispatch は、いくつか（16 個か、要素数の合計が 262144 まで）まとめて提出します。大きな仕事はすぐ提出して、GPU が CPU と並んで動けるようにします。

**仕組み:** Buffer の領域はページ単位で確保し、ページの保護で CPU の読み書きを見張ります（`runtime/wgpu.c`）。

| 状態 | ページの保護 | CPU が触ると |
| --- | --- | --- |
| CPU の写しだけが新しい | なし | （何も起きない）。次に GPU に渡すとき送る |
| CPU と GPU の写しが同じ | 読み取り専用 | 書くと、GPU の写しを古いとして保護を外す（例外 1 回） |
| GPU の写しだけが新しい | 読み書き禁止 | 読み戻して読み取り専用にする（例外 1 回） |

- 1 つの Buffer について、例外は GPU との往復 1 回あたり高々 2 回です（ページごとではなく、Buffer の全体をまとめて切り替えます）。
- 読み戻しは、同じメモリの別の場所（別名の写し）に書きます。プログラムから見える側は読み戻しのあいだも読み書き禁止のままなので、ほかのスレッドの CPU の書き込みが読み戻しに上書きされることはありません。
- `UM_WGPU_LAZY=0` で、この仕組みをやめ、dispatch のたびに送って読み戻します（比較用）。

**注意:**

- **デバッガで実行すると、アクセス違反の例外（first-chance exception）が見えます。** ランタイムが処理する例外なので、続行してかまいません（Visual Studio では、この例外で止まらないように設定できます）。
- **OS の関数に Buffer の領域を直接渡すと、読み戻しの前の中身が渡ることはありませんが、関数が失敗することがあります**（OS の中では保護の例外を処理できないため）。UM の組み込みでは、Buffer を受け取る `windowPresent` が先に領域を読んで読み戻しを済ませます。ほかの組み込みは Buffer を OS に直接渡しません。
- **Buffer は 1 つにつき 1 ページ以上を使います**（Windows では、アドレスの範囲を 64 KiB 単位で確保します）。小さい Buffer をとても多く作るプログラムでは、メモリの使い方が増えます。GPU に渡した Buffer は、GPU 側にも同じ大きさの写しを持ちます。

- **WebAssembly の組み込み（`wasmLoad` など。[docs/wasm.ja.md](wasm.ja.md)）と同じプログラムで使えます。** `wasmWrite` が GPU の書いた Buffer を読むと読み戻し、`wasmRead` が書いた Buffer は次に GPU に渡すとき送り直します。wasmtime も保護の例外を使いますが、どちらも自分の領域で起きた例外だけを扱うので、並んで動きます。

例: [tests/wgpu/transfers.um](../tests/wgpu/transfers.um)（CPU と GPU を行き来する並びと、複数のスレッドの読み書き）、[tests/wgpu/with_wasm.um](../tests/wgpu/with_wasm.um)（GPU の結果を wasm のプラグインに渡して戻す）。

## 自動判断（`dispatch`）

kernel ごとに CPU と GPU の時間を見込み、速いほうを選びます。見込みには、実測した時間と、**Buffer がいまどちらにあるか**（送る量・読み戻す量）を入れます。

```
GPU の時間 = 実行 + 送る量 × 送る速さ + 書く Buffer の量 × 読み戻す速さ × CPU がその結果を読む割合
CPU の時間 = 実行 + 読み戻す量 × 読み戻す速さ
```

- **送る量**は、渡す Buffer のうち GPU の写しが古いもの（CPU が書いたか、まだ GPU に送っていない）の大きさです。**読み戻す量**は、GPU だけが新しい Buffer の大きさです（CPU で実行すると、例外で読み戻すため）。どちらも dispatch のたびに Buffer の状態から数えます。入力も出力もすでに GPU にあれば、GPU の時間に写しは入らず、CPU の時間には読み戻しが入るので、GPU を選びます。
- **最初の 2 回は CPU で実行して測ります。** CPU での時間が 100 マイクロ秒未満の仕事は、GPU を試しません。
- **それより重い仕事は、GPU でも 2 回試します。** 試す回は、前の仕事を終わらせ、古い Buffer を送り終えるまでを測って送る速さとし、それから完了まで待って実行の時間を測り、書いた Buffer を読み戻して読み戻す速さを測ります。送る速さと読み戻す速さはプログラム全体で 1 つです。CPU の実行の時間からは、そのあいだの読み戻しの時間を除きます。
- **割合**は、kernel が GPU で書いた結果を、次に GPU が書く前に、kernel の外の CPU のコード（表示など）が読んだかどうかを覚えてならしたものです（はじめは 0.5）。CPU で実行した kernel が読んだときの読み戻しは、その kernel の CPU の時間の側に入るので、割合には数えません。
- 選ばれなかった側も、ときどき測り直します（Metal と同じ）。`UM_GPU_THRESHOLD=N` で、「要素数が N 以上なら GPU」という固定の規則になります。

**「全部 CPU」から抜け出す（周期の比較）:** Buffer の置き場所だけで判断すると、どの Buffer も CPU にあるときは、どの kernel も単独では送る分だけ遅く見えて、全部を GPU で続けて実行すれば速い場合でも CPU を選び続けます。そこで、同じ kernel が繰り返し実行されるプログラム（動画の 1 コマごとの描画など）では、次の 2 つの判断を交互に試し、繰り返しの周期が短いほうを使います。

| 判断 | 中身 |
| --- | --- |
| ふつう | 上の式のとおり |
| GPU に寄せる | 送る費用を見込まない（GPU で続けて使えば、送るのは初めの 1 回だけだから） |

- 周期は、目印の kernel（送る費用だけで CPU を選んだ最初の kernel）が実行される間隔です。切り替えた直後の数回を除き、6 回の周期を比べます（いちばん長いものを除いた平均）。
- 試すのは、送る費用だけで CPU を選んでいる kernel があるときと、GPU に寄せているあいだのときどきです（ふつうに戻して確かめる）。GPU に寄せるのは 5%、ふつうに戻すのは 10% 以上速いときです。結果が変わらなければ、次に試すまでの間隔を延ばします。
- `UM_GPU_TRACE=1` で比べた周期と選んだ判断を、`UM_GPU_TRACE=2` で dispatch ごとの選択と量を出します。

例: [tests/wgpu/placement.um](../tests/wgpu/placement.um)（判断が GPU と CPU を行き来しても、どのフレームの結果も CPU と同じ）、[bench/gpu/chain.um](../bench/gpu/chain.um)（下の「速さ」の「動画編集ソフトの描画に近い並び」）。

## GPU を優先する（`gpu.prefer`）・GPU があるか（`gpu.available`）

```ts
gpu.prefer(true);              // このあとの dispatch / dispatchGroups は、GPU が使えるかぎり GPU で実行する
dispatch(fx, n, mut canvas, k); // GPU がなければ（GPU の経路のないビルド、UM_GPU=0、GPU なし）CPU で実行する
gpu.prefer(false);             // 自動判断に戻す。どちらも前の値を返す
if (gpu.available()) { ... }   // 実行時に GPU が使えるか
```

- **どのビルドでもコンパイルできます**（`gpu.dispatch` は GPU の経路のないビルドではコンパイルエラーなので、共通のコードには書けません）。GPU の経路のないビルドでは、`gpu.prefer` は印を覚えて前の値を返すだけで（dispatch はいつも CPU）、`gpu.available()` は `false` です。
- `gpu.prefer(true)` のあいだも、GPU で実行できない kernel（64 ビットの整数・simdgroup の集計。上の「CPU で実行する kernel」）は CPU で実行します。
- 印はプログラム全体で 1 つです（スレッドごとではありません）。Metal の経路と wasm32 の WebGPU でも同じ意味です。
- 自動判断で足りる場合は `dispatch` のままで構いません。`gpu.prefer` は、GPU で続けて処理すると分かっている描画の並びを、初めのフレームから GPU で実行したいときに使います。

例: [tests/wgpu/prefer.um](../tests/wgpu/prefer.um)（GPU の経路のないビルドと wasm32 でも同じ出力になることも確かめる）。

## 実行時に kernel を読み込む（`gpu.loadKernel` / `gpu.runKernel`）

バイナリに入っていない kernel（WGSL の文字列）を実行時に読み込み、GPU で実行できます。MozaicCut のプラグイン（`.mzplug`）のように、効果を別に配る使い方のためのものです。

```ts
// WGSL のソースを読み込む。GPU が使えないか、WGSL をコンパイルできなければ null
const k = gpu.loadKernel(wgsl);            // u32 | null
if (k !== null) {
  // dispatch と同じ書き方で、kernel の引数を順に渡す（書く Buffer には mut）。実行を頼めたら true
  const ok = gpu.runKernel(k, w * h, src, mut dst, w, h, args);
}
```

| 組み込み | 意味 |
| --- | --- |
| `gpu.loadKernel(wgsl: string): u32 \| null` | WGSL を GPU に読み込み、kernel の番号を返す。GPU が使えない（`UM_WGPU=1` でないビルド、GPU がない、`UM_GPU=0`）か、WGSL をコンパイルできなければ `null`（コンパイルの誤りは標準エラー出力に出す） |
| `gpu.runKernel(kernel: u32, count: u32, ...引数): bool` | 読み込んだ kernel を、添字 0 から count 未満について GPU で実行する。実行を頼めたら `true`。束縛の並びが kernel と合わない（Buffer の数や `mut` が違うなど）と `false`（理由を標準エラー出力に出す）で、Buffer は変わらない。`UM_WGPU=1` でないビルドではいつも `false` |

- **引数は、Buffer / RacyBuffer と値を、kernel の引数の順に並べます**（dispatch と同じ。添字の引数は書かない）。値に使える型は `i32` / `u32` / `f32` / `bool` と、それらの struct・固定長配列（vec・行列を含む）、Buffer の要素にはさらに 8 / 16 ビットの整数です。64 ビットの整数と `f32` 以外の浮動小数点は使えません（`kernel-type`）。リテラルは型を書きます（`0.5 as f32`）。
- **実行は dispatch と同じく遅延します**（結果を CPU が読むときに読み戻す。続けて `runKernel` を呼べば GPU の中で続けて使う）。
- **GPU がない環境のために、CPU の版を用意してください。** MozaicCut のプラグインは、`gpu.loadKernel` が `null` を返したら（または `gpu.runKernel` が `false` を返したら）wasm の CPU 版に戻ります。

例: [tests/wgpu/load.um](../tests/wgpu/load.um)（プラグインと同じ形の kernel を読み込み、CPU の結果と比べる）。

### 束縛の並び（読み込める WGSL の約束）

`gpu.loadKernel` は、UM の `--emit-wgsl` が kernel について出す WGSL を、そのまま読み込めます。手で書いた WGSL も、同じ並びなら読み込めます。

- **入口は `@compute fn um_main`** で、`@workgroup_size(64)`。添字 `i` は `gid.y * um_params.row + gid.x` で、`i >= um_params.count` の実行は何もしない。
- **`@group(0) @binding(0)` は `var<storage, read> um_params: UmParams`。** `UmParams` の中身は次の順です（どれも 4 バイト単位で、隙間なく並ぶ）:
  1. `count: u32` — 実行する個数（`runKernel` の count）
  2. `row: u32` — 1 行の要素数（ランタイムが決める。2^30 未満）
  3. Buffer ごとに `len_<名前>: u32` — その Buffer の要素数（引数の順）
  4. 値の引数ごとに `p_<名前>: <型>` — 値（引数の順。`bool` は `u32` の 0 / 1）
- **`@group(0) @binding(1)` からは Buffer を引数の順に**: `var<storage, read>`（`mut` なしで渡す）か `var<storage, read_write>`（`mut` を付けて渡す）の `array<要素の型>`。要素の struct は WGSL の struct（4 バイト単位で隙間なく並ぶ）。8 / 16 ビットの整数の Buffer は `array<u32>`（書くなら `array<atomic<u32>>`）に詰めて渡します。
- 名前は自由です（並びだけが意味を持ちます）。`runKernel` に渡す値の型と順が `UmParams` と合わないと、kernel は誤った値を読みます（検出できません）。Buffer の数・`mut` の有無が合わないと `false` になります。

MozaicCut のプラグインの形（入力と出力の `Buffer<vec4<f32>>`、`u32` の w / h、`f32[8]` の引数）なら、`--emit-wgsl` は次の束縛を出します（`kernel function fx(i: u32, src: Buffer<vec4<f32>>, dst: mut Buffer<vec4<f32>>, w: u32, h: u32, args: f32[8])`）。

```wgsl
struct S_vec4__f32 {
  f_x: f32,
  f_y: f32,
  f_z: f32,
  f_w: f32,
}

struct UmParams {
  count: u32,
  row: u32,
  len_src: u32,
  len_dst: u32,
  p_w: u32,
  p_h: u32,
  p_args: array<f32, 8>,
}

@group(0) @binding(0) var<storage, read> um_params: UmParams;
@group(0) @binding(1) var<storage, read> b_src: array<S_vec4__f32>;
@group(0) @binding(2) var<storage, read_write> b_dst: array<S_vec4__f32>;
```

ホスト側は、同じ順で `gpu.runKernel(k, w * h, src, mut dst, w, h, args)` と呼びます。

## バックエンドの選び方

- **Windows の既定は Vulkan です**（wgpu が選ぶ順）。`UM_WGPU_BACKEND=dx12` で Direct3D 12 にできます。Direct3D 12 では、wgpu はシェーダを FXC（Windows に入っている古いコンパイラ）でコンパイルします。`tests/metal` と `tests/wgpu` は両方の API で CPU と一致しますが、FXC はまれに正しい WGSL をコンパイルできないことがあるので、既定の Vulkan を勧めます。
- **Mac の既定は Metal の経路（`UM_METAL=1`）のままにします。** Apple Silicon は CPU と GPU がメモリを共有しているので、Metal の経路は Buffer をコピーせずに GPU に渡し、常駐の仕組みなどで往復の手間も小さくしています。wgpu の経路は、Mac でも Buffer を GPU 側に写します。Mac での両者の速さは、この版では測っていません（下の「速さ」の手順で測れます）。

## 速さ

計測のプログラムは `bench/gpu/` にあります（どれも初めに CPU と GPU の結果が一致することを確かめます）。

```sh
UM_WGPU=1 node src/cli.ts build bench/gpu/pixels.um --target host -o pixels && ./pixels
UM_WGPU=1 node src/cli.ts build bench/gpu/blur.um   --target host -o blur   && ./blur
UM_WGPU=1 node src/cli.ts build bench/gpu/small.um  --target host -o small  && ./small
UM_WGPU=1 node src/cli.ts build bench/gpu/chain.um  --target host -o chain  && ./chain
# Mac の Metal と比べるときは、同じものを UM_METAL=1 で作る
```

測った環境: Windows 11、AMD Ryzen 9 3900X（12 コア 24 スレッド）、NVIDIA GeForce RTX 5060 Ti、wgpu-native v29.0.1.1（2026-11-21）。3 回の中央の値。CPU は全コア（`cpu.dispatch`）。

**1280x720 の画素ごとの kernel**（`pixels.um`。`vec4<f32>` の画素に色の行列・トーンカーブ・平方根。1 フレームあたり）

| 実行 | Vulkan | Direct3D 12 |
| --- | --- | --- |
| CPU（全コア） | 0.80 ms | 0.64 ms |
| GPU、毎フレーム CPU が結果を読む | 4.14 ms | 4.06 ms |
| GPU、2 つの kernel を続けて実行し、読まない | 0.56 ms（1 つあたり 0.28 ms） | 0.62 ms |
| 自動判断、毎フレーム読む | 1.07 ms（ほぼ CPU を選ぶ）。Buffer の置き場所を見込む版: 1.26 ms※ | 1.01 ms |

**1280x720 の箱ぼかし**（`blur.um`。半径 8、横と縦の 2 回。1 フレームあたり）

| 実行 | Vulkan | Direct3D 12 |
| --- | --- | --- |
| CPU（全コア） | 5.74 ms | 5.54 ms |
| GPU、毎フレーム CPU が結果を読む | 4.34 ms | 4.30 ms |
| GPU、読まない | 0.43 ms | 0.49 ms |
| 自動判断、毎フレーム読む | 6.36 ms（CPU を選ぶ）。Buffer の置き場所を見込む版: 5.72 ms※（同じときの CPU 6.14 ms、GPU で毎フレーム読む 5.11 ms） | 6.75 ms |

**小さな dispatch の繰り返し**（`small.um`。4096 要素、前の結果を次の入力に。1 回あたり）

| 実行 | Vulkan | Direct3D 12 |
| --- | --- | --- |
| CPU（全コア） | 25.5 µs | 25.7 µs |
| GPU、最後にだけ読む | 17.4 µs | 48.1 µs |
| GPU、毎回 CPU が読む | 105 µs | 181 µs |
| 自動判断、最後にだけ読む | 27.2 µs（CPU を選ぶ）。Buffer の置き場所を見込む版: 21.7 µs※（同じときの CPU 33.1 µs） | 24.6 µs |

**動画編集ソフトの描画に近い並び**（`chain.um`。毎フレーム、CPU が元の画像を書き、全画面の kernel 10 個（キャンバスを消す、元の画像を重ねる、明るさ・コントラスト・彩度、ぼかしの横と縦、周辺減光、色味、窓の画素に写す）を実行し、CPU が窓の画素の一部を描いて全体を読む。キャンバスは `vec4<f32>` の 14.7 MB。1 フレームあたり。Vulkan。2026-11-21、Buffer の置き場所を見込む版）

| 実行 | 時間 |
| --- | --- |
| CPU（全コア、`cpu.dispatch`） | 9.02 ms |
| GPU（`gpu.dispatch`） | 4.98 ms |
| `gpu.prefer(true)` と `dispatch` | 5.42 ms |
| 自動判断（600 フレーム。効果を 1 つずつ足して始める） | 6.15 ms（初めの 200 フレームを除くと 5.68 ms） |
| 同じ、前の版の自動判断（Buffer の置き場所を見込まない） | 12.2 ms（同じく 8.25 ms） |

- 前の版は、効果を 1 つずつ足していくと、どの kernel も前後が CPU のまま単独で GPU を試されて、送る時間の分だけ遅く見え、ほとんどのフレームを CPU で実行します（600 フレームで CPU の実行が 1438 回。この版は 198 回）。この版は、周期の比較で「GPU に寄せる」判断を選び、GPU で続けて実行します。
- 自動判断が `gpu.dispatch` より遅いのは、ときどきの測り直しと周期の比較の試しの分です。初めから GPU と分かっているなら `gpu.prefer(true)` を使えます。

※ の値は、ほかの仕事で機械が混んでいたときに測ったもので、表のほかの値（前の版）と同じ条件ではありません。3 回の中央の値。

**読み方:**

- **GPU の計算そのものは速く、時間の多くは写しです。** 1280x720 の `vec4<f32>`（14.7 MB）を GPU から読み戻すのに、この機械では約 3.5 ms かかります（GPU からホストへの写しが約 2.5 ms、CPU の領域への写しが約 1 ms）。結果を毎フレーム CPU で読むなら、軽い kernel は CPU の全コアのほうが速く、重い kernel（ぼかし）は GPU が速くなります。
- **効果を GPU で続けてかけ、最後に 1 回だけ読む使い方が最も速くなります**（ぼかしは 13 倍、画素ごとの kernel は 1 つあたり 2.5 倍以上）。動画の編集のように効果を重ねる処理では、途中の結果を CPU で読まないように並べてください。
- **Buffer を GPU に置いたままにする仕組みの効果**（`UM_WGPU_LAZY=0` で、dispatch のたびに送って読み戻したときとの比較。Vulkan）: 画素ごとの kernel を続けて 2 回 18.6 ms → 0.56 ms、ぼかし（読まない）18.5 ms → 0.43 ms、小さな dispatch 182 µs → 17 µs。
- **Direct3D 12 は、1 回の提出の手間が Vulkan より大きく**、小さな dispatch では遅くなります。この機械では Vulkan を勧めます。
- **Mac（Metal の経路と wgpu の経路）は、この版では測っていません。** Mac の既定を Metal の経路にしたのは、上の「バックエンドの選び方」の理由（コピーのない共有）によります。

## 環境変数

コンパイル時:

| 変数 | 意味 |
| --- | --- |
| `UM_WGPU=1` | wgpu の経路を使う（ネイティブの `--target host`） |
| `UM_WGPU_HOME` | wgpu-native を展開した場所 |

実行時（`runtime/wgpu.c`）:

| 変数 | 意味 |
| --- | --- |
| `UM_GPU=0` | GPU を使いません。`dispatch` はすべて CPU、`gpu.dispatch` は停止、`gpu.loadKernel` は `null` |
| `UM_GPU_THRESHOLD=N` | 自動判断の実測をやめ、要素数が N 以上なら GPU にします |
| `UM_GPU_TRACE=1` | 使う GPU と API、dispatch・送り・読み戻しと時間、自動判断の測定と周期の比較を標準エラー出力に出します。`2` なら dispatch ごとの選択と量も |
| `UM_WGPU_BACKEND` | `vulkan` / `dx12` / `metal` / `gl` のどれかに限ります（既定は wgpu が選ぶ） |
| `UM_WGPU_LAZY=0` | 比較用。Buffer を GPU に置いたままにせず、dispatch のたびに送って読み戻します |

テスト（`node tests/run.ts`）:

| 変数 | 意味 |
| --- | --- |
| `UM_TEST_WGPU_BACKENDS=vulkan,dx12` | wgpu のテストを、API ごとに実行します（既定は wgpu が選ぶ API で 1 回） |

## テスト

`node tests/run.ts` は、wgpu-native が見つかり（`UM_WGPU_HOME` など）GPU が使えれば、`tests/metal/*.um` と `tests/wgpu/*.um` を `UM_WGPU=1` で作って GPU で実行し、`// expect:` と照合します（どれも GPU の結果を CPU の結果と比べるプログラムです）。64 ビットの整数や simdgroup の集計を `gpu.dispatch` で使う、Metal だけのテストには `// wgpu-error: コード` があり、そのコードでコンパイルが止まることを確かめます。見つからなければ、wgpu のテストは飛ばします。`// cpu-build` のあるテスト（`tests/wgpu/prefer.um`）は、GPU の経路のないビルド（`UM_WGPU` なしのネイティブと wasm32）でも実行して、同じ出力になることを確かめます（wgpu-native がなくても実行します）。

## まだないもの・未確認

- Linux と macOS での実行（コードはありますが、確かめていません）。Windows の ARM64（wgpu-native のリリースに版があります）。
- GPU を選ぶ指定（いまは wgpu が「高性能」として選ぶ 1 つ）。
- 読み戻しと CPU のコピーを重ねること（いまは GPU から写し終えてから CPU の領域に写します）。
