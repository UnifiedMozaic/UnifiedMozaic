# ブラウザで動かす

UM のプログラムを wasm32 にして、ブラウザで動かします。ウィンドウ（`windowOpen` など）は canvas になり、描画（[描画](render.ja.md)）は WebGPU か CPU で行います。ネイティブと同じソースが、そのまま動きます。

## 作り方

```sh
node src/cli.ts build examples/window/cube.um --web -o cube-web
python3 -m http.server -d cube-web 8000     # などで HTTP で配る（file:// では wasm を読み込めない）
# ブラウザで http://localhost:8000/ を開く
```

`--web` は、次のファイルを書いたフォルダを作ります（`-o` で場所を指定。なければ `<名前>-web`）。

| ファイル | 中身 |
| --- | --- |
| `index.html` | 読み込むページ（`runtime/web/index.html`）。canvas と、`print` の出力の欄がある |
| `app.wasm` | プログラム（`--target wasm32` と同じ） |
| `host.mjs`、`web.mjs` | ランタイム（`runtime/host.mjs` と `runtime/web.mjs`） |

URL で、プログラムに引数を渡せます。

| URL | 意味 |
| --- | --- |
| `?args=a,b` | `args()` が `["a", "b"]` を返す |
| `?gpu=0` | WebGPU を使わない（描画も kernel も CPU で行う） |

自分のページに組み込むときは、`host.mjs` の `instantiate(bytes, { canvas, print, ... })` で読み込み、`await exports.main()` を呼びます（`index.html` が例）。

## 要るもの

- **JSPI（WebAssembly JavaScript Promise Integration）。** 表示（`windowPresent`・`gpu.endFrame`）と待つ `pollWindowEvent` のたびに、wasm を一時停止してブラウザに画面の更新と入力の処理をさせます。こうすると、`while (running) { 入力; 描く; 表示 }` のループのまま動きます。Chrome / Edge は 137 から使えます。Firefox は 153 で標準で有効になる予定、Safari はまだです（2026 年 11 月の時点。最新の対応は各ブラウザの情報を確かめてください）。JSPI のないブラウザでは、表示しても待たないので、ループの間に画面が更新されません。
- **WebGPU（任意）。** あれば、描画（`gpu.draw`）と kernel の `dispatch` を GPU で行います。なければ CPU で行います（同じ結果）。

## ブラウザでのウィンドウ

| ネイティブ | ブラウザ |
| --- | --- |
| `windowOpen` | canvas を使う（1 つ目は `index.html` の canvas。2 つ目からは作ってページに足す）。題名はページの題名になる |
| 大きさ | canvas の大きさ（CSS の画素）。`windowPixelSize` は canvas の画素（`devicePixelRatio` 倍） |
| イベント | キー（SDL と同じキーコード）、文字の入力、マウス、ホイール、大きさの変化、焦点。キーは canvas に焦点があるときだけ届く（クリックで焦点が移る） |
| `windowPresent` | canvas の 2D に描き、次の描画の時機（`requestAnimationFrame`）まで待つ |
| `gpu.beginFrame` | WebGPU があれば canvas の WebGPU の面に描く。canvas は 2D と WebGPU のどちらかしか使えないので、GPU で描き始めた canvas に `windowPresent` は使えない |
| `ticksNs` / `sleepMs` | `performance.now()` / `setTimeout`（眠っているあいだは一時停止） |
| 全画面 | `windowSetFullscreen` は canvas を全画面にする（ブラウザの決まりで、クリックやキーの処理の中からしか効かないことがある） |
| カーソル | `windowSetCursor` / `windowShowCursor` は canvas の CSS の cursor |
| マウスの相対モード | `windowSetRelativeMouse` は Pointer Lock。クリックの中からしか入れないので、入れたあとの最初のクリックで入る。Esc で抜けたら、次のクリックで入り直す |
| クリップボード | 書くのは `navigator.clipboard`。読むのはまだない（空） |
| 音（`audioOpen` など） | Web Audio。出力ごとに、決めた標本化周波数の `AudioContext` を作る。数（書き込んだ・待ち行列にある・再生した）の意味はネイティブと同じ。ブラウザは、ページの上のクリックかキーの前には音を鳴らさないので、それまでは `audioResume` しても再生した数が進まない（最初のクリックかキーで鳴り始める） |

**まだないもの:** OS のダイアログ・メニューバー、ファイルのドロップ、IME の編集中の文字列、ウィンドウの位置。

## 確かめた環境

- Chromium 158（Playwright。ヘッドレス）。`tests/run.ts` の「ブラウザ」の節が、Playwright があれば `tests/web/*.um`（音は自動再生を許して開く）と描画のテスト（`tests/wgpu/draw*.um`）を、WebGPU（SwiftShader）ありとなしで開いて出力を照合する。
- WebGPU の画像に描いて読み戻す描画は、SwiftShader で CPU と同じ結果になった（辺のすぐ近くの数画素を除く）。
- WebGPU で canvas に描くフレーム（`gpu.beginFrame`）は、エラーなく動くことまでを確かめた。このヘッドレスの環境では、WebGPU の canvas を画面に出すと WebGPU が使えなくなる（UM を使わない最小のページでも同じ）ため、表示された絵は確かめていない。実際のブラウザでの確認が要る。
