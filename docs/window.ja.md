# ウィンドウ

ネイティブのプログラムは、ウィンドウを開いて画素を表示し、マウスとキーボードの入力を受け取れます。
中身は [SDL3](https://www.libsdl.org/) で、macOS・Windows・Linux で同じコードが動きます。wasm32 では使えません。

UI もここから先はすべて UM で書く、という前提の最小限の入口です。ボタンや文字の描画などは、
`Buffer<u32>` に kernel で描きます。

## 準備

SDL3（3.2.12 以上）が必要です。ウィンドウを使うプログラムをビルドしたときだけリンクします。

```sh
# macOS
brew install sdl3

# Linux（Debian / Ubuntu。ディストリビューションに 3.2.12 以上がなければ、ソースからビルドして PKG_CONFIG_PATH を通す）
sudo apt install libsdl3-dev
```

Windows では [SDL のリリース](https://github.com/libsdl-org/SDL/releases) から `SDL3-devel-*-VC.zip` を取ってきて展開し、
環境変数 `UM_SDL3` にその場所を設定します。`SDL3.dll` は実行ファイルの隣に自動で写します。

```powershell
$env:UM_SDL3 = "C:\SDL3-3.2.24"
```

探す順番は、`UM_SDL3`（`include/SDL3/SDL.h` のある場所）→ `pkg-config sdl3` → Homebrew の決まった場所です。

## 例

```sh
node src/cli.ts build examples/window/paint.um --target host -o paint
./paint
```

四角をマウスでドラッグするか、矢印キーで動かします（Shift で速く）。Esc で終わります。

## 組み込み関数

どれもメインスレッドから呼びます（macOS のウィンドウはメインスレッドでしか扱えないため）。違反すると停止します。

| 関数 | 説明 |
| --- | --- |
| `windowOpen(title: string, width: i32, height: i32): i32 \| null` | ウィンドウを開き、番号（1 から）を返す。開けなければ `null`。大きさは変えられる |
| `windowClose(window: i32): void` | ウィンドウを閉じる |
| `windowPresent(window: i32, pixels: Buffer<u32>, width: i32, height: i32): void` | `0xAARRGGBB` の画素を `width × height` 並べたものを、ウィンドウ全体に引き伸ばして表示する。画面の更新（vsync）に合わせて待つ |
| `pollWindowEvent(timeoutMs: i32): WindowEvent \| null` | イベントを1つ取り出す。なければ `timeoutMs` ミリ秒まで待つ（`0` は待たない、負は来るまで待つ） |

番号が開いていないウィンドウのもの、大きさが正でない、`pixels` が `width × height` より短い、のどれかなら停止します。

毎フレーム描き直すアニメーションなら `pollWindowEvent(0)` を、入力があるまで描き直さなくてよいエディタのような画面なら
`pollWindowEvent(-1)`（または上限つきの待ち時間）を使うと、CPU を使い続けずに済みます。

## イベント

```ts
enum WindowEvent {
  Close(window: i32),                                     // 閉じるボタン（閉じるかどうかはプログラムが決める）
  Resize(window: i32, width: i32, height: i32),
  MouseMove(window: i32, x: i32, y: i32),
  MouseDown(window: i32, x: i32, y: i32, button: i32),    // button: 1 左、2 中、3 右
  MouseUp(window: i32, x: i32, y: i32, button: i32),
  Wheel(window: i32, dx: i32, dy: i32),                   // 上・右が正
  KeyDown(window: i32, key: Key, mods: Mods, repeat: bool),
  KeyUp(window: i32, key: Key, mods: Mods),
  Text(window: i32, codepoint: u32),                      // 入力された文字（IME の確定も含む）
  Quit,                                                   // アプリの終了の要求
}
```

`Key` は `Enter`・`Escape`・`Backspace`・`Tab`・`Space`・`Delete`・矢印・`Home`・`End`・`PageUp`・`PageDown`・`F(n)`、
文字のキーは `Char(codepoint)`（英字は小文字）、それ以外は `Other(code)` です。文字の入力は `KeyDown` ではなく `Text` で受け取ります
（Shift や IME を通した結果の文字が来ます）。`Mods` は `shift`・`ctrl`・`alt`・`meta`（Command / Windows キー）の `bool` です。

座標はウィンドウの座標系です。Retina のような高密度の画面でも、`windowOpen` に渡した大きさの単位で届きます。

```ts
while (running) {
  while (true) {
    const e = pollWindowEvent(0);
    if (e == null) {
      break;
    }
    match (e) {
      WindowEvent.Close(w) => { running = false; }
      WindowEvent.KeyDown(w, key, mods, repeat) => {
        match (key) {
          Key.Escape => { running = false; }
          _ => {}
        }
      }
      _ => {}
    }
  }
  dispatch(draw, px.length as u32, mut px, ...);
  windowPresent(win, px, width, height);
}
```

## しくみ

- 組み込みの部分は `runtime/window.c`（C と SDL3）。`WindowEvent`・`Key`・`Mods`・`pollWindowEvent` は `runtime/window.um`（UM）で、
  プログラムがこれらの名前を使うときだけ標準の関数として加わります。ウィンドウを使わないプログラムは SDL3 をリンクしません。
- 低い層の `windowPoll(event: mut Buffer<i32>, timeoutMs: i32): bool` も呼べます。`event` の先頭 6 要素に
  `[種類, ウィンドウ, a, b, c, d]` を書きます（種類の番号は `runtime/window.c` を見てください）。
- 表示は SDL のレンダラ（macOS は Metal、Windows は Direct3D）のテクスチャに写して描きます。
