# ウィンドウと音

ネイティブのプログラムは、ウィンドウを開いて画素を表示し、マウスとキーボードの入力を受け取り、音を鳴らせます。
ファイルのダイアログ、ファイルのドロップ、クリップボード、カーソルの形、IME の入力中の文字、ウィンドウの操作、
複数のディスプレイと高密度の画面（HiDPI）、高精度の時計も使えます。
中身は [SDL3](https://www.libsdl.org/) で、macOS・Windows・Linux で同じコードが動きます。ブラウザ（wasm32）では、ウィンドウは canvas に、音は Web Audio になります（ダイアログ・メニューバーなどはまだない。[ブラウザで動かす](web.ja.md)）。

ライブラリ `um:window` です。`import * from "um:window";` で読み込みます（[ライブラリ](libraries.ja.md)）。

UI はすべて UM で書く、という前提の入口です。OS の標準の部品（ボタンなど）は使いません。ボタンや文字の描画などは、
`Buffer<u32>` に kernel で描きます。OS に任せるのは、ウィンドウ・入力・音・ファイルのダイアログ・クリップボードだけです。

## 準備

SDL3（3.2.12 以上）が必要です。ウィンドウか音を使うプログラムをビルドしたときだけリンクします。

```sh
# macOS
brew install sdl3

# Linux（Debian / Ubuntu。ディストリビューションに 3.2.12 以上がなければ、ソースからビルドして PKG_CONFIG_PATH を通す）
sudo apt install libsdl3-dev
```

Windows では [SDL のリリース](https://github.com/libsdl-org/SDL/releases) から `SDL3-devel-*-VC.zip` を取ってきて展開し、
環境変数 `UM_SDL3` にその場所を設定します。`SDL3.dll` は実行ファイルの隣に自動で写します（配布するときは同梱します）。

```powershell
$env:UM_SDL3 = "C:\SDL3-3.4.20"
```

探す順番は、`UM_SDL3`（`include/SDL3/SDL.h` のある場所）→ `pkg-config sdl3` → Homebrew の決まった場所です。

## 例

```sh
node src/cli.ts build examples/window/paint.um --target host -o paint
./paint
```

| 例 | 内容 |
| --- | --- |
| [paint.um](../examples/window/paint.um) | 四角をマウスでドラッグするか、矢印キーで動かす |
| [demo.um](../examples/window/demo.um) | 3D のデモ「宝石あつめ」。立方体を動かして宝石を集める（WASD・矢印・Space・マウスのドラッグ）。光・市松模様・霧のシェーダ、1 フレームに何回もの `gpu.draw`。GPU でも CPU でも、ブラウザ（`--web`）でも動く |
| [cube.um](../examples/window/cube.um) | 回る立方体。シェーダ（vertex / fragment function）で描き、`gpu.beginFrame` / `gpu.endFrame` で表示する（[描画](render.ja.md)）。`UM_WGPU=1` なら GPU で、なければ CPU で描く。Space で止める |
| [tone.um](../examples/window/tone.um) | 440 Hz の音を鳴らし、再生した標本の数を時計にして経過時間を出す（ウィンドウなし） |
| [open_dialog.um](../examples/window/open_dialog.um) | 開く・保存・フォルダのダイアログ。ボタンの上でカーソルが手になる。Ctrl+C / Ctrl+V でクリップボード |
| [drop.um](../examples/window/drop.um) | ファイルやフォルダを落とすと、パスと位置が出る。ドラッグ中は落とす先が明るくなる |
| [ime.um](../examples/window/ime.um) | IME の変換中の文字列をプログラムが描き、候補の一覧をカーソルの近くに出す |
| [present_bench.um](../examples/window/present_bench.um) | `windowPresent` と `windowPresentRect` の速さを比べる |

## 決まりごと

- ウィンドウの関数はメインスレッドから呼びます（macOS のウィンドウはメインスレッドでしか扱えないため）。違反すると停止します。
  例外は、時計と眠り（`ticksNs`・`sleepNs`・`sleepMs`）、`windowError`、音の書き込みと問い合わせ（`audioWrite`・`audioQueued`・
  `audioPlayed` など）で、どのスレッドからも呼べます。`audioOpen` と `audioClose` はメインスレッドから。
- ウィンドウの番号が開いていないもの、大きさが正でない、`pixels` が `width × height` より短い、矩形が画素の外に出る、
  音の標本の数がチャンネルの数の倍数でない、のどれかなら停止します（原因コード 12）。
- 失敗（ウィンドウを開けない、音のデバイスがない、など）は停止せず `null` / `false` を返します。理由は `windowError()` で読めます。

## ウィンドウと表示

| 関数 | 説明 |
| --- | --- |
| `windowOpen(title: string, width: i32, height: i32): i32 \| null` | ウィンドウを開き、番号（1 から）を返す。開けなければ `null`。大きさは変えられる。文字の入力は始まった状態 |
| `windowClose(window: i32): void` | ウィンドウを閉じる |
| `windowPresent(window, pixels: Buffer<u32>, width: i32, height: i32): void` | `0xAARRGGBB` の画素を `width × height` 並べたものを、ウィンドウ全体に引き伸ばして表示する。画面の更新（vsync）に合わせて待つ |
| `windowPresentRect(window, pixels, width, height, x, y, w, h): void` | `windowPresent` と同じだが、変わった矩形 `(x, y, w, h)` だけを写す。残りは前回の表示のまま。`w` か `h` が 0 なら写さずに表示だけ |
| `windowUpdateRect(window, pixels, width, height, x, y, w, h): void` | 矩形を写すだけで表示しない。変わった矩形が複数あるときは、`windowUpdateRect` を並べて、最後の1つを `windowPresentRect` にする |
| `windowSetVSync(window, on: bool): bool` | 表示を画面の更新に合わせるか（開いた直後は合わせる） |
| `pollWindowEvent(timeoutMs: i32): WindowEvent \| null` | イベントを1つ取り出す。なければ `timeoutMs` ミリ秒まで待つ（`0` は待たない、負は来るまで待つ） |

毎フレーム描き直すアニメーション（再生中）なら `pollWindowEvent(0)` を、入力があるまで描き直さなくてよい画面（停止中の編集画面）なら
`pollWindowEvent(-1)`（または上限つきの待ち時間）を使うと、CPU を使い続けずに済みます。待っている間に隠れていた部分が見えたときなどは
`Exposed` が届くので、そのとき描き直します。

### 変わった矩形だけを送る（windowPresentRect）

`windowPresent` は毎回、画面全体を GPU に写します。動画編集の画面では、毎フレーム変わるのは再生ヘッドや時刻の表示など一部だけです。
`windowPresentRect` / `windowUpdateRect` は変わった矩形だけを写します。大きさが変わった、描画の装置が作り直された、などで前回の表示が
残っていないときは、自動で全体を写します（その場合は `Exposed` も届くので、全体を描き直してください）。

速さ（[present_bench.um](../examples/window/present_bench.um)、1920×1080、vsync なし、Windows 11・Direct3D 11、Ryzen 9 3900X）:

| 方法 | 1 フレーム |
| --- | --- |
| `windowPresent`（全体の 8 MB を写す） | 約 2.1 ms |
| `windowUpdateRect` + `windowPresentRect`（4×1080 と 200×40 の2つ） | 約 0.4 ms |

### ウィンドウの操作

| 関数 | 説明 |
| --- | --- |
| `windowSetTitle(window, title: string): bool` / `windowTitle(window): string` | 題名 |
| `windowSetSize(window, width, height): bool` / `windowSize(window): WindowSize` | 大きさ（ウィンドウの座標） |
| `windowSetPosition(window, x, y): bool` / `windowPosition(window): WindowPos` | 位置（画面の座標。左のディスプレイでは負にもなる） |
| `windowSetMinimumSize(window, width, height): bool` | 利用者が縮められる下限（`0, 0` で制限なし） |
| `windowMaximize` / `windowMinimize` / `windowRestore(window): bool` | 最大化・最小化・元に戻す |
| `windowSetFullscreen(window, on: bool): bool` | 全画面（デスクトップの解像度のまま、枠をなくして画面を覆う） |
| `windowState(window): WindowState` | `fullscreen`・`maximized`・`minimized`・`focused`・`hidden` の `bool` |
| `windowSetResizable(window, on: bool): bool` | 大きさを変えられるか |
| `windowRaise(window): bool` | 前に出して焦点を移す |
| `windowSetIcon(window, pixels: Buffer<u32>, width, height): bool` | 窓のアイコン（題名の帯・タスクバー・Alt+Tab）。下の「アイコン」 |

`WindowSize` は `{ width, height }`、`WindowPos` は `{ x, y }`、`DisplayRect` は `{ x, y, width, height }`（どれも `i32`）です。

### アイコン

`windowSetIcon(window, pixels, width, height)` は、`0xAARRGGBB` の画素を `width × height` 並べたものを窓のアイコンにします
（`windowPresent` と同じ並び）。大きさは 32×32 か 64×64 くらいが目安です（OS が縮めます）。SDL が画素を写すので、呼び終えたら
`Buffer` は手放してかまいません。ドライバが窓のアイコンを持たないとき（`offscreen` など）は `false` を返します。

これは動いている窓のアイコンです。エクスプローラーなどに出る実行ファイル（.exe）のアイコンは別で、ビルドのときにリソースとして
埋め込みます。Windows では、`.ico` を指すリソースの定義を `llvm-rc` でリソース（`.res`）にして、`um build --link-input` で渡します。

```powershell
# app.rc の中身: 1 ICON "app.ico"（最初のアイコンが exe のアイコンになる）
& "C:\Program Files\LLVM\bin\llvm-rc.exe" -fo app.res app.rc
node src/cli.ts build app.um --target host -o app --link-input app.res
```

`.ico` には 16×16・32×32・48×48・256×256 などの大きさを入れておくと、OS が場面に合うものを選びます。

UM の実行ファイルはコンソールのプログラムなので、エクスプローラーからダブルクリックすると、窓の後ろに黒いコンソールの窓も出ます。
窓だけのアプリにするには、リンカへの指示（Windows のサブシステムと、入口を `main` のままにする指示）だけを持つオブジェクトを
作って、同じく `--link-input` で渡します。標準出力・標準エラー出力は、起動した側がつないだもの（パイプなど）にはそのまま出ます
（ダブルクリックで起動したときは、どこにも出ません）。

```powershell
# gui.c の中身:
#   #pragma comment(linker, "/SUBSYSTEM:WINDOWS")
#   #pragma comment(linker, "/ENTRY:mainCRTStartup")
& "C:\Program Files\LLVM\bin\clang.exe" -c gui.c -o gui.obj
node src/cli.ts build app.um --target host -o app --link-input app.res --link-input gui.obj
```

macOS の `.app` のアイコン（`Info.plist` と `.icns`）と Linux のデスクトップのアイコン（`.desktop` ファイル）は、実行ファイルの外に置くものなので、
`--link-input` は使いません（配布の形を作るときに置きます）。

### 高密度の画面（HiDPI）とディスプレイ

座標（マウスの位置、`windowSize`）はウィンドウの座標です。画素の数とは違うことがあります。

| 関数 | 説明 |
| --- | --- |
| `windowPixelSize(window): WindowSize` | 描画の面の画素の大きさ。くっきり描くには、この大きさの `Buffer` に描いて `windowPresent` に渡す |
| `windowPixelDensity(window): f64` | ウィンドウの座標 1 あたりの画素の数（macOS の Retina なら 2。Windows ではふつう 1） |
| `windowDisplayScale(window): f64` | 文字やボタンを大きく描く倍率（Windows の 150% の設定なら 1.5、Retina なら 2） |
| `windowDisplay(window): i32` | ウィンドウのあるディスプレイの番号（0 から。分からなければ -1） |
| `displayCount(): i32` | ディスプレイの数 |
| `displayBounds(index): DisplayRect \| null` / `displayUsableBounds(index)` | 範囲（画面の座標）／タスクバーなどを除いた範囲。番号が正しくなければ `null` |
| `displayName(index): string` | 名前 |
| `displayContentScale(index): f64` | 拡大の設定（125% なら 1.25） |
| `displayRefreshRate(index): f64` | 画面の更新の頻度（Hz。分からなければ 0） |

おすすめの描き方: `windowPixelSize` の大きさで描き、UI の寸法は `windowDisplayScale` 倍にし、マウスの座標は
`windowPixelDensity` 倍して画素の位置にします。Windows では座標がそのまま画素で、倍率は `windowDisplayScale` で分かります。
macOS では座標が論理的な単位で、`windowPixelDensity` が 2 になります。倍率の違うディスプレイへ移すと `ScaleChanged` と
`PixelSizeChanged` が届きます。ディスプレイの増減や設定の変化では `DisplaysChanged` が届くので、番号を読み直します。

## イベント

```ts
enum WindowEvent {
  Close(window: i32),                                     // 閉じるボタン（閉じるかどうかはプログラムが決める）
  Resize(window: i32, width: i32, height: i32),
  MouseMove(window: i32, x: i32, y: i32),
  MouseDown(window: i32, x: i32, y: i32, button: i32),    // button: 1 左、2 中、3 右
  MouseUp(window: i32, x: i32, y: i32, button: i32),
  MouseRelative(window: i32, dx: f64, dy: f64),           // 相対モードのマウスの動き（右・下が正。ウィンドウの座標の単位）
  Wheel(window: i32, dx: i32, dy: i32, fx: f64, fy: f64), // 上・右が正。dx, dy はノッチ（整数）、fx, fy は細かい量（ノッチ単位の小数。トラックパッドの滑らかなスクロール）
  KeyDown(window: i32, key: Key, mods: Mods, repeat: bool),
  KeyUp(window: i32, key: Key, mods: Mods),
  Text(window: i32, codepoint: u32),                      // 入力された文字（IME の確定も含む）
  Quit,                                                   // アプリの終了の要求
  DropBegin(window: i32),                                 // ドロップ: Begin → (Position | File | Text)* → Complete
  DropFile(window: i32, path: string, x: i32, y: i32),    // ファイルかフォルダ1つ（複数なら1つずつ届く）
  DropText(window: i32, text: string, x: i32, y: i32),
  DropPosition(window: i32, x: i32, y: i32),              // ドラッグ中の位置
  DropComplete(window: i32, x: i32, y: i32),
  TextEditing(window: i32, text: string, start: i32, length: i32),  // IME の変換中の文字列
  Dialog(request: i32, paths: string[], filter: i32),     // ダイアログの結果（paths が空なら取り消し）
  DialogFailed(request: i32),                             // ダイアログを出せなかった
  PixelSizeChanged(window: i32, width: i32, height: i32), // 描画の面の画素の大きさが変わった
  ScaleChanged(window: i32, scale: f64),                  // 表示の倍率が変わった
  Focus(window: i32, gained: bool),
  MouseEnter(window: i32),
  MouseLeave(window: i32),
  Moved(window: i32, x: i32, y: i32),
  StateChanged(window: i32, state: WindowState),          // 最大化・最小化・元に戻す・全画面の出入り
  Exposed(window: i32),                                   // 描き直しが要る
  DisplaysChanged,                                        // ディスプレイの増減・配置・倍率の変化
  Menu(id: string),                                       // メニューバーの項目が選ばれた（id は windowMenuItem に渡したもの）
}
```

`Key` は `Enter`・`Escape`・`Backspace`・`Tab`・`Space`・`Delete`・矢印・`Home`・`End`・`PageUp`・`PageDown`・`F(n)`、
文字のキーは `Char(codepoint)`（英字は小文字）、それ以外は `Other(code)` です。文字の入力は `KeyDown` ではなく `Text` で受け取ります
（Shift や IME を通した結果の文字が来ます）。`Mods` は `shift`・`ctrl`・`alt`・`meta`（Command / Windows キー）の `bool` です。

`match` には `_ => {}` を書いておくと、あとで選択肢が増えてもそのままコンパイルできます。

```ts
while (running) {
  while (true) {
    const e = pollWindowEvent(0);
    if (e == null) {
      break;
    }
    match (e) {
      WindowEvent.Close(w) => { running = false; }
      WindowEvent.DropFile(w, path, x, y) => { importMedia(path); }
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

## ファイルのダイアログ

OS の標準のダイアログを出します。どれもすぐに戻り、要求の番号を返します（出せなければ `null`）。結果は後で
`WindowEvent.Dialog(request, paths, filter)` で届きます。`paths` が空なら取り消しです。

| 関数 | 説明 |
| --- | --- |
| `windowOpenFileDialog(window, title, location, filters: DialogFilter[], many: bool): i32 \| null` | 開く。`many` で複数選べる |
| `windowSaveFileDialog(window, title, location, filters): i32 \| null` | 保存。`location` にファイルの名前まで入れると、それが最初に入る |
| `windowOpenFolderDialog(window, title, location, many): i32 \| null` | フォルダを選ぶ |

`window` は親にするウィンドウ（0 なら親なし）、`title` と `location`（最初のフォルダ）は空なら OS に任せます。
`DialogFilter` は `{ name: string; pattern: string }` で、`pattern` は拡張子を `;` で区切ったもの（`"mp4;mov;mkv"`）、`"*"` はすべてです。
`filter` は選ばれた絞り込みの番号です（分からなければ -1）。

```ts
const filters: DialogFilter[] = [{ name: "動画", pattern: "mp4;mov;mkv" }, { name: "すべて", pattern: "*" }];
const req = windowOpenFileDialog(win, "素材を読み込む", "", filters, true);
// ... pollWindowEvent で WindowEvent.Dialog(r, paths, filter) を待つ（r == req）
```

ダイアログが出ている間も、プログラムのイベントの輪は回り続けます（描き直しや再生を止めなくてよい）。

## メニューバー（macOS）

macOS の画面の上のメニューバー（`NSMenu`）を組み立てます。macOS の本物の画面（`windowHasNativeMenuBar()` が `true`）のときだけ働き、
Windows・Linux・`offscreen` では何もしません（`false` を返します）。選ばれた項目は `WindowEvent.Menu(id)` で届きます。

| 関数 | 説明 |
| --- | --- |
| `windowPlatform(): string` | 動いている OS の名前（`"macos"`・`"windows"`・`"linux"`） |
| `windowHasNativeMenuBar(): bool` | OS のメニューバーが使えるか（macOS で、`offscreen` / `dummy` でないとき `true`） |
| `windowMenuBegin(): void` | メニューバーの組み立てを始める |
| `windowMenuBeginAppMenu(title: string): void` | macOS のアプリのメニュー（左端の太字のメニュー）を始める |
| `windowMenuBeginMenu(title: string): void` | トップレベルのメニュー（`"ファイル"`・`"編集"` など）を始める |
| `windowMenuBeginWindowMenu(title: string): void` | macOS 標準の「ウィンドウ」メニューを始める |
| `windowMenuBeginSubMenu(title: string): void` | サブメニューを始める |
| `windowMenuEndMenu(): void` | いま組み立てているメニュー（またはサブメニュー）を閉じる |
| `windowMenuSep(): void` | 区切りの線を足す |
| `windowMenuItem(id, label, key: string, mods: i32, enabled, checked: bool): void` | 項目を 1 つ足す。選ばれると `WindowEvent.Menu(id)` が届く。`mods` は `1` Shift・`2` Ctrl・`4` Alt/Option・`8` Meta/Command |
| `windowMenuRole(role: MenuRole, label: string): void` | OS 標準の動作（`Services`・`Hide`・`HideOthers`・`ShowAll`・`Minimize`・`Zoom`・`Fullscreen`・`BringAllFront`）を足す |
| `windowMenuCommit(): bool` | 組み立てたメニューバーを OS に反映する |
| `windowMenuSetItemState(id: string, enabled, checked: bool): bool` | 反映済みの項目の有効／無効とチェックの印だけを変える |

## ファイルのドロップ

ウィンドウにファイルやフォルダを落とすと、`DropBegin` → `DropPosition`（ドラッグ中、何度も）→ `DropFile`（1つずつ）→ `DropComplete` の順に届きます。
位置はウィンドウの座標です。ドラッグ中に落とす先（素材の箱かタイムラインか）を強調するには `DropPosition` を使います
（OS によっては `DropPosition` が来ないこともあります）。

## クリップボード

| 関数 | 説明 |
| --- | --- |
| `clipboardText(): string` | クリップボードの文字（なければ空） |
| `clipboardSetText(text: string): bool` | 書き込む |
| `clipboardHasText(): bool` | 文字があるか |

## カーソル

| 関数 | 説明 |
| --- | --- |
| `windowSetCursor(c: CursorShape): bool` | カーソルの形を変える（すべてのウィンドウで同じ） |
| `windowShowCursor(show: bool): bool` | 見せる・隠す |

`CursorShape` は `Default`（矢印）・`Text`（文字の入力）・`Wait`（待ち）・`Crosshair`（十字）・`Progress`（処理中だが操作できる）・
`ResizeNWSE`・`ResizeNESW`・`ResizeEW`（左右の矢印）・`ResizeNS`（上下の矢印）・`Move`・`NotAllowed`（禁止）・`Pointer`（手）です。

### マウスの相対モード

一人称の視点のように、マウスの動きの量で向きを変えるときに使います。カーソルを隠してウィンドウに閉じ込め、画面の端でも
止まらずに動きの量を届けます。

| 関数 | 説明 |
| --- | --- |
| `windowSetRelativeMouse(window, on: bool): bool` | 入れる・切る。入れているあいだは `MouseMove` の代わりに `MouseRelative(window, dx, dy)` が届く |
| `windowRelativeMouse(window): bool` | 入っているか |

```ts
WindowEvent.MouseRelative(w, dx, dy) => {
  yaw -= (dx as f32) * 0.003;
  pitch -= (dy as f32) * 0.003;
}
```

ブラウザでは Pointer Lock を使います。ブラウザの決まりで、クリックの中からしか入れないので、`windowSetRelativeMouse(w, true)`
のあとの最初のクリックで入ります。Esc を押すとブラウザが抜けさせますが、もう一度クリックすれば入り直します
（`windowRelativeMouse` は、いま入っているかを返します）。

## 文字の入力と IME

| 関数 | 説明 |
| --- | --- |
| `windowImeComposition(on: bool): bool` | IME の変換中の文字列をプログラムが描くか。最初の `windowOpen` より前に呼ぶ（後では `false`） |
| `windowStartTextInput(window)` / `windowStopTextInput(window): bool` | 文字の入力を始める・止める（開いた直後は入力中） |
| `windowTextInputActive(window): bool` | 入力中か |
| `windowSetTextInputArea(window, x, y, width, height, cursor): bool` | 入力欄の矩形（ウィンドウの座標）と、欄の中のカーソルの x。IME の候補の一覧がこの近くに出る |

既定では、変換中の文字列は OS が描き、確定した文字だけが `Text` で届きます。`windowImeComposition(true)` にすると、
変換中の文字列が `TextEditing(window, text, start, length)` で届き、プログラムが入力欄の中に描きます（`start` と `length` は
選んでいる文節で、単位は文字。なければ -1）。`text` が空になったら変換は終わりです（確定した文字は続く `Text` で届きます）。
候補の一覧は OS が `windowSetTextInputArea` の近くに出すので、カーソルが動くたびに知らせます。

文字の入力を止めると、文字のキーがすべて `KeyDown` で届き、IME も働きません（ショートカットだけを受け付ける画面向け。
テキスト欄に焦点が移ったら `windowStartTextInput` で再開します）。

## 音の出力

既定の出力のデバイスに、f32 の標本を書き込んで鳴らします。標本はチャンネルを交互に並べます（ステレオなら L R L R …）。
1 フレーム = チャンネルの数の標本です。SDL が出力のデバイスの形式（標本化周波数など）への変換をします。
ブラウザでは Web Audio で鳴らします（[ブラウザで動かす](web.ja.md)。最初のクリックかキーまでは鳴りません）。

| 関数 | 説明 |
| --- | --- |
| `audioOpen(sampleRate: i32, channels: i32): i32 \| null` | 開いて出力の番号を返す。音のデバイスがなければ `null`。**一時停止の状態で開く** |
| `audioWrite(stream, samples: Buffer<f32>, start: i64, count: i64): bool` | `samples[start ..]` から `count` 個の標本を待ち行列に足す（`count` が負なら終わりまで。チャンネルの数の倍数） |
| `audioQueued(stream): i64` | まだ鳴らしていない（待ち行列にある）フレームの数。これが減ったら書き足す |
| `audioPause` / `audioResume(stream): bool` | 一時停止・再開 |
| `audioPaused(stream): bool` | 一時停止中か |
| `audioClear(stream): bool` | 待ち行列を捨てる（シークのとき）。捨てた分は `audioPlayed` に数えない |
| `audioPlayed(stream): i64` | 開いてから再生した（デバイスに渡った）フレームの数 |
| `audioWritten(stream): i64` | 開いてから書き込んだフレームの数（捨てた分を引く）。`audioPlayed + audioQueued` に等しい |
| `audioLatency(stream): i64` | デバイスの中にたまる分のフレームの数（渡ってから耳に届くまでの遅れ。分からなければ 0） |
| `audioSetGain(stream, gain: f64): bool` | 音量（1.0 で元のまま） |
| `audioClose(stream): void` | 閉じる |

### 映像との同期

音を時計にします。今の再生位置（秒）は、おおよそ次のとおりです。

```ts
const t = ((audioPlayed(s) - audioLatency(s)) as f64) / (rate as f64) + startOffset;
```

`audioPlayed` はデバイスに渡った分で、CPU の時計と違って音の途切れや一時停止に合わせて止まります。映像の
フレームは、この時刻に合うものを表示します。シークでは `audioPause` → `audioClear` → 新しい位置の標本を書く → `audioResume` とし、
`startOffset` を新しい位置から `audioPlayed` を引いた分にします。

待ち行列は 50〜200 ミリ秒ほど保つのが目安です（短いと途切れ、長いと操作への反応が遅れます）。`audioWrite` はほかのスレッドからも
呼べるので、デコードのスレッドから直接書き足せます。

## 時刻

| 関数 | 説明 |
| --- | --- |
| `ticksNs(): i64` | 逆戻りしない時計（ナノ秒）。どのスレッドからも呼べる。起点は決まっていない（SDL を初期化する前の最初の呼び出しでは 0 になる）ので、差だけを使う |
| `sleepNs(ns: i64)` / `sleepMs(ms: i64)` | 指定の時間だけ眠る。1 ミリ秒より短くても正確（最後は待ち続けて合わせる） |

`clock()`（秒の `f64`）は SDL3 なしで使えます。`ticksNs` と `sleep*` は SDL3 をリンクします。

## しくみ

- 組み込みの部分は `runtime/window.c`（ウィンドウ・入力・ダイアログ・クリップボードなど）と `runtime/audio.c`（音）で、C と SDL3 です。
  名前のある関数と `WindowEvent` などの型は `lib/window.um`（UM のライブラリ `um:window`）にあり、`import * from "um:window";` で読み込みます。
  ウィンドウも音も使わないプログラムは SDL3 をリンクしません。
- `lib/window.um` は、次の低い層の組み込みを操作番号で呼び分けています（番号は `runtime/window.c` と `runtime/audio.c` の enum）。これらは `lib/window.um` の中だけで使え、プログラムからは呼べません。
  - `windowPoll(event: mut Buffer<i32>, timeoutMs: i32): bool` — `event` の先頭 6 要素に `[種類, ウィンドウ, a, b, c, d]` を書く
  - `windowCtl(op: i32, window: i32, io: mut Buffer<i64>): i64` — 引数を `io` から読み、結果を `io` と戻り値で返す
  - `windowCtlF(op: i32, window: i32, a: i64): f64` / `windowSetText(op, window, a, text: string): i64` / `windowGetText(op, window, a): string`
- イベントの文字列（ドロップのパス、IME の文字列、ダイアログのパス）は、次の `windowPoll` まで C の側に置いてあり、
  `pollWindowEvent` が `windowGetText` で読みます。
- SDL のダイアログは非同期で、結果はほかのスレッドから届くことがあります。届いた結果は SDL のイベントの列に積み、
  `pollWindowEvent` がメインスレッドで `Dialog` にします。
- 表示は SDL のレンダラ（macOS は Metal、Windows は Direct3D）のテクスチャに写して描きます。高密度の画面では、描画の面を画素の大きさで作ります
  （`SDL_WINDOW_HIGH_PIXEL_DENSITY`）。
- 試験（`node tests/run.ts`）は、`SDL_VIDEO_DRIVER=offscreen` と `SDL_AUDIO_DRIVER=dummy` で画面と音のデバイスなしに実行します。
  ドロップと IME は人の操作がないと起きないので、試験用の関数 `windowTestEvent(window, 種類, 文字列)` で SDL のイベントを積み、
  変換の経路を確かめます（種類は `runtime/window.c` の `WS_TEST_EVENT`）。ダイアログは自動の試験がありません。
