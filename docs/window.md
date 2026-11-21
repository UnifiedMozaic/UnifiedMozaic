# Windows and input

[日本語](window.ja.md)

Native programs can open a window, show pixels in it and receive mouse and keyboard input.
It is built on [SDL3](https://www.libsdl.org/), so the same code runs on macOS, Windows and Linux. It is not available on wasm32.

This is a minimal entry point: everything above it — buttons, text, layout — is meant to be written in UM,
by drawing into a `Buffer<u32>` with kernels.

## Setup

SDL3 3.2.12 or later is needed, and it is linked only into programs that use a window.

```sh
brew install sdl3              # macOS
sudo apt install libsdl3-dev   # Debian / Ubuntu (or build SDL3 from source and set PKG_CONFIG_PATH)
```

On Windows, download `SDL3-devel-*-VC.zip` from the [SDL releases](https://github.com/libsdl-org/SDL/releases), unzip it,
and set `UM_SDL3` to that folder. `SDL3.dll` is copied next to the executable.

The compiler looks in `UM_SDL3` (the folder with `include/SDL3/SDL.h`), then `pkg-config sdl3`, then Homebrew's usual places.

## Example

```sh
node src/cli.ts build examples/window/paint.um --target host -o paint
./paint
```

Drag the square with the mouse or move it with the arrow keys (Shift for bigger steps). Esc quits.

## Built-in functions

All of them must be called from the main thread (macOS only allows windows there); otherwise the program stops.

| Function | |
| --- | --- |
| `windowOpen(title: string, width: i32, height: i32): i32 \| null` | Opens a resizable window and returns its number (from 1), or `null` |
| `windowClose(window: i32): void` | Closes the window |
| `windowPresent(window: i32, pixels: Buffer<u32>, width: i32, height: i32): void` | Shows `width × height` pixels (`0xAARRGGBB`), stretched over the window; waits for vsync |
| `pollWindowEvent(timeoutMs: i32): WindowEvent \| null` | Takes one event; if there is none, waits up to `timeoutMs` ms (`0` does not wait, negative waits forever) |

An unknown window number, a size that is not positive, or `pixels` shorter than `width × height` stops the program.

## Events

```ts
enum WindowEvent {
  Close(window: i32),
  Resize(window: i32, width: i32, height: i32),
  MouseMove(window: i32, x: i32, y: i32),
  MouseDown(window: i32, x: i32, y: i32, button: i32),    // 1 left, 2 middle, 3 right
  MouseUp(window: i32, x: i32, y: i32, button: i32),
  Wheel(window: i32, dx: i32, dy: i32),                   // up / right are positive
  KeyDown(window: i32, key: Key, mods: Mods, repeat: bool),
  KeyUp(window: i32, key: Key, mods: Mods),
  Text(window: i32, codepoint: u32),                      // typed text, including IME input
  Quit,
}
```

`Key` has `Enter`, `Escape`, `Backspace`, `Tab`, `Space`, `Delete`, the arrows, `Home`, `End`, `PageUp`, `PageDown` and `F(n)`;
other character keys are `Char(codepoint)` (lower case) and the rest are `Other(code)`. Read typed text from `Text`, not `KeyDown`.
`Mods` holds the `bool`s `shift`, `ctrl`, `alt` and `meta` (Command / Windows key). Coordinates are in window units, also on high-density screens.

## How it works

The built-ins live in `runtime/window.c` (C and SDL3). `WindowEvent`, `Key`, `Mods` and `pollWindowEvent` are written in UM in
`runtime/window.um` and are added to a program only when it uses one of those names, so programs without a window never link SDL3.
The lower-level `windowPoll(event: mut Buffer<i32>, timeoutMs: i32): bool` writes `[kind, window, a, b, c, d]` into `event`.

## More: audio, dialogs, drop, clipboard, cursors, IME, window control, HiDPI, time

The Japanese page ([window.ja.md](window.ja.md)) has the full reference; this is a summary. All of these are written in UM in
`runtime/window.um` on top of a few numbered low-level built-ins (`windowCtl`, `windowCtlF`, `windowSetText`, `windowGetText`, `audioWrite`).

- **Partial present**: `windowPresentRect(window, pixels, width, height, x, y, w, h)` uploads only the changed rectangle and presents;
  `windowUpdateRect(...)` uploads without presenting (for several rectangles). 1920×1080, D3D11: full present ≈ 2.1 ms, two small rects ≈ 0.4 ms.
  `windowSetVSync(window, on)`.
- **Audio** (`runtime/audio.c`): `audioOpen(rate, channels): i32 | null` (opens paused; `null` when there is no device),
  `audioWrite(stream, samples: Buffer<f32>, start, count)` (interleaved f32), `audioQueued`, `audioPause`, `audioResume`, `audioPaused`,
  `audioClear`, `audioPlayed` (frames handed to the device — use it as the A/V clock), `audioWritten`, `audioLatency`, `audioSetGain`, `audioClose`.
  Writes and queries may come from any thread.
- **File dialogs** (asynchronous): `windowOpenFileDialog(window, title, location, filters: DialogFilter[], many)`, `windowSaveFileDialog`,
  `windowOpenFolderDialog` return a request number; the result arrives as `WindowEvent.Dialog(request, paths, filter)` (empty `paths` = cancelled)
  or `DialogFailed(request)`.
- **Drop**: `DropBegin`, `DropPosition(x, y)`, `DropFile(path, x, y)`, `DropText(text, x, y)`, `DropComplete`.
- **Clipboard**: `clipboardText()`, `clipboardSetText(text)`, `clipboardHasText()`.
- **Cursors**: `windowSetCursor(CursorShape.Pointer)` etc., `windowShowCursor(show)`.
- **IME**: call `windowImeComposition(true)` before the first `windowOpen` to receive `TextEditing(window, text, start, length)` and draw the
  composition yourself; `windowSetTextInputArea(window, x, y, w, h, cursor)` places the OS candidate list; `windowStartTextInput` / `windowStopTextInput`.
- **Window control**: `windowSetTitle`, `windowTitle`, `windowSetSize`, `windowSize`, `windowSetPosition`, `windowPosition`, `windowSetMinimumSize`,
  `windowMaximize`, `windowMinimize`, `windowRestore`, `windowSetFullscreen`, `windowState`, `windowSetResizable`, `windowRaise`.
- **Window icon**: `windowSetIcon(window, pixels: Buffer<u32>, width, height): bool` sets the icon of a running window
  (`0xAARRGGBB`, e.g. 32×32 or 64×64; returns `false` when the video driver has no window icons, such as `offscreen`).
- **HiDPI and displays**: `windowPixelSize`, `windowPixelDensity`, `windowDisplayScale`, `windowDisplay`, `displayCount`, `displayBounds`,
  `displayUsableBounds`, `displayName`, `displayContentScale`, `displayRefreshRate`; events `PixelSizeChanged`, `ScaleChanged`, `DisplaysChanged`.
- **Other events**: `Focus`, `MouseEnter`, `MouseLeave`, `Moved`, `StateChanged`, `Exposed` (redraw needed).
- **Time**: `ticksNs()`, `sleepNs(ns)`, `sleepMs(ms)` (any thread).

Examples: `examples/window/tone.um`, `open_dialog.um`, `drop.um`, `ime.um`, `present_bench.um`.
