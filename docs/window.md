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
