# Running WebAssembly

[日本語](wasm.ja.md)

Native programs can load WebAssembly modules (`.wasm`) and call their functions. It is built on the
[wasmtime](https://wasmtime.dev/) C API, which compiles wasm to native code (JIT), so plugins run close to native speed.
This is meant for shipping OS-independent plugins that run isolated from the host program.

- A module runs only inside its instance. By default it gets no host functions (imports) and no WASI, so it cannot touch files, the network or the host's memory.
- Traps, running out of fuel or time, and load failures do not stop the program: they come back as `Result.Err(reason)`.
- Several threads can use separate instances at the same time.
- Not available on wasm32 (except `wasmCustomSection`).

The Japanese page has the full reference; this page is a summary.

## Setup

Download the C API for your OS from the [wasmtime releases](https://github.com/bytecodealliance/wasmtime/releases)
(`wasmtime-v*-x86_64-windows-c-api.zip`, `-aarch64-macos-c-api.tar.xz`, `-x86_64-linux-c-api.tar.xz`, ...), unpack it and set
`UM_WASMTIME` to that folder (the one with `include/wasmtime.h` and `lib`). It is linked only into programs that use these functions.
The compiler looks in `UM_WASMTIME`, then `pkg-config wasmtime`, then Homebrew's `/opt/homebrew/opt/wasmtime`.
On Windows `wasmtime.dll` is copied next to the executable. Tested with wasmtime 51.0.0 on Windows 11 (x64) and Linux
(Ubuntu 24.04 on WSL, x64); macOS and the `pkg-config` lookup are not tested yet.

## Functions

| Function | Meaning |
| --- | --- |
| `wasmLoad(bytes: u8[]): Result<i64, string>` / `wasmLoadFile(path)` | Compile a module (cached on disk); returns a module handle |
| `wasmFromWat(text: string): Result<u8[], string>` | Text format to binary |
| `wasmNew(module: i64, limits: WasmLimits): Result<i64, string>` | Create an instance (own memory and globals) |
| `wasmCall(instance: i64, name: string, args: WasmValue[]): Result<WasmValue[], string>` | Call an exported function (i32 / i64 / f32 / f64) |
| `wasmWrite(instance, offset, src[, start, count]): bool` / `wasmRead(instance, offset, mut dst[, start, count]): bool` | Bulk copy between wasm memory and a numeric array or Buffer |
| `wasmMemorySize(instance): i64` | Memory size in bytes |
| `wasmExports(module): WasmExport[]` | Exported functions, memories, ... |
| `wasmCustomSection(bytes: u8[], name: string): u8[] \| null` | Read a custom section without compiling (plugin manifests) |
| `wasmFreeInstance(instance)` / `wasmFreeModule(module)` | Free handles |

`WasmLimits` has `fuel` (per call), `timeoutMs` (per call, epoch based), `memoryBytes` and `stubImports` (fill imports with trapping functions); 0 means no limit.
Fuel and epoch checks slow generated code down (15–35%), so each instance runs code compiled with only the checks its limits need.
A handle that was freed or has the wrong kind stops the program (code 13). Using the same instance from two threads at once fails for the later caller.

## Plugins written in UM (`--target wasm32-plugin`)

`node src/cli.ts build plugin.um --target wasm32-plugin -o plugin.wasm` builds a module with **no imports**:
`print` and other I/O are build errors (`plugin-import`), kernels are dispatched on one CPU thread, and stops become `unreachable` traps
(the host reads the reason with the exported `umPanicCode()` / `umPanicLine()`).
Exported functions take `Buffer<T>` parameters as two i32s (element offset, element count), plus numbers and bool; the host allocates
with the exported `umAlloc(bytes)` / `umFree(at)`. SIMD (simd128) is enabled. See [examples/wasm/um_plugin.um](../examples/wasm/um_plugin.um).

## Examples

`examples/wasm/`: `add.um`, `pixels.um`, `limits.um`, `threads.um`, `custom_section.um`, `um_plugin.um`, and `bench/` for speed measurements.
