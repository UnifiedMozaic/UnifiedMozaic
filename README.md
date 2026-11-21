<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo/logo-white.svg">
    <img src="assets/logo/logo-black.svg" alt="UnifiedMozaic logo" width="120">
  </picture>
</p>

# UnifiedMozaic

**One language. Every core.**

[日本語](README.ja.md)

UnifiedMozaic is a small, statically typed language with TypeScript-like syntax. It compiles through LLVM to native executables and WebAssembly, with no garbage collector, and lets you write CPU and GPU code in the same `.um` source.

> **Status: experimental.** The language and its runtime are developed by one person and change often. The detailed documentation is written in Japanese.

```ts
struct Particle {
  pos: f32;
  vel: f32;
}

// A kernel runs once per index, on the CPU (all cores) or on the GPU.
kernel function integrate(i: u32, ps: mut Buffer<Particle>, dt: f32): void {
  ps[i].pos = ps[i].pos + ps[i].vel * dt;
}

export function main(): i32 {
  let ps = new Buffer<Particle>(1000);        // fixed length, zero-filled
  ps[0].vel = 2;
  dispatch(integrate, ps.length, mut ps, 0.5);
  print(ps[0].pos);                           // 1
  return 0;
}
```

## Features

- **TypeScript-like syntax, explicit types.** Sized numeric types (`i8`–`i64`, `u8`–`u64`, `f16`/`f32`/`f64`, and small floats `f4`/`f8`), `struct`, `class` (no inheritance), `enum` with `match`, interfaces, generics, strings, nullable types. No implicit conversions.
- **Ownership instead of GC.** Arrays and other heap values have a single owner. Arguments are borrowed (default), mutably borrowed (`mut`) or moved (`own`), checked at compile time.
- **Safe by default.** Out-of-bounds access, integer division by zero and oversized shifts stop the program with a clear message; integer overflow wraps; there is no undefined behaviour and no `unsafe`.
- **One source for CPU and GPU.** `kernel` functions have the same meaning on every backend. They run on a CPU thread pool, on GPUs on Windows, Linux and macOS through wgpu (WGSL), on Apple GPUs through Metal (MSL), and on WebGPU in the browser. `dispatch` chooses CPU or GPU by measurement; `gpu.submit` / `gpu.ready` / `gpu.wait` run GPU work asynchronously.
- **Threads.** `thread.spawn` / `thread.join`, with `RacyBuffer` as the only shared mutable data (atomic element access, so no data races).
- **Low-level speed when you need it.** 128-bit SIMD types (`i8x16`, `i16x8`, `f32x4`, …), `Math.prefetch`, masked shifts, bounds-check elimination, bulk `fill`.
- **Diagnostics for humans and AI.** Errors carry a code, a location and a hint, and `check --json` gives machine-readable diagnostics.
- **No npm dependencies.** The compiler is plain TypeScript run directly by Node.js; the only external tool is `clang`.

## Requirements

- Node.js 22.18 or later (runs the TypeScript sources directly)
- `clang` from LLVM 18 or later; `wasm-ld` for wasm32 output
  - On macOS, Xcode's clang cannot target wasm32. Run `brew install llvm lld`; wasm32 builds then use it automatically.
  - On Windows, install LLVM (`winget install LLVM.LLVM`) and the MSVC tools from Visual Studio or the Build Tools ("Desktop development with C++").
- For GPU execution on any OS: wgpu-native ([docs/wgpu.ja.md](docs/wgpu.ja.md), Japanese)
- For GPU execution on a Mac through Metal: Apple Silicon with Xcode or the Command Line Tools (Metal)

Step-by-step instructions, including the `um` command (`npm link`) and troubleshooting, are in [docs/setup.md](docs/setup.md).

## Quick start

```sh
git clone https://github.com/UnifiedMozaic/UnifiedMozaic.git
cd UnifiedMozaic

# Native executable for this machine
node src/cli.ts build examples/structs.um --target host -o structs
./structs

# Native executable that runs kernels on the GPU (Windows, Linux, macOS; set UM_WGPU_HOME to wgpu-native)
UM_WGPU=1 node src/cli.ts build examples/kernels.um --target host -o kernels

# Native executable that runs kernels on the GPU through Metal (Apple Silicon)
UM_METAL=1 node src/cli.ts build examples/kernels.um --target host -o kernels
./kernels

# WebAssembly (default target)
node src/cli.ts build examples/structs.um -o structs.wasm
node runtime/run-node.mjs structs.wasm

# Type-check only (add --json for machine-readable diagnostics)
node src/cli.ts check examples/structs.um

# Inspect generated code
node src/cli.ts build examples/kernels.um --emit-llvm -o kernels.ll
node src/cli.ts build examples/kernels.um --target host --emit-msl -o kernels   # kernels.<name>.metal
```

## Documentation

- [Setup](docs/setup.md) — installing Node.js and clang, the `um` command, running the tests
- [Language guide (Japanese)](docs/guide.ja.md) — syntax, ownership, kernels and buffers, GPU execution, threads, safety guarantees
- [GPU execution with wgpu (Japanese)](docs/wgpu.ja.md) — `UM_WGPU=1`, buffer transfers, loading kernels at run time with `gpu.loadKernel`, performance
- [Windows and input](docs/window.md) — opening a window, showing pixels, mouse and keyboard events (native, SDL3)
- [Running WebAssembly](docs/wasm.md) — loading and calling isolated wasm modules, plugins written in UM (native, wasmtime)
- [Video and audio (Japanese)](docs/media.ja.md) — file information, frame-accurate seeking, decoding (with hardware decoding), encoding to H.264 / AAC and more (native, FFmpeg)
- [examples/](examples) — runnable programs; each one doubles as a test (`// expect:` lines)
- [vscode-extension](https://github.com/UnifiedMozaic/vscode-extension) — VS Code extension (syntax highlighting, type checking on save, build and run)
- [SECURITY.md](SECURITY.md) — security policy and how to report a vulnerability

## Tests

```sh
node tests/run.ts
```

This runs every example on wasm32 and native and compares the output, checks programs that must fail to compile (`tests/errors/`) or must stop at run time (`tests/panics/`), and runs thread tests under ThreadSanitizer and memory tests under AddressSanitizer or valgrind when available. GPU tests (`tests/metal/`, `tests/wgpu/`) run on a Mac with Metal and wherever wgpu-native is found (`UM_WGPU_HOME`). wasm32 tests are skipped when no wasm32-capable `clang` is found.

## Project layout

```
src/        compiler: lexer → parser → checker → codegen (LLVM IR) → driver (clang)
            plus msl.ts (Metal) and wgsl.ts (wgpu and WebGPU) kernel translators, gpuload.ts (kernels loaded at run time)
runtime/    native runtime (C), wgpu runtime (C), Metal runtime (Objective-C), wasm runtime and JS host
examples/   sample programs (also tests)
tests/      test runner and test cases
assets/     logo
docs/       documentation
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this project, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
