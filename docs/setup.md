# Setup

How to get the UnifiedMozaic compiler running on your machine. You need only **Node.js** and **clang**; there are no npm packages to install.

[日本語](setup.ja.md)

## Supported platforms

| Platform | Native | GPU (wgpu) | GPU (Metal) | wasm32 |
| --- | --- | --- | --- | --- |
| macOS (Apple Silicon) | yes | not yet verified | yes | yes (needs Homebrew's LLVM and lld) |
| Linux (x86_64 / arm64) | yes | not yet verified | — | yes (needs `lld`) |
| Windows (x86_64) | yes (needs MSVC from Visual Studio or the Build Tools) | yes | — | yes |

GPU (wgpu) needs wgpu-native; see [docs/wgpu.ja.md](wgpu.ja.md) (Japanese): unzip a release from https://github.com/gfx-rs/wgpu-native/releases and set `UM_WGPU_HOME` to that folder.

## 1. Node.js (22.18 or later)

The compiler is TypeScript that Node.js runs directly (type stripping, enabled by default since 22.18).

```sh
# macOS
brew install node

# Linux (example with nvm; distribution packages are often too old)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
nvm install 22
```

```sh
node --version   # v22.18.0 or later
```

## 2. clang (LLVM 18 or later)

```sh
# macOS: the clang from Xcode or the Command Line Tools (also provides the Metal framework)
xcode-select --install

# Linux (Debian / Ubuntu): lld provides wasm-ld for wasm32 linking
sudo apt install clang lld

# Windows: the official LLVM build (clang, lld and wasm-ld)
winget install LLVM.LLVM
```

```sh
clang --version  # 18 or later
```

**wasm32 on macOS:** Xcode's clang cannot target wasm32. Install Homebrew's LLVM and lld; when they are present, wasm32 builds use them automatically (native and Metal builds keep using Xcode's clang).

```sh
brew install llvm lld
```

**Windows:** native programs are linked with the MSVC tools and the Windows SDK. Install Visual Studio (Community is fine) or the Build Tools for Visual Studio with the "Desktop development with C++" workload; clang finds them on its own (no developer command prompt needed). The LLVM installer does not add itself to `PATH` by default; when no clang is on `PATH`, `C:Program FilesLLVMinclang.exe` is used automatically.

```powershell
winget install Microsoft.VisualStudio.2022.BuildTools --override "--wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

To use a different clang, set `UM_CLANG` to its path (it is then used for every target).

## 3. Get the compiler

```sh
git clone https://github.com/UnifiedMozaic/UnifiedMozaic.git
cd UnifiedMozaic
```

## 4. The `um` command (optional)

Run `npm link` inside the repository to put `um` on your `PATH`. It is a link, so there is nothing to reinstall after `git pull`.

```sh
npm link
um --help
```

- If you get `um: command not found`, check that the `bin` directory under `npm prefix -g` is on your `PATH`.
- To remove it: `npm unlink -g unifiedmozaic`.
- Without `npm link`, replace `um` with `node src/cli.ts` below.
- Installing from the npm registry (`npm install -g`) is not supported yet.

## 5. Check that it works

```sh
# Native
um build examples/structs.um --target host -o structs
./structs                   # prints 32, 1.5, 2.5, 3.5, 7, 1

# Native (Windows): a name without an extension gets .exe
um build examples/structs.um --target host -o structs
.structs.exe

# wgpu (any OS): kernels can run on the GPU; UM_WGPU_HOME points to the unzipped wgpu-native release
UM_WGPU=1 um build examples/gpu_submit.um --target host -o gpu_submit

# Metal (Mac): kernels can run on the GPU
UM_METAL=1 um build examples/gpu_submit.um --target host -o gpu_submit
./gpu_submit                # prints 30, true, 285, 3

# wasm32 (with a wasm32-capable clang)
um build examples/structs.um -o structs.wasm
node runtime/run-node.mjs structs.wasm
```

Programs that open a window ([docs/window.md](window.md)) also need SDL3.
Programs that run WebAssembly modules ([docs/wasm.md](wasm.md)) also need the wasmtime C API.

## 6. Tests

```sh
node tests/run.ts           # same as npm test
```

This takes a few minutes. Anything your machine lacks is skipped with a note:

- no wasm32-capable clang (for example a Mac without Homebrew's LLVM) → wasm32 tests are skipped
- none of AddressSanitizer, valgrind or `leaks` (Mac) → memory-checked runs are skipped; on Linux, `sudo apt install valgrind` adds them
- no Metal → `tests/metal/` is only compiled for arm64
- Windows → AddressSanitizer cannot check for leaks there (use-after-free, out-of-bounds access and double frees are still checked), and there is no ThreadSanitizer, so the race checks of `tests/threads` are skipped

## 7. Editor

A VS Code extension lives at [UnifiedMozaic/vscode-extension](https://github.com/UnifiedMozaic/vscode-extension) (highlighting, type checking on save, build and run with F5). Install the `.vsix` from its [releases](https://github.com/UnifiedMozaic/vscode-extension/releases) and set `unifiedmozaic.compilerPath` to this repository.

```sh
code --install-extension unifiedmozaic-0.1.0.vsix
```

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `ERR_UNKNOWN_FILE_EXTENSION ".ts"`, or a `SyntaxError` at a type annotation | Node.js is too old; use 22.18 or later |
| `clang が見つかりません` / `the LLVM toolchain (clang) failed` | Install clang, or set `UM_CLANG` to its path |
| `この clang は wasm32 を出力できない` (this clang cannot target wasm32) | macOS: `brew install llvm lld`. Linux: install `lld` |
| Windows: `unable to find a Visual Studio installation`, or `link.exe` / `kernel32.lib` not found | Install Visual Studio or the Build Tools with "Desktop development with C++" (MSVC and the Windows SDK) |
| Windows: garbled non-ASCII text in the console | Native programs switch the console's output code page to UTF-8 while they run; if text is still garbled, check that the console font has those characters |
| `gpu.dispatch` is a compile error | The native GPU paths need `UM_WGPU=1` (any OS) or `UM_METAL=1` (Mac); otherwise use `dispatch` or `cpu.dispatch` |
| `wgpu-native was not found` | Unzip a wgpu-native release and set `UM_WGPU_HOME` to that folder |
| `panic: the GPU is not available` at run time | `gpu.dispatch` stops when there is no GPU; `dispatch` (automatic) falls back to the CPU |
