# 環境構築

UnifiedMozaic のコンパイラを手元で使えるようにする手順です。必要なのは **Node.js** と **clang** だけで、npm パッケージのインストールはありません。

[English](setup.md)

## 対応している環境

| 環境 | ネイティブ | GPU（wgpu） | GPU（Metal） | wasm32 |
| --- | --- | --- | --- | --- |
| macOS（Apple Silicon） | ○ | △（未確認） | ○ | ○（Homebrew の LLVM と lld が要る） |
| Linux（x86_64 / arm64） | ○ | △（未確認） | — | ○（`lld` が要る） |
| Windows（x86_64） | ○（Visual Studio か Build Tools の MSVC が要る） | ○ | — | ○ |

GPU（wgpu）には wgpu-native が要ります（[wgpu での GPU 実行](wgpu.ja.md)の「準備」）。

## 1. Node.js（22.18 以上）

コンパイラは TypeScript のソースを Node.js がそのまま実行します（型を消すだけの実行。22.18 から既定で使えます）。

```sh
# macOS
brew install node

# Linux（nvm を使う例。ディストリビューションの nodejs は古いことが多い）
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
nvm install 22
```

```sh
node --version   # v22.18.0 以上
```

## 2. clang（LLVM 18 以上）

```sh
# macOS: Xcode か Command Line Tools の clang を使う（Metal のフレームワークも入る）
xcode-select --install

# Linux（Debian / Ubuntu）: lld は wasm32 のリンクに使う（wasm-ld）
sudo apt install clang lld

# Windows: LLVM の公式のビルド（clang・lld・wasm-ld が入る）
winget install LLVM.LLVM
```

```sh
clang --version  # 18 以上
```

**macOS で wasm32 も出したいとき**: Xcode の clang は wasm32 を出せないので、Homebrew の LLVM と lld を入れます。入っていれば、wasm32 のビルドには自動でそちらを使います（ネイティブと Metal は Xcode の clang のまま）。

```sh
brew install llvm lld
```

**Windows**: ネイティブのリンクには、MSVC のツールと Windows SDK を使います。Visual Studio（Community でよい）か Build Tools for Visual Studio を、「C++ によるデスクトップ開発」を選んで入れてください。clang は自動でそれを見つけます（開発者用のコマンドプロンプトでなくてよい）。LLVM のインストーラは既定では `PATH` に足しませんが、`PATH` に clang がなければ `C:Program FilesLLVMinclang.exe` を自動で使います。

```powershell
winget install Microsoft.VisualStudio.2022.BuildTools --override "--wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

別の clang を使いたいときは、環境変数 `UM_CLANG` に場所を書きます（すべてのターゲットでその clang を使う）。

## 3. コンパイラを取ってくる

```sh
git clone https://github.com/UnifiedMozaic/UnifiedMozaic.git
cd UnifiedMozaic
```

## 4. `um` コマンドを使えるようにする（任意）

リポジトリの中で `npm link` を実行すると、`um` コマンドがどこからでも使えるようになります。リンクなので、`git pull` したあとに入れ直す必要はありません。

```sh
npm link
um --help
```

- `um: command not found` のときは、`npm prefix -g` の下の `bin` が `PATH` に入っているか確かめてください。
- やめるときは `npm unlink -g unifiedmozaic`。
- `npm link` を使わない場合は、以下の `um` を `node src/cli.ts` に読み替えてください。
- npm のレジストリからのインストール（`npm install -g`）にはまだ対応していません。

## 5. 動くことを確かめる

```sh
# ネイティブ
um build examples/structs.um --target host -o structs
./structs                   # 32、1.5、2.5、3.5、7、1 と出る

# ネイティブ（Windows）。拡張子のない名前には .exe が付く
um build examples/structs.um --target host -o structs
.structs.exe

# wgpu（どの OS でも）。kernel を GPU でも実行する。UM_WGPU_HOME に wgpu-native を展開した場所
UM_WGPU=1 um build examples/gpu_submit.um --target host -o gpu_submit

# Metal（Mac）。kernel を GPU でも実行する
UM_METAL=1 um build examples/gpu_submit.um --target host -o gpu_submit
./gpu_submit                # 30、true、285、3 と出る

# wasm32（wasm32 を出せる clang があるとき）
um build examples/structs.um -o structs.wasm
node runtime/run-node.mjs structs.wasm
```

ウィンドウを開くプログラム（[docs/window.ja.md](window.ja.md)）には、ほかに SDL3 が要ります。
WebAssembly のモジュールを実行するプログラム（[docs/wasm.ja.md](wasm.ja.md)）には、ほかに wasmtime の C API が要ります。

## 6. テスト

```sh
node tests/run.ts           # npm test でも同じ
```

数分かかります。環境にないものは、理由を表示して飛ばします。

- wasm32 を出せる clang がない（Homebrew の LLVM のない Mac など）→ wasm32 のテストを飛ばす。
- AddressSanitizer・valgrind・`leaks`（Mac）のどれもない → メモリ検査つきの実行を飛ばす。Linux では `sudo apt install valgrind` で足せます。
- Metal がない → `tests/metal/` は arm64 向けに作れることだけを確かめる。
- Windows → AddressSanitizer で解放漏れは調べられない（解放後使用・範囲外アクセス・二重解放だけを調べる）。ThreadSanitizer はないので、`tests/threads` の競合検査を飛ばす。

## 7. エディタ

VSCode 向けの拡張が [UnifiedMozaic/vscode-extension](https://github.com/UnifiedMozaic/vscode-extension) にあります（色分け、保存時の型検査、F5 でビルドして実行）。[リリース](https://github.com/UnifiedMozaic/vscode-extension/releases) の `.vsix` を入れ、設定の `unifiedmozaic.compilerPath` にこのリポジトリの場所を書きます。

```sh
code --install-extension unifiedmozaic-0.1.0.vsix
```

## うまくいかないとき

| 症状 | 対処 |
| --- | --- |
| `ERR_UNKNOWN_FILE_EXTENSION ".ts"` や、型の注釈での `SyntaxError` | Node.js が古い。22.18 以上にする |
| `clang が見つかりません` / `the LLVM toolchain (clang) failed` | clang を入れるか、`UM_CLANG` で場所を指定する |
| `この clang は wasm32 を出力できない` | macOS は `brew install llvm lld`。Linux は `lld` を入れる |
| Windows で `unable to find a Visual Studio installation`、`link.exe` や `kernel32.lib` が見つからない | Visual Studio か Build Tools に「C++ によるデスクトップ開発」（MSVC と Windows SDK）を入れる |
| Windows のコンソールで日本語が化ける | ネイティブのプログラムは、コンソールに出す間だけ出力の文字コードを UTF-8 にする。化けるときは、コンソールのフォントがその文字を持っているか確かめる |
| `gpu.dispatch` がコンパイルエラーになる | ネイティブの GPU の経路は `UM_WGPU=1`（どの OS でも。[wgpu での GPU 実行](wgpu.ja.md)）か `UM_METAL=1`（Mac）を付けたビルドだけ。付けないときは `dispatch` か `cpu.dispatch` を使う |
| `wgpu-native was not found` | wgpu-native のリリースを展開し、`UM_WGPU_HOME` にその場所を入れる（[wgpu での GPU 実行](wgpu.ja.md)の「準備」） |
| 実行時に `panic: the GPU is not available` | `gpu.dispatch` は GPU がないと停止する。`dispatch`（自動）なら CPU で動く |
