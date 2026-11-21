#!/usr/bin/env bash
# CI で使うライブラリを $UM_DEPS（actions/cache で残す場所）に揃え、テストが見る環境変数を $GITHUB_ENV に書く。
#   wasmtime の C API  → UM_WASMTIME
#   wgpu-native        → UM_WGPU_HOME
#   SDL3               → UM_SDL3
#   FFmpeg（LGPL の共有ライブラリ）→ UM_FFMPEG_HOME
#   LLVM（Windows だけ）→ UM_CLANG と PATH
# どれも公式の配布物（GitHub の releases）から取る。配布物のない組み合わせ（macOS と Linux の SDL3、macOS の FFmpeg）は、
# 同じ版のソース（GitHub の releases / タグ）からビルドする。すでに $UM_DEPS にあるもの（キャッシュから戻したもの）は取り直さない。
# 版を変えたら、このファイルが変わるのでキャッシュの鍵も変わる（ci.yml の hashFiles）。
set -euo pipefail

WASMTIME_VERSION=51.0.0
WGPU_VERSION=29.0.1.1
SDL3_VERSION=3.4.20
FFMPEG_VERSION=9.0
LLVM_VERSION=23.1.6

DEPS="${UM_DEPS:?UM_DEPS is not set}"
mkdir -p "$DEPS"
JOBS=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)

case "$(uname -s)" in
  MINGW* | MSYS* | CYGWIN*) os=windows ;;
  Darwin) os=macos ;;
  Linux) os=linux ;;
  *) echo "unsupported OS: $(uname -s)" >&2; exit 1 ;;
esac

# 取ってきて展開する。$1 = URL、$2 = 展開する場所。
fetch() {
  local url=$1 dest=$2 tmp
  tmp=$(mktemp -d)
  echo "download: $url"
  curl -fsSL --retry 3 -o "$tmp/archive" "$url"
  mkdir -p "$dest"
  case "$url" in
    *.zip) unzip -q "$tmp/archive" -d "$dest" ;;
    *.tar.xz) tar -xJf "$tmp/archive" -C "$dest" ;;
    *.tar.gz) tar -xzf "$tmp/archive" -C "$dest" ;;
  esac
  rm -rf "$tmp"
}

# Windows ではテスト（Node）に渡すパスを Windows の形にする。
native_path() {
  if [ "$os" = windows ]; then cygpath -w "$1"; else printf '%s' "$1"; fi
}

setenv() {
  echo "$1=$(native_path "$2")" >> "${GITHUB_ENV:-/dev/null}"
  echo "$1=$(native_path "$2")"
}

# macOS: 配布物の dylib の名前（install name）が @rpath で始まらないと、リンクした実行ファイルが起動時に見つけられない
# （UM はライブラリの場所を rpath で渡す）。そうなっていなければ @rpath/<名前> に直して、署名し直す。
rpath_install_name() {
  local lib=$1 id
  id=$(otool -D "$lib" | tail -n 1)
  echo "$(basename "$lib") install name: $id"
  case "$id" in
    @rpath/*) ;;
    *) install_name_tool -id "@rpath/$(basename "$lib")" "$lib"
       codesign --force --sign - "$lib" ;;
  esac
}

# --- LLVM（Windows だけ） ---
# runner に入っている LLVM は、runner の Visual Studio の C++ 標準ライブラリより古いことがある（LLVM 20 に対して
# 「Clang 22 or newer」を求められ、tests/msl の C++ の検査が通らない）。手元と同じ版の公式の配布物を使う。
# 配布物は大きいので、使うもの（clang・clang++・lld・llvm-rc・llvm-readobj・DLL・clang のリソース）だけ展開する。
if [ "$os" = windows ]; then
  llvm_name=clang+llvm-$LLVM_VERSION-x86_64-pc-windows-msvc
  llvm_dir="$DEPS/llvm-$LLVM_VERSION"
  if [ ! -f "$llvm_dir/bin/clang.exe" ]; then
    tmp=$(mktemp -d)
    echo "download: LLVM $LLVM_VERSION"
    curl -fsSL --retry 3 -o "$tmp/llvm.tar.xz" "https://github.com/llvm/llvm-project/releases/download/llvmorg-$LLVM_VERSION/$llvm_name.tar.xz"
    mkdir -p "$tmp/x"
    # .tar.zst のほうが小さいが、長い窓の zstd なので Windows の tar（bsdtar）では展開できない。
    xz -T0 -dc "$tmp/llvm.tar.xz" | tar -x --wildcards -C "$tmp/x" \
      "$llvm_name/bin/clang.exe" "$llvm_name/bin/clang++.exe" "$llvm_name/bin/lld-link.exe" "$llvm_name/bin/wasm-ld.exe" \
      "$llvm_name/bin/llvm-rc.exe" "$llvm_name/bin/llvm-readobj.exe" "$llvm_name/bin/*.dll" "$llvm_name/lib/clang"
    rm -rf "$llvm_dir"
    mv "$tmp/x/$llvm_name" "$llvm_dir"
    rm -rf "$tmp"
  fi
  setenv UM_CLANG "$llvm_dir/bin/clang.exe"
  native_path "$llvm_dir/bin" >> "${GITHUB_PATH:-/dev/null}"
  "$llvm_dir/bin/clang.exe" --version
fi

# --- wasmtime の C API ---
case "$os" in
  windows) wt_name=wasmtime-v$WASMTIME_VERSION-x86_64-windows-c-api; wt_ext=zip ;;
  macos) wt_name=wasmtime-v$WASMTIME_VERSION-aarch64-macos-c-api; wt_ext=tar.xz ;;
  linux) wt_name=wasmtime-v$WASMTIME_VERSION-x86_64-linux-c-api; wt_ext=tar.xz ;;
esac
if [ ! -f "$DEPS/$wt_name/include/wasmtime.h" ]; then
  fetch "https://github.com/bytecodealliance/wasmtime/releases/download/v$WASMTIME_VERSION/$wt_name.$wt_ext" "$DEPS"
fi
[ "$os" = macos ] && rpath_install_name "$DEPS/$wt_name/lib/libwasmtime.dylib"
setenv UM_WASMTIME "$DEPS/$wt_name"

# --- wgpu-native ---
case "$os" in
  windows) wgpu_zip=wgpu-windows-x86_64-msvc-release.zip ;;
  macos) wgpu_zip=wgpu-macos-aarch64-release.zip ;;
  linux) wgpu_zip=wgpu-linux-x86_64-release.zip ;;
esac
wgpu_dir="$DEPS/wgpu-native-v$WGPU_VERSION"
if [ ! -f "$wgpu_dir/include/webgpu/webgpu.h" ]; then
  fetch "https://github.com/gfx-rs/wgpu-native/releases/download/v$WGPU_VERSION/$wgpu_zip" "$wgpu_dir"
fi
[ "$os" = macos ] && rpath_install_name "$wgpu_dir/lib/libwgpu_native.dylib"
setenv UM_WGPU_HOME "$wgpu_dir"

# --- SDL3 ---
sdl_dir="$DEPS/SDL3-$SDL3_VERSION"
if [ "$os" = windows ]; then
  if [ ! -f "$sdl_dir/include/SDL3/SDL.h" ]; then
    fetch "https://github.com/libsdl-org/SDL/releases/download/release-$SDL3_VERSION/SDL3-devel-$SDL3_VERSION-VC.zip" "$DEPS"
  fi
else
  # macOS と Linux には開発用の配布物がないので、同じ版のソースからビルドする。
  if [ ! -f "$sdl_dir/include/SDL3/SDL.h" ]; then
    src=$(mktemp -d)
    fetch "https://github.com/libsdl-org/SDL/releases/download/release-$SDL3_VERSION/SDL3-$SDL3_VERSION.tar.gz" "$src"
    extra=()
    # Linux の runner には画面がなく、テストは offscreen と dummy の駆動を使うので、X11 / Wayland なしでビルドする。
    if [ "$os" = linux ]; then extra+=(-DSDL_UNIX_CONSOLE_BUILD=ON); fi
    cmake -S "$src/SDL3-$SDL3_VERSION" -B "$src/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$sdl_dir" \
      -DCMAKE_INSTALL_LIBDIR=lib -DSDL_SHARED=ON -DSDL_STATIC=OFF -DSDL_TESTS=OFF -DSDL_EXAMPLES=OFF ${extra[@]+"${extra[@]}"}
    cmake --build "$src/build" --parallel "$JOBS"
    cmake --install "$src/build"
    rm -rf "$src"
  fi
fi
setenv UM_SDL3 "$sdl_dir"

# --- FFmpeg（LGPL の共有ライブラリ） ---
case "$os" in
  windows | linux)
    if [ "$os" = windows ]; then ff_name=ffmpeg-n$FFMPEG_VERSION-latest-win64-lgpl-shared-$FFMPEG_VERSION; ff_ext=zip
    else ff_name=ffmpeg-n$FFMPEG_VERSION-latest-linux64-lgpl-shared-$FFMPEG_VERSION; ff_ext=tar.xz; fi
    if [ ! -f "$DEPS/$ff_name/include/libavformat/avformat.h" ]; then
      fetch "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/$ff_name.$ff_ext" "$DEPS"
    fi
    setenv UM_FFMPEG_HOME "$DEPS/$ff_name"
    ;;
  macos)
    # macOS の LGPL の共有ライブラリの配布物はないので、同じ版のタグのソースから LGPL の構成でビルドする
    # （GPL の部品は入れない。H.264 の符号化は OS の VideoToolbox を使う）。
    ff_dir="$DEPS/ffmpeg-n$FFMPEG_VERSION-lgpl-shared"
    if [ ! -f "$ff_dir/include/libavformat/avformat.h" ]; then
      src=$(mktemp -d)
      fetch "https://github.com/FFmpeg/FFmpeg/archive/refs/tags/n$FFMPEG_VERSION.tar.gz" "$src"
      (
        cd "$src/FFmpeg-n$FFMPEG_VERSION"
        ./configure --prefix="$ff_dir" --enable-shared --disable-static --enable-version3 \
          --disable-autodetect --enable-videotoolbox --enable-audiotoolbox --enable-zlib \
          --disable-doc --disable-ffplay --disable-ffprobe
        make -j"$JOBS"
        make install
      )
      rm -rf "$src"
    fi
    setenv UM_FFMPEG_HOME "$ff_dir"
    ;;
esac

echo "deps ready in $DEPS"
