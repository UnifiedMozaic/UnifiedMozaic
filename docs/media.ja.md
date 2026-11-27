# 動画・音声（FFmpeg）

ネイティブのプログラムは、動画・音声のファイルを読み（情報、コマ単位で正確なシーク、順の展開、音の取り出し）、
書けます（H.264 / HEVC / AV1 と AAC などで MP4・MOV・MKV に）。中身は [FFmpeg](https://ffmpeg.org/) のライブラリ
（libavformat / libavcodec / libavutil / libswscale / libswresample）で、macOS・Windows・Linux で同じコードが動きます。
wasm32 では使えません（コンパイルエラー `not-implemented`）。

`import { mediaOpenVideo, mediaFrame } from "um:media";` のように、ライブラリ `um:media` から読み込んで使います（[ライブラリ](libraries.ja.md)）。

動画編集ソフトのように、素材を事前に別の形式へ展開せず、プログラムの中でその場で展開・シーク・圧縮するためのものです。
特定のアプリのためのものではなく、UM の汎用の機能です。

## 準備: FFmpeg のライブラリを入れる

FFmpeg 7.1 以降の**共有ライブラリ**のビルドが必要です（FFmpeg 9.0 で確かめています）。動画・音声の関数を使うプログラムを
ビルドしたときだけリンクします。

**Windows**: [BtbN/FFmpeg-Builds のリリース](https://github.com/BtbN/FFmpeg-Builds/releases) から
`ffmpeg-n9.0-latest-win64-lgpl-shared-9.0.zip` のような **lgpl-shared** のものを取って展開し、環境変数 `UM_FFMPEG_HOME` に
その場所を設定します。`gpl` の付いたビルド（と、gyan.dev の essentials / full のビルド）は GPL なので、配布するプログラムには
使わないでください（下の「ライセンス」）。

```powershell
$env:UM_FFMPEG_HOME = "C:\Users\you\ffmpeg-lgpl\ffmpeg-n9.0-latest-win64-lgpl-shared-9.0"
```

ビルドすると、`avformat-*.dll`・`avcodec-*.dll`・`avutil-*.dll`・`swscale-*.dll`・`swresample-*.dll` を実行ファイルの隣に
写します（DLL は実行ファイルと同じ場所にないと起動できないため）。配布するときは一緒に置いてください（合わせて約 120 MB。
大半は avcodec で、BtbN のビルドは多くの形式と外部の符号器を含むため。必要な形式だけで FFmpeg をビルドすれば小さくできます）。

**macOS / Linux**: `UM_FFMPEG_HOME`（`include/libavformat/avformat.h` と `lib/libavformat.dylib` / `.so` がある場所）、
なければ `pkg-config` の `libavformat` など、macOS ではさらに Homebrew の場所（`brew install ffmpeg`）を探します。
ライブラリの場所は実行ファイルに書き込みます（rpath）。Homebrew や Linux のディストリビューションの FFmpeg は、多くが
GPL の部品（libx264 など）を含むビルドです。手元で試すには問題ありませんが、配布するなら LGPL の構成で FFmpeg をビルドしてください。

探す順番は、`UM_FFMPEG_HOME` → `pkg-config` → Homebrew の決まった場所です。`mediaVersion()` が、実際に読み込んだライブラリの
版とライセンス（`LGPL version 3 or later` など）を返します。

## 例

```sh
node src/cli.ts build examples/media/info.um --target host -o info
./info clip.mp4
```

| 例 | 内容 |
| --- | --- |
| [examples/media/info.um](../examples/media/info.um) | ファイルの情報（長さ、大きさ、コマの速さ、画素の形式、色空間、回転、音） |
| [examples/media/frame_to_bmp.um](../examples/media/frame_to_bmp.um) | 指定の時刻のコマを1つ取り出して BMP に書く |
| [examples/media/count_frames.um](../examples/media/count_frames.um) | 最初から順に読んでコマを数える（ハードウェアの展開も選べる） |
| [examples/media/audio.um](../examples/media/audio.um) | 音を 48 kHz のステレオの f32 で読む。途中へのシーク |
| [examples/media/write_mp4.um](../examples/media/write_mp4.um) | 映像と音を作って MP4（H.264 + AAC）に書き、読み直して確かめる |
| [examples/media/gpu/grade.um](../examples/media/gpu/grade.um) | 展開したコマを GPU（`UM_WGPU=1`）の kernel で色補正し、MP4 に書く |
| [examples/media/bench/bench.um](../examples/media/bench/bench.um) | 展開とシークの速さを測る |

テスト（`node tests/run.ts`）は、素材を ffmpeg のコマンドで一時フォルダに作り（色にコマの番号を入れた H.264・B フレームつきの
MPEG-4・10 ビットの ProRes、途中で切れたファイル、動画でないファイル）、`examples/media` と `tests/media` を実行します。
FFmpeg のライブラリか ffmpeg のコマンドが見つからなければ飛ばします。

## 読む

```ts
import { mediaOpenVideo, mediaInfoOf } from "um:media";
match (mediaOpenVideo("clip.mp4", "auto")) {           // Result<i64, string>（読み手の番号）
  Result.Err(why) => { eprint(why); }
  Result.Ok(r) => {
    const info = mediaInfoOf(r);                          // MediaInfo
    let px = new Buffer<u32>(info.width * info.height);
    match (mediaFrame(r, 12.5, mut px, MediaPixels.Argb32, 0, 0)) {   // 12.5 秒に表示されるコマ
      MediaRead.Frame(t) => { ... }                       // t はそのコマの時刻（秒）
      MediaRead.End => { ... }                            // 最後のコマより後ろ
      MediaRead.Failed(why) => { eprint(why); }
    }
    while (true) {                                         // 続きを順に
      match (mediaNextFrame(r, mut px, MediaPixels.Argb32, 0, 0)) { ... }
    }
    mediaClose(r);
  }
}
```

| 関数 | 説明 |
| --- | --- |
| `mediaProbe(path: string): Result<MediaInfo, string>` | 情報だけを読む（復号器は開かない） |
| `mediaOpenVideo(path: string, hwaccel: string): Result<i64, string>` | 映像の読み手を開く。`hwaccel` は下の「ハードウェアの展開」 |
| `mediaOpenAudio(path: string): Result<i64, string>` | 音の読み手を開く |
| `mediaInfoOf(reader: i64): MediaInfo` | 読み手の情報（`hwaccel` は実際に使っている展開） |
| `mediaFrame<T>(reader, time: f64, out: mut Buffer<T>, pixels: MediaPixels, width: i32, height: i32): MediaRead` | `time` 秒に表示されるコマを `out` に書く。`width` / `height` が 0 なら素材の大きさ、違えば拡大縮小する |
| `mediaNextFrame<T>(reader, out: mut Buffer<T>, pixels, width, height): MediaRead` | 次のコマ（開いた直後は最初のコマ、`mediaFrame` のあとはその次） |
| `mediaFrameScaled<T>(reader, time, out, pixels, width, height, scale: MediaScale): MediaRead` | 拡大縮小のフィルタを選ぶ版（`time` が負なら次のコマ）。`Fast` / `Bilinear` / `Bicubic`（既定）/ `Lanczos` |
| `mediaSamples(reader, time: f64, out: mut Buffer<f32>, sampleRate: i32): MediaAudioRead` | `time` 秒からの音を、ステレオの f32 を交互に（L, R, L, R, ...）`out` の長さの半分の組まで書く |
| `mediaNextSamples(reader, out: mut Buffer<f32>, sampleRate: i32): MediaAudioRead` | 続きの音 |
| `mediaClose(reader: i64): void` | 閉じる |

```ts
enum MediaRead { Frame(time: f64), End, Failed(reason: string) }
enum MediaAudioRead { Samples(time: f64, count: i32), End, Failed(reason: string) }
```

- **時刻は秒（f64）で、最初のコマ（ストリームの最初の時刻）を 0 とします。** `mediaFrame` は「その時刻に表示されているコマ」、
  つまり時刻が `time` 以下で最も遅いコマを返します（25 コマ毎秒で `time = 0.999` なら 24 番目のコマ、時刻 0.96）。
  最初のコマより前の時刻なら最初のコマ、最後のコマの表示の終わりより後ろなら `End` です。コマ番号 `n` を読むには
  `n * fpsDen / fpsNum` を渡します（可変フレームレートの素材でも、時刻で指定すれば正しいコマが返ります）。
- **読み手は、映像か音のどちらかを読みます**（読まないストリームのパケットは、ファイルを読む段階で捨てます）。
  同じファイルの映像と音を読むときは、`mediaOpenVideo` と `mediaOpenAudio` で2つ開きます（ふつうは別のスレッドで読むので、
  そのほうが速い）。
- 同じ読み手のコマは、次の `mediaFrame` / `mediaNextFrame` まで復号した形で覚えておくので、同じ時刻をもう一度頼むと
  （大きさや形式を変えても）復号し直しません。

### 情報（MediaInfo）

| フィールド | 内容 |
| --- | --- |
| `duration`、`format`、`bitRate` | 全体の長さ（秒。分からなければ -1）、入れ物の名前（`mov,mp4,m4a,3gp,3g2,mj2`、`matroska,webm` など）、ビット毎秒 |
| `hasVideo`、`width`、`height` | 映像があるか、大きさ（画素） |
| `fpsNum` / `fpsDen` | コマの速さ（29.97 なら 30000 / 1001） |
| `frameCount` | コマの数（入れ物に書いてなければ長さ × コマの速さ。分からなければ -1） |
| `videoCodec`、`videoProfile` | `h264` / `High`、`hevc` / `Main 10`、`prores` / `HQ` など |
| `pixelFormat`、`bitDepth` | `yuv420p`、`yuv420p10le`、`yuv422p10le` など。成分のビット数 |
| `colorSpace`、`colorPrimaries`、`colorTransfer`、`colorRange` | 色の行列（`bt709`、`bt2020nc`）、色域（`bt709`、`bt2020`）、伝達特性（`bt709`、`smpte2084`（PQ）、`arib-std-b67`（HLG））、範囲（`tv` = 限られた範囲、`pc` = いっぱい）。書いていなければ `unknown` |
| `rotation` | 表示するときに時計回りに回す角度（0 / 90 / 180 / 270。スマートフォンの縦の動画）。コマは回さずに返すので、表示する側で回す |
| `sarNum` / `sarDen` | 画素の縦横比（ふつうは 1:1。アナモルフィックの素材で違う） |
| `videoDuration`、`videoBitRate` | 映像のストリームの長さとビット毎秒 |
| `hasAudio`、`sampleRate`、`channels`、`channelLayout`、`audioCodec`、`sampleFormat`、`audioDuration`、`audioBitRate` | 音 |
| `hwaccel` | 映像の読み手が使っているハードウェアの展開（`d3d11va`、`cuda` など。なければ `none`） |

### 画素の形式（MediaPixels）

`Buffer` の要素の型は形式で決まります。違う型や短い `Buffer` を渡すと `Failed`（理由つき）で、プログラムは止まりません。

| 形式 | Buffer | 中身 |
| --- | --- | --- |
| `Rgba8` | `Buffer<u8>`（幅 × 高さ × 4） | R, G, B, A を 1 バイトずつ |
| `Argb32` | `Buffer<u32>`（幅 × 高さ） | `0xAARRGGBB`。`windowPresent` にそのまま渡せる |
| `RgbaF32` | `Buffer<f32>`（幅 × 高さ × 4）か `Buffer<vec4<f32>>`（幅 × 高さ） | 0〜1。kernel にそのまま渡せる |
| `Rgba16` | `Buffer<u16>`（幅 × 高さ × 4） | 0〜65535。10 ビットの素材を丸めずに受け取る |
| `Yuv420p8` | `Buffer<u8>`（幅 × 高さ + 2 × ⌈幅/2⌉ × ⌈高さ/2⌉） | Y、U、V の面を順に。素材の色の行列と範囲のまま |
| `Yuv420p10` | `Buffer<u16>`（同じ長さ） | 同じ並びで 0〜1023 |

- **RGB の形式は、素材の色の行列（BT.601 / BT.709 / BT.2020）と範囲（限られた範囲 / いっぱい）で YUV から変換し、
  0〜255（0〜1）いっぱいの範囲にします。** 素材に書いていなければ、高さ 720 以上は BT.709、それ未満は BT.601、範囲は
  限られた範囲（`yuvj` の形式だけいっぱい）とみなします。色域と伝達特性（HDR の PQ / HLG を含む）は変換しません
  （値は素材の伝達特性のまま。色の管理は M4 で UM の kernel で行う前提です）。
- `RgbaF32` と `Rgba16` は、内部で 16 ビットの RGBA に変換してから作るので、10 / 12 ビットの素材の精度を保ちます。
- 拡大縮小と色の変換は libswscale で行います（CPU の複数のスレッドで 1 枚を分けて変換）。`MediaScale.Fast` はプレビュー向け、`Lanczos` は書き出し向けです。

### コマ単位で正確なシーク

`mediaFrame` は次のように、指定の時刻のコマを必ず返します（B フレームで表示の順と復号の順が違っても同じ）。

1. 直前に返したコマ（か、その次に復号済みのコマ）がその時刻を含むなら、それを返す（復号しない）。
2. 目標が今の位置より後ろで、間にキーフレームがない（入れ物の索引で分かる）か 0.5 秒以内なら、シークせずに読み進める。
   1 コマずつ後ろへ進むスクラブや再生は、順に読むのと同じ速さです。
3. そうでなければ、目標以前のキーフレームへシークし、目標のコマまで復号して途中のコマは捨てる。シークが目標より後ろに
   着いたら（索引が粗い入れ物）、1 秒、2 秒、4 秒、... と前から読み直す。

1 コマずつ前へ戻る（逆再生）のは、毎回キーフレームから復号し直すので遅くなります（下の「速さ」。キーフレームの間隔が長い
素材ほど遅い。編集ソフトでは、戻る向きのコマを先読みしてためておくのが定石です）。

### 音

- 出力は常に**ステレオの f32**（L, R, L, R, ...、-1〜1）で、標本化周波数は呼ぶ側が決めます（変換は libswresample）。
  素材が 1 チャンネルなら両方に、3 チャンネル以上なら 2 チャンネルにまとめます。
- `mediaSamples` のシークは標本単位です。素材と同じ標本化周波数なら、順に読んだものとずれは 0 標本、変換する（44.1 kHz の
  素材を 48 kHz で読むなど）ときは ±1 標本（約 21 マイクロ秒）以内です（tests/media/audio_seek.um）。
- 最初より前の時刻は、間を無音で埋めます（素材の音が映像より遅れて始まる場合に、時刻を揃えたまま読める）。
- 途中で標本化周波数を変えると、続きの時刻から読み直します。

### ハードウェアの展開

`mediaOpenVideo` の `hwaccel`:

| 値 | 動き |
| --- | --- |
| `"none"` | ソフトウェア（CPU の全コア。フレームとスライスのスレッド） |
| `"auto"` | 使えるものを順に試す。Windows: `d3d11va` → `cuda` → `dxva2`、macOS: `videotoolbox`、Linux: `vaapi` → `cuda` → `vdpau` |
| `"d3d11va"`、`"dxva2"`、`"d3d12va"`、`"cuda"`（NVDEC）、`"qsv"`、`"videotoolbox"`、`"vaapi"`、`"vulkan"` | その方式 |

- **使えなければ、黙ってソフトウェアに戻ります**（GPU がない、その形式・プロファイルに対応していない、など）。実際に
  使っているものは `mediaInfoOf(r).hwaccel` で分かります。知らない名前だけは失敗です。
- 同じ形式の復号器が複数あるときは、GPU の経路を持つものを選びます（AV1 では、既定の libdav1d には GPU の経路がないので、
  FFmpeg の内蔵の av1 復号器を使う）。
- 復号したコマは GPU のメモリから CPU に写してから変換します。このため、**小さい素材や軽い形式では、ソフトウェアのほうが
  速いことがあります**（下の「速さ」）。CPU の手が空くので、4K や HEVC / AV1 の 10 ビット、多数の素材を同時に読む場合に向きます。

## 書く

```ts
match (mediaCreate("out.mp4")) {                         // Result<i64, string>（書き手の番号）
  Result.Err(why) => { eprint(why); }
  Result.Ok(w) => {
    let v = new MediaVideoSettings();                      // 既定: "h264"、1920x1080、30 コマ毎秒
    v.width = 1280; v.height = 720;
    mediaAddVideoStream(w, v);                             // Result<string, string>（選ばれた符号器の名前）
    mediaAddAudioStream(w, new MediaAudioSettings());      // 既定: "aac"、48 kHz、ステレオ
    // コマと音を順に（交互に）書く
    mediaWriteFrame(w, px, MediaPixels.Argb32, 1280, 720); // string | null（null なら成功）
    mediaWriteSamples(w, samples, -1);                     // 交互の f32。-1 は samples 全体
    mediaFinish(w);                                        // string | null。書き手は閉じる
  }
}
```

| 関数 | 説明 |
| --- | --- |
| `mediaCreate(path: string): Result<i64, string>` | 入れ物は拡張子で決まる（`.mp4`、`.mov`、`.mkv`、`.webm`、`.wav` など） |
| `mediaAddVideoStream(writer, s: MediaVideoSettings): Result<string, string>` | 映像のストリームを足し、使う符号器の名前を返す。最初のコマを書く前に |
| `mediaAddAudioStream(writer, s: MediaAudioSettings): Result<string, string>` | 音のストリームを足す |
| `mediaWriteFrame<T>(writer, pixels: Buffer<T>, format: MediaPixels, width, height): string \| null` | コマを1つ書く（時刻は 0 から 1 コマずつ進む）。大きさが違えば拡大縮小する |
| `mediaWriteSamples(writer, samples: Buffer<f32>, count: i64): string \| null` | 音を書く（チャンネルを交互に並べた f32。`count` は標本の組の数、負なら全体） |
| `mediaFinish(writer): string \| null` | 残りを書き出してファイルを閉じる。成功しても失敗しても書き手は閉じる |
| `mediaEncoders(probe: bool): MediaEncoder[]` | 符号器の一覧。`probe` が真なら実際に開けるものだけ（数秒かかることがある） |

```ts
class MediaVideoSettings {
  encoder: string = "h264";   // 符号器の名前（"h264_nvenc"、"libopenh264" など）か形式（"h264"、"hevc"、"av1"）
  width: i32 = 1920;  height: i32 = 1080;
  fpsNum: i32 = 30;  fpsDen: i32 = 1;
  bitRate: i64 = 0;           // 0 なら大きさとコマの速さから（1080p30 で約 6 Mbps）
  options: string = "";       // 符号器の設定 "key=value:key=value"（"preset=p5:rc=vbr"、"pix_fmt=yuv420p10le" など）
}
class MediaAudioSettings { encoder: string = "aac"; sampleRate: i32 = 48000; channels: i32 = 2; bitRate: i64 = 0; options: string = ""; }
```

- **形式の名前（`"h264"`、`"hevc"`、`"av1"`）を渡すと、使える符号器を次の順に試します**: NVENC → AMF → QSV → VideoToolbox →
  VAAPI → MediaFoundation → D3D12 → Vulkan → ソフトウェア。GPU がない、ドライバが古いなどで開けないものは飛ばします。
  選ばれた符号器の名前が `mediaAddVideoStream` の結果です。
- RGB の形式のコマは、BT.709 の限られた範囲の YUV（ふつうは yuv420p）にして符号化し、ファイルにもそう印を付けます。
  YUV の形式のコマは、BT.709 の限られた範囲の値として扱います。
- キーフレームは 2 秒ごと、B フレームは 2 枚まで（符号器が対応していれば）。MP4 / MOV は索引をファイルの先頭に置きます
  （`faststart`。書き終わりに 1 回写し直す）。
- `options` の知らないキーは失敗です（打ち間違いを黙って無視しない）。`pix_fmt` は符号器に渡す画素の形式です。
- 映像と音は、時刻の順にだいたい交互に書いてください（片方を全部書いてからもう片方を書くと、入れ物に書く前にためる量が増えます）。
- 書き込みが一度失敗した書き手は、以後の書き込みも失敗します（ファイルは不完全）。`mediaFinish` で閉じてください。

## ライセンス（LGPL の範囲で使えるもの）

FFmpeg のライブラリは、構成によって LGPL か GPL になります。UM のプログラム（と MozaicCut）は、**LGPL のビルドを動的に
リンクする**前提です。LGPL の部品は、利用者が DLL を差し替えられる形（共有ライブラリ）で同梱すれば、プログラム自体の
ライセンスは自由に選べます（LGPL の条件: FFmpeg のライセンス文と、使った FFmpeg のソース（か入手方法）を添える）。
BtbN の lgpl のビルドは `--enable-version3` なので、**LGPL version 3 以降**です（`mediaVersion()` で確かめられます）。
GPL の部品（`--enable-gpl` が要るもの）は使いません。

| 用途 | LGPL のビルドで使えるもの | 使わないもの（GPL） |
| --- | --- | --- |
| H.264 の圧縮 | `h264_nvenc`（NVIDIA）、`h264_amf`（AMD）、`h264_qsv`（Intel）、`h264_videotoolbox`（macOS）、`h264_mf`（Windows の MediaFoundation）、`h264_vaapi`（Linux）、`libopenh264`（Cisco の OpenH264。ソフトウェア） | `libx264` |
| HEVC の圧縮 | `hevc_nvenc`、`hevc_amf`、`hevc_qsv`、`hevc_videotoolbox`、`hevc_mf`、`libkvazaar`（ソフトウェア） | `libx265` |
| AV1 の圧縮 | `av1_nvenc`（RTX 40 以降）、`av1_amf`、`av1_qsv`、`libsvtav1`、`libaom-av1`、`librav1e` | — |
| 中間形式・可逆 | `prores_ks`（ProRes）、`dnxhd`（DNxHD / HR）、`ffv1`、`mjpeg`、`png` | — |
| 音の圧縮 | `aac`（FFmpeg 内蔵）、`libopus`、`flac`、`alac`、`libmp3lame`、`pcm_s16le` / `pcm_s24le` / `pcm_f32le` | `libfdk_aac`（非自由。配布できない） |
| 展開 | ほぼすべて（H.264 / HEVC / AV1 / VP9 / ProRes / DNxHD など内蔵の復号器、libdav1d、ハードウェアの展開） | — |

注意:

- **H.264 / HEVC / AAC の特許は、ライセンスとは別の話です。** ハードウェアの符号器（NVENC など）と OS の MediaFoundation は、
  特許の扱いを GPU・OS の側が持ちます。OpenH264 は、Cisco が配る**バイナリ**を使った場合だけ Cisco が特許料を負担します
  （BtbN のビルドはソースからビルドして組み込んでいるので、この扱いにならない）。ソフトウェアの H.264 で配布するなら、
  Cisco のバイナリを読み込む構成にするか、ハードウェアの符号器を使うのが無難です。HEVC のソフトウェアの圧縮も同様です。
- `h264_mf` は Windows に入っている MediaFoundation の符号器です（GPU があればそのハードウェアの符号器、なければ Microsoft の
  ソフトウェアの符号器）。どの Windows でも使えますが、画質の設定の幅は狭いです。
- AV1（AOMedia の特許のライセンスはロイヤリティなし）と ProRes / DNxHD（編集の中間形式）は、特許の面でも扱いやすい選択肢です。

`mediaEncoders(true)` で、その機械で実際に使える符号器が分かります。この機械（Windows 11、NVIDIA の GPU）では、H.264 は
`h264_nvenc`・`h264_mf`・`libopenh264` が使え、`h264_amf`（AMD 用）と `h264_qsv`（Intel 用）は開けません（理由つきの失敗）。

## スレッド

- **どの関数も、どのスレッドからでも呼べます。** 読み手・書き手は、それぞれ別の FFmpeg の状態（ファイル、復号器、変換）を
  持つので、**別々の読み手をスレッドごとに同時に使えます**（先読みのスレッドで展開し、メインスレッドで表示する、など）。
  番号は i64 なので、`thread.spawn` にそのまま渡せます。
- 同じ読み手・書き手を 2 つのスレッドが同時に使おうとしたら、後から来たほうが失敗します（`the reader is being used by another
  thread (use one per thread)`）。止まりはしませんが、読み手はスレッドごとに開いてください。
- ソフトウェアの展開は、1 つの読み手が最大 8 本のスレッドを使います（FFmpeg のフレームとスライスのスレッド。論理 CPU が
  8 より少なければその数）。画素の変換（libswscale）は 1 つの読み手につき最大 4 本です。FFmpeg の既定（論理 CPU の数）の
  ままだと、読み手をいくつも同時に開いたときにスレッドが多くなりすぎ、待ち合わせの手間で CPU を食うためです。
  数は環境変数で変えられます: `UM_MEDIA_THREADS`（展開。0 なら論理 CPU の数）、`UM_SWS_THREADS`（変換。0 なら論理 CPU の数）。
  読み手を多数同時に使うときは、ハードウェアの展開のほうが CPU を空けられます。

例: [tests/media/threads.um](../tests/media/threads.um)。

## 失敗と停止

- **開けない・壊れた・読めないファイル、合わない引数（Buffer の型や長さ、知らない符号器や設定）、使えない符号器では止まらず、
  理由を返します**（`Result.Err`、`MediaRead.Failed`、`string | null`）。理由は英語の1行で、FFmpeg が出したエラーの行を添えます
  （`broken.mp4: Invalid data found when processing input (mov,mp4,m4a,3gp,3g2,mj2: moov atom not found)` など）。
- 途中が壊れたファイルは、壊れたコマを飛ばして読み続けます（続けて 64 回失敗したら `Failed`）。
- FFmpeg のログは標準エラー出力に出しません。`UM_MEDIA_LOG=1` を付けて実行すると、すべて出します。
- **閉じた番号や種類の違う番号（読み手に書き手の番号など）を渡すと停止します**（`panic: invalid media call (an unknown or closed
  reader / writer)`）。使っている最中の読み手を別のスレッドが閉じても停止します。これらはプログラムの誤りです。

## 速さ

[examples/media/bench/bench.um](../examples/media/bench/bench.um) で測りました。Windows 11、Ryzen 9 3900X（12 コア 24 スレッド）、
NVIDIA GeForce RTX 5060 Ti、FFmpeg 9.0（BtbN の lgpl-shared）。素材は 1920x1080、30 コマ毎秒、20 秒（600 コマ）、
20 Mbps、キーフレームは 2 秒ごと、B フレーム 2 枚（NVENC で作成）。ほかの作業と同時に測ったので、数字は 2〜3 割揺れます
（何回か測った範囲）。

**順に読む（コマ毎秒）**

| 素材 | 展開 | Yuv420p8 | Argb32 | RgbaF32 | Argb32 960x540 |
| --- | --- | --- | --- | --- | --- |
| H.264 High 8 ビット | ソフトウェア | 290〜470 | 205〜250 | 60〜90 | 315〜395 |
| | d3d11va | 190〜300 | 110 | 35 | 185 |
| | cuda（NVDEC） | 345〜440 | 145 | 55 | 120 |
| HEVC Main 10 | ソフトウェア | 125〜150 | 140 | 85 | 160 |
| | d3d11va | 75〜165 | 115 | 70 | 170 |
| | cuda（NVDEC） | 80〜265 | 165 | 85 | 235 |

**シーク（ミリ秒、Argb32 の素材の大きさ）**

| 素材 | 展開 | 飛び飛びの時刻 | 1 コマずつ後ろへ | 1 コマずつ前へ |
| --- | --- | --- | --- | --- |
| H.264 | ソフトウェア | 125〜190 | 6.5〜9 | 105〜185 |
| | d3d11va | 110 | 9 | 105 |
| | cuda | 185 | 17 | 190 |
| HEVC Main 10 | ソフトウェア | 315 | 14 | 300 |
| | d3d11va | 140 | 11 | 130 |
| | cuda | 90 | 8 | 85 |

読み方:

- **1080p の再生（30〜60 コマ毎秒）とスクラブ（1 コマずつ後ろへ: 7〜17 ms）は、どの組み合わせでも足ります。**
  飛び飛びのシークは、キーフレームから目標まで平均 1 秒分を復号するので 0.1〜0.3 秒かかります。
- 1080p を RGB にする変換（libswscale）は、展開と同じくらい重い処理です。FFmpeg 9 の新しい API（`sws_scale_frame`）で
  1 枚を複数のスレッドで変換しています（1 スレッドの古い API では Argb32 が 42 コマ毎秒だった）。表は変換のスレッドを
  論理 CPU の数（24）にしていたときの数字で、今の既定（4 本）では 1080p の Argb32 は約 130 コマ毎秒、960x540 は約 350 コマ毎秒です
  （展開のスレッドを 8 本にしたことによる差は、Yuv420p8 で 1 割以内）。`UM_SWS_THREADS=0` で表の速さに戻ります。プレビューの大きさ
  （960x540）で受け取ると速くなります。
- `RgbaF32` は 16 ビットの RGBA を経由して f32 に直す分（1080p で 1 コマ 32 MB）遅くなります。kernel で色を扱うときに使い、
  表示だけなら `Argb32` を使ってください。
- **ハードウェアの展開は、この機械の 1080p では速さの得になりません**（コマを GPU から CPU に写す手間があるため）。
  10 ビットの HEVC のように CPU の展開が重い素材では、CPU を空けたままソフトウェアと同じくらいの速さが出ます。
  4K や、多数の素材を同時に読む場合（CPU のコアを取り合う）に `auto` を使うのがよいです。
- 展開 → GPU の kernel（`UM_WGPU=1`）→ 符号化（NVENC）の流れ（[examples/media/gpu/grade.um](../examples/media/gpu/grade.um)、
  RgbaF32 で受け取り、露出・彩度・周辺減光をかけて Argb32 で H.264 に書く）は、1080p で約 55 コマ毎秒でした
  （1 コマあたり、展開と変換 11.5 ms、kernel 4.1 ms、符号化 2.3 ms）。

## 組み込み関数（低水準）

`lib/media.um` の関数は、次の組み込み関数（`runtime/media.c`）を使って書いてあります。ふつうは上の関数を使ってください。

| 関数 | 説明 |
| --- | --- |
| `mediaOpen(path, kind: i32, hwaccel): i64 \| null` | kind: 0 情報だけ、1 映像、2 音 |
| `mediaClose(reader)` | |
| `mediaInfoNumbers(reader): f64[]` / `mediaInfoText(reader): string[]` | 情報の数と文字列（並びは runtime/media.c） |
| `mediaReadVideo(reader, time, out: mut 数の Buffer か配列, format: i32, width, height, filter: i32): f64` | コマの時刻。終わりは -1、失敗は -2 |
| `mediaReadAudio(reader, time, out: mut f32 の Buffer か配列, sampleRate): i64` | 標本の組の数。終わりは 0、失敗は -1 |
| `mediaLastTime(reader): f64` | 最後に読んだコマ・音の先頭の時刻 |
| `mediaCreateWriter(path): i64 \| null`、`mediaAddVideo(...)`、`mediaAddAudio(...)`、`mediaWriteVideo(...)`、`mediaWriteAudio(...)`、`mediaFinishWriter(writer): bool`、`mediaWriterText(writer): string[]` | 書く |
| `mediaEncoderList(probe: bool): string[]` | 5 つずつ: 名前、`video` / `audio`、形式、説明、`hw` / `sw` |
| `mediaError(): string` | このスレッドで最後に失敗した理由 |
| `mediaVersion(): string` | FFmpeg のライブラリの版とライセンス |

## まだないもの

- 色域と伝達特性の変換（HDR → SDR のトーンマッピング、色の管理）。値は素材のまま返す
- コマの回転（`rotation` は情報として返すだけ）、インターレースの解除
- 字幕・データのストリーム、複数の映像・音のストリームの選択（最も適したものを 1 つずつ選ぶ）
- GPU のメモリのまま kernel に渡す経路（いまはハードウェアの展開でも CPU のメモリに写す）
- 書き出しのハードウェアの面への直接の転送、2 パスの符号化
- 読み手のネットワーク（http など）の扱いの設定（FFmpeg が対応する URL は開けるが、時間切れなどは設定できない）
