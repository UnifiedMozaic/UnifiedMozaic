// 動画・音声の組み込み（mediaOpen / mediaReadVideo / mediaReadAudio / mediaCreate / mediaWriteVideo など）の実装。
// FFmpeg のライブラリ（libavformat / libavcodec / libavutil / libswscale / libswresample）を使う。
// プログラムが動画・音声の組み込みを使ったときだけ、ドライバが native.c と一緒にリンクする。
// FFmpeg は LGPL のビルドを共有ライブラリとして動的にリンクする前提（GPL の部分は使わない。docs/media.ja.md）。
//
// 決まりごと:
// - 読み手（reader）と書き手（writer）は、64 ビットの番号（ハンドル）で表す。種類、表の場所、世代を詰めてあるので、
//   閉じた番号や種類の違う番号を渡すと停止する（20）。
// - どの関数もどのスレッドからでも呼べる。読み手・書き手は、それぞれ別の FFmpeg の状態（demuxer、復号器、変換）を持つので、
//   別々の読み手をスレッドごとに同時に使える。同じ読み手を2つのスレッドが同時に使おうとしたら、後から来たほうを
//   失敗にする（理由は mediaError）。使っている最中の読み手・書き手を閉じようとしたら停止（21）。
// - 失敗（開けない、壊れたデータ、使えない符号器、Buffer の型や長さが合わない、など）では停止せず、失敗を返す。
//   理由の文字列はスレッドごとに覚えておき、mediaError() で読む（runtime/media.um がこれを Result や enum にまとめる）。
// - 時刻は秒（f64）。ストリームの最初の時刻を 0 とする。
// - 読み手は、開くときに映像か音のどちらを読むかを決める（もう片方のパケットは demuxer の段階で捨てる）。
//   同じファイルの映像と音を読むときは、読み手を2つ開く（ふつうは別のスレッドで読むので、そのほうが速い）。
// - FFmpeg のログは標準エラー出力に出さず、最後のエラーの行を失敗の理由に使う（UM_MEDIA_LOG=1 で出す）。

#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <math.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include "win32.h"
#else
#include <pthread.h>
#endif
#if !defined(_WIN32) && !defined(__APPLE__)
#include <dlfcn.h>
#endif

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/audio_fifo.h>
#include <libavutil/channel_layout.h>
#include <libavutil/display.h>
#include <libavutil/hwcontext.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>

_Noreturn void um_panic(int32_t code, int32_t line);
void *um_alloc(size_t count, size_t elem_size);

// um_panic の原因コード（native.c の表と src/codegen.ts の並びと揃えること）。
#define UM_PANIC_MEDIA_HANDLE 20  // 番号が正しくない（閉じた、種類違い）
#define UM_PANIC_MEDIA_BUSY 21    // ほかのスレッドが使っている最中の読み手・書き手を閉じようとした

#define KIND_READER 1
#define KIND_WRITER 2
#define MAX_SLOTS 1024  // 読み手と書き手の、それぞれ同時に持てる数

// 読み手が読むもの（runtime/media.um と揃える）。
#define READ_INFO 0   // 情報だけ（復号器を開かない）
#define READ_VIDEO 1
#define READ_AUDIO 2

// 画素の形式（runtime/media.um の MediaPixels と揃える）。
#define PX_RGBA8 0      // Buffer<u8>: R, G, B, A の順に 1 バイトずつ
#define PX_ARGB32 1     // Buffer<u32>: 0xAARRGGBB（windowPresent にそのまま渡せる）
#define PX_RGBAF32 2    // Buffer<f32> か Buffer<vec4<f32>>: R, G, B, A を 0〜1 で
#define PX_RGBA16 3     // Buffer<u16>: R, G, B, A を 0〜65535 で
#define PX_YUV420P8 4   // Buffer<u8>: Y（幅×高さ）、U、V（それぞれ (幅+1)/2 × (高さ+1)/2）の順。素材の色の行列と範囲のまま
#define PX_YUV420P10 5  // Buffer<u16>: 同じ並びで、値は 0〜1023

// Buffer の要素の型（src/codegen.ts が渡す）。整数ならバイト数、浮動小数点なら 0x100 | バイト数。
#define SC_F32 0x104

// 拡大縮小のフィルタ（runtime/media.um の MediaScale と揃える）。
#define FILTER_FAST 0
#define FILTER_BILINEAR 1
#define FILTER_BICUBIC 2
#define FILTER_LANCZOS 3

// ---- 失敗の理由（スレッドごと） ----

static _Thread_local char last_error[1024];
static _Thread_local char last_log[512];  // このスレッドで FFmpeg が最後に出したエラーの行
static int log_verbose;
static pthread_once_t init_once = PTHREAD_ONCE_INIT;

static void set_error(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(last_error, sizeof last_error, fmt, ap);
  va_end(ap);
}

// FFmpeg のエラー番号を理由にする。FFmpeg がこのスレッドでエラーの行を出していれば、それも添える。
static void set_error_av(const char *what, int err) {
  char buf[AV_ERROR_MAX_STRING_SIZE];
  av_strerror(err, buf, sizeof buf);
  if (last_log[0]) set_error("%s: %s (%s)", what, buf, last_log);
  else set_error("%s: %s", what, buf);
}

static void log_callback(void *avcl, int level, const char *fmt, va_list vl) {
  if (level > AV_LOG_ERROR && !log_verbose) return;
  char line[512];
  int prefix = 1;
  va_list copy;
  va_copy(copy, vl);
  av_log_format_line2(avcl, level, fmt, copy, line, sizeof line, &prefix);
  va_end(copy);
  if (log_verbose) fputs(line, stderr);
  if (level <= AV_LOG_ERROR) {
    // 失敗の理由に使う行は「部品の名前: 文」にする（FFmpeg の接頭辞 "[h264 @ 0000...]" のアドレスは見せない）。
    char text[512];
    prefix = 0;
    va_copy(copy, vl);
    av_log_format_line2(avcl, level, fmt, copy, text, sizeof text, &prefix);
    va_end(copy);
    size_t n = strlen(text);
    while (n > 0 && (text[n - 1] == '\n' || text[n - 1] == '\r' || text[n - 1] == ' ')) text[--n] = 0;
    const AVClass *cls = avcl ? *(const AVClass *const *)avcl : NULL;
    const char *who = cls && cls->item_name ? cls->item_name(avcl) : NULL;
    if (n > 0) {
      if (who) snprintf(last_log, sizeof last_log, "%s: %s", who, text);
      else snprintf(last_log, sizeof last_log, "%s", text);
    }
  }
}

static void media_init(void) {
  const char *v = getenv("UM_MEDIA_LOG");
  log_verbose = v && v[0] == '1';
  av_log_set_callback(log_callback);
  if (log_verbose) av_log_set_level(AV_LOG_VERBOSE);
}

static void begin(void) {
  pthread_once(&init_once, media_init);
  last_log[0] = 0;
}

// ---- 表 ----

typedef struct {
  void *items[MAX_SLOTS];
  uint32_t gens[MAX_SLOTS];
} um_table;

static um_table readers, writers;
static pthread_mutex_t table_lock = PTHREAD_MUTEX_INITIALIZER;

static int64_t table_add(um_table *t, int kind, void *item) {
  pthread_mutex_lock(&table_lock);
  for (int i = 0; i < MAX_SLOTS; i++) {
    if (!t->items[i]) {
      t->items[i] = item;
      int64_t h = ((int64_t)kind << 56) | ((int64_t)(t->gens[i] & 0xffffffffu) << 24) | (int64_t)(i + 1);
      pthread_mutex_unlock(&table_lock);
      return h;
    }
  }
  pthread_mutex_unlock(&table_lock);
  return 0;
}

// 番号から場所を求める（table_lock を持って呼ぶ）。正しくなければ停止。
static int slot_of(um_table *t, int kind, int64_t h, int32_t line) {
  int64_t i = (h & 0xffffff) - 1;
  uint32_t gen = (uint32_t)((h >> 24) & 0xffffffff);
  if (h <= 0 || (h >> 56) != kind || i < 0 || i >= MAX_SLOTS || !t->items[i] || t->gens[i] != gen) {
    pthread_mutex_unlock(&table_lock);
    um_panic(UM_PANIC_MEDIA_HANDLE, line);
  }
  return (int)i;
}

// 読み手・書き手を借りる（先頭に _Atomic int busy を持つ構造体）。ほかのスレッドが使っている最中なら NULL（理由を書く）。
static void *acquire(um_table *t, int kind, int64_t h, int32_t line) {
  pthread_mutex_lock(&table_lock);
  void *item = t->items[slot_of(t, kind, h, line)];
  int expected = 0;
  int ok = atomic_compare_exchange_strong((_Atomic int *)item, &expected, 1);
  pthread_mutex_unlock(&table_lock);
  if (!ok) {
    set_error("the %s is being used by another thread (use one per thread)", kind == KIND_READER ? "reader" : "writer");
    return NULL;
  }
  return item;
}

static void release(void *item) { atomic_store((_Atomic int *)item, 0); }

// 表から外す。使っている最中なら停止。
static void *table_remove(um_table *t, int kind, int64_t h, int32_t line) {
  pthread_mutex_lock(&table_lock);
  int i = slot_of(t, kind, h, line);
  void *item = t->items[i];
  int expected = 0;
  if (!atomic_compare_exchange_strong((_Atomic int *)item, &expected, 1)) {
    pthread_mutex_unlock(&table_lock);
    um_panic(UM_PANIC_MEDIA_BUSY, line);
  }
  t->items[i] = NULL;
  t->gens[i]++;
  pthread_mutex_unlock(&table_lock);
  return item;
}

// ---- UM の値を返す ----

typedef struct {
  unsigned char *data;
  size_t len, cap;
} um_bytes;

static void give_string(const char *s, size_t n, unsigned char **out, size_t *len) {
  *out = NULL;
  *len = 0;
  if (n == 0) return;
  unsigned char *p = um_alloc(n, 1);
  memcpy(p, s, n);
  *out = p;
  *len = n;
}

// 文字列の配列（string[]）を作る。
static void give_strings(const char *const *items, size_t n, um_bytes **out, size_t *len) {
  *out = NULL;
  *len = 0;
  if (n == 0) return;
  um_bytes *a = um_alloc(n, sizeof(um_bytes));
  for (size_t k = 0; k < n; k++) {
    const char *s = items[k] ? items[k] : "";
    size_t sl = strlen(s);
    give_string(s, sl, &a[k].data, &a[k].len);
    a[k].cap = a[k].len;
  }
  *out = a;
  *len = n;
}

// UM の文字列（NUL で終わらない）を C の文字列にする。0 のバイトを含むなら NULL。
static char *c_string(const char *s, size_t n) {
  if (memchr(s, 0, n)) return NULL;
  char *p = malloc(n + 1);
  if (!p) um_panic(5, 0);
  memcpy(p, s, n);
  p[n] = 0;
  return p;
}

// ---- 画素の形式 ----

static const char *px_name(int fmt) {
  switch (fmt) {
    case PX_RGBA8: return "Rgba8";
    case PX_ARGB32: return "Argb32";
    case PX_RGBAF32: return "RgbaF32";
    case PX_RGBA16: return "Rgba16";
    case PX_YUV420P8: return "Yuv420p8";
    case PX_YUV420P10: return "Yuv420p10";
    default: return "?";
  }
}

// 形式に合う Buffer の要素の型と、必要なバイト数。合わなければ 0 を返し、理由を書く。
static size_t px_bytes(int fmt, int w, int h, int32_t scalar, size_t have_bytes) {
  size_t px = (size_t)w * (size_t)h;
  size_t cw = (size_t)(w + 1) / 2, ch = (size_t)(h + 1) / 2;
  size_t need;
  int32_t want;
  const char *types;
  switch (fmt) {
    case PX_RGBA8: need = px * 4; want = 1; types = "Buffer<u8>"; break;
    case PX_ARGB32: need = px * 4; want = 4; types = "Buffer<u32>"; break;
    case PX_RGBAF32: need = px * 16; want = SC_F32; types = "Buffer<f32> or Buffer<vec4<f32>>"; break;
    case PX_RGBA16: need = px * 8; want = 2; types = "Buffer<u16>"; break;
    case PX_YUV420P8: need = px + 2 * cw * ch; want = 1; types = "Buffer<u8>"; break;
    case PX_YUV420P10: need = (px + 2 * cw * ch) * 2; want = 2; types = "Buffer<u16>"; break;
    default: set_error("unknown pixel format %d", fmt); return 0;
  }
  if (scalar != want) {
    set_error("pixels in %s need a %s", px_name(fmt), types);
    return 0;
  }
  if (have_bytes < need) {
    set_error("the Buffer is too short for %dx%d %s (it needs %zu elements)", w, h, px_name(fmt),
              need / (size_t)(want & 0xff));
    return 0;
  }
  return need;
}

// UM の形式 → swscale に渡す FFmpeg の形式。RgbaF32 は 16 ビットの RGBA を経由する。
static enum AVPixelFormat px_av(int fmt) {
  switch (fmt) {
    case PX_RGBA8: return AV_PIX_FMT_RGBA;
    case PX_ARGB32: return AV_PIX_FMT_RGB32;  // u32 として読んだ値が 0xAARRGGBB（エンディアンによらない）
    case PX_RGBAF32: return AV_PIX_FMT_RGBA64;
    case PX_RGBA16: return AV_PIX_FMT_RGBA64;
    case PX_YUV420P8: return AV_PIX_FMT_YUV420P;
    case PX_YUV420P10: return AV_PIX_FMT_YUV420P10;
    default: return AV_PIX_FMT_NONE;
  }
}

static int px_is_rgb(int fmt) { return fmt <= PX_RGBA16; }

// UM の Buffer の中に、形式どおりの面（plane）を並べる。base は 16 ビットの RGBA の中間の領域のこともある。
static void px_planes(int fmt, uint8_t *base, int w, int h, uint8_t *data[4], int lines[4]) {
  memset(data, 0, 4 * sizeof data[0]);
  memset(lines, 0, 4 * sizeof lines[0]);
  int cw = (w + 1) / 2, ch = (h + 1) / 2;
  switch (fmt) {
    case PX_RGBA8:
    case PX_ARGB32: data[0] = base; lines[0] = w * 4; break;
    case PX_RGBAF32:
    case PX_RGBA16: data[0] = base; lines[0] = w * 8; break;
    case PX_YUV420P8:
      data[0] = base; lines[0] = w;
      data[1] = base + (size_t)w * h; lines[1] = cw;
      data[2] = data[1] + (size_t)cw * ch; lines[2] = cw;
      break;
    case PX_YUV420P10:
      data[0] = base; lines[0] = w * 2;
      data[1] = base + (size_t)w * h * 2; lines[1] = cw * 2;
      data[2] = data[1] + (size_t)cw * ch * 2; lines[2] = cw * 2;
      break;
  }
}

static int sws_flags_of(int filter) {
  switch (filter) {
    case FILTER_FAST: return SWS_FAST_BILINEAR;
    case FILTER_BILINEAR: return SWS_BILINEAR | SWS_FULL_CHR_H_INT;
    case FILTER_LANCZOS: return SWS_LANCZOS | SWS_FULL_CHR_H_INT | SWS_FULL_CHR_H_INP | SWS_ACCURATE_RND;
    default: return SWS_BICUBIC | SWS_FULL_CHR_H_INT | SWS_ACCURATE_RND;
  }
}

// FFmpeg の色の行列。指定がなければ大きさで推す（HD 以上は BT.709、それ未満は BT.601）。
static enum AVColorSpace guess_colorspace(enum AVColorSpace cs, int height) {
  if (cs != AVCOL_SPC_UNSPECIFIED && cs != AVCOL_SPC_RESERVED) return cs;
  return height >= 720 ? AVCOL_SPC_BT709 : AVCOL_SPC_SMPTE170M;
}

static int is_yuv(enum AVPixelFormat f) {
  const AVPixFmtDescriptor *d = av_pix_fmt_desc_get(f);
  return d && !(d->flags & AV_PIX_FMT_FLAG_RGB) && d->nb_components >= 3;
}

// YUV の値の範囲。指定がなければ、yuvj の形式だけ full（0〜255）、ほかは限られた範囲とみなす。
static enum AVColorRange guess_range(enum AVColorRange r, enum AVPixelFormat f) {
  if (r == AVCOL_RANGE_JPEG || r == AVCOL_RANGE_MPEG) return r;
  int full = f == AV_PIX_FMT_YUVJ420P || f == AV_PIX_FMT_YUVJ422P || f == AV_PIX_FMT_YUVJ444P || f == AV_PIX_FMT_YUVJ440P;
  return full ? AVCOL_RANGE_JPEG : AVCOL_RANGE_MPEG;
}

// 画素の変換（拡大縮小と色の変換）。libswscale の新しい API（sws_scale_frame）を使う: 色の行列と範囲はフレームの
// 情報から決まり、1 枚を複数のスレッドで分けて変換する。src と dst の色域・伝達特性は同じにする（色の管理はしない）。
// dst の画素は呼ぶ側の領域（data / lines）に書く。成功なら 1。
static int scale_into(SwsContext **ctx, int flags, const AVFrame *src, enum AVPixelFormat dfmt, int dw, int dh,
                      uint8_t *data[4], int lines[4], enum AVColorSpace dcs, enum AVColorRange drange) {
  if (!*ctx) {
    *ctx = sws_alloc_context();
    if (!*ctx) um_panic(5, 0);
    (*ctx)->threads = 0;  // CPU の数に合わせる
  }
  (*ctx)->flags = (unsigned)flags;
  AVFrame *dst = av_frame_alloc();
  if (!dst) um_panic(5, 0);
  dst->width = dw;
  dst->height = dh;
  dst->format = dfmt;
  for (int i = 0; i < 4; i++) {
    dst->data[i] = data[i];
    dst->linesize[i] = lines[i];
  }
  dst->colorspace = dcs;
  dst->color_range = drange;
  dst->color_primaries = src->color_primaries;
  dst->color_trc = src->color_trc;
  dst->chroma_location = AVCHROMA_LOC_LEFT;
  int err = sws_scale_frame(*ctx, dst, src);
  av_frame_free(&dst);  // data は呼ぶ側の領域（buf を持たないので解放されない）
  if (err < 0) {
    set_error_av("cannot convert the pixels", err);
    return 0;
  }
  return 1;
}

// 16 ビットの RGBA → 0〜1 の f32。
static void rgba16_to_f32(const uint16_t *src, float *dst, size_t n) {
  const float k = 1.0f / 65535.0f;
  for (size_t i = 0; i < n; i++) dst[i] = (float)src[i] * k;
}

static void f32_to_rgba16(const float *src, uint16_t *dst, size_t n) {
  for (size_t i = 0; i < n; i++) {
    float v = src[i];
    v = v != v ? 0.0f : v < 0.0f ? 0.0f : v > 1.0f ? 1.0f : v;  // NaN は 0
    dst[i] = (uint16_t)(v * 65535.0f + 0.5f);
  }
}

// ---- 読み手 ----

typedef struct {
  _Atomic int busy;  // 先頭に置く（acquire が見る）
  int kind;          // READ_*
  AVFormatContext *fmt;
  int vi, ai;  // 映像・音のストリームの番号（なければ -1）
  int si;      // 読むストリーム
  AVStream *st;
  AVCodecContext *dec;
  AVBufferRef *hw_device;
  enum AVPixelFormat hw_fmt;
  char hw_name[32];  // 使っているハードウェアの展開（なければ "none"）
  AVPacket *pkt;
  AVFrame *frame;  // 復号の受け皿
  AVFrame *sw;     // ハードウェアの面から写す先
  AVFrame *cur;    // 最後に返したコマ（映像）
  int has_cur;
  AVFrame *ahead;  // シークの途中で目標より先まで復号したコマ（次に順に読むときに返す）
  int has_ahead;
  int flushing;    // demuxer が終わり、復号器に終わりを伝えた
  int dec_eof;     // 復号器が終わった
  int64_t start;   // ストリームの最初の時刻（ストリームの時間の単位）
  int64_t next_pts;  // 時刻のないコマのための、次のコマの推定の時刻
  int64_t frame_dur; // 1 コマの長さの既定（ストリームの時間の単位）
  int errors;        // 続けて起きた復号の失敗
  double last_time;
  SwsContext *sws;
  uint16_t *tmp16;
  size_t tmp16_cap;
  // 音
  SwrContext *swr;
  int swr_rate;
  AVChannelLayout swr_in_layout;
  int swr_in_fmt, swr_in_rate;
  AVAudioFifo *fifo;  // 変換済みの標本（ステレオの f32）
  double fifo_time;   // fifo の先頭の標本の時刻
  int fifo_rate;
  int swr_flushed;
} um_reader;

static double ts_to_sec(um_reader *r, int64_t pts) { return (double)(pts - r->start) * av_q2d(r->st->time_base); }

static int64_t sec_to_ts(um_reader *r, double t) {
  AVRational tb = r->st->time_base;
  return r->start + (int64_t)llrint(t * (double)tb.den / (double)tb.num);
}

static void free_reader(um_reader *r) {
  avcodec_free_context(&r->dec);
  av_buffer_unref(&r->hw_device);
  avformat_close_input(&r->fmt);
  av_packet_free(&r->pkt);
  av_frame_free(&r->frame);
  av_frame_free(&r->sw);
  av_frame_free(&r->cur);
  av_frame_free(&r->ahead);
  sws_free_context(&r->sws);
  free(r->tmp16);
  swr_free(&r->swr);
  av_channel_layout_uninit(&r->swr_in_layout);
  if (r->fifo) av_audio_fifo_free(r->fifo);
  free(r);
}

// ハードウェアの展開の種類の候補（"auto" のとき、前から順に試す）。
static const char *const AUTO_HW[] = {
#if defined(_WIN32)
  "d3d11va", "cuda", "dxva2",
#elif defined(__APPLE__)
  "videotoolbox",
#else
  "vaapi", "cuda", "vdpau",
#endif
  NULL,
};

static enum AVPixelFormat get_hw_format(AVCodecContext *ctx, const enum AVPixelFormat *fmts) {
  um_reader *r = ctx->opaque;
  for (const enum AVPixelFormat *p = fmts; *p != AV_PIX_FMT_NONE; p++) {
    if (*p == r->hw_fmt) return *p;
  }
  // この素材（プロファイルなど）はハードウェアで展開できない。ソフトウェアに戻る。
  snprintf(r->hw_name, sizeof r->hw_name, "none");
  for (const enum AVPixelFormat *p = fmts; *p != AV_PIX_FMT_NONE; p++) {
    const AVPixFmtDescriptor *d = av_pix_fmt_desc_get(*p);
    if (d && !(d->flags & AV_PIX_FMT_FLAG_HWACCEL)) return *p;
  }
  return AV_PIX_FMT_NONE;
}

// Linux: ハードウェアの展開に要る共有ライブラリが読めるか。FFmpeg のビルドによっては（BtbN の配布物など）、libva や
// libvdpau を「初めて呼ぶときに dlopen する」形でリンクしてあり、ライブラリのない機械で試すと、エラーを返さずに
// プログラムごと止まる（abort）。そこで、試す前に読めることを確かめ、読めなければその種類は使えないものとして扱う。
// vaapi は DRM の経路（libva-drm）を先に試し、X11 の経路（libva-x11）は X のディスプレイを開けたときだけ使う。
static int hw_libs_present(const char *name) {
#if !defined(_WIN32) && !defined(__APPLE__)
  const char *libs[4] = { NULL, NULL, NULL, NULL };
  if (strcmp(name, "vaapi") == 0) {
    libs[0] = "libva.so.2";
    libs[1] = "libva-drm.so.2";
    const char *display = getenv("DISPLAY");
    if (display && display[0]) libs[2] = "libva-x11.so.2";
  } else if (strcmp(name, "vdpau") == 0) {
    libs[0] = "libvdpau.so.1";
  }
  for (int i = 0; libs[i]; i++) {
    void *h = dlopen(libs[i], RTLD_LAZY | RTLD_LOCAL);
    if (!h) return 0;
    dlclose(h);
  }
#else
  (void)name;
#endif
  return 1;
}

// 名前のハードウェアの展開を準備する。使えなければ 0。
static int try_hw(um_reader *r, const AVCodec *codec, const char *name) {
  enum AVHWDeviceType type = av_hwdevice_find_type_by_name(name);
  if (type == AV_HWDEVICE_TYPE_NONE || !hw_libs_present(name)) return 0;
  for (int i = 0;; i++) {
    const AVCodecHWConfig *cfg = avcodec_get_hw_config(codec, i);
    if (!cfg) return 0;
    if ((cfg->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX) && cfg->device_type == type) {
      AVBufferRef *dev = NULL;
      if (av_hwdevice_ctx_create(&dev, type, NULL, NULL, 0) < 0) return 0;
      r->hw_device = dev;
      r->hw_fmt = cfg->pix_fmt;
      snprintf(r->hw_name, sizeof r->hw_name, "%s", name);
      return 1;
    }
  }
}

// 映像の復号器を選び、ハードウェアの展開を準備する。hw が "none" 以外なら、その形式の復号器のうち GPU の経路を持つものを探す
// （AV1 では、既定の libdav1d には GPU の経路がなく、内蔵の av1 にはある）。どれも使えなければ既定の復号器でソフトウェア。
static int pick_hw(um_reader *r, const AVCodec **codec, const char *name) {
  if (try_hw(r, *codec, name)) return 1;
  void *it = NULL;
  const AVCodec *c;
  while ((c = av_codec_iterate(&it))) {
    if (c != *codec && av_codec_is_decoder(c) && c->id == (*codec)->id && try_hw(r, c, name)) {
      *codec = c;
      return 1;
    }
  }
  return 0;
}

static int open_decoder(um_reader *r, const char *hw) {
  AVCodecParameters *par = r->st->codecpar;
  const AVCodec *codec = avcodec_find_decoder(par->codec_id);
  if (!codec) {
    set_error("no decoder for %s in this FFmpeg build", avcodec_get_name(par->codec_id));
    return 0;
  }
  snprintf(r->hw_name, sizeof r->hw_name, "none");
  int use_hw = 0;
  if (r->kind == READ_VIDEO && hw[0] && strcmp(hw, "none") != 0) {
    if (strcmp(hw, "auto") == 0) {
      for (int k = 0; AUTO_HW[k] && !use_hw; k++) use_hw = pick_hw(r, &codec, AUTO_HW[k]);
    } else {
      if (av_hwdevice_find_type_by_name(hw) == AV_HWDEVICE_TYPE_NONE) {
        set_error("unknown hardware decoder '%s' (use none, auto, d3d11va, dxva2, cuda, qsv, videotoolbox, vaapi, vulkan, ...)", hw);
        return 0;
      }
      use_hw = pick_hw(r, &codec, hw);  // 使えなければソフトウェアに戻る（hwaccel の情報は "none" になる）
    }
  }
  r->dec = avcodec_alloc_context3(codec);
  if (!r->dec) um_panic(5, 0);
  int err = avcodec_parameters_to_context(r->dec, par);
  if (err < 0) { set_error_av("cannot set up the decoder", err); return 0; }
  r->dec->pkt_timebase = r->st->time_base;
  r->dec->opaque = r;
  if (use_hw) {
    r->dec->hw_device_ctx = av_buffer_ref(r->hw_device);
    r->dec->get_format = get_hw_format;
    // ハードウェアの展開は1スレッドで足りる（コマのスレッドは遅延を増やすだけ）。
    r->dec->thread_count = 1;
  } else {
    r->dec->thread_count = 0;  // CPU の数に合わせる
    r->dec->thread_type = FF_THREAD_FRAME | FF_THREAD_SLICE;
  }
  err = avcodec_open2(r->dec, codec, NULL);
  if (err < 0) { set_error_av("cannot open the decoder", err); return 0; }
  return 1;
}

int64_t um_media_open(const char *path, size_t path_len, int32_t kind, const char *hw, size_t hw_len, int32_t line) {
  (void)line;
  begin();
  if (kind < READ_INFO || kind > READ_AUDIO) {
    set_error("unknown reader kind %d", kind);
    return 0;
  }
  char *p = c_string(path, path_len);
  char *h = c_string(hw, hw_len);
  if (!p || !h) {
    set_error("the path contains a zero byte");
    free(p);
    free(h);
    return 0;
  }
  um_reader *r = calloc(1, sizeof *r);
  if (!r) um_panic(5, 0);
  r->kind = kind;
  r->vi = r->ai = r->si = -1;
  r->last_time = -1;
  int64_t handle = 0;
  int err = avformat_open_input(&r->fmt, p, NULL, NULL);
  if (err < 0) { set_error_av(p, err); goto fail; }
  err = avformat_find_stream_info(r->fmt, NULL);
  if (err < 0) { set_error_av("cannot read the stream information", err); goto fail; }
  r->vi = av_find_best_stream(r->fmt, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
  r->ai = av_find_best_stream(r->fmt, AVMEDIA_TYPE_AUDIO, -1, -1, NULL, 0);
  // 表紙の画像（attached_pic）は映像として扱わない。
  if (r->vi >= 0 && (r->fmt->streams[r->vi]->disposition & AV_DISPOSITION_ATTACHED_PIC)) r->vi = -1;
  if (r->vi < 0) r->vi = -1;
  if (r->ai < 0) r->ai = -1;
  if (kind == READ_VIDEO) {
    if (r->vi < 0) { set_error("%s has no video stream", p); goto fail; }
    r->si = r->vi;
  } else if (kind == READ_AUDIO) {
    if (r->ai < 0) { set_error("%s has no audio stream", p); goto fail; }
    r->si = r->ai;
  }
  if (kind != READ_INFO) {
    // 読まないストリームは demuxer の段階で捨てる。
    for (unsigned i = 0; i < r->fmt->nb_streams; i++) {
      if ((int)i != r->si) r->fmt->streams[i]->discard = AVDISCARD_ALL;
    }
    r->st = r->fmt->streams[r->si];
    r->start = r->st->start_time != AV_NOPTS_VALUE ? r->st->start_time : 0;
    r->next_pts = r->start;
    if (kind == READ_VIDEO) {
      AVRational fr = r->st->avg_frame_rate.num > 0 ? r->st->avg_frame_rate : r->st->r_frame_rate;
      r->frame_dur = fr.num > 0 ? av_rescale_q(1, av_inv_q(fr), r->st->time_base) : 1;
      if (r->frame_dur <= 0) r->frame_dur = 1;
    }
    if (!open_decoder(r, h)) goto fail;
    r->pkt = av_packet_alloc();
    r->frame = av_frame_alloc();
    r->sw = av_frame_alloc();
    r->cur = av_frame_alloc();
    r->ahead = av_frame_alloc();
    if (!r->pkt || !r->frame || !r->sw || !r->cur || !r->ahead) um_panic(5, 0);
  }
  handle = table_add(&readers, KIND_READER, r);
  if (!handle) { set_error("too many open readers (%d)", MAX_SLOTS); goto fail; }
  free(p);
  free(h);
  return handle;
fail:
  free(p);
  free(h);
  free_reader(r);
  return 0;
}

void um_media_close(int64_t h, int32_t line) {
  um_reader *r = table_remove(&readers, KIND_READER, h, line);
  free_reader(r);
}

// 情報の数（f64[]）。並びは runtime/media.um の mediaInfoOf と揃える。
#define INFO_NUMBERS 20
void um_media_info_numbers(int64_t h, double **out, size_t *len, int32_t line) {
  *out = NULL;
  *len = 0;
  um_reader *r = acquire(&readers, KIND_READER, h, line);
  double *v = um_alloc(INFO_NUMBERS, sizeof(double));
  memset(v, 0, INFO_NUMBERS * sizeof(double));
  *out = v;
  *len = INFO_NUMBERS;
  if (!r) { v[0] = -2; return; }  // 使用中（ふつうは起きない）
  AVFormatContext *f = r->fmt;
  v[0] = f->duration > 0 ? (double)f->duration / AV_TIME_BASE : -1;
  v[1] = f->bit_rate > 0 ? (double)f->bit_rate : 0;
  if (r->vi >= 0) {
    AVStream *s = f->streams[r->vi];
    AVCodecParameters *p = s->codecpar;
    v[2] = 1;
    v[3] = p->width;
    v[4] = p->height;
    AVRational fr = s->avg_frame_rate.num > 0 ? s->avg_frame_rate : s->r_frame_rate;
    v[5] = fr.num;
    v[6] = fr.den;
    v[7] = s->nb_frames > 0 ? (double)s->nb_frames
         : (s->duration > 0 && fr.num > 0) ? floor((double)s->duration * av_q2d(s->time_base) * av_q2d(fr) + 0.5)
         : (f->duration > 0 && fr.num > 0) ? floor((double)f->duration / AV_TIME_BASE * av_q2d(fr) + 0.5) : -1;
    const AVPixFmtDescriptor *d = av_pix_fmt_desc_get((enum AVPixelFormat)p->format);
    v[8] = d ? d->comp[0].depth : 0;
    double rot = 0;
    const AVPacketSideData *sd = av_packet_side_data_get(p->coded_side_data, p->nb_coded_side_data, AV_PKT_DATA_DISPLAYMATRIX);
    if (sd && sd->size >= 9 * 4) {
      double a = av_display_rotation_get((const int32_t *)sd->data);
      if (a == a) rot = -a;  // FFmpeg は反時計回り。時計回りの角度にする
    }
    rot = fmod(floor(rot / 90.0 + 0.5) * 90.0, 360.0);
    if (rot < 0) rot += 360;
    v[9] = rot;
    AVRational sar = av_guess_sample_aspect_ratio(f, s, NULL);
    v[10] = sar.num > 0 ? sar.num : 1;
    v[11] = sar.num > 0 ? sar.den : 1;
    v[12] = s->duration > 0 ? (double)s->duration * av_q2d(s->time_base) : v[0];
    v[13] = p->bit_rate > 0 ? (double)p->bit_rate : 0;
  }
  if (r->ai >= 0) {
    AVStream *s = f->streams[r->ai];
    AVCodecParameters *p = s->codecpar;
    v[14] = 1;
    v[15] = p->sample_rate;
    v[16] = p->ch_layout.nb_channels;
    v[17] = s->duration > 0 ? (double)s->duration * av_q2d(s->time_base) : v[0];
    v[18] = p->bit_rate > 0 ? (double)p->bit_rate : 0;
  }
  release(r);
}

// 情報の文字列（string[]）。並びは runtime/media.um の mediaInfoOf と揃える。
void um_media_info_text(int64_t h, um_bytes **out, size_t *len, int32_t line) {
  um_reader *r = acquire(&readers, KIND_READER, h, line);
  const char *t[12] = { 0 };
  char layout[128] = "";
  if (r) {
    AVFormatContext *f = r->fmt;
    t[0] = f->iformat ? f->iformat->name : "";
    if (r->vi >= 0) {
      AVCodecParameters *p = f->streams[r->vi]->codecpar;
      t[1] = avcodec_get_name(p->codec_id);
      t[2] = av_get_pix_fmt_name((enum AVPixelFormat)p->format);
      t[3] = av_color_space_name(p->color_space);
      t[4] = av_color_primaries_name(p->color_primaries);
      t[5] = av_color_transfer_name(p->color_trc);
      t[6] = av_color_range_name(p->color_range);
      t[7] = avcodec_profile_name(p->codec_id, p->profile);
    }
    if (r->ai >= 0) {
      AVCodecParameters *p = f->streams[r->ai]->codecpar;
      t[8] = avcodec_get_name(p->codec_id);
      av_channel_layout_describe(&p->ch_layout, layout, sizeof layout);
      t[9] = layout;
      t[10] = av_get_sample_fmt_name((enum AVSampleFormat)p->format);
    }
    t[11] = r->kind == READ_VIDEO ? r->hw_name : "none";
  }
  give_strings(t, 12, out, len);
  if (r) release(r);
}

// 1 コマ復号して r->frame に入れる（ハードウェアの面は CPU に写す）。1 = コマ、0 = 終わり、-1 = 失敗。
static int decode_one(um_reader *r) {
  for (;;) {
    int err = avcodec_receive_frame(r->dec, r->frame);
    if (err == 0) {
      r->errors = 0;
      if (r->frame->format == r->hw_fmt && r->hw_device) {
        av_frame_unref(r->sw);
        err = av_hwframe_transfer_data(r->sw, r->frame, 0);
        if (err < 0) { set_error_av("cannot copy the frame from the GPU", err); av_frame_unref(r->frame); return -1; }
        av_frame_copy_props(r->sw, r->frame);
        av_frame_unref(r->frame);
        av_frame_move_ref(r->frame, r->sw);
      }
      // 時刻。なければ前のコマから推す。
      int64_t pts = r->frame->best_effort_timestamp;
      if (pts == AV_NOPTS_VALUE) pts = r->frame->pts;
      if (pts == AV_NOPTS_VALUE) pts = r->next_pts;
      r->frame->pts = pts;
      int64_t dur = r->frame->duration > 0 ? r->frame->duration : r->frame_dur;
      if (r->kind == READ_AUDIO && r->frame->duration <= 0 && r->frame->sample_rate > 0) {
        dur = av_rescale_q(r->frame->nb_samples, (AVRational){ 1, r->frame->sample_rate }, r->st->time_base);
      }
      r->frame->duration = dur > 0 ? dur : 1;
      r->next_pts = pts + r->frame->duration;
      return 1;
    }
    if (err == AVERROR_EOF) {
      r->dec_eof = 1;
      return 0;
    }
    if (err != AVERROR(EAGAIN)) {
      // 壊れたコマ。いくつかは飛ばして続ける（続けて失敗が多すぎれば、読めないファイルとする）。
      if (++r->errors > 64) { set_error_av("cannot decode", err); return -1; }
      continue;
    }
    if (r->flushing) {
      r->dec_eof = 1;
      return 0;
    }
    // パケットを1つ読んで復号器に渡す。
    err = av_read_frame(r->fmt, r->pkt);
    if (err == AVERROR_EOF || (err < 0 && avio_feof(r->fmt->pb))) {
      r->flushing = 1;
      avcodec_send_packet(r->dec, NULL);
      continue;
    }
    if (err < 0) {
      if (++r->errors > 64) { set_error_av("cannot read", err); return -1; }
      continue;
    }
    if (r->pkt->stream_index != r->si) {
      av_packet_unref(r->pkt);
      continue;
    }
    err = avcodec_send_packet(r->dec, r->pkt);
    av_packet_unref(r->pkt);
    if (err < 0 && err != AVERROR(EAGAIN)) {
      if (++r->errors > 64) { set_error_av("cannot decode", err); return -1; }
    }
  }
}

static int do_seek(um_reader *r, int64_t ts) {
  int err = av_seek_frame(r->fmt, r->si, ts, AVSEEK_FLAG_BACKWARD);
  if (err < 0) {
    // 索引のない形式などで後ろ向きに探せなければ、どちら向きでも近いところへ。
    err = avformat_seek_file(r->fmt, r->si, INT64_MIN, ts, ts, 0);
  }
  if (err < 0) { set_error_av("cannot seek", err); return 0; }
  avcodec_flush_buffers(r->dec);
  r->flushing = 0;
  r->dec_eof = 0;
  r->has_ahead = 0;
  r->has_cur = 0;
  r->errors = 0;
  av_frame_unref(r->cur);
  av_frame_unref(r->ahead);
  r->next_pts = ts;
  return 1;
}

static int frame_covers(AVFrame *f, int64_t ts) { return f->pts <= ts && ts < f->pts + f->duration; }

// 映像: 目標の時刻 ts に表示されるコマを r->cur に入れる。1 = あった、0 = 終わりより後、-1 = 失敗。
static int seek_video(um_reader *r, int64_t ts) {
  if (r->has_cur && frame_covers(r->cur, ts)) return 1;
  if (r->has_ahead && frame_covers(r->ahead, ts)) {
    av_frame_unref(r->cur);
    av_frame_move_ref(r->cur, r->ahead);
    r->has_cur = 1;
    r->has_ahead = 0;
    return 1;
  }
  // 今の位置から前へ復号したほうが速いか: 目標が今より後ろで、間にキーフレームがない（索引で分かる）か、近い。
  int forward = 0;
  if (r->has_cur && ts > r->cur->pts && !r->dec_eof) {
    int idx = av_index_search_timestamp(r->st, ts, AVSEEK_FLAG_BACKWARD);
    const AVIndexEntry *e = idx >= 0 ? avformat_index_get_entry(r->st, idx) : NULL;
    double gap = (double)(ts - r->cur->pts) * av_q2d(r->st->time_base);
    if ((e && e->timestamp <= r->cur->pts) || gap < 0.5) forward = 1;
  }
  int64_t seek_ts = ts;
  int seeked = 0;
  if (!forward) {
    if (!do_seek(r, seek_ts)) return -1;
    seeked = 1;
  }
  for (;;) {
    int got;
    if (r->has_ahead) {
      av_frame_unref(r->frame);
      av_frame_move_ref(r->frame, r->ahead);
      r->has_ahead = 0;
      got = 1;
    } else {
      got = decode_one(r);
    }
    if (got < 0) return -1;
    if (got == 0) {
      // 終わり。目標が最後のコマの中なら、そのコマ。
      if (r->has_cur && frame_covers(r->cur, ts)) return 1;
      if (r->has_cur && ts < r->cur->pts + r->cur->duration) return 1;
      return 0;
    }
    if (frame_covers(r->frame, ts)) {
      av_frame_unref(r->cur);
      av_frame_move_ref(r->cur, r->frame);
      r->has_cur = 1;
      return 1;
    }
    if (r->frame->pts > ts) {
      if (r->has_cur) {
        // 目標は前のコマ（コマの長さが揺れる素材）。いま復号したコマは次に返す。
        av_frame_unref(r->ahead);
        av_frame_move_ref(r->ahead, r->frame);
        r->has_ahead = 1;
        return 1;
      }
      if (seeked && seek_ts > r->start && seeked < 6) {
        // シークが目標より後ろに着いた（索引が粗い）。もっと前から読み直す。
        int64_t back = av_rescale_q(1 << (seeked - 1), (AVRational){ 1, 1 }, r->st->time_base);
        seek_ts = ts - back > r->start ? ts - back : r->start;
        if (!do_seek(r, seek_ts)) return -1;
        seeked++;
        continue;
      }
      // 目標が最初のコマより前。最初のコマを返す。
      av_frame_unref(r->cur);
      av_frame_move_ref(r->cur, r->frame);
      r->has_cur = 1;
      return 1;
    }
    // 目標より前のコマ。覚えておいて次へ。
    av_frame_unref(r->cur);
    av_frame_move_ref(r->cur, r->frame);
    r->has_cur = 1;
  }
}

// r->cur を UM の Buffer に変換して書く。成功なら 1。
static int convert_video(um_reader *r, uint8_t *out, size_t out_bytes, int32_t scalar, int fmt, int w, int h, int filter) {
  AVFrame *f = r->cur;
  if (w <= 0) w = f->width;
  if (h <= 0) h = f->height;
  if (w > 16384 || h > 16384) { set_error("the size %dx%d is too large", w, h); return 0; }
  size_t need = px_bytes(fmt, w, h, scalar, out_bytes);
  if (!need) return 0;
  enum AVPixelFormat sfmt = (enum AVPixelFormat)f->format;
  enum AVPixelFormat dfmt = px_av(fmt);
  // 書いていない色の行列と範囲は推した値にする（RGB の素材ならそのまま）。
  if (is_yuv(sfmt)) {
    f->colorspace = guess_colorspace(f->colorspace, f->height);
    f->color_range = guess_range(f->color_range, sfmt);
  }
  uint8_t *base = out;
  if (fmt == PX_RGBAF32) {
    size_t n = (size_t)w * h * 4;
    if (r->tmp16_cap < n) {
      free(r->tmp16);
      r->tmp16 = malloc(n * sizeof(uint16_t));
      if (!r->tmp16) um_panic(5, 0);
      r->tmp16_cap = n;
    }
    base = (uint8_t *)r->tmp16;
  }
  uint8_t *data[4];
  int lines[4];
  px_planes(fmt, base, w, h, data, lines);
  // RGB にするときは 0〜255 いっぱいに、YUV のまま取り出すときは素材の行列と範囲のまま。
  enum AVColorSpace dcs = px_is_rgb(fmt) ? AVCOL_SPC_RGB : f->colorspace;
  enum AVColorRange drange = px_is_rgb(fmt) ? AVCOL_RANGE_JPEG : f->color_range;
  if (!scale_into(&r->sws, sws_flags_of(filter), f, dfmt, w, h, data, lines, dcs, drange)) return 0;
  if (fmt == PX_RGBAF32) rgba16_to_f32(r->tmp16, (float *)out, (size_t)w * h * 4);
  return 1;
}

// 映像のコマを1つ読む。time が負なら次のコマ、0 以上ならその時刻に表示されるコマ。
// 戻り値はコマの時刻（秒、0 以上）。終わりなら -1、失敗なら -2。
double um_media_read_video(int64_t h, double time, uint8_t *out, size_t len, size_t esize, int32_t scalar,
                           int32_t fmt, int32_t w, int32_t ht, int32_t filter, int32_t line) {
  begin();
  um_reader *r = acquire(&readers, KIND_READER, h, line);
  if (!r) return -2;
  double result = -2;
  if (r->kind != READ_VIDEO) {
    set_error("this reader was not opened for video (use mediaOpenVideo)");
    goto done;
  }
  if (time != time) { set_error("the time is NaN"); goto done; }
  int got;
  if (time < 0) {
    if (r->has_ahead) {
      av_frame_unref(r->cur);
      av_frame_move_ref(r->cur, r->ahead);
      r->has_ahead = 0;
      r->has_cur = 1;
      got = 1;
    } else {
      got = decode_one(r);
      if (got == 1) {
        av_frame_unref(r->cur);
        av_frame_move_ref(r->cur, r->frame);
        r->has_cur = 1;
      }
    }
  } else {
    got = seek_video(r, sec_to_ts(r, time));
  }
  if (got < 0) goto done;
  if (got == 0) { result = -1; goto done; }
  if (!convert_video(r, out, len * esize, scalar, fmt, w, ht, filter)) goto done;
  result = ts_to_sec(r, r->cur->pts);
  if (result < 0) result = 0;
  r->last_time = result;
done:
  release(r);
  return result;
}

// ---- 音を読む ----

static int setup_swr(um_reader *r, AVFrame *f, int rate) {
  int fmt = f->format;
  if (r->swr && r->swr_rate == rate && r->swr_in_fmt == fmt && r->swr_in_rate == f->sample_rate &&
      av_channel_layout_compare(&r->swr_in_layout, &f->ch_layout) == 0) {
    return 1;
  }
  swr_free(&r->swr);
  AVChannelLayout stereo = AV_CHANNEL_LAYOUT_STEREO;
  AVChannelLayout in = { 0 };
  if (f->ch_layout.order == AV_CHANNEL_ORDER_UNSPEC || !av_channel_layout_check(&f->ch_layout)) {
    av_channel_layout_default(&in, f->ch_layout.nb_channels > 0 ? f->ch_layout.nb_channels : 2);
  } else {
    av_channel_layout_copy(&in, &f->ch_layout);
  }
  int err = swr_alloc_set_opts2(&r->swr, &stereo, AV_SAMPLE_FMT_FLT, rate, &in, (enum AVSampleFormat)fmt, f->sample_rate, 0, NULL);
  av_channel_layout_uninit(&in);
  if (err >= 0) err = swr_init(r->swr);
  if (err < 0) { set_error_av("cannot convert the audio", err); swr_free(&r->swr); return 0; }
  r->swr_rate = rate;
  r->swr_in_fmt = fmt;
  r->swr_in_rate = f->sample_rate;
  av_channel_layout_uninit(&r->swr_in_layout);
  av_channel_layout_copy(&r->swr_in_layout, &f->ch_layout);
  return 1;
}

static void fifo_reset(um_reader *r, int rate) {
  if (!r->fifo || r->fifo_rate != rate) {
    if (r->fifo) av_audio_fifo_free(r->fifo);
    r->fifo = av_audio_fifo_alloc(AV_SAMPLE_FMT_FLT, 2, 4096);
    if (!r->fifo) um_panic(5, 0);
    r->fifo_rate = rate;
  }
  av_audio_fifo_reset(r->fifo);
}

// 変換した標本を fifo に足す。in が NULL なら変換器に残った分を出す。
static int push_converted(um_reader *r, AVFrame *in) {
  int max_out = swr_get_out_samples(r->swr, in ? in->nb_samples : 0);
  if (max_out <= 0) return 1;
  float *buf = malloc((size_t)max_out * 2 * sizeof(float));
  if (!buf) um_panic(5, 0);
  uint8_t *outp[1] = { (uint8_t *)buf };
  int n = swr_convert(r->swr, outp, max_out, in ? (const uint8_t *const *)in->extended_data : NULL, in ? in->nb_samples : 0);
  if (n < 0) { free(buf); set_error_av("cannot convert the audio", n); return 0; }
  if (n > 0) {
    void *planes[1] = { buf };
    if (av_audio_fifo_write(r->fifo, planes, n) < n) um_panic(5, 0);
  }
  free(buf);
  return 1;
}

// fifo の先頭から n 標本を捨てる。
static void fifo_drop(um_reader *r, int n) {
  if (n <= 0) return;
  int have = av_audio_fifo_size(r->fifo);
  if (n > have) n = have;
  av_audio_fifo_drain(r->fifo, n);
  r->fifo_time += (double)n / r->fifo_rate;
}

// 無音を fifo に足す。
static void push_silence(um_reader *r, int n) {
  float *z = calloc((size_t)n * 2, sizeof(float));
  if (!z) um_panic(5, 0);
  void *planes[1] = { z };
  if (av_audio_fifo_write(r->fifo, planes, n) < n) um_panic(5, 0);
  free(z);
}

// 復号して fifo に少なくとも want 標本あるようにする（終わりならあるだけ）。1 = 成功、-1 = 失敗。
// fifo が空のときに来た最初のコマで、fifo の先頭の時刻を決める。
static int fill_fifo(um_reader *r, int want, int rate) {
  while (av_audio_fifo_size(r->fifo) < want) {
    if (r->dec_eof) {
      if (r->swr && !r->swr_flushed) {
        r->swr_flushed = 1;
        if (!push_converted(r, NULL)) return -1;
        continue;
      }
      return 1;
    }
    int got = decode_one(r);
    if (got < 0) return -1;
    if (got == 0) continue;
    if (!setup_swr(r, r->frame, rate)) return -1;
    if (av_audio_fifo_size(r->fifo) == 0) {
      double t = ts_to_sec(r, r->frame->pts) - (double)swr_get_delay(r->swr, rate) / rate;
      r->fifo_time = t;
    }
    if (!push_converted(r, r->frame)) return -1;
    av_frame_unref(r->frame);
  }
  return 1;
}

// 音を読む。time が負なら続き、0 以上ならその時刻から。out にステレオの f32 を交互に書く（L, R, L, R, ...）。
// 戻り値は書いた標本の組の数（1 以上）。終わりなら 0、失敗なら -1。書いた先頭の時刻は mediaLastTime。
int64_t um_media_read_audio(int64_t h, double time, float *out, size_t len, size_t esize, int32_t scalar, int32_t rate, int32_t line) {
  begin();
  um_reader *r = acquire(&readers, KIND_READER, h, line);
  if (!r) return -1;
  int64_t result = -1;
  if (r->kind != READ_AUDIO) {
    set_error("this reader was not opened for audio (use mediaOpenAudio)");
    goto done;
  }
  if (scalar != SC_F32 || esize != 4) { set_error("audio samples need a Buffer<f32> or f32[]"); goto done; }
  if (rate < 1000 || rate > 768000) { set_error("the sample rate %d is out of range (1000 to 768000)", rate); goto done; }
  if (time != time) { set_error("the time is NaN"); goto done; }
  int cap = len / 2 > (size_t)INT32_MAX ? INT32_MAX : (int)(len / 2);
  if (cap == 0) { set_error("the Buffer has room for no stereo samples"); goto done; }
  if (!r->fifo) {
    // 初めて: time が負なら、今の位置（最初）から読む。
    fifo_reset(r, rate);
  } else if (r->fifo_rate != rate) {
    // 標本化周波数が変わった: 続きの時刻から読み直す。
    double from = r->fifo_time;
    fifo_reset(r, rate);
    swr_free(&r->swr);
    r->last_time = -1;
    if (time < 0) time = from;
  }
  if (time >= 0) {
    double end = r->fifo_time + (double)av_audio_fifo_size(r->fifo) / rate;
    // 今の位置のすぐ後ろなら、シークせずに読み進める。
    int in_fifo = r->last_time >= 0 && time >= r->fifo_time - 0.5 / rate && time < end + 0.5;
    if (!in_fifo) {
      // シークする（少し前から読み、余分を捨てる）。
      int64_t ts = sec_to_ts(r, time > 0.1 ? time - 0.1 : 0);
      if (!do_seek(r, ts)) goto done;
      fifo_reset(r, rate);
      swr_free(&r->swr);
      r->swr_flushed = 0;
      if (fill_fifo(r, 1, rate) < 0) goto done;
      if (av_audio_fifo_size(r->fifo) > 0 && r->fifo_time > time + 0.0005) {
        // シークが目標より後ろに着いた。最初から読み直す（音はたいてい全部のパケットがキーフレーム）。
        if (!do_seek(r, r->start)) goto done;
        fifo_reset(r, rate);
        swr_free(&r->swr);
        r->swr_flushed = 0;
        if (fill_fifo(r, 1, rate) < 0) goto done;
      }
      if (av_audio_fifo_size(r->fifo) > 0 && r->fifo_time > time) {
        // 目標がストリームの最初より前: 間を無音で埋める。
        double gap = r->fifo_time - time;
        int n = (int)llrint(gap * rate);
        if (n > 0 && gap < 60) {
          float *keep = malloc((size_t)av_audio_fifo_size(r->fifo) * 2 * sizeof(float));
          if (!keep) um_panic(5, 0);
          int k = av_audio_fifo_size(r->fifo);
          void *kp[1] = { keep };
          av_audio_fifo_read(r->fifo, kp, k);
          push_silence(r, n);
          av_audio_fifo_write(r->fifo, kp, k);
          free(keep);
          r->fifo_time -= (double)n / rate;
        }
      }
    }
    // 目標より前の標本を捨てる。
    for (;;) {
      int n = (int)llrint((time - r->fifo_time) * rate);
      if (n <= 0) break;
      int have = av_audio_fifo_size(r->fifo);
      if (n <= have) { fifo_drop(r, n); break; }
      fifo_drop(r, have);
      if (r->dec_eof && r->swr_flushed) break;
      if (fill_fifo(r, 1, rate) < 0) goto done;
      if (av_audio_fifo_size(r->fifo) == 0) break;
    }
    // 標本1つより細かいずれは、頼まれた時刻に揃える（返す時刻が 1.4999937 のようにならないように）。
    if (fabs(r->fifo_time - time) < 1.0 / rate) r->fifo_time = time;
  }
  if (fill_fifo(r, cap, rate) < 0) goto done;
  int n = av_audio_fifo_size(r->fifo);
  if (n > cap) n = cap;
  if (n == 0) { result = 0; goto done; }
  void *planes[1] = { out };
  r->last_time = r->fifo_time < 0 ? 0 : r->fifo_time;
  av_audio_fifo_read(r->fifo, planes, n);
  r->fifo_time += (double)n / rate;
  result = n;
done:
  release(r);
  return result;
}

double um_media_last_time(int64_t h, int32_t line) {
  um_reader *r = acquire(&readers, KIND_READER, h, line);
  if (!r) return -1;
  double t = r->last_time;
  release(r);
  return t;
}

// ---- 書き手 ----

typedef struct {
  _Atomic int busy;
  AVFormatContext *oc;
  char *path;
  int header;  // ヘッダを書いた
  int failed;  // 書けなくなった（以後の書き込みは失敗にする）
  AVCodecContext *venc, *aenc;
  AVStream *vst, *ast;
  AVPacket *pkt;
  AVFrame *vframe;
  int64_t vpts;
  SwsContext *sws;
  AVFrame *src;  // 受け取った画素を包むフレーム（変換の入力）
  uint16_t *tmp16;
  size_t tmp16_cap;
  AVFrame *aframe;
  AVAudioFifo *afifo;  // 受け取った標本（交互の f32）
  SwrContext *aswr;    // 交互の f32 → 符号器の形式
  int64_t apts;
  int achannels;
  char vname[64], aname[64];
} um_writer;

static void free_writer(um_writer *w) {
  avcodec_free_context(&w->venc);
  avcodec_free_context(&w->aenc);
  if (w->oc) {
    if (w->oc->pb && !(w->oc->oformat->flags & AVFMT_NOFILE)) avio_closep(&w->oc->pb);
    avformat_free_context(w->oc);
  }
  av_packet_free(&w->pkt);
  av_frame_free(&w->vframe);
  av_frame_free(&w->aframe);
  sws_free_context(&w->sws);
  av_frame_free(&w->src);
  free(w->tmp16);
  if (w->afifo) av_audio_fifo_free(w->afifo);
  swr_free(&w->aswr);
  free(w->path);
  free(w);
}

int64_t um_media_create(const char *path, size_t path_len, int32_t line) {
  (void)line;
  begin();
  char *p = c_string(path, path_len);
  if (!p) { set_error("the path contains a zero byte"); return 0; }
  um_writer *w = calloc(1, sizeof *w);
  if (!w) um_panic(5, 0);
  w->path = p;
  int err = avformat_alloc_output_context2(&w->oc, NULL, NULL, p);
  if (err < 0 || !w->oc) {
    set_error("cannot tell the container format from the file name '%s' (use .mp4, .mov, .mkv, ...)", p);
    free_writer(w);
    return 0;
  }
  w->pkt = av_packet_alloc();
  if (!w->pkt) um_panic(5, 0);
  int64_t h = table_add(&writers, KIND_WRITER, w);
  if (!h) { set_error("too many open writers (%d)", MAX_SLOTS); free_writer(w); return 0; }
  return h;
}

// ハードウェアの符号器か（名前で判断する。FFmpeg の能力の印は付いていないものがある）。
static int encoder_is_hw(const AVCodec *c) {
  static const char *const parts[] = { "_nvenc", "_amf", "_qsv", "_videotoolbox", "_mf", "_vaapi", "_d3d12va", "_vulkan", "_v4l2m2m", "_mediacodec", NULL };
  if (c->capabilities & AV_CODEC_CAP_HARDWARE) return 1;
  for (int i = 0; parts[i]; i++) {
    if (strstr(c->name, parts[i])) return 1;
  }
  return 0;
}

// 符号器の優先順位（小さいほど先に試す）: ハードウェア（GPU の専用回路）→ ソフトウェア。
static int encoder_rank(const AVCodec *c) {
  static const char *const order[] = { "_nvenc", "_amf", "_qsv", "_videotoolbox", "_vaapi", "_mf", "_d3d12va", "_vulkan", NULL };
  for (int i = 0; order[i]; i++) {
    if (strstr(c->name, order[i])) return i;
  }
  if (encoder_is_hw(c)) return 20;
  if (c->capabilities & AV_CODEC_CAP_EXPERIMENTAL) return 40;
  return 30;
}

// 符号器が受け付ける画素の形式から選ぶ（10 ビットを望むなら 10 ビットの形式を先に）。
static enum AVPixelFormat pick_pix_fmt(AVCodecContext *ctx, const AVCodec *c, enum AVPixelFormat want) {
  const void *list = NULL;
  int n = 0;
  if (avcodec_get_supported_config(ctx, c, AV_CODEC_CONFIG_PIX_FORMAT, 0, &list, &n) < 0 || !list || n == 0) {
    return want != AV_PIX_FMT_NONE ? want : AV_PIX_FMT_YUV420P;
  }
  const enum AVPixelFormat *fmts = list;
  enum AVPixelFormat prefs8[] = { AV_PIX_FMT_YUV420P, AV_PIX_FMT_NV12, AV_PIX_FMT_YUV422P, AV_PIX_FMT_YUV444P, AV_PIX_FMT_NONE };
  enum AVPixelFormat prefs10[] = { AV_PIX_FMT_YUV420P10, AV_PIX_FMT_P010, AV_PIX_FMT_YUV422P10, AV_PIX_FMT_NONE };
  if (want != AV_PIX_FMT_NONE) {
    for (int i = 0; i < n; i++) if (fmts[i] == want) return want;
  }
  for (int k = 0; prefs8[k] != AV_PIX_FMT_NONE; k++) {
    for (int i = 0; i < n; i++) if (fmts[i] == prefs8[k]) return prefs8[k];
  }
  for (int k = 0; prefs10[k] != AV_PIX_FMT_NONE; k++) {
    for (int i = 0; i < n; i++) if (fmts[i] == prefs10[k]) return prefs10[k];
  }
  // ハードウェアの面（d3d11 など）しか受け付けない形式は飛ばす。
  for (int i = 0; i < n; i++) {
    const AVPixFmtDescriptor *d = av_pix_fmt_desc_get(fmts[i]);
    if (d && !(d->flags & AV_PIX_FMT_FLAG_HWACCEL)) return fmts[i];
  }
  return AV_PIX_FMT_NONE;
}

// 映像の符号器を1つ開いてみる。使えれば ctx を返す（理由は last_error）。
static AVCodecContext *open_video_encoder(const AVCodec *c, AVFormatContext *oc, int width, int height, int fps_num, int fps_den,
                                          int64_t bitrate, const char *options) {
  AVCodecContext *ctx = avcodec_alloc_context3(c);
  if (!ctx) um_panic(5, 0);
  AVDictionary *opts = NULL;
  if (options && options[0] && av_dict_parse_string(&opts, options, "=", ":", 0) < 0) {
    set_error("cannot read the encoder options '%s' (write key=value:key=value)", options);
    avcodec_free_context(&ctx);
    return NULL;
  }
  enum AVPixelFormat want = AV_PIX_FMT_NONE;
  const AVDictionaryEntry *pf = av_dict_get(opts, "pix_fmt", NULL, 0);
  if (pf) {
    want = av_get_pix_fmt(pf->value);
    if (want == AV_PIX_FMT_NONE) {
      set_error("unknown pix_fmt '%s'", pf->value);
      av_dict_free(&opts);
      avcodec_free_context(&ctx);
      return NULL;
    }
    av_dict_set(&opts, "pix_fmt", NULL, 0);
  }
  ctx->width = width;
  ctx->height = height;
  ctx->time_base = (AVRational){ fps_den, fps_num };
  ctx->framerate = (AVRational){ fps_num, fps_den };
  ctx->pix_fmt = pick_pix_fmt(ctx, c, want);
  if (ctx->pix_fmt == AV_PIX_FMT_NONE) {
    set_error("%s accepts no pixel format that can be written from memory", c->name);
    av_dict_free(&opts);
    avcodec_free_context(&ctx);
    return NULL;
  }
  if (want != AV_PIX_FMT_NONE && ctx->pix_fmt != want) {
    set_error("%s does not accept pix_fmt %s", c->name, av_get_pix_fmt_name(want));
    av_dict_free(&opts);
    avcodec_free_context(&ctx);
    return NULL;
  }
  // 0 なら大きさとコマの速さから決める（1080p30 で約 6 Mbps）。
  ctx->bit_rate = bitrate > 0 ? bitrate : (int64_t)((double)width * height * fps_num / fps_den * 0.1);
  ctx->gop_size = (int)(2.0 * fps_num / fps_den + 0.5);  // 2 秒ごとにキーフレーム（編集ソフトで読みやすく）
  if (ctx->gop_size < 1) ctx->gop_size = 1;
  ctx->max_b_frames = 2;
  ctx->sample_aspect_ratio = (AVRational){ 1, 1 };
  ctx->color_range = AVCOL_RANGE_MPEG;
  ctx->colorspace = AVCOL_SPC_BT709;
  ctx->color_primaries = AVCOL_PRI_BT709;
  ctx->color_trc = AVCOL_TRC_BT709;
  if (oc && (oc->oformat->flags & AVFMT_GLOBALHEADER)) ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
  int err = avcodec_open2(ctx, c, &opts);
  if (err < 0) {
    char what[160];
    snprintf(what, sizeof what, "cannot open the encoder %s", c->name);
    set_error_av(what, err);
    av_dict_free(&opts);
    avcodec_free_context(&ctx);
    return NULL;
  }
  const AVDictionaryEntry *left = av_dict_get(opts, "", NULL, AV_DICT_IGNORE_SUFFIX);
  if (left) {
    set_error("the encoder %s has no option '%s'", c->name, left->key);
    av_dict_free(&opts);
    avcodec_free_context(&ctx);
    return NULL;
  }
  av_dict_free(&opts);
  return ctx;
}

static enum AVSampleFormat pick_sample_fmt(AVCodecContext *ctx, const AVCodec *c) {
  const void *list = NULL;
  int n = 0;
  if (avcodec_get_supported_config(ctx, c, AV_CODEC_CONFIG_SAMPLE_FORMAT, 0, &list, &n) < 0 || !list || n == 0) return AV_SAMPLE_FMT_FLTP;
  const enum AVSampleFormat *fmts = list;
  enum AVSampleFormat prefs[] = { AV_SAMPLE_FMT_FLTP, AV_SAMPLE_FMT_FLT, AV_SAMPLE_FMT_S32P, AV_SAMPLE_FMT_S32, AV_SAMPLE_FMT_S16P, AV_SAMPLE_FMT_S16 };
  for (size_t k = 0; k < sizeof prefs / sizeof prefs[0]; k++) {
    for (int i = 0; i < n; i++) if (fmts[i] == prefs[k]) return prefs[k];
  }
  return fmts[0];
}

static AVCodecContext *open_audio_encoder(const AVCodec *c, AVFormatContext *oc, int rate, int channels, int64_t bitrate, const char *options) {
  AVCodecContext *ctx = avcodec_alloc_context3(c);
  if (!ctx) um_panic(5, 0);
  AVDictionary *opts = NULL;
  if (options && options[0] && av_dict_parse_string(&opts, options, "=", ":", 0) < 0) {
    set_error("cannot read the encoder options '%s' (write key=value:key=value)", options);
    avcodec_free_context(&ctx);
    return NULL;
  }
  ctx->sample_rate = rate;
  av_channel_layout_default(&ctx->ch_layout, channels);
  ctx->sample_fmt = pick_sample_fmt(ctx, c);
  ctx->bit_rate = bitrate > 0 ? bitrate : (c->id == AV_CODEC_ID_AAC || c->id == AV_CODEC_ID_MP3 ? 96000LL * channels : 0);
  ctx->time_base = (AVRational){ 1, rate };
  if (oc && (oc->oformat->flags & AVFMT_GLOBALHEADER)) ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
  int err = avcodec_open2(ctx, c, &opts);
  if (err < 0) {
    char what[160];
    snprintf(what, sizeof what, "cannot open the encoder %s", c->name);
    set_error_av(what, err);
    av_dict_free(&opts);
    avcodec_free_context(&ctx);
    return NULL;
  }
  const AVDictionaryEntry *left = av_dict_get(opts, "", NULL, AV_DICT_IGNORE_SUFFIX);
  if (left) {
    set_error("the encoder %s has no option '%s'", c->name, left->key);
    av_dict_free(&opts);
    avcodec_free_context(&ctx);
    return NULL;
  }
  av_dict_free(&opts);
  return ctx;
}

// 名前が符号器なら1つ、形式（h264 など）なら、その形式の符号器を優先順位の順に並べる（最大 32）。
static int encoder_candidates(const char *name, enum AVMediaType type, const AVCodec **out) {
  const AVCodec *exact = avcodec_find_encoder_by_name(name);
  if (exact && exact->type == type) {
    out[0] = exact;
    return 1;
  }
  const AVCodecDescriptor *d = avcodec_descriptor_get_by_name(name);
  if (!d || d->type != type) return 0;
  int n = 0;
  void *it = NULL;
  const AVCodec *c;
  while ((c = av_codec_iterate(&it)) && n < 32) {
    if (av_codec_is_encoder(c) && c->id == d->id) out[n++] = c;
  }
  // 優先順位で並べる（挿入ソート。数は少ない）。
  for (int i = 1; i < n; i++) {
    const AVCodec *x = out[i];
    int j = i - 1;
    while (j >= 0 && encoder_rank(out[j]) > encoder_rank(x)) { out[j + 1] = out[j]; j--; }
    out[j + 1] = x;
  }
  return n;
}

static um_writer *writer_for_setup(int64_t h, int32_t line) {
  um_writer *w = acquire(&writers, KIND_WRITER, h, line);
  if (!w) return NULL;
  if (w->header) {
    set_error("streams must be added before the first frame is written");
    release(w);
    return NULL;
  }
  return w;
}

int32_t um_media_add_video(int64_t h, const char *enc, size_t enc_len, int32_t width, int32_t height, int32_t fps_num, int32_t fps_den,
                           int64_t bitrate, const char *opt, size_t opt_len, int32_t line) {
  begin();
  um_writer *w = writer_for_setup(h, line);
  if (!w) return 0;
  int ok = 0;
  char *name = c_string(enc, enc_len);
  char *options = c_string(opt, opt_len);
  if (!name || !options) { set_error("the encoder name or options contain a zero byte"); goto done; }
  if (w->venc) { set_error("the writer already has a video stream"); goto done; }
  if (width <= 0 || height <= 0 || width > 16384 || height > 16384) { set_error("invalid video size %dx%d", width, height); goto done; }
  if (fps_num <= 0 || fps_den <= 0) { set_error("invalid frame rate %d/%d", fps_num, fps_den); goto done; }
  const AVCodec *cands[32];
  int n = encoder_candidates(name, AVMEDIA_TYPE_VIDEO, cands);
  if (n == 0) { set_error("no video encoder named '%s' in this FFmpeg build (see mediaEncoders)", name); goto done; }
  char first_error[1024] = "";
  for (int i = 0; i < n && !w->venc; i++) {
    if (avformat_query_codec(w->oc->oformat, cands[i]->id, FF_COMPLIANCE_NORMAL) == 0) {
      set_error("the %s container cannot hold %s", w->oc->oformat->name, avcodec_get_name(cands[i]->id));
    } else {
      w->venc = open_video_encoder(cands[i], w->oc, width, height, fps_num, fps_den, bitrate, options);
    }
    if (!w->venc && !first_error[0]) snprintf(first_error, sizeof first_error, "%s", last_error);
    if (w->venc) snprintf(w->vname, sizeof w->vname, "%s", cands[i]->name);
  }
  if (!w->venc) {
    if (n > 1) set_error("no %s encoder could be opened (first error: %s)", name, first_error);
    goto done;
  }
  w->vst = avformat_new_stream(w->oc, NULL);
  if (!w->vst) um_panic(5, 0);
  avcodec_parameters_from_context(w->vst->codecpar, w->venc);
  w->vst->time_base = w->venc->time_base;
  w->vst->avg_frame_rate = w->venc->framerate;
  w->vframe = av_frame_alloc();
  if (!w->vframe) um_panic(5, 0);
  w->vframe->format = w->venc->pix_fmt;
  w->vframe->width = width;
  w->vframe->height = height;
  if (av_frame_get_buffer(w->vframe, 0) < 0) um_panic(5, 0);
  ok = 1;
done:
  free(name);
  free(options);
  release(w);
  return ok;
}

int32_t um_media_add_audio(int64_t h, const char *enc, size_t enc_len, int32_t rate, int32_t channels, int64_t bitrate,
                           const char *opt, size_t opt_len, int32_t line) {
  begin();
  um_writer *w = writer_for_setup(h, line);
  if (!w) return 0;
  int ok = 0;
  char *name = c_string(enc, enc_len);
  char *options = c_string(opt, opt_len);
  if (!name || !options) { set_error("the encoder name or options contain a zero byte"); goto done; }
  if (w->aenc) { set_error("the writer already has an audio stream"); goto done; }
  if (rate < 1000 || rate > 768000) { set_error("invalid sample rate %d", rate); goto done; }
  if (channels < 1 || channels > 8) { set_error("invalid channel count %d (1 to 8)", channels); goto done; }
  const AVCodec *cands[32];
  int n = encoder_candidates(name, AVMEDIA_TYPE_AUDIO, cands);
  if (n == 0) { set_error("no audio encoder named '%s' in this FFmpeg build (see mediaEncoders)", name); goto done; }
  char first_error[1024] = "";
  for (int i = 0; i < n && !w->aenc; i++) {
    if (avformat_query_codec(w->oc->oformat, cands[i]->id, FF_COMPLIANCE_NORMAL) == 0) {
      set_error("the %s container cannot hold %s", w->oc->oformat->name, avcodec_get_name(cands[i]->id));
    } else {
      w->aenc = open_audio_encoder(cands[i], w->oc, rate, channels, bitrate, options);
    }
    if (!w->aenc && !first_error[0]) snprintf(first_error, sizeof first_error, "%s", last_error);
    if (w->aenc) snprintf(w->aname, sizeof w->aname, "%s", cands[i]->name);
  }
  if (!w->aenc) {
    if (n > 1) set_error("no %s encoder could be opened (first error: %s)", name, first_error);
    goto done;
  }
  w->ast = avformat_new_stream(w->oc, NULL);
  if (!w->ast) um_panic(5, 0);
  avcodec_parameters_from_context(w->ast->codecpar, w->aenc);
  w->ast->time_base = w->aenc->time_base;
  w->achannels = channels;
  w->afifo = av_audio_fifo_alloc(AV_SAMPLE_FMT_FLT, channels, 4096);
  w->aframe = av_frame_alloc();
  if (!w->afifo || !w->aframe) um_panic(5, 0);
  int err = swr_alloc_set_opts2(&w->aswr, &w->aenc->ch_layout, w->aenc->sample_fmt, rate, &w->aenc->ch_layout, AV_SAMPLE_FMT_FLT, rate, 0, NULL);
  if (err >= 0) err = swr_init(w->aswr);
  if (err < 0) { set_error_av("cannot set up the audio conversion", err); avcodec_free_context(&w->aenc); goto done; }
  ok = 1;
done:
  free(name);
  free(options);
  release(w);
  return ok;
}

static int write_header(um_writer *w) {
  if (w->header) return 1;
  if (!w->venc && !w->aenc) { set_error("the writer has no streams (call mediaAddVideo or mediaAddAudio first)"); return 0; }
  int err;
  if (!(w->oc->oformat->flags & AVFMT_NOFILE)) {
    err = avio_open(&w->oc->pb, w->path, AVIO_FLAG_WRITE);
    if (err < 0) { set_error_av(w->path, err); w->failed = 1; return 0; }
  }
  AVDictionary *opts = NULL;
  // MP4 / MOV は索引を先頭に置く（読み込みとシークが速くなる。書き終わりに1回写し直す）。
  if (strstr(w->oc->oformat->name, "mp4") || strstr(w->oc->oformat->name, "mov")) av_dict_set(&opts, "movflags", "+faststart", 0);
  err = avformat_write_header(w->oc, &opts);
  av_dict_free(&opts);
  if (err < 0) { set_error_av("cannot write the file header", err); w->failed = 1; return 0; }
  w->header = 1;
  return 1;
}

// 符号器から出てきたパケットをすべてファイルに書く。
static int drain(um_writer *w, AVCodecContext *enc, AVStream *st) {
  for (;;) {
    int err = avcodec_receive_packet(enc, w->pkt);
    if (err == AVERROR(EAGAIN) || err == AVERROR_EOF) return 1;
    if (err < 0) { set_error_av("cannot encode", err); w->failed = 1; return 0; }
    av_packet_rescale_ts(w->pkt, enc->time_base, st->time_base);
    w->pkt->stream_index = st->index;
    err = av_interleaved_write_frame(w->oc, w->pkt);
    if (err < 0) { set_error_av("cannot write", err); w->failed = 1; return 0; }
  }
}

static int encode(um_writer *w, AVCodecContext *enc, AVStream *st, AVFrame *f) {
  int err = avcodec_send_frame(enc, f);
  if (err < 0 && err != AVERROR_EOF) { set_error_av("cannot encode", err); w->failed = 1; return 0; }
  return drain(w, enc, st);
}

static um_writer *writer_for_write(int64_t h, int32_t line) {
  um_writer *w = acquire(&writers, KIND_WRITER, h, line);
  if (!w) return NULL;
  if (w->failed) {
    set_error("an earlier write failed; the file is incomplete");
    release(w);
    return NULL;
  }
  if (!write_header(w)) {
    release(w);
    return NULL;
  }
  return w;
}

int32_t um_media_write_video(int64_t h, const uint8_t *px, size_t len, size_t esize, int32_t scalar, int32_t fmt, int32_t width, int32_t height, int32_t line) {
  begin();
  um_writer *w = acquire(&writers, KIND_WRITER, h, line);
  if (!w) return 0;
  int ok = 0;
  if (!w->venc) { set_error("the writer has no video stream (call mediaAddVideo first)"); goto done; }
  release(w);
  w = writer_for_write(h, line);
  if (!w) return 0;
  if (width <= 0 || height <= 0 || width > 16384 || height > 16384) { set_error("invalid frame size %dx%d", width, height); goto done; }
  if (!px_bytes(fmt, width, height, scalar, len * esize)) goto done;
  const uint8_t *src = px;
  if (fmt == PX_RGBAF32) {
    size_t n = (size_t)width * height * 4;
    if (w->tmp16_cap < n) {
      free(w->tmp16);
      w->tmp16 = malloc(n * sizeof(uint16_t));
      if (!w->tmp16) um_panic(5, 0);
      w->tmp16_cap = n;
    }
    f32_to_rgba16((const float *)px, w->tmp16, n);
    src = (const uint8_t *)w->tmp16;
  }
  uint8_t *sdata[4];
  int slines[4];
  px_planes(fmt, (uint8_t *)src, width, height, sdata, slines);
  // RGB は 0〜255 いっぱいとして読み、BT.709 の限られた範囲の YUV にする（符号器にもそう印を付けてある）。
  // YUV の入力は BT.709 の限られた範囲とみなす。
  if (!w->src) {
    w->src = av_frame_alloc();
    if (!w->src) um_panic(5, 0);
  }
  AVFrame *in = w->src;
  in->width = width;
  in->height = height;
  in->format = px_av(fmt);
  for (int i = 0; i < 4; i++) {
    in->data[i] = sdata[i];
    in->linesize[i] = slines[i];
  }
  in->colorspace = px_is_rgb(fmt) ? AVCOL_SPC_RGB : AVCOL_SPC_BT709;
  in->color_range = px_is_rgb(fmt) ? AVCOL_RANGE_JPEG : AVCOL_RANGE_MPEG;
  in->color_primaries = AVCOL_PRI_BT709;
  in->color_trc = AVCOL_TRC_BT709;
  if (av_frame_make_writable(w->vframe) < 0) um_panic(5, 0);
  if (!scale_into(&w->sws, SWS_BICUBIC | SWS_FULL_CHR_H_INP | SWS_ACCURATE_RND, in, w->venc->pix_fmt, w->venc->width, w->venc->height,
                  w->vframe->data, w->vframe->linesize, AVCOL_SPC_BT709, AVCOL_RANGE_MPEG)) {
    goto done;
  }
  w->vframe->pts = w->vpts++;
  ok = encode(w, w->venc, w->vst, w->vframe);
done:
  release(w);
  return ok;
}

// fifo から1つの音のコマを符号器に送る。n は標本の組の数。
static int encode_audio_chunk(um_writer *w, int n) {
  AVFrame *f = w->aframe;
  av_frame_unref(f);
  f->nb_samples = n;
  f->format = w->aenc->sample_fmt;
  f->sample_rate = w->aenc->sample_rate;
  av_channel_layout_copy(&f->ch_layout, &w->aenc->ch_layout);
  if (av_frame_get_buffer(f, 0) < 0) um_panic(5, 0);
  float *tmp = malloc((size_t)n * w->achannels * sizeof(float));
  if (!tmp) um_panic(5, 0);
  void *tp[1] = { tmp };
  int got = av_audio_fifo_read(w->afifo, tp, n);
  if (got < n) memset(tmp + (size_t)got * w->achannels, 0, (size_t)(n - got) * w->achannels * sizeof(float));
  const uint8_t *in[1] = { (const uint8_t *)tmp };
  int conv = swr_convert(w->aswr, f->extended_data, n, in, n);
  free(tmp);
  if (conv < 0) { set_error_av("cannot convert the audio", conv); w->failed = 1; return 0; }
  f->pts = w->apts;
  w->apts += n;
  return encode(w, w->aenc, w->ast, f);
}

static int audio_frame_size(um_writer *w) {
  int fs = w->aenc->frame_size;
  return fs > 0 ? fs : 1024;
}

int32_t um_media_write_audio(int64_t h, const float *s, size_t len, size_t esize, int32_t scalar, int64_t count, int32_t line) {
  begin();
  um_writer *w = acquire(&writers, KIND_WRITER, h, line);
  if (!w) return 0;
  int ok = 0;
  if (!w->aenc) { set_error("the writer has no audio stream (call mediaAddAudio first)"); goto done; }
  release(w);
  w = writer_for_write(h, line);
  if (!w) return 0;
  if (scalar != SC_F32 || esize != 4) { set_error("audio samples need a Buffer<f32> or f32[]"); goto done; }
  size_t frames = len / (size_t)w->achannels;
  if (count >= 0) {
    if ((uint64_t)count > frames) { set_error("count %lld is more than the %zu samples in the Buffer", (long long)count, frames); goto done; }
    frames = (size_t)count;
  }
  while (frames > 0) {
    int n = frames > (1 << 20) ? (1 << 20) : (int)frames;
    void *planes[1] = { (void *)s };
    if (av_audio_fifo_write(w->afifo, planes, n) < n) um_panic(5, 0);
    s += (size_t)n * w->achannels;
    frames -= (size_t)n;
    int fs = audio_frame_size(w);
    while (av_audio_fifo_size(w->afifo) >= fs) {
      if (!encode_audio_chunk(w, fs)) goto done;
    }
  }
  ok = 1;
done:
  release(w);
  return ok;
}

// 残りを書き出してファイルを閉じる。書き手は閉じる（成功しても失敗しても）。
int32_t um_media_finish(int64_t h, int32_t line) {
  begin();
  um_writer *w = table_remove(&writers, KIND_WRITER, h, line);
  int ok = 0;
  if (w->failed) { set_error("an earlier write failed; the file is incomplete"); goto done; }
  if (!write_header(w)) goto done;
  if (w->aenc) {
    int left = av_audio_fifo_size(w->afifo);
    if (left > 0) {
      int fs = audio_frame_size(w);
      // 最後の短いコマを受け付けない符号器には、無音を足して1コマにする。
      int n = (w->aenc->codec->capabilities & (AV_CODEC_CAP_SMALL_LAST_FRAME | AV_CODEC_CAP_VARIABLE_FRAME_SIZE)) ? left : fs;
      if (!encode_audio_chunk(w, n)) goto done;
    }
    if (!encode(w, w->aenc, w->ast, NULL)) goto done;
  }
  if (w->venc && !encode(w, w->venc, w->vst, NULL)) goto done;
  int err = av_write_trailer(w->oc);
  if (err < 0) { set_error_av("cannot finish the file", err); goto done; }
  ok = 1;
done:
  free_writer(w);
  return ok;
}

// 書き手が使う符号器の名前（形式で頼んだときに、実際に選ばれたもの）。[映像, 音]。
void um_media_writer_text(int64_t h, um_bytes **out, size_t *len, int32_t line) {
  um_writer *w = acquire(&writers, KIND_WRITER, h, line);
  const char *t[2] = { w ? w->vname : "", w ? w->aname : "" };
  give_strings(t, 2, out, len);
  if (w) release(w);
}

// ---- 符号器の一覧 ----

// 一覧に載せる形式（編集ソフトの書き出しで使うもの）。
static const enum AVCodecID LISTED[] = {
  AV_CODEC_ID_H264, AV_CODEC_ID_HEVC, AV_CODEC_ID_AV1, AV_CODEC_ID_VP9, AV_CODEC_ID_VP8, AV_CODEC_ID_PRORES, AV_CODEC_ID_DNXHD,
  AV_CODEC_ID_MJPEG, AV_CODEC_ID_FFV1, AV_CODEC_ID_PNG,
  AV_CODEC_ID_AAC, AV_CODEC_ID_OPUS, AV_CODEC_ID_MP3, AV_CODEC_ID_FLAC, AV_CODEC_ID_ALAC, AV_CODEC_ID_VORBIS,
  AV_CODEC_ID_PCM_S16LE, AV_CODEC_ID_PCM_S24LE, AV_CODEC_ID_PCM_F32LE,
};

// 実際に開けるか試す（ハードウェアの符号器は、その GPU がなければ開けない）。
static int encoder_works(const AVCodec *c) {
  AVCodecContext *ctx;
  if (c->type == AVMEDIA_TYPE_VIDEO) ctx = open_video_encoder(c, NULL, 1280, 720, 30, 1, 0, "");
  else ctx = open_audio_encoder(c, NULL, 48000, 2, 0, "");
  if (!ctx) return 0;
  avcodec_free_context(&ctx);
  return 1;
}

// 符号器の一覧。1つの符号器は 5 つの文字列: 名前、"video" / "audio"、形式、説明、"hw" / "sw"。
// probe が真なら、実際に開けるものだけ（時間がかかる: ハードウェアの符号器ごとに数十〜数百ミリ秒）。
void um_media_encoders(int32_t probe, um_bytes **out, size_t *len) {
  begin();
  const char *items[5 * 256];
  size_t n = 0;
  void *it = NULL;
  const AVCodec *c;
  while ((c = av_codec_iterate(&it)) && n + 5 <= sizeof items / sizeof items[0]) {
    if (!av_codec_is_encoder(c)) continue;
    int listed = 0;
    for (size_t k = 0; k < sizeof LISTED / sizeof LISTED[0]; k++) listed |= c->id == LISTED[k];
    if (!listed) continue;
    if (probe && !encoder_works(c)) continue;
    items[n++] = c->name;
    items[n++] = c->type == AVMEDIA_TYPE_VIDEO ? "video" : "audio";
    items[n++] = avcodec_get_name(c->id);
    items[n++] = c->long_name ? c->long_name : "";
    items[n++] = encoder_is_hw(c) ? "hw" : "sw";
  }
  give_strings(items, n, out, len);
  last_error[0] = 0;
}

void um_media_error(unsigned char **out, size_t *len) { give_string(last_error, strlen(last_error), out, len); }

// FFmpeg のライブラリの版（"libavformat 62.3.100, libavcodec ..."）。
void um_media_version(unsigned char **out, size_t *len) {
  char buf[256];
  unsigned f = avformat_version(), c = avcodec_version(), u = avutil_version();
  snprintf(buf, sizeof buf, "libavformat %u.%u.%u, libavcodec %u.%u.%u, libavutil %u.%u.%u (%s)", f >> 16, (f >> 8) & 0xff, f & 0xff,
           c >> 16, (c >> 8) & 0xff, c & 0xff, u >> 16, (u >> 8) & 0xff, u & 0xff, avcodec_license());
  give_string(buf, strlen(buf), out, len);
}
