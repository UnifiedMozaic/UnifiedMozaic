// 音の出力の組み込みの実装。SDL3 の audio stream を使う。runtime/window.c と一緒にリンクする。
//   audioWrite(stream, samples: Buffer<f32>, start, count)  標本を書き込む
//   windowCtl の操作番号 100 番台                            開く・閉じる・問い合わせ・一時停止・再開・消去
// 名前のある関数（audioOpen など）は runtime/window.um にある。
//
// 決まりごと:
// - 出力の番号は 1 から。0 は「開けなかった」（音のデバイスがない、など。audioOpen は null を返す）。
// - 標本は f32（-1.0 〜 1.0）で、チャンネルを交互に並べる（ステレオなら L R L R ...）。1 フレーム = チャンネルの数の標本。
// - 開く・閉じるはメインスレッドから。書き込みと問い合わせはどのスレッドからも呼べる（SDL の audio stream は
//   内部で排他する）。ただし同じ出力を閉じながら別のスレッドで使ってはいけない。
// - 時計: 「再生した」フレーム数 = 書き込んだフレーム数 − まだ待ち行列にあるフレーム数。デバイスの中の分
//   （AC_LATENCY）だけ、実際に耳に届くのはさらに遅れる。

#include <SDL3/SDL.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

_Noreturn void um_panic(int32_t code, int32_t line);
int32_t um_is_main(void);

#define UM_PANIC_WINDOW_THREAD 11
#define UM_PANIC_WINDOW_ARGS 12

// 操作番号（window.um と揃える）。stream は windowCtl の window の引数で渡す。
enum {
  AC_OPEN = 100,     // io[0] = 標本化周波数, io[1] = チャンネルの数 → 出力の番号（0 は開けなかった）。一時停止の状態で開く
  AC_CLOSE = 101,
  AC_QUEUED = 102,   // → 待ち行列にあるフレームの数
  AC_PAUSE = 103,
  AC_RESUME = 104,
  AC_CLEAR = 105,    // 待ち行列を捨てる（捨てた分は「再生した」に数えない）
  AC_PLAYED = 106,   // → 再生した（デバイスに渡った）フレームの数。開いてからの合計
  AC_PAUSED = 107,   // → 一時停止中なら 1
  AC_LATENCY = 108,  // → デバイスの中にたまる分のフレームの数（出力の標本化周波数に換算。不明なら 0）
  AC_WRITTEN = 109,  // → 書き込んだフレームの数。開いてからの合計（消去した分は引く）
  AC_GAIN = 110,     // io[0] = 音量 × 1000（1000 で元のまま）
};

#define MAX_STREAMS 8

typedef struct {
  SDL_AudioStream *s;
  int channels;
  int rate;
  int64_t written;  // 書き込んだフレームの数（消去した分は引く）
} um_audio;

static um_audio streams[MAX_STREAMS];  // 番号 n は streams[n - 1]
static int audio_ready;

static um_audio *lookup(int32_t id, int32_t line) {
  if (id < 1 || id > MAX_STREAMS || !streams[id - 1].s) um_panic(UM_PANIC_WINDOW_ARGS, line);
  return &streams[id - 1];
}

static int64_t queued_frames(um_audio *a) {
  int bytes = SDL_GetAudioStreamQueued(a->s);
  return bytes > 0 ? bytes / (int)(sizeof(float) * (size_t)a->channels) : 0;
}

static int32_t audio_open(int64_t rate, int64_t channels, int32_t line) {
  if (!um_is_main()) um_panic(UM_PANIC_WINDOW_THREAD, line);
  if (rate <= 0 || rate > 768000 || channels < 1 || channels > 8) um_panic(UM_PANIC_WINDOW_ARGS, line);
  if (!audio_ready) {
    if (!SDL_InitSubSystem(SDL_INIT_AUDIO)) return 0;
    audio_ready = 1;
  }
  int slot = -1;
  for (int i = 0; i < MAX_STREAMS; i++) {
    if (!streams[i].s) { slot = i; break; }
  }
  if (slot < 0) return 0;
  SDL_AudioSpec spec = { SDL_AUDIO_F32, (int)channels, (int)rate };
  SDL_AudioStream *s = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec, NULL, NULL);
  if (!s) return 0;  // 音のデバイスがない、など（理由は windowError）
  streams[slot] = (um_audio){ s, (int)channels, (int)rate, 0 };
  return slot + 1;
}

// audioWrite: samples[start .. start + count) を書き込む。count が負なら終わりまで。count はチャンネルの数の倍数。
int32_t um_audio_write(int32_t id, const float *samples, size_t len, int64_t start, int64_t count, int32_t line) {
  um_audio *a = lookup(id, line);
  if (start < 0 || (uint64_t)start > (uint64_t)len) um_panic(UM_PANIC_WINDOW_ARGS, line);
  if (count < 0) count = (int64_t)len - start;
  if ((uint64_t)count > (uint64_t)len - (uint64_t)start || count % a->channels != 0) um_panic(UM_PANIC_WINDOW_ARGS, line);
  if (count == 0) return 1;
  // UM_WGPU=1 のビルドでは、GPU が書いた標本をまだ CPU に読み戻していないことがある（runtime/wgpu.c）。先に1回読んで済ませる。
  (void)*(volatile const float *)(samples + start);
  // SDL は int のバイト数で受け取るので、大きいときは分けて書く。
  // 書き込みと数の更新を1つの錠の中で行う（ほかのスレッドの AC_PLAYED が途中の値を見ないように。錠は再帰できる）。
  const int64_t chunk = (int64_t)a->channels * 1048576;
  int ok = 1;
  SDL_LockAudioStream(a->s);
  for (int64_t done = 0; done < count && ok; done += chunk) {
    int64_t n = count - done < chunk ? count - done : chunk;
    ok = SDL_PutAudioStreamData(a->s, samples + start + done, (int)(n * (int64_t)sizeof(float)));
    if (ok) a->written += n / a->channels;
  }
  SDL_UnlockAudioStream(a->s);
  return ok;
}

int64_t um_audio_ctl(int32_t op, int32_t id, int64_t *io, size_t len, int32_t line) {
  if (op == AC_OPEN) {
    if (len < 2) um_panic(UM_PANIC_WINDOW_ARGS, line);
    return audio_open(io[0], io[1], line);
  }
  um_audio *a = lookup(id, line);
  switch (op) {
    case AC_CLOSE:
      if (!um_is_main()) um_panic(UM_PANIC_WINDOW_THREAD, line);
      SDL_DestroyAudioStream(a->s);
      memset(a, 0, sizeof *a);
      return 1;
    case AC_QUEUED: return queued_frames(a);
    case AC_PAUSE: return SDL_PauseAudioStreamDevice(a->s);
    case AC_RESUME: return SDL_ResumeAudioStreamDevice(a->s);
    case AC_CLEAR: {
      // 捨てる分を書き込んだ数から引く（再生した位置は動かない）。
      SDL_LockAudioStream(a->s);
      a->written -= queued_frames(a);
      int ok = SDL_ClearAudioStream(a->s);
      SDL_UnlockAudioStream(a->s);
      return ok;
    }
    case AC_PLAYED: {
      SDL_LockAudioStream(a->s);
      int64_t r = a->written - queued_frames(a);
      SDL_UnlockAudioStream(a->s);
      return r < 0 ? 0 : r;
    }
    case AC_PAUSED: return SDL_AudioStreamDevicePaused(a->s);
    case AC_LATENCY: {
      SDL_AudioSpec dev;
      int frames = 0;
      if (!SDL_GetAudioDeviceFormat(SDL_GetAudioStreamDevice(a->s), &dev, &frames) || dev.freq <= 0) return 0;
      return (int64_t)frames * a->rate / dev.freq;
    }
    case AC_WRITTEN: {
      SDL_LockAudioStream(a->s);
      int64_t r = a->written;
      SDL_UnlockAudioStream(a->s);
      return r;
    }
    case AC_GAIN:
      if (len < 1 || io[0] < 0) um_panic(UM_PANIC_WINDOW_ARGS, line);
      return SDL_SetAudioStreamGain(a->s, (float)io[0] / 1000.0f);
  }
  um_panic(UM_PANIC_WINDOW_ARGS, line);
}
