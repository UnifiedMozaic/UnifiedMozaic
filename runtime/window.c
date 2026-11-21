// ウィンドウの組み込み（windowOpen / windowClose / windowPoll / windowPresent）の実装。SDL3 を使う。
// プログラムがウィンドウの組み込みを使ったときだけ、ドライバが native.c と一緒にリンクする（SDL3 3.2.12 以降）。
// ウィンドウ・入力・表示を SDL3 にまかせるので、macOS・Windows・Linux で同じコードが動く。
//
// 決まりごと:
// - どの関数もメインスレッドから呼ぶ（macOS のウィンドウはメインスレッドでしか扱えないため）。違反は停止。
// - ウィンドウの番号は 1 から。0 は「開けなかった」（windowOpen は null を返す）。
// - 画素は u32 の 0xAARRGGBB（SDL の ARGB8888。メモリ上の並びはエンディアンによらず、u32 として読んだ値で決まる）。
// - イベントは Buffer<i32> の先頭 6 要素に [種類, ウィンドウ, a, b, c, d] で書く。種類の番号は prelude.um の
//   pollWindowEvent と揃えること。

#include <SDL3/SDL.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

_Noreturn void um_panic(int32_t code, int32_t line);
int32_t um_is_main(void);

// um_panic の原因コード（native.c の表と src/codegen.ts の並びと揃えること）。
#define UM_PANIC_WINDOW_THREAD 11  // メインスレッド以外から呼んだ
#define UM_PANIC_WINDOW_ARGS 12    // 番号・大きさ・Buffer の長さが正しくない

// イベントの種類（prelude.um の WINDOW_EVENT_* と揃える）。
enum {
  EV_CLOSE = 1,       // 閉じるボタン（a, b, c, d は 0）
  EV_RESIZE = 2,      // a = 幅, b = 高さ（ウィンドウの座標系）
  EV_MOUSE_MOVE = 3,  // a = x, b = y
  EV_MOUSE_DOWN = 4,  // a = x, b = y, c = ボタン（1 左, 2 中, 3 右）
  EV_MOUSE_UP = 5,    // 同上
  EV_WHEEL = 6,       // a = 横, b = 縦（ノッチ単位。上・右が正）
  EV_KEY_DOWN = 7,    // a = キー（SDL のキーコード）, b = 修飾キー, c = 押しっぱなしの繰り返しなら 1
  EV_KEY_UP = 8,      // a = キー, b = 修飾キー
  EV_TEXT = 9,        // a = 入力された文字（Unicode のコードポイント）
  EV_QUIT = 10,       // アプリの終了の要求（最後のウィンドウを閉じた、Cmd+Q など）
};
#define EV_FIELDS 6

// 修飾キー（prelude.um の MOD_* と揃える）。
#define MOD_SHIFT 1
#define MOD_CTRL 2
#define MOD_ALT 4
#define MOD_META 8  // macOS の Command、Windows のキー

#define MAX_WINDOWS 16

typedef struct {
  SDL_Window *win;
  SDL_Renderer *ren;
  SDL_Texture *tex;
  int tex_w, tex_h;
} um_window;

static um_window windows[MAX_WINDOWS];  // 番号 n は windows[n - 1]
static int sdl_ready;

// まだ返していない入力の文字（1つのテキスト入力のイベントに複数の文字が入ることがあるので、1文字ずつ返す）。
static struct {
  int32_t window;
  uint32_t cps[64];
  int n, next;
} pending_text;

static void main_only(int32_t line) {
  if (!um_is_main()) um_panic(UM_PANIC_WINDOW_THREAD, line);
}

static um_window *lookup(int32_t id, int32_t line) {
  if (id < 1 || id > MAX_WINDOWS || !windows[id - 1].win) um_panic(UM_PANIC_WINDOW_ARGS, line);
  return &windows[id - 1];
}

// SDL のウィンドウ ID → 番号（閉じたウィンドウや知らないウィンドウなら 0）。
static int32_t number_of(SDL_WindowID wid) {
  for (int i = 0; i < MAX_WINDOWS; i++) {
    if (windows[i].win && SDL_GetWindowID(windows[i].win) == wid) return i + 1;
  }
  return 0;
}

int32_t um_window_open(const char *title, size_t title_len, int32_t width, int32_t height, int32_t line) {
  main_only(line);
  if (width <= 0 || height <= 0) um_panic(UM_PANIC_WINDOW_ARGS, line);
  if (!sdl_ready) {
    if (!SDL_Init(SDL_INIT_VIDEO)) return 0;
    sdl_ready = 1;
  }
  int slot = -1;
  for (int i = 0; i < MAX_WINDOWS; i++) {
    if (!windows[i].win) { slot = i; break; }
  }
  if (slot < 0) return 0;
  // UM の文字列は NUL で終わらないので、写して終端を付ける。
  char name[256];
  size_t n = title_len < sizeof name - 1 ? title_len : sizeof name - 1;
  if (n) memcpy(name, title, n);
  name[n] = 0;
  SDL_Window *win = NULL;
  SDL_Renderer *ren = NULL;
  if (!SDL_CreateWindowAndRenderer(name, width, height, SDL_WINDOW_RESIZABLE, &win, &ren)) return 0;
  SDL_SetRenderVSync(ren, 1);  // 表示を画面の更新に合わせる（毎フレーム描き直すループが CPU を使い切らないように）
  SDL_StartTextInput(win);
  windows[slot] = (um_window){ win, ren, NULL, 0, 0 };
  return slot + 1;
}

void um_window_close(int32_t id, int32_t line) {
  main_only(line);
  um_window *w = lookup(id, line);
  if (w->tex) SDL_DestroyTexture(w->tex);
  SDL_DestroyRenderer(w->ren);
  SDL_DestroyWindow(w->win);
  memset(w, 0, sizeof *w);
  if (pending_text.window == id) pending_text.n = pending_text.next = 0;
}

void um_window_present(int32_t id, const uint32_t *pixels, size_t len, int32_t width, int32_t height, int32_t line) {
  main_only(line);
  um_window *w = lookup(id, line);
  if (width <= 0 || height <= 0 || (uint64_t)width * (uint64_t)height > (uint64_t)len) um_panic(UM_PANIC_WINDOW_ARGS, line);
  if (!w->tex || w->tex_w != width || w->tex_h != height) {
    if (w->tex) SDL_DestroyTexture(w->tex);
    w->tex = SDL_CreateTexture(w->ren, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, width, height);
    if (!w->tex) { w->tex_w = w->tex_h = 0; return; }
    w->tex_w = width;
    w->tex_h = height;
  }
  SDL_UpdateTexture(w->tex, NULL, pixels, width * 4);
  SDL_RenderClear(w->ren);
  SDL_RenderTexture(w->ren, w->tex, NULL, NULL);  // ウィンドウ全体に引き伸ばす
  SDL_RenderPresent(w->ren);
}

static int32_t mods_of(SDL_Keymod m) {
  int32_t r = 0;
  if (m & SDL_KMOD_SHIFT) r |= MOD_SHIFT;
  if (m & SDL_KMOD_CTRL) r |= MOD_CTRL;
  if (m & SDL_KMOD_ALT) r |= MOD_ALT;
  if (m & SDL_KMOD_GUI) r |= MOD_META;
  return r;
}

// UTF-8 を読んで pending_text に積む。
static void queue_text(int32_t window, const char *s) {
  pending_text.window = window;
  pending_text.n = pending_text.next = 0;
  const unsigned char *p = (const unsigned char *)s;
  while (*p && pending_text.n < 64) {
    uint32_t cp;
    int extra;
    if (*p < 0x80) { cp = *p; extra = 0; }
    else if ((*p & 0xE0) == 0xC0) { cp = *p & 0x1F; extra = 1; }
    else if ((*p & 0xF0) == 0xE0) { cp = *p & 0x0F; extra = 2; }
    else if ((*p & 0xF8) == 0xF0) { cp = *p & 0x07; extra = 3; }
    else { p++; continue; }  // 壊れたバイトは飛ばす
    p++;
    int ok = 1;
    for (int i = 0; i < extra; i++) {
      if ((*p & 0xC0) != 0x80) { ok = 0; break; }
      cp = (cp << 6) | (*p++ & 0x3F);
    }
    if (ok) pending_text.cps[pending_text.n++] = cp;
  }
}

static int take_text(int32_t *out) {
  if (pending_text.next >= pending_text.n) return 0;
  out[0] = EV_TEXT;
  out[1] = pending_text.window;
  out[2] = (int32_t)pending_text.cps[pending_text.next++];
  return 1;
}

// SDL のイベントを [種類, ウィンドウ, a, b, c, d] にする。扱わないイベントなら 0。
static int translate(const SDL_Event *e, int32_t *out) {
  switch (e->type) {
    case SDL_EVENT_QUIT:
      out[0] = EV_QUIT;
      return 1;
    case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
      out[0] = EV_CLOSE;
      out[1] = number_of(e->window.windowID);
      return out[1] != 0;
    case SDL_EVENT_WINDOW_RESIZED:
      out[0] = EV_RESIZE;
      out[1] = number_of(e->window.windowID);
      out[2] = e->window.data1;
      out[3] = e->window.data2;
      return out[1] != 0;
    case SDL_EVENT_MOUSE_MOTION:
      out[0] = EV_MOUSE_MOVE;
      out[1] = number_of(e->motion.windowID);
      out[2] = (int32_t)e->motion.x;
      out[3] = (int32_t)e->motion.y;
      return out[1] != 0;
    case SDL_EVENT_MOUSE_BUTTON_DOWN:
    case SDL_EVENT_MOUSE_BUTTON_UP:
      out[0] = e->type == SDL_EVENT_MOUSE_BUTTON_DOWN ? EV_MOUSE_DOWN : EV_MOUSE_UP;
      out[1] = number_of(e->button.windowID);
      out[2] = (int32_t)e->button.x;
      out[3] = (int32_t)e->button.y;
      out[4] = e->button.button;
      return out[1] != 0;
    case SDL_EVENT_MOUSE_WHEEL: {
      int32_t flip = e->wheel.direction == SDL_MOUSEWHEEL_FLIPPED ? -1 : 1;
      out[0] = EV_WHEEL;
      out[1] = number_of(e->wheel.windowID);
      out[2] = e->wheel.integer_x * flip;
      out[3] = e->wheel.integer_y * flip;
      return out[1] != 0 && (out[2] != 0 || out[3] != 0);
    }
    case SDL_EVENT_KEY_DOWN:
    case SDL_EVENT_KEY_UP:
      out[0] = e->type == SDL_EVENT_KEY_DOWN ? EV_KEY_DOWN : EV_KEY_UP;
      out[1] = number_of(e->key.windowID);
      out[2] = (int32_t)e->key.key;
      out[3] = mods_of(e->key.mod);
      out[4] = e->key.repeat ? 1 : 0;
      return out[1] != 0;
    case SDL_EVENT_TEXT_INPUT: {
      int32_t id = number_of(e->text.windowID);
      if (!id || !e->text.text) return 0;
      queue_text(id, e->text.text);
      return take_text(out);
    }
    default:
      return 0;
  }
}

int32_t um_window_poll(int32_t *event, size_t len, int32_t timeout_ms, int32_t line) {
  main_only(line);
  if (len < EV_FIELDS) um_panic(UM_PANIC_WINDOW_ARGS, line);
  memset(event, 0, EV_FIELDS * sizeof *event);
  if (take_text(event)) return 1;
  if (!sdl_ready) return 0;  // まだウィンドウを開いていない
  Uint64 deadline = timeout_ms > 0 ? SDL_GetTicks() + (Uint64)timeout_ms : 0;
  for (;;) {
    SDL_Event e;
    int got;
    if (timeout_ms == 0) {
      got = SDL_PollEvent(&e);
    } else if (timeout_ms < 0) {
      got = SDL_WaitEvent(&e);
    } else {
      Uint64 now = SDL_GetTicks();
      if (now >= deadline) got = SDL_PollEvent(&e);
      else got = SDL_WaitEventTimeout(&e, (Sint32)(deadline - now));
    }
    if (!got) return 0;
    memset(event, 0, EV_FIELDS * sizeof *event);
    if (translate(&e, event)) return 1;
    // 扱わないイベントは捨てて次を見る（待つ場合は、残りの時間だけ待つ）。
  }
}
