// ウィンドウの組み込みの実装。SDL3 を使う。
//   windowOpen / windowClose / windowPoll / windowPresent / windowPresentRect / windowUpdateRect
//   操作番号で呼び分ける windowCtl / windowCtlF / windowSetText / windowGetText（名前のある関数は runtime/window.um）
// 音の出力は runtime/audio.c にあり、windowCtl の操作番号 100 番台から呼ぶ。
// プログラムがウィンドウか音の組み込みを使ったときだけ、ドライバが native.c と一緒にリンクする（SDL3 3.2.12 以降）。
// ウィンドウ・入力・表示を SDL3 にまかせるので、macOS・Windows・Linux で同じコードが動く。
//
// 決まりごと:
// - どの関数もメインスレッドから呼ぶ（macOS のウィンドウはメインスレッドでしか扱えないため）。違反は停止。
//   例外は時計と眠り（WC_TICKS_NS / WC_SLEEP_NS / WF_OP_SECONDS）、SDL の失敗の理由、音の書き込みと問い合わせ。
// - ウィンドウの番号は 1 から。0 は「開けなかった」（windowOpen は null を返す）。
// - 画素は u32 の 0xAARRGGBB（SDL の ARGB8888。メモリ上の並びはエンディアンによらず、u32 として読んだ値で決まる）。
// - イベントは Buffer<i32> の先頭 6 要素に [種類, ウィンドウ, a, b, c, d] で書く。種類の番号は window.um の
//   pollWindowEvent と揃えること。文字列を持つイベントの文字列は windowGetText(WG_EVENT_STRING, 0, i) で読む。

#include <SDL3/SDL.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

_Noreturn void um_panic(int32_t code, int32_t line);
int32_t um_is_main(void);
void *um_alloc(size_t count, size_t elem_size);

// um_panic の原因コード（native.c の表と src/codegen.ts の並びと揃えること）。
#define UM_PANIC_WINDOW_THREAD 11  // メインスレッド以外から呼んだ
#define UM_PANIC_WINDOW_ARGS 12    // 番号・大きさ・Buffer の長さ・操作番号が正しくない

// イベントの種類（window.um の pollWindowEvent と揃える）。
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
  EV_DROP_BEGIN = 11,     // ドロップの始まり
  EV_DROP_FILE = 12,      // a = x, b = y, 文字列 0 = パス（フォルダのこともある）
  EV_DROP_TEXT = 13,      // a = x, b = y, 文字列 0 = 文字
  EV_DROP_POSITION = 14,  // a = x, b = y（ドラッグ中の位置）
  EV_DROP_COMPLETE = 15,  // a = x, b = y（ドロップの終わり）
  EV_TEXT_EDITING = 16,   // a = 選択の始まり, b = 選択の長さ（文字単位。なければ -1）, 文字列 0 = 編集中の文字列
  EV_DIALOG = 17,         // ウィンドウ = 要求の番号, a = パスの数（-1 は失敗）, b = 選んだ絞り込みの番号（-1 は不明）, 文字列 = パス
  EV_PIXEL_SIZE = 18,     // a = 幅, b = 高さ（描画の面の画素）
  EV_SCALE = 19,          // ウィンドウの表示の倍率が変わった
  EV_FOCUS = 20,          // a = 1 なら入力の焦点を得た、0 なら失った
  EV_MOUSE_ENTER = 21,    // a = 1 ならマウスが入った、0 なら出た
  EV_MOVED = 22,          // a = x, b = y（画面の座標）
  EV_STATE = 23,          // 最大化・最小化・元に戻す・全画面の出入り。a = 状態のビット（WF_*）
  EV_EXPOSED = 24,        // 描き直しが要る（隠れていた部分が見えた、描画の装置が作り直されたなど）
  EV_DISPLAYS = 25,       // ディスプレイの増減・配置・倍率の変化（ウィンドウは 0）
};
#define EV_FIELDS 6

// 修飾キー（window.um の modsOf と揃える）。
#define MOD_SHIFT 1
#define MOD_CTRL 2
#define MOD_ALT 4
#define MOD_META 8  // macOS の Command、Windows のキー

// ウィンドウの状態のビット（window.um の stateOf と揃える）。
#define WF_FULLSCREEN 1
#define WF_MAXIMIZED 2
#define WF_MINIMIZED 4
#define WF_FOCUSED 8
#define WF_HIDDEN 16

#define MAX_WINDOWS 16

typedef struct {
  SDL_Window *win;
  SDL_Renderer *ren;
  SDL_Texture *tex;
  int tex_w, tex_h;
  int tex_valid;  // テクスチャの中身が前回の表示のまま残っているか（windowPresentRect は残りを使う）
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

// 映像（とイベント）の仕組みを用意する。ウィンドウを開く前のダイアログやクリップボードでも使う。
static int ensure_video(void) {
  if (!sdl_ready) {
    if (!SDL_Init(SDL_INIT_VIDEO)) return 0;
    sdl_ready = 1;
  }
  return 1;
}

// UM の文字列（NUL で終わらない）を、SDL_free で解放する C の文字列にする。
static char *cstr(const char *s, size_t n) {
  char *r = SDL_malloc(n + 1);
  if (!r) return NULL;
  if (n) memcpy(r, s, n);
  r[n] = 0;
  return r;
}

int32_t um_window_open(const char *title, size_t title_len, int32_t width, int32_t height, int32_t line) {
  main_only(line);
  if (width <= 0 || height <= 0) um_panic(UM_PANIC_WINDOW_ARGS, line);
  if (!ensure_video()) return 0;
  int slot = -1;
  for (int i = 0; i < MAX_WINDOWS; i++) {
    if (!windows[i].win) { slot = i; break; }
  }
  if (slot < 0) return 0;
  char *name = cstr(title, title_len);
  if (!name) return 0;
  SDL_Window *win = NULL;
  SDL_Renderer *ren = NULL;
  // 高密度の画面（Retina など）では、描画の面を画素の大きさで作る（座標はウィンドウの座標のまま）。
  int ok = SDL_CreateWindowAndRenderer(name, width, height, SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY, &win, &ren);
  SDL_free(name);
  if (!ok) return 0;
  SDL_SetRenderVSync(ren, 1);  // 表示を画面の更新に合わせる（毎フレーム描き直すループが CPU を使い切らないように）
  SDL_StartTextInput(win);
  windows[slot] = (um_window){ win, ren, NULL, 0, 0, 0 };
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

// ---- 表示 ----

// テクスチャを width × height に合わせる（作り直したら中身は空なので tex_valid を 0 にする）。
static int fit_texture(um_window *w, int32_t width, int32_t height) {
  if (!w->tex || w->tex_w != width || w->tex_h != height) {
    if (w->tex) SDL_DestroyTexture(w->tex);
    w->tex = SDL_CreateTexture(w->ren, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, width, height);
    w->tex_valid = 0;
    if (!w->tex) { w->tex_w = w->tex_h = 0; return 0; }
    w->tex_w = width;
    w->tex_h = height;
  }
  return 1;
}

static void show_texture(um_window *w) {
  SDL_RenderClear(w->ren);
  SDL_RenderTexture(w->ren, w->tex, NULL, NULL);  // ウィンドウ全体に引き伸ばす
  SDL_RenderPresent(w->ren);
}

static void check_pixels(size_t len, int32_t width, int32_t height, int32_t line) {
  if (width <= 0 || height <= 0 || (uint64_t)width * (uint64_t)height > (uint64_t)len) um_panic(UM_PANIC_WINDOW_ARGS, line);
}

void um_window_present(int32_t id, const uint32_t *pixels, size_t len, int32_t width, int32_t height, int32_t line) {
  main_only(line);
  um_window *w = lookup(id, line);
  check_pixels(len, width, height, line);
  if (!fit_texture(w, width, height)) return;
  SDL_UpdateTexture(w->tex, NULL, pixels, width * 4);
  w->tex_valid = 1;
  show_texture(w);
}

// 変わった矩形 (x, y, rw, rh) だけをテクスチャに写し、show が 1 なら表示する（windowPresentRect / windowUpdateRect）。
// テクスチャの残りは前回の表示のまま。大きさが変わった・描画の装置が作り直されたなどで残りがないときは、全体を写す。
void um_window_present_rect(int32_t id, const uint32_t *pixels, size_t len, int32_t width, int32_t height,
                            int32_t x, int32_t y, int32_t rw, int32_t rh, int32_t show, int32_t line) {
  main_only(line);
  um_window *w = lookup(id, line);
  check_pixels(len, width, height, line);
  if (x < 0 || y < 0 || rw < 0 || rh < 0 || (int64_t)x + rw > width || (int64_t)y + rh > height) um_panic(UM_PANIC_WINDOW_ARGS, line);
  if (!fit_texture(w, width, height)) return;
  if (!w->tex_valid) {
    SDL_UpdateTexture(w->tex, NULL, pixels, width * 4);
    w->tex_valid = 1;
  } else if (rw > 0 && rh > 0) {
    SDL_Rect r = { x, y, rw, rh };
    SDL_UpdateTexture(w->tex, &r, pixels + (size_t)y * (size_t)width + (size_t)x, width * 4);
  }
  if (show) show_texture(w);
}

// ---- イベントの文字列 ----
// 最後に windowPoll が返したイベントの文字列（ドロップのパス、IME の編集中の文字列、ダイアログで選んだパス）。
// 次の windowPoll で捨てる。
static char **ev_strings;
static int ev_nstrings;

static void clear_event_strings(void) {
  for (int i = 0; i < ev_nstrings; i++) SDL_free(ev_strings[i]);
  SDL_free(ev_strings);
  ev_strings = NULL;
  ev_nstrings = 0;
}

// s を写してイベントの文字列に足す（s が NULL なら空の文字列）。
static void add_event_string(const char *s) {
  char **grown = SDL_realloc(ev_strings, (size_t)(ev_nstrings + 1) * sizeof *grown);
  if (!grown) return;
  ev_strings = grown;
  char *copy = SDL_strdup(s ? s : "");
  if (copy) ev_strings[ev_nstrings++] = copy;
}

// ---- ファイルのダイアログ ----
// SDL のダイアログは非同期で、結果はほかのスレッドから届くことがある。届いた結果は自前のイベント（user event）に
// 入れて SDL のイベントの列に積み、windowPoll が EV_DIALOG にする。

typedef struct {
  int32_t request;
  int count;  // -1 は失敗
  int filter;
  char **paths;
  SDL_DialogFileFilter *filters;  // SDL がコールバックを呼ぶまで生かしておく
  int nfilters;
} dialog_result;

static Uint32 dialog_event_type;  // SDL_RegisterEvents で得た番号（0 はまだ）
static int32_t next_request = 1;

// 次に開くダイアログの設定（windowSetText で貯め、開いたら空にする）。
static struct {
  char *title, *location, *accept;
  SDL_DialogFileFilter *filters;
  int nfilters;
} staged;

static void free_filters(SDL_DialogFileFilter *f, int n) {
  for (int i = 0; i < n; i++) {
    SDL_free((void *)f[i].name);
    SDL_free((void *)f[i].pattern);
  }
  SDL_free(f);
}

static void clear_staged(void) {
  SDL_free(staged.title);
  SDL_free(staged.location);
  SDL_free(staged.accept);
  free_filters(staged.filters, staged.nfilters);
  memset(&staged, 0, sizeof staged);
}

static void free_result(dialog_result *r) {
  for (int i = 0; r->paths && i < r->count; i++) SDL_free(r->paths[i]);
  SDL_free(r->paths);
  free_filters(r->filters, r->nfilters);
  SDL_free(r);
}

static void SDLCALL dialog_done(void *userdata, const char *const *filelist, int filter) {
  dialog_result *r = userdata;
  r->filter = filter;
  r->count = -1;
  if (filelist) {
    int n = 0;
    while (filelist[n]) n++;
    r->paths = SDL_calloc((size_t)n + 1, sizeof *r->paths);
    if (r->paths) {
      r->count = 0;
      for (int i = 0; i < n; i++) {
        char *p = SDL_strdup(filelist[i]);
        if (p) r->paths[r->count++] = p;
      }
    }
  }
  SDL_Event e;
  SDL_zero(e);
  e.type = dialog_event_type;
  e.user.data1 = r;
  if (!SDL_PushEvent(&e)) free_result(r);  // 積めなかった（終了中など）。結果は捨てる
}

// kind: 0 開く、1 保存、2 フォルダ。window は 0（親なし）か開いているウィンドウ。要求の番号（1 から）を返す。0 は開けなかった。
static int32_t show_dialog(int kind, int32_t window, int many, int32_t line) {
  SDL_Window *parent = window ? lookup(window, line)->win : NULL;
  if (!ensure_video()) { clear_staged(); return 0; }
  if (!dialog_event_type) {
    dialog_event_type = SDL_RegisterEvents(1);
    if (!dialog_event_type) { clear_staged(); return 0; }
  }
  dialog_result *r = SDL_calloc(1, sizeof *r);
  if (!r) { clear_staged(); return 0; }
  r->request = next_request++;
  r->filters = staged.filters;
  r->nfilters = staged.nfilters;
  staged.filters = NULL;
  staged.nfilters = 0;
  SDL_PropertiesID props = SDL_CreateProperties();
  if (r->nfilters && kind != 2) {
    SDL_SetPointerProperty(props, SDL_PROP_FILE_DIALOG_FILTERS_POINTER, r->filters);
    SDL_SetNumberProperty(props, SDL_PROP_FILE_DIALOG_NFILTERS_NUMBER, r->nfilters);
  }
  if (parent) SDL_SetPointerProperty(props, SDL_PROP_FILE_DIALOG_WINDOW_POINTER, parent);
  if (staged.location) SDL_SetStringProperty(props, SDL_PROP_FILE_DIALOG_LOCATION_STRING, staged.location);
  if (staged.title) SDL_SetStringProperty(props, SDL_PROP_FILE_DIALOG_TITLE_STRING, staged.title);
  if (staged.accept) SDL_SetStringProperty(props, SDL_PROP_FILE_DIALOG_ACCEPT_STRING, staged.accept);
  SDL_SetBooleanProperty(props, SDL_PROP_FILE_DIALOG_MANY_BOOLEAN, many != 0);
  SDL_FileDialogType type = kind == 0 ? SDL_FILEDIALOG_OPENFILE : kind == 1 ? SDL_FILEDIALOG_SAVEFILE : SDL_FILEDIALOG_OPENFOLDER;
  int32_t request = r->request;
  SDL_ShowFileDialogWithProperties(type, dialog_done, r, props);  // r はコールバックが受け取る
  SDL_DestroyProperties(props);
  clear_staged();
  return request;
}

// ---- カーソル ----
static SDL_Cursor *cursors[SDL_SYSTEM_CURSOR_COUNT];

static int set_cursor(int64_t shape) {
  if (shape < 0 || shape >= SDL_SYSTEM_CURSOR_COUNT) return 0;
  if (!ensure_video()) return 0;
  if (!cursors[shape]) cursors[shape] = SDL_CreateSystemCursor((SDL_SystemCursor)shape);
  return cursors[shape] && SDL_SetCursor(cursors[shape]);
}

// ---- ディスプレイ ----
// ディスプレイは SDL_GetDisplays の並びの番号（0 から）で指す。並びは増減で変わるので、EV_DISPLAYS のあとは読み直す。
static SDL_DisplayID display_at(int64_t index) {
  if (!ensure_video()) return 0;
  int n = 0;
  SDL_DisplayID *ids = SDL_GetDisplays(&n);
  SDL_DisplayID id = ids && index >= 0 && index < n ? ids[index] : 0;
  SDL_free(ids);
  return id;
}

static int64_t display_index(SDL_DisplayID id) {
  int n = 0;
  SDL_DisplayID *ids = SDL_GetDisplays(&n);
  int64_t r = -1;
  for (int i = 0; ids && i < n; i++) {
    if (ids[i] == id) r = i;
  }
  SDL_free(ids);
  return r;
}

static int32_t state_bits(SDL_Window *win) {
  SDL_WindowFlags f = SDL_GetWindowFlags(win);
  int32_t r = 0;
  if (f & SDL_WINDOW_FULLSCREEN) r |= WF_FULLSCREEN;
  if (f & SDL_WINDOW_MAXIMIZED) r |= WF_MAXIMIZED;
  if (f & SDL_WINDOW_MINIMIZED) r |= WF_MINIMIZED;
  if (f & SDL_WINDOW_INPUT_FOCUS) r |= WF_FOCUSED;
  if (f & SDL_WINDOW_HIDDEN) r |= WF_HIDDEN;
  return r;
}

// ---- windowCtl(op, window, io): 引数を io から読み、結果を io に書いて、戻り値も返す ----
// 番号は window.um と揃える。
enum {
  WC_SET_SIZE = 1,         // io[0] = 幅, io[1] = 高さ
  WC_GET_SIZE = 2,         // → io[0], io[1]
  WC_SET_POS = 3,          // io[0] = x, io[1] = y（画面の座標）
  WC_GET_POS = 4,          // → io[0], io[1]
  WC_SET_MIN_SIZE = 5,     // io[0] = 幅, io[1] = 高さ
  WC_MAXIMIZE = 6,
  WC_MINIMIZE = 7,
  WC_RESTORE = 8,
  WC_SET_FULLSCREEN = 9,   // io[0] = 1 で全画面（デスクトップの解像度のまま）、0 で戻す
  WC_STATE = 10,           // → 状態のビット（WF_*）
  WC_PIXEL_SIZE = 11,      // → io[0], io[1]（描画の面の画素の大きさ）
  WC_DISPLAY = 12,         // → ウィンドウのあるディスプレイの番号（なければ -1）
  WC_DISPLAY_COUNT = 13,   // → ディスプレイの数
  WC_DISPLAY_BOUNDS = 14,  // io[0] = 番号 → io[0..3] = x, y, 幅, 高さ。戻り値 1（番号が正しくなければ 0）
  WC_DISPLAY_USABLE = 15,  // 同上（タスクバーなどを除いた範囲）
  WC_START_TEXT = 16,      // 文字の入力を始める
  WC_STOP_TEXT = 17,       // 止める
  WC_TEXT_ACTIVE = 18,     // → 入力中なら 1
  WC_TEXT_AREA = 19,       // io[0..3] = 入力欄の矩形（ウィンドウの座標）, io[4] = 欄の中のカーソルの x（欄の左端から）
  WC_CURSOR = 20,          // io[0] = SDL_SystemCursor の番号（ウィンドウは 0）
  WC_SHOW_CURSOR = 21,     // io[0] = 1 で見せる、0 で隠す
  WC_VSYNC = 22,           // io[0] = 1 で表示を画面の更新に合わせる、0 で合わせない
  WC_RAISE = 23,           // 前に出して焦点を移す
  WC_IME_COMPOSITION = 24, // io[0] = 1 で IME の編集中の文字列をプログラムが描く（SDL を初期化する前だけ。できなければ 0）
  WC_CLIPBOARD_HAS = 25,   // → クリップボードに文字があれば 1
  WC_SET_RESIZABLE = 26,   // io[0] = 1 で大きさを変えられる
  WC_TICKS_NS = 30,        // → 逆戻りしない時計（ナノ秒）。どのスレッドからも呼べる
  WC_SLEEP_NS = 31,        // io[0] ナノ秒だけ眠る（短い時間も正確に）。どのスレッドからも呼べる
  WC_DIALOG_OPEN = 40,     // io[0] = 1 で複数選べる → 要求の番号（0 は開けなかった）
  WC_DIALOG_SAVE = 41,
  WC_DIALOG_FOLDER = 42,   // io[0] = 1 で複数選べる
  WC_DIALOG_RESET = 43,    // 貯めたダイアログの設定を捨てる
};

// 音の出力（runtime/audio.c）。op は 100 番台。
int64_t um_audio_ctl(int32_t op, int32_t stream, int64_t *io, size_t len, int32_t line);

static void need(size_t len, size_t n, int32_t line) {
  if (len < n) um_panic(UM_PANIC_WINDOW_ARGS, line);
}

static int in_i32(int64_t v) { return v >= INT32_MIN && v <= INT32_MAX; }

int64_t um_window_ctl(int32_t op, int32_t window, int64_t *io, size_t len, int32_t line) {
  if (op >= 100) return um_audio_ctl(op, window, io, len, line);
  if (op == WC_TICKS_NS) return (int64_t)SDL_GetTicksNS();
  if (op == WC_SLEEP_NS) {
    need(len, 1, line);
    if (io[0] > 0) SDL_DelayPrecise((Uint64)io[0]);
    return 0;
  }
  main_only(line);
  // ウィンドウを指す操作では、開いているかを確かめる。
  SDL_Window *win = NULL;
  switch (op) {
    case WC_DISPLAY_COUNT: case WC_DISPLAY_BOUNDS: case WC_DISPLAY_USABLE: case WC_CURSOR: case WC_SHOW_CURSOR:
    case WC_IME_COMPOSITION: case WC_CLIPBOARD_HAS: case WC_DIALOG_OPEN: case WC_DIALOG_SAVE: case WC_DIALOG_FOLDER:
    case WC_DIALOG_RESET:
      break;
    default:
      win = lookup(window, line)->win;
  }
  int w = 0, h = 0;
  switch (op) {
    case WC_SET_SIZE:
      need(len, 2, line);
      if (io[0] <= 0 || io[1] <= 0 || !in_i32(io[0]) || !in_i32(io[1])) um_panic(UM_PANIC_WINDOW_ARGS, line);
      return SDL_SetWindowSize(win, (int)io[0], (int)io[1]);
    case WC_GET_SIZE:
      need(len, 2, line);
      SDL_GetWindowSize(win, &w, &h);
      io[0] = w;
      io[1] = h;
      return 1;
    case WC_SET_POS:
      need(len, 2, line);
      if (!in_i32(io[0]) || !in_i32(io[1])) um_panic(UM_PANIC_WINDOW_ARGS, line);
      return SDL_SetWindowPosition(win, (int)io[0], (int)io[1]);
    case WC_GET_POS:
      need(len, 2, line);
      SDL_GetWindowPosition(win, &w, &h);
      io[0] = w;
      io[1] = h;
      return 1;
    case WC_SET_MIN_SIZE:
      need(len, 2, line);
      if (io[0] < 0 || io[1] < 0 || !in_i32(io[0]) || !in_i32(io[1])) um_panic(UM_PANIC_WINDOW_ARGS, line);
      return SDL_SetWindowMinimumSize(win, (int)io[0], (int)io[1]);
    case WC_MAXIMIZE: return SDL_MaximizeWindow(win);
    case WC_MINIMIZE: return SDL_MinimizeWindow(win);
    case WC_RESTORE: return SDL_RestoreWindow(win);
    case WC_SET_FULLSCREEN:
      need(len, 1, line);
      return SDL_SetWindowFullscreen(win, io[0] != 0);
    case WC_STATE: return state_bits(win);
    case WC_PIXEL_SIZE:
      need(len, 2, line);
      SDL_GetWindowSizeInPixels(win, &w, &h);
      io[0] = w;
      io[1] = h;
      return 1;
    case WC_DISPLAY: {
      SDL_DisplayID id = SDL_GetDisplayForWindow(win);
      return id ? display_index(id) : -1;
    }
    case WC_DISPLAY_COUNT: {
      if (!ensure_video()) return 0;
      int n = 0;
      SDL_free(SDL_GetDisplays(&n));
      return n;
    }
    case WC_DISPLAY_BOUNDS:
    case WC_DISPLAY_USABLE: {
      need(len, 4, line);
      SDL_DisplayID id = display_at(io[0]);
      SDL_Rect r;
      if (!id || !(op == WC_DISPLAY_BOUNDS ? SDL_GetDisplayBounds(id, &r) : SDL_GetDisplayUsableBounds(id, &r))) return 0;
      io[0] = r.x;
      io[1] = r.y;
      io[2] = r.w;
      io[3] = r.h;
      return 1;
    }
    case WC_START_TEXT: return SDL_StartTextInput(win);
    case WC_STOP_TEXT: return SDL_StopTextInput(win);
    case WC_TEXT_ACTIVE: return SDL_TextInputActive(win);
    case WC_TEXT_AREA: {
      need(len, 5, line);
      for (int i = 0; i < 5; i++) {
        if (!in_i32(io[i])) um_panic(UM_PANIC_WINDOW_ARGS, line);
      }
      SDL_Rect r = { (int)io[0], (int)io[1], (int)io[2], (int)io[3] };
      return SDL_SetTextInputArea(win, &r, (int)io[4]);
    }
    case WC_CURSOR:
      need(len, 1, line);
      return set_cursor(io[0]);
    case WC_SHOW_CURSOR:
      need(len, 1, line);
      if (!ensure_video()) return 0;
      return io[0] ? SDL_ShowCursor() : SDL_HideCursor();
    case WC_VSYNC:
      need(len, 1, line);
      return SDL_SetRenderVSync(windows[window - 1].ren, io[0] ? 1 : 0);
    case WC_RAISE: return SDL_RaiseWindow(win);
    case WC_IME_COMPOSITION:
      need(len, 1, line);
      if (sdl_ready) return 0;  // SDL を初期化したあとでは効かない
      return SDL_SetHint(SDL_HINT_IME_IMPLEMENTED_UI, io[0] ? "composition" : "none");
    case WC_CLIPBOARD_HAS:
      if (!ensure_video()) return 0;
      return SDL_HasClipboardText();
    case WC_SET_RESIZABLE:
      need(len, 1, line);
      return SDL_SetWindowResizable(win, io[0] != 0);
    case WC_DIALOG_OPEN:
    case WC_DIALOG_SAVE:
    case WC_DIALOG_FOLDER:
      need(len, 1, line);
      return show_dialog(op - WC_DIALOG_OPEN, window, op != WC_DIALOG_SAVE && io[0] != 0, line);
    case WC_DIALOG_RESET:
      clear_staged();
      return 1;
  }
  um_panic(UM_PANIC_WINDOW_ARGS, line);
}

// ---- windowCtlF(op, window, a): 小数を返す操作 ----
enum {
  WF_OP_DISPLAY_SCALE = 1,  // ウィンドウの表示の倍率（UI をこの倍率で大きく描く。125% の設定なら 1.25）
  WF_OP_PIXEL_DENSITY = 2,  // ウィンドウの座標 1 あたりの画素の数（Retina なら 2）
  WF_OP_CONTENT_SCALE = 3,  // a = ディスプレイの番号 → そのディスプレイの倍率（番号が正しくなければ 0）
  WF_OP_REFRESH_RATE = 4,   // a = ディスプレイの番号 → 画面の更新の頻度（Hz。不明なら 0）
  WF_OP_SECONDS = 5,        // 逆戻りしない時計（秒）。どのスレッドからも呼べる
};

double um_window_ctl_f(int32_t op, int32_t window, int64_t a, int32_t line) {
  if (op == WF_OP_SECONDS) return (double)SDL_GetTicksNS() / 1e9;
  main_only(line);
  switch (op) {
    case WF_OP_DISPLAY_SCALE: return SDL_GetWindowDisplayScale(lookup(window, line)->win);
    case WF_OP_PIXEL_DENSITY: return SDL_GetWindowPixelDensity(lookup(window, line)->win);
    case WF_OP_CONTENT_SCALE: {
      SDL_DisplayID id = display_at(a);
      return id ? SDL_GetDisplayContentScale(id) : 0.0;
    }
    case WF_OP_REFRESH_RATE: {
      SDL_DisplayID id = display_at(a);
      const SDL_DisplayMode *m = id ? SDL_GetCurrentDisplayMode(id) : NULL;
      return m ? m->refresh_rate : 0.0;
    }
  }
  um_panic(UM_PANIC_WINDOW_ARGS, line);
}

// ---- windowSetText(op, window, a, text): 文字列を渡す操作 ----
enum {
  WS_TITLE = 1,            // ウィンドウの題名
  WS_CLIPBOARD = 2,        // クリップボードに文字を書く（ウィンドウは 0）
  WS_DIALOG_TITLE = 3,     // 次に開くダイアログの題名
  WS_DIALOG_LOCATION = 4,  // 次に開くダイアログの最初のフォルダかファイル
  WS_DIALOG_FILTER = 5,    // 次に開くダイアログの絞り込みを足す。"名前|mp4;mov" の形（拡張子は ; で区切る。* はすべて）
  WS_DIALOG_ACCEPT = 6,    // 次に開くダイアログの決定のボタンの文字
  WS_TEST_EVENT = 90,      // 試験用: SDL のイベントを作って積む。a = 12 ドロップのファイル、13 ドロップの文字、16 IME の編集中、
                           // 11 ドロップの始まり、15 ドロップの終わり。text = 文字列。位置は (10, 20)、編集の選択は (1, 2)
};

// 試験用のイベントの文字列。SDL はイベントの中のポインタの先を写さないので、取り出すまで残しておく（直近の 16 個）。
static char *test_strings[16];
static int test_next;

static const char *keep_test_string(const char *s) {
  SDL_free(test_strings[test_next]);
  test_strings[test_next] = SDL_strdup(s);
  const char *r = test_strings[test_next];
  test_next = (test_next + 1) % 16;
  return r;
}

// 試験用に SDL のイベントを積む（ドロップや IME は人の操作がないと起きないので、変換の経路だけを確かめる）。
static int64_t push_test_event(int32_t window, int64_t kind, const char *text, int32_t line) {
  SDL_Window *win = lookup(window, line)->win;
  SDL_Event e;
  SDL_zero(e);
  switch (kind) {
    case EV_DROP_BEGIN: e.type = SDL_EVENT_DROP_BEGIN; break;
    case EV_DROP_FILE: e.type = SDL_EVENT_DROP_FILE; break;
    case EV_DROP_TEXT: e.type = SDL_EVENT_DROP_TEXT; break;
    case EV_DROP_COMPLETE: e.type = SDL_EVENT_DROP_COMPLETE; break;
    case EV_TEXT_EDITING:
      e.type = SDL_EVENT_TEXT_EDITING;
      e.edit.windowID = SDL_GetWindowID(win);
      e.edit.text = keep_test_string(text);
      e.edit.start = 1;
      e.edit.length = 2;
      return SDL_PushEvent(&e);
    default:
      um_panic(UM_PANIC_WINDOW_ARGS, line);
  }
  e.drop.windowID = SDL_GetWindowID(win);
  e.drop.x = 10;
  e.drop.y = 20;
  if (kind == EV_DROP_FILE || kind == EV_DROP_TEXT) e.drop.data = keep_test_string(text);
  return SDL_PushEvent(&e);
}

int64_t um_window_set_text(int32_t op, int32_t window, int64_t a, const char *text, size_t text_len, int32_t line) {
  (void)a;
  main_only(line);
  if (op == WS_TITLE) lookup(window, line);
  char *s = cstr(text, text_len);
  if (!s) return 0;
  switch (op) {
    case WS_TEST_EVENT: {
      int64_t r = push_test_event(window, a, s, line);
      SDL_free(s);
      return r;
    }
    case WS_TITLE: {
      int64_t r = SDL_SetWindowTitle(windows[window - 1].win, s);
      SDL_free(s);
      return r;
    }
    case WS_CLIPBOARD: {
      int64_t r = ensure_video() && SDL_SetClipboardText(s);
      SDL_free(s);
      return r;
    }
    case WS_DIALOG_TITLE: SDL_free(staged.title); staged.title = s; return 1;
    case WS_DIALOG_LOCATION: SDL_free(staged.location); staged.location = s; return 1;
    case WS_DIALOG_ACCEPT: SDL_free(staged.accept); staged.accept = s; return 1;
    case WS_DIALOG_FILTER: {
      char *bar = strchr(s, '|');
      const char *pattern = bar ? bar + 1 : "*";
      if (bar) *bar = 0;
      int64_t r = 0;
      SDL_DialogFileFilter *grown = SDL_realloc(staged.filters, (size_t)(staged.nfilters + 1) * sizeof *grown);
      if (grown) {
        staged.filters = grown;
        char *n = SDL_strdup(s), *p = SDL_strdup(pattern);
        if (n && p) {
          staged.filters[staged.nfilters].name = n;
          staged.filters[staged.nfilters].pattern = p;
          staged.nfilters++;
          r = 1;
        } else {
          SDL_free(n);
          SDL_free(p);
        }
      }
      SDL_free(s);
      return r;
    }
  }
  SDL_free(s);
  um_panic(UM_PANIC_WINDOW_ARGS, line);
}

// C の文字列を、UM の文字列（um_alloc で確保したバイト列）として返す。
static void give_string(const char *s, unsigned char **out, size_t *len) {
  size_t n = s ? strlen(s) : 0;
  unsigned char *p = um_alloc(n, 1);
  if (n) memcpy(p, s, n);
  *out = p;
  *len = n;
}

// ---- windowGetText(op, window, a): 文字列を返す操作 ----
enum {
  WG_CLIPBOARD = 1,     // クリップボードの文字（なければ空）
  WG_EVENT_STRING = 2,  // 最後に windowPoll が返したイベントの a 番目の文字列（なければ空）
  WG_DISPLAY_NAME = 3,  // a = ディスプレイの番号
  WG_TITLE = 4,         // ウィンドウの題名
  WG_ERROR = 5,         // SDL の最後の失敗の理由（どのスレッドからも呼べる。そのスレッドのもの）
};

void um_window_get_text(int32_t op, int32_t window, int64_t a, unsigned char **out, size_t *len, int32_t line) {
  if (op == WG_ERROR) {
    give_string(SDL_GetError(), out, len);
    return;
  }
  main_only(line);
  switch (op) {
    case WG_CLIPBOARD: {
      char *s = ensure_video() ? SDL_GetClipboardText() : NULL;
      give_string(s, out, len);
      SDL_free(s);
      return;
    }
    case WG_EVENT_STRING:
      give_string(a >= 0 && a < ev_nstrings ? ev_strings[a] : NULL, out, len);
      return;
    case WG_DISPLAY_NAME: {
      SDL_DisplayID id = display_at(a);
      give_string(id ? SDL_GetDisplayName(id) : NULL, out, len);
      return;
    }
    case WG_TITLE:
      give_string(SDL_GetWindowTitle(lookup(window, line)->win), out, len);
      return;
  }
  um_panic(UM_PANIC_WINDOW_ARGS, line);
}

// ---- イベント ----

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
    case SDL_EVENT_TEXT_EDITING:
      out[0] = EV_TEXT_EDITING;
      out[1] = number_of(e->edit.windowID);
      out[2] = e->edit.start;
      out[3] = e->edit.length;
      if (!out[1]) return 0;
      add_event_string(e->edit.text);
      return 1;
    case SDL_EVENT_DROP_BEGIN:
    case SDL_EVENT_DROP_FILE:
    case SDL_EVENT_DROP_TEXT:
    case SDL_EVENT_DROP_POSITION:
    case SDL_EVENT_DROP_COMPLETE:
      out[0] = e->type == SDL_EVENT_DROP_BEGIN ? EV_DROP_BEGIN
             : e->type == SDL_EVENT_DROP_FILE ? EV_DROP_FILE
             : e->type == SDL_EVENT_DROP_TEXT ? EV_DROP_TEXT
             : e->type == SDL_EVENT_DROP_POSITION ? EV_DROP_POSITION
             : EV_DROP_COMPLETE;
      out[1] = number_of(e->drop.windowID);
      if (e->type != SDL_EVENT_DROP_BEGIN) {
        out[2] = (int32_t)e->drop.x;
        out[3] = (int32_t)e->drop.y;
      }
      if (!out[1]) return 0;
      if (e->type == SDL_EVENT_DROP_FILE || e->type == SDL_EVENT_DROP_TEXT) add_event_string(e->drop.data);
      return 1;
    case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
      out[0] = EV_PIXEL_SIZE;
      out[1] = number_of(e->window.windowID);
      out[2] = e->window.data1;
      out[3] = e->window.data2;
      return out[1] != 0;
    case SDL_EVENT_WINDOW_DISPLAY_SCALE_CHANGED:
      out[0] = EV_SCALE;
      out[1] = number_of(e->window.windowID);
      return out[1] != 0;
    case SDL_EVENT_WINDOW_FOCUS_GAINED:
    case SDL_EVENT_WINDOW_FOCUS_LOST:
      out[0] = EV_FOCUS;
      out[1] = number_of(e->window.windowID);
      out[2] = e->type == SDL_EVENT_WINDOW_FOCUS_GAINED;
      return out[1] != 0;
    case SDL_EVENT_WINDOW_MOUSE_ENTER:
    case SDL_EVENT_WINDOW_MOUSE_LEAVE:
      out[0] = EV_MOUSE_ENTER;
      out[1] = number_of(e->window.windowID);
      out[2] = e->type == SDL_EVENT_WINDOW_MOUSE_ENTER;
      return out[1] != 0;
    case SDL_EVENT_WINDOW_MOVED:
      out[0] = EV_MOVED;
      out[1] = number_of(e->window.windowID);
      out[2] = e->window.data1;
      out[3] = e->window.data2;
      return out[1] != 0;
    case SDL_EVENT_WINDOW_MAXIMIZED:
    case SDL_EVENT_WINDOW_MINIMIZED:
    case SDL_EVENT_WINDOW_RESTORED:
    case SDL_EVENT_WINDOW_ENTER_FULLSCREEN:
    case SDL_EVENT_WINDOW_LEAVE_FULLSCREEN: {
      int32_t id = number_of(e->window.windowID);
      if (!id) return 0;
      out[0] = EV_STATE;
      out[1] = id;
      out[2] = state_bits(windows[id - 1].win);
      return 1;
    }
    case SDL_EVENT_WINDOW_EXPOSED:
      out[0] = EV_EXPOSED;
      out[1] = number_of(e->window.windowID);
      return out[1] != 0;
    case SDL_EVENT_RENDER_TARGETS_RESET:
    case SDL_EVENT_RENDER_DEVICE_RESET:
    case SDL_EVENT_RENDER_DEVICE_LOST: {
      // テクスチャの中身がなくなったかもしれない。次の表示は全体を写し、プログラムには描き直してもらう。
      int32_t id = number_of(e->render.windowID);
      for (int i = 0; i < MAX_WINDOWS; i++) {
        if (windows[i].win && (!id || id == i + 1)) windows[i].tex_valid = 0;
      }
      out[0] = EV_EXPOSED;
      out[1] = id;
      return id != 0;
    }
    case SDL_EVENT_DISPLAY_ADDED:
    case SDL_EVENT_DISPLAY_REMOVED:
    case SDL_EVENT_DISPLAY_MOVED:
    case SDL_EVENT_DISPLAY_CONTENT_SCALE_CHANGED:
      out[0] = EV_DISPLAYS;
      return 1;
    default:
      if (dialog_event_type && e->type == dialog_event_type) {
        dialog_result *r = e->user.data1;
        out[0] = EV_DIALOG;
        out[1] = r->request;
        out[2] = r->count;
        out[3] = r->filter;
        for (int i = 0; i < r->count; i++) add_event_string(r->paths[i]);
        free_result(r);
        return 1;
      }
      return 0;
  }
}

int32_t um_window_poll(int32_t *event, size_t len, int32_t timeout_ms, int32_t line) {
  main_only(line);
  if (len < EV_FIELDS) um_panic(UM_PANIC_WINDOW_ARGS, line);
  memset(event, 0, EV_FIELDS * sizeof *event);
  clear_event_strings();
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
