// WebAssembly の組み込み（wasmCompile / wasmInstantiate / wasmInvoke / wasmRead / wasmWrite など）の実装。
// wasmtime の C API（JIT。https://github.com/bytecodealliance/wasmtime）を使う。
// プログラムが wasm の組み込みを使ったときだけ、ドライバが native.c と一緒にリンクする。
//
// 決まりごと:
// - 読み込んだモジュールとインスタンスは、64 ビットの番号（ハンドル）で表す。種類（モジュールかインスタンスか）、
//   表の場所、世代を詰めてあるので、解放済みの番号や種類の違う番号を渡すと停止する（13）。
// - どの関数もどのスレッドからでも呼べる。エンジンとモジュールはスレッドの間で共有し、インスタンスごとに
//   wasmtime の Store を分ける（Store は単一スレッド用）。同じインスタンスを2つのスレッドが同時に使おうとしたら、
//   後から来たほうを失敗にする（理由は wasmError）。使っている最中のインスタンスを解放しようとしたら停止（14）。
// - 失敗（読み込めない、トラップ、燃料切れ、時間切れ、範囲外など）では停止せず、失敗を返す。理由の文字列は
//   スレッドごとに覚えておき、wasmError() で読む（runtime/wasm.um がこれを Result にまとめる）。
// - 既定では、モジュールにホストの関数（import）も WASI も渡さない。import を持つモジュールは
//   インスタンスを作れない（stubImports を付けると、import を「呼ぶとトラップする関数」で埋める）。
// - 上限: 燃料（呼び出し1回あたりの命令の目安。0 は上限なし）、時間（呼び出し1回あたりのミリ秒。epoch で測る）、
//   メモリ（インスタンスの線形メモリのバイト数）。
// - 燃料と epoch の検査は、生成するコードに入れると遅くなる（燃料と epoch で 15〜35%）。そこでエンジンを3つ持ち、
//   インスタンスの上限に合わせて使い分ける: 上限なし → 検査なし、時間だけ → epoch、燃料あり → 燃料と epoch。
//   モジュールは読み込むときに「検査なし」でコンパイルし（壊れていればここで失敗）、ほかの2つは使うときにコンパイルする
//   （バイト列の写しを持っておく。ディスクのキャッシュが効く）。

#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdarg.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include "win32.h"
#else
#include <pthread.h>
#include <unistd.h>
#endif

#include <wasmtime.h>

_Noreturn void um_panic(int32_t code, int32_t line);
void *um_alloc(size_t count, size_t elem_size);

// um_panic の原因コード（native.c の表と src/codegen.ts の並びと揃えること）。
#define UM_PANIC_WASM_HANDLE 13  // 番号が正しくない（解放済み、種類違い）か、配列の範囲の指定が正しくない
#define UM_PANIC_WASM_BUSY 14    // ほかのスレッドが使っている最中のインスタンスを解放しようとした

#define KIND_MODULE 1
#define KIND_INSTANCE 2
#define MAX_SLOTS 4096  // モジュールとインスタンスの、それぞれ同時に持てる数

// 値の種類（runtime/wasm.um の WasmValue と揃える。wasmtime の WASMTIME_I32 などと同じ番号）。
#define VK_I32 0
#define VK_I64 1
#define VK_F32 2
#define VK_F64 3

// 上限を付けないときの燃料と epoch の期限（どちらも「実質なし」）。
#define FUEL_UNLIMITED ((uint64_t)1 << 62)
#define EPOCH_FAR ((uint64_t)1 << 62)

// エンジンの種類（生成するコードに入れる検査）。
#define ENGINE_PLAIN 0  // 検査なし
#define ENGINE_EPOCH 1  // 時間の上限（epoch）
#define ENGINE_FUEL 2   // 燃料と時間の上限
#define ENGINES 3

typedef struct {
  uint8_t *bytes;  // コンパイルし直すためのバイト列の写し
  size_t len;
  wasmtime_module_t *mods[ENGINES];  // エンジンごとのコンパイル結果（まだなら NULL）
  pthread_mutex_t lock;              // mods を埋めるとき
  _Atomic int refs;                  // 表から 1、インスタンスを作っている最中の呼び出しから 1 ずつ
} um_wmodule;

typedef struct {
  int engine;  // ENGINE_*
  wasmtime_store_t *store;
  wasmtime_context_t *ctx;
  wasmtime_instance_t instance;
  wasmtime_memory_t memory;
  int has_memory;
  uint64_t fuel;        // 呼び出し1回あたりの燃料（0 は上限なし）
  uint64_t timeout_ms;  // 呼び出し1回あたりの時間（0 は上限なし）
  _Atomic int busy;     // 使っている最中なら 1
} um_winstance;

// 表。場所 i の世代は gens[i]（解放のたびに 1 増やす）。番号 = 種類 << 56 | 世代 << 24 | (場所 + 1)。
typedef struct {
  void *items[MAX_SLOTS];
  uint32_t gens[MAX_SLOTS];
} um_table;

static um_table modules, instances;
static pthread_mutex_t table_lock = PTHREAD_MUTEX_INITIALIZER;

static wasm_engine_t *_Atomic engines[ENGINES];
static pthread_mutex_t engine_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t ticker_once = PTHREAD_ONCE_INIT;

// ---- 失敗の理由（スレッドごと） ----

static _Thread_local char last_error[2048];

static void set_error(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(last_error, sizeof last_error, fmt, ap);
  va_end(ap);
}

// wasmtime の文を1行にまとめる。呼び出し履歴つきの文（"... wasm backtrace: ... Caused by: 理由"）は理由だけにし、
// 複数の行は "; " でつなぐ。
static void set_error_tidy(const char *prefix, const char *msg, size_t n) {
  const char *cause = NULL;
  for (size_t i = 0; i + 10 <= n; i++) {
    if (memcmp(msg + i, "Caused by:", 10) == 0) cause = msg + i + 10;
  }
  const char *p = cause ? cause : msg;
  const char *end = msg + n;
  size_t at = (size_t)snprintf(last_error, sizeof last_error, "%s", prefix);
  int first = 1;
  while (p < end && at + 1 < sizeof last_error) {
    while (p < end && (*p == ' ' || *p == '\n' || *p == '\r' || *p == 0)) p++;
    const char *q = p;
    while (q < end && *q != '\n' && *q != 0) q++;
    const char *r = q;
    while (r > p && (r[-1] == ' ' || r[-1] == '\r')) r--;
    if (r > p) {
      at += (size_t)snprintf(last_error + at, sizeof last_error - at, "%s%.*s", first ? "" : "; ", (int)(r - p), p);
      first = 0;
    }
    p = q;
  }
}

// wasmtime のエラーを理由にする（エラーは解放する）。
static void set_error_from(const char *what, wasmtime_error_t *err) {
  wasm_name_t msg;
  wasmtime_error_message(err, &msg);
  char prefix[300];
  snprintf(prefix, sizeof prefix, "%s: ", what);
  set_error_tidy(prefix, msg.data, msg.size);
  wasm_byte_vec_delete(&msg);
  wasmtime_error_delete(err);
}

// トラップの理由。原因の分かるトラップは短い決まった文にする（wasmtime の文には呼び出し履歴が付いて長いため）。
static void set_error_from_trap(const char *fn, size_t fn_len, wasm_trap_t *trap) {
  wasmtime_trap_code_t code;
  const char *why = NULL;
  if (wasmtime_trap_code(trap, &code)) {
    switch (code) {
      case WASMTIME_TRAP_CODE_STACK_OVERFLOW: why = "stack overflow"; break;
      case WASMTIME_TRAP_CODE_MEMORY_OUT_OF_BOUNDS: why = "out of bounds memory access"; break;
      case WASMTIME_TRAP_CODE_HEAP_MISALIGNED: why = "misaligned atomic memory access"; break;
      case WASMTIME_TRAP_CODE_TABLE_OUT_OF_BOUNDS: why = "out of bounds table access"; break;
      case WASMTIME_TRAP_CODE_INDIRECT_CALL_TO_NULL: why = "indirect call to a null table entry"; break;
      case WASMTIME_TRAP_CODE_BAD_SIGNATURE: why = "indirect call signature mismatch"; break;
      case WASMTIME_TRAP_CODE_INTEGER_OVERFLOW: why = "integer overflow"; break;
      case WASMTIME_TRAP_CODE_INTEGER_DIVISION_BY_ZERO: why = "integer divide by zero"; break;
      case WASMTIME_TRAP_CODE_BAD_CONVERSION_TO_INTEGER: why = "invalid conversion to integer"; break;
      case WASMTIME_TRAP_CODE_UNREACHABLE_CODE_REACHED: why = "unreachable executed"; break;
      case WASMTIME_TRAP_CODE_INTERRUPT: why = "timeout (the call ran longer than timeoutMs)"; break;
      case WASMTIME_TRAP_CODE_OUT_OF_FUEL: why = "out of fuel (the call ran longer than the fuel limit)"; break;
      default: break;
    }
  }
  if (why) {
    set_error("trap in '%.*s': %s", (int)fn_len, fn, why);
  } else {
    // 原因コードのないトラップ。wasmtime の文から理由を取り出す。
    wasm_message_t msg;
    wasm_trap_message(trap, &msg);
    char prefix[300];
    snprintf(prefix, sizeof prefix, "trap in '%.*s': ", (int)fn_len, fn);
    set_error_tidy(prefix, msg.data, msg.size);
    wasm_byte_vec_delete(&msg);
  }
  wasm_trap_delete(trap);
}

// ---- エンジンと時計 ----

// エンジン（コンパイラと、生成するコードの設定）。最初に使うときに作る。
static wasm_engine_t *get_engine(int kind) {
  wasm_engine_t *e = atomic_load(&engines[kind]);
  if (e) return e;
  pthread_mutex_lock(&engine_lock);
  e = atomic_load(&engines[kind]);
  if (!e) {
    wasm_config_t *config = wasm_config_new();
    // 燃料と epoch による中断の検査を、生成するコードに入れるかどうか。
    wasmtime_config_consume_fuel_set(config, kind == ENGINE_FUEL);
    wasmtime_config_epoch_interruption_set(config, kind != ENGINE_PLAIN);
    // コンパイル結果のキャッシュ（ディスク）。同じバイト列の2回目からは、コンパイルせずに読み込む。
    // UM_WASM_CACHE=0 で切る。キャッシュの場所と大きさの上限は wasmtime の設定ファイルに従う。
    const char *cache = getenv("UM_WASM_CACHE");
    if (!cache || strcmp(cache, "0") != 0) {
      wasmtime_error_t *err = wasmtime_config_cache_config_load(config, NULL);
      if (err) wasmtime_error_delete(err);  // キャッシュが使えなくても、毎回コンパイルするだけ
    }
    e = wasm_engine_new_with_config(config);
    atomic_store(&engines[kind], e);
  }
  pthread_mutex_unlock(&engine_lock);
  return e;
}

// epoch を「起動からのミリ秒」として進める係。時間の上限を使うインスタンスを初めて作ったときに起こす。
// 寝る時間は OS の精度に左右されるので、実際に経った時間だけ epoch を進める。
static double now_ms(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec * 1000.0 + (double)t.tv_nsec / 1e6;
}

static void *ticker_main(void *arg) {
  (void)arg;
  double origin = now_ms();
  uint64_t ticks = 0;
  for (;;) {
#ifdef _WIN32
    Sleep(1);
#else
    struct timespec d = { 0, 1000000 };
    nanosleep(&d, NULL);
#endif
    uint64_t want = (uint64_t)(now_ms() - origin);
    while (ticks < want) {
      // epoch を使うエンジン（まだ作っていなければ飛ばす。期限は「今の epoch から」なので、途中から数えてよい）。
      for (int k = ENGINE_EPOCH; k < ENGINES; k++) {
        wasm_engine_t *e = atomic_load(&engines[k]);
        if (e) wasmtime_engine_increment_epoch(e);
      }
      ticks++;
    }
  }
  return NULL;
}

static void ticker_init(void) {
  pthread_t t;
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setstacksize(&attr, 65536);
  pthread_create(&t, &attr, ticker_main, NULL);
  pthread_attr_destroy(&attr);
}

// ---- 表 ----

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
    um_panic(UM_PANIC_WASM_HANDLE, line);
  }
  return (int)i;
}

// インスタンスを借りる。ほかのスレッドが使っている最中なら NULL（理由を書く）。
static um_winstance *acquire(int64_t h, int32_t line) {
  pthread_mutex_lock(&table_lock);
  um_winstance *w = instances.items[slot_of(&instances, KIND_INSTANCE, h, line)];
  int expected = 0;
  int ok = atomic_compare_exchange_strong(&w->busy, &expected, 1);
  pthread_mutex_unlock(&table_lock);
  if (!ok) {
    set_error("the instance is being used by another thread (use one instance per thread)");
    return NULL;
  }
  return w;
}

static void release(um_winstance *w) { atomic_store(&w->busy, 0); }

// UM の文字列や配列として返すために、um_alloc で写す。
static void give_string(const char *s, size_t n, unsigned char **out, size_t *len) {
  *out = NULL;
  *len = 0;
  if (n == 0) return;
  unsigned char *p = um_alloc(n, 1);
  memcpy(p, s, n);
  *out = p;
  *len = n;
}

// ---- モジュール ----

// バイト列（wasm のバイナリ）をコンパイルして、モジュールの番号を返す。失敗なら 0。
// 「検査なし」のエンジンでコンパイルし（壊れたバイナリはここで失敗）、バイト列の写しを持っておく。
int64_t um_wasm_compile(const uint8_t *bytes, size_t len, int32_t line) {
  wasmtime_module_t *m = NULL;
  wasmtime_error_t *err = wasmtime_module_new(get_engine(ENGINE_PLAIN), bytes, len, &m);
  if (err) {
    set_error_from("could not compile the module", err);
    return 0;
  }
  um_wmodule *w = calloc(1, sizeof *w);
  if (!w || !(w->bytes = malloc(len ? len : 1))) um_panic(5, line);
  memcpy(w->bytes, bytes, len);
  w->len = len;
  w->mods[ENGINE_PLAIN] = m;
  pthread_mutex_t init = PTHREAD_MUTEX_INITIALIZER;
  w->lock = init;
  atomic_store(&w->refs, 1);
  int64_t h = table_add(&modules, KIND_MODULE, w);
  if (!h) {
    wasmtime_module_delete(m);
    free(w->bytes);
    free(w);
    set_error("too many modules are loaded (at most %d)", MAX_SLOTS);
    return 0;
  }
  return h;
}

// モジュールの参照を1つ減らし、なくなったら消す（インスタンスは wasmtime のモジュールを自分で持っているので、先に消してよい）。
static void module_unref(um_wmodule *w) {
  if (atomic_fetch_sub(&w->refs, 1) != 1) return;
  for (int k = 0; k < ENGINES; k++) {
    if (w->mods[k]) wasmtime_module_delete(w->mods[k]);
  }
  free(w->bytes);
  free(w);
}

// 番号からモジュールを借りる（使い終わったら module_unref）。
static um_wmodule *module_ref(int64_t h, int32_t line) {
  pthread_mutex_lock(&table_lock);
  um_wmodule *w = modules.items[slot_of(&modules, KIND_MODULE, h, line)];
  atomic_fetch_add(&w->refs, 1);
  pthread_mutex_unlock(&table_lock);
  return w;
}

void um_wasm_free_module(int64_t h, int32_t line) {
  pthread_mutex_lock(&table_lock);
  int i = slot_of(&modules, KIND_MODULE, h, line);
  um_wmodule *w = modules.items[i];
  modules.items[i] = NULL;
  modules.gens[i]++;
  pthread_mutex_unlock(&table_lock);
  module_unref(w);
}

// エンジン kind 向けのコンパイル結果（なければここでコンパイルする）。失敗なら NULL（理由を書く）。
static wasmtime_module_t *module_for(um_wmodule *w, int kind) {
  pthread_mutex_lock(&w->lock);
  if (!w->mods[kind]) {
    wasmtime_error_t *err = wasmtime_module_new(get_engine(kind), w->bytes, w->len, &w->mods[kind]);
    if (err) {
      w->mods[kind] = NULL;
      set_error_from("could not compile the module", err);
    }
  }
  wasmtime_module_t *m = w->mods[kind] ? wasmtime_module_clone(w->mods[kind]) : NULL;
  pthread_mutex_unlock(&w->lock);
  return m;
}

static char valtype_char(const wasm_valtype_t *t) {
  switch (wasm_valtype_kind(t)) {
    case WASM_I32: return 'i';
    case WASM_I64: return 'I';
    case WASM_F32: return 'f';
    case WASM_F64: return 'F';
    case WASMTIME_V128: return 'v';  // wasm.h には名前がない（wasmtime の拡張で、番号は WASMTIME_V128 と同じ）
    default: return 'r';  // 参照（funcref / externref など）
  }
}

// export の一覧を string[] で返す。1つの export は「種類の1文字 + 引数の型 + ':' + 戻り値の型 + ':' + 名前」。
// 種類は f（関数）、m（メモリ）、g（グローバル）、t（テーブル）、x（そのほか）。型は i / I / f / F（i32 / i64 / f32 / f64）、
// v（v128）、r（参照）。runtime/wasm.um の wasmExports が WasmExport の配列にする。
typedef struct {
  unsigned char *data;
  size_t len, cap;
} um_bytes;

void um_wasm_exports(int64_t h, um_bytes **out, size_t *len, int32_t line) {
  *out = NULL;
  *len = 0;
  // 一覧を作るあいだに解放されないよう、モジュールの参照を1つ増やしておく。
  um_wmodule *w = module_ref(h, line);
  wasmtime_module_t *m = wasmtime_module_clone(w->mods[ENGINE_PLAIN]);
  module_unref(w);
  wasm_exporttype_vec_t ex;
  wasmtime_module_exports(m, &ex);
  if (ex.size > 0) {
    um_bytes *a = um_alloc(ex.size, sizeof(um_bytes));
    for (size_t k = 0; k < ex.size; k++) {
      const wasm_name_t *name = wasm_exporttype_name(ex.data[k]);
      const wasm_externtype_t *et = wasm_exporttype_type(ex.data[k]);
      char sig[600];
      size_t n = 0;
      switch (wasm_externtype_kind(et)) {
        case WASM_EXTERN_FUNC: {
          const wasm_functype_t *ft = wasm_externtype_as_functype_const(et);
          const wasm_valtype_vec_t *ps = wasm_functype_params(ft);
          const wasm_valtype_vec_t *rs = wasm_functype_results(ft);
          sig[n++] = 'f';
          for (size_t j = 0; j < ps->size && n < 280; j++) sig[n++] = valtype_char(ps->data[j]);
          sig[n++] = ':';
          for (size_t j = 0; j < rs->size && n < 560; j++) sig[n++] = valtype_char(rs->data[j]);
          sig[n++] = ':';
          break;
        }
        case WASM_EXTERN_MEMORY: sig[n++] = 'm'; sig[n++] = ':'; sig[n++] = ':'; break;
        case WASM_EXTERN_GLOBAL: sig[n++] = 'g'; sig[n++] = ':'; sig[n++] = ':'; break;
        case WASM_EXTERN_TABLE: sig[n++] = 't'; sig[n++] = ':'; sig[n++] = ':'; break;
        default: sig[n++] = 'x'; sig[n++] = ':'; sig[n++] = ':'; break;
      }
      size_t total = n + name->size;
      unsigned char *p = um_alloc(total, 1);
      memcpy(p, sig, n);
      if (name->size) memcpy(p + n, name->data, name->size);
      a[k].data = p;
      a[k].len = total;
      a[k].cap = total;
    }
    *out = a;
    *len = ex.size;
  }
  wasm_exporttype_vec_delete(&ex);
  wasmtime_module_delete(m);
}

// ---- インスタンス ----

// 呼び出しの前に、燃料と時間の上限を設定し直す（上限は呼び出し1回ごと）。燃料の上限がなくても時間の上限があれば、
// 燃料のエンジンでは燃料を「実質なし」にしておく。
static void arm_limits(um_winstance *w) {
  if (w->engine == ENGINE_FUEL) wasmtime_context_set_fuel(w->ctx, w->fuel ? w->fuel : FUEL_UNLIMITED);
  if (w->engine != ENGINE_PLAIN) wasmtime_context_set_epoch_deadline(w->ctx, w->timeout_ms ? w->timeout_ms : EPOCH_FAR);
}

// インスタンスを作る。fuel / timeout_ms / memory_bytes は 0 で上限なし。stub_imports が 0 でなければ、
// import を「呼ぶとトラップする関数」で埋める（0 なら import を持つモジュールは失敗）。失敗なら 0。
int64_t um_wasm_instantiate(int64_t mh, int64_t fuel, int64_t timeout_ms, int64_t memory_bytes, int32_t stub_imports, int32_t line) {
  if (fuel < 0 || timeout_ms < 0 || memory_bytes < 0) {
    set_error("limits must not be negative (0 means no limit)");
    return 0;
  }
  // 上限に合わせてエンジンを選ぶ（検査を入れるほど遅くなるので、要らない検査は入れない）。
  int kind = fuel > 0 ? ENGINE_FUEL : timeout_ms > 0 ? ENGINE_EPOCH : ENGINE_PLAIN;
  um_wmodule *wm = module_ref(mh, line);
  wasmtime_module_t *m = module_for(wm, kind);
  module_unref(wm);
  if (!m) return 0;

  wasm_engine_t *e = get_engine(kind);
  if (timeout_ms > 0) pthread_once(&ticker_once, ticker_init);
  um_winstance *w = calloc(1, sizeof *w);
  if (!w) um_panic(5, line);
  w->engine = kind;
  w->fuel = (uint64_t)fuel;
  w->timeout_ms = (uint64_t)timeout_ms;
  w->store = wasmtime_store_new(e, NULL, NULL);
  w->ctx = wasmtime_store_context(w->store);
  if (memory_bytes > 0) wasmtime_store_limiter(w->store, memory_bytes, -1, -1, -1, -1);
  // start 関数（インスタンスを作るときに走る）にも同じ上限をかける。
  arm_limits(w);

  wasmtime_linker_t *linker = wasmtime_linker_new(e);
  wasmtime_error_t *err = NULL;
  if (stub_imports) err = wasmtime_linker_define_unknown_imports_as_traps(linker, m);
  wasm_trap_t *trap = NULL;
  if (!err) err = wasmtime_linker_instantiate(linker, w->ctx, m, &w->instance, &trap);
  wasmtime_linker_delete(linker);
  wasmtime_module_delete(m);
  if (err || trap) {
    if (err) set_error_from("could not instantiate the module", err);
    else set_error_from_trap("start", 5, trap);
    wasmtime_store_delete(w->store);
    free(w);
    return 0;
  }
  // 写す先のメモリ: "memory" という名前の export、なければ最初に見つかった export のメモリ。
  wasmtime_extern_t item;
  if (wasmtime_instance_export_get(w->ctx, &w->instance, "memory", 6, &item) && item.kind == WASMTIME_EXTERN_MEMORY) {
    w->memory = item.of.memory;
    w->has_memory = 1;
  } else {
    char *name;
    size_t name_len;
    for (size_t k = 0; !w->has_memory && wasmtime_instance_export_nth(w->ctx, &w->instance, k, &name, &name_len, &item); k++) {
      if (item.kind == WASMTIME_EXTERN_MEMORY) {
        w->memory = item.of.memory;
        w->has_memory = 1;
      }
    }
  }
  int64_t h = table_add(&instances, KIND_INSTANCE, w);
  if (!h) {
    wasmtime_store_delete(w->store);
    free(w);
    set_error("too many instances exist (at most %d)", MAX_SLOTS);
    return 0;
  }
  return h;
}

void um_wasm_free_instance(int64_t h, int32_t line) {
  pthread_mutex_lock(&table_lock);
  int i = slot_of(&instances, KIND_INSTANCE, h, line);
  um_winstance *w = instances.items[i];
  int expected = 0;
  if (!atomic_compare_exchange_strong(&w->busy, &expected, 1)) {
    pthread_mutex_unlock(&table_lock);
    um_panic(UM_PANIC_WASM_BUSY, line);
  }
  instances.items[i] = NULL;
  instances.gens[i]++;
  pthread_mutex_unlock(&table_lock);
  wasmtime_store_delete(w->store);
  free(w);
}


static const char *kind_name(int k) {
  switch (k) {
    case VK_I32: return "i32";
    case VK_I64: return "i64";
    case VK_F32: return "f32";
    case VK_F64: return "f64";
    default: return "an unsupported type";
  }
}

static int kind_of(const wasm_valtype_t *t) {
  switch (wasm_valtype_kind(t)) {
    case WASM_I32: return VK_I32;
    case WASM_I64: return VK_I64;
    case WASM_F32: return VK_F32;
    case WASM_F64: return VK_F64;
    default: return -1;
  }
}

#define MAX_VALUES 64

// export された関数 name を呼ぶ。引数 k は kinds[k] の種類で、整数なら ints[k]、小数なら floats[k] から取る。
// 戻り値も同じ形で kinds / ints / floats の先頭から書き、個数を返す。失敗なら -1（理由は wasmError）。
// 3つの配列は、引数の数と戻り値の数のどちらよりも長くなければならない。
int32_t um_wasm_invoke(int64_t h, const char *name, size_t name_len, int32_t *kinds, size_t kinds_len, int64_t *ints, size_t ints_len,
                       double *floats, size_t floats_len, int32_t nargs, int32_t line) {
  size_t room = kinds_len < ints_len ? kinds_len : ints_len;
  if (floats_len < room) room = floats_len;
  if (nargs < 0 || (size_t)nargs > room) um_panic(UM_PANIC_WASM_HANDLE, line);
  um_winstance *w = acquire(h, line);
  if (!w) return -1;
  int32_t result = -1;
  wasmtime_extern_t item;
  if (!wasmtime_instance_export_get(w->ctx, &w->instance, name, name_len, &item)) {
    set_error("the module has no export named '%.*s'", (int)name_len, name);
    goto done;
  }
  if (item.kind != WASMTIME_EXTERN_FUNC) {
    set_error("the export '%.*s' is not a function", (int)name_len, name);
    if (item.kind != WASMTIME_EXTERN_MEMORY) wasmtime_extern_delete(&item);
    goto done;
  }
  wasm_functype_t *ft = wasmtime_func_type(w->ctx, &item.of.func);
  const wasm_valtype_vec_t *ps = wasm_functype_params(ft);
  const wasm_valtype_vec_t *rs = wasm_functype_results(ft);
  if (ps->size != (size_t)nargs) {
    set_error("'%.*s' takes %d arguments but %d were given", (int)name_len, name, (int)ps->size, (int)nargs);
    goto done_type;
  }
  if (ps->size > MAX_VALUES || rs->size > MAX_VALUES || rs->size > room) {
    set_error("'%.*s' has too many parameters or results (%d and %d)", (int)name_len, name, (int)ps->size, (int)rs->size);
    goto done_type;
  }
  wasmtime_val_t args[MAX_VALUES], results[MAX_VALUES];
  for (size_t k = 0; k < ps->size; k++) {
    int want = kind_of(ps->data[k]);
    if (want < 0) {
      set_error("parameter %d of '%.*s' has a type that cannot be passed from UM (only i32, i64, f32 and f64)", (int)k + 1, (int)name_len, name);
      goto done_type;
    }
    if (kinds[k] != want) {
      set_error("argument %d of '%.*s' must be %s, but %s was given", (int)k + 1, (int)name_len, name, kind_name(want), kind_name(kinds[k]));
      goto done_type;
    }
    args[k].kind = (wasmtime_valkind_t)want;
    switch (want) {
      case VK_I32: args[k].of.i32 = (int32_t)ints[k]; break;
      case VK_I64: args[k].of.i64 = ints[k]; break;
      case VK_F32: args[k].of.f32 = (float)floats[k]; break;
      default: args[k].of.f64 = floats[k]; break;
    }
  }
  for (size_t k = 0; k < rs->size; k++) {
    if (kind_of(rs->data[k]) < 0) {
      set_error("result %d of '%.*s' has a type that cannot be returned to UM (only i32, i64, f32 and f64)", (int)k + 1, (int)name_len, name);
      goto done_type;
    }
  }
  arm_limits(w);
  wasm_trap_t *trap = NULL;
  wasmtime_error_t *err = wasmtime_func_call(w->ctx, &item.of.func, args, ps->size, results, rs->size, &trap);
  if (err) {
    // ホスト側で起きたトラップ（埋めた import を呼んだなど）は、エラーとして来る。
    char what[300];
    snprintf(what, sizeof what, "trap in '%.*s'", (int)name_len, name);
    set_error_from(what, err);
    goto done_type;
  }
  if (trap) {
    set_error_from_trap(name, name_len, trap);
    goto done_type;
  }
  for (size_t k = 0; k < rs->size; k++) {
    kinds[k] = results[k].kind;
    ints[k] = 0;
    floats[k] = 0;
    switch (results[k].kind) {
      case VK_I32: ints[k] = results[k].of.i32; break;
      case VK_I64: ints[k] = results[k].of.i64; break;
      case VK_F32: floats[k] = results[k].of.f32; break;
      default: floats[k] = results[k].of.f64; break;
    }
  }
  result = (int32_t)rs->size;
done_type:
  wasm_functype_delete(ft);
done:
  release(w);
  return result;
}

// ---- メモリ ----

// 写す範囲: UM の配列の start 番目から count 個（count が負なら終わりまで）。範囲外なら停止。
static size_t range_bytes(size_t len, size_t elem, int64_t start, int64_t count, size_t *first, int32_t line) {
  if (start < 0 || (uint64_t)start > len) um_panic(UM_PANIC_WASM_HANDLE, line);
  uint64_t n = count < 0 ? (uint64_t)len - (uint64_t)start : (uint64_t)count;
  if (n > (uint64_t)len - (uint64_t)start) um_panic(UM_PANIC_WASM_HANDLE, line);
  *first = (size_t)start * elem;
  return (size_t)n * elem;
}

static uint8_t *memory_span(um_winstance *w, int64_t offset, size_t bytes) {
  if (!w->has_memory) {
    set_error("the module exports no memory");
    return NULL;
  }
  uint8_t *base = wasmtime_memory_data(w->ctx, &w->memory);
  size_t size = wasmtime_memory_data_size(w->ctx, &w->memory);
  if (offset < 0 || (uint64_t)offset > size || bytes > size - (size_t)offset) {
    set_error("out of bounds: %zu bytes at offset %lld, but the memory has %zu bytes", bytes, (long long)offset, size);
    return NULL;
  }
  return base + offset;
}

// インスタンスのメモリの offset バイト目から、UM の配列（data の start 番目から count 個、1個 elem バイト）へ写す。
int32_t um_wasm_read(int64_t h, int64_t offset, void *data, size_t len, size_t elem, int64_t start, int64_t count, int32_t line) {
  size_t first;
  size_t bytes = range_bytes(len, elem, start, count, &first, line);
  um_winstance *w = acquire(h, line);
  if (!w) return 0;
  uint8_t *p = memory_span(w, offset, bytes);
  if (p && bytes) memcpy((uint8_t *)data + first, p, bytes);
  release(w);
  return p != NULL;
}

// UM の配列からインスタンスのメモリへ写す（um_wasm_read の逆向き）。
int32_t um_wasm_write(int64_t h, int64_t offset, const void *data, size_t len, size_t elem, int64_t start, int64_t count, int32_t line) {
  size_t first;
  size_t bytes = range_bytes(len, elem, start, count, &first, line);
  um_winstance *w = acquire(h, line);
  if (!w) return 0;
  uint8_t *p = memory_span(w, offset, bytes);
  if (p && bytes) memcpy(p, (const uint8_t *)data + first, bytes);
  release(w);
  return p != NULL;
}

// メモリの大きさ（バイト数）。メモリがない・ほかのスレッドが使っている最中なら -1。
int64_t um_wasm_memory_size(int64_t h, int32_t line) {
  um_winstance *w = acquire(h, line);
  if (!w) return -1;
  int64_t n = -1;
  if (w->has_memory) n = (int64_t)wasmtime_memory_data_size(w->ctx, &w->memory);
  else set_error("the module exports no memory");
  release(w);
  return n;
}

// ---- そのほか ----

// テキスト形式（.wat）を wasm のバイナリにする。できたら 1（*out に新しい領域）、できなければ 0。
int32_t um_wasm_wat(const char *text, size_t len, unsigned char **out, size_t *out_len) {
  *out = NULL;
  *out_len = 0;
  wasm_byte_vec_t bin;
  wasmtime_error_t *err = wasmtime_wat2wasm(text, len, &bin);
  if (err) {
    set_error_from("could not parse the text format", err);
    return 0;
  }
  give_string(bin.data, bin.size, out, out_len);
  wasm_byte_vec_delete(&bin);
  return 1;
}

// このスレッドで最後に失敗した理由。
void um_wasm_error(unsigned char **out, size_t *len) { give_string(last_error, strlen(last_error), out, len); }
