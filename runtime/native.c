// UnifiedMozaic のネイティブ用ランタイム（最小）。
// 生成コードから呼ばれる関数だけを置く。
// 入力を受け取るのは、標準入力（um_read_line）と、プログラムが名前を指定したファイル（um_read_file）だけ。
// POSIX（macOS・Linux）と Windows（clang の MSVC ターゲット）で同じファイル。Windows の差は runtime/win32.h と、
// このファイルの `#ifdef _WIN32` の部分（標準入力、ファイル、コマンドライン引数、コンソールの文字コード）。

#ifdef _WIN32
// getenv などの標準の関数を「安全でない」とする MSVC の警告を出さない（どれも使い方を確かめてある）。
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include "win32.h"
#else
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <sched.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "f16.h"
#include "smallfloat.h"

void um_print_i64(int64_t v) { printf("%" PRId64 "\n", v); }

void um_print_u64(uint64_t v) { printf("%" PRIu64 "\n", v); }

void um_print_bool(int32_t v) { puts(v ? "true" : "false"); }

// JavaScript の Number.prototype.toString と同じ表記で出す（wasm 側の出力と一致させるため）。
// 元の値に戻せる最短の桁数を求め、ECMAScript の規則どおりに小数点と指数を置く。
// out は 80 バイト以上。書いた文字数を返す（終端の 0 は数えない）。
static int fmt_f64(double v, char *out) {
  if (isnan(v)) return snprintf(out, 80, "NaN");
  if (isinf(v)) return snprintf(out, 80, v > 0 ? "Infinity" : "-Infinity");
  if (v == 0) return snprintf(out, 80, "0");

  // "%.*e" の形（d.ddde±XX）で、元に戻せる最短の桁数を探す。
  char sci[40];
  for (int p = 1; p <= 17; p++) {
    snprintf(sci, sizeof sci, "%.*e", p - 1, v);
    if (strtod(sci, NULL) == v) break;
    // v が2の累乗のとき、元に戻せる範囲は絶対値の大きい側の方が広い。ちょうど中間の値を printf が
    // 小さい側へ丸めると戻せないので、最後の桁を1つ大きくした表記も試す。
    char up[40];
    snprintf(up, sizeof up, "%s", sci);
    char *e = strchr(up, 'e');
    char *d = e - 1;
    while (d >= up && (*d == '9' || *d == '.')) {
      if (*d == '9') *d = '0';
      d--;
    }
    if (d >= up && *d >= '0' && *d <= '8') {
      (*d)++;
      if (strtod(up, NULL) == v) {
        snprintf(sci, sizeof sci, "%s", up);
        break;
      }
    }
  }

  // 数字の列（digits、k 桁）と、小数点の位置 n（値 = 0.digits × 10^n）に分ける。
  char digits[24];
  int k = 0;
  const char *s = sci;
  int negative = *s == '-';
  if (negative) s++;
  for (; *s && *s != 'e'; s++) {
    if (*s != '.') digits[k++] = *s;
  }
  int n = atoi(s + 1) + 1;
  while (k > 1 && digits[k - 1] == '0') k--;

  int o = 0;
  if (negative) out[o++] = '-';
  if (k <= n && n <= 21) {
    for (int i = 0; i < k; i++) out[o++] = digits[i];
    for (int i = k; i < n; i++) out[o++] = '0';
  } else if (0 < n && n <= 21) {
    for (int i = 0; i < n; i++) out[o++] = digits[i];
    out[o++] = '.';
    for (int i = n; i < k; i++) out[o++] = digits[i];
  } else if (-6 < n && n <= 0) {
    out[o++] = '0';
    out[o++] = '.';
    for (int i = n; i < 0; i++) out[o++] = '0';
    for (int i = 0; i < k; i++) out[o++] = digits[i];
  } else {
    out[o++] = digits[0];
    if (k > 1) {
      out[o++] = '.';
      for (int i = 1; i < k; i++) out[o++] = digits[i];
    }
    o += snprintf(out + o, 80 - (size_t)o, "e%c%d", n - 1 < 0 ? '-' : '+', n - 1 < 0 ? 1 - n : n - 1);
  }
  out[o] = 0;
  return o;
}

void um_print_f64(double v) {
  char out[80];
  fmt_f64(v, out);
  puts(out);
}

// print 1回の出力（1行）は、ほかのスレッドの出力と混ざらないようにする（文字列は2回の stdio 呼び出しなので囲む）。
// ほかの print は1回の stdio 呼び出しで、stdio が呼び出しごとに鍵を取る。
void um_print_str(const char *data, size_t len) {
  flockfile(stdout);
  if (len > 0) fwrite(data, 1, len, stdout);
  putchar_unlocked('\n');
  funlockfile(stdout);
}

// ---- スレッド（thread.spawn / spawnEach / join） ----
// 生成コードが引数の塊（[0] に戻り値、[1..] に引数）を用意し、ここが OS のスレッドを起こして入口を呼ぶ。
// 同時に存在できるのは UM_THREAD_MAX 本まで（メインと kernel 用のスレッドプールは数えない）。
#define UM_THREAD_MAX 256
static _Atomic long um_threads_alive;    // 起動して、まだ合流していない本数
static _Atomic long um_threads_running;  // 起動して、まだ関数から戻っていない本数（停止のときに使う）
static pthread_t um_main_thread;
static size_t um_page;

__attribute__((constructor)) static void um_thread_init(void) {
  um_main_thread = pthread_self();
  um_page = (size_t)sysconf(_SC_PAGESIZE);
}

// 停止（終了コード 101）。ほかのスレッドが動いているときは、atexit の片付け（スレッドプールや Metal の状態）が
// 動いているスレッドの使う物を壊すので、_exit で終わる。stdout は先に流してある。runtime/metal.m も使う。
_Noreturn void um_exit_failure(void) {
  if (atomic_load(&um_threads_running) > 0 || !pthread_equal(pthread_self(), um_main_thread)) {
    fflush(stderr);
    _exit(101);
  }
  exit(101);
}

static _Atomic int um_panicking;

// 原因コードは src/codegen.ts の PANIC_* と揃えること。
_Noreturn void um_panic(int32_t code, int32_t line) {
  const char *msg = "unknown";
  switch (code) {
    case 1: msg = "index out of bounds"; break;
    case 2: msg = "division by zero"; break;
    case 3: msg = "shift amount too large"; break;
    case 4: msg = "pop from an empty array"; break;
    case 5: msg = "out of memory"; break;
    case 6: msg = "dispatch count exceeds the length of a written Buffer"; break;
    case 7: msg = "the GPU is not available"; break;
    case 8: msg = "could not start a thread"; break;
    case 9: msg = "readLine and inputReady can only be called from the main thread"; break;
    case 10: msg = "invalid group size (it must be 1 to 1024, and groups times size must fit in u32)"; break;
    case 11: msg = "window functions can only be called from the main thread"; break;
    case 12: msg = "invalid window call (unknown window, a size that is not positive, or a Buffer that is too short)"; break;
  }
  // 複数のスレッドが同時に停止しても、メッセージは1回だけ。2番目以降は、終わるまで寝る。
  if (atomic_exchange(&um_panicking, 1)) {
    for (;;) pause();
  }
  fflush(stdout);
  fprintf(stderr, "panic: %s (line %d)\n", msg, (int)line);
  um_exit_failure();
}

typedef void (*um_thread_fn)(void *block);
typedef struct {
  pthread_t id;
  um_thread_fn fn;
  void *block;
} um_thread_one;
typedef struct {
  size_t n;
  um_thread_one t[];
} um_thread_set;

static void *um_thread_main(void *arg) {
  um_thread_one *t = arg;
  t->fn(t->block);
  atomic_fetch_sub(&um_threads_running, 1);
  return NULL;
}

// スタックの大きさ。macOS の pthread の既定（512 KiB）では探索の再帰が足りないので、64 MiB を予約する
// （使った分だけが実際のメモリになる）。UM_THREAD_STACK_MB で変えられる。
static size_t um_thread_stack(void) {
  static _Atomic size_t cached;
  size_t v = atomic_load(&cached);
  if (v) return v;
  size_t mb = 64;
  const char *env = getenv("UM_THREAD_STACK_MB");
  if (env) {
    char *end;
    long x = strtol(env, &end, 10);
    if (end != env && *end == 0 && x >= 1 && x <= 4096) mb = (size_t)x;
  }
  v = mb << 20;
  if (v < (size_t)PTHREAD_STACK_MIN) v = (size_t)PTHREAD_STACK_MIN;
  atomic_store(&cached, v);
  return v;
}

// n 本のスレッドを起こす。k 本目は fn(blocks + k * size) を実行する。持ち手を返す（n が 0 でもよい）。
void *um_thread_start(um_thread_fn fn, char *blocks, size_t size, size_t n, int32_t line) {
  if (n > UM_THREAD_MAX || atomic_fetch_add(&um_threads_alive, (long)n) + (long)n > UM_THREAD_MAX) um_panic(8, line);
  um_thread_set *set = malloc(sizeof(um_thread_set) + n * sizeof(um_thread_one));
  if (!set) um_panic(5, line);
  set->n = n;
  pthread_attr_t attr;
  if (pthread_attr_init(&attr) != 0) um_panic(8, line);
  pthread_attr_setstacksize(&attr, um_thread_stack());
  for (size_t k = 0; k < n; k++) {
    set->t[k].fn = fn;
    set->t[k].block = blocks + k * size;
    atomic_fetch_add(&um_threads_running, 1);
    if (pthread_create(&set->t[k].id, &attr, um_thread_main, &set->t[k]) != 0) {
      atomic_fetch_sub(&um_threads_running, 1);
      um_panic(8, line);
    }
  }
  pthread_attr_destroy(&attr);
  return set;
}

// 持ち手のスレッドがすべて終わるのを待ち、持ち手を手放す。スレッドがしたことは、ここから戻ったあとすべて見える。
void um_thread_join(void *h) {
  um_thread_set *set = h;
  for (size_t k = 0; k < set->n; k++) pthread_join(set->t[k].id, NULL);
  atomic_fetch_sub(&um_threads_alive, (long)set->n);
  free(set);
}

int32_t um_thread_cpus(void) {
  long n = sysconf(_SC_NPROCESSORS_ONLN);
  return n < 1 ? 1 : n > INT32_MAX ? INT32_MAX : (int32_t)n;
}

// `thread.yield()`: CPU をほかの動けるスレッドに譲る（なければすぐ戻る）。RacyBuffer の印を見ながら待つループ用。
void um_thread_yield(void) { sched_yield(); }

// readLine / inputReady を呼んでよいか（メインスレッドか）。
int32_t um_is_main(void) { return pthread_equal(pthread_self(), um_main_thread) != 0; }

// ---- ヒープ ----
// 要素数 × 要素サイズの掛け算は、あふれを必ず検査する。

static size_t checked_bytes(size_t count, size_t elem_size) {
  size_t bytes;
  if (__builtin_mul_overflow(count, elem_size, &bytes)) um_panic(5, 0);
  return bytes;
}

void *um_alloc(size_t count, size_t elem_size) {
  size_t bytes = checked_bytes(count, elem_size);
  if (bytes == 0) return NULL;
  void *p = malloc(bytes);
  if (!p) um_panic(5, 0);
  return p;
}

// GPU とコピーなしで共有できる領域: ページ境界に揃え、大きさもページ単位に切り上げ、ゼロで埋める。
// Metal を使うビルドでは、Buffer はすべてこれで確保する（runtime/metal.m がこの前提で領域を GPU に渡す）。
void *um_alloc_shared(size_t count, size_t elem_size) {
  size_t bytes = checked_bytes(count, elem_size);
  size_t page = (size_t)sysconf(_SC_PAGESIZE);
  if (bytes > SIZE_MAX - page) um_panic(5, 0);
  size_t rounded = (bytes + page - 1) / page * page;
  if (rounded == 0) rounded = page;
#ifdef _WIN32
  // Windows には Metal がないので、ページ境界に揃える必要はない（_aligned_malloc の領域は free で解放できない）。
  void *p = calloc(1, rounded);
  if (!p) um_panic(5, 0);
#else
  void *p = NULL;
  if (posix_memalign(&p, page, rounded) != 0 || !p) um_panic(5, 0);
  memset(p, 0, rounded);
#endif
  return p;
}

void *um_alloc_zeroed(size_t count, size_t elem_size) {
  size_t bytes = checked_bytes(count, elem_size);
  if (bytes == 0) return NULL;
  void *p = calloc(1, bytes);
  if (!p) um_panic(5, 0);
  return p;
}

// Metal のランタイムが設定する呼び出し口。GPU 側で待機させてある kernel が使っている領域を解放する前に、
// その待機を片付ける必要がある（runtime/metal.m の um_on_free）。Metal を使わないビルドでは NULL のまま。
// Metal の準備で一度だけ設定する。どのスレッドからも読むのでアトミックにする。
// GPU に渡す領域はすべてページ境界から始まる（um_alloc_shared）ので、そうでない領域（文字列や配列の多く）は呼ばない。
typedef void (*um_free_fn)(void *);
_Atomic(um_free_fn) um_free_hook = NULL;

void um_free(void *p) {
  if (p && ((uintptr_t)p & (um_page - 1)) == 0) {
    um_free_fn hook = atomic_load_explicit(&um_free_hook, memory_order_acquire);
    if (hook) hook(p);
  }
  free(p);
}

// 伸ばした後の容量（空なら4、それ以外は2倍）。
// 配列の構造体そのものを受け取らない形にしてある。構造体のアドレスをランタイムに渡すと、
// コンパイラが長さや容量をレジスタに置けなくなり、push が遅くなるため。
size_t um_grow_cap(size_t cap) {
  size_t next;
  if (cap == 0) return 4;
  if (__builtin_mul_overflow(cap, (size_t)2, &next)) um_panic(5, 0);
  return next;
}

// 要素の領域を new_cap 個ぶんに広げ、新しい場所を返す。len は使っている要素数（wasm 側と引数を揃えている）。
void *um_grow_data(void *data, size_t len, size_t new_cap, size_t elem_size) {
  (void)len;
  void *p = realloc(data, checked_bytes(new_cap, elem_size));
  if (!p) um_panic(5, 0);
  return p;
}

// ---- kernel の並列実行 ----
// dispatch は、添字の範囲 [0, n) を小さな区間に分け、複数のスレッドで実行する。
// kernel が書けるのは自分の添字の要素だけなので、区間どうしは競合しない。
// 実行順によらず結果は同じになる。

typedef void (*um_range_fn)(uint32_t begin, uint32_t end, void *ctx);

// これより少ない要素数では、ふつうはスレッドを起こすコストの方が大きいので、その場で実行する。
// ただし、1要素あたりの計算が重い kernel は、個数が少なくても分けたほうが速い。
// そこで、その場で実行したときの時間を測っておき、次に同じ kernel が来たとき、
// 見込みの時間が UM_PARALLEL_MIN_SECONDS 以上なら複数のスレッドで実行する。
// （ワーカーは仕事のあとしばらく起きたまま待つので、続けて呼ばれるかぎり、分ける手間は数マイクロ秒で済む。）
#define UM_PARALLEL_MIN 4096
#define UM_PARALLEL_MIN_SECONDS 25e-6
#define UM_MAX_THREADS 64
#define UM_SMALL_TRACKED 8

// 少ない個数で実行した kernel（区間を実行する関数ごと）の、1要素あたりの時間。
// 呼び出したスレッドごとに持つ（thread.spawn のスレッドが同時に dispatch してもよいように）。
static _Thread_local struct {
  void *fn;
  double per_elem;
  uint32_t parallel_runs;  // 複数のスレッドで続けて実行した回数（ときどき測り直すため）
} um_small[UM_SMALL_TRACKED];
static _Thread_local unsigned um_small_next;

static double um_seconds(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

// ワーカーは、仕事を終えたあとしばらく（UM_POOL_SPIN_US マイクロ秒）は寝ずに次の仕事を待つ。
// 寝ているスレッドを起こすには数十マイクロ秒かかるので、dispatch を短い間隔でくり返す使い方
// （探索の末端の評価など）では、起きたまま待つほうがずっと速い。次が来なければ寝る。
// 環境変数 UM_POOL_SPIN_US で変えられる（0 なら待たずにすぐ寝る）。
#define UM_POOL_SPIN_US 100

static struct {
  pthread_mutex_t lock;
  pthread_cond_t wake;
  pthread_cond_t done;
  pthread_t threads[UM_MAX_THREADS];
  int workers;        // 呼び出したスレッド以外の数
  double spin_seconds;
  _Atomic int busy;   // いま仕事を出しているスレッドがいるか（同時に出せる仕事は1つ）
  _Atomic int stopping;
  _Atomic unsigned long job;   // 仕事の通し番号。増えたら新しい仕事がある
  _Atomic unsigned long open;  // 受け付け中の仕事の番号（0 なら受け付けていない）
  _Atomic int inside;          // いま pool の中身（fn など）を見ているワーカーの数
  _Atomic uint64_t finished;   // いまの仕事で、実行し終えた要素の数
  _Atomic int sleepers;        // 寝ているワーカーの数（0 なら、起こすための合図を省ける）
  um_range_fn fn;
  void *ctx;
  uint64_t n;
  uint64_t chunk;
  _Atomic uint64_t next;
} pool = {.lock = PTHREAD_MUTEX_INITIALIZER, .wake = PTHREAD_COND_INITIALIZER, .done = PTHREAD_COND_INITIALIZER};

// 寝ずに待つあいだ、CPU に「待っている」ことを伝える（電力と、同じコアのもう一方のスレッドのため）。
static inline void um_cpu_relax(void) {
#if defined(__aarch64__) || defined(__arm__)
  __asm__ volatile("yield");
#elif defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#endif
}

// 区間を1つずつ取って実行する。取り出しはアトミックなので、同じ区間を2回実行することはない。
static void pool_run_chunks(void) {
  for (;;) {
    uint64_t begin = atomic_fetch_add(&pool.next, pool.chunk);
    if (begin >= pool.n) return;
    uint64_t end = begin + pool.chunk;
    if (end > pool.n) end = pool.n;
    pool.fn((uint32_t)begin, (uint32_t)end, pool.ctx);
    if (atomic_fetch_add(&pool.finished, end - begin) + (end - begin) == pool.n) return;
  }
}

static void *pool_worker(void *arg) {
  (void)arg;
  unsigned long seen = 0;
  for (;;) {
    // 新しい仕事を待つ。まずは寝ずに待ち、来なければ寝る。
    int found = 0;
    if (pool.spin_seconds > 0) {
      double until = um_seconds() + pool.spin_seconds;
      for (unsigned i = 0;; i++) {
        if (atomic_load(&pool.job) != seen || atomic_load(&pool.stopping)) {
          found = 1;
          break;
        }
        um_cpu_relax();
        if ((i & 63) == 63 && um_seconds() > until) break;
      }
    }
    if (!found) {
      pthread_mutex_lock(&pool.lock);
      // sleepers を増やしてから仕事の番号を確かめる。仕事を出す側は、番号を増やしてから sleepers を見るので、
      // 「寝る直前に仕事が来たのに起こされない」ことは起きない。
      atomic_fetch_add(&pool.sleepers, 1);
      while (atomic_load(&pool.job) == seen && !atomic_load(&pool.stopping)) pthread_cond_wait(&pool.wake, &pool.lock);
      atomic_fetch_sub(&pool.sleepers, 1);
      pthread_mutex_unlock(&pool.lock);
    }
    if (atomic_load(&pool.stopping)) break;
    seen = atomic_load(&pool.job);
    // 仕事に加わる。仕事を出した側は、全部の要素が終わった時点で先へ進む（全員の参加を待たない）ので、
    // 遅れて来たワーカーは、その仕事がまだ受け付け中かを確かめてからでないと pool の中身を読めない。
    // inside を増やしてから open を見る。出した側は open を閉じてから inside が 0 になるのを待つので、
    // 次の仕事の中身を書いている最中に読むことはない。
    atomic_fetch_add(&pool.inside, 1);
    if (atomic_load(&pool.open) == seen) {
      pool_run_chunks();
      if (atomic_load(&pool.finished) == pool.n) {
        // 仕事を出した側が寝て待っているかもしれないので、知らせる。
        pthread_mutex_lock(&pool.lock);
        pthread_cond_signal(&pool.done);
        pthread_mutex_unlock(&pool.lock);
      }
    }
    atomic_fetch_sub(&pool.inside, 1);
  }
  return NULL;
}

// プロセスの終了時にワーカーを止めて回収する。
static void pool_stop(void) {
  pthread_mutex_lock(&pool.lock);
  atomic_store(&pool.stopping, 1);
  pthread_cond_broadcast(&pool.wake);
  pthread_mutex_unlock(&pool.lock);
  for (int i = 0; i < pool.workers; i++) pthread_join(pool.threads[i], NULL);
  pool.workers = 0;
}

// 使うスレッド数（呼び出したスレッドを含む）。環境変数 UM_THREADS で上書きできる。
// 複数のスレッドから同時に呼ばれうるので、準備は pthread_once で1回だけ行う。
static pthread_once_t pool_once = PTHREAD_ONCE_INIT;
static void pool_init(void);
static int pool_start(void) {
  pthread_once(&pool_once, pool_init);
  return pool.workers + 1;
}
static void pool_init(void) {
  long want = sysconf(_SC_NPROCESSORS_ONLN);
  const char *env = getenv("UM_THREADS");
  if (env) {
    char *end;
    long v = strtol(env, &end, 10);
    if (end != env && *end == 0 && v >= 1) want = v;
  }
  if (want < 1) want = 1;
  if (want > UM_MAX_THREADS) want = UM_MAX_THREADS;
  long spin_us = UM_POOL_SPIN_US;
  const char *spin = getenv("UM_POOL_SPIN_US");
  if (spin) {
    char *end;
    long v = strtol(spin, &end, 10);
    if (end != spin && *end == 0 && v >= 0 && v <= 1000000) spin_us = v;
  }
  pool.spin_seconds = (double)spin_us * 1e-6;
  for (long i = 0; i < want - 1; i++) {
    if (pthread_create(&pool.threads[pool.workers], NULL, pool_worker, NULL) != 0) break;
    pool.workers++;
  }
  if (pool.workers > 0) atexit(pool_stop);
}

void um_parallel_for(um_range_fn fn, void *ctx, uint32_t n) {
  if (n == 0) return;
  uint64_t min_chunk = 256;
  if (n < UM_PARALLEL_MIN) {
    int k = -1;
    for (int i = 0; i < UM_SMALL_TRACKED; i++) {
      if (um_small[i].fn == (void *)fn) k = i;
    }
    // 重いと分かっている kernel は、下の並列の経路へ進む。64 回に1回は、その場で実行して測り直す。
    int heavy = k >= 0 && um_small[k].per_elem * n >= UM_PARALLEL_MIN_SECONDS && n >= 2;
    if (heavy && ++um_small[k].parallel_runs < 64) {
      min_chunk = 1;
    } else {
      double t0 = um_seconds();
      fn(0, n, ctx);
      double per_elem = (um_seconds() - t0) / n;
      if (k < 0) {
        k = (int)(um_small_next++ % UM_SMALL_TRACKED);
        um_small[k].fn = (void *)fn;
        um_small[k].per_elem = per_elem;
      } else {
        um_small[k].per_elem = um_small[k].per_elem * 0.5 + per_elem * 0.5;
      }
      um_small[k].parallel_runs = 0;
      return;
    }
  }
  int threads = pool_start();
  // プールは同時に1つの仕事だけ。ほかのスレッド（thread.spawn）が使っている最中なら、待たずに自分だけで実行する
  // （探索のスレッドが全コアを使っているときに割り込ませても速くならない）。
  if (threads <= 1 || atomic_exchange(&pool.busy, 1)) {
    fn(0, n, ctx);
    return;
  }

  // 前の仕事は閉じてあり、ワーカーは誰も pool の中身を見ていない。中身を書いてから受け付けを始める。
  pool.fn = fn;
  pool.ctx = ctx;
  pool.n = n;
  // 1スレッドあたり数個の区間に分ける。仕事の重さに偏りがあっても、空いたスレッドが残りを引き受けられる。
  pool.chunk = n / ((uint64_t)threads * 8);
  if (pool.chunk < min_chunk) pool.chunk = min_chunk;
  atomic_store(&pool.next, 0);
  atomic_store(&pool.finished, 0);
  unsigned long id = atomic_load(&pool.job) + 1;
  atomic_store(&pool.open, id);
  atomic_store(&pool.job, id);
  if (atomic_load(&pool.sleepers) > 0) {
    pthread_mutex_lock(&pool.lock);
    pthread_cond_broadcast(&pool.wake);
    pthread_mutex_unlock(&pool.lock);
  }

  pool_run_chunks(); // 呼び出したスレッドも働く

  // 全部の要素が終わるのを待つ。たいていすぐ終わるので、まずは寝ずに待つ。
  // OS に止められているワーカーがいても、その分はほかのスレッドが引き受けるので、全員を待つ必要はない。
  if (pool.spin_seconds > 0) {
    double until = um_seconds() + pool.spin_seconds;
    for (unsigned i = 0; atomic_load(&pool.finished) != n; i++) {
      um_cpu_relax();
      if ((i & 63) == 63 && um_seconds() > until) break;
    }
  }
  if (atomic_load(&pool.finished) != n) {
    pthread_mutex_lock(&pool.lock);
    while (atomic_load(&pool.finished) != n) pthread_cond_wait(&pool.done, &pool.lock);
    pthread_mutex_unlock(&pool.lock);
  }
  // 受け付けを閉じ、中身を見ているワーカーがいなくなるのを待つ（区間を取りに来ただけなので、すぐいなくなる）。
  atomic_store(&pool.open, 0);
  while (atomic_load(&pool.inside) != 0) um_cpu_relax();
  atomic_store(&pool.busy, 0);
}

// ---- 一括の書き込み（xs.fill） ----

// 要素 [a, b) を、ビット列 bits の下位 size バイトにする（1 つのスレッドで）。
// racy（RacyBuffer）では、ほかのスレッドが同時に読み書きするので、要素ごとに不可分な relaxed の書き込みにする
// （ちぎれた値は見えない）。そうでなければ普通に書く（全バイトが同じ値なら memset、ほかはベクトル化される）。
static void um_fill_range(void *data, uint64_t a, uint64_t b, int32_t size, uint64_t bits, int32_t racy) {
  if (a >= b) return;
  uint64_t pattern = bits & (size == 8 ? ~(uint64_t)0 : (((uint64_t)1 << (size * 8)) - 1));
  int same = 1;
  for (int k = 1; k < size; k++) same &= ((pattern >> (k * 8)) & 0xff) == (pattern & 0xff);
  if (!racy && same) {
    memset((char *)data + a * (uint64_t)size, (int)(pattern & 0xff), (size_t)((b - a) * (uint64_t)size));
    return;
  }
  switch (size) {
    case 1: {
      uint8_t *d = data;
      for (uint64_t i = a; i < b; i++) __atomic_store_n(&d[i], (uint8_t)pattern, __ATOMIC_RELAXED);
      return;
    }
    case 2: {
      uint16_t *d = data;
      if (racy) for (uint64_t i = a; i < b; i++) __atomic_store_n(&d[i], (uint16_t)pattern, __ATOMIC_RELAXED);
      else for (uint64_t i = a; i < b; i++) d[i] = (uint16_t)pattern;
      return;
    }
    case 4: {
      uint32_t *d = data;
      if (racy) for (uint64_t i = a; i < b; i++) __atomic_store_n(&d[i], (uint32_t)pattern, __ATOMIC_RELAXED);
      else for (uint64_t i = a; i < b; i++) d[i] = (uint32_t)pattern;
      return;
    }
    default: {
      uint64_t *d = data;
      if (racy) for (uint64_t i = a; i < b; i++) __atomic_store_n(&d[i], pattern, __ATOMIC_RELAXED);
      else for (uint64_t i = a; i < b; i++) d[i] = pattern;
      return;
    }
  }
}

typedef struct {
  void *data;
  uint64_t a, b, per;  // per: 1 区間の要素数
  uint64_t bits;
  int32_t size, racy;
} um_fill_job;

static void um_fill_chunks(uint32_t lo, uint32_t hi, void *ctx) {
  um_fill_job *j = ctx;
  uint64_t from = j->a + (uint64_t)lo * j->per;
  uint64_t to = j->a + (uint64_t)hi * j->per;
  if (to > j->b) to = j->b;
  um_fill_range(j->data, from, to, j->size, j->bits, j->racy);
}

// 16 MiB 以上なら、4 KiB ずつの区間に分けてスレッドプール（um_parallel_for）で書く（区間が 4096 個以上になり、
// 並列の経路に直接入る）。置換表のような大きな表を消す時間を、メモリの帯域の限りまで縮める。
#define UM_FILL_PARALLEL_BYTES (16u << 20)
void um_fill(void *data, uint64_t a, uint64_t b, int32_t size, uint64_t bits, int32_t racy) {
  if (a >= b) return;
  uint64_t bytes = (b - a) * (uint64_t)size;
  if (bytes < UM_FILL_PARALLEL_BYTES) {
    um_fill_range(data, a, b, size, bits, racy);
    return;
  }
  um_fill_job j = {data, a, b, 4096 / (uint64_t)size, bits, size, racy};
  uint64_t n = (b - a + j.per - 1) / j.per;
  while (n > 0x7fffffffu) {  // 8 TiB を超える範囲は区間を大きくする
    j.per *= 2;
    n = (b - a + j.per - 1) / j.per;
  }
  um_parallel_for(um_fill_chunks, &j, (uint32_t)n);
}

// ---- 文字列 ----
// 文字列は UTF-8 のバイト列。長さ 0 のときデータの場所は NULL でよい。

// a と b をつないだ新しい領域を返す。長さの合計があふれる場合は停止する。
char *um_str_concat(const char *a, size_t alen, const char *b, size_t blen) {
  if (alen > SIZE_MAX - blen) um_panic(5, 0);
  char *out = um_alloc(alen + blen, 1);
  if (alen > 0) memcpy(out, a, alen);
  if (blen > 0) memcpy(out + alen, b, blen);
  return out;
}

int32_t um_str_eq(const char *a, size_t alen, const char *b, size_t blen) {
  if (alen != blen) return 0;
  return alen == 0 || memcmp(a, b, alen) == 0;
}

static char *str_from(const char *text, int n, size_t *len) {
  *len = (size_t)n;
  char *out = um_alloc((size_t)n, 1);
  memcpy(out, text, (size_t)n);
  return out;
}

char *um_str_from_i64(int64_t v, size_t *len) {
  char buf[32];
  return str_from(buf, snprintf(buf, sizeof buf, "%" PRId64, v), len);
}

char *um_str_from_u64(uint64_t v, size_t *len) {
  char buf[32];
  return str_from(buf, snprintf(buf, sizeof buf, "%" PRIu64, v), len);
}

char *um_str_from_f64(double v, size_t *len) {
  char buf[80];
  return str_from(buf, fmt_f64(v, buf), len);
}

// ---- OS の違い（ファイル、標準入力、コンソール） ----
// ファイルの名前は UTF-8 の文字列（0 で終わる写し）。Windows では UTF-16 に直して開く（ANSI のコードページを通さない）。
// 開き方 how: 0 = 読む、1 = 作って（あれば中身を捨てて）書く、2 = 終わりに書き足す（なければ作る）。
// os_stat / os_fstat は種類を返す: 0 = 調べられない、1 = ふつうのファイル、2 = フォルダ、3 = そのほか。
#define OS_NONE 0
#define OS_FILE 1
#define OS_DIR 2
#define OS_OTHER 3

#ifdef _WIN32

// 0 で終わる UTF-8 を UTF-16 にする（新しく確保する）。UTF-8 として正しくなければ NULL（別の名前に化けたファイルを開かない）。
static WCHAR *os_wide(const char *z) {
  int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, z, -1, NULL, 0);
  if (n <= 0) return NULL;
  WCHAR *w = malloc((size_t)n * sizeof(WCHAR));
  if (!w) um_panic(5, 0);
  MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, z, -1, w, n);
  return w;
}

static int os_open(const char *z, int how) {
  WCHAR *w = os_wide(z);
  if (!w) return -1;
  int flags = how == 0 ? _O_RDONLY : _O_WRONLY | _O_CREAT | (how == 1 ? _O_TRUNC : _O_APPEND);
  int fd = _wopen(w, flags | _O_BINARY | _O_NOINHERIT, _S_IREAD | _S_IWRITE);
  free(w);
  return fd;
}

// n は UM_IO_CHUNK 以下で呼ぶ（unsigned int に収まる）。
static ssize_t os_read(int fd, void *buf, size_t n) { return _read(fd, buf, (unsigned)n); }
static ssize_t os_write(int fd, const void *buf, size_t n) { return _write(fd, buf, (unsigned)n); }
static ssize_t os_pread(int fd, void *buf, size_t n, int64_t offset) {
  if (_lseeki64(fd, offset, SEEK_SET) < 0) return -1;
  return _read(fd, buf, (unsigned)n);
}
static int os_close(int fd) { return _close(fd); }

static int os_kind(unsigned short mode) {
  return (mode & _S_IFMT) == _S_IFREG ? OS_FILE : (mode & _S_IFMT) == _S_IFDIR ? OS_DIR : OS_OTHER;
}

static int os_fstat(int fd, int64_t *size) {
  struct _stat64 st;
  if (_fstat64(fd, &st) != 0) return OS_NONE;
  *size = st.st_size;
  return os_kind(st.st_mode);
}

static int os_stat(const char *z, int64_t *size) {
  WCHAR *w = os_wide(z);
  if (!w) return OS_NONE;
  struct _stat64 st;
  int r = _wstat64(w, &st);
  free(w);
  if (r != 0) return OS_NONE;
  *size = st.st_size;
  return os_kind(st.st_mode);
}

// 標準入力。コンソールからは ReadConsoleW で UTF-16 を読んで UTF-8 にする（コンソールの文字コードによらず、
// 文字列は UTF-8 になる）。パイプやファイルからはバイト列のまま読む。Windows の標準入力に「待たない」設定はない。
static struct {
  int ready;
  HANDLE h;
  DWORD type;   // GetFileType の値
  int console;  // コンソールから読む
  int started;  // パイプやファイルから一度でも読んだ（先頭の BOM を確かめた）
  unsigned char text[4096 * 3];  // UTF-8 にして、まだ渡していない分 [at, n)
  size_t at, n;
  WCHAR high;   // 前回の終わりの上位サロゲート（次に読んだ分の先頭とつなぐ）
} um_win_in;

static void os_stdin_init(void) {
  if (um_win_in.ready) return;
  um_win_in.ready = 1;
  um_win_in.h = GetStdHandle(STD_INPUT_HANDLE);
  DWORD mode;
  um_win_in.type = um_win_in.h && um_win_in.h != INVALID_HANDLE_VALUE ? GetFileType(um_win_in.h) : FILE_TYPE_UNKNOWN;
  um_win_in.console = um_win_in.type == FILE_TYPE_CHAR && GetConsoleMode(um_win_in.h, &mode);
}

// 標準入力から最大 want バイト読む。読めたバイト数を返す。入力の終わり、または読めないときは 0。
static ssize_t os_read_stdin(unsigned char *buf, size_t want) {
  os_stdin_init();
  if (!um_win_in.console) {
    for (;;) {
      int n = _read(0, buf, (unsigned)want);
      if (n < 0 && errno == EINTR) continue;
      if (n <= 0) return 0;  // 書く側が閉じたパイプは 0（入力の終わり）
      // Windows PowerShell 5.1 は、パイプでプログラムに渡す文字列の先頭に UTF-8 の BOM を付ける。
      // 入力の先頭の BOM は1回だけ読み飛ばす（1行目が "﻿usi" のようにならないように）。
      if (!um_win_in.started) {
        um_win_in.started = 1;
        if (n >= 3 && buf[0] == 0xEF && buf[1] == 0xBB && buf[2] == 0xBF) {
          memmove(buf, buf + 3, (size_t)n - 3);
          n -= 3;
          if (n == 0) continue;
        }
      }
      return n;
    }
  }
  while (um_win_in.at == um_win_in.n) {
    WCHAR w[4096];
    DWORD k = 0, got = 0;
    if (um_win_in.high) {
      w[k++] = um_win_in.high;
      um_win_in.high = 0;
    }
    if (!ReadConsoleW(um_win_in.h, w + k, 4096 - k, &got, NULL)) return 0;
    if (got == 0) return 0;
    k += got;
    // Ctrl+Z（0x1A）で始まる行は入力の終わり（Windows のコンソールの決まり）。
    if (w[0] == 0x1A) return 0;
    if (w[k - 1] >= 0xD800 && w[k - 1] <= 0xDBFF) um_win_in.high = w[--k];
    int n = k ? WideCharToMultiByte(CP_UTF8, 0, w, (int)k, (char *)um_win_in.text, (int)sizeof um_win_in.text, NULL, NULL) : 0;
    um_win_in.at = 0;
    um_win_in.n = n > 0 ? (size_t)n : 0;
  }
  size_t n = um_win_in.n - um_win_in.at;
  if (n > want) n = want;
  memcpy(buf, um_win_in.text + um_win_in.at, n);
  um_win_in.at += n;
  return (ssize_t)n;
}

// 標準入力を待たずに読めるか: 1 = 読める（または終わっている）、0 = まだ何も届いていない、-1 = 確かめられない。
static int os_stdin_ready(void) {
  os_stdin_init();
  if (um_win_in.console) {
    if (um_win_in.at < um_win_in.n) return 1;
    // コンソールは Enter が押されるまで1行を渡さない。入力の列に Enter があれば、ReadConsoleW は待たずに戻る。
    DWORD count = 0;
    if (!GetNumberOfConsoleInputEvents(um_win_in.h, &count)) return -1;
    if (count == 0) return 0;
    INPUT_RECORD *recs = malloc(count * sizeof *recs);
    if (!recs) um_panic(5, 0);
    DWORD got = 0;
    int ready = 0;
    if (PeekConsoleInputW(um_win_in.h, recs, count, &got)) {
      for (DWORD i = 0; i < got && !ready; i++) {
        ready = recs[i].EventType == KEY_EVENT && recs[i].Event.KeyEvent.bKeyDown && recs[i].Event.KeyEvent.wVirtualKeyCode == VK_RETURN;
      }
    }
    free(recs);
    return ready;
  }
  if (um_win_in.type == FILE_TYPE_PIPE) {
    DWORD avail = 0;
    // 書く側が閉じたパイプでは失敗する。そのときの読み込みは待たずに「入力の終わり」を返すので、-1 でよい。
    if (!PeekNamedPipe(um_win_in.h, NULL, 0, NULL, &avail, NULL)) return -1;
    return avail > 0;
  }
  return 1;  // ファイルや NUL は待たずに読める
}

// 改行を "\r\n" に変えない（出力を macOS・Linux・wasm と同じバイト列にする。コンソールは "\n" だけで改行する）。
// 文字列は UTF-8 なので、コンソールに出すときはコンソールの出力の文字コードを UTF-8 にし、終了時に戻す。
static UINT um_win_saved_cp;
static void um_win_restore_cp(void) { SetConsoleOutputCP(um_win_saved_cp); }

__attribute__((constructor)) static void um_win_init(void) {
  _setmode(0, _O_BINARY);
  _setmode(1, _O_BINARY);
  _setmode(2, _O_BINARY);
  DWORD mode;
  if (GetConsoleMode(GetStdHandle(STD_OUTPUT_HANDLE), &mode) || GetConsoleMode(GetStdHandle(STD_ERROR_HANDLE), &mode)) {
    um_win_saved_cp = GetConsoleOutputCP();
    if (um_win_saved_cp != CP_UTF8 && SetConsoleOutputCP(CP_UTF8)) atexit(um_win_restore_cp);
  }
}

#else

static int os_open(const char *z, int how) {
  int flags = how == 0 ? O_RDONLY : O_WRONLY | O_CREAT | (how == 1 ? O_TRUNC : O_APPEND);
  return open(z, flags | O_CLOEXEC, 0644);
}

static ssize_t os_read(int fd, void *buf, size_t n) { return read(fd, buf, n); }
static ssize_t os_write(int fd, const void *buf, size_t n) { return write(fd, buf, n); }
static ssize_t os_pread(int fd, void *buf, size_t n, int64_t offset) { return pread(fd, buf, n, (off_t)offset); }
static int os_close(int fd) { return close(fd); }

static int os_kind(mode_t mode) { return S_ISREG(mode) ? OS_FILE : S_ISDIR(mode) ? OS_DIR : OS_OTHER; }

static int os_fstat(int fd, int64_t *size) {
  struct stat st;
  if (fstat(fd, &st) != 0) return OS_NONE;
  *size = (int64_t)st.st_size;
  return os_kind(st.st_mode);
}

static int os_stat(const char *z, int64_t *size) {
  struct stat st;
  if (stat(z, &st) != 0) return OS_NONE;
  *size = (int64_t)st.st_size;
  return os_kind(st.st_mode);
}

#endif

// ---- 入出力（readLine / inputReady / clock / readFile / writeFile） ----

// 標準入力は stdio を通さず、自前のバッファで read(2) する。
// 「バッファに1行あるか」を自分で分かるようにして、um_input_ready が正確に答えられるようにするため。
// [start, end) がまだ渡していないバイト。行の長さに上限はない（バッファを伸ばす）。
// seen は、start から数えて「改行がない」と分かっているバイト数（同じ所を何度も探さないため）。
static struct {
  unsigned char *buf;
  size_t start, end, cap, seen;
  int eof;
} um_in;

// 1回の read / write に頼む大きさの上限。macOS は INT_MAX を超える大きさを EINVAL で断るので、
// 大きな読み書きは必ずこの大きさ以下に分けて繰り返す。
#define UM_IO_CHUNK ((size_t)1 << 30)

// まだ渡していない分の中の最初の改行。なければ NULL。
static const unsigned char *in_newline(void) {
  size_t have = um_in.end - um_in.start;
  if (have <= um_in.seen) return NULL;
  const unsigned char *nl = memchr(um_in.buf + um_in.start + um_in.seen, '\n', have - um_in.seen);
  if (!nl) um_in.seen = have;
  return nl;
}

// 標準入力からもう少し読む。読めたら 1。入力の終わり（または読めないとき）は 0 を返し、以後は読まない。
// 標準入力が「待たない」設定（O_NONBLOCK。親のプロセスが付けたものを引き継ぐことがある）で、まだ何も
// 届いていないときは、wait が 0 でなければ届くまで待ち、0 なら -1 を返す（入力の終わりではない）。
static int in_fill(int wait) {
  if (um_in.eof) return 0;
  if (um_in.start > 0) {
    // 渡し終えた分を詰める。
    memmove(um_in.buf, um_in.buf + um_in.start, um_in.end - um_in.start);
    um_in.end -= um_in.start;
    um_in.start = 0;
  }
  if (um_in.end == um_in.cap) {
    size_t cap = um_in.cap == 0 ? 4096 : um_in.cap;
    if (um_in.cap != 0 && __builtin_mul_overflow(um_in.cap, (size_t)2, &cap)) um_panic(5, 0);
    unsigned char *p = realloc(um_in.buf, cap);
    if (!p) um_panic(5, 0);
    um_in.buf = p;
    um_in.cap = cap;
  }
  size_t want = um_in.cap - um_in.end;
  if (want > UM_IO_CHUNK) want = UM_IO_CHUNK;
#ifdef _WIN32
  // Windows の標準入力に「待たない」設定はない（wait が 0 のときは、os_stdin_ready が読めると答えたあとに呼ばれる）。
  (void)wait;
  ssize_t got = os_read_stdin(um_in.buf + um_in.end, want);
  if (got > 0) {
    um_in.end += (size_t)got;
    return 1;
  }
  um_in.eof = 1;
  return 0;
#else
  for (;;) {
    ssize_t n = read(0, um_in.buf + um_in.end, want);
    if (n > 0) {
      um_in.end += (size_t)n;
      return 1;
    }
    if (n < 0 && errno == EINTR) continue;
    if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
      if (!wait) return -1;
      struct pollfd p = {.fd = 0, .events = POLLIN, .revents = 0};
      if (poll(&p, 1, -1) >= 0 || errno == EINTR) continue;
    }
    um_in.eof = 1;
    return 0;
  }
#endif
}

// バッファの [start, start + n) を1行として渡す（末尾の "\r" を1つ除く）。skip は読み飛ばす改行の数（0 か 1）。
static void in_take(size_t n, size_t skip, unsigned char **out, size_t *len) {
  const unsigned char *line = um_in.buf + um_in.start;
  um_in.start += n + skip;
  um_in.seen = 0;
  if (n > 0 && line[n - 1] == '\r') n--;
  unsigned char *copy = um_alloc(n, 1);
  if (n > 0) memcpy(copy, line, n);
  *out = copy;
  *len = n;
  if (um_in.start == um_in.end && um_in.eof) {
    // 入力が終わり、すべて渡した。バッファはもう要らない。
    free(um_in.buf);
    um_in.buf = NULL;
    um_in.start = um_in.end = um_in.cap = 0;
  }
}

// 標準入力から1行読む。行があれば 1 を返し、*out に新しく確保した領域（長さ 0 なら NULL）、*len に長さを書く。
// 入力の終わりでは 0 を返す（*out = NULL、*len = 0）。待つ前に標準出力を流す。
int32_t um_read_line(unsigned char **out, size_t *len) {
  *out = NULL;
  *len = 0;
  fflush(stdout);
  for (;;) {
    const unsigned char *nl = in_newline();
    if (nl) {
      in_take((size_t)(nl - (um_in.buf + um_in.start)), 1, out, len);
      return 1;
    }
    if (!in_fill(1)) {
      // 入力の終わり。改行で終わっていない最後の行があれば、それを渡す。
      size_t have = um_in.end - um_in.start;
      if (have == 0) {
        free(um_in.buf);
        um_in.buf = NULL;
        um_in.start = um_in.end = um_in.cap = um_in.seen = 0;
        return 0;
      }
      in_take(have, 0, out, len);
      return 1;
    }
  }
}

// um_read_line が待たずに戻るなら 1: バッファに1行そろっているか、入力が終わっているとき。標準出力も流す。
// OS が「読める」と答えた分は、待たずにバッファへ取り込んでから判断する。行の途中まで（改行がまだ）しか
// 届いていなければ 0（そこで um_read_line を呼ぶと、残りが届くまで待つことになるため）。
int32_t um_input_ready(void) {
  fflush(stdout);
  // 1回の呼び出しで取り込む回数には上限を付ける（改行のない入力が流れ続けても、ここから戻れるように）。
  for (int round = 0; round < 16; round++) {
    if (in_newline()) return 1;
    if (um_in.eof) return 1;
#ifdef _WIN32
    int r = os_stdin_ready();
#else
    struct pollfd p = {.fd = 0, .events = POLLIN, .revents = 0};
    int r;
    do {
      r = poll(&p, 1, 0);
    } while (r < 0 && errno == EINTR);
#endif
    if (r == 0) return 0;
    // 確かめられないとき（r < 0）は 1 にする。um_read_line は、読めなければすぐ「入力の終わり」として戻る。
    if (r < 0) return 1;
    // 読めるか、閉じられている。取り込む（poll が読めると答えた直後なので待たない）。
    if (in_fill(0) < 0) return 0;
  }
  return in_newline() != NULL || um_in.eof;
}

// プログラムの開始からの秒数（逆戻りしない時計）。
static double um_clock_origin;
__attribute__((constructor)) static void um_clock_init(void) { um_clock_origin = um_seconds(); }

double um_clock(void) { return um_seconds() - um_clock_origin; }

// 文字列は 0 で終わっていないので、OS に渡す前に 0 で終わる写しを作る。途中に 0 を含む名前は受け付けない
// （OS がそこで名前を切ってしまい、プログラムが指定したのとは別のファイルを開くことになるため）。
static char *path_copy(const char *path, size_t plen) {
  if (plen == 0 || plen == SIZE_MAX || memchr(path, 0, plen)) return NULL;
  char *z = malloc(plen + 1);
  if (!z) um_panic(5, 0);
  memcpy(z, path, plen);
  z[plen] = 0;
  return z;
}

// ファイル全体を読む。読めたら 1 を返し、*out に新しく確保した領域（空なら NULL）、*len に長さを書く。
// 読めなければ 0（*out = NULL、*len = 0）。
int32_t um_read_file(const char *path, size_t plen, unsigned char **out, size_t *len) {
  *out = NULL;
  *len = 0;
  char *z = path_copy(path, plen);
  if (!z) return 0;
  int fd = os_open(z, 0);
  free(z);
  if (fd < 0) return 0;
  // ふつうのファイルなら、大きさ + 1 バイトを最初に確保する（最後の「もうない」の確認まで伸ばさずに済む）。
  size_t first = 65536;
  int64_t size = 0;
  if (os_fstat(fd, &size) == OS_FILE && size > 0 && (uint64_t)size < SIZE_MAX) first = (size_t)size + 1;
  unsigned char *buf = NULL;
  size_t n = 0, cap = 0;
  for (;;) {
    if (n == cap) {
      size_t next = first;
      if (cap != 0 && __builtin_mul_overflow(cap, (size_t)2, &next)) um_panic(5, 0);
      unsigned char *p = realloc(buf, next);
      if (!p) um_panic(5, 0);
      buf = p;
      cap = next;
    }
    ssize_t r = os_read(fd, buf + n, cap - n > UM_IO_CHUNK ? UM_IO_CHUNK : cap - n);
    if (r < 0 && errno == EINTR) continue;
    if (r < 0) {
      // フォルダなど、読めないもの。
      free(buf);
      os_close(fd);
      return 0;
    }
    if (r == 0) break;
    n += (size_t)r;
  }
  os_close(fd);
  if (n == 0) {
    free(buf);
    return 1;
  }
  // 余った分を返す（配列の容量は長さと同じとして扱われる）。
  unsigned char *fit = realloc(buf, n);
  *out = fit ? fit : buf;
  *len = n;
  return 1;
}

// ファイルを作り（あれば中身を捨てて）、バイト列を書く。すべて書けたら 1、そうでなければ 0。
int32_t um_write_file(const char *path, size_t plen, const unsigned char *data, size_t len) {
  char *z = path_copy(path, plen);
  if (!z) return 0;
  int fd = os_open(z, 1);
  free(z);
  if (fd < 0) return 0;
  size_t done = 0;
  while (done < len) {
    ssize_t r = os_write(fd, data + done, len - done > UM_IO_CHUNK ? UM_IO_CHUNK : len - done);
    if (r < 0 && errno == EINTR) continue;
    if (r <= 0) {
      os_close(fd);
      return 0;
    }
    done += (size_t)r;
  }
  return os_close(fd) == 0;
}

// ---- ファイルの一部と追記（fileSize / readFileRange / appendFile） ----

// ふつうのファイルの大きさ（バイト数）。読めないもの、ふつうのファイルでないもの（フォルダなど）は 0 を返す。
int32_t um_file_size(const char *path, size_t plen, int64_t *out) {
  *out = 0;
  char *z = path_copy(path, plen);
  if (!z) return 0;
  int64_t size = 0;
  int kind = os_stat(z, &size);
  free(z);
  if (kind != OS_FILE || size < 0) return 0;
  *out = size;
  return 1;
}

// offset バイト目から最大 want バイトを読む。ファイルの終わりで止まるので、短く（長さ 0 にも）なりうる。
// ふつうのファイルでないもの（/dev/urandom など）は、offset が 0 のときだけ先頭から読む。
// 読めたら 1 を返し、*out に新しく確保した領域（空なら NULL）、*len に長さを書く。読めなければ 0。
int32_t um_read_file_range(const char *path, size_t plen, int64_t offset, int64_t want, unsigned char **out, size_t *len) {
  *out = NULL;
  *len = 0;
  if (offset < 0 || want < 0 || offset > INT64_MAX - want) return 0;
  char *z = path_copy(path, plen);
  if (!z) return 0;
  int fd = os_open(z, 0);
  free(z);
  if (fd < 0) return 0;
  int64_t size = 0;
  int kind = os_fstat(fd, &size);
  if (kind == OS_NONE || kind == OS_DIR) {
    os_close(fd);
    return 0;
  }
  int regular = kind == OS_FILE;
  if (!regular && offset != 0) {
    os_close(fd);
    return 0;
  }
  // ふつうのファイルなら、ファイルに残っている分より多くは確保しない。
  if (regular) {
    int64_t rest = size > offset ? size - offset : 0;
    if (want > rest) want = rest;
  }
  if ((uint64_t)want > (uint64_t)(SIZE_MAX / 2)) {
    os_close(fd);
    return 0;
  }
  size_t cap = (size_t)want, n = 0;
  unsigned char *buf = cap > 0 ? malloc(cap) : NULL;
  if (cap > 0 && !buf) {
    os_close(fd);
    return 0;
  }
  while (n < cap) {
    size_t step = cap - n > UM_IO_CHUNK ? UM_IO_CHUNK : cap - n;
    ssize_t r = regular ? os_pread(fd, buf + n, step, offset + (int64_t)n) : os_read(fd, buf + n, step);
    if (r < 0 && errno == EINTR) continue;
    if (r < 0) {
      free(buf);
      os_close(fd);
      return 0;
    }
    if (r == 0) break;  // 読んでいるあいだにファイルが短くなった、または入力の終わり。
    n += (size_t)r;
  }
  os_close(fd);
  if (n == 0) {
    free(buf);
    return 1;
  }
  if (n < cap) {
    unsigned char *fit = realloc(buf, n);
    if (fit) buf = fit;
  }
  *out = buf;
  *len = n;
  return 1;
}

// ファイルの終わりにバイト列を書き足す（なければ作る）。すべて書けたら 1、そうでなければ 0。
int32_t um_append_file(const char *path, size_t plen, const unsigned char *data, size_t len) {
  char *z = path_copy(path, plen);
  if (!z) return 0;
  int fd = os_open(z, 2);
  free(z);
  if (fd < 0) return 0;
  size_t done = 0;
  while (done < len) {
    ssize_t r = os_write(fd, data + done, len - done > UM_IO_CHUNK ? UM_IO_CHUNK : len - done);
    if (r < 0 && errno == EINTR) continue;
    if (r <= 0) {
      os_close(fd);
      return 0;
    }
    done += (size_t)r;
  }
  return os_close(fd) == 0;
}

// ---- コマンドライン引数（args）と標準エラー出力（eprint） ----

// 生成コードの main(argc, argv) が最初に渡す。args() はプログラム名（argv[0]）を除いた残り。
static int um_argc;
static char **um_argv;
void um_set_args(int32_t argc, char **argv) {
#ifdef _WIN32
  // Windows の argv は ANSI のコードページの文字列なので使わず、UTF-16 のコマンドラインから UTF-8 の並びを作る。
  // 作った並びはプログラムの終わりまで使う（解放しない）。
  int n = 0;
  LPWSTR *w = CommandLineToArgvW(GetCommandLineW(), &n);
  if (w && n > 0) {
    char **a = calloc((size_t)n + 1, sizeof *a);
    if (!a) um_panic(5, 0);
    for (int i = 0; i < n; i++) {
      int len = WideCharToMultiByte(CP_UTF8, 0, w[i], -1, NULL, 0, NULL, NULL);
      a[i] = malloc(len > 0 ? (size_t)len : 1);
      if (!a[i]) um_panic(5, 0);
      a[i][0] = 0;
      if (len > 0) WideCharToMultiByte(CP_UTF8, 0, w[i], -1, a[i], len, NULL, NULL);
    }
    LocalFree(w);
    um_argc = n;
    um_argv = a;
    return;
  }
  if (w) LocalFree(w);
#endif
  um_argc = argc;
  um_argv = argv;
}

// %um.Array（要素の場所、長さ、容量）と同じ並び。
typedef struct {
  unsigned char *data;
  size_t len, cap;
} um_bytes;

// 引数を string[] として新しく確保して渡す（*out は要素の場所、*len は個数。なければ NULL と 0）。
void um_args(um_bytes **out, size_t *len) {
  *out = NULL;
  *len = 0;
  if (um_argc <= 1 || !um_argv) return;
  size_t n = (size_t)um_argc - 1;
  um_bytes *a = um_alloc(n, sizeof(um_bytes));
  for (size_t i = 0; i < n; i++) {
    const char *s = um_argv[i + 1];
    size_t l = s ? strlen(s) : 0;
    unsigned char *p = um_alloc(l, 1);
    if (l > 0) memcpy(p, s, l);
    a[i].data = p;
    a[i].len = l;
    a[i].cap = l;
  }
  *out = a;
  *len = n;
}

// 標準エラー出力に1行書く（改行を付ける）。標準エラー出力はまとめずにすぐ書く。
void um_eprint(const char *data, size_t len) {
  flockfile(stderr);
  if (len > 0) fwrite(data, 1, len, stderr);
  fputc('\n', stderr);
  funlockfile(stderr);
}
