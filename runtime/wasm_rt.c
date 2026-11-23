// UnifiedMozaic の wasm 用の最小Cランタイム。
// LLVMは構造体や配列のコピー・初期化を memcpy / memset / memmove の呼び出しに変えることがある。
// wasm では libc をリンクしないので、ここで提供する。-fno-builtin でコンパイルすること
// （そうしないと、このループ自体が memset 呼び出しに置き換えられて無限再帰になる）。

typedef __SIZE_TYPE__ size_t;

#include "f16.h"
#include "smallfloat.h"

// wasm では、LLVM は f16 の値を f32 として持ち回り、メモリとの出し入れのときにこれらを呼ぶ。
// f16 は16ビットの整数として受け渡される。
float __extendhfsf2(uint16_t h) {
  union { uint32_t bits; float value; } u = {.bits = um_h2f(h)};
  return u.value;
}

uint16_t __truncdfhf2(double x) {
  union { double value; uint64_t bits; } u = {.value = x};
  return um_d2h(u.bits);
}

uint16_t __truncsfhf2(float x) {
  return __truncdfhf2((double)x); // f32 から f64 への変換は正確
}

void *memset(void *dst, int c, size_t n) {
  unsigned char *d = dst;
  for (size_t i = 0; i < n; i++) d[i] = (unsigned char)c;
  return dst;
}

void *memcpy(void *restrict dst, const void *restrict src, size_t n) {
  unsigned char *d = dst;
  const unsigned char *s = src;
  for (size_t i = 0; i < n; i++) d[i] = s[i];
  return dst;
}

void *memmove(void *dst, const void *src, size_t n) {
  unsigned char *d = dst;
  const unsigned char *s = src;
  if (d < s) {
    for (size_t i = 0; i < n; i++) d[i] = s[i];
  } else {
    for (size_t i = n; i > 0; i--) d[i - 1] = s[i - 1];
  }
  return dst;
}

// ---- ヒープ ----
// 2のべき乗のサイズ別に空きリストを持つ、単純なアロケータ。
// 各ブロックは 16 バイトのヘッダ（サイズクラス）+ 本体。本体は 16 バイト境界に揃う。
// 隣り合う空きの結合はしない（単純さを優先。断片化は後の段階で改善する）。

// 停止。ふつうの wasm32 ではホスト（runtime/host.mjs）の関数を import する。プラグイン（UM_PLUGIN。--target wasm32-plugin）では
// import を持たないので、このファイルの終わりで定義する（原因を覚えてからトラップする）。
#ifdef UM_PLUGIN
_Noreturn void um_panic(int code, int line);
#else
__attribute__((import_module("env"), import_name("um_panic"))) _Noreturn void um_panic(int code, int line);
#endif

extern unsigned char __heap_base;

#define UM_PAGE 65536ull
#define UM_HEADER 16
#define UM_CLASSES 28 /* 本体の最大は 16 << 27 = 2GiB */

static size_t heap_top;
static void *free_lists[UM_CLASSES];

static size_t class_bytes(int c) { return (size_t)16 << c; }

static size_t checked_bytes(size_t count, size_t elem_size) {
  size_t bytes;
  if (__builtin_mul_overflow(count, elem_size, &bytes)) um_panic(5, 0);
  return bytes;
}

static void *alloc_bytes(size_t bytes) {
  int c = 0;
  while (class_bytes(c) < bytes) {
    c++;
    if (c >= UM_CLASSES) um_panic(5, 0);
  }
  if (free_lists[c]) {
    void *p = free_lists[c];
    free_lists[c] = *(void **)p;
    return p;
  }
  if (heap_top == 0) heap_top = ((size_t)&__heap_base + 15) & ~(size_t)15;
  // 32ビットのアドレス空間を超えないことを 64ビットで確かめる。
  unsigned long long end = (unsigned long long)heap_top + UM_HEADER + class_bytes(c);
  unsigned long long have = (unsigned long long)__builtin_wasm_memory_size(0) * UM_PAGE;
  if (end > 0xFFFFFFFFull) um_panic(5, 0);
  if (end > have) {
    unsigned long long pages = (end - have + UM_PAGE - 1) / UM_PAGE;
    if (__builtin_wasm_memory_grow(0, (size_t)pages) == (size_t)-1) um_panic(5, 0);
  }
  size_t *header = (size_t *)heap_top;
  header[0] = (size_t)c;
  heap_top = (size_t)end;
  return (unsigned char *)header + UM_HEADER;
}

static int class_of(void *p) { return (int)*(size_t *)((unsigned char *)p - UM_HEADER); }

void *um_alloc(size_t count, size_t elem_size) {
  size_t bytes = checked_bytes(count, elem_size);
  return bytes == 0 ? 0 : alloc_bytes(bytes);
}

void *um_alloc_zeroed(size_t count, size_t elem_size) {
  size_t bytes = checked_bytes(count, elem_size);
  if (bytes == 0) return 0;
  // 空きリストから再利用したブロックには前の内容が残っているので、必ず消す。
  return memset(alloc_bytes(bytes), 0, bytes);
}

void um_free(void *p) {
  if (!p) return;
  int c = class_of(p);
  *(void **)p = free_lists[c];
  free_lists[c] = p;
}

// 伸ばした後の容量（空なら4、それ以外は2倍）。runtime/native.c と同じ。
size_t um_grow_cap(size_t cap) {
  size_t next;
  if (cap == 0) return 4;
  if (__builtin_mul_overflow(cap, (size_t)2, &next)) um_panic(5, 0);
  return next;
}

// 要素の領域を new_cap 個ぶんに広げ、新しい場所を返す。使っている len 個の要素を写す。
void *um_grow_data(void *data, size_t len, size_t new_cap, size_t elem_size) {
  void *p = alloc_bytes(checked_bytes(new_cap, elem_size));
  if (data) {
    memcpy(p, data, checked_bytes(len, elem_size));
    um_free(data);
  }
  return p;
}

// ---- 文字列 ----
// 文字列は UTF-8 のバイト列。長さ 0 のときデータの場所は 0 でよい。

// f64 を JavaScript と同じ表記で buf（80 バイト以上）に書き、文字数を返す。ホスト側が持つ。
int um_fmt_f64(double v, char *buf) __attribute__((import_module("env"), import_name("um_fmt_f64")));

char *um_str_concat(const char *a, size_t alen, const char *b, size_t blen) {
  if (alen > (size_t)-1 - blen) um_panic(5, 0);
  char *out = um_alloc(alen + blen, 1);
  if (alen > 0) memcpy(out, a, alen);
  if (blen > 0) memcpy(out + alen, b, blen);
  return out;
}

int um_str_eq(const char *a, size_t alen, const char *b, size_t blen) {
  if (alen != blen) return 0;
  for (size_t i = 0; i < alen; i++) {
    if (a[i] != b[i]) return 0;
  }
  return 1;
}

static char *str_from(const char *text, size_t n, size_t *len) {
  *len = n;
  char *out = um_alloc(n, 1);
  if (n > 0) memcpy(out, text, n);
  return out;
}

char *um_str_from_u64(uint64_t v, size_t *len) {
  char buf[24];
  size_t n = 0;
  do {
    buf[sizeof buf - 1 - n++] = (char)('0' + v % 10);
    v /= 10;
  } while (v != 0);
  return str_from(buf + sizeof buf - n, n, len);
}

char *um_str_from_i64(int64_t v, size_t *len) {
  if (v >= 0) return um_str_from_u64((uint64_t)v, len);
  char buf[24];
  size_t n = 0;
  uint64_t u = 0 - (uint64_t)v; // 最小値でもあふれない
  do {
    buf[sizeof buf - 1 - n++] = (char)('0' + u % 10);
    u /= 10;
  } while (u != 0);
  buf[sizeof buf - 1 - n++] = '-';
  return str_from(buf + sizeof buf - n, n, len);
}

char *um_str_from_f64(double v, size_t *len) {
  char buf[80];
  int n = um_fmt_f64(v, buf);
  return str_from(buf, n < 0 || n > 80 ? 0 : (size_t)n, len);
}

// ---- 入出力（readLine / inputReady / clock / readFile / writeFile） ----
// wasm の中からは入出力ができないので、すべてホスト（runtime/host.mjs）に頼む。
// 長さの分からないデータ（1行、ファイルの中身）は2段階で受け取る:
// まずホストがデータを用意して長さを返し、こちらが領域を確保してから、ホストがそこへ写す。

// 次の1行を用意して、そのバイト数を返す。入力がなければ（または終わっていれば）-1。
int um_host_line(void) __attribute__((import_module("env"), import_name("um_host_line")));
// ファイルの中身を用意して、そのバイト数を返す。読めなければ -1。
int um_host_read_file(const char *path, size_t plen) __attribute__((import_module("env"), import_name("um_host_read_file")));
// 直前に用意したデータを dst へ写す（用意したバイト数ぶん）。
void um_host_take(unsigned char *dst) __attribute__((import_module("env"), import_name("um_host_take")));
int um_host_input_ready(void) __attribute__((import_module("env"), import_name("um_host_input_ready")));
double um_host_clock(void) __attribute__((import_module("env"), import_name("um_host_clock")));
int um_host_write_file(const char *path, size_t plen, const unsigned char *data, size_t len)
    __attribute__((import_module("env"), import_name("um_host_write_file")));

// ホストが用意した n バイトを、新しく確保した領域に受け取る。n が負なら「なし」（0 を返す）。
static int take_pending(int n, unsigned char **out, size_t *len) {
  *out = 0;
  *len = 0;
  if (n < 0) return 0;
  if (n > 0) {
    unsigned char *p = um_alloc((size_t)n, 1);
    um_host_take(p);
    *out = p;
    *len = (size_t)n;
  }
  return 1;
}

int um_read_line(unsigned char **out, size_t *len) { return take_pending(um_host_line(), out, len); }

int um_input_ready(void) { return um_host_input_ready() != 0; }

double um_clock(void) { return um_host_clock(); }

int um_read_file(const char *path, size_t plen, unsigned char **out, size_t *len) {
  return take_pending(um_host_read_file(path, plen), out, len);
}

int um_write_file(const char *path, size_t plen, const unsigned char *data, size_t len) {
  return um_host_write_file(path, plen, data, len) != 0;
}

// ---- ファイルの一部と追記、コマンドライン引数、標準エラー出力 ----
// i64 の引数と結果は、ホストでは BigInt になる。

// ふつうのファイルの大きさ。なければ（読めなければ）負の値。
long long um_host_file_size(const char *path, size_t plen) __attribute__((import_module("env"), import_name("um_host_file_size")));
// ファイルの一部を用意して、そのバイト数を返す。読めなければ -1。
int um_host_read_file_range(const char *path, size_t plen, long long offset, long long len)
    __attribute__((import_module("env"), import_name("um_host_read_file_range")));
int um_host_append_file(const char *path, size_t plen, const unsigned char *data, size_t len)
    __attribute__((import_module("env"), import_name("um_host_append_file")));
// 引数の個数と、i 番目の引数を用意してそのバイト数を返す。
int um_host_argc(void) __attribute__((import_module("env"), import_name("um_host_argc")));
int um_host_arg(int i) __attribute__((import_module("env"), import_name("um_host_arg")));
void um_host_eprint(const char *data, size_t len) __attribute__((import_module("env"), import_name("um_host_eprint")));

int um_file_size(const char *path, size_t plen, long long *out) {
  long long n = um_host_file_size(path, plen);
  *out = n < 0 ? 0 : n;
  return n >= 0;
}

int um_read_file_range(const char *path, size_t plen, long long offset, long long len, unsigned char **out, size_t *outlen) {
  if (offset < 0 || len < 0) {
    *out = 0;
    *outlen = 0;
    return 0;
  }
  return take_pending(um_host_read_file_range(path, plen, offset, len), out, outlen);
}

int um_append_file(const char *path, size_t plen, const unsigned char *data, size_t len) {
  return um_host_append_file(path, plen, data, len) != 0;
}

// %um.Array（要素の場所、長さ、容量）と同じ並び。
typedef struct {
  unsigned char *data;
  size_t len, cap;
} um_bytes;

void um_args(um_bytes **out, size_t *len) {
  *out = 0;
  *len = 0;
  int n = um_host_argc();
  if (n <= 0) return;
  um_bytes *a = um_alloc((size_t)n, sizeof(um_bytes));
  for (int i = 0; i < n; i++) {
    unsigned char *p;
    size_t l;
    take_pending(um_host_arg(i), &p, &l);
    a[i].data = p;
    a[i].len = l;
    a[i].cap = l;
  }
  *out = a;
  *len = (size_t)n;
}

void um_eprint(const char *data, size_t len) { um_host_eprint(data, len); }

// ---- ファイルに直接読む・開いたままのファイル・フォルダとパス ----
// 開いたファイルの番号はホスト（runtime/host.mjs）が管理する。知らない・閉じた番号には -2 が返るので、停止する（16）。

int um_host_file_open(const char *path, size_t plen, int mode) __attribute__((import_module("env"), import_name("um_host_file_open")));
// 読んだデータを用意して、そのバイト数を返す（失敗 -1、知らない番号 -2）。at が負なら今の位置から。
int um_host_file_read(int id, long long at, int len) __attribute__((import_module("env"), import_name("um_host_file_read")));
int um_host_file_write(int id, const unsigned char *data, size_t len) __attribute__((import_module("env"), import_name("um_host_file_write")));
int um_host_file_seek(int id, long long offset) __attribute__((import_module("env"), import_name("um_host_file_seek")));
long long um_host_file_length(int id) __attribute__((import_module("env"), import_name("um_host_file_length")));
int um_host_file_close(int id) __attribute__((import_module("env"), import_name("um_host_file_close")));
// 名前の一覧を用意して個数を返す（読めなければ -1）。i 番目の名前を用意してそのバイト数を返す。
int um_host_list_dir(const char *path, size_t plen) __attribute__((import_module("env"), import_name("um_host_list_dir")));
int um_host_list_entry(int i) __attribute__((import_module("env"), import_name("um_host_list_entry")));
int um_host_make_dir(const char *path, size_t plen) __attribute__((import_module("env"), import_name("um_host_make_dir")));
int um_host_is_dir(const char *path, size_t plen) __attribute__((import_module("env"), import_name("um_host_is_dir")));
// kind: 0 = cwd、1 = exePath、2 = exeDir、3 = absPath、4 = envVar。結果を用意してそのバイト数を返す（なければ -1）。
int um_host_path(int kind, const char *path, size_t plen) __attribute__((import_module("env"), import_name("um_host_path")));

static size_t file_range(size_t blen, long long start, long long count, int line) {
  if (start < 0 || (unsigned long long)start > blen) um_panic(16, line);
  size_t rest = blen - (size_t)start;
  if (count < 0) return rest;
  if ((unsigned long long)count > rest) um_panic(16, line);
  return (size_t)count;
}

// 用意された n バイト（max 以下）を dst へ写す。
static long long take_into(int n, unsigned char *dst, size_t max) {
  if (n < 0) return -1;
  if ((size_t)n > max) n = (int)max;  // ホストが切り詰めるので、ここには来ない
  if (n > 0) um_host_take(dst);
  return n;
}

int um_read_file_into(const char *path, size_t plen, long long offset, unsigned char *buf, size_t blen, long long boff, long long want,
                      long long *out, int line) {
  *out = 0;
  if (want < 0) um_panic(16, line);
  size_t n = file_range(blen, boff, want, line);
  if (offset < 0) return 0;
  long long got = take_into(um_host_read_file_range(path, plen, offset, (long long)n), buf + boff, n);
  if (got < 0) return 0;
  *out = got;
  return 1;
}

int um_file_open(const char *path, size_t plen, const char *mode, size_t mlen, int line) {
  int how = -1;
  if (mlen == 1 && mode[0] == 'r') how = 0;
  else if (mlen == 1 && mode[0] == 'w') how = 1;
  else if (mlen == 1 && mode[0] == 'a') how = 2;
  else if (mlen == 2 && mode[0] == 'r' && mode[1] == '+') how = 3;
  if (how < 0) um_panic(16, line);
  int id = um_host_file_open(path, plen, how);
  return id > 0 ? id : 0;
}

int um_file_read(int id, int positional, long long offset, unsigned char *buf, size_t blen, long long start, long long count,
                 long long *out, int line) {
  *out = 0;
  size_t n = file_range(blen, start, count, line);
  if (positional && offset < 0) {
    // 番号だけは確かめる（ネイティブと同じく、知らない番号なら停止）。
    if (um_host_file_length(id) == -2) um_panic(16, line);
    return 0;
  }
  if (n > 0x7fffffff) n = 0x7fffffff;
  int r = um_host_file_read(id, positional ? offset : -1, (int)n);
  if (r == -2) um_panic(16, line);
  long long got = take_into(r, buf + start, n);
  if (got < 0) return 0;
  *out = got;
  return 1;
}

int um_file_write(int id, const unsigned char *data, size_t len, long long start, long long count, int line) {
  size_t n = file_range(len, start, count, line);
  int r = um_host_file_write(id, data + start, n);
  if (r == -2) um_panic(16, line);
  return r > 0;
}

int um_file_seek(int id, long long offset, int line) {
  int r = um_host_file_seek(id, offset);
  if (r == -2) um_panic(16, line);
  return r > 0;
}

int um_file_length(int id, long long *out, int line) {
  long long n = um_host_file_length(id);
  if (n == -2) um_panic(16, line);
  *out = n < 0 ? 0 : n;
  return n >= 0;
}

void um_file_close(int id, int line) {
  if (um_host_file_close(id) == -2) um_panic(16, line);
}

int um_list_dir(const char *path, size_t plen, um_bytes **out, size_t *len) {
  *out = 0;
  *len = 0;
  int n = um_host_list_dir(path, plen);
  if (n < 0) return 0;
  if (n == 0) return 1;
  um_bytes *a = um_alloc((size_t)n, sizeof(um_bytes));
  for (int i = 0; i < n; i++) {
    unsigned char *p;
    size_t l;
    take_pending(um_host_list_entry(i), &p, &l);
    a[i].data = p;
    a[i].len = l;
    a[i].cap = l;
  }
  *out = a;
  *len = (size_t)n;
  return 1;
}

int um_make_dir(const char *path, size_t plen) { return um_host_make_dir(path, plen) > 0; }

int um_is_dir(const char *path, size_t plen) { return um_host_is_dir(path, plen) > 0; }

void um_path_query(int kind, const char *path, size_t plen, unsigned char **out, size_t *len) {
  take_pending(um_host_path(kind, path, plen), out, len);
}

// ---- プロセスとファイルの操作（processRun / processStart / processId / fileRename / fileRemove / dirRemove）----
// ホスト（runtime/host.mjs）の口を呼ぶ。口がなければ失敗（processRun は -1、processId は 0、ほかは false）を返す。
// args は string[] の要素の並び（{ 場所, 長さ, 容量 } が個数ぶん）で、ホストがメモリから直接読む。
int um_host_process_run(const char *path, size_t plen, const um_bytes *args, size_t n) __attribute__((import_module("env"), import_name("um_host_process_run")));
int um_host_process_start(const char *path, size_t plen, const um_bytes *args, size_t n) __attribute__((import_module("env"), import_name("um_host_process_start")));
int um_host_process_id(void) __attribute__((import_module("env"), import_name("um_host_process_id")));
int um_host_file_rename(const char *from, size_t flen, const char *to, size_t tlen) __attribute__((import_module("env"), import_name("um_host_file_rename")));
int um_host_file_remove(const char *path, size_t plen) __attribute__((import_module("env"), import_name("um_host_file_remove")));
int um_host_dir_remove(const char *path, size_t plen) __attribute__((import_module("env"), import_name("um_host_dir_remove")));

int um_process_run(const char *path, size_t plen, const um_bytes *args, size_t n) { return um_host_process_run(path, plen, args, n); }
int um_process_start(const char *path, size_t plen, const um_bytes *args, size_t n) { return um_host_process_start(path, plen, args, n) > 0; }
int um_process_id(void) { return um_host_process_id(); }
int um_file_rename(const char *from, size_t flen, const char *to, size_t tlen) { return um_host_file_rename(from, flen, to, tlen) > 0; }
int um_file_remove(const char *path, size_t plen) { return um_host_file_remove(path, plen) > 0; }
int um_dir_remove(const char *path, size_t plen) { return um_host_dir_remove(path, plen) > 0; }

// ---- スレッド ----
// wasm32 にはスレッドがないので、thread.spawn / spawnEach は控えておくだけにし、合流の時点（thread.join か
// ブロックの終わりの自動の合流）でその場で最後まで実行する（spawnEach は要素の順）。起動の時点で実行しないのは、
// 「止める合図が立つまで回る」補助スレッドが、合図を立てる側より先に走ると終わらなくなるため。
// 型検査の凍結により、合流までに貸した値は変わらないので、結果はネイティブで同じ順に実行した場合と同じ。
typedef void (*um_thread_fn)(void *block);
typedef struct {
  um_thread_fn fn;
  char *blocks;
  size_t size, n;
} um_thread_set;

static int um_in_thread;  // スレッドの関数を実行している深さ（readLine / inputReady を止めるため）

void *um_thread_start(um_thread_fn fn, char *blocks, size_t size, size_t n, int line) {
  (void)line;
  um_thread_set *set = um_alloc(1, sizeof(um_thread_set));
  set->fn = fn;
  set->blocks = blocks;
  set->size = size;
  set->n = n;
  return set;
}

void um_thread_join(void *h) {
  um_thread_set *set = h;
  um_in_thread++;
  for (size_t k = 0; k < set->n; k++) set->fn(set->blocks + k * set->size);
  um_in_thread--;
  um_free(set);
}

int um_thread_cpus(void) { return 1; }

// wasm32 にはほかのスレッドがないので、何もしない。
void um_thread_yield(void) {}

int um_is_main(void) { return um_in_thread == 0; }

// xs.fill(v, start, end): 要素 [a, b) を、ビット列 bits の下位 size バイトにする。
// wasm32 にはスレッドがないので、RacyBuffer（racy）も普通に書く。
void um_fill(void *data, size_t a, size_t b, int32_t size, uint64_t bits, int32_t racy) {
  (void)racy;
  if (a >= b) return;
  switch (size) {
    case 1: {
      unsigned char *d = data;
      for (size_t i = a; i < b; i++) d[i] = (unsigned char)bits;
      return;
    }
    case 2: {
      uint16_t *d = data;
      for (size_t i = a; i < b; i++) d[i] = (uint16_t)bits;
      return;
    }
    case 4: {
      uint32_t *d = data;
      for (size_t i = a; i < b; i++) d[i] = (uint32_t)bits;
      return;
    }
    default: {
      uint64_t *d = data;
      for (size_t i = a; i < b; i++) d[i] = bits;
      return;
    }
  }
}

// ---- HTTP（httpStart など）----
// wasm32 ではまだ使えない。httpStart は番号を返すが、要求はすぐに失敗の状態（3）になり、httpError が理由を返す。
// 番号の検査はしない（どの番号も同じ失敗の要求として扱う）。

static const char http_unsupported[] = "HTTP requests are not available in wasm32 programs yet (build with --target host)";

long long um_http_start(const char *m, size_t ml, const char *u, size_t ul, const char *h, size_t hl, const unsigned char *b, size_t bl,
                        int timeout_ms, int line) {
  (void)m, (void)ml, (void)u, (void)ul, (void)h, (void)hl, (void)b, (void)bl, (void)timeout_ms, (void)line;
  static long long next = 0;
  return ++next;
}
int um_http_state(long long h, int line) { (void)h, (void)line; return 3; }
int um_http_status(long long h, int line) { (void)h, (void)line; return 0; }
void um_http_read(long long h, unsigned char **out, size_t *len, int line) { (void)h, (void)line; *out = 0; *len = 0; }
int um_http_read_line(long long h, unsigned char **out, size_t *len, int line) { (void)h, (void)line; *out = 0; *len = 0; return 0; }
void um_http_headers(long long h, unsigned char **out, size_t *len, int line) { (void)h, (void)line; *out = 0; *len = 0; }
void um_http_error(long long h, unsigned char **out, size_t *len, int line) {
  (void)h, (void)line;
  size_t n = sizeof http_unsupported - 1;
  unsigned char *p = um_alloc(n, 1);
  memcpy(p, http_unsupported, n);
  *out = p;
  *len = n;
}
int um_http_wait(long long h, int timeout_ms, int line) { (void)h, (void)timeout_ms, (void)line; return 0; }
void um_http_cancel(long long h, int line) { (void)h, (void)line; }
void um_http_close(long long h, int line) { (void)h, (void)line; }

#ifdef UM_PLUGIN
// ---- プラグイン（--target wasm32-plugin）----
// ホストのプログラムに組み込んで使う wasm。import を持たない。ホストは次の export を使う:
//   umAlloc(bytes) -> 位置   メモリを確保する（16 バイト境界。0 バイトでも 0 でない位置を返す）。足りなければトラップ。
//   umFree(位置)             umAlloc で確保したメモリを返す。
//   umPanicCode() / umPanicLine()  最後の停止の原因コードと行番号（停止していなければ 0）。停止は unreachable のトラップになるので、
//                            ホストはトラップのあとにこれを呼んで理由を知る（原因コードは src/codegen.ts の PANIC_* と同じ）。
// export した UM の関数は、Buffer の引数を (要素の位置, 要素の数) の2つの i32 で受け取る（src/codegen.ts の pluginExport）。

static int panic_code, panic_line;

_Noreturn void um_panic(int code, int line) {
  panic_code = code;
  panic_line = line;
  __builtin_trap();
}

__attribute__((export_name("umAlloc"))) void *um_plugin_alloc(size_t bytes) { return alloc_bytes(bytes ? bytes : 1); }

__attribute__((export_name("umFree"))) void um_plugin_free(void *p) { um_free(p); }

__attribute__((export_name("umPanicCode"))) int um_plugin_panic_code(void) { return panic_code; }

__attribute__((export_name("umPanicLine"))) int um_plugin_panic_line(void) { return panic_line; }
#endif
