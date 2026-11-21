// Windows（clang の x86_64-pc-windows-msvc / aarch64-pc-windows-msvc）で runtime/native.c を動かすための小さな互換層。
// native.c が使う POSIX の機能のうち、スレッド・鍵・時計・CPU 数・ページの大きさなどを Win32 の API で用意する。
// ファイルと標準入力は native.c の os_* 関数が Windows 用の実装を持つ（UTF-8 の名前、コンソールの入力のため）。
// native.c からだけ include する。

#ifndef UM_WIN32_H
#define UM_WIN32_H

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <shellapi.h>

#include <errno.h>
#include <fcntl.h>
#include <io.h>
#include <process.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

// CommandLineToArgvW（args() が UTF-16 の引数を UTF-8 にするのに使う）。
#pragma comment(lib, "shell32.lib")

typedef intptr_t ssize_t;

// ---- スレッド（pthread の一部） ----

typedef struct {
  HANDLE handle;
  DWORD id;
} pthread_t;

typedef struct {
  size_t stack;
} pthread_attr_t;

#define PTHREAD_STACK_MIN 65536

static inline pthread_t pthread_self(void) {
  pthread_t t = {NULL, GetCurrentThreadId()};
  return t;
}

static inline int pthread_equal(pthread_t a, pthread_t b) { return a.id == b.id; }

static inline int pthread_attr_init(pthread_attr_t *a) {
  a->stack = 0;
  return 0;
}

static inline int pthread_attr_setstacksize(pthread_attr_t *a, size_t size) {
  a->stack = size;
  return 0;
}

static inline int pthread_attr_destroy(pthread_attr_t *a) {
  (void)a;
  return 0;
}

typedef struct {
  void *(*fn)(void *);
  void *arg;
} um_win_start;

static unsigned __stdcall um_win_thread_entry(void *p) {
  um_win_start s = *(um_win_start *)p;
  free(p);
  s.fn(s.arg);
  return 0;
}

// スタックの大きさは「予約」として渡す（使った分だけが実際のメモリになる。POSIX の pthread と同じ）。
static inline int pthread_create(pthread_t *t, const pthread_attr_t *attr, void *(*fn)(void *), void *arg) {
  um_win_start *s = malloc(sizeof *s);
  if (!s) return EAGAIN;
  s->fn = fn;
  s->arg = arg;
  unsigned id = 0;
  size_t stack = attr ? attr->stack : 0;
  uintptr_t h = _beginthreadex(NULL, (unsigned)(stack > 0xffffffffu ? 0xffffffffu : stack), um_win_thread_entry, s,
                               stack ? STACK_SIZE_PARAM_IS_A_RESERVATION : 0, &id);
  if (h == 0) {
    free(s);
    return EAGAIN;
  }
  t->handle = (HANDLE)h;
  t->id = id;
  return 0;
}

static inline int pthread_join(pthread_t t, void **ret) {
  (void)ret;
  WaitForSingleObject(t.handle, INFINITE);
  CloseHandle(t.handle);
  return 0;
}

// 鍵と条件変数は SRW ロックと CONDITION_VARIABLE（どちらも静的に初期化でき、解放が要らない）。
typedef SRWLOCK pthread_mutex_t;
typedef CONDITION_VARIABLE pthread_cond_t;
#define PTHREAD_MUTEX_INITIALIZER SRWLOCK_INIT
#define PTHREAD_COND_INITIALIZER CONDITION_VARIABLE_INIT

static inline int pthread_mutex_lock(pthread_mutex_t *m) {
  AcquireSRWLockExclusive(m);
  return 0;
}

static inline int pthread_mutex_unlock(pthread_mutex_t *m) {
  ReleaseSRWLockExclusive(m);
  return 0;
}

static inline int pthread_cond_wait(pthread_cond_t *c, pthread_mutex_t *m) {
  SleepConditionVariableSRW(c, m, INFINITE, 0);
  return 0;
}

static inline int pthread_cond_signal(pthread_cond_t *c) {
  WakeConditionVariable(c);
  return 0;
}

static inline int pthread_cond_broadcast(pthread_cond_t *c) {
  WakeAllConditionVariable(c);
  return 0;
}

typedef INIT_ONCE pthread_once_t;
#define PTHREAD_ONCE_INIT INIT_ONCE_STATIC_INIT

static BOOL CALLBACK um_win_once_entry(PINIT_ONCE once, PVOID param, PVOID *ctx) {
  (void)once;
  (void)ctx;
  ((void (*)(void))param)();
  return TRUE;
}

static inline int pthread_once(pthread_once_t *once, void (*fn)(void)) {
  InitOnceExecuteOnce(once, um_win_once_entry, (PVOID)fn, NULL);
  return 0;
}

static inline int sched_yield(void) {
  SwitchToThread();
  return 0;
}

// ---- システムの情報 ----

#define _SC_PAGESIZE 1
#define _SC_NPROCESSORS_ONLN 2

static inline long sysconf(int name) {
  if (name == _SC_PAGESIZE) {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (long)si.dwPageSize;
  }
  if (name == _SC_NPROCESSORS_ONLN) {
    // 64 を超える論理プロセッサは複数のグループに分かれる。Windows 11 ではスレッドはグループをまたいで割り当てられる。
    DWORD n = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    return n ? (long)n : 1;
  }
  return -1;
}

// ---- 時計（逆戻りしない時計だけ） ----

#define CLOCK_MONOTONIC 1

static inline int clock_gettime(int clk, struct timespec *t) {
  (void)clk;
  static LARGE_INTEGER freq;  // 0 なら未取得（どのスレッドが取っても同じ値になる）
  LARGE_INTEGER now;
  if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
  QueryPerformanceCounter(&now);
  t->tv_sec = (time_t)(now.QuadPart / freq.QuadPart);
  t->tv_nsec = (long)((now.QuadPart % freq.QuadPart) * 1000000000LL / freq.QuadPart);
  return 0;
}

// ---- stdio ----

#define flockfile _lock_file
#define funlockfile _unlock_file
#define putchar_unlocked(c) _putc_nolock((c), stdout)

// 停止の2番目以降のスレッドが、終わるまで寝るのに使う。
static inline int pause(void) {
  Sleep(INFINITE);
  return -1;
}

#endif
