// UnifiedMozaic の wgpu ランタイム。kernel を wgpu-native（WebGPU の C API）で GPU に実行させる。
// Windows（Direct3D 12 / Vulkan）、Linux（Vulkan）、macOS（Metal）で同じに動く。
//
// コンパイラは UM_WGPU=1 のネイティブのビルドで、GPU を使う呼び出しがあるときだけこのファイルと wgpu-native をリンクする
// （src/driver.ts）。kernel は WGSL（src/wgsl.ts）にしてバイナリに埋め込み、初めて使うときにパイプラインを作る。
//
// 引数の並びは Metal（runtime/metal.m）と WebGPU（runtime/host.mjs）と同じ:
//   table: Buffer 1つにつき 64 ビットの語が4つ（データの場所、長さ、1要素のバイト数、書き込むかどうか）。
//   block: Buffer 以外の引数を、コンパイラが 32 ビットの語で詰めたもの。
// シェーダ側の引数の塊（UmParams、binding 0）は、個数、1行の要素数、各 Buffer の長さ、block の順。
// Buffer は binding 1 から、引数の順に並ぶ（docs/wgpu.ja.md の「束縛の並び」）。
//
// ---- CPU と GPU の写し（docs/wgpu.ja.md の「Buffer の写し」）----
// CPU と GPU のメモリが分かれている GPU（NVIDIA や AMD の単体の GPU）では、Buffer の中身を写す必要がある。
// 写しを最小にするため、Buffer の領域（um_wgpu_alloc がページ単位で確保する）ごとに、CPU 側と GPU 側の
// どちらが新しいかを覚え、ページの保護で CPU の読み書きを見張る:
//   CPU だけが新しい（GPU の写しがない・古い） → 保護なし。dispatch に渡したとき GPU へ送る。
//   両方が同じ                                  → 読み取り専用。CPU が書くと（1回の例外で）「CPU だけが新しい」に戻る。
//   GPU だけが新しい（kernel が書いた）           → 読み書きとも禁止。CPU が触ると（1回の例外で）読み戻す。
// こうすると、GPU で続けて実行する dispatch のあいだでは写しが起きず、dispatch は提出するだけで完了を待たない
// （結果を CPU が読むときに、読み戻しが完了を待つ）。UM_WGPU_LAZY=0 で、この仕組みをやめて dispatch のたびに
// 送って読み戻す（比較用）。
//
// 例外の処理（Windows は例外ハンドラ、ほかは SIGSEGV / SIGBUS）は、自分の確保した領域で起きたものだけを扱う。
// GPU の鍵（um_lock）を持っているスレッドでは扱わない（ランタイムは鍵を持ったまま保護した領域に触らない）。
//
// 領域は同じメモリを2か所に写す（Windows はページファイルを裏にした共有メモリ、ほかは memfd / shm）。プログラムが使う
// 場所（base）は保護を変え、ランタイムは保護しない別名（alias）から読み書きする。読み戻しのあいだもプログラム側は
// 読み書き禁止のままなので、ほかのスレッドの CPU の書き込みが読み戻しに上書きされることはない（触れば鍵を待つ）。

#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS  // getenv などの「安全でない」という警告を出さない（native.c と同じ）
#endif
#ifdef __linux__
// memfd_create を使う。glibc は最初のヘッダを読むときにこの指定を見るので、どの #include よりも前に置く。
#define _GNU_SOURCE
#endif
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#endif

#include <webgpu/webgpu.h>
#include <webgpu/wgpu.h>

extern _Noreturn void um_exit_failure(void);
extern _Noreturn void um_panic(int32_t code, int32_t line);
// runtime/native.c の um_free が、ページ境界から始まる領域を解放する前に呼ぶ。1 を返したら、こちらで解放済み。
typedef int (*um_free_owned_fn)(void *);
extern _Atomic(um_free_owned_fn) um_free_owned_hook;

// 1つのワークグループの要素数（src/wgsl.ts の WORKGROUP_SIZE と揃える）。
#define UM_WG 64u

// ---- 鍵・時計・ページ ----

static _Thread_local int um_lock_depth;
#ifdef _WIN32
static SRWLOCK um_srw = SRWLOCK_INIT;
static void um_lock(void) {
  AcquireSRWLockExclusive(&um_srw);
  um_lock_depth++;
}
static void um_unlock(void) {
  um_lock_depth--;
  ReleaseSRWLockExclusive(&um_srw);
}
static double um_now(void) {
  static LARGE_INTEGER freq;
  LARGE_INTEGER t;
  if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
  QueryPerformanceCounter(&t);
  return (double)t.QuadPart / (double)freq.QuadPart;
}
static size_t um_pagesize(void) {
  SYSTEM_INFO si;
  GetSystemInfo(&si);
  return si.dwPageSize;
}
#else
static pthread_mutex_t um_mutex = PTHREAD_MUTEX_INITIALIZER;
static void um_lock(void) {
  pthread_mutex_lock(&um_mutex);
  um_lock_depth++;
}
static void um_unlock(void) {
  um_lock_depth--;
  pthread_mutex_unlock(&um_mutex);
}
static double um_now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
static size_t um_pagesize(void) { return (size_t)sysconf(_SC_PAGESIZE); }
#endif

static void um_fail(const char *what, WGPUStringView msg) {
  fflush(stdout);
  if (msg.data && msg.length > 0) {
    int n = msg.length == WGPU_STRLEN ? (int)strlen(msg.data) : (int)msg.length;
    fprintf(stderr, "panic: the GPU could not run a kernel (%s): %.*s\n", what, n, msg.data);
  } else {
    fprintf(stderr, "panic: the GPU could not run a kernel (%s)\n", what);
  }
  um_exit_failure();
}
static void um_fail0(const char *what) { um_fail(what, (WGPUStringView){NULL, 0}); }

static WGPUStringView um_sv(const char *s) { return (WGPUStringView){s, strlen(s)}; }

// ---- 環境変数 ----

static int um_env_done;
static int um_gpu_off;       // UM_GPU=0: GPU を使わない
static int um_lazy = 1;      // UM_WGPU_LAZY=0: dispatch のたびに送って読み戻す（比較用）
static int um_trace;         // UM_GPU_TRACE=1: 送り・読み戻し・dispatch を stderr に出す（2 なら自動判断の選択も）
static int um_fixed_rule;    // UM_GPU_THRESHOLD=n: 実測をやめ、要素数が n 以上なら GPU
static uint32_t um_threshold;

static void um_env_init(void) {
  if (um_env_done) return;
  um_env_done = 1;
  const char *v = getenv("UM_GPU");
  if (v && strcmp(v, "0") == 0) um_gpu_off = 1;
  v = getenv("UM_WGPU_LAZY");
  if (v && strcmp(v, "0") == 0) um_lazy = 0;
  v = getenv("UM_GPU_TRACE");
  if (v && (*v == '1' || *v == '2') && v[1] == 0) um_trace = *v - '0';
  v = getenv("UM_GPU_THRESHOLD");
  if (v && *v) {
    um_fixed_rule = 1;
    um_threshold = (uint32_t)strtoul(v, NULL, 10);
  }
}

// ---- 領域（Buffer の中身）----

enum { UM_RW = 0, UM_RO = 1, UM_NONE = 2 };

typedef struct {
  unsigned char *base;  // プログラムが使う場所（保護を変える）
  unsigned char *alias; // 同じメモリの別名（保護しない。ランタイムだけが使う）
#ifdef _WIN32
  HANDLE section;
#endif
  size_t size;          // 確保したバイト数（ページ単位）
  uint64_t used;        // GPU と写し合うバイト数（4 の倍数）
  WGPUBuffer gbuf;      // GPU 側の写し（初めて dispatch に渡したときに作る）
  WGPUBuffer staging;   // 読み戻し用（初めて読み戻すときに作る）
  uint8_t host_valid, gpu_valid, prot;
  int32_t writer;       // 最後に GPU で書いた kernel の番号（自動判断の読み戻しの見積もり用。なければ -1）
  uint32_t wcount;      // そのときの要素数
} um_region;

// 自動判断の読み戻しの見積もり（後ろの「自動判断」で定義する）。
static void um_note_read(um_region *r, double seconds);
static void um_note_unread(int32_t writer);
static int um_forced_readback;  // 自動判断の試しの回の読み戻し（CPU が読んだのではない）
static double um_rb_spent;      // 読み戻しにかかった時間の合計（CPU で実行した kernel の時間から除くため）

// base の昇順に並べた表。
static um_region **um_regions;
static size_t um_nregions, um_capregions;
static _Atomic uintptr_t um_lo = UINTPTR_MAX, um_hi = 0;

static size_t um_region_lower(uintptr_t p) {
  size_t lo = 0, hi = um_nregions;
  while (lo < hi) {
    size_t mid = (lo + hi) / 2;
    if ((uintptr_t)um_regions[mid]->base < p) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}
static um_region *um_region_at(uintptr_t p) {
  size_t i = um_region_lower(p);
  return i < um_nregions && (uintptr_t)um_regions[i]->base == p ? um_regions[i] : NULL;
}
static um_region *um_region_containing(uintptr_t p) {
  size_t i = um_region_lower(p + 1);
  if (i == 0) return NULL;
  um_region *r = um_regions[i - 1];
  return p >= (uintptr_t)r->base && p < (uintptr_t)r->base + r->size ? r : NULL;
}

static void um_protect(um_region *r, int prot) {
  if (r->prot == prot) return;
#ifdef _WIN32
  DWORD old;
  DWORD p = prot == UM_RW ? PAGE_READWRITE : prot == UM_RO ? PAGE_READONLY : PAGE_NOACCESS;
  if (!VirtualProtect(r->base, r->size, p, &old)) um_fail0("changing the protection of a buffer");
#else
  int p = prot == UM_RW ? PROT_READ | PROT_WRITE : prot == UM_RO ? PROT_READ : PROT_NONE;
  if (mprotect(r->base, r->size, p) != 0) um_fail0("changing the protection of a buffer");
#endif
  r->prot = (uint8_t)prot;
}
// 2つの印から決まる保護にする。
// UM_WGPU_LAZY=0 では保護しない（dispatch のたびに読み戻すので、CPU が古い写しを見ることはない）。
static void um_reprotect(um_region *r) {
  if (!um_lazy) return;
  if (!r->host_valid) um_protect(r, UM_NONE);
  else if (r->gpu_valid) um_protect(r, UM_RO);
  else um_protect(r, UM_RW);
}

// ---- GPU ----

static int um_ready;  // 0 = まだ調べていない、1 = 使える、-1 = 使えない
static WGPUInstance um_inst;
static WGPUAdapter um_adapter;
static WGPUDevice um_dev;
static WGPUQueue um_queue;
static WGPULimits um_limits;
static WGPUCommandEncoder um_enc;  // まだ提出していない dispatch を積んだ命令（um_gpu_flush で提出する）
static uint32_t um_enc_count;      // um_enc に積んだ dispatch の数
static uint64_t um_enc_elems;      // um_enc に積んだ dispatch の要素数の合計
#define UM_BATCH_COUNT 16
#define UM_BATCH_ELEMS 262144
static uint64_t um_submits;        // 提出した回数（gpu.submit の券）
static _Atomic uint64_t um_done;   // 終わったことが分かっている券
static WGPUSubmissionIndex um_last_index;

// 読み戻しをまだしていない、提出前の dispatch が書いた領域（UM_WGPU_LAZY=0 と、表にない Buffer のため）。
typedef struct {
  um_region *r;        // 表にある領域（なければ NULL）
  WGPUBuffer temp;     // 表にない Buffer の、一時的な GPU 側のバッファ
  void *host;
  uint64_t bytes;
} um_eager;
static um_eager *um_eagers;
static size_t um_neagers, um_capeagers;

static void um_wait_flag(_Atomic int *flag) {
  while (!atomic_load(flag)) {
    wgpuDevicePoll(um_dev, 1, NULL);
    wgpuInstanceProcessEvents(um_inst);
  }
}

static void um_on_adapter(WGPURequestAdapterStatus status, WGPUAdapter adapter, WGPUStringView msg, void *u1, void *u2) {
  (void)msg, (void)u2;
  if (status == WGPURequestAdapterStatus_Success) um_adapter = adapter;
  *(int *)u1 = 1;
}
static void um_on_device(WGPURequestDeviceStatus status, WGPUDevice device, WGPUStringView msg, void *u1, void *u2) {
  (void)msg, (void)u2;
  if (status == WGPURequestDeviceStatus_Success) um_dev = device;
  *(int *)u1 = 1;
}
// 検証の失敗など、捕まえていない誤り。生成したシェーダと引数の誤りなので、止める。
static void um_on_error(WGPUDevice const *device, WGPUErrorType type, WGPUStringView msg, void *u1, void *u2) {
  (void)device, (void)type, (void)u1, (void)u2;
  um_fail("wgpu", msg);
}

static WGPUInstanceBackend um_backend_from_env(void) {
  const char *b = getenv("UM_WGPU_BACKEND");
  if (!b || !*b) return WGPUInstanceBackend_Primary;
  if (strcmp(b, "vulkan") == 0) return WGPUInstanceBackend_Vulkan;
  if (strcmp(b, "dx12") == 0) return WGPUInstanceBackend_DX12;
  if (strcmp(b, "metal") == 0) return WGPUInstanceBackend_Metal;
  if (strcmp(b, "gl") == 0) return WGPUInstanceBackend_GL;
  fprintf(stderr, "warning: unknown UM_WGPU_BACKEND '%s' (expected vulkan, dx12, metal or gl); using the default\n", b);
  return WGPUInstanceBackend_Primary;
}

static const char *um_backend_name(WGPUBackendType t) {
  switch (t) {
    case WGPUBackendType_D3D12: return "Direct3D 12";
    case WGPUBackendType_Vulkan: return "Vulkan";
    case WGPUBackendType_Metal: return "Metal";
    case WGPUBackendType_OpenGL: return "OpenGL";
    case WGPUBackendType_OpenGLES: return "OpenGL ES";
    default: return "other";
  }
}

// 準備（鍵を持って呼ぶ）。初めて GPU を使おうとしたときに1回だけ行う。
static int um_init_locked(void) {
  if (um_ready) return um_ready > 0;
  um_ready = -1;
  um_env_init();
  if (um_gpu_off) return 0;

  WGPUInstanceExtras extras;
  memset(&extras, 0, sizeof extras);
  extras.chain.sType = (WGPUSType)WGPUSType_InstanceExtras;
  extras.backends = um_backend_from_env();
  WGPUInstanceDescriptor idesc = WGPU_INSTANCE_DESCRIPTOR_INIT;
  idesc.nextInChain = &extras.chain;
  um_inst = wgpuCreateInstance(&idesc);
  if (!um_inst) return 0;

  _Atomic int got = 0;
  WGPURequestAdapterOptions opt = WGPU_REQUEST_ADAPTER_OPTIONS_INIT;
  opt.powerPreference = WGPUPowerPreference_HighPerformance;
  WGPURequestAdapterCallbackInfo aci = WGPU_REQUEST_ADAPTER_CALLBACK_INFO_INIT;
  aci.mode = WGPUCallbackMode_AllowProcessEvents;
  aci.callback = um_on_adapter;
  aci.userdata1 = (void *)&got;
  wgpuInstanceRequestAdapter(um_inst, &opt, aci);
  while (!atomic_load(&got)) wgpuInstanceProcessEvents(um_inst);
  if (!um_adapter) return 0;

  // アダプタが許す上限をそのまま求める（グループの大きさ 1024、大きな Buffer、多くの Buffer の引数のため）。
  WGPULimits want = WGPU_LIMITS_INIT;
  if (wgpuAdapterGetLimits(um_adapter, &want) != WGPUStatus_Success) return 0;
  WGPUDeviceDescriptor ddesc = WGPU_DEVICE_DESCRIPTOR_INIT;
  ddesc.label = um_sv("UnifiedMozaic");
  ddesc.requiredLimits = &want;
  ddesc.uncapturedErrorCallbackInfo.callback = um_on_error;
  atomic_store(&got, 0);
  WGPURequestDeviceCallbackInfo dci = WGPU_REQUEST_DEVICE_CALLBACK_INFO_INIT;
  dci.mode = WGPUCallbackMode_AllowProcessEvents;
  dci.callback = um_on_device;
  dci.userdata1 = (void *)&got;
  wgpuAdapterRequestDevice(um_adapter, &ddesc, dci);
  while (!atomic_load(&got)) wgpuInstanceProcessEvents(um_inst);
  if (!um_dev) return 0;
  um_queue = wgpuDeviceGetQueue(um_dev);
  um_limits = (WGPULimits)WGPU_LIMITS_INIT;
  if (wgpuDeviceGetLimits(um_dev, &um_limits) != WGPUStatus_Success) return 0;
  if (um_trace) {
    WGPUAdapterInfo info = WGPU_ADAPTER_INFO_INIT;
    wgpuAdapterGetInfo(um_adapter, &info);
    fprintf(stderr, "wgpu: %.*s (%s)\n", (int)info.device.length, info.device.data ? info.device.data : "", um_backend_name(info.backendType));
  }
  um_ready = 1;
  return 1;
}

int32_t um_gpu_available(void) {
  um_lock();
  int ok = um_init_locked();
  um_unlock();
  return ok;
}

// 積んである dispatch を提出する（待たない）。鍵を持って呼ぶ。
static void um_submit_locked(void) {
  if (um_enc) {
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(um_enc, NULL);
    wgpuCommandEncoderRelease(um_enc);
    um_enc = NULL;
    um_enc_count = 0;
    um_enc_elems = 0;
    um_last_index = wgpuQueueSubmitForIndex(um_queue, 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    um_submits++;
  }
}

static WGPUBuffer um_make_buffer(uint64_t size, WGPUBufferUsage usage, int mapped) {
  WGPUBufferDescriptor d = WGPU_BUFFER_DESCRIPTOR_INIT;
  d.size = size;
  d.usage = usage;
  d.mappedAtCreation = mapped ? 1 : 0;
  WGPUBuffer b = wgpuDeviceCreateBuffer(um_dev, &d);
  if (!b) um_fail0("allocating GPU memory");
  return b;
}

static void um_on_mapped(WGPUMapAsyncStatus status, WGPUStringView msg, void *u1, void *u2) {
  (void)u2;
  if (status != WGPUMapAsyncStatus_Success) um_fail("reading a buffer back", msg);
  atomic_store((_Atomic int *)u1, 1);
}

// GPU のバッファ src の先頭 bytes バイトを host に読み戻す（staging は MapRead のバッファ）。鍵を持って呼ぶ。
static void um_copy_back_locked(WGPUBuffer src, WGPUBuffer staging, void *host, uint64_t bytes) {
  um_submit_locked();
  WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(um_dev, NULL);
  wgpuCommandEncoderCopyBufferToBuffer(enc, src, 0, staging, 0, bytes);
  WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, NULL);
  wgpuCommandEncoderRelease(enc);
  um_last_index = wgpuQueueSubmitForIndex(um_queue, 1, &cmd);
  wgpuCommandBufferRelease(cmd);
  _Atomic int done = 0;
  WGPUBufferMapCallbackInfo ci = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
  ci.mode = WGPUCallbackMode_AllowProcessEvents;
  ci.callback = um_on_mapped;
  ci.userdata1 = (void *)&done;
  wgpuBufferMapAsync(staging, WGPUMapMode_Read, 0, (size_t)bytes, ci);
  double t0 = um_trace ? um_now() : 0;
  um_wait_flag(&done);
  double t1 = um_trace ? um_now() : 0;
  const void *p = wgpuBufferGetConstMappedRange(staging, 0, (size_t)bytes);
  if (!p) um_fail0("mapping a buffer");
  memcpy(host, p, (size_t)bytes);
  wgpuBufferUnmap(staging);
  if (um_trace) fprintf(stderr, "wgpu: wait %.3f ms, copy %.3f ms\n", (t1 - t0) * 1000.0, (um_now() - t1) * 1000.0);
}

// 領域の中身を GPU から読み戻す（GPU だけが新しいとき）。鍵を持って呼ぶ。
static void um_readback_locked(um_region *r) {
  double t0 = um_now();
  if (!r->staging) r->staging = um_make_buffer(r->used, WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst, 0);
  // 別名に書く。プログラム側は読み書き禁止のまま（ほかのスレッドは触れば鍵を待つ）。
  um_copy_back_locked(r->gbuf, r->staging, r->alias, r->used);
  r->host_valid = 1;
  um_reprotect(r);
  double dt = um_now() - t0;
  um_rb_spent += dt;
  um_note_read(r, dt);
  if (um_trace) fprintf(stderr, "wgpu: read back %llu bytes %.3f ms\n", (unsigned long long)r->used, dt * 1000.0);
}

// 領域の中身を GPU へ送る（CPU だけが新しいとき）。鍵を持って呼ぶ。
static void um_upload_locked(um_region *r) {
  if (!r->gbuf) {
    r->gbuf = um_make_buffer(r->used, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc, 0);
  }
  // 積んである dispatch が、この Buffer の前の中身を読むかもしれない。送る命令はそれより先に実行されるので、先に提出する。
  um_submit_locked();
  // 送る前に読み取り専用にする。送っているあいだにほかのスレッドが書けば、例外で（鍵を待ってから）GPU の写しを古いとする。
  r->gpu_valid = 1;
  um_reprotect(r);
  wgpuQueueWriteBuffer(um_queue, r->gbuf, 0, r->alias, (size_t)r->used);
  if (um_trace) fprintf(stderr, "wgpu: sent %llu bytes\n", (unsigned long long)r->used);
}

// 提出したあと、読み戻しを待っている領域を読み戻す。鍵を持って呼ぶ。
static void um_drain_eager_locked(void) {
  if (um_neagers == 0) return;
  um_submit_locked();
  for (size_t i = 0; i < um_neagers; i++) {
    um_eager *e = &um_eagers[i];
    if (e->r) {
      if (!e->r->host_valid) um_readback_locked(e->r);
      if (!um_lazy) e->r->gpu_valid = 0;  // 比較用の経路: 次の dispatch でも送り直す
    } else {
      WGPUBuffer staging = um_make_buffer(e->bytes, WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst, 0);
      um_copy_back_locked(e->temp, staging, e->host, e->bytes);
      wgpuBufferRelease(staging);
      wgpuBufferRelease(e->temp);
    }
  }
  um_neagers = 0;
}

static void um_push_eager(um_region *r, WGPUBuffer temp, void *host, uint64_t bytes) {
  if (um_neagers == um_capeagers) {
    um_capeagers = um_capeagers ? um_capeagers * 2 : 16;
    um_eagers = realloc(um_eagers, um_capeagers * sizeof *um_eagers);
    if (!um_eagers) um_panic(5, 0);
  }
  um_eagers[um_neagers++] = (um_eager){r, temp, host, bytes};
}

// ---- 例外（CPU が保護した領域に触った）----

// 触った場所が自分の領域なら、読み戻すか保護を外して 1 を返す（触った命令をやり直す）。そうでなければ 0。
// write: 1 = 書き込み、0 = 読み込み、-1 = 分からない（POSIX）。
static int um_on_fault(uintptr_t addr, int write) {
  if (addr < atomic_load(&um_lo) || addr >= atomic_load(&um_hi)) return 0;
  um_lock();
  um_region *r = um_region_containing(addr);
  if (!r) {
    um_unlock();
    return 0;
  }
  int was_readable = r->host_valid;
  if (!r->host_valid) um_readback_locked(r);
  // 読めるのに例外になったなら書き込み（読み取り専用だった）。CPU の写しが新しくなるので、GPU の写しは古くなる。
  if (write == 1 || (write < 0 && was_readable && r->prot == UM_RO)) {
    r->gpu_valid = 0;
    um_protect(r, UM_RW);
  }
  um_unlock();
  return 1;
}

#ifdef _WIN32
static LONG CALLBACK um_veh(PEXCEPTION_POINTERS ep) {
  EXCEPTION_RECORD *er = ep->ExceptionRecord;
  if (er->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || er->NumberParameters < 2 || um_lock_depth > 0) {
    return EXCEPTION_CONTINUE_SEARCH;
  }
  int write = er->ExceptionInformation[0] == 1;
  return um_on_fault((uintptr_t)er->ExceptionInformation[1], write) ? EXCEPTION_CONTINUE_EXECUTION : EXCEPTION_CONTINUE_SEARCH;
}
static void um_install_fault_handler(void) { AddVectoredExceptionHandler(1, um_veh); }
#else
static struct sigaction um_old_segv, um_old_bus;
static void um_sig(int sig, siginfo_t *info, void *ctx) {
  if (um_lock_depth == 0 && um_on_fault((uintptr_t)info->si_addr, -1)) return;
  // 自分の領域ではない。前の処理に渡す（既定の処理なら、元に戻して命令をやり直させ、既定の終わり方にする）。
  struct sigaction *old = sig == SIGSEGV ? &um_old_segv : &um_old_bus;
  if (old->sa_flags & SA_SIGINFO) {
    old->sa_sigaction(sig, info, ctx);
  } else if (old->sa_handler != SIG_DFL && old->sa_handler != SIG_IGN) {
    old->sa_handler(sig);
  } else {
    sigaction(sig, old, NULL);
  }
}
static void um_install_fault_handler(void) {
  struct sigaction sa;
  memset(&sa, 0, sizeof sa);
  sa.sa_sigaction = um_sig;
  sa.sa_flags = SA_SIGINFO | SA_NODEFER;
  sigemptyset(&sa.sa_mask);
  sigaction(SIGSEGV, &sa, &um_old_segv);
  sigaction(SIGBUS, &sa, &um_old_bus);
}
#endif

// ---- 確保と解放 ----

static int um_free_owned(void *p);
static int um_hooks_done;

void *um_wgpu_alloc(size_t count, size_t elem) {
  size_t bytes;
  if (__builtin_mul_overflow(count, elem, &bytes)) um_panic(5, 0);
  size_t page = um_pagesize();
  if (bytes > SIZE_MAX - 2 * page) um_panic(5, 0);
  size_t size = bytes == 0 ? page : (bytes + page - 1) / page * page;
  um_region *r = calloc(1, sizeof *r);
  if (!r) um_panic(5, 0);
  void *p, *alias;
#ifdef _WIN32
  // ページファイルを裏にした共有メモリ（中身は 0）を2か所に写す。
  r->section = CreateFileMappingW(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, (DWORD)((uint64_t)size >> 32), (DWORD)size, NULL);
  if (!r->section) um_panic(5, 0);
  p = MapViewOfFile(r->section, FILE_MAP_ALL_ACCESS, 0, 0, size);
  alias = MapViewOfFile(r->section, FILE_MAP_ALL_ACCESS, 0, 0, size);
  if (!p || !alias) um_panic(5, 0);
#else
  int fd = -1;
#ifdef __linux__
  fd = memfd_create("um-buffer", MFD_CLOEXEC);
#endif
  if (fd < 0) {
    // memfd がない（macOS、古い Linux）。名前を付けて作り、すぐ名前を消す。
    char name[64];
    static _Atomic unsigned um_shm_seq;
    snprintf(name, sizeof name, "/um-%ld-%u", (long)getpid(), atomic_fetch_add(&um_shm_seq, 1));
    fd = shm_open(name, O_RDWR | O_CREAT | O_EXCL, 0600);
    if (fd >= 0) shm_unlink(name);
  }
  if (fd < 0 || ftruncate(fd, (off_t)size) != 0) um_panic(5, 0);
  p = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  alias = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (p == MAP_FAILED || alias == MAP_FAILED) um_panic(5, 0);
#endif
  r->base = p;
  r->alias = alias;
  r->size = size;
  r->used = bytes < 4 ? 4 : (bytes + 3) / 4 * 4;
  r->host_valid = 1;
  r->prot = UM_RW;
  r->writer = -1;

  um_lock();
  if (!um_hooks_done) {
    um_hooks_done = 1;
    um_env_init();
    if (um_lazy) um_install_fault_handler();
    atomic_store_explicit(&um_free_owned_hook, um_free_owned, memory_order_release);
  }
  if (um_nregions == um_capregions) {
    um_capregions = um_capregions ? um_capregions * 2 : 64;
    um_regions = realloc(um_regions, um_capregions * sizeof *um_regions);
    if (!um_regions) um_panic(5, 0);
  }
  size_t i = um_region_lower((uintptr_t)p);
  memmove(&um_regions[i + 1], &um_regions[i], (um_nregions - i) * sizeof *um_regions);
  um_regions[i] = r;
  um_nregions++;
  if ((uintptr_t)p < atomic_load(&um_lo)) atomic_store(&um_lo, (uintptr_t)p);
  if ((uintptr_t)p + size > atomic_load(&um_hi)) atomic_store(&um_hi, (uintptr_t)p + size);
  um_unlock();
  return p;
}

static int um_free_owned(void *p) {
  if ((uintptr_t)p < atomic_load(&um_lo) || (uintptr_t)p >= atomic_load(&um_hi)) return 0;
  um_lock();
  size_t i = um_region_lower((uintptr_t)p);
  if (i >= um_nregions || um_regions[i]->base != p) {
    um_unlock();
    return 0;
  }
  um_region *r = um_regions[i];
  memmove(&um_regions[i], &um_regions[i + 1], (um_nregions - i - 1) * sizeof *um_regions);
  um_nregions--;
  // 読み戻しを待っている控えからも外す（解放した領域には書かない）。
  for (size_t k = 0; k < um_neagers; k++) {
    if (um_eagers[k].r == r) um_eagers[k].r = NULL, um_eagers[k].temp = NULL, um_eagers[k].bytes = 0;
  }
  size_t w = 0;
  for (size_t k = 0; k < um_neagers; k++) {
    if (um_eagers[k].r || um_eagers[k].temp) um_eagers[w++] = um_eagers[k];
  }
  um_neagers = w;
  // GPU のバッファは参照を放すだけ（提出済みの命令が使っていれば、wgpu が終わるまで生かしておく）。
  if (r->gbuf) wgpuBufferRelease(r->gbuf);
  if (r->staging) wgpuBufferRelease(r->staging);
  um_unlock();
#ifdef _WIN32
  UnmapViewOfFile(r->base);
  UnmapViewOfFile(r->alias);
  CloseHandle(r->section);
#else
  munmap(r->base, r->size);
  munmap(r->alias, r->size);
#endif
  free(r);
  return 1;
}

// ---- パイプライン ----

typedef struct {
  int32_t key;      // 埋め込んだ kernel は番号（0 以上）、実行時に読み込んだ kernel は -(番号 + 1)
  uint32_t group;   // dispatchGroups のグループの大きさ（0 は普通の dispatch）
  uint64_t mask;    // 書き込む Buffer の印（i ビット目 = i 番目の Buffer）
  int32_t nbufs;
  WGPUComputePipeline pipeline;  // NULL なら、作れなかった（実行時に読み込んだ kernel のみ）
  WGPUBindGroupLayout layout;
} um_pipe;
static um_pipe *um_pipes;
static size_t um_npipes, um_cappipes;

// 実行時に読み込んだ kernel（gpu.loadKernel）。
static WGPUShaderModule *um_loaded;
static size_t um_nloaded, um_caploaded;

typedef struct {
  _Atomic int done;
  int failed;
  char msg[1024];
} um_scope;
static void um_on_pop(WGPUPopErrorScopeStatus status, WGPUErrorType type, WGPUStringView msg, void *u1, void *u2) {
  (void)u2;
  um_scope *s = u1;
  if (status != WGPUPopErrorScopeStatus_Success || type != WGPUErrorType_NoError) {
    s->failed = 1;
    size_t n = msg.data ? (msg.length == WGPU_STRLEN ? strlen(msg.data) : msg.length) : 0;
    if (n >= sizeof s->msg) n = sizeof s->msg - 1;
    if (n) memcpy(s->msg, msg.data, n);
    s->msg[n] = 0;
  }
  atomic_store(&s->done, 1);
}
// 検証の誤りを捕まえる範囲。push のあとの呼び出しで起きた誤りを、pop が返す（止めない）。鍵を持って呼ぶ。
static void um_scope_push(void) { wgpuDevicePushErrorScope(um_dev, WGPUErrorFilter_Validation); }
static int um_scope_pop(char *msg, size_t cap) {
  um_scope s;
  memset(&s, 0, sizeof s);
  WGPUPopErrorScopeCallbackInfo ci = WGPU_POP_ERROR_SCOPE_CALLBACK_INFO_INIT;
  ci.mode = WGPUCallbackMode_AllowProcessEvents;
  ci.callback = um_on_pop;
  ci.userdata1 = &s;
  wgpuDevicePopErrorScope(um_dev, ci);
  while (!atomic_load(&s.done)) wgpuInstanceProcessEvents(um_inst);
  if (s.failed && msg) snprintf(msg, cap, "%s", s.msg);
  return !s.failed;
}

static WGPUShaderModule um_module(const char *src, size_t len) {
  WGPUShaderSourceWGSL wgsl = WGPU_SHADER_SOURCE_WGSL_INIT;
  wgsl.code = (WGPUStringView){src, len};
  WGPUShaderModuleDescriptor d = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
  d.nextInChain = &wgsl.chain;
  return wgpuDeviceCreateShaderModule(um_dev, &d);
}

// パイプラインを探すか作る。鍵を持って呼ぶ。soft なら、作れないとき（実行時に読み込んだ kernel）に止めずに NULL を返す。
static um_pipe *um_pipeline_locked(int32_t key, const char *src, int32_t src_len, uint32_t group, const uint64_t *table, int32_t nbufs, int soft) {
  uint64_t mask = 0;
  for (int32_t i = 0; i < nbufs; i++) {
    if (table[i * 4 + 3] != 0) mask |= 1ull << i;
  }
  for (size_t i = 0; i < um_npipes; i++) {
    um_pipe *p = &um_pipes[i];
    if (p->key == key && p->group == group && p->mask == mask && p->nbufs == nbufs) return p->pipeline ? p : NULL;
  }
  if (um_npipes == um_cappipes) {
    um_cappipes = um_cappipes ? um_cappipes * 2 : 16;
    um_pipes = realloc(um_pipes, um_cappipes * sizeof *um_pipes);
    if (!um_pipes) um_panic(5, 0);
  }
  um_pipe *p = &um_pipes[um_npipes++];
  memset(p, 0, sizeof *p);
  p->key = key;
  p->group = group;
  p->mask = mask;
  p->nbufs = nbufs;

  if (soft) um_scope_push();
  WGPUShaderModule module;
  if (key >= 0) {
    module = um_module(src, (size_t)src_len);
  } else {
    module = um_loaded[-key - 1];
    wgpuShaderModuleAddRef(module);
  }
  // 使っていない Buffer があっても並びがずれないよう、束縛の配置は明示する（runtime/host.mjs と同じ）。
  WGPUBindGroupLayoutEntry entries[65];
  for (int32_t i = 0; i <= nbufs; i++) {
    WGPUBindGroupLayoutEntry e = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    e.binding = (uint32_t)i;
    e.visibility = WGPUShaderStage_Compute;
    e.buffer.type = i > 0 && (mask >> (i - 1) & 1) ? WGPUBufferBindingType_Storage : WGPUBufferBindingType_ReadOnlyStorage;
    entries[i] = e;
  }
  WGPUBindGroupLayoutDescriptor ld = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
  ld.entryCount = (size_t)nbufs + 1;
  ld.entries = entries;
  p->layout = wgpuDeviceCreateBindGroupLayout(um_dev, &ld);
  WGPUPipelineLayoutDescriptor pld = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
  pld.bindGroupLayoutCount = 1;
  pld.bindGroupLayouts = &p->layout;
  WGPUPipelineLayout pl = wgpuDeviceCreatePipelineLayout(um_dev, &pld);
  WGPUComputePipelineDescriptor cd = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
  cd.layout = pl;
  cd.compute.module = module;
  cd.compute.entryPoint = um_sv("um_main");
  WGPUConstantEntry wg = WGPU_CONSTANT_ENTRY_INIT;
  if (group) {
    // dispatchGroups の kernel は、グループの大きさ（シェーダの定数 um_wg）ごとに別のパイプラインになる。
    wg.key = um_sv("um_wg");
    wg.value = (double)group;
    cd.compute.constantCount = 1;
    cd.compute.constants = &wg;
  }
  p->pipeline = wgpuDeviceCreateComputePipeline(um_dev, &cd);
  wgpuPipelineLayoutRelease(pl);
  wgpuShaderModuleRelease(module);
  if (soft) {
    char msg[1024];
    if (!um_scope_pop(msg, sizeof msg)) {
      fprintf(stderr, "warning: gpu.runKernel: the kernel cannot be run with these arguments: %s\n", msg);
      if (p->pipeline) wgpuComputePipelineRelease(p->pipeline);
      p->pipeline = NULL;
    }
  }
  if (!p->pipeline) {
    if (!soft) um_fail0("creating the pipeline");
    return NULL;
  }
  return p;
}

// ---- dispatch ----

// dispatch を1つ積む（提出はしない）。鍵を持って呼ぶ。eager_all なら、書いた Buffer を提出のあとで読み戻す。
static void um_encode_locked(um_pipe *pipe, uint32_t count, uint32_t group, const void *block, int32_t block_bytes,
                             const uint64_t *table, int32_t nbufs, int eager_all) {
  if (group > um_limits.maxComputeInvocationsPerWorkgroup || group > um_limits.maxComputeWorkgroupSizeX) {
    fflush(stdout);
    fprintf(stderr, "panic: the group size %u is larger than this GPU allows (%u)\n", group, um_limits.maxComputeInvocationsPerWorkgroup);
    um_exit_failure();
  }
  // 要素数をワークグループの2次元の格子に割り当てる。dispatchGroups では、グループがそのままワークグループ。
  uint32_t groups = group ? count / group : (uint32_t)(((uint64_t)count + UM_WG - 1) / UM_WG);
  uint32_t maxdim = um_limits.maxComputeWorkgroupsPerDimension;
  uint32_t x = groups < maxdim ? groups : maxdim;
  uint32_t y = (uint32_t)(((uint64_t)groups + x - 1) / x);
  if (y > maxdim) um_fail0("too many elements for one dispatch");

  WGPUBindGroupEntry entries[65];
  WGPUBuffer temps[64];
  um_region *regs[64];
  for (int32_t i = 0; i < nbufs; i++) {
    void *host = (void *)(uintptr_t)table[i * 4];
    uint64_t len = table[i * 4 + 1], eb = table[i * 4 + 2];
    um_region *r = um_region_at((uintptr_t)host);
    regs[i] = r;
    temps[i] = NULL;
    WGPUBindGroupEntry e = WGPU_BIND_GROUP_ENTRY_INIT;
    e.binding = (uint32_t)i + 1;
    if (r) {
      if (!r->gpu_valid || !um_lazy) um_upload_locked(r);
      e.buffer = r->gbuf;
      e.size = r->used;
    } else {
      // um_wgpu_alloc で確保していない Buffer（ふつうは起きない）。一時的なバッファに送り、提出のあとで読み戻す。
      uint64_t used = (len * eb + 3) / 4 * 4;
      uint64_t size = used < 4 ? 4 : used;
      temps[i] = um_make_buffer(size, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc, 0);
      um_submit_locked();
      if (used > 0) wgpuQueueWriteBuffer(um_queue, temps[i], 0, host, (size_t)used);
      e.buffer = temps[i];
      e.size = size;
    }
    entries[i + 1] = e;
  }

  // 引数の塊: 個数、1行の要素数、各 Buffer の長さ、値の引数（コンパイラが詰めた並びのまま）。
  uint64_t header = (uint64_t)(2 + nbufs) * 4;
  uint64_t psize = (header + (uint64_t)block_bytes + 3) / 4 * 4;
  WGPUBuffer params = um_make_buffer(psize, WGPUBufferUsage_Storage, 1);
  unsigned char *pp = wgpuBufferGetMappedRange(params, 0, (size_t)psize);
  if (!pp) um_fail0("mapping the argument buffer");
  memset(pp, 0, (size_t)psize);
  uint32_t words[2] = {group ? groups : count, group ? x : x * UM_WG};
  memcpy(pp, words, sizeof words);
  for (int32_t i = 0; i < nbufs; i++) {
    uint32_t len = (uint32_t)table[i * 4 + 1];
    memcpy(pp + 8 + 4 * i, &len, 4);
  }
  if (block_bytes > 0) memcpy(pp + header, block, (size_t)block_bytes);
  wgpuBufferUnmap(params);
  WGPUBindGroupEntry pe = WGPU_BIND_GROUP_ENTRY_INIT;
  pe.binding = 0;
  pe.buffer = params;
  pe.size = psize;
  entries[0] = pe;

  WGPUBindGroupDescriptor bd = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
  bd.layout = pipe->layout;
  bd.entryCount = (size_t)nbufs + 1;
  bd.entries = entries;
  WGPUBindGroup bg = wgpuDeviceCreateBindGroup(um_dev, &bd);

  if (!um_enc) um_enc = wgpuDeviceCreateCommandEncoder(um_dev, NULL);
  um_enc_count++;
  um_enc_elems += count;
  WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(um_enc, NULL);
  wgpuComputePassEncoderSetPipeline(pass, pipe->pipeline);
  wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, NULL);
  if (groups > 0) wgpuComputePassEncoderDispatchWorkgroups(pass, x, y, 1);
  wgpuComputePassEncoderEnd(pass);
  wgpuComputePassEncoderRelease(pass);
  wgpuBindGroupRelease(bg);
  wgpuBufferRelease(params);

  // 書いた Buffer は GPU だけが新しくなる。CPU が触れば読み戻す（遅延）。
  for (int32_t i = 0; i < nbufs; i++) {
    int writes = table[i * 4 + 3] != 0;
    um_region *r = regs[i];
    if (r) {
      if (writes) {
        // 前に GPU が書いた結果を、CPU が一度も読まずに上書きする（その kernel の結果は GPU の中で使われた）。
        if (!r->host_valid && r->writer >= 0) um_note_unread(r->writer);
        r->writer = pipe->key >= 0 && pipe->key < 256 ? pipe->key : -1;
        r->wcount = count;
        r->host_valid = 0;
        r->gpu_valid = 1;
        um_reprotect(r);
        if (!um_lazy || eager_all) um_push_eager(r, NULL, NULL, 0);
      }
    } else if (writes) {
      um_push_eager(NULL, temps[i], (void *)(uintptr_t)table[i * 4], (table[i * 4 + 1] * table[i * 4 + 2] + 3) / 4 * 4);
    } else {
      wgpuBufferRelease(temps[i]);
    }
  }
  if (um_trace) fprintf(stderr, "wgpu: dispatch %u (%u x %u groups)\n", count, x, y);
}

// ---- 自動判断（dispatch）----
// 実測と Buffer の置き場所から、速いと見込まれるほうを選ぶ（docs/wgpu.ja.md「自動判断」）。
//   GPU の時間 = 実行 + 送る量（GPU の写しが古い Buffer）× 送る速さ
//              + 書く Buffer の量 × 読み戻す速さ × CPU がその結果を読む割合
//   CPU の時間 = 実行 + 読み戻す量（GPU だけが新しい Buffer）× 読み戻す速さ
// 実行の時間は kernel ごとに測る（GPU は試しの回に、送り終えてから完了まで待って測る。CPU は読み戻しの時間を除く）。
// 送る速さと読み戻す速さはプログラム全体で1つ（試しの回に測る）。
// 割合は、その kernel が GPU で書いた結果を、次に GPU が書く前に、kernel の外の CPU のコード（表示など）が読んだか
// （1）読まなかったか（0）を、ならして覚える。CPU で実行した kernel が読んだときの読み戻しは、その kernel の CPU の
// 時間の側に入るので、割合には数えない。
//
// Buffer の置き場所だけで判断すると、どの Buffer も CPU にあるときは、どの kernel も単独では送る分だけ遅く見えて
// 「全部 CPU」から抜け出せない（GPU で続けて実行すれば速い場合でも）。そこで、同じ kernel が繰り返し実行される
// プログラムでは、送る費用を見込まない判断（「GPU に寄せる」）と、見込む判断（ふつう）を交互に試し、
// 繰り返しの周期（目印の kernel が実行される間隔）が短いほうを使う。試すのは、送る費用だけで CPU を選んでいる
// kernel があるときと、GPU に寄せているあいだ（ときどき、ふつうに戻して確かめる）。

// GPU を候補にする最小の仕事の大きさ（CPU で実行したときの秒数）。wgpu は提出して読み戻すまで 0.1〜0.3 ms かかる。
#define UM_GPU_MIN_SECONDS 100e-6
#define UM_MAX_TRACKED 256
#define UM_REPROBE 64
typedef struct {
  uint32_t cpu_runs, gpu_runs, since_probe, backoff;
  int last_winner;
  double cpu_per_elem, gpu_per_elem;
  uint32_t rb_seen;
  double rb_ratio;
} um_stat;
static um_stat um_stats[UM_MAX_TRACKED];
static _Thread_local int32_t um_cpu_kernel = -1, um_gpu_kernel_measure = -1;
static _Thread_local double um_cpu_start, um_cpu_rb0;
static _Thread_local uint32_t um_next_group;
static int um_cpu_running;  // 自動判断で CPU を選んで実行中の kernel の数（その読み戻しは割合に数えない）

// 送る速さと読み戻す速さ（秒 / バイト）。測るまでは 1 バイト 0.25 ns（4 GB/s）と見込む。
static double um_up_rate = 0.25e-9, um_rb_rate = 0.25e-9;
static uint32_t um_up_runs, um_rb_runs;

static void um_blend(double *slot, double sample, uint32_t runs) {
  if (runs == 0) *slot = sample;
  else if (runs == 1) *slot = sample < *slot ? sample : *slot;
  else *slot = *slot * 0.75 + sample * 0.25;
}

void um_gpu_group(int32_t size) { um_next_group = (uint32_t)size; }
static uint32_t um_take_group(void) {
  uint32_t g = um_next_group;
  um_next_group = 0;
  return g;
}

// ---- 周期の比較（GPU に寄せるか）----

static int um_lean;               // 1 = GPU に寄せる（送る費用を見込まない）
static int32_t um_anchor = -1;    // 周期を測る目印の kernel
static double um_anchor_last;
static int um_exp_phase;          // 0 = 待つ、1 = いまの判断で測る、2 = もう一方の判断で測る
static uint32_t um_exp_skip, um_exp_n, um_exp_wait, um_exp_backoff;
static double um_exp_samples[8], um_exp_base;
static int um_stuck;              // 送る費用だけで CPU を選んだ kernel があった
#define UM_EXP_SAMPLES 6

// 周期の代表値: いちばん長いもの（たまたま遅れた回）を除いた平均。
static double um_exp_typical(void) {
  double sum = 0, max = 0;
  for (uint32_t i = 0; i < UM_EXP_SAMPLES; i++) {
    sum += um_exp_samples[i];
    if (um_exp_samples[i] > max) max = um_exp_samples[i];
  }
  return (sum - max) / (UM_EXP_SAMPLES - 1);
}

// 目印の kernel の判断のたびに呼ぶ（鍵を持って）。
static void um_exp_tick(void) {
  double now = um_now();
  double period = um_anchor_last > 0 ? now - um_anchor_last : 0;
  um_anchor_last = now;
  if (um_exp_phase == 0) {
    if (um_exp_wait > 0) um_exp_wait--;
    else if (um_lean || um_stuck) um_exp_phase = 1, um_exp_skip = 1, um_exp_n = 0;
    return;
  }
  // 判断を切り替えた直後の数回は、Buffer の移動を含むので数えない。
  if (um_exp_skip > 0) {
    um_exp_skip--;
    return;
  }
  um_exp_samples[um_exp_n++] = period;
  if (um_exp_n < UM_EXP_SAMPLES) return;
  double typical = um_exp_typical();
  if (um_exp_phase == 1) {
    um_exp_base = typical;
    um_lean = !um_lean;
    um_exp_phase = 2, um_exp_skip = 2, um_exp_n = 0;
    return;
  }
  // GPU に寄せた判断は、試しに CPU で実行した kernel があっても GPU に戻りやすい。ふつうの判断は、そのとき Buffer が
  // CPU に戻ると「全部 CPU」に落ちることがあるので、寄せた判断からふつうに戻すのは、はっきり速いときだけにする。
  int keep = typical < um_exp_base * (um_lean ? 0.95 : 0.9);
  if (um_trace) {
    fprintf(stderr, "gpu: period %.3f ms (%s, tried) vs %.3f ms (%s): use %s\n", typical * 1000.0, um_lean ? "lean to GPU" : "normal",
            um_exp_base * 1000.0, um_lean ? "normal" : "lean to GPU", (keep ? um_lean : !um_lean) ? "lean to GPU" : "normal");
  }
  if (keep) {
    um_exp_backoff = 0;
  } else {
    um_lean = !um_lean;
    if (um_exp_backoff < 6) um_exp_backoff++;
  }
  um_exp_phase = 0;
  um_exp_wait = 16u << um_exp_backoff;
  um_stuck = 0;
}

// ---- 判断 ----

// Buffer の表から、GPU で実行するときに送る量・書く量と、CPU で実行するときに読み戻す量を数える（鍵を持って）。
static void um_bytes_locked(const uint64_t *table, int32_t nbufs, double *up, double *out, double *back) {
  *up = *out = *back = 0;
  for (int32_t i = 0; i < nbufs; i++) {
    um_region *r = um_region_at((uintptr_t)table[i * 4]);
    double bytes = r ? (double)r->used : (double)(table[i * 4 + 1] * table[i * 4 + 2]);
    if (!r || !r->gpu_valid || !um_lazy) *up += bytes;
    if (table[i * 4 + 3] != 0) *out += bytes;
    if (r && !r->host_valid) *back += bytes;
  }
}

int32_t um_gpu_pick_bufs(int32_t kernel, uint32_t count, const uint64_t *table, int32_t nbufs) {
  um_env_init();
  if (um_gpu_off) return 0;
  if (um_fixed_rule || kernel < 0 || kernel >= UM_MAX_TRACKED) return count >= um_threshold && um_gpu_available();
  um_lock();
  um_stat *s = &um_stats[kernel];
  double up, out, back;
  um_bytes_locked(table, nbufs, &up, &out, &back);
  int gpu, measure = 0;
  double cpu_time = s->cpu_per_elem * count + back * um_rb_rate;
  if (s->cpu_runs < 2) gpu = 0;
  else if (cpu_time < UM_GPU_MIN_SECONDS) gpu = 0;
  else if (s->gpu_runs < 2) gpu = 1, measure = 1;
  else {
    double seen = s->rb_seen ? s->rb_ratio : 0.5;
    double base = s->gpu_per_elem * count + out * um_rb_rate * seen;
    double send = up * um_up_rate;
    double gpu_time = base + (um_lean ? 0 : send);
    gpu = gpu_time < cpu_time;
    // 送る費用がなければ GPU が速い kernel。周期の比較の目印にする。
    if (!um_lean && !gpu && base < cpu_time) {
      um_stuck = 1;
      if (um_anchor < 0) um_anchor = kernel;
    }
    if (gpu != s->last_winner) {
      s->last_winner = gpu;
      s->backoff = 0;
      s->since_probe = 0;
    }
    double ratio = gpu ? cpu_time / gpu_time : gpu_time / cpu_time;
    if (!(ratio >= 1)) ratio = 1;
    if (ratio > 64) ratio = 64;
    if (++s->since_probe >= (uint32_t)(UM_REPROBE * ratio) << s->backoff) {
      s->since_probe = 0;
      if (s->backoff < 6) s->backoff++;
      gpu = !gpu;
      measure = gpu;
    }
  }
  if (kernel == um_anchor) um_exp_tick();
  int ok = !gpu || um_init_locked();
  if (gpu && !ok) gpu = 0;
  if (!gpu) um_cpu_running++;
  um_unlock();
  if (gpu) {
    if (measure) um_gpu_kernel_measure = kernel;
  } else {
    um_cpu_kernel = kernel;
    um_cpu_rb0 = um_rb_spent;
    um_cpu_start = um_now();
  }
  if (um_trace >= 2) {
    fprintf(stderr, "pick: kernel %d count %u -> %s (send %.0f, write %.0f, back %.0f bytes)\n", kernel, count, gpu ? "gpu" : "cpu", up, out, back);
  }
  return gpu;
}

// Buffer の表を渡さない版（表がなければ、置き場所は見込まない）。
int32_t um_gpu_pick(int32_t kernel, uint32_t count) { return um_gpu_pick_bufs(kernel, count, NULL, 0); }

// 読み戻し（鍵を持って呼ぶ）: 試しの回なら読み戻す速さを測る。CPU のコードが読んだのなら、割合を 1 に寄せる。
static void um_note_read(um_region *r, double seconds) {
  if (um_forced_readback && r->used >= 65536) {
    um_blend(&um_rb_rate, seconds / (double)r->used, um_rb_runs);
    if (um_rb_runs < 2) um_rb_runs++;
  }
  int32_t k = r->writer;
  r->writer = -1;
  if (k < 0 || k >= UM_MAX_TRACKED || um_forced_readback || um_cpu_running > 0) return;
  um_stat *s = &um_stats[k];
  s->rb_ratio = s->rb_seen ? s->rb_ratio * 0.5 + 0.5 : 1.0;
  s->rb_seen = 1;
}
static void um_note_unread(int32_t k) {
  if (k < 0 || k >= UM_MAX_TRACKED) return;
  um_stat *s = &um_stats[k];
  s->rb_ratio = s->rb_seen ? s->rb_ratio * 0.5 : 0.0;
  s->rb_seen = 1;
}

void um_gpu_cpu_done(int32_t kernel, uint32_t count) {
  if (kernel != um_cpu_kernel) return;
  um_cpu_kernel = -1;
  double elapsed = um_now() - um_cpu_start;
  um_lock();
  um_cpu_running--;
  // GPU だけが新しい Buffer を読み戻した時間は除く（その分は判断のときに Buffer の量から見込む）。
  elapsed -= um_rb_spent - um_cpu_rb0;
  if (elapsed < 0) elapsed = 0;
  if (count > 0) {
    um_stat *s = &um_stats[kernel];
    um_blend(&s->cpu_per_elem, elapsed / count, s->cpu_runs);
    if (s->cpu_runs < 2) s->cpu_runs++;
  }
  um_unlock();
  if (um_trace) fprintf(stderr, "cpu: kernel %d count %u %.3f ms\n", kernel, count, elapsed * 1000.0);
}

// 空の命令を提出して、GPU が終わるまで待つ（鍵を持って）。それまでに積んだ送りもすべて終わる。
static void um_idle_locked(void) {
  um_submit_locked();
  WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(um_dev, NULL);
  WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, NULL);
  wgpuCommandEncoderRelease(enc);
  um_last_index = wgpuQueueSubmitForIndex(um_queue, 1, &cmd);
  wgpuCommandBufferRelease(cmd);
  wgpuDevicePoll(um_dev, 1, NULL);
}

static void um_dispatch(int32_t kernel, const char *src, int32_t src_len, uint32_t count, const void *block,
                        int32_t block_bytes, const uint64_t *table, int32_t nbufs, int submit) {
  uint32_t group = um_take_group();
  int measure = um_gpu_kernel_measure == kernel;
  um_gpu_kernel_measure = -1;
  if (count == 0) return;
  if (nbufs > 64) um_fail0("too many buffers");
  um_lock();
  if (!um_init_locked()) um_fail0("no GPU");
  um_pipe *p = um_pipeline_locked(kernel, src, src_len, group, table, nbufs, 0);
  if (measure && um_lazy) {
    // 試す回: 前の仕事を終わらせ、古い Buffer を送り終えるまでを送る速さとして測り、それから実行の時間を測る。
    um_idle_locked();
    double t0 = um_now();
    uint64_t sent = 0;
    for (int32_t i = 0; i < nbufs; i++) {
      um_region *r = um_region_at((uintptr_t)table[i * 4]);
      if (r && !r->gpu_valid) {
        um_upload_locked(r);
        sent += r->used;
      }
    }
    if (sent > 0) {
      um_idle_locked();
      if (sent >= 65536) {
        um_blend(&um_up_rate, (um_now() - t0) / (double)sent, um_up_runs);
        if (um_up_runs < 2) um_up_runs++;
      }
    }
    double t1 = um_now();
    um_encode_locked(p, count, group, block, block_bytes, table, nbufs, 1);
    um_submit_locked();
    wgpuDevicePoll(um_dev, 1, NULL);
    double elapsed = um_now() - t1;
    um_stat *s = &um_stats[kernel];
    um_blend(&s->gpu_per_elem, elapsed / count, s->gpu_runs);
    if (s->gpu_runs < 2) s->gpu_runs++;
    // 書いた Buffer を読み戻して、読み戻す速さを測る（um_note_read）。
    um_forced_readback = 1;
    um_drain_eager_locked();
    um_forced_readback = 0;
    if (um_trace) fprintf(stderr, "gpu: kernel %d count %u %.3f ms (measured, sent %llu bytes)\n", kernel, count, elapsed * 1000.0, (unsigned long long)sent);
  } else {
    um_encode_locked(p, count, group, block, block_bytes, table, nbufs, 0);
    if (submit || !um_lazy) um_submit_locked();
    if (!um_lazy) um_drain_eager_locked();
  }
  um_unlock();
}

void um_gpu_dispatch(int32_t kernel, const char *src, int32_t src_len, uint32_t count, const void *block,
                     int32_t block_bytes, const uint64_t *table, int32_t nbufs) {
  um_dispatch(kernel, src, src_len, count, block, block_bytes, table, nbufs, 0);
}

// 積んである dispatch を提出する。完了は待たない（CPU が結果を読むときに、読み戻しが待つ）。
void um_gpu_flush(void) {
  um_lock();
  // 小さな仕事は、いくつかまとめてから提出する（提出1回に 20 µs ほどかかるため）。大きな仕事はすぐ提出して、
  // GPU が CPU と並んで動けるようにする。まとめているあいだも、CPU が結果に触れば読み戻しの前に提出する。
  if (um_ready > 0 && (um_enc_count >= UM_BATCH_COUNT || um_enc_elems >= UM_BATCH_ELEMS)) um_submit_locked();
  um_unlock();
}

// ---- gpu.submit / gpu.ready / gpu.wait ----

static void um_on_work_done(WGPUQueueWorkDoneStatus status, WGPUStringView msg, void *u1, void *u2) {
  (void)status, (void)msg, (void)u2;
  uint64_t t = (uint64_t)(uintptr_t)u1;
  uint64_t cur = atomic_load(&um_done);
  while (cur < t && !atomic_compare_exchange_weak(&um_done, &cur, t)) {
  }
}

int64_t um_gpu_submit(int32_t kernel, const char *src, int32_t src_len, uint32_t count, const void *block,
                      int32_t block_bytes, const uint64_t *table, int32_t nbufs) {
  um_dispatch(kernel, src, src_len, count, block, block_bytes, table, nbufs, 1);
  if (count == 0) return 0;
  um_lock();
  uint64_t t = um_submits;
  WGPUQueueWorkDoneCallbackInfo ci = WGPU_QUEUE_WORK_DONE_CALLBACK_INFO_INIT;
  ci.mode = WGPUCallbackMode_AllowProcessEvents;
  ci.callback = um_on_work_done;
  ci.userdata1 = (void *)(uintptr_t)t;
  wgpuQueueOnSubmittedWorkDone(um_queue, ci);
  um_unlock();
  return (int64_t)t;
}

int32_t um_gpu_ready(int64_t t) {
  if (t <= 0 || atomic_load(&um_done) >= (uint64_t)t) return 1;
  um_lock();
  wgpuDevicePoll(um_dev, 0, NULL);
  wgpuInstanceProcessEvents(um_inst);
  um_unlock();
  return atomic_load(&um_done) >= (uint64_t)t;
}

void um_gpu_wait(int64_t t) {
  while (!um_gpu_ready(t)) {
    um_lock();
    wgpuDevicePoll(um_dev, 1, NULL);
    wgpuInstanceProcessEvents(um_inst);
    um_unlock();
  }
}

// gpu.flags（Metal の gpu.submit の印）。wgpu では効果がなく、覚えて前の値を返すだけ。
static _Thread_local uint32_t um_flags;
int32_t um_gpu_set_flags(int32_t f) {
  uint32_t old = um_flags;
  um_flags = (uint32_t)f;
  return (int32_t)old;
}

// ---- 実行時に読み込む kernel（gpu.loadKernel / gpu.runKernel。docs/wgpu.ja.md）----

// WGSL のソースを読み込む。番号 + 1 を返す。GPU が使えないか、WGSL をコンパイルできなければ 0。
int32_t um_gpu_load(const char *src, size_t len) {
  um_lock();
  if (!um_init_locked()) {
    um_unlock();
    return 0;
  }
  um_scope_push();
  WGPUShaderModule m = um_module(src, len);
  char msg[1024];
  int ok = um_scope_pop(msg, sizeof msg);
  if (!ok || !m) {
    fprintf(stderr, "warning: gpu.loadKernel: the WGSL could not be compiled: %s\n", msg);
    if (m) wgpuShaderModuleRelease(m);
    um_unlock();
    return 0;
  }
  if (um_nloaded == um_caploaded) {
    um_caploaded = um_caploaded ? um_caploaded * 2 : 8;
    um_loaded = realloc(um_loaded, um_caploaded * sizeof *um_loaded);
    if (!um_loaded) um_panic(5, 0);
  }
  um_loaded[um_nloaded++] = m;
  int32_t id = (int32_t)um_nloaded;
  um_unlock();
  return id;
}

// 読み込んだ kernel を実行する。実行を頼めたら 1。束縛の並びが kernel と合わないなど、実行できなければ 0（警告を出す）。
int32_t um_gpu_run(int32_t id, uint32_t count, const void *block, int32_t block_bytes, const uint64_t *table, int32_t nbufs) {
  if (id <= 0 || nbufs > 64) return 0;
  um_lock();
  if (um_ready <= 0 || (size_t)id > um_nloaded) {
    um_unlock();
    return 0;
  }
  if (count == 0) {
    um_unlock();
    return 1;
  }
  um_pipe *p = um_pipeline_locked(-id, NULL, 0, 0, table, nbufs, 1);
  if (!p) {
    um_unlock();
    return 0;
  }
  // 引数の大きさが kernel と合わないと、検証の誤りは命令を閉じるときに出る。それを捕まえるため、別の命令にして閉じる。
  // 失敗したときに戻せるよう、Buffer の印を控えておく（送るのは成功しても失敗しても正しい）。
  um_submit_locked();
  uint8_t saved[64][2];
  for (int32_t i = 0; i < nbufs; i++) {
    um_region *r = um_region_at((uintptr_t)table[i * 4]);
    saved[i][0] = r ? r->host_valid : 1;
    saved[i][1] = r ? r->gpu_valid : 0;
  }
  um_scope_push();
  um_encode_locked(p, count, 0, block, block_bytes, table, nbufs, 0);
  WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(um_enc, NULL);
  wgpuCommandEncoderRelease(um_enc);
  um_enc = NULL;
  um_enc_count = 0;
  um_enc_elems = 0;
  char msg[1024];
  int ok = um_scope_pop(msg, sizeof msg);
  if (ok) {
    um_last_index = wgpuQueueSubmitForIndex(um_queue, 1, &cmd);
    um_submits++;
    if (!um_lazy) um_drain_eager_locked();
  } else {
    fprintf(stderr, "warning: gpu.runKernel: the kernel cannot be run with these arguments: %s\n", msg);
    // 書くはずだった Buffer の印を元に戻す（命令は提出しないので、GPU の写しは変わらない）。
    for (int32_t i = 0; i < nbufs; i++) {
      um_region *r = um_region_at((uintptr_t)table[i * 4]);
      if (r && table[i * 4 + 3] != 0) {
        r->host_valid = saved[i][0];
        r->gpu_valid = saved[i][1];
        um_reprotect(r);
      }
    }
    for (size_t k = 0; k < um_neagers; k++) {
      if (um_eagers[k].temp) wgpuBufferRelease(um_eagers[k].temp);
    }
    um_neagers = 0;
  }
  wgpuCommandBufferRelease(cmd);
  um_unlock();
  return ok;
}

// ---- 描画（gpu.beginImage / gpu.draw / gpu.endImage。src/render.ts、docs/render.ja.md）----
// 画像は色（BGRA8Unorm。読み戻すと 0xAARRGGBB の u32 と同じバイトの並び）と深度（Depth32Float）のテクスチャの組。
// draw は dispatch と同じ命令（um_enc）に描画のパスを積むので、前後の dispatch との順序は保たれる。
// 最初の draw のパスが色と深度を埋め（clear）、あとの draw は前の結果に重ねて描く。

static void um_render_panic(const char *msg) {
  fflush(stdout);
  fprintf(stderr, "panic: %s\n", msg);
  um_exit_failure();
}

typedef struct {
  int used;
  uint32_t width, height;
  uint32_t clear;   // 0xAARRGGBB
  int drawn;        // 色と深度を埋めるパスをもう積んだか
  int32_t window;   // ウィンドウに描くフレーム（gpu.beginFrame）なら、そのウィンドウの番号。画像なら 0
  int skip;         // ウィンドウの面を得られなかったフレーム（描かずに捨てる）
  WGPUTextureFormat format;
  WGPUTexture color, depth;
  WGPUTextureView color_view, depth_view;
} um_image;
static um_image *um_images;
static size_t um_nimages, um_capimages;

static um_image *um_image_at(int32_t image, const char *what) {
  if (image < 1 || (size_t)image > um_nimages || !um_images[image - 1].used) {
    char msg[128];
    snprintf(msg, sizeof msg, "%s: %d is not an image from gpu.beginImage (or it has already ended)", what, (int)image);
    um_render_panic(msg);
  }
  return &um_images[image - 1];
}

static WGPUTexture um_texture(uint32_t w, uint32_t h, WGPUTextureFormat format, WGPUTextureUsage usage) {
  WGPUTextureDescriptor d = WGPU_TEXTURE_DESCRIPTOR_INIT;
  d.size.width = w;
  d.size.height = h;
  d.size.depthOrArrayLayers = 1;
  d.format = format;
  d.usage = usage;
  WGPUTexture t = wgpuDeviceCreateTexture(um_dev, &d);
  if (!t) um_fail0("creating a texture");
  return t;
}

// GPU が使えなければ 0 を返す（画像は CPU に用意され、CPU で描く。runtime/render.um）。
// 空いている画像の場所を取る（鍵を持って呼ぶ）。
static size_t um_image_slot_locked(void) {
  for (size_t i = 0; i < um_nimages; i++) {
    if (!um_images[i].used) return i;
  }
  if (um_nimages == um_capimages) {
    um_capimages = um_capimages ? um_capimages * 2 : 4;
    um_images = realloc(um_images, um_capimages * sizeof *um_images);
    if (!um_images) um_panic(5, 0);
  }
  return um_nimages++;
}

int32_t um_gpu_image_begin(int32_t width, int32_t height, int32_t clear) {
  um_lock();
  if (!um_init_locked()) {
    um_unlock();
    return 0;
  }
  if (width <= 0 || height <= 0 || (uint32_t)width > um_limits.maxTextureDimension2D || (uint32_t)height > um_limits.maxTextureDimension2D) {
    char msg[160];
    snprintf(msg, sizeof msg, "gpu.beginImage: the size %d x %d is not between 1 and %u", (int)width, (int)height, um_limits.maxTextureDimension2D);
    um_render_panic(msg);
  }
  size_t slot = um_image_slot_locked();
  um_image *im = &um_images[slot];
  memset(im, 0, sizeof *im);
  im->used = 1;
  im->width = (uint32_t)width;
  im->height = (uint32_t)height;
  im->clear = (uint32_t)clear;
  im->format = WGPUTextureFormat_BGRA8Unorm;
  im->color = um_texture(im->width, im->height, WGPUTextureFormat_BGRA8Unorm, WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc);
  im->depth = um_texture(im->width, im->height, WGPUTextureFormat_Depth32Float, WGPUTextureUsage_RenderAttachment);
  im->color_view = wgpuTextureCreateView(im->color, NULL);
  im->depth_view = wgpuTextureCreateView(im->depth, NULL);
  um_unlock();
  return (int32_t)slot + 1;
}

// 描画のパイプライン（シェーダの組の番号、Buffer の数、色の形式ごと）。
typedef struct {
  int32_t key;
  int32_t nbufs;
  WGPUTextureFormat format;
  WGPURenderPipeline pipeline;
  WGPUBindGroupLayout layout;
} um_rpipe;
static um_rpipe *um_rpipes;
static size_t um_nrpipes, um_caprpipes;

static um_rpipe *um_rpipeline_locked(int32_t key, const char *src, int32_t src_len, int32_t nbufs, WGPUTextureFormat format) {
  for (size_t i = 0; i < um_nrpipes; i++) {
    if (um_rpipes[i].key == key && um_rpipes[i].nbufs == nbufs && um_rpipes[i].format == format) return &um_rpipes[i];
  }
  if (um_nrpipes == um_caprpipes) {
    um_caprpipes = um_caprpipes ? um_caprpipes * 2 : 8;
    um_rpipes = realloc(um_rpipes, um_caprpipes * sizeof *um_rpipes);
    if (!um_rpipes) um_panic(5, 0);
  }
  um_rpipe *p = &um_rpipes[um_nrpipes++];
  memset(p, 0, sizeof *p);
  p->key = key;
  p->nbufs = nbufs;
  p->format = format;
  WGPUShaderModule module = um_module(src, (size_t)src_len);
  // 束縛はどれも読むだけ。vertex と fragment のどちらからも見える。
  WGPUBindGroupLayoutEntry entries[65];
  for (int32_t i = 0; i <= nbufs; i++) {
    WGPUBindGroupLayoutEntry e = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    e.binding = (uint32_t)i;
    e.visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment;
    e.buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
    entries[i] = e;
  }
  WGPUBindGroupLayoutDescriptor ld = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
  ld.entryCount = (size_t)nbufs + 1;
  ld.entries = entries;
  p->layout = wgpuDeviceCreateBindGroupLayout(um_dev, &ld);
  WGPUPipelineLayoutDescriptor pld = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
  pld.bindGroupLayoutCount = 1;
  pld.bindGroupLayouts = &p->layout;
  WGPUPipelineLayout pl = wgpuDeviceCreatePipelineLayout(um_dev, &pld);

  WGPUColorTargetState target = WGPU_COLOR_TARGET_STATE_INIT;
  target.format = format;
  target.writeMask = WGPUColorWriteMask_All;
  WGPUFragmentState fragment = WGPU_FRAGMENT_STATE_INIT;
  fragment.module = module;
  fragment.entryPoint = um_sv("um_fs");
  fragment.targetCount = 1;
  fragment.targets = &target;
  WGPUDepthStencilState depth = WGPU_DEPTH_STENCIL_STATE_INIT;
  depth.format = WGPUTextureFormat_Depth32Float;
  depth.depthWriteEnabled = WGPUOptionalBool_True;
  depth.depthCompare = WGPUCompareFunction_Less;

  WGPURenderPipelineDescriptor rd = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
  rd.layout = pl;
  rd.vertex.module = module;
  rd.vertex.entryPoint = um_sv("um_vs");
  rd.primitive.topology = WGPUPrimitiveTopology_TriangleList;
  rd.primitive.cullMode = WGPUCullMode_None;
  rd.depthStencil = &depth;
  rd.multisample.count = 1;
  rd.multisample.mask = 0xFFFFFFFFu;
  rd.fragment = &fragment;
  p->pipeline = wgpuDeviceCreateRenderPipeline(um_dev, &rd);
  wgpuPipelineLayoutRelease(pl);
  wgpuShaderModuleRelease(module);
  if (!p->pipeline) um_fail0("creating the render pipeline");
  return p;
}

// 画像に描画のパスを1つ積む。pipe が NULL なら、埋めるだけ（何も描かない）。鍵を持って呼ぶ。
static void um_render_pass_locked(um_image *im, um_rpipe *pipe, WGPUBindGroup bg, uint32_t count) {
  WGPURenderPassColorAttachment ca = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
  ca.view = im->color_view;
  ca.loadOp = im->drawn ? WGPULoadOp_Load : WGPULoadOp_Clear;
  ca.storeOp = WGPUStoreOp_Store;
  ca.clearValue.r = ((im->clear >> 16) & 255) / 255.0;
  ca.clearValue.g = ((im->clear >> 8) & 255) / 255.0;
  ca.clearValue.b = (im->clear & 255) / 255.0;
  ca.clearValue.a = ((im->clear >> 24) & 255) / 255.0;
  WGPURenderPassDepthStencilAttachment da = WGPU_RENDER_PASS_DEPTH_STENCIL_ATTACHMENT_INIT;
  da.view = im->depth_view;
  da.depthLoadOp = im->drawn ? WGPULoadOp_Load : WGPULoadOp_Clear;
  da.depthStoreOp = WGPUStoreOp_Store;
  da.depthClearValue = 1.0f;
  WGPURenderPassDescriptor rp = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
  rp.colorAttachmentCount = 1;
  rp.colorAttachments = &ca;
  rp.depthStencilAttachment = &da;
  if (!um_enc) um_enc = wgpuDeviceCreateCommandEncoder(um_dev, NULL);
  WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(um_enc, &rp);
  if (pipe) {
    wgpuRenderPassEncoderSetPipeline(pass, pipe->pipeline);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, bg, 0, NULL);
    if (count > 0) wgpuRenderPassEncoderDraw(pass, count, 1, 0, 0);
  }
  wgpuRenderPassEncoderEnd(pass);
  wgpuRenderPassEncoderRelease(pass);
  im->drawn = 1;
}

void um_gpu_draw(int32_t image, int32_t key, const char *src, int32_t src_len, int32_t count, const void *block, int32_t block_bytes,
                 const uint64_t *table, int32_t nbufs) {
  um_lock();
  um_image *im = um_image_at(image, "gpu.draw");
  if (im->skip) {
    um_unlock();
    return;
  }
  um_rpipe *pipe = um_rpipeline_locked(key, src, src_len, nbufs, im->format);

  // Buffer は dispatch と同じく、CPU だけが新しければ GPU へ送る（描画は読むだけなので、GPU の写しは新しいまま）。
  WGPUBindGroupEntry entries[65];
  for (int32_t i = 0; i < nbufs; i++) {
    void *host = (void *)(uintptr_t)table[i * 4];
    um_region *r = um_region_at((uintptr_t)host);
    if (!r) um_render_panic("gpu.draw: a Buffer was not allocated for the GPU");
    if (!r->gpu_valid || !um_lazy) um_upload_locked(r);
    WGPUBindGroupEntry e = WGPU_BIND_GROUP_ENTRY_INIT;
    e.binding = (uint32_t)i + 1;
    e.buffer = r->gbuf;
    e.size = r->used;
    entries[i + 1] = e;
  }
  // 引数の塊: 頂点の数、0（row。um_z の元）、各 Buffer の長さ、値の引数（dispatch と同じ並び）。
  uint64_t header = (uint64_t)(2 + nbufs) * 4;
  uint64_t psize = (header + (uint64_t)block_bytes + 3) / 4 * 4;
  WGPUBuffer params = um_make_buffer(psize, WGPUBufferUsage_Storage, 1);
  unsigned char *pp = wgpuBufferGetMappedRange(params, 0, (size_t)psize);
  if (!pp) um_fail0("mapping the argument buffer");
  memset(pp, 0, (size_t)psize);
  uint32_t words[2] = {(uint32_t)count, 0};
  memcpy(pp, words, sizeof words);
  for (int32_t i = 0; i < nbufs; i++) {
    uint32_t len = (uint32_t)table[i * 4 + 1];
    memcpy(pp + 8 + 4 * i, &len, 4);
  }
  if (block_bytes > 0) memcpy(pp + header, block, (size_t)block_bytes);
  wgpuBufferUnmap(params);
  WGPUBindGroupEntry pe = WGPU_BIND_GROUP_ENTRY_INIT;
  pe.binding = 0;
  pe.buffer = params;
  pe.size = psize;
  entries[0] = pe;
  WGPUBindGroupDescriptor bd = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
  bd.layout = pipe->layout;
  bd.entryCount = (size_t)nbufs + 1;
  bd.entries = entries;
  WGPUBindGroup bg = wgpuDeviceCreateBindGroup(um_dev, &bd);

  um_render_pass_locked(im, pipe, bg, (uint32_t)count);
  wgpuBindGroupRelease(bg);
  wgpuBufferRelease(params);
  um_enc_count++;
  if (um_trace) fprintf(stderr, "wgpu: draw %d vertices\n", (int)count);
  um_unlock();
}

void um_gpu_image_end(int32_t image, uint32_t *pixels, size_t len) {
  um_lock();
  um_image *im = um_image_at(image, "gpu.endImage");
  if (im->window) um_render_panic("gpu.endImage: this is a frame of a window; end it with gpu.endFrame");
  uint64_t n = (uint64_t)im->width * im->height;
  if ((uint64_t)len < n) {
    char msg[200];
    snprintf(msg, sizeof msg, "gpu.endImage: the Buffer holds %llu elements but the image has %u x %u pixels",
             (unsigned long long)len, im->width, im->height);
    um_render_panic(msg);
  }
  if (!im->drawn) um_render_pass_locked(im, NULL, NULL, 0);

  // テクスチャから読み戻し用のバッファへ写す（1行のバイト数は 256 の倍数）。
  uint32_t row = (im->width * 4 + 255) / 256 * 256;
  uint64_t bytes = (uint64_t)row * im->height;
  WGPUBuffer staging = um_make_buffer(bytes, WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst, 0);
  if (!um_enc) um_enc = wgpuDeviceCreateCommandEncoder(um_dev, NULL);
  WGPUTexelCopyTextureInfo from = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
  from.texture = im->color;
  WGPUTexelCopyBufferInfo to = WGPU_TEXEL_COPY_BUFFER_INFO_INIT;
  to.buffer = staging;
  to.layout.bytesPerRow = row;
  to.layout.rowsPerImage = im->height;
  WGPUExtent3D size = {im->width, im->height, 1};
  wgpuCommandEncoderCopyTextureToBuffer(um_enc, &from, &to, &size);
  um_submit_locked();
  _Atomic int done = 0;
  WGPUBufferMapCallbackInfo ci = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
  ci.mode = WGPUCallbackMode_AllowProcessEvents;
  ci.callback = um_on_mapped;
  ci.userdata1 = (void *)&done;
  wgpuBufferMapAsync(staging, WGPUMapMode_Read, 0, (size_t)bytes, ci);
  um_wait_flag(&done);
  const unsigned char *src = wgpuBufferGetConstMappedRange(staging, 0, (size_t)bytes);
  if (!src) um_fail0("mapping a buffer");

  // pixels が GPU と写し合う領域なら、ランタイムの別名に書き、CPU の写しを新しいとする。
  um_region *r = um_region_at((uintptr_t)pixels);
  unsigned char *dst = (unsigned char *)pixels;
  if (r) {
    if (!r->host_valid) um_readback_locked(r);  // 画像より後ろの要素を保つ
    dst = r->alias;
  }
  for (uint32_t y = 0; y < im->height; y++) memcpy(dst + (size_t)y * im->width * 4, src + (size_t)y * row, (size_t)im->width * 4);
  wgpuBufferUnmap(staging);
  wgpuBufferRelease(staging);
  if (r) {
    r->host_valid = 1;
    r->gpu_valid = 0;
    um_reprotect(r);
  }

  wgpuTextureViewRelease(im->color_view);
  wgpuTextureViewRelease(im->depth_view);
  wgpuTextureRelease(im->color);
  wgpuTextureRelease(im->depth);
  im->used = 0;
  um_unlock();
}

// ---- ウィンドウに描く（gpu.beginFrame / gpu.endFrame）----
// ウィンドウごとに wgpu の surface を1つ持つ。初めてのフレームで、ウィンドウ（runtime/window.c の um_window_native）から
// OS の窓の情報をもらって作る。そのときウィンドウは SDL の描画の装置を手放す（以後 windowPresent は使えない）。
// GPU が使えなければ um_window_native を呼ばずに 0 を返し、フレームは CPU で描いて windowPresent で表示する（runtime/render_window.um）。

typedef struct {
  int32_t kind;
  void *a;
  void *b;
  uint64_t c;
} um_native_window;
typedef int32_t (*um_native_fn)(int32_t window, um_native_window *out, int32_t line);

#define UM_MAX_SURFACES 16
static struct {
  WGPUSurface surface;
  WGPUTextureFormat format;
  uint32_t width, height;  // 設定した大きさ（0 はまだ設定していない）
} um_surfaces[UM_MAX_SURFACES];

static void um_configure_locked(int slot, uint32_t width, uint32_t height) {
  WGPUSurfaceConfiguration c = WGPU_SURFACE_CONFIGURATION_INIT;
  c.device = um_dev;
  c.format = um_surfaces[slot].format;
  c.usage = WGPUTextureUsage_RenderAttachment;
  c.width = width;
  c.height = height;
  c.presentMode = WGPUPresentMode_Fifo;  // 画面の更新（vsync）に合わせる
  c.alphaMode = WGPUCompositeAlphaMode_Auto;
  wgpuSurfaceConfigure(um_surfaces[slot].surface, &c);
  um_surfaces[slot].width = width;
  um_surfaces[slot].height = height;
}

static WGPUSurface um_make_surface_locked(const um_native_window *nw) {
  WGPUSurfaceDescriptor d = WGPU_SURFACE_DESCRIPTOR_INIT;
  WGPUSurfaceSourceWindowsHWND hwnd = WGPU_SURFACE_SOURCE_WINDOWS_HWND_INIT;
  WGPUSurfaceSourceMetalLayer metal = WGPU_SURFACE_SOURCE_METAL_LAYER_INIT;
  WGPUSurfaceSourceXlibWindow xlib = WGPU_SURFACE_SOURCE_XLIB_WINDOW_INIT;
  WGPUSurfaceSourceWaylandSurface wl = WGPU_SURFACE_SOURCE_WAYLAND_SURFACE_INIT;
  switch (nw->kind) {
    case 1:
      hwnd.hwnd = nw->a;
      hwnd.hinstance = nw->b;
      d.nextInChain = &hwnd.chain;
      break;
    case 2:
      metal.layer = nw->a;
      d.nextInChain = &metal.chain;
      break;
    case 3:
      xlib.display = nw->a;
      xlib.window = nw->c;
      d.nextInChain = &xlib.chain;
      break;
    case 4:
      wl.display = nw->a;
      wl.surface = nw->b;
      d.nextInChain = &wl.chain;
      break;
    default:
      return NULL;
  }
  return wgpuInstanceCreateSurface(um_inst, &d);
}

int32_t um_gpu_frame_begin(int32_t window, um_native_fn native, int32_t width, int32_t height, int32_t clear, int32_t line) {
  if (width <= 0 || height <= 0 || window < 1 || window > UM_MAX_SURFACES) return 0;
  um_lock();
  if (!um_init_locked()) {
    um_unlock();
    return 0;
  }
  int slot = window - 1;
  if (!um_surfaces[slot].surface) {
    um_native_window nw;
    if (!native(window, &nw, line)) um_render_panic("gpu.beginFrame: could not get the window from the OS for the GPU");
    WGPUSurface surface = um_make_surface_locked(&nw);
    if (!surface) um_render_panic("gpu.beginFrame: could not create a GPU surface for the window");
    WGPUSurfaceCapabilities caps = WGPU_SURFACE_CAPABILITIES_INIT;
    if (wgpuSurfaceGetCapabilities(surface, um_adapter, &caps) != WGPUStatus_Success || caps.formatCount == 0) {
      um_render_panic("gpu.beginFrame: the GPU cannot draw to this window");
    }
    // 画像と同じ BGRA8Unorm（色をそのまま書く）を選ぶ。なければ RGBA8Unorm、それもなければ最初の形式。
    WGPUTextureFormat format = caps.formats[0];
    for (size_t i = 0; i < caps.formatCount; i++) {
      if (caps.formats[i] == WGPUTextureFormat_RGBA8Unorm) format = caps.formats[i];
    }
    for (size_t i = 0; i < caps.formatCount; i++) {
      if (caps.formats[i] == WGPUTextureFormat_BGRA8Unorm) format = caps.formats[i];
    }
    wgpuSurfaceCapabilitiesFreeMembers(caps);
    um_surfaces[slot].surface = surface;
    um_surfaces[slot].format = format;
    um_surfaces[slot].width = 0;
  }
  if (um_surfaces[slot].width != (uint32_t)width || um_surfaces[slot].height != (uint32_t)height) um_configure_locked(slot, (uint32_t)width, (uint32_t)height);

  size_t i = um_image_slot_locked();
  um_image *im = &um_images[i];
  memset(im, 0, sizeof *im);
  im->used = 1;
  im->width = (uint32_t)width;
  im->height = (uint32_t)height;
  im->clear = (uint32_t)clear;
  im->window = window;
  im->format = um_surfaces[slot].format;
  // 前に積んだ dispatch を先に提出しておく（面を待つあいだに GPU が進められるように）。
  um_submit_locked();
  WGPUSurfaceTexture st = WGPU_SURFACE_TEXTURE_INIT;
  for (int attempt = 0; attempt < 2; attempt++) {
    wgpuSurfaceGetCurrentTexture(um_surfaces[slot].surface, &st);
    if (st.status == WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal || st.status == WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal) break;
    if (st.texture) wgpuTextureRelease(st.texture);
    st.texture = NULL;
    // 大きさが変わった・面を失ったなど。設定し直してもう一度だけ試す。
    um_configure_locked(slot, (uint32_t)width, (uint32_t)height);
  }
  if (!st.texture) {
    im->skip = 1;  // このフレームは描かずに捨てる（最小化の途中など）
    um_unlock();
    return (int32_t)i + 1;
  }
  im->color = st.texture;
  im->color_view = wgpuTextureCreateView(im->color, NULL);
  im->depth = um_texture(im->width, im->height, WGPUTextureFormat_Depth32Float, WGPUTextureUsage_RenderAttachment);
  im->depth_view = wgpuTextureCreateView(im->depth, NULL);
  um_unlock();
  return (int32_t)i + 1;
}

void um_gpu_frame_end(int32_t frame) {
  um_lock();
  um_image *im = um_image_at(frame, "gpu.endFrame");
  if (!im->window) um_render_panic("gpu.endFrame: this is an image; end it with gpu.endImage");
  if (!im->skip) {
    if (!im->drawn) um_render_pass_locked(im, NULL, NULL, 0);
    um_submit_locked();
    wgpuSurfacePresent(um_surfaces[im->window - 1].surface);
    wgpuTextureViewRelease(im->color_view);
    wgpuTextureViewRelease(im->depth_view);
    wgpuTextureRelease(im->color);
    wgpuTextureRelease(im->depth);
  }
  im->used = 0;
  um_unlock();
}
