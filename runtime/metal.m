// UnifiedMozaic の Metal ランタイム（macOS）。kernel を GPU で実行する。
//
// 試験中: コンパイラは UM_METAL=1 のときだけこのファイルをリンクする。
//
// Buffer はコピーせずに GPU と共有する（ユニファイドメモリ）。Buffer の領域はページ境界に揃えて
// 確保してあるので、そのまま GPU のバッファとして渡せる。UM_GPU_COPY=1 を付けると、比較のために
// 「コピーして実行、コピーして戻す」方式に切り替わる。
//
// 引数の並びは wasm 版（runtime/host.mjs）と同じ:
//   table: Buffer 1つにつき 64 ビットの語が4つ（データの場所、長さ、1要素のバイト数、書き込むかどうか）。
//   block: Buffer 以外の引数を、コンパイラが 32 ビットの語で詰めたもの。
// シェーダ側の引数の塊（UmParams）は、個数、1行の要素数、各 Buffer の長さ、block の順。

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach_time.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// 1行に並べる要素の数。um_main は gid.y * row + gid.x を添字にする。
#define UM_ROW 16384u

static id<MTLDevice> um_device;
static id<MTLCommandQueue> um_queue;
static NSMutableDictionary<NSNumber *, id<MTLComputePipelineState>> *um_pipelines;
// UM_GPU_THRESHOLD を付けると、実測による判断をやめて「要素数がこの数以上なら GPU」という固定の規則になる。
static uint32_t um_threshold = 0;
static int um_fixed_rule = 0;
// 実測による判断で、GPU を候補にする最小の仕事の大きさ（CPU で実行したときの秒数）。
// GPU は1回渡すだけで 0.1 ms ほどかかるので、CPU でそれより十分に短く終わる仕事は GPU を試しもしない
// （試すと、その kernel のシェーダのコンパイルも走る）。要素数ではなく時間で見るのは、
// 「少ない個数の重い仕事」（探索の末端の評価など）を GPU に回せるようにするため。
#define UM_GPU_MIN_SECONDS 50e-6

// 自動判断のための実測値（kernel ごと）。時間は秒。
//   CPU: 1要素あたりの時間。
//   GPU: 要素数によらない手間（命令を出して終わるのを待つ時間）と、1要素あたりの時間。
// どちらも最初の2回は小さいほうを取り（初回は準備の時間が混ざるため）、そのあとはならして更新する。
#define UM_MAX_TRACKED 256
#define UM_REPROBE 64  // 選ばれなかった側を、この回数に1回だけ試して測り直す（差が大きいほど間隔を広げる）
typedef struct {
  uint32_t cpu_runs, gpu_runs, since_probe;
  uint32_t backoff;   // 測り直しても勝ち負けが変わらなかった回数（そのたびに測り直しの間隔を倍にする）
  int last_winner;    // 前回の判断で速いと見込んだ側（0 = CPU、1 = GPU）
  double cpu_per_elem, gpu_fixed, gpu_per_elem;
} um_stat;
static um_stat um_stats[UM_MAX_TRACKED];
// CPU で実行中の kernel（時間を測っている）。スレッドごと（thread.spawn のスレッドが別々に dispatch する）。
static _Thread_local int32_t um_pending_kernel = -1;
static _Thread_local CFAbsoluteTime um_pending_start;

// ---- 複数のスレッドからの呼び出し（thread.spawn）----
// 鍵は4つ。取る順番は um_park_lock → um_gpu_lock → um_async_lock で、um_table_lock は最後（ほかの鍵を取らない）。
//   um_park_lock:  常駐の枠（um_slot など）。常駐の経路は GPU の結果を待つあいだも持つので、ほかのスレッドが
//                  使っている最中の gpu.dispatch は待たずに普通に提出する（um_park_try）。
//   um_gpu_lock:   gpu.submit の券の表（um_async）と GPU 側のバッファの控え（um_async_wrap）。命令を作るあいだだけ持ち、
//                  GPU を待つあいだ（gpu.wait の見張り、完了待ち、検算の計算）は放す。
//                  急ぎの列（um_queue_urgent）を作るのと、書いた領域の控え（um_async_lastwrite）もこの鍵で守る。
//   um_async_lock: ためている命令（um_open_cmd など、列ごと）。Metal の完了の処理（別のスレッド）からも取る。
//                  完了の印（um_async_fin）は完了の処理がアトミックに書くだけで、鍵は使わない。
//   um_table_lock: パイプラインの表と自動判断の実測値（um_stats）。
//   um_ap_lock:    gpu.submit の常駐の枠ごとの鍵（um_ap。待機分の補充）。um_gpu_lock のあと、um_async_lock の前に取る。
// まだ提出していない dispatch（um_pending）は文の並びの話なので、スレッドごとに持つ。
static pthread_mutex_t um_table_lock = PTHREAD_MUTEX_INITIALIZER;
// GPU が覚えている領域があるか（gpu.submit の控えの数 + 常駐の枠の数）。0 なら um_on_free は鍵を取らずに戻る。
static _Atomic int um_gpu_regions;
typedef void (*um_free_fn)(void *);
extern _Atomic(um_free_fn) um_free_hook;
extern _Noreturn void um_exit_failure(void);
static void um_on_free(void *p);
static void um_ap_init_locks(void);
static void um_ap_release(void *p);
static void um_ap_idle(void);
static void um_chains_flush_all(void);

static void um_blend(double *slot, double sample, uint32_t runs) {
  if (runs == 0) *slot = sample;
  else if (runs == 1) *slot = sample < *slot ? sample : *slot;
  else *slot = *slot * 0.75 + sample * 0.25;
}
static int um_zero_copy = 1;  // UM_GPU_COPY=1 で、コピーなしの共有をやめて毎回コピーする（比較用）
static int um_check_enabled = 1;  // UM_GPU_RESIDENT=status で、検算による完了の確認を使わない（um_check を参照）
static int um_resident = 1;  // UM_GPU_RESIDENT=0 で、待機 kernel（常駐）の経路を使わない
static int um_trace = 0;  // UM_GPU_TRACE=1 で、dispatch ごとに共有・コピーした Buffer の数と時間を stderr に出す
static int um_ready = 0;  // 0 = まだ調べていない、1 = 使える、-1 = 使えない

// GPUStartTime / GPUEndTime と同じ基準の現在時刻（秒）。
static double um_gpu_clock(void) {
  static mach_timebase_info_data_t tb;
  if (tb.denom == 0) mach_timebase_info(&tb);
  return (double)mach_absolute_time() * tb.numer / tb.denom / 1e9;
}

static void um_metal_fail(const char *what, NSError *error) {
  fprintf(stderr, "panic: the GPU could not run a kernel (%s)%s%s\n", what, error ? ": " : "",
          error ? [[error localizedDescription] UTF8String] : "");
  um_exit_failure();
}

// ---- gpu.submit の印（gpu.flags）----
// 呼んだスレッドで、このあとの gpu.submit に付ける印。0（既定）なら今までと同じ動き。
//   UM_SUBMIT_NOCHECK: 検算をしない（検算の kernel を積まない）。完了は Metal の status と、完了の印（um_async_fin）で知る。
//                      結果の語の中に世代（札）を書く kernel なら、呼ぶ側が札を見て完了を判断できる（README の gpu.submit）。
//   UM_SUBMIT_URGENT:  急ぎの列（2 本目の MTLCommandQueue）に積む。UM_INFLIGHT_MAX でためず、HOLD でなければすぐ提出する。
//   UM_SUBMIT_MAILBOX: 検算なしの submit を gpu.submit の常駐（待機命令）に乗せ（急ぎの列でなくても）、合図で待たせる代わりに
//                      郵便受けの列で待たせる（「郵便受けの列」。動き出すまでの 60〜100 µs がなくなる）。UM_GPU_MAILBOX=1 で、
//                      印がなくても常駐に乗る並びの全部に使う。UM_GPU_MAILBOX=0 で、印があっても使わない。
//   UM_SUBMIT_HOLD:    積むが提出しない。同じ列への次の HOLD なしの submit、gpu.commit()、その券の ready / wait、
//                      ためた数が UM_OPEN_MAX に達したとき、のどれかで提出される。
#define UM_SUBMIT_NOCHECK 1u
#define UM_SUBMIT_URGENT 2u
#define UM_SUBMIT_HOLD 4u
#define UM_SUBMIT_MAILBOX 8u  // 常駐の経路に乗るなら、郵便受けの列を使う（「郵便受けの列」。並びのどれか 1 つに付いていればよい）
static _Thread_local uint32_t um_submit_flags;
static int um_force_check = -1;  // UM_GPU_CHECK: -1 = 印に従う（既定）、0 = 必ず検算しない、1 = 必ず検算する
static int um_allow_urgent = 1;  // UM_GPU_URGENT=0 で URGENT を無視する（列は 1 本）
static int um_allow_hold = 1;    // UM_GPU_HOLD=0 で HOLD を無視する（毎回今までどおり提出を判断する）
// gpu.submit の常駐（待機命令）の経路を使う submit（「gpu.submit の常駐」）。UM_GPU_ASYNC_RESIDENT で選ぶ:
//   1（既定）= 検算なしで急ぎの列（URGENT）の submit、all = 検算なしの submit すべて、0 = 使わない（今までの経路）。
static int um_ap_enabled = 1;
// UM_GPU_MAILBOX=1 で、gpu.submit の常駐の待機命令が、合図（MTLSharedEvent）の代わりに郵便受けを見る（「郵便受けの列」を参照）。
static int um_mb_enabled = 0;
static int um_mb_off = 0;       // 郵便受けを使わない（UM_GPU_MAILBOX=0、または空振りが多くてやめた。um_mq_disable）
static int um_mb_signal = 0;    // UM_GPU_MAILBOX_SIGNAL=1 で、待機命令の最後に合図を出す操作を入れる（「郵便受けの列」。既定は入れない）
static int um_mb_asked = 0;     // UM_GPU_MAILBOX=1 が付いていた（途中でやめても、計測の行は出す）
static int um_mq_on;            // 郵便受けの経路を使っている（「郵便受けの列」）
static void um_mq_kick(void);   // 郵便受けの列の見張りをどかす（ほかの列に命令を出したあとで呼ぶ）
static void um_mq_judge(uint64_t skipped, int cold);

// ---- gpu.submit の計測（UM_GPU_STATS=1 で、終わるときに stderr に出す）----
static int um_gs_on = 0;
static _Atomic uint64_t um_gs_submit[2];   // 列ごとの submit の数
static _Atomic uint64_t um_gs_nocheck;     // NOCHECK で検算を積まなかった数
static _Atomic uint64_t um_gs_hold;        // HOLD で提出を見送った数
static _Atomic uint64_t um_gs_by_check;    // ready / wait が検算の一致で完了とした数
static _Atomic uint64_t um_gs_by_status;   // ready / wait が Metal の status で完了とした数
static _Atomic uint64_t um_gs_by_fin;      // ready / wait が完了の印（um_async_fin）で済ませた数
static _Atomic uint64_t um_gs_blocking;    // waitUntilCompleted まで待った数
static _Atomic uint64_t um_gs_blocking_us; // その合計（マイクロ秒）
static _Atomic uint64_t um_gs_take_waits;  // 券の表が満杯で、いちばん古い提出を待った数
static _Atomic uint64_t um_gs_reclaimed;   // 券の表が満杯のとき、完了の印で回収した枠の数
static _Atomic uint64_t um_gs_cross_waits; // 別の列の書き込みを待った数（um_async_lastwrite）
static _Atomic uint64_t um_gs_ap_jobs;     // gpu.submit の常駐: 待機命令で走らせた submit の数
static _Atomic uint64_t um_gs_ap_fires;    // 同じく、合図を出した回数（HOLD でつないだ並び 1 つで 1 回）
static _Atomic uint64_t um_gs_ap_normal;   // 同じく、対象になれたが形が待機していなかったので普通に積んだ submit の数
static _Atomic uint64_t um_gs_ap_starts;   // 同じく、待機を始めた形の数
static _Atomic uint64_t um_gs_ap_stops;    // 同じく、待機をやめた数（解放、使われない、など）
static _Atomic uint64_t um_gs_mq_skips;    // 郵便受けの列: 形が合わずに空振りさせた待機分の数
static _Atomic uint64_t um_gs_mq_flushes;  // 同じく、待機分をまとめて空振りさせた回数（休み、寿命、形をやめる）
static _Atomic uint64_t um_gs_mq_cold;     // 同じく、依頼のときにその形の待機分がなく、その場で出した回数
static _Atomic uint64_t um_gs_mq_kicks;    // 同じく、ほかの列に命令を出したので待機分を空振りさせた回数
static _Atomic uint64_t um_gs_gpu_cmds;   // 非同期の命令のうち、GPU の実行時間を足した数（列の命令と待機命令）
static _Atomic uint64_t um_gs_gpu_ns;     // その GPU の実行時間（GPUEndTime - GPUStartTime）の合計（ナノ秒）
static _Atomic uint64_t um_gs_gpu_first;  // 最初の命令の GPUStartTime（ナノ秒。0 ならまだない）
static _Atomic uint64_t um_gs_gpu_last;   // 最後の命令の GPUEndTime（ナノ秒）
static void um_gs_add(_Atomic uint64_t *c, uint64_t v) {
  if (um_gs_on) atomic_fetch_add_explicit(c, v, memory_order_relaxed);
}
// 終わった命令の GPU の実行時間を足す（UM_GPU_STATS のときだけ）。完了の処理から呼ぶ。
static void um_gs_gpu_time(id<MTLCommandBuffer> done) {
  if (!um_gs_on) return;
  double a = [done GPUStartTime], b = [done GPUEndTime];
  if (!(b > a) || a <= 0) return;
  uint64_t na = (uint64_t)(a * 1e9), nb = (uint64_t)(b * 1e9);
  atomic_fetch_add_explicit(&um_gs_gpu_cmds, 1, memory_order_relaxed);
  atomic_fetch_add_explicit(&um_gs_gpu_ns, nb - na, memory_order_relaxed);
  uint64_t z = 0;
  atomic_compare_exchange_strong(&um_gs_gpu_first, &z, na);
  uint64_t cur = atomic_load(&um_gs_gpu_last);
  while (nb > cur && !atomic_compare_exchange_weak(&um_gs_gpu_last, &cur, nb)) {}
}
// gpu.submit の常駐の往復の内訳（UM_GPU_STATS のときだけ。完了の処理から呼ぶ）: 合図 → GPU が始めた、GPU の実行、
// GPU が終えた → 完了の処理が走った（Metal の完了の知らせの遅れ）。どれもナノ秒の合計。始めた時刻が合図より前のとき
// （合図を待つあいだも GPU の時間に数えられた）は数を別にする。
static _Atomic uint64_t um_gs_lat_n, um_gs_lat_pre, um_gs_lat_start_ns, um_gs_lat_run_ns, um_gs_lat_note_ns;
static _Atomic uint64_t um_gs_lat_hist[8];  // 合図 → GPU が始めた: <10, <20, <40, <80, <160, <320, <640, ≥640 µs
static _Atomic uint64_t um_gs_chain_n, um_gs_chain_ns;  // 並びの最初の submit から合図まで（um_chain_run）
static void um_gs_ap_lat(id<MTLCommandBuffer> done, double fire) {
  if (!um_gs_on || fire <= 0) return;
  double a = [done GPUStartTime], b = [done GPUEndTime], now = um_gpu_clock();
  if (!(b > a) || a <= 0) return;
  atomic_fetch_add_explicit(&um_gs_lat_n, 1, memory_order_relaxed);
  if (a < fire) {
    atomic_fetch_add_explicit(&um_gs_lat_pre, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&um_gs_lat_run_ns, (uint64_t)((b - fire) * 1e9), memory_order_relaxed);
  } else {
    atomic_fetch_add_explicit(&um_gs_lat_start_ns, (uint64_t)((a - fire) * 1e9), memory_order_relaxed);
    atomic_fetch_add_explicit(&um_gs_lat_run_ns, (uint64_t)((b - a) * 1e9), memory_order_relaxed);
    double us = (a - fire) * 1e6, lim = 10;
    int h = 0;
    while (h < 7 && us >= lim) { h++; lim *= 2; }
    atomic_fetch_add_explicit(&um_gs_lat_hist[h], 1, memory_order_relaxed);
  }
  if (now > b) atomic_fetch_add_explicit(&um_gs_lat_note_ns, (uint64_t)((now - b) * 1e9), memory_order_relaxed);
}
static void um_gs_print(void) {
#define UM_GS(x) (unsigned long long)atomic_load(&(x))
  {
    uint64_t n = atomic_load(&um_gs_lat_n), pre = atomic_load(&um_gs_lat_pre);
    if (n > 0) {
      uint64_t cn = atomic_load(&um_gs_chain_n);
      fprintf(stderr, "gpu ap chain: first_submit->fire_us %.1f\n", cn ? atomic_load(&um_gs_chain_ns) / 1e3 / cn : 0.0);
      fprintf(stderr, "gpu ap latency: fires %llu start_before_fire %llu | fire->start_us %.1f run_us %.1f end->handler_us %.1f | start hist",
              (unsigned long long)n, (unsigned long long)pre, n > pre ? atomic_load(&um_gs_lat_start_ns) / 1e3 / (n - pre) : 0.0,
              atomic_load(&um_gs_lat_run_ns) / 1e3 / n, atomic_load(&um_gs_lat_note_ns) / 1e3 / n);
      for (int h = 0; h < 8; h++) fprintf(stderr, " %llu", UM_GS(um_gs_lat_hist[h]));
      fprintf(stderr, "\n");
    }
  }
  {
    uint64_t n = atomic_load(&um_gs_gpu_cmds), ns = atomic_load(&um_gs_gpu_ns);
    uint64_t span = atomic_load(&um_gs_gpu_last) - atomic_load(&um_gs_gpu_first);
    fprintf(stderr, "gpu time: cmds %llu busy_ms %.1f span_ms %.1f busy_per_cmd_us %.1f busy_ratio %.3f\n",
            (unsigned long long)n, ns / 1e6, span / 1e6, n ? ns / 1e3 / n : 0.0, span ? (double)ns / span : 0.0);
  }
  fprintf(stderr,
          "gpu stats: submit normal %llu urgent %llu nocheck %llu hold %llu | done by check %llu status %llu fin %llu | "
          "blocking waits %llu (%llu us) | table waits %llu reclaimed %llu | cross-queue waits %llu | "
          "resident submits %llu fires %llu (eligible but normal %llu) starts %llu stops %llu\n",
          UM_GS(um_gs_submit[0]), UM_GS(um_gs_submit[1]), UM_GS(um_gs_nocheck), UM_GS(um_gs_hold), UM_GS(um_gs_by_check),
          UM_GS(um_gs_by_status), UM_GS(um_gs_by_fin), UM_GS(um_gs_blocking), UM_GS(um_gs_blocking_us),
          UM_GS(um_gs_take_waits), UM_GS(um_gs_reclaimed), UM_GS(um_gs_cross_waits), UM_GS(um_gs_ap_jobs),
          UM_GS(um_gs_ap_fires), UM_GS(um_gs_ap_normal), UM_GS(um_gs_ap_starts), UM_GS(um_gs_ap_stops));
  if (um_mb_asked || um_mq_on || um_mb_off) {
    fprintf(stderr, "gpu mailbox: skipped %llu flushes %llu (kicked by other queues %llu) fires without a parked command %llu%s\n",
            UM_GS(um_gs_mq_skips), UM_GS(um_gs_mq_flushes), UM_GS(um_gs_mq_kicks), UM_GS(um_gs_mq_cold),
            um_mb_off ? " | turned off" : "");
  }
#undef UM_GS
}

// 環境変数を読む（1回だけ）。GPU の準備（um_metal_init）とは分けてある。CPU だけで済むプログラムでは、
// GPU の準備（数十ミリ秒かかる）をしないで済ませるため。
static pthread_once_t um_env_once = PTHREAD_ONCE_INIT;
static int um_gpu_off = 0;  // UM_GPU=0 で GPU を使わない
#define UM_AWAKE_PIPE_MAX 8
static int um_awake_pipe = 0;  // UM_GPU_AWAKE_PIPE（um_awake_main）
static int um_awake_block = 0;  // UM_GPU_AWAKE_BLOCK=1（実験用）: 郵便受けのあいだも、起こしておく kernel の完了を待ち続ける
static int um_awake_quiet = 0;  // UM_GPU_AWAKE_QUIET=1（実験用）: GPU を起こしておく kernel を出さない（待機の片付けだけする）
static long um_awake_len = -1;  // UM_GPU_AWAKE_LEN: GPU を起こしておく kernel の長さを固定する（-1 = 形ごとに選ぶ）
static void um_env_init_once(void);
static void um_env_init(void) { pthread_once(&um_env_once, um_env_init_once); }
static void um_env_init_once(void) {
  const char *off = getenv("UM_GPU");
  if (off && strcmp(off, "0") == 0) um_gpu_off = 1;
  const char *awl = getenv("UM_GPU_AWAKE_LEN");
  if (awl) um_awake_len = atol(awl);
  const char *awb = getenv("UM_GPU_AWAKE_BLOCK");
  if (awb && strcmp(awb, "1") == 0) um_awake_block = 1;
  const char *awq = getenv("UM_GPU_AWAKE_QUIET");
  if (awq && strcmp(awq, "1") == 0) um_awake_quiet = 1;
  const char *awp = getenv("UM_GPU_AWAKE_PIPE");
  if (awp) {
    um_awake_pipe = atoi(awp);
    if (um_awake_pipe < 0) um_awake_pipe = 0;
    if (um_awake_pipe > UM_AWAKE_PIPE_MAX) um_awake_pipe = UM_AWAKE_PIPE_MAX;
  }
  const char *t = getenv("UM_GPU_THRESHOLD");
  if (t) {
    char *end;
    long v = strtol(t, &end, 10);
    if (end != t && *end == 0 && v >= 0) {
      um_threshold = (uint32_t)v;
      um_fixed_rule = 1;
    }
  }
  const char *copy = getenv("UM_GPU_COPY");
  if (copy && strcmp(copy, "1") == 0) um_zero_copy = 0;
  const char *resident = getenv("UM_GPU_RESIDENT");
  if (resident && strcmp(resident, "0") == 0) um_resident = 0;
  if (resident && strcmp(resident, "status") == 0) um_check_enabled = 0;
  const char *trace = getenv("UM_GPU_TRACE");
  if (trace && strcmp(trace, "1") == 0) um_trace = 1;
  // gpu.submit の印（gpu.flags）を環境変数で上書きする。um_gpu_submit_inner の「実際に使う印」を参照。
  const char *check = getenv("UM_GPU_CHECK");
  if (check && strcmp(check, "0") == 0) um_force_check = 0;
  if (check && strcmp(check, "1") == 0) um_force_check = 1;
  const char *urgent = getenv("UM_GPU_URGENT");
  if (urgent && strcmp(urgent, "0") == 0) um_allow_urgent = 0;
  const char *hold = getenv("UM_GPU_HOLD");
  if (hold && strcmp(hold, "0") == 0) um_allow_hold = 0;
  const char *ap = getenv("UM_GPU_ASYNC_RESIDENT");
  if (ap && strcmp(ap, "0") == 0) um_ap_enabled = 0;
  if (ap && strcmp(ap, "all") == 0) um_ap_enabled = 2;
  const char *mb = getenv("UM_GPU_MAILBOX");
  if (mb && strcmp(mb, "1") == 0) um_mb_enabled = um_mb_asked = 1;
  if (mb && strcmp(mb, "0") == 0) um_mb_off = 1;
  const char *mbs = getenv("UM_GPU_MAILBOX_SIGNAL");
  if (mbs && strcmp(mbs, "1") == 0) um_mb_signal = 1;
  const char *stats = getenv("UM_GPU_STATS");
  if (stats && strcmp(stats, "1") == 0) {
    um_gs_on = 1;
    atexit(um_gs_print);
  }
}

// 準備は1回だけ（どのスレッドが先に呼んでもよい）。解放の呼び出し口（um_free_hook）もここで設定する。
static pthread_once_t um_init_once = PTHREAD_ONCE_INIT;
static void um_metal_init_once(void) {
  um_ready = -1;
  um_env_init();
  if (um_gpu_off) return;
  um_device = MTLCreateSystemDefaultDevice();
  if (!um_device) return;
  um_queue = [um_device newCommandQueue];
  if (!um_queue) return;
  um_pipelines = [NSMutableDictionary dictionary];
  um_ap_init_locks();
  atomic_store_explicit(&um_free_hook, um_on_free, memory_order_release);
  um_ready = 1;
}
static int um_metal_init(void) {
  pthread_once(&um_init_once, um_metal_init_once);
  return um_ready > 0;
}

int32_t um_gpu_available(void) { return um_metal_init(); }

// まだ提出していない dispatch（um_gpu_dispatch が積み、um_gpu_flush がまとめて提出して完了を待つ）。
// 生成コードは、続けて書かれた dispatch 文のあいだでは um_gpu_flush を呼ばない。その分が1回の提出にまとまり、
// 「命令を出して終わるのを待つ」固定の手間が1回で済む。
// 積むときは引数を写しておくだけで、GPU への命令は um_gpu_flush で作る。そこで、同じ並びが待機していれば
// そちら（常駐）へ渡し、なければ普通に提出する。
#define UM_MAX_BATCH 64
#define UM_BATCH_BUFS 16    // 積んでおける dispatch の Buffer の数（これより多い dispatch は、積まずにすぐ実行する）
#define UM_BATCH_BLOCK 512  // 同じく、Buffer 以外の引数のバイト数
typedef struct {
  int32_t kernel, src_len, nbufs, block_bytes;
  const char *src;
  uint32_t count;
  uint32_t group;  // dispatchGroups のグループの大きさ（0 は普通の dispatch）
  uint64_t table[UM_BATCH_BUFS * 4];
  unsigned char block[UM_BATCH_BLOCK];
} um_job;
static _Thread_local um_job um_pending[UM_MAX_BATCH];
static _Thread_local int um_pending_len;

// ---- グループ（dispatchGroups / gpu.submitGroups）----
// 生成コードは um_gpu_dispatch / um_gpu_submit の直前に um_gpu_group でグループの大きさを渡す。
// 次の um_gpu_dispatch / um_gpu_submit がそれを使って 0 に戻す（スレッドごと）。
// グループの dispatch では、count は全体のスレッド数（グループの数 × 大きさ）。シェーダ（src/msl.ts の groups の形）には、
// 個数としてグループの数、1行としてグループの数を渡し、dispatchThreadgroups で大きさちょうどのグループを並べる。
static _Thread_local uint32_t um_next_group;
void um_gpu_group(int32_t size) { um_next_group = (uint32_t)size; }
static uint32_t um_take_group(void) {
  uint32_t g = um_next_group;
  um_next_group = 0;
  return g;
}

// 格子の形。n は、普通の dispatch ではスレッド数、グループの dispatch ではグループの数。
typedef struct {
  uint32_t n, row, rows, group;
} um_grid;
static um_grid um_grid_of(uint32_t count, uint32_t group) {
  um_grid g;
  g.group = group;
  g.n = group ? count / group : count;
  g.row = g.n < UM_ROW ? g.n : UM_ROW;
  g.rows = (g.n + g.row - 1) / g.row;
  return g;
}
static void um_encode_grid(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline, um_grid g) {
  if (g.group) {
    if (g.group > [pipeline maxTotalThreadsPerThreadgroup]) {
      fprintf(stderr, "gpu: group size %u, but this kernel allows at most %lu threads per group on this GPU\n", g.group,
              (unsigned long)[pipeline maxTotalThreadsPerThreadgroup]);
      um_metal_fail("the group size is too large for this kernel", nil);
    }
    [enc dispatchThreadgroups:MTLSizeMake(g.row, g.rows, 1) threadsPerThreadgroup:MTLSizeMake(g.group, 1, 1)];
    return;
  }
  NSUInteger width = [pipeline threadExecutionWidth];
  if (width > g.row) width = g.row;
  [enc dispatchThreads:MTLSizeMake(g.row, g.rows, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
}

void um_gpu_flush(void);
static id<MTLComputePipelineState> um_pipeline(int32_t kernel, const char *src, int32_t src_len);
static id<MTLComputePipelineState> um_pipeline_bl(int32_t kernel, const char *src, int32_t src_len);

// 常駐の経路（待機命令。dispatchThreadgroupsWithIndirectBuffer）の格子。待機命令の threadsPerThreadgroup は形を
// 決めたときに固定した width（普通の dispatch は threadExecutionWidth、グループの dispatch はグループの大きさ）で、
// 個数は合図の前に引数（words[0] = count、words[1] = row）と間接 dispatch の格子（groups[0..2]）に書く。
//   普通の dispatch: 1行を width の倍数に切り上げ、格子は (row / width, rows) 個のまとまり。count はスレッド数。
//   グループの dispatch: um_grid_of と同じ（count / row はグループの数で数える。src/msl.ts の groups の形）。
//     格子は (row, rows) 個のグループで、大きさはちょうどグループの大きさ。普通の提出（um_encode_grid）と同じ形。
static void um_resident_grid(uint32_t count, uint32_t group, NSUInteger width, uint32_t *words, uint32_t *groups) {
  if (group) {
    um_grid g = um_grid_of(count, group);
    words[0] = g.n;
    words[1] = g.row;
    groups[0] = g.row;
    groups[1] = g.rows;
    groups[2] = 1;
    return;
  }
  uint32_t row = count < UM_ROW ? (uint32_t)((count + width - 1) / width * width) : UM_ROW;
  uint32_t rows = (count + row - 1) / row;
  words[0] = count;
  words[1] = row;
  groups[0] = row / (uint32_t)width;
  groups[1] = rows;
  groups[2] = 1;
}

// 待機させる形の1つの dispatch の threadsPerThreadgroup を決める。使えなければ 0。
// 普通の dispatch は、1行の幅（UM_ROW）をスレッドのまとまりで割り切れないと個数を実行時に決められない。
// グループの dispatch は、グループの大きさがこの kernel の上限を超えるなら使わない（普通の経路が止める）。
static NSUInteger um_resident_width(id<MTLComputePipelineState> pipeline, uint32_t group) {
  if (!pipeline) return 0;
  if (group) return group <= [pipeline maxTotalThreadsPerThreadgroup] ? group : 0;
  NSUInteger width = [pipeline threadExecutionWidth];
  if (width == 0 || UM_ROW % width != 0) return 0;
  return width;
}

// ---- 常駐（待機 kernel）----
// 同じ kernel（または同じ並びの kernel）を同じ Buffer の組で何度も呼ぶ使い方のための経路。
// GPU の上で回り続ける kernel は CPU の書き込みを読めない（値が新しくなるのは GPU の仕事の区切りだけ）ので、
// 代わりに「先に提出して、合図を待たせておく」。
//   - 「形」は、1回の提出にまとまる dispatch の並び（続けて書かれた dispatch 文。1つだけでもよい）と、
//     それぞれの kernel・Buffer の組。同じ形を UM_PARK_AFTER 回見たら、その形の命令を UM_PARK_AHEAD 個、
//     合図待ちで提出しておく。形は UM_PARK_SLOTS 個まで同時に待機させておける。
//   - 次に同じ形の dispatch が来たら、引数と個数を共有メモリに書いて合図（MTLSharedEvent）を進めるだけ。
//     待機していた kernel は走り出すときに引数と個数を読む（個数は間接 dispatch）。
//   - 合図を出してから GPU が動き出すまでを短くするため、使っているあいだは軽い kernel を回し続けて
//     GPU を起こしたままにする。しばらく呼ばれなければ止める。
//   - 形が変わったとき、関係する Buffer が解放されるときは、待機分を空振りさせて（個数 0）片付ける。
// UM_GPU_RESIDENT=0 でこの経路を使わない。
#define UM_PARK_AFTER 8
#define UM_PARK_AHEAD 32
#define UM_PARK_BUFS UM_BATCH_BUFS
#define UM_PARK_SEQ 8    // 1つの形に含められる dispatch の数
#define UM_PARK_SLOTS 4  // 同時に待機させておける形の数
// 形ごとの枠。state: 0 = 空き、1 = 回数を数えている、2 = 待機中。
static struct {
  int state;
  int hits;
  uint64_t last_use;
  int n;  // 並んでいる dispatch の数
  struct {
    int32_t kernel, src_len, nbufs, block_bytes;
    const char *src;
    uint64_t sig[UM_PARK_BUFS * 4];  // Buffer ごとの（場所、長さ、要素の大きさ、書き込むか）
    uint32_t group;                  // グループの大きさ（0 は普通の dispatch。形の一部）
    NSUInteger width;                // 1 つのスレッドのまとまり（threadsPerThreadgroup）の大きさ。グループなら group
  } d[UM_PARK_SEQ];
  uint64_t committed, fired;  // 合図の値（枠ごとに増え続ける）
  // GPU を起こしておく kernel の長さの選択（um_awake_lengths の番号）。runs は待機を始めてからの実行回数。
  uint32_t runs;
  CFAbsoluteTime last_fire;  // 最後に合図を出した（または待機を始めた）時刻。um_park_idle を参照
  int awake;
  double trial_sum[2];
  uint32_t trial_n[2];
} um_slot[UM_PARK_SLOTS];
// 枠ごとの Metal の物。合図（共有イベント）と待ち行列は枠ごとに分ける。1つの合図を共有すると、
// ある形に合図を出したときに、ほかの形の待機分まで走り出してしまう。
static id<MTLCommandQueue> um_slot_queue[UM_PARK_SLOTS];
static id<MTLSharedEvent> um_slot_event[UM_PARK_SLOTS];
static id<MTLComputePipelineState> um_slot_pipeline[UM_PARK_SLOTS][UM_PARK_SEQ];
static id<MTLBuffer> um_slot_params[UM_PARK_SLOTS][UM_PARK_SEQ], um_slot_indirect[UM_PARK_SLOTS][UM_PARK_SEQ];
static id<MTLBuffer> um_slot_buf[UM_PARK_SLOTS][UM_PARK_SEQ * UM_PARK_BUFS];
static NSMutableArray<id<MTLCommandBuffer>> *um_slot_cmds[UM_PARK_SLOTS];  // まだ走っていない待機分（先頭が次に走る）
static double um_park_started_gpu;            // 合図を出した時刻を、GPU の時刻（GPUStartTime など）と同じ基準で
static uint64_t um_park_tick;                 // 常駐の対象になりうる提出を数える（枠の古さの判断に使う）
static id<MTLCommandBuffer> um_park_running;  // 合図を出して、完了を待っている1つ
static int um_park_running_slot;
static CFAbsoluteTime um_park_started;
static uint32_t um_park_counts[UM_PARK_SEQ];  // いま走っている仕事の、dispatch ごとの個数

// ---- 検算による完了の確認 ----
// Metal の完了の知らせは、GPU が終わってから数十マイクロ秒遅れて届く。GPU が書いた値はそれより前に CPU から
// 見え始めるが、場所ごとに見えるタイミングが違い、前回と同じ値のこともあるので、「値が見えた」だけでは
// 完了と判断できない。そこで、待機させる命令の最後に検算の kernel を足す。GPU は、仕事が書いた範囲の
// 64 語ごとの検算値を、依頼の番号を混ぜて共有メモリに書く。CPU は、自分に見えている出力から同じ値を計算し、
// 全部一致したら結果を取り込む（完了の知らせを待たない）。一致するのは、今回の検算値が見えていて、かつ
// 出力のすべての語が GPU の最終的な値と同じに見えているときだけ。合わないあいだは見直しを続け、
// 先に Metal が完了を知らせたらそれで終わる。
// 仕事が書く範囲が大きいとき（UM_CHECK_MAX_WORDS を超える）は、CPU の検算が重くなるので使わない。
// UM_GPU_RESIDENT=status で、この確認を使わず必ず完了の知らせを待つ。
//
// 検算値の作り方（GPU の um_check と CPU の um_check_matches で同じ。変えるときは両方）:
//   s = fmix64(fmix64(依頼の番号 + 定数) + 何番目の 64 語か)
//   語 j ごとに s = fmix64(s ^ (j << 32 | data[j]))     （j は Buffer の先頭からの語の番号）
// fmix64 は MurmurHash3 の仕上げの混ぜ（全単射で、入力の1ビットの違いが出力の全体に広がる）。
// 以前は「語ごとの値 × 定数」の和だった。和は語について線形なので、古い出力と新しい出力の差が打ち消し合うと
// （例: 2 語の出力が (s, s) と (s', s') で、s ^ s' が 0x9E3779B1 のビットの中だけにある）、古い出力のまま
// 今回の検算値と一致し、誤った結果を通した（評価関数の 2 語の出力で、40 万回に 21 回など。
// tests/metal/checksum.um は 10 万回に数十〜数百回を再現する）。
// いまの形で誤って一致するのは、次の2つの場合だが、どちらも1回の確認につき 2^-58 程度より小さい:
//   - 古い出力 + 今回の検算値: 依頼の番号から作った出発点 s は同じで、語の並びだけが違う。最初に違う語で
//     s が分かれ（1 歩は s について全単射）、そのあと再び合流するには、どこかの語で s ^ s' = (語の差) と
//     ならなければならない。s ^ s' は依頼の番号ごとに変わる 64 ビットの擬似乱数なので、プログラムの出力が
//     それに合う確率は 1 歩あたり 2^-64、64 語で 2^-58。最後の値が偶然そろう確率も 2^-64。
//   - 古い出力 + 古い検算値（前の依頼が書いたもの）: 出発点が依頼の番号ごとに違うので、同じ並びでも
//     最後の値は独立な擬似乱数どうしになり、一致するのは 2^-64 程度。依頼の番号を外すと、この場合が
//     必ず一致してしまうので、外さないこと。
// 語の番号 j を混ぜるので、語を入れ替えた出力も別の値になる。
// CPU 側の手間（Apple M4、-O2）: 1〜8 語 約 0.06 マイクロ秒（ほとんどが出発点を作る 2 回の fmix64）、
// 3000 語 約 2、65536 語 約 40 マイクロ秒（8 つの 64 語を並べて計算する。以前の和は 2 語で約 0.002、
// 3000 語で約 1、65536 語で約 20 マイクロ秒）。
#define UM_PARK_CHECKS 4
#define UM_CHECK_MAX_WORDS 65536
static id<MTLComputePipelineState> um_check_pipeline;
static NSUInteger um_check_width;
static uint64_t um_check_ticket;
static struct {
  int n;  // 検算する Buffer の数。0 なら検算を使わない
  struct {
    int d, i;            // 形の中の何番目の dispatch の、何番目の Buffer か
    uint32_t elem_words;
    int whole;           // 1 なら Buffer 全体（RacyBuffer）、0 なら先頭から個数ぶん
    uint32_t words;      // いま走っている仕事で検算する語数
    const void *sums;    // um_check_sums の contents（見直すたびに [contents] を呼ばないよう控えておく）
  } c[UM_PARK_CHECKS];
} um_check[UM_PARK_SLOTS];
static id<MTLBuffer> um_check_sums[UM_PARK_SLOTS][UM_PARK_CHECKS];
static id<MTLBuffer> um_check_params[UM_PARK_SLOTS][UM_PARK_CHECKS];    // [0] 依頼の番号の下位、[1] 語数、[2] 番号の上位
static id<MTLBuffer> um_check_indirect[UM_PARK_SLOTS][UM_PARK_CHECKS];
// 検算で完了とみなしたが、Metal の完了の知らせはまだ確かめていない命令。次に使うときに、失敗していないか確かめる。
static id<MTLCommandBuffer> um_park_unchecked;

// GPU を起こしたままにするスレッド。最後に待機 kernel を使ってから 50 ms たつと休む。
// 回し続ける kernel の1回の長さ（計算の回数）で、往復の時間が変わる。長いと GPU が動き出すまでが安定し、
// 短いと完了の知らせが早く届くことがある。どちらが速いかは仕事によって違う（軽い仕事は長いほう、
// GPU の計算が数十マイクロ秒かかる仕事は短いほうが速かった）ので、待機している形ごとに両方を試して選ぶ。
static const uint32_t um_awake_lengths[2] = {100000, 100};
#define UM_AWAKE_TRIAL 256     // それぞれの長さを試す回数（前半は GPU の状態が落ち着くまでの分で、数えない）
#define UM_AWAKE_RETRY 8192    // 選んだあと、この回数ごとに試し直す
static _Atomic double um_awake_until;
static pthread_t um_awake_thread;
static int um_awake_started;
static pthread_once_t um_awake_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t um_park_lock = PTHREAD_MUTEX_INITIALIZER;  // um_park_idle を参照
static id<MTLComputePipelineState> um_awake_pipeline;
static id<MTLBuffer> um_awake_buf;
static id<MTLComputePipelineState> um_mb_pipeline;  // 郵便受けの見張り（um_mb_poll。「郵便受けの列」を参照）

static void um_park_idle(void);
static void *um_awake_main(void *arg) {
  (void)arg;
  id<MTLBuffer> mine = um_awake_buf;
  id<MTLCommandQueue> q = [um_device newCommandQueue];
  CFAbsoluteTime next_idle = 0;
  __strong id<MTLCommandBuffer> flight[UM_AWAKE_PIPE_MAX + 1];
  int nflight = 0;
  for (;;) {
    CFAbsoluteTime now = CFAbsoluteTimeGetCurrent();
    if (now > atomic_load(&um_awake_until) || um_awake_quiet) {
      usleep(2000);
      um_park_idle();
      um_ap_idle();
      continue;
    }
    // 起きているあいだも待機の片付けを見る。gpu.submit やほかの形の合図が続くと um_awake_until が延び続け、
    // 休む側の分岐に来ないので、使われなくなった形の待機が残って Metal に打ち切られていた。
    if (now >= next_idle) {
      next_idle = now + 0.1;
      um_park_idle();
      um_ap_idle();
    }
    @autoreleasepool {
      id<MTLCommandBuffer> cmd = [q commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
      [enc setComputePipelineState:um_awake_pipeline];
      [enc setBuffer:mine offset:0 atIndex:0];
      [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      [enc endEncoding];
      [cmd commit];
      // UM_GPU_AWAKE_PIPE=n: 終わるのを待たずに n 個まで重ねて出す（GPU への指示を途切れさせない）。0（既定）は 1 つずつ待つ。
      if (um_awake_pipe > 0) {
        flight[nflight++] = cmd;
        if (nflight > um_awake_pipe) {
          [flight[0] waitUntilCompleted];
          for (int i = 1; i < nflight; i++) flight[i - 1] = flight[i];
          flight[--nflight] = nil;
        }
      } else if (um_mq_on && !um_awake_block) {
        // 郵便受けの列の見張りが回っているあいだ、ほかの列の命令は走れない（命令の切れ目でしか切り替わらない）。
        // 終わるのを待ち続けると待機の片付け（um_ap_idle）が見張りの寿命まで遅れるので、片付けを見ながら待つ。
        while ([cmd status] < MTLCommandBufferStatusCompleted) {
          CFAbsoluteTime t = CFAbsoluteTimeGetCurrent();
          if (t >= next_idle) {
            next_idle = t + 0.1;
            um_park_idle();
            um_ap_idle();
          }
          usleep(50);
        }
      } else {
        [cmd waitUntilCompleted];
      }
    }
  }
  return NULL;
}

static void um_awake_start_once(void);
static int um_awake_start(void) {
  pthread_once(&um_awake_once, um_awake_start_once);
  return um_awake_started > 0;
}
static void um_awake_start_once(void) {
  um_awake_started = -1;
  NSError *error = nil;
  id<MTLLibrary> lib = [um_device newLibraryWithSource:
      @"#include <metal_stdlib>\nusing namespace metal;\n"
       "kernel void um_awake(device uint *s [[buffer(0)]], uint3 gid [[thread_position_in_grid]]) {\n"
       "  uint acc = gid.x + s[0];\n"
       "  for (uint j = 0; j < s[2]; j++) { acc ^= acc << 13; acc ^= acc >> 17; acc ^= acc << 5; acc += j; }\n"
       "  s[1] = acc;\n"
       "}\n"
       // 検算（「検算による完了の確認」を参照）。スレッド g が、data の g 番目の 64 語の検算値を sums[g] に書く。
       // p[0] / p[2] は依頼の番号の下位 / 上位 32 ビット、p[1] は語数。CPU 側の um_check_matches と同じ計算。
       "static ulong um_fmix64(ulong k) {\n"
       "  k ^= k >> 33; k *= 0xff51afd7ed558ccdUL; k ^= k >> 33; k *= 0xc4ceb9fe1a85ec53UL; k ^= k >> 33;\n"
       "  return k;\n"
       "}\n"
       "kernel void um_check(device ulong *sums [[buffer(0)]], const device uint *data [[buffer(1)]],\n"
       "                     const device uint *p [[buffer(2)]], uint3 gid [[thread_position_in_grid]]) {\n"
       "  uint words = p[1];\n"
       "  uint base = gid.x * 64u;\n"
       "  if (base >= words) return;\n"
       "  ulong t = ulong(p[0]) | (ulong(p[2]) << 32);\n"
       "  ulong s = um_fmix64(um_fmix64(t + 0x9E3779B97F4A7C15UL) + ulong(gid.x));\n"
       "  for (uint j = base; j < min(base + 64u, words); j++) s = um_fmix64(s ^ ((ulong(j) << 32) | ulong(data[j])));\n"
       "  sums[gid.x] = s;\n"
       "}\n"
       // 郵便受けの見張り（「郵便受けの列」を参照）。枠（64 バイトおき）を 1 回ずつ読み、番号が seq 以上になっていたら抜ける。
       // m = (枠の数, 間を空けずに見る枠の数, seq, 0)。fast 個を過ぎたら、それまでの時間の 1/64 ずつ間を空ける
       // （見張りの寿命を延ばす。待った時間に対する遅れは 2% 弱）。st には（見つけたか、見た枠の数、seq、空回りの値）を書く。
       "kernel void um_mb_poll(device atomic_uint *mb [[buffer(0)]], constant uint4 &m [[buffer(1)]], device uint *st [[buffer(2)]],\n"
       "                       uint3 gid [[thread_position_in_grid]]) {\n"
       "  if (gid.x != 0) return;\n"
       "  uint n = m.x, fast = m.y, seq = m.z, found = 0, k = 0, acc = seq, total = fast * 32u;\n"
       "  for (; k < n; k++) {\n"
       "    if (int(atomic_load_explicit(&mb[k * 16u], memory_order_relaxed) - seq) >= 0) { found = 1; break; }\n"
       "    if (k >= fast) {\n"
       "      uint w = total >> 6;\n"
       "      for (uint j = 0; j < w; j++) acc = acc * 1664525u + 1013904223u;\n"
       "      total += w + 32u;\n"
       "    }\n"
       "  }\n"
       "  st[0] = found; st[1] = k; st[2] = seq; st[3] = acc;\n"
       "}\n"
options:nil error:&error];
  id<MTLFunction> fn = lib ? [lib newFunctionWithName:@"um_awake"] : nil;
  id<MTLFunction> check = lib ? [lib newFunctionWithName:@"um_check"] : nil;
  um_check_pipeline = check ? [um_device newComputePipelineStateWithFunction:check error:&error] : nil;
  um_check_width = um_check_pipeline ? [um_check_pipeline threadExecutionWidth] : 0;
  um_awake_pipeline = fn ? [um_device newComputePipelineStateWithFunction:fn error:&error] : nil;
  id<MTLFunction> poll = lib ? [lib newFunctionWithName:@"um_mb_poll"] : nil;
  um_mb_pipeline = poll ? [um_device newComputePipelineStateWithFunction:poll error:&error] : nil;
  um_awake_buf = [um_device newBufferWithLength:64 options:MTLResourceStorageModeShared];
  if (!um_awake_pipeline || !um_awake_buf) return;
  memset([um_awake_buf contents], 0, 64);
  ((uint32_t *)[um_awake_buf contents])[2] = um_awake_len >= 0 ? (uint32_t)um_awake_len : um_awake_lengths[0];
  if (pthread_create(&um_awake_thread, NULL, um_awake_main, NULL) != 0) return;
  pthread_detach(um_awake_thread);
  um_awake_started = 1;
}

// 検算値の計算（CPU 側。GPU の um_check と同じ）。
static inline uint64_t um_fmix64(uint64_t k) {
  k ^= k >> 33;
  k *= 0xff51afd7ed558ccdull;
  k ^= k >> 33;
  k *= 0xc4ceb9fe1a85ec53ull;
  k ^= k >> 33;
  return k;
}
static inline uint64_t um_check_seed(uint64_t ticket, uint32_t g) {
  return um_fmix64(um_fmix64(ticket + 0x9E3779B97F4A7C15ull) + g);
}
// GPU が書いている最中かもしれない場所なので、アトミックな読み出しにする（順序の保証は要らない）。
static inline uint32_t um_check_load(const uint32_t *data, uint32_t j) {
  return atomic_load_explicit((const _Atomic uint32_t *)&data[j], memory_order_relaxed);
}

// data の先頭 words 語の検算値が、GPU が sums に書いた値（依頼の番号 ticket のもの）と全部一致すれば 1。
// 1 つの 64 語の計算は前の語の結果を待つ一続きの計算なので、8 つの 64 語を並べて計算して待ち時間を重ねる。
#define UM_CHECK_LANES 8
static int um_check_matches(uint64_t ticket, const uint32_t *data, uint32_t words, const void *sums_mem) {
  const _Atomic uint64_t *sums = (const _Atomic uint64_t *)sums_mem;
  uint32_t groups = (words + 63) / 64;
  for (uint32_t g = 0; g < groups; g += UM_CHECK_LANES) {
    uint32_t lanes = groups - g < UM_CHECK_LANES ? groups - g : UM_CHECK_LANES;
    uint64_t s[UM_CHECK_LANES];
    uint32_t end[UM_CHECK_LANES];
    for (uint32_t q = 0; q < lanes; q++) {
      s[q] = um_check_seed(ticket, g + q);
      end[q] = (g + q) * 64 + 64 < words ? (g + q) * 64 + 64 : words;
    }
    if (lanes == UM_CHECK_LANES && end[UM_CHECK_LANES - 1] == (g + UM_CHECK_LANES) * 64) {
      // 8 つとも 64 語ちょうど（よくある場合）。
      for (uint32_t i = 0; i < 64; i++) {
        for (uint32_t q = 0; q < UM_CHECK_LANES; q++) {
          uint32_t j = (g + q) * 64 + i;
          s[q] = um_fmix64(s[q] ^ ((uint64_t)j << 32 | um_check_load(data, j)));
        }
      }
    } else {
      for (uint32_t i = 0; i < 64; i++) {
        for (uint32_t q = 0; q < lanes; q++) {
          uint32_t j = (g + q) * 64 + i;
          if (j < end[q]) s[q] = um_fmix64(s[q] ^ ((uint64_t)j << 32 | um_check_load(data, j)));
        }
      }
    }
    int bad = 0;
    for (uint32_t q = 0; q < lanes; q++) bad |= s[q] != atomic_load_explicit(&sums[g + q], memory_order_relaxed);
    if (bad) return 0;
  }
  return 1;
}

// 合図を出した待機 kernel が終わるのを待つ。短い仕事が多いので、しばらくは寝ずに完了の状態を見続ける。
//
// 完了は必ず Metal の完了の状態（status）で判断する。試して使えなかった方法:
//   - kernel のあとに「終わりの印」を共有メモリに書かせ、それが CPU から見えたら完了とみなす
//     → 印が見えても、仕事の出力はまだ古い値のことがある（GPU の書き込みは場所ごとに別々のタイミングで見える）。
//       往復は約 60 マイクロ秒まで縮むが、結果が正しくない。
//   - Metal に共有イベントで完了を知らせさせ、その値を見続ける → 正しいが、status を見るより遅い。
// 検算で完了とみなした命令が、Metal から見ても失敗していないことを確かめる。まだ知らせが届いていなければ待つ。
static void um_park_settle(void) {
  id<MTLCommandBuffer> cmd = um_park_unchecked;
  if (!cmd) return;
  um_park_unchecked = nil;
  if ([cmd status] < MTLCommandBufferStatusCompleted) [cmd waitUntilCompleted];
  if ([cmd status] != MTLCommandBufferStatusCompleted) um_metal_fail("running the kernel", [cmd error]);
}

// 枠 k でいま走っている仕事の出力が、CPU から全部見えているかを検算で確かめる。
static int um_park_verified(int k) {
  for (int c = 0; c < um_check[k].n; c++) {
    const uint32_t *data = (const uint32_t *)(uintptr_t)um_slot[k].d[um_check[k].c[c].d].sig[um_check[k].c[c].i * 4];
    if (!um_check_matches(um_check_ticket, data, um_check[k].c[c].words, um_check[k].c[c].sums)) return 0;
  }
  return 1;
}

static void um_park_wait(void) {
  id<MTLCommandBuffer> cmd = um_park_running;
  if (!cmd) return;
  int k = um_park_running_slot;
  CFAbsoluteTime spin_until = um_park_started + 0.0005;
  int verified = 0;
  for (;;) {
    if (um_check[k].n > 0 && um_park_verified(k)) {
      verified = 1;
      break;
    }
    int done;
    @autoreleasepool {  // um_gpu_wait の説明を参照
      done = [cmd status] >= MTLCommandBufferStatusCompleted;
    }
    if (done) break;
    if (CFAbsoluteTimeGetCurrent() > spin_until) {
      [cmd waitUntilCompleted];
      break;
    }
  }
  double total = CFAbsoluteTimeGetCurrent() - um_park_started;
  double busy = total;
  if (verified) {
    um_park_unchecked = cmd;
  } else {
    if ([cmd status] != MTLCommandBufferStatusCompleted) um_metal_fail("running the kernel", [cmd error]);
    busy = [cmd GPUEndTime] - [cmd GPUStartTime];
    if (busy < 0 || busy > total) busy = total;
  }
  // 起こしておく kernel の長さを試している最中なら、この往復の時間を記録する。極端に長いもの（割り込みなど）は除く。
  uint32_t phase = um_slot[k].runs % UM_AWAKE_RETRY;
  if (phase < UM_AWAKE_TRIAL * 2) {
    int which = phase < UM_AWAKE_TRIAL ? 0 : 1;
    if (phase == 0) {
      um_slot[k].trial_sum[0] = um_slot[k].trial_sum[1] = 0;
      um_slot[k].trial_n[0] = um_slot[k].trial_n[1] = 0;
    }
    // 切り替えた直後は、前の長さの kernel がまだ回っていたり、GPU の状態が落ち着いていなかったりするので数えない。
    if (phase % UM_AWAKE_TRIAL >= UM_AWAKE_TRIAL / 2 && total < 0.001) {
      um_slot[k].trial_sum[which] += total;
      um_slot[k].trial_n[which]++;
    }
    if (phase == UM_AWAKE_TRIAL * 2 - 1) {
      double a = um_slot[k].trial_n[0] ? um_slot[k].trial_sum[0] / um_slot[k].trial_n[0] : 1;
      double b = um_slot[k].trial_n[1] ? um_slot[k].trial_sum[1] / um_slot[k].trial_n[1] : 1;
      um_slot[k].awake = b < a;
      if (um_trace) {
        fprintf(stderr, "gpu: resident kernel %d: keep-awake length %u (%.3f ms vs %.3f ms)\n", um_slot[k].d[0].kernel,
                um_awake_lengths[um_slot[k].awake], a * 1000.0, b * 1000.0);
      }
    }
  }
  um_slot[k].runs++;
  // 自動判断のための実測。GPU が計算していた時間を要素数で割り、残りを固定の手間として dispatch の数で分ける。
  // 検算で先に進んだ場合は、GPU の時刻がまだ分からないので更新しない。
  int n = um_slot[k].n;
  double elems = 0;
  for (int d = 0; d < n; d++) elems += um_park_counts[d];
  pthread_mutex_lock(&um_table_lock);
  for (int d = 0; d < n && !verified; d++) {
    int32_t kernel = um_slot[k].d[d].kernel;
    if (kernel >= 0 && kernel < UM_MAX_TRACKED && elems > 0) {
      um_stat *s = &um_stats[kernel];
      um_blend(&s->gpu_per_elem, busy / elems, s->gpu_runs);
      um_blend(&s->gpu_fixed, (total - busy) / n, s->gpu_runs);
      if (s->gpu_runs < 2) s->gpu_runs++;
    }
  }
  pthread_mutex_unlock(&um_table_lock);
  if (um_trace && verified) {
    fprintf(stderr, "gpu: resident kernel %d (%d in a row) count %u: %.3f ms (taken when the checksum matched)\n",
            um_slot[k].d[0].kernel, n, um_park_counts[0], total * 1000.0);
  } else if (um_trace) {
    // 内訳: GPU の計算、合図から GPU が動き出すまで、GPU が終わってから完了が分かるまで。
    fprintf(stderr, "gpu: resident kernel %d (%d in a row) count %u: %.3f ms (gpu %.3f, to start %.3f, after end %.3f)\n",
            um_slot[k].d[0].kernel, n, um_park_counts[0], total * 1000.0, busy * 1000.0,
            ([cmd GPUStartTime] - um_park_started_gpu) * 1000.0, (um_gpu_clock() - [cmd GPUEndTime]) * 1000.0);
  }
  um_park_running = nil;
}

// 枠 k の待機分を1つ足す（形に含まれる dispatch を、順に1つの命令に入れる）。
static void um_park_commit_one(int k) {
  id<MTLCommandBuffer> cmd = [um_slot_queue[k] commandBuffer];
  if (!cmd) um_metal_fail("creating the command buffer", nil);
  [cmd encodeWaitForEvent:um_slot_event[k] value:um_slot[k].committed + 1];
  id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
  if (!enc) um_metal_fail("creating the command buffer", nil);
  for (int d = 0; d < um_slot[k].n; d++) {
    [enc setComputePipelineState:um_slot_pipeline[k][d]];
    [enc setBuffer:um_slot_params[k][d] offset:0 atIndex:0];
    for (int32_t i = 0; i < um_slot[k].d[d].nbufs; i++) {
      [enc setBuffer:um_slot_buf[k][d * UM_PARK_BUFS + i] offset:0 atIndex:(NSUInteger)(i + 1)];
    }
    [enc dispatchThreadgroupsWithIndirectBuffer:um_slot_indirect[k][d]
                           indirectBufferOffset:0
                          threadsPerThreadgroup:MTLSizeMake(um_slot[k].d[d].width, 1, 1)];
  }
  // 検算。仕事が書く Buffer と同じ GPU 側のバッファを読むので、仕事のあとに実行される。
  for (int c = 0; c < um_check[k].n; c++) {
    [enc setComputePipelineState:um_check_pipeline];
    [enc setBuffer:um_check_sums[k][c] offset:0 atIndex:0];
    [enc setBuffer:um_slot_buf[k][um_check[k].c[c].d * UM_PARK_BUFS + um_check[k].c[c].i] offset:0 atIndex:1];
    [enc setBuffer:um_check_params[k][c] offset:0 atIndex:2];
    [enc dispatchThreadgroupsWithIndirectBuffer:um_check_indirect[k][c]
                           indirectBufferOffset:0
                          threadsPerThreadgroup:MTLSizeMake(um_check_width, 1, 1)];
  }
  [enc endEncoding];
  [cmd commit];
  [um_slot_cmds[k] addObject:cmd];
  um_slot[k].committed++;
}

// 枠の状態を変える。空きかどうかが変わったら、GPU が覚えている領域の数（um_gpu_regions）も合わせる。
static void um_park_state(int k, int state) {
  int was = um_slot[k].state != 0, now = state != 0;
  um_slot[k].state = state;
  if (was != now) atomic_fetch_add(&um_gpu_regions, now ? 1 : -1);
}

// 枠 k が持っている Metal の物を手放す。
static void um_park_clear(int k) {
  [um_slot_cmds[k] removeAllObjects];
  for (int d = 0; d < UM_PARK_SEQ; d++) {
    um_slot_pipeline[k][d] = nil;
    um_slot_params[k][d] = nil;
    um_slot_indirect[k][d] = nil;
    for (int i = 0; i < UM_PARK_BUFS; i++) um_slot_buf[k][d * UM_PARK_BUFS + i] = nil;
  }
  for (int c = 0; c < UM_PARK_CHECKS; c++) {
    um_check_sums[k][c] = nil;
    um_check_params[k][c] = nil;
    um_check_indirect[k][c] = nil;
  }
  um_check[k].n = 0;
}

// 枠 k の待機をやめて、枠を空ける。残っている待機分は、個数 0 で空振りさせて終わるのを待つ
// （提出した命令は取り消せないため）。
static void um_park_stop(int k) {
  if (um_slot[k].state != 2) {
    um_park_state(k, 0);
    return;
  }
  @autoreleasepool {
    um_park_wait();
    um_park_settle();
    if ([um_slot_cmds[k] count] > 0) {
      for (int c = 0; c < um_check[k].n; c++) {
        memset([um_check_indirect[k][c] contents], 0, 12);
        ((uint32_t *)[um_check_params[k][c] contents])[1] = 0;  // 語数 0: 走ってしまっても何も書かない
      }
      for (int d = 0; d < um_slot[k].n; d++) {
        uint32_t *groups = (uint32_t *)[um_slot_indirect[k][d] contents];
        groups[0] = 0;
        groups[1] = 0;
        groups[2] = 0;
        ((uint32_t *)[um_slot_params[k][d] contents])[0] = 0;  // count = 0: 走ってしまっても何も書かない
      }
      id<MTLCommandBuffer> last = [um_slot_cmds[k] lastObject];
      um_slot[k].fired = um_slot[k].committed;
      um_slot_event[k].signaledValue = um_slot[k].fired;
      um_mq_kick();
      [last waitUntilCompleted];
    }
    um_park_clear(k);
    um_park_state(k, 0);
    if (um_trace) fprintf(stderr, "gpu: resident stopped for kernel %d\n", um_slot[k].d[0].kernel);
  }
}

// 解放されようとしている領域を待機 kernel が使っていれば、その待機をやめる（um_free から呼ばれる）。
// 放っておくと、同じ場所に確保し直された別の領域ではなく、解放済みの領域に GPU が書くことがある。
// 回数を数えているだけの枠も空ける（同じ場所に別の Buffer が確保されると、同じ形に見えてしまう）。
static void um_park_release(void *p) {
  pthread_mutex_lock(&um_park_lock);
  for (int k = 0; k < UM_PARK_SLOTS; k++) {
    if (um_slot[k].state == 0) continue;
    int used = 0;
    for (int d = 0; d < um_slot[k].n && !used; d++) {
      for (int32_t i = 0; i < um_slot[k].d[d].nbufs; i++) {
        if ((void *)(uintptr_t)um_slot[k].d[d].sig[i * 4] == p) {
          used = 1;
          break;
        }
      }
    }
    if (used) um_park_stop(k);
  }
  pthread_mutex_unlock(&um_park_lock);
}

// 合図を待たせた命令は、長く待たせすぎると Metal が打ち切る（Apple M4・macOS 27 で、約 3〜6 秒待たせると
// kIOGPUCommandBufferCallbackErrorTimeout。打ち切られたあとに合図を出すと、その命令が失敗して停止していた）。
// そこで、GPU を起こしておくスレッドが、待機中の枠のうち UM_PARK_IDLE_STOP 秒より長く合図が出ていないものを
// 片付ける（um_park_stop）。同じ形がまた続けて来れば、UM_PARK_AFTER 回で待機を始め直す。
// 見るのは、スレッドが休んでいるあいだ（2 ms ごと）と、起きているあいだ（0.1 秒ごと。gpu.submit やほかの形の
// 合図で起きたままになるため）。
// 枠の状態を2つのスレッドから触るので、常駐の経路（um_park_try / um_park_release / ここ）は um_park_lock を持って動く。
#define UM_PARK_IDLE_STOP 1.0
static void um_park_idle(void) {
  // trylock にはしない。常駐の経路は GPU の結果を待つあいだも枠を握っているので、別の形の gpu.dispatch が
  // 続くと trylock はほぼ必ず失敗し、使われなくなった形の待機が片付かなかった。待つのは長くても dispatch 1回ぶん。
  pthread_mutex_lock(&um_park_lock);
  CFAbsoluteTime now = CFAbsoluteTimeGetCurrent();
  for (int k = 0; k < UM_PARK_SLOTS; k++) {
    if (um_slot[k].state == 2 && now - um_slot[k].last_fire > UM_PARK_IDLE_STOP) {
      if (um_trace) fprintf(stderr, "gpu: resident kernel %d idle for %.1f s\n", um_slot[k].d[0].kernel, now - um_slot[k].last_fire);
      um_park_stop(k);
    }
  }
  pthread_mutex_unlock(&um_park_lock);
}

// ---- 非同期の提出（gpu.submit） ----
// 終わるのを待たずに戻り、券（0 でない番号）を返す。gpu.ready / gpu.wait が券で完了を確かめる。
// 渡せるのは RacyBuffer だけ（コンパイラが検査する）なので、CPU が同じ領域を同時に触ってもメモリは壊れない。
// ただし解放は別で、GPU がまだ使っている領域を解放させないよう、um_free の前に完了を待つ（um_on_free）。
//
// 複数のスレッドから呼べる。券の表と控えは um_gpu_lock で守る。券は全体で一意で、どのスレッドの券でも、
// どのスレッドから ready / wait してもよい。券の下位 UM_ASYNC_BITS ビットが表の枠の番号。
// 待つ人（ready / wait / 解放）は、枠に waiting を付けて鍵を放し、Metal の完了の状態と検算を鍵の外で見る。
// waiting が付いている枠は空けない（終わったと分かったら done を付け、最後の待つ人が空ける）。
#define UM_ASYNC_BITS 10
#define UM_ASYNC_MAX (1 << UM_ASYNC_BITS)
static pthread_mutex_t um_gpu_lock = PTHREAD_MUTEX_INITIALIZER;
static struct {
  uint64_t id;  // 0 なら空き
  int nbufs;
  void *data[UM_BATCH_BUFS];
  // 検算による完了の確認（um_check を参照）。この仕事が書く Buffer ごとに、場所と語数。nchecks が 0 なら使わない。
  int nchecks;
  const uint32_t *check_data[UM_PARK_CHECKS];
  uint32_t check_words[UM_PARK_CHECKS];
  // 検算値の場所（um_async_sums の contents）。見直すたびに [contents] を呼ばないよう、控えておく
  // （[contents] は Buffer を自動解放の枠に入れる。枠の外で何度も呼ぶとメモリが増え続けた）。
  const void *check_sums[UM_PARK_CHECKS];
  int waiting;  // 鍵を放してこの券を見ているスレッドの数
  int done;     // 終わったと分かっている（waiting が 0 になったら空ける）
  // gpu.submit の常駐: まだ投げたスレッドの並び（HOLD でつないでいる最中）に入っている券なら、その並び。
  // 並びから外して命令に入れる最中は、chain も um_async_cmd も空（um_async_cmd_of がしばらく待つ）。
  struct um_chain *chain;
} um_async[UM_ASYNC_MAX];
static id<MTLBuffer> um_async_sums[UM_ASYNC_MAX][UM_PARK_CHECKS];
static id<MTLBuffer> um_async_check_params[UM_ASYNC_MAX][UM_PARK_CHECKS];
static id<MTLCommandBuffer> um_async_cmd[UM_ASYNC_MAX];
static uint64_t um_async_seq = 1;  // 券の上位の部分（提出ごとに増える）
static int um_async_free[UM_ASYNC_MAX];
static int um_async_nfree = -1;    // -1 = まだ空きの並びを作っていない
static int um_async_live;
// 投げるたびに作り直さないよう、取っておくもの。
//   um_async_wrap: 領域の場所 → GPU 側のバッファ（その領域が解放されるときに捨てる。um_on_free）
//   um_async_params: 番号 i の提出が使う引数の領域（大きさが足りるあいだ使い回す）
static NSMutableDictionary<NSNumber *, id<MTLBuffer>> *um_async_wrap;

// ---- 追跡しない Buffer（gpu.untracked）----
// um_untracked: 領域の場所の集合（um_gpu_lock で守る。領域が解放されるときに外す。um_on_free）。
// gpu.submit の経路（um_async_wrap）で、この領域の GPU 側のバッファを MTLResourceHazardTrackingModeUntracked で作る。
// Metal は追跡しないバッファについて、命令（command buffer）・列どうしの書き込みの順番を待たない。そのため、同じ
// 作業用の Buffer の別々の部分を使う仕事どうしが、並んで走れる。安全の決まり（呼ぶ側が守る）: 同時に走りうる GPU の
// 仕事どうしで、同じ要素を一方が書き、もう一方が読む・書くことがない。1つの命令の中の dispatch の順番（HOLD でつないだ
// 並びと、検算の kernel）は、dispatch の前に memoryBarrier を入れて守る。普通の dispatch 文の経路（待つ）は今までどおり。
static NSMutableSet<NSNumber *> *um_untracked;
static _Atomic int um_untracked_n;  // um_untracked の数（um_on_free が素通りしてよいかを、鍵なしで見るため）
static int um_is_untracked(const void *data) {
  return um_untracked && [um_untracked containsObject:@((uintptr_t)data)];
}
static MTLResourceOptions um_wrap_options(const void *data) {
  return MTLResourceStorageModeShared | (um_is_untracked(data) ? MTLResourceHazardTrackingModeUntracked : 0);
}
// 控えてある GPU 側のバッファ b を、大きさと追跡の有無が変わっていなければそのまま使えるか。
static int um_wrap_ok(id<MTLBuffer> b, size_t rounded, const void *data) {
  if (!b || [b length] != rounded) return 0;
  return ([b hazardTrackingMode] == MTLHazardTrackingModeUntracked) == (um_is_untracked(data) != 0);
}
static id<MTLBuffer> um_async_params[UM_ASYNC_MAX];
// 投げられた仕事は、GPU に余裕があればすぐ提出する。先に出した命令が UM_INFLIGHT_MAX 個たまっているあいだに
// 投げられた分は、1つの命令（um_open_cmd）にためておき、先の命令が1つ終わった時点でまとめて提出する。書く側は1つずつ
// 投げるだけで、提出の手間（1回 約 0.1 ms）が何件かで分け合われる。GPU が空いているときは待たせないので、
// それで遅くなることはない。
// 前の提出が終わったことは、Metal が別のスレッドで呼ぶ完了の処理で知る。そこからも um_open_cmd を
// 触るので、um_async_lock で守る。
static pthread_mutex_t um_async_lock = PTHREAD_MUTEX_INITIALIZER;
// 列は 2 本。0 は今までの列（um_queue）、1 は急ぎの列（um_queue_urgent。最初の URGENT の submit で作る）。
// 急ぎの評価が、先に投げた大きいまとまりの後ろに並ばないようにするため。
static id<MTLCommandQueue> um_queue_urgent;
static id<MTLCommandBuffer> um_open_cmd[2];       // ためている最中の命令（まだ提出していない）
static id<MTLComputeCommandEncoder> um_open_enc[2];
static int um_open_n[2];                          // そこに入っている仕事の数
static int um_open_held[2];                       // 最後に入れた仕事が HOLD（完了の処理から続けて提出しない）
static int um_inflight[2];                        // 提出して、まだ終わっていない命令の数
// 命令をこの数まで出しているあいだは、次をためる。GPU は命令を数個まで重ねて処理できるので、1 にすると
// かえって遅い。実測（32 局面の評価を 8 件預けて回す形、Apple M4）: 1 → 84、2〜3 → 34、4 → 30、
// 6 → 29、ためない → 38 マイクロ秒/件。急ぎの列（1）には使わない（いつもすぐ提出する）。
#define UM_INFLIGHT_MAX 4
// 1つの命令にためる仕事の上限（これに達したら、GPU が忙しくても、HOLD でも提出する）。
#define UM_OPEN_MAX 32

// ---- 完了の印 ----
// 命令の完了の処理（Metal のスレッド）が、その命令に入っていた券を、券の表の枠ごとに書く。
// ready / wait は、枠の印が自分の券と同じなら、Metal にも検算にも聞かずに完了と分かる。
// 券は増え続けるので、大きいほうだけを残す（検算で先に空けた枠が使い回されたあとに、前の命令の完了の処理が
// 遅れて来ても、新しい券の印を消さない）。
static _Atomic uint64_t um_async_fin[UM_ASYNC_MAX];
static _Atomic int um_async_failed;  // 完了の処理で、失敗した命令を見た
typedef struct {
  int n;
  int slot[UM_OPEN_MAX];
  uint64_t ticket[UM_OPEN_MAX];
} um_fin_list;
static um_fin_list um_open_fins[2];  // ためている命令に入っている券
// 書く Buffer（領域の場所）→ 最後に書いた券と列（ticket << 1 | q）。列をまたいで同じ領域を使うときの安全網。
static NSMutableDictionary<NSNumber *, NSNumber *> *um_async_lastwrite;

static void um_fin_mark(int slot, uint64_t ticket) {
  uint64_t cur = atomic_load(&um_async_fin[slot]);
  while (cur < ticket && !atomic_compare_exchange_weak(&um_async_fin[slot], &cur, ticket)) {
  }
}

// ためてある列 q の命令を提出する。um_async_lock を持った状態で呼ぶ。
static void um_async_commit_open(int q) {
  if (!um_open_cmd[q] || um_open_n[q] == 0) return;
  id<MTLCommandBuffer> cmd = um_open_cmd[q];
  [um_open_enc[q] endEncoding];
  um_fin_list *fins = (um_fin_list *)malloc(sizeof(um_fin_list));
  if (!fins) um_metal_fail("allocating the completion list", nil);
  *fins = um_open_fins[q];
  [cmd addCompletedHandler:^(id<MTLCommandBuffer> done) {
    // 終わったら、券ごとに完了の印を書き、列 0 ならそのあいだにたまった分を続けて提出する。
    // Metal のスレッドで呼ばれるので、ここにも自動解放の枠を置く（um_gpu_wait の説明を参照）。
    @autoreleasepool {
      if ([done status] != MTLCommandBufferStatusCompleted) atomic_store(&um_async_failed, 1);
      um_gs_gpu_time(done);
      for (int k = 0; k < fins->n; k++) um_fin_mark(fins->slot[k], fins->ticket[k]);
      free(fins);
      pthread_mutex_lock(&um_async_lock);
      um_inflight[q]--;
      if (q == 0 && um_inflight[0] < UM_INFLIGHT_MAX && !um_open_held[0]) um_async_commit_open(0);
      pthread_mutex_unlock(&um_async_lock);
    }
  }];
  um_inflight[q]++;
  um_open_cmd[q] = nil;
  um_open_enc[q] = nil;
  um_open_n[q] = 0;
  um_open_held[q] = 0;
  um_open_fins[q].n = 0;
  [cmd commit];
  um_mq_kick();
}

// cmd がまだためている最中の命令なら、いま提出する（待つ前に必ず呼ぶ）。どの鍵を持っていてもよい（um_async_lock 以外）。
static void um_async_flush_cmd(id<MTLCommandBuffer> cmd) {
  pthread_mutex_lock(&um_async_lock);
  for (int q = 0; q < 2; q++) {
    if (cmd == um_open_cmd[q]) um_async_commit_open(q);
  }
  pthread_mutex_unlock(&um_async_lock);
}

// cmd が HOLD でためている最中の命令なら、いま提出する（gpu.ready が呼ぶ。HOLD の券を ready で見続けても
// 止まらないように）。HOLD でない理由（列 0 が忙しい）でためている命令は、今までどおり完了の処理に任せる。
static void um_async_flush_held(id<MTLCommandBuffer> cmd) {
  pthread_mutex_lock(&um_async_lock);
  for (int q = 0; q < 2; q++) {
    if (cmd == um_open_cmd[q] && um_open_held[q]) um_async_commit_open(q);
  }
  pthread_mutex_unlock(&um_async_lock);
}

// `gpu.flags(f)`: このスレッドの、このあとの gpu.submit に付ける印を f にする。前の値を返す。
uint32_t um_gpu_set_flags(uint32_t f) {
  uint32_t old = um_submit_flags;
  um_submit_flags = f;
  return old;
}

// `gpu.commit()`: HOLD でためてある命令を（どちらの列も）いま提出する。gpu.submit の常駐で、スレッドの並びに
// ためてある submit も（どのスレッドのものも）積んで提出する。
void um_gpu_commit(void) {
  if (um_ready <= 0) return;
  @autoreleasepool {  // um_gpu_wait の説明を参照
    um_chains_flush_all();
    pthread_mutex_lock(&um_async_lock);
    um_async_commit_open(0);
    um_async_commit_open(1);
    pthread_mutex_unlock(&um_async_lock);
  }
}

// `gpu.untracked(b)`: Buffer b の領域を、gpu.submit の経路で Metal に追跡させない（um_untracked を参照）。
// 次にその領域を使う submit から効く。領域が解放されると元に戻る。
void um_gpu_untracked(void *data) {
  if (!data || !um_metal_init()) return;
  @autoreleasepool {
    pthread_mutex_lock(&um_gpu_lock);
    if (!um_untracked) um_untracked = [NSMutableSet set];
    NSNumber *key = @((uintptr_t)data);
    if (![um_untracked containsObject:key]) {
      [um_untracked addObject:key];
      atomic_fetch_add(&um_untracked_n, 1);
    }
    pthread_mutex_unlock(&um_gpu_lock);
  }
}

// 券 t の枠。なければ -1。um_gpu_lock を持って呼ぶ。
static int um_async_find(uint64_t t) {
  if (t == 0) return -1;
  int i = (int)(t & (UM_ASYNC_MAX - 1));
  return um_async[i].id == t ? i : -1;
}

// 終わった枠を空ける。待っている人がいれば、印だけ付けて最後の人に任せる。um_gpu_lock を持って呼ぶ。
static void um_async_release(int i) {
  if (um_async[i].waiting > 0) {
    um_async[i].done = 1;
    return;
  }
  um_async_cmd[i] = nil;
  um_async[i].id = 0;
  um_async[i].done = 0;
  um_async[i].chain = NULL;
  um_async_free[um_async_nfree++] = i;
  um_async_live--;
}

static id<MTLCommandBuffer> um_async_cmd_of(int i, uint64_t t);

// 鍵の外で見るための、検算の控え。
typedef struct {
  uint64_t id;
  int n;
  const uint32_t *data[UM_PARK_CHECKS];
  uint32_t words[UM_PARK_CHECKS];
  const void *sums[UM_PARK_CHECKS];
} um_check_snap;

static void um_async_snap(int i, um_check_snap *s) {
  s->id = um_async[i].id;
  s->n = um_async[i].nchecks;
  for (int c = 0; c < s->n; c++) {
    s->data[c] = um_async[i].check_data[c];
    s->words[c] = um_async[i].check_words[c];
    s->sums[c] = um_async[i].check_sums[c];
  }
}

// 提出の結果が、CPU から全部見えているかを検算で確かめる（um_park_verified と同じ考え方）。鍵は要らない
// （waiting を付けた枠の控えなので、そのあいだ検算値の領域は使い回されない）。
static int um_snap_verified(const um_check_snap *s) {
  if (s->n == 0) return 0;
  for (int c = 0; c < s->n; c++) {
    if (!um_check_matches(s->id, s->data[c], s->words[c], s->sums[c])) return 0;
  }
  return 1;
}

// 鍵を持たずに、命令 cmd の完了か検算の一致を待つ。検算で先に進んだら 1、Metal の完了の知らせで終わったら 0
// （命令が失敗していれば停止する）。短い仕事が多いので、しばらくは寝ずに見続ける。
static int um_async_await(id<MTLCommandBuffer> cmd, const um_check_snap *s) {
  um_async_flush_cmd(cmd);
  CFAbsoluteTime until = CFAbsoluteTimeGetCurrent() + 0.0005;
  for (;;) {
    int done;
    @autoreleasepool {
      done = [cmd status] >= MTLCommandBufferStatusCompleted;
    }
    if (done) break;
    // 完了の知らせより前でも、検算が合えば結果は読める。
    if (s && um_snap_verified(s)) {
      um_gs_add(&um_gs_by_check, 1);
      return 1;
    }
    if (CFAbsoluteTimeGetCurrent() > until) break;
    // 見張るあいだは CPU を譲る。コアの数より多いスレッドが GPU を待つとき、待っているだけのスレッドが
    // 計算するスレッドの CPU を奪わないように（ほかに動けるスレッドがなければ、すぐ戻ってくる）。
    sched_yield();
  }
  @autoreleasepool {
    if ([cmd status] < MTLCommandBufferStatusCompleted) {
      CFAbsoluteTime t0 = um_gs_on ? CFAbsoluteTimeGetCurrent() : 0;
      [cmd waitUntilCompleted];
      if (um_gs_on) {
        um_gs_add(&um_gs_blocking, 1);
        um_gs_add(&um_gs_blocking_us, (uint64_t)((CFAbsoluteTimeGetCurrent() - t0) * 1e6));
      }
    }
    if ([cmd status] != MTLCommandBufferStatusCompleted) um_metal_fail("running the kernel", [cmd error]);
  }
  um_gs_add(&um_gs_by_status, 1);
  return 0;
}

// 空いている枠を1つ取る。なければ、いちばん古い提出（待っている人のいないもの）が終わるのを待って空ける。
// um_gpu_lock を持って呼ぶ（待つあいだは放す）。
static int um_async_take(void) {
  if (um_async_nfree < 0) {
    for (int i = 0; i < UM_ASYNC_MAX; i++) um_async_free[i] = UM_ASYNC_MAX - 1 - i;
    um_async_nfree = UM_ASYNC_MAX;
  }
  for (;;) {
    if (um_async_nfree > 0) return um_async_free[--um_async_nfree];
    // 完了の印が付いている枠（ready / wait を呼ばれないままの券。札で完了を判断する呼び手は、補助の券に
    // ready を呼ばない）をまとめて空ける。それでも空かないときだけ、いちばん古いものを待つ。
    for (int i = 0; i < UM_ASYNC_MAX; i++) {
      if (um_async[i].id != 0 && um_async[i].waiting == 0 && atomic_load(&um_async_fin[i]) == um_async[i].id) {
        um_async_release(i);
        um_gs_add(&um_gs_reclaimed, 1);
      }
    }
    if (um_async_nfree > 0) continue;
    if (atomic_load(&um_async_failed)) um_metal_fail("running the kernel", nil);
    int oldest = -1;
    for (int i = 0; i < UM_ASYNC_MAX; i++) {
      if (um_async[i].waiting == 0 && (oldest < 0 || um_async[i].id < um_async[oldest].id)) oldest = i;
    }
    if (oldest < 0) {
      // 全部の枠に待っている人がいる（1024 本のスレッドが同時に待っている）。少し待って見直す。
      pthread_mutex_unlock(&um_gpu_lock);
      usleep(50);
      pthread_mutex_lock(&um_gpu_lock);
      continue;
    }
    uint64_t t = um_async[oldest].id;
    um_async[oldest].waiting++;
    id<MTLCommandBuffer> cmd = um_async_cmd_of(oldest, t);
    um_gs_add(&um_gs_take_waits, 1);
    pthread_mutex_unlock(&um_gpu_lock);
    if (cmd) um_async_await(cmd, NULL);
    cmd = nil;
    pthread_mutex_lock(&um_gpu_lock);
    um_async[oldest].waiting--;
    if (um_async[oldest].id == t) um_async_release(oldest);
  }
}

// 領域が解放される前に呼ばれる（um_free から。ページ境界の領域だけ）。その領域を使っている GPU の仕事を片付ける。
static void um_on_free_inner(void *p);
static void um_on_free(void *p) {
  if (atomic_load(&um_gpu_regions) == 0 && atomic_load(&um_untracked_n) == 0) return;
  @autoreleasepool {  // um_gpu_wait の説明を参照
    um_on_free_inner(p);
  }
}
static void um_on_free_inner(void *p) {
  um_park_release(p);
  pthread_mutex_lock(&um_gpu_lock);
  NSNumber *key = @((uintptr_t)p);
  if (um_untracked && [um_untracked containsObject:key]) {
    [um_untracked removeObject:key];
    atomic_fetch_sub(&um_untracked_n, 1);
  }
  if (um_async_wrap && um_async_wrap[key]) {
    // この領域を使う提出が終わるまで待つ（鍵は放して待つ）。取っておいた GPU 側のバッファは、そのあと捨てる。
    for (;;) {
      int found = -1;
      for (int i = 0; i < UM_ASYNC_MAX && um_async_live > 0 && found < 0; i++) {
        if (um_async[i].id == 0 || um_async[i].done) continue;
        if (um_async[i].waiting == 0 && atomic_load(&um_async_fin[i]) == um_async[i].id) {
          um_async_release(i);  // 完了の印が付いている（もう終わっている）
          continue;
        }
        for (int b = 0; b < um_async[i].nbufs; b++) {
          if (um_async[i].data[b] == p) {
            found = i;
            break;
          }
        }
      }
      if (found < 0) break;
      uint64_t t = um_async[found].id;
      um_async[found].waiting++;
      id<MTLCommandBuffer> cmd = um_async_cmd_of(found, t);
      pthread_mutex_unlock(&um_gpu_lock);
      if (cmd) um_async_await(cmd, NULL);
      cmd = nil;
      pthread_mutex_lock(&um_gpu_lock);
      um_async[found].waiting--;
      if (um_async[found].id == t) um_async_release(found);
    }
    [um_async_wrap removeObjectForKey:key];
    [um_async_lastwrite removeObjectForKey:key];
    atomic_fetch_sub(&um_gpu_regions, 1);
  }
  // gpu.submit の常駐で、この領域を使う形の待機をやめる（解放済みの領域に GPU が書かないように）。
  um_ap_release(p);
  pthread_mutex_unlock(&um_gpu_lock);
}

// 枠 k の形で待機を始める。UM_PARK_AHEAD 個を合図待ちにする。うまくいけば state が 2 になる。
static void um_park_start(int k) {
  @autoreleasepool {
    um_slot_queue[k] = [um_device newCommandQueueWithMaxCommandBufferCount:UM_PARK_AHEAD * 2 + 8];
    um_slot_event[k] = [um_device newSharedEvent];
    um_slot_cmds[k] = [NSMutableArray array];
    if (!um_slot_queue[k] || !um_slot_event[k] || !um_awake_start()) return;
    size_t page = (size_t)getpagesize();
    for (int d = 0; d < um_slot[k].n; d++) {
      int32_t nbufs = um_slot[k].d[d].nbufs;
      int32_t block_bytes = um_slot[k].d[d].block_bytes;
      const uint64_t *table = um_slot[k].d[d].sig;
      id<MTLComputePipelineState> pipeline = um_pipeline(um_slot[k].d[d].kernel, um_slot[k].d[d].src, um_slot[k].d[d].src_len);
      NSUInteger width = um_resident_width(pipeline, um_slot[k].d[d].group);
      size_t header = (size_t)(2 + nbufs) * 4;
      id<MTLBuffer> params = [um_device newBufferWithLength:header + (size_t)block_bytes options:MTLResourceStorageModeShared];
      id<MTLBuffer> indirect = [um_device newBufferWithLength:12 options:MTLResourceStorageModeShared];
      if (width == 0 || !params || !indirect) {
        um_park_clear(k);
        return;
      }
      memset([params contents], 0, header + (size_t)block_bytes);
      memset([indirect contents], 0, 12);
      um_slot[k].d[d].width = width;
      um_slot_pipeline[k][d] = pipeline;
      um_slot_params[k][d] = params;
      um_slot_indirect[k][d] = indirect;
      for (int32_t i = 0; i < nbufs; i++) {
        void *data = (void *)(uintptr_t)table[i * 4];
        size_t len = (size_t)table[i * 4 + 1];
        size_t size = (len > 0 ? len : 1) * (size_t)table[i * 4 + 2];
        size_t rounded = (size + page - 1) / page * page;
        id<MTLBuffer> b = nil;
        // 同じ領域には、この形の中で必ず同じ GPU 側のバッファを使う（前の kernel の書き込みを次の kernel が待つように）。
        for (int d2 = 0; d2 <= d && !b; d2++) {
          int32_t limit = d2 == d ? i : um_slot[k].d[d2].nbufs;
          for (int32_t j = 0; j < limit; j++) {
            if (um_slot[k].d[d2].sig[j * 4] == table[i * 4]) {
              b = um_slot_buf[k][d2 * UM_PARK_BUFS + j];
              break;
            }
          }
        }
        if (!b) b = [um_device newBufferWithBytesNoCopy:data length:rounded options:MTLResourceStorageModeShared deallocator:nil];
        if (!b) {
          um_park_clear(k);
          return;
        }
        um_slot_buf[k][d * UM_PARK_BUFS + i] = b;
      }
    }
    // 検算する Buffer を決める: この形の dispatch が書く Buffer すべて（同じ領域は1回だけ）。
    // 多すぎるか、大きすぎる場合は検算を使わず、Metal の完了の知らせを待つ。
    um_check[k].n = 0;
    int usable = um_check_enabled && um_check_pipeline && um_check_width > 0;
    for (int d = 0; d < um_slot[k].n && usable; d++) {
      for (int32_t i = 0; i < um_slot[k].d[d].nbufs && usable; i++) {
        const uint64_t *e = &um_slot[k].d[d].sig[i * 4];
        if (e[3] == 0) continue;
        int seen = 0;
        for (int c = 0; c < um_check[k].n; c++) {
          if (um_slot[k].d[um_check[k].c[c].d].sig[um_check[k].c[c].i * 4] == e[0]) {
            // 同じ領域を別の dispatch も書く。どこまで書くか（個数）が違いうるので、全体を検算する。
            um_check[k].c[c].whole = 1;
            seen = 1;
          }
        }
        if (seen) continue;
        uint64_t max_words = (e[1] > 0 ? e[1] : 1) * (e[2] / 4);
        if (um_check[k].n == UM_PARK_CHECKS || e[2] % 4 != 0 || max_words > UM_CHECK_MAX_WORDS) {
          usable = 0;
          break;
        }
        int c = um_check[k].n++;
        um_check[k].c[c].d = d;
        um_check[k].c[c].i = i;
        um_check[k].c[c].elem_words = (uint32_t)(e[2] / 4);
        // グループの dispatch は、書く場所がスレッドの番号と揃うとは限らない（グループごとに1つ書く kernel など）ので、
        // 個数ぶんではなく全体を検算する（個数は全体のスレッド数で、Buffer の長さを超えうる）。
        um_check[k].c[c].whole = e[3] == 2 || um_slot[k].d[d].group != 0;
        um_check[k].c[c].words = 0;
        um_check_sums[k][c] = [um_device newBufferWithLength:(size_t)((max_words + 63) / 64 + um_check_width) * 8
                                                     options:MTLResourceStorageModeShared];
        um_check_params[k][c] = [um_device newBufferWithLength:16 options:MTLResourceStorageModeShared];
        um_check_indirect[k][c] = [um_device newBufferWithLength:12 options:MTLResourceStorageModeShared];
        if (!um_check_sums[k][c] || !um_check_params[k][c] || !um_check_indirect[k][c]) {
          usable = 0;
          break;
        }
        memset([um_check_sums[k][c] contents], 0, [um_check_sums[k][c] length]);
        um_check[k].c[c].sums = [um_check_sums[k][c] contents];
        memset([um_check_params[k][c] contents], 0, 16);
        memset([um_check_indirect[k][c] contents], 0, 12);
      }
    }
    if (!usable) {
      for (int c = 0; c < UM_PARK_CHECKS; c++) {
        um_check_sums[k][c] = nil;
        um_check_params[k][c] = nil;
        um_check_indirect[k][c] = nil;
      }
      um_check[k].n = 0;
    }
    um_slot[k].committed = 0;
    um_slot[k].fired = 0;
    um_slot[k].runs = 0;
    um_slot[k].awake = 0;
    um_slot[k].last_fire = CFAbsoluteTimeGetCurrent();
    um_park_state(k, 2);
    for (int i = 0; i < UM_PARK_AHEAD; i++) um_park_commit_one(k);
    if (um_trace) fprintf(stderr, "gpu: resident started for kernel %d (%d in a row)\n", um_slot[k].d[0].kernel, um_slot[k].n);
  }
}

// 枠 k で待機している kernel に仕事を渡す: 引数と個数を共有メモリに書き、合図を進める。
static void um_park_fire(int k, const um_job *jobs) {
  @autoreleasepool {
    // 前回、検算で先に進んだ命令が失敗していなかったかを、ここで確かめる（知らせはもう届いている頃）。
    if (um_park_unchecked && [um_park_unchecked status] >= MTLCommandBufferStatusCompleted) um_park_settle();
    um_park_started = CFAbsoluteTimeGetCurrent();
    um_slot[k].last_fire = um_park_started;
    if (um_trace) um_park_started_gpu = um_gpu_clock();
    atomic_store(&um_awake_until, um_park_started + 0.05);
    // 起こしておく kernel の長さ。試している最中は順に両方、そのあとは選んだほう。
    uint32_t phase = um_slot[k].runs % UM_AWAKE_RETRY;
    int which = phase < UM_AWAKE_TRIAL ? 0 : phase < UM_AWAKE_TRIAL * 2 ? 1 : um_slot[k].awake;
    if (um_awake_len < 0) ((uint32_t *)[um_awake_buf contents])[2] = um_awake_lengths[which];
    for (int d = 0; d < um_slot[k].n; d++) {
      const um_job *j = &jobs[d];
      uint32_t *words = (uint32_t *)[um_slot_params[k][d] contents];
      um_resident_grid(j->count, um_slot[k].d[d].group, um_slot[k].d[d].width, words,
                       (uint32_t *)[um_slot_indirect[k][d] contents]);
      for (int32_t i = 0; i < j->nbufs; i++) words[2 + i] = (uint32_t)j->table[i * 4 + 1];
      if (j->block_bytes > 0) memcpy((char *)words + (size_t)(2 + j->nbufs) * 4, j->block, (size_t)j->block_bytes);
      um_park_counts[d] = j->count;
    }
    // 検算の範囲と依頼の番号。
    if (um_check[k].n > 0) {
      um_check_ticket++;
      for (int c = 0; c < um_check[k].n; c++) {
        const um_job *j = &jobs[um_check[k].c[c].d];
        uint64_t elems = um_check[k].c[c].whole ? j->table[um_check[k].c[c].i * 4 + 1] : j->count;
        uint32_t words = (uint32_t)(elems * um_check[k].c[c].elem_words);
        um_check[k].c[c].words = words;
        uint32_t *p = (uint32_t *)[um_check_params[k][c] contents];
        p[0] = (uint32_t)um_check_ticket;
        p[1] = words;
        p[2] = (uint32_t)(um_check_ticket >> 32);
        uint32_t *groups = (uint32_t *)[um_check_indirect[k][c] contents];
        groups[0] = (uint32_t)(((words + 63) / 64 + um_check_width - 1) / um_check_width);
        groups[1] = 1;
        groups[2] = 1;
      }
    }
    um_park_running = um_slot_cmds[k][0];
    um_park_running_slot = k;
    [um_slot_cmds[k] removeObjectAtIndex:0];
    um_slot[k].fired++;
    um_slot_event[k].signaledValue = um_slot[k].fired;
    um_mq_kick();
    // GPU が走っているあいだに、待機分を1つ補充する。
    um_park_commit_one(k);
  }
}

static int um_park_same(int k, const um_job *jobs, int n) {
  if (um_slot[k].state == 0 || um_slot[k].n != n) return 0;
  for (int d = 0; d < n; d++) {
    if (um_slot[k].d[d].kernel != jobs[d].kernel || um_slot[k].d[d].src != jobs[d].src || um_slot[k].d[d].group != jobs[d].group ||
        um_slot[k].d[d].nbufs != jobs[d].nbufs || um_slot[k].d[d].block_bytes != jobs[d].block_bytes ||
        memcmp(um_slot[k].d[d].sig, jobs[d].table, (size_t)jobs[d].nbufs * 4 * sizeof(uint64_t)) != 0) {
      return 0;
    }
  }
  return 1;
}

// 積んである dispatch の並びを常駐の経路で実行できるなら、実行して終わるまで待ち、1 を返す。
// できなければ、形の回数を数えて 0 を返す（呼び出し側が普通に提出する）。
static int um_park_try_locked(const um_job *jobs, int n);
static int um_park_try(const um_job *jobs, int n) {
  // ほかのスレッドが常駐の経路を使っている最中（GPU の結果を待っている）なら、待たずに普通に提出する。
  if (pthread_mutex_trylock(&um_park_lock) != 0) return 0;
  int done = um_park_try_locked(jobs, n);
  pthread_mutex_unlock(&um_park_lock);
  return done;
}
static int um_park_try_locked(const um_job *jobs, int n) {
  um_park_tick++;
  int k = -1;
  for (int i = 0; i < UM_PARK_SLOTS; i++) {
    if (um_park_same(i, jobs, n)) {
      k = i;
      break;
    }
  }
  if (k < 0) {
    // 初めて見る形。空きの枠か、いちばん長く使われていない枠に入れる。待機中の枠を追い出すのは、
    // しばらく（256 回ぶん）使われていないときだけにする。
    for (int i = 0; i < UM_PARK_SLOTS; i++) {
      if (um_slot[i].state == 0) {
        k = i;
        break;
      }
      if (um_slot[i].state == 2 && um_park_tick - um_slot[i].last_use < 256) continue;
      if (k < 0 || um_slot[i].last_use < um_slot[k].last_use) k = i;
    }
    if (k < 0) return 0;
    um_park_stop(k);
    um_park_state(k, 1);
    um_slot[k].hits = 0;
    um_slot[k].n = n;
    for (int d = 0; d < n; d++) {
      um_slot[k].d[d].kernel = jobs[d].kernel;
      um_slot[k].d[d].src = jobs[d].src;
      um_slot[k].d[d].src_len = jobs[d].src_len;
      um_slot[k].d[d].group = jobs[d].group;
      um_slot[k].d[d].nbufs = jobs[d].nbufs;
      um_slot[k].d[d].block_bytes = jobs[d].block_bytes;
      memcpy(um_slot[k].d[d].sig, jobs[d].table, (size_t)jobs[d].nbufs * 4 * sizeof(uint64_t));
    }
  }
  um_slot[k].last_use = um_park_tick;
  if (um_slot[k].state == 1) {
    if (++um_slot[k].hits < UM_PARK_AFTER) return 0;
    um_slot[k].hits = 0;  // 始められなかった場合は、また数え直す
    um_park_start(k);
    if (um_slot[k].state != 2) return 0;
  }
  um_park_fire(k, jobs);
  um_park_wait();
  return 1;
}

// 自動判断。GPU が使えなければ CPU。
// それ以外は、この kernel を CPU と GPU で実行したときの実測から、速いと見込まれるほうを選ぶ。
// 先に CPU で2回測り、GPU に渡す価値がある大きさ（UM_GPU_MIN_SECONDS）なら GPU でも2回試す。CPU と GPU の結果は同じなので、どちらを選んでも答えは変わらない。
int32_t um_gpu_pick(int32_t kernel, uint32_t count) {
  um_env_init();
  if (um_gpu_off) return 0;
  if (um_fixed_rule || kernel < 0 || kernel >= UM_MAX_TRACKED) {
    if (count >= um_threshold && um_metal_init()) return 1;
    um_gpu_flush();
    return 0;
  }
  pthread_mutex_lock(&um_table_lock);
  um_stat *s = &um_stats[kernel];
  int gpu;
  double cpu_time = s->cpu_per_elem * count;
  if (s->cpu_runs < 2) gpu = 0;
  else if (cpu_time < UM_GPU_MIN_SECONDS) gpu = 0;
  else if (s->gpu_runs < 2) gpu = 1;
  else {
    double gpu_time = s->gpu_fixed + s->gpu_per_elem * count;
    gpu = gpu_time < cpu_time;
    // 負けている側を測り直す間隔。差が大きいほど広げ（遅い側を試す損を、全体の 1/64 程度に抑える）、
    // 測り直しても勝ち負けが変わらないあいだは、そのたびに倍にする（最大 64 倍）。勝ち負けが入れ替わったら戻す。
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
    }
  }
  pthread_mutex_unlock(&um_table_lock);
  // GPU の準備は、初めて GPU を使おうとしたときにする。
  if (gpu && !um_metal_init()) gpu = 0;
  if (!gpu) {
    um_gpu_flush();  // CPU で実行する前に、積んである GPU の仕事を終わらせる（同じ Buffer を使うかもしれない）
    um_pending_kernel = kernel;
    um_pending_start = CFAbsoluteTimeGetCurrent();
  }
  return gpu;
}

// 自動判断で CPU が選ばれた dispatch が終わったときに、生成コードから呼ばれる。
void um_gpu_cpu_done(int32_t kernel, uint32_t count) {
  if (kernel != um_pending_kernel || count == 0) return;
  um_pending_kernel = -1;
  um_stat *s = &um_stats[kernel];
  double elapsed = CFAbsoluteTimeGetCurrent() - um_pending_start;
  pthread_mutex_lock(&um_table_lock);
  um_blend(&s->cpu_per_elem, elapsed / count, s->cpu_runs);
  if (s->cpu_runs < 2) s->cpu_runs++;
  pthread_mutex_unlock(&um_table_lock);
  if (um_trace) fprintf(stderr, "cpu: kernel %d count %u %.3f ms\n", kernel, count, elapsed * 1000.0);
}

static id<MTLComputePipelineState> um_pipeline_locked(int32_t kernel, const char *src, int32_t src_len);
static id<MTLComputePipelineState> um_pipeline(int32_t kernel, const char *src, int32_t src_len) {
  pthread_mutex_lock(&um_table_lock);
  id<MTLComputePipelineState> p = um_pipeline_locked(kernel, src, src_len);
  pthread_mutex_unlock(&um_table_lock);
  return p;
}
static id<MTLComputePipelineState> um_pipeline_locked(int32_t kernel, const char *src, int32_t src_len) {
  NSNumber *key = @(kernel);
  id<MTLComputePipelineState> p = um_pipelines[key];
  if (p) return p;
  NSString *code = [[NSString alloc] initWithBytes:src length:(NSUInteger)src_len encoding:NSUTF8StringEncoding];
  if (!code) um_metal_fail("the shader source is not valid UTF-8", nil);
  MTLCompileOptions *options = [MTLCompileOptions new];
  // CPU と同じ結果にするため、浮動小数点の近似を切る。
  if (@available(macOS 15.0, *)) {
    options.mathMode = MTLMathModeSafe;
    options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
  } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    options.fastMathEnabled = NO;
#pragma clang diagnostic pop
  }
  NSError *error = nil;
  id<MTLLibrary> library = [um_device newLibraryWithSource:code options:options error:&error];
  if (!library) um_metal_fail("compiling the shader", error);
  id<MTLFunction> fn = [library newFunctionWithName:@"um_main"];
  if (!fn) um_metal_fail("the shader has no um_main", nil);
  p = [um_device newComputePipelineStateWithFunction:fn error:&error];
  if (!p) um_metal_fail("creating the pipeline", error);
  um_pipelines[key] = p;
  return p;
}
// 同じ kernel の、Buffer をポインタの表から受け取る入口（um_main_bl。src/msl.ts）のパイプライン。なければ nil
// （Buffer の引数がない kernel、古いコンパイラの出力）。
static NSMutableDictionary<NSNumber *, id> *um_pipelines_bl;
static id<MTLComputePipelineState> um_pipeline_bl(int32_t kernel, const char *src, int32_t src_len) {
  pthread_mutex_lock(&um_table_lock);
  if (!um_pipelines_bl) um_pipelines_bl = [NSMutableDictionary dictionary];
  NSNumber *key = @(kernel);
  id cached = um_pipelines_bl[key];
  if (!cached) {
    cached = [NSNull null];
    NSString *code = [[NSString alloc] initWithBytes:src length:(NSUInteger)src_len encoding:NSUTF8StringEncoding];
    MTLCompileOptions *options = [MTLCompileOptions new];
    if (@available(macOS 15.0, *)) {
      options.mathMode = MTLMathModeSafe;
      options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
    } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
      options.fastMathEnabled = NO;
#pragma clang diagnostic pop
    }
    NSError *error = nil;
    id<MTLLibrary> library = code ? [um_device newLibraryWithSource:code options:options error:&error] : nil;
    id<MTLFunction> fn = library ? [library newFunctionWithName:@"um_main_bl"] : nil;
    id<MTLComputePipelineState> p = fn ? [um_device newComputePipelineStateWithFunction:fn error:&error] : nil;
    if (p) cached = p;
    um_pipelines_bl[key] = cached;
  }
  pthread_mutex_unlock(&um_table_lock);
  return cached == [NSNull null] ? nil : (id<MTLComputePipelineState>)cached;
}

// dispatch の並びを1つの命令にして GPU に提出し、終わるまで待つ（常駐を使わない、普通の経路）。
// Buffer はすべてコピーなしで渡せるものに限る。
static void um_submit(const um_job *jobs, int n) {
  @autoreleasepool {
    CFAbsoluteTime started = CFAbsoluteTimeGetCurrent();
    size_t page = (size_t)getpagesize();
    id<MTLCommandBuffer> cmd = [um_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) um_metal_fail("creating the command buffer", nil);
    // この提出の中でコピーなしで渡した領域（場所 → GPU 側のバッファ）。同じ領域には必ず同じバッファを使う。
    // 別々のバッファとして渡すと、Metal は同じメモリだと分からず、前の kernel の書き込みを待たずに次を実行してしまう。
    NSMutableDictionary<NSNumber *, id<MTLBuffer>> *wrapped = [NSMutableDictionary dictionary];
    for (int d = 0; d < n; d++) {
      const um_job *j = &jobs[d];
      id<MTLComputePipelineState> pipeline = um_pipeline(j->kernel, j->src, j->src_len);
      um_grid grid = um_grid_of(j->count, j->group);

      // 引数の塊。
      size_t header = (size_t)(2 + j->nbufs) * 4;
      id<MTLBuffer> params = [um_device newBufferWithLength:header + (size_t)j->block_bytes options:MTLResourceStorageModeShared];
      if (!params) um_metal_fail("allocating the parameter block", nil);
      uint32_t *words = (uint32_t *)[params contents];
      words[0] = grid.n;
      words[1] = grid.row;
      for (int32_t i = 0; i < j->nbufs; i++) words[2 + i] = (uint32_t)j->table[i * 4 + 1];
      if (j->block_bytes > 0) memcpy((char *)[params contents] + header, j->block, (size_t)j->block_bytes);

      [enc setComputePipelineState:pipeline];
      [enc setBuffer:params offset:0 atIndex:0];
      for (int32_t i = 0; i < j->nbufs; i++) {
        void *data = (void *)(uintptr_t)j->table[i * 4];
        size_t len = (size_t)j->table[i * 4 + 1];
        // 長さ 0 でも1要素ぶん確保してある（添字の切り詰め先を常に有効にするため。CPU 側と同じ）。
        size_t size = (len > 0 ? len : 1) * (size_t)j->table[i * 4 + 2];
        // Buffer の領域はページ境界に揃えてページ単位で確保してある（um_alloc_shared）ので、
        // コピーせずにそのまま GPU に渡す。ユニファイドメモリでは CPU と GPU が同じ場所を見る。
        size_t rounded = (size + page - 1) / page * page;
        NSNumber *key = @((uintptr_t)data);
        id<MTLBuffer> b = wrapped[key];
        if (!b) {
          b = [um_device newBufferWithBytesNoCopy:data length:rounded options:MTLResourceStorageModeShared deallocator:nil];
          if (!b) um_metal_fail("sharing a buffer with the GPU", nil);
          wrapped[key] = b;
        }
        [enc setBuffer:b offset:0 atIndex:(NSUInteger)(i + 1)];
      }
      um_encode_grid(enc, pipeline, grid);
    }
    [enc endEncoding];
    [cmd commit];
    um_mq_kick();
    [cmd waitUntilCompleted];
    if ([cmd status] != MTLCommandBufferStatusCompleted) um_metal_fail("running the kernel", [cmd error]);

    // 自動判断のための実測。GPU が計算していた時間を要素数で割り、残りを固定の手間として dispatch の数で分ける。
    double total = CFAbsoluteTimeGetCurrent() - started;
    double busy = [cmd GPUEndTime] - [cmd GPUStartTime];
    if (busy < 0 || busy > total) busy = total;
    double elems = 0;
    for (int d = 0; d < n; d++) elems += jobs[d].count;
    for (int d = 0; d < n; d++) {
      int32_t kernel = jobs[d].kernel;
      if (kernel >= 0 && kernel < UM_MAX_TRACKED) {
        pthread_mutex_lock(&um_table_lock);
        um_stat *s = &um_stats[kernel];
        um_blend(&s->gpu_per_elem, busy / elems, s->gpu_runs);
        um_blend(&s->gpu_fixed, (total - busy) / n, s->gpu_runs);
        if (s->gpu_runs < 2) s->gpu_runs++;
        pthread_mutex_unlock(&um_table_lock);
      }
      if (um_trace) {
        fprintf(stderr, "gpu: kernel %d count %u shared %d copied 0\n", kernel, jobs[d].count, jobs[d].nbufs);
      }
    }
    // 全体の時間と、そのうち GPU が実際に計算していた時間。
    if (um_trace) fprintf(stderr, "gpu: submitted %d kernel(s) %.3f ms (gpu %.3f)\n", n, total * 1000.0, busy * 1000.0);
  }
}

// コピーなしで渡せない Buffer がある dispatch（または積んでおけない大きさの dispatch）を、1つだけで実行する。
// 渡せない Buffer は写して実行し、終わったら写し戻す。
static void um_run_alone(int32_t kernel, const char *src, int32_t src_len, uint32_t count, uint32_t group, const void *block,
                         int32_t block_bytes, const uint64_t *table, int32_t nbufs) {
  @autoreleasepool {
    CFAbsoluteTime started = CFAbsoluteTimeGetCurrent();
    size_t page = (size_t)getpagesize();
    id<MTLComputePipelineState> pipeline = um_pipeline(kernel, src, src_len);
    um_grid grid = um_grid_of(count, group);
    size_t header = (size_t)(2 + nbufs) * 4;
    id<MTLBuffer> params = [um_device newBufferWithLength:header + (size_t)block_bytes options:MTLResourceStorageModeShared];
    if (!params) um_metal_fail("allocating the parameter block", nil);
    uint32_t *words = (uint32_t *)[params contents];
    words[0] = grid.n;
    words[1] = grid.row;
    for (int32_t i = 0; i < nbufs; i++) words[2 + i] = (uint32_t)table[i * 4 + 1];
    if (block_bytes > 0) memcpy((char *)[params contents] + header, block, (size_t)block_bytes);

    id<MTLCommandBuffer> cmd = [um_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) um_metal_fail("creating the command buffer", nil);
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:params offset:0 atIndex:0];
    NSMutableArray<id<MTLBuffer>> *buffers = [NSMutableArray arrayWithCapacity:(NSUInteger)nbufs];
    char shared[64] = {0};  // コピーなしで渡した Buffer
    int n_shared = 0;
    for (int32_t i = 0; i < nbufs; i++) {
      void *data = (void *)(uintptr_t)table[i * 4];
      size_t len = (size_t)table[i * 4 + 1];
      size_t elem = (size_t)table[i * 4 + 2];
      size_t used = len * elem;
      size_t size = (len > 0 ? len : 1) * elem;
      size_t rounded = (size + page - 1) / page * page;
      id<MTLBuffer> b = nil;
      if (um_zero_copy && data && ((uintptr_t)data % page) == 0) {
        // 同じ領域が2回渡されていれば、同じ GPU 側のバッファを使う。
        for (int32_t j = 0; j < i; j++) {
          if (shared[j] && table[j * 4] == table[i * 4]) b = buffers[(NSUInteger)j];
        }
        if (!b) b = [um_device newBufferWithBytesNoCopy:data length:rounded options:MTLResourceStorageModeShared deallocator:nil];
      }
      if (b) {
        shared[i] = 1;
        n_shared++;
      } else {
        b = [um_device newBufferWithLength:size options:MTLResourceStorageModeShared];
        if (!b) um_metal_fail("allocating a buffer", nil);
        if (used > 0) memcpy([b contents], data, used);
        else memset([b contents], 0, size);
      }
      [buffers addObject:b];
      [enc setBuffer:b offset:0 atIndex:(NSUInteger)(i + 1)];
    }
    um_encode_grid(enc, pipeline, grid);
    [enc endEncoding];
    [cmd commit];
    um_mq_kick();
    [cmd waitUntilCompleted];
    if ([cmd status] != MTLCommandBufferStatusCompleted) um_metal_fail("running the kernel", [cmd error]);
    for (int32_t i = 0; i < nbufs; i++) {
      if (table[i * 4 + 3] == 0 || shared[i]) continue;
      size_t used = (size_t)table[i * 4 + 1] * (size_t)table[i * 4 + 2];
      if (used > 0) memcpy((void *)(uintptr_t)table[i * 4], [buffers[(NSUInteger)i] contents], used);
    }

    double total = CFAbsoluteTimeGetCurrent() - started;
    double busy = [cmd GPUEndTime] - [cmd GPUStartTime];
    if (busy < 0 || busy > total) busy = total;
    if (kernel >= 0 && kernel < UM_MAX_TRACKED) {
      pthread_mutex_lock(&um_table_lock);
      um_stat *s = &um_stats[kernel];
      um_blend(&s->gpu_per_elem, busy / count, s->gpu_runs);
      um_blend(&s->gpu_fixed, total - busy, s->gpu_runs);
      if (s->gpu_runs < 2) s->gpu_runs++;
      pthread_mutex_unlock(&um_table_lock);
    }
    if (um_trace) {
      fprintf(stderr, "gpu: kernel %d count %u shared %d copied %d\n", kernel, count, n_shared, nbufs - n_shared);
      fprintf(stderr, "gpu: submitted 1 kernel(s) %.3f ms (gpu %.3f)\n", total * 1000.0, busy * 1000.0);
    }
  }
}

// 積んである dispatch をまとめて実行し、終わるまで待つ。積んでいなければ何もしない。
// 同じ並びが待機していれば（または待機を始められれば）常駐の経路で、そうでなければ普通に提出する。
void um_gpu_flush(void) {
  int n = um_pending_len;
  if (n == 0) return;
  um_pending_len = 0;
  @autoreleasepool {  // um_park_wait の待ちも含める（um_gpu_wait の説明を参照）
    // グループの dispatch も常駐の経路に乗る（形にグループの大きさを含める。um_resident_grid）。
    if (um_resident && n <= UM_PARK_SEQ && um_park_try(um_pending, n)) return;
    um_submit(um_pending, n);
  }
}

// dispatch を1つ積む。実行と完了待ちは um_gpu_flush で行う（生成コードが、次の文が dispatch でなければすぐ呼ぶ）。
void um_gpu_dispatch(int32_t kernel, const char *src, int32_t src_len, uint32_t count, const void *block,
                     int32_t block_bytes, const uint64_t *table, int32_t nbufs) {
  uint32_t group = um_take_group();
  if (count == 0) return;
  if (!um_metal_init()) um_metal_fail("no GPU", nil);
  if (nbufs > 64) um_metal_fail("too many buffers", nil);
  // コピーして渡す Buffer があるか、積んでおける大きさを超える dispatch は、積んである仕事を先に終わらせてから
  // 1つだけで実行する（前の dispatch が同じ Buffer に書いているかもしれない）。
  size_t page = (size_t)getpagesize();
  int copies = 0;
  for (int32_t i = 0; i < nbufs; i++) {
    void *data = (void *)(uintptr_t)table[i * 4];
    if (!um_zero_copy || !data || ((uintptr_t)data % page) != 0) copies++;
  }
  if (copies > 0 || nbufs > UM_BATCH_BUFS || block_bytes > UM_BATCH_BLOCK) {
    um_gpu_flush();
    um_run_alone(kernel, src, src_len, count, group, block, block_bytes, table, nbufs);
    return;
  }
  if (um_pending_len == UM_MAX_BATCH) um_gpu_flush();
  um_job *j = &um_pending[um_pending_len++];
  j->kernel = kernel;
  j->src = src;
  j->src_len = src_len;
  j->count = count;
  j->group = group;
  j->nbufs = nbufs;
  j->block_bytes = block_bytes;
  memcpy(j->table, table, (size_t)nbufs * 4 * sizeof(uint64_t));
  if (block_bytes > 0) memcpy(j->block, block, (size_t)block_bytes);
}

// 列をまたぐ安全網: この仕事が使う領域を、別の列のまだ終わっていない仕事が書いていれば、それが終わるのを待つ
// （列のあいだでは、Metal は前の命令の書き込みを待たない）。列の番号 q は 0 / 1 が普通の列、2 + k が
// gpu.submit の常駐の枠 k の列（um_ap_queue）。控えは (券 << 8) | q。um_gpu_lock を持って呼ぶ（待つあいだは放す）。
// 1 つの領域をいつも同じ列で使う呼び手（探索の評価など）では起きない。読むだけの仕事が別の列で走っているところに
// 書く場合（読みのあとの書き込み）は見ない。
static void um_async_cross_wait(const uint64_t *table, int32_t nbufs, int q) {
  if (!um_async_lastwrite) return;
  for (;;) {
    int found = -1;
    uint64_t t = 0;
    for (int32_t i = 0; i < nbufs && found < 0; i++) {
      NSNumber *key = @((uintptr_t)table[i * 4]);
      // 追跡しない領域は待たない（同時に走る仕事どうしで重ならないことを、呼ぶ側が守る。um_untracked）。
      if (um_is_untracked((const void *)(uintptr_t)table[i * 4])) continue;
      NSNumber *v = um_async_lastwrite[key];
      if (!v) continue;
      uint64_t e = [v unsignedLongLongValue];
      if ((int)(e & 0xff) == q) continue;
      uint64_t tk = e >> 8;
      int j = um_async_find(tk);
      if (j < 0 || um_async[j].done || atomic_load(&um_async_fin[j]) == tk) {
        [um_async_lastwrite removeObjectForKey:key];
        continue;
      }
      found = j;
      t = tk;
    }
    if (found < 0) return;
    um_gs_add(&um_gs_cross_waits, 1);
    um_async[found].waiting++;
    id<MTLCommandBuffer> cmd = um_async_cmd_of(found, t);
    pthread_mutex_unlock(&um_gpu_lock);
    if (cmd) um_async_await(cmd, NULL);
    cmd = nil;
    pthread_mutex_lock(&um_gpu_lock);
    um_async[found].waiting--;
    if (um_async[found].id == t) um_async_release(found);
  }
}

// `gpu.submit`: kernel の実行を GPU に頼み、終わるのを待たずに戻る。券を返す（0 は「もう終わっている」）。
// スレッドの印（gpu.flags）で、検算をしない（NOCHECK）、急ぎの列に積む（URGENT）、提出を見送る（HOLD）を選べる。
static uint64_t um_gpu_submit_inner(int32_t kernel, const char *src, int32_t src_len, uint32_t count, uint32_t group,
                                    const void *block, int32_t block_bytes, const uint64_t *table, int32_t nbufs);
uint64_t um_gpu_submit(int32_t kernel, const char *src, int32_t src_len, uint32_t count, const void *block,
                       int32_t block_bytes, const uint64_t *table, int32_t nbufs) {
  uint32_t group = um_take_group();
  @autoreleasepool {  // um_gpu_wait の説明を参照
    return um_gpu_submit_inner(kernel, src, src_len, count, group, block, block_bytes, table, nbufs);
  }
}
// 実際に使う列（um_gpu_lock を持って呼ぶ）。URGENT なら急ぎの列（最初に使うときに作る。作れなければ今までの列）。
static int um_async_queue_for(uint32_t eff) {
  if (!(eff & UM_SUBMIT_URGENT)) return 0;
  if (!um_queue_urgent) um_queue_urgent = [um_device newCommandQueue];
  return um_queue_urgent ? 1 : 0;
}

// 表の枠 slot に券 ticket を設定する（um_gpu_lock を持って呼ぶ）。命令（um_async_cmd）はまだ空。
static void um_async_open_slot(int slot, uint64_t ticket, const uint64_t *table, int32_t nbufs) {
  um_async[slot].id = ticket;
  um_async[slot].nbufs = nbufs;
  for (int32_t i = 0; i < nbufs; i++) um_async[slot].data[i] = (void *)(uintptr_t)table[i * 4];
  um_async[slot].nchecks = 0;
  um_async[slot].waiting = 0;
  um_async[slot].done = 0;
  um_async[slot].chain = NULL;
  um_async_cmd[slot] = nil;
  um_async_live++;
}

// 券 ticket（表の枠 slot。um_async_open_slot で設定済み）の仕事を、列 q のためている命令に入れ、提出するかどうかを
// 決める（今までの gpu.submit の経路）。um_gpu_lock を持って呼ぶ（別の列の書き込みを待つあいだは放す）。
// checking なら、書く Buffer の検算の kernel も積む。
static void um_async_encode(int32_t kernel, const char *src, int32_t src_len, uint32_t count, uint32_t group, const void *block,
                            int32_t block_bytes, const uint64_t *table, int32_t nbufs, int slot, uint64_t ticket, int q,
                            int hold, int checking) {
  size_t page = (size_t)getpagesize();
  id<MTLComputePipelineState> pipeline = um_pipeline(kernel, src, src_len);
  um_grid grid = um_grid_of(count, group);
  size_t header = (size_t)(2 + nbufs) * 4;
  um_async_cross_wait(table, nbufs, q);
  // この枠の前の提出は終わっているので、引数の領域は使い回せる。
  id<MTLBuffer> params = um_async_params[slot];
  if (!params || [params length] < header + (size_t)block_bytes) {
    size_t want = header + (size_t)block_bytes;
    params = [um_device newBufferWithLength:(want < 256 ? 256 : want) options:MTLResourceStorageModeShared];
    if (!params) um_metal_fail("allocating the parameter block", nil);
    um_async_params[slot] = params;
  }
  uint32_t *words = (uint32_t *)[params contents];
  words[0] = grid.n;
  words[1] = grid.row;
  for (int32_t i = 0; i < nbufs; i++) words[2 + i] = (uint32_t)table[i * 4 + 1];
  if (block_bytes > 0) memcpy((char *)[params contents] + header, block, (size_t)block_bytes);

  if (!um_async_wrap) um_async_wrap = [NSMutableDictionary dictionary];
  id<MTLBuffer> made[UM_BATCH_BUFS];
  for (int32_t i = 0; i < nbufs; i++) {
    void *data = (void *)(uintptr_t)table[i * 4];
    size_t len = (size_t)table[i * 4 + 1];
    size_t size = (len > 0 ? len : 1) * (size_t)table[i * 4 + 2];
    size_t rounded = (size + page - 1) / page * page;
    // 同じ領域には必ず同じ GPU 側のバッファを使う（1つの命令にまとめたとき、前の仕事の書き込みを次が待つように）。
    NSNumber *key = @((uintptr_t)data);
    id<MTLBuffer> b = um_async_wrap[key];
    if (!um_wrap_ok(b, rounded, data)) {
      b = [um_device newBufferWithBytesNoCopy:data length:rounded options:um_wrap_options(data) deallocator:nil];
      if (!b) um_metal_fail("sharing a buffer with the GPU", nil);
      if (!um_async_wrap[key]) atomic_fetch_add(&um_gpu_regions, 1);
      um_async_wrap[key] = b;
    }
    made[i] = b;
  }

  // 検算の用意: この仕事が書く Buffer（同じ領域は1回だけ）。多すぎる・大きすぎる場合は使わない。
  int nchecks = 0, usable = checking;
  id<MTLBuffer> check_data[UM_PARK_CHECKS];
  for (int32_t i = 0; i < nbufs && usable; i++) {
    if (table[i * 4 + 3] == 0) continue;
    int seen = 0;
    for (int c = 0; c < nchecks; c++) seen |= um_async[slot].check_data[c] == (const uint32_t *)(uintptr_t)table[i * 4];
    if (seen) continue;
    // 4語目が 2（RacyBuffer）なら、どこにでも書きうるので全体を検算する。
    uint64_t elems = table[i * 4 + 3] == 2 ? table[i * 4 + 1] : count;
    uint64_t words64 = elems * (table[i * 4 + 2] / 4);
    if (nchecks == UM_PARK_CHECKS || table[i * 4 + 2] % 4 != 0 || words64 == 0 || words64 > UM_CHECK_MAX_WORDS) {
      usable = 0;
      break;
    }
    size_t need = (size_t)((words64 + 63) / 64) * 8;
    if (!um_async_sums[slot][nchecks] || [um_async_sums[slot][nchecks] length] < need) {
      um_async_sums[slot][nchecks] = [um_device newBufferWithLength:(need < 256 ? 256 : need) options:MTLResourceStorageModeShared];
    }
    if (!um_async_check_params[slot][nchecks]) {
      um_async_check_params[slot][nchecks] = [um_device newBufferWithLength:16 options:MTLResourceStorageModeShared];
    }
    if (!um_async_sums[slot][nchecks] || !um_async_check_params[slot][nchecks]) {
      usable = 0;
      break;
    }
    uint32_t *cp = (uint32_t *)[um_async_check_params[slot][nchecks] contents];
    cp[0] = (uint32_t)ticket;
    cp[1] = (uint32_t)words64;
    cp[2] = (uint32_t)(ticket >> 32);
    um_async[slot].check_data[nchecks] = (const uint32_t *)(uintptr_t)table[i * 4];
    um_async[slot].check_words[nchecks] = (uint32_t)words64;
    um_async[slot].check_sums[nchecks] = [um_async_sums[slot][nchecks] contents];
    check_data[nchecks] = made[i];
    nchecks++;
  }
  um_async[slot].nchecks = usable ? nchecks : 0;

  pthread_mutex_lock(&um_async_lock);
  while (!um_open_cmd[q]) {
    // 命令は um_async_lock の外で作る。キューの「終わっていない命令」が上限（64 個）に達していると、作る呼び出しは
    // 先の命令が終わるまで待つ。終わりの処理（um_async_commit_open の完了の処理）は um_async_lock を取るので、
    // 鍵を持ったまま待つと、どちらも進まなくなる（多くのスレッドが投げ続けて、ためた命令が UM_OPEN_MAX ごとに
    // 提出され続けたときに起きた）。命令を作るのは um_gpu_lock を持つこの経路だけなので、鍵を取り直したあとも
    // ふつうは空のまま。完了の処理が提出して空にした場合も、ここで作り直す。
    pthread_mutex_unlock(&um_async_lock);
    id<MTLCommandBuffer> fresh = [(q == 1 ? um_queue_urgent : um_queue) commandBuffer];
    if (!fresh) um_metal_fail("creating the command buffer", nil);
    pthread_mutex_lock(&um_async_lock);
    if (!um_open_cmd[q]) {
      um_open_cmd[q] = fresh;
      um_open_enc[q] = [fresh computeCommandEncoder];
      if (!um_open_enc[q]) um_metal_fail("creating the command buffer", nil);
      um_open_n[q] = 0;
      um_open_held[q] = 0;
      um_open_fins[q].n = 0;
    }
  }
  id<MTLComputeCommandEncoder> enc = um_open_enc[q];
  // 追跡しない Buffer を使う仕事は、同じ命令の前の dispatch の書き込みを Metal が待たないので、ここで待たせる。
  int untracked = 0;
  for (int32_t i = 0; i < nbufs; i++) untracked |= [made[i] hazardTrackingMode] == MTLHazardTrackingModeUntracked;
  if (untracked && um_open_n[q] > 0) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
  [enc setComputePipelineState:pipeline];
  [enc setBuffer:params offset:0 atIndex:0];
  for (int32_t i = 0; i < nbufs; i++) [enc setBuffer:made[i] offset:0 atIndex:(NSUInteger)(i + 1)];
  um_encode_grid(enc, pipeline, grid);
  if (untracked && um_async[slot].nchecks > 0) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
  // 検算。仕事が書く Buffer と同じ GPU 側のバッファを読むので、仕事のあとに実行される。
  for (int c = 0; c < um_async[slot].nchecks; c++) {
    NSUInteger groups = (um_async[slot].check_words[c] + 63) / 64;
    [enc setComputePipelineState:um_check_pipeline];
    [enc setBuffer:um_async_sums[slot][c] offset:0 atIndex:0];
    [enc setBuffer:check_data[c] offset:0 atIndex:1];
    [enc setBuffer:um_async_check_params[slot][c] offset:0 atIndex:2];
    [enc dispatchThreads:MTLSizeMake(groups, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(um_check_width < groups ? um_check_width : groups, 1, 1)];
  }
  um_fin_list *fins = &um_open_fins[q];
  fins->slot[fins->n] = slot;
  fins->ticket[fins->n] = ticket;
  fins->n++;
  um_open_n[q]++;
  um_open_held[q] = hold;
  um_async_cmd[slot] = um_open_cmd[q];
  // 提出するかどうか。ためた数が上限に達したら、いつも提出する（um_open_fins の大きさもこれで足りる）。
  //   HOLD: それ以外では提出しない（同じ列への次の HOLD なしの submit などで提出される）。
  //   列 1（急ぎ）: すぐ提出する。
  //   列 0: GPU が空いていれば（前の提出が終わっていれば）すぐ提出する。忙しければ、ためておく。
  int commit_now = um_open_n[q] >= UM_OPEN_MAX || (!hold && (q == 1 || um_inflight[0] < UM_INFLIGHT_MAX));
  if (commit_now) um_async_commit_open(q);
  pthread_mutex_unlock(&um_async_lock);
  um_gs_add(&um_gs_submit[q], 1);
  // 書く領域を、最後に書いた券と列として控える（um_async_cross_wait）。列が 1 本しかない設定では要らない。
  if (um_allow_urgent || um_ap_enabled) {
    if (!um_async_lastwrite) um_async_lastwrite = [NSMutableDictionary dictionary];
    for (int32_t i = 0; i < nbufs; i++) {
      if (table[i * 4 + 3] != 0) um_async_lastwrite[@((uintptr_t)table[i * 4])] = @((ticket << 8) | (uint64_t)q);
    }
  }
}

// ---- gpu.submit の常駐（待機命令）----
// 常駐（um_slot。gpu.dispatch 用）と同じ考え方を gpu.submit に使う。同じ「形」の submit が続くなら、その形の命令を
// 先に提出して合図（MTLSharedEvent）を待たせておき、submit は引数と個数を共有メモリに書いて合図を進めるだけにする
// （命令を作って提出する手間と、提出から GPU が動き出すまでの一部を省く。tools/metal/roundtrip.m）。
//   - 形は、1つのスレッドが HOLD でつないで投げた submit の並び（HOLD なしの submit で終わる。1つだけでもよい。
//     最大 UM_PARK_SEQ 個）と、それぞれの kernel・Buffer の組。並びの途中の submit は、そのスレッドの並び（um_chain）に
//     ためておき、並びが終わった時点で、形が待機していれば合図を出し、していなければ今までどおり列に積む。
//   - 同じ形の並びを UM_PARK_AFTER 回見たら、待機を始める（命令を UM_AP_AHEAD 個、合図待ちで提出しておく）。
//     形は UM_AP_SLOTS 個まで。合図を出すたびに1つ補充する。
//   - 形ごとに命令の列（MTLCommandQueue）と合図を分ける。別の形（たとえば評価の置き場ごと）の仕事は別の列なので、
//     GPU の上で同時に走れる（以前は全スレッドの仕事が 1 本の急ぎの列に並び、GPU 側の 1 件あたりの手間で詰まっていた）。
//     同じ領域を別の列のまだ終わっていない仕事が書いていれば、先にそれを待つ（um_async_cross_wait）。
//   - 待機分は、それぞれ引数の領域（枠ごとに UM_AP_RING 個の輪の1つ）と完了の控え（um_ap_fin）を持つ。
//     完了は普通の経路と同じく、Metal の status、完了の印（um_async_fin。命令の完了の処理が書く）、呼び手の札で知る。
//     GPU は走り出すときに引数を読み（合図を出す前に書く）、結果は命令の区切りで見える（常駐と同じ決まり）。
//   - 対象は検算を使わない（NOCHECK か、UM_GPU_RESIDENT=status）、急ぎの列（URGENT）の submit だけ
//     （UM_GPU_ASYNC_RESIDENT=all で急ぎでないものも。um_ap_enabled を参照）。検算は券ごとに検算値の領域が要り、
//     先に作っておく命令には入れにくいため。Buffer が 17 個以上、Buffer 以外の引数が 512 バイトを超えるもの、
//     コピーして渡す Buffer があるものも対象外（今までの経路）。グループの submit（gpu.submitGroups）も対象で、
//     形にグループの大きさを含める（待機命令の格子は um_resident_grid。普通の submit と混ぜた並びも 1 つの形）。
//   - 並びの途中の券を別のスレッドが見る（ready / wait / 解放 / 券の表が満杯）ときは、その並びを今までの経路でいま積む
//     （um_async_cmd_of）。gpu.commit() も全スレッドの並びを積む。
//   - 関係する Buffer が解放されるとき（um_on_free）と、UM_PARK_IDLE_STOP 秒合図が出ていないとき（um_ap_idle。
//     合図を長く待たせた命令は Metal に打ち切られる）は、待機分を個数 0 で空振りさせて片付ける。
//   - 使っているあいだは GPU を起こしておく（um_awake_until。常駐と同じスレッド。最後の submit から 50 ms で休む）。
// UM_GPU_ASYNC_RESIDENT=0 でこの経路を使わない（今までどおり、submit ごとに列に積む）。
// 枠の状態は um_gpu_lock と枠の鍵（um_ap_lock）の両方を持って変える。待機分の補充（um_ap_refill）は、submit が
// um_gpu_lock を放してから、枠の鍵だけで行う（命令を作る数マイクロ秒のあいだ、ほかのスレッドの submit を止めない）。
// 鍵の順番は um_gpu_lock → um_ap_lock → um_async_lock。
#define UM_AP_SLOTS 64
#define UM_AP_AHEAD 4
#define UM_AP_RING 16       // 枠ごとの引数の領域の数（待機分 UM_AP_AHEAD 個 + 走っている最中のもの）
#define UM_AP_PSTRIDE 768   // 引数の領域の中の、dispatch 1つぶんの引数の塊の間隔（256 の倍数）
#define UM_AP_INDIRECT (UM_PARK_SEQ * UM_AP_PSTRIDE)  // 間接 dispatch の個数の場所（dispatch ごとに 16 バイト）
#define UM_AP_STATUS (UM_AP_INDIRECT + UM_PARK_SEQ * 16)  // 郵便受けの見張りが書く状態（16 バイト。見つけたか・枠の数・番号）
#define UM_AP_RINGBYTES (UM_AP_STATUS + 64)
typedef struct {
  _Atomic int n;  // 合図を出す直前に書く（それまで 0）。命令の完了の処理が読む
  double fire_t;  // 合図を出した時刻（um_gpu_clock。UM_GPU_STATS の往復の内訳）
  int slot[UM_PARK_SEQ];
  uint64_t ticket[UM_PARK_SEQ];
} um_ap_fin;
static struct {
  int state;  // 0 = 空き、1 = 回数を数えている、2 = 待機中
  int hits;
  uint64_t last_use;
  uint64_t epoch;  // 待機を始める・やめるたびに増える（補充が、やめて始め直した枠に足さないように）
  int n;
  struct {
    int32_t kernel, src_len, nbufs, block_bytes;
    const char *src;
    uint64_t sig[UM_PARK_BUFS * 4];
    uint32_t group;  // グループの大きさ（0 は普通の dispatch。形の一部）
    NSUInteger width;
  } d[UM_PARK_SEQ];
  uint64_t committed, fired;  // 合図の値（枠ごとに増え続ける）。番号 c の待機分は合図 c + 1 で走る
  CFAbsoluteTime last_fire;
  int generic;  // 郵便受けの列の形: Buffer の場所は形に含めない（依頼ごとにポインタの表で渡す。sig は要素の大きさと書くかだけ見る）
} um_ap[UM_AP_SLOTS];
// 郵便受けの列の形が使ったことのある Buffer（待機命令は、これらを useResource で知らせてから出す）。
#define UM_AP_RES 256
#define UM_AP_PTRS 640  // 引数の塊（UM_AP_PSTRIDE）の中の、ポインタの表の場所（8 バイト × UM_PARK_BUFS）
static id<MTLBuffer> um_ap_res_buf[UM_AP_SLOTS][UM_AP_RES];
static void *um_ap_res_data[UM_AP_SLOTS][UM_AP_RES];
static uint64_t um_ap_res_gpu[UM_AP_SLOTS][UM_AP_RES];
static int um_ap_res_n[UM_AP_SLOTS];
static int um_ap_res_untracked[UM_AP_SLOTS];  // 追跡しない Buffer が混ざっている（dispatch のあいだに壁を入れる）
static pthread_mutex_t um_ap_lock[UM_AP_SLOTS];  // um_metal_init_once で用意する
static _Atomic int um_ap_parked;                 // 待機中の枠の数（um_ap_idle が鍵を取らずに戻るため）
static uint64_t um_ap_tick;
static id<MTLCommandQueue> um_ap_queue[UM_AP_SLOTS];
static id<MTLSharedEvent> um_ap_event[UM_AP_SLOTS];
static id<MTLComputePipelineState> um_ap_pipeline[UM_AP_SLOTS][UM_PARK_SEQ];
static id<MTLBuffer> um_ap_buf[UM_AP_SLOTS][UM_PARK_SEQ * UM_PARK_BUFS];
static id<MTLBuffer> um_ap_ring[UM_AP_SLOTS][UM_AP_RING];
static unsigned char *um_ap_ring_mem[UM_AP_SLOTS][UM_AP_RING];  // um_ap_ring の contents（毎回 [contents] を呼ばない）
static id<MTLCommandBuffer> um_ap_cmd[UM_AP_SLOTS][UM_AP_RING];  // 番号 c の待機分は [c % UM_AP_RING]
static um_ap_fin *um_ap_fins[UM_AP_SLOTS][UM_AP_RING];          // その完了の控え（完了の処理が解放する）
// submit が um_gpu_lock を放したあとに補充する枠（-1 = なし）と、そのときの epoch。
static _Thread_local int um_ap_refill_k = -1;
static _Thread_local uint64_t um_ap_refill_epoch;

static void um_ap_init_locks(void) {
  for (int k = 0; k < UM_AP_SLOTS; k++) pthread_mutex_init(&um_ap_lock[k], NULL);
}

static int um_ap_same(int k, const um_job *jobs, int n) {
  if (um_ap[k].state == 0 || um_ap[k].n != n) return 0;
  for (int d = 0; d < n; d++) {
    if (um_ap[k].d[d].kernel != jobs[d].kernel || um_ap[k].d[d].src != jobs[d].src || um_ap[k].d[d].group != jobs[d].group ||
        um_ap[k].d[d].nbufs != jobs[d].nbufs || um_ap[k].d[d].block_bytes != jobs[d].block_bytes) {
      return 0;
    }
    if (um_ap[k].generic) {
      // Buffer の場所と長さは形に含めない（要素の大きさと、書くかどうかだけ）。
      for (int32_t i = 0; i < jobs[d].nbufs; i++) {
        if (um_ap[k].d[d].sig[i * 4 + 2] != jobs[d].table[i * 4 + 2] || um_ap[k].d[d].sig[i * 4 + 3] != jobs[d].table[i * 4 + 3]) return 0;
      }
    } else if (memcmp(um_ap[k].d[d].sig, jobs[d].table, (size_t)jobs[d].nbufs * 4 * sizeof(uint64_t)) != 0) {
      return 0;
    }
  }
  return 1;
}

// 並び jobs（n 個）が、枠 k の形（郵便受けの列の形）の並びの一部として順に現れるか。現れるなら、jobs[j] に当たる
// dispatch の番号を map[j] に入れて 1（同じ長さなら map[j] = j）。足りない dispatch は個数 0 のまま空振りになる
// （1 つ 約 1.4 µs）。並びの長さが違うだけの仕事（評価の、差分の kernel がある仕事とない仕事など）を 1 つの形で受ける。
static int um_ap_embed(int k, const um_job *jobs, int n, int *map) {
  if (um_ap[k].state == 0 || !um_ap[k].generic || um_ap[k].n < n) return 0;
  int d = 0;
  for (int j = 0; j < n; j++) {
    for (;; d++) {
      if (d == um_ap[k].n) return 0;
      if (um_ap[k].d[d].kernel != jobs[j].kernel || um_ap[k].d[d].src != jobs[j].src || um_ap[k].d[d].group != jobs[j].group ||
          um_ap[k].d[d].nbufs != jobs[j].nbufs || um_ap[k].d[d].block_bytes != jobs[j].block_bytes) {
        continue;
      }
      int same = 1;
      for (int32_t i = 0; i < jobs[j].nbufs && same; i++) {
        same = um_ap[k].d[d].sig[i * 4 + 2] == jobs[j].table[i * 4 + 2] && um_ap[k].d[d].sig[i * 4 + 3] == jobs[j].table[i * 4 + 3];
      }
      if (same) break;
    }
    if (map) map[j] = d;
    d++;
  }
  return 1;
}

// 枠 k が持っている Metal の物を手放す（枠の鍵を持って呼ぶ）。
static void um_ap_clear(int k) {
  for (int r = 0; r < UM_AP_RING; r++) {
    um_ap_cmd[k][r] = nil;
    um_ap_fins[k][r] = NULL;
    um_ap_ring[k][r] = nil;
    um_ap_ring_mem[k][r] = NULL;
  }
  for (int d = 0; d < UM_PARK_SEQ; d++) {
    um_ap_pipeline[k][d] = nil;
    for (int i = 0; i < UM_PARK_BUFS; i++) um_ap_buf[k][d * UM_PARK_BUFS + i] = nil;
  }
  um_ap_queue[k] = nil;
  um_ap_event[k] = nil;
  for (int i = 0; i < um_ap_res_n[k]; i++) um_ap_res_buf[k][i] = nil;
  um_ap_res_n[k] = 0;
  um_ap_res_untracked[k] = 0;
}

// 表の 1 行（場所、長さ、要素の大きさ、書くか）の領域の、GPU 側のバッファ（um_gpu_lock を持って呼ぶ）。今までの経路と
// 同じものを使う（解放のときに um_on_free が控えを見つけられるように）。作れなければ nil。
static id<MTLBuffer> um_ap_wrap(const uint64_t *e) {
  size_t page = (size_t)getpagesize();
  if (!um_async_wrap) um_async_wrap = [NSMutableDictionary dictionary];
  void *data = (void *)(uintptr_t)e[0];
  size_t len = (size_t)e[1];
  size_t size = (len > 0 ? len : 1) * (size_t)e[2];
  size_t rounded = (size + page - 1) / page * page;
  NSNumber *key = @((uintptr_t)data);
  id<MTLBuffer> b = um_async_wrap[key];
  if (!um_wrap_ok(b, rounded, data)) {
    b = [um_device newBufferWithBytesNoCopy:data length:rounded options:um_wrap_options(data) deallocator:nil];
    if (!b) return nil;
    if (!um_async_wrap[key]) atomic_fetch_add(&um_gpu_regions, 1);
    um_async_wrap[key] = b;
  }
  return b;
}

// ---- 郵便受けの列（UM_GPU_MAILBOX=1。待機命令が合図の代わりに郵便受けを見る）----
// 合図（MTLSharedEvent）で待たせた命令は、合図を出してから GPU が動き出すまでに 60〜100 µs かかる（GPU を管理している側が
// 命令を始める手間）。そこで待機命令の先頭に見張りの kernel（um_mb_poll、1 スレッド）を置き、命令を先に走らせておく。
// 見張りは共有メモリの「郵便受け」を見て、番号が自分の番号以上になったら抜け、続きの dispatch（間接。個数は引数の領域から）が走る。
//   - 走っている kernel は、一度読んだ場所の新しい値を見られない（GPU のキャッシュに古い値が残る）。まだ読んでいない行は
//     メモリから読むので、CPU がいま書いた値が見える（tools/metal/spin_fresh.m で遅れ 1 µs 前後）。そこで郵便受けを
//     UM_MB_SLOTS 個の枠（64 バイトおき）にし、見張りは枠を 1 回ずつ順に読む。CPU は全部の枠に番号を書く（um_mq_post）。
//     命令の切れ目では GPU のキャッシュが捨てられるので、次の命令の見張りは同じ枠をまた先頭から使える（spin_ring.m）。
//   - GPU は別々の列の命令をほぼ 1 つずつしか走らせないので、回り続ける見張りは全体で 1 つにする（形ごとに列を分けて
//     見張りを置くと、ほかの形の見張りが GPU を塞ぐ）。そこで郵便受けの経路では、全部の形の待機命令を 1 本の列（um_mq_queue）に
//     並べる。同じ列に出してある命令は、前の命令が終わると 0.2 µs で始まる。
//   - 番号は列で 1 つ（増え続ける。番号 c の待機分は、郵便受けの番号が c + 1 以上で走る）。「以上」で比べるので、前の命令が
//     まだ走っているうちに次の番号を書いてもよい（古い値を読んでも小さいだけで、走り出すことはない）。
//   - 郵便受けの列の形は、kernel の並びだけで決める（um_ap[k].generic。Buffer の場所は含めない）。待機命令の kernel は
//     Buffer を [[buffer(n)]] で受け取らず、引数の塊に置いたポインタの表（MTLBuffer の gpuAddress。UM_AP_PTRS）から読む
//     入口（um_main_bl。src/msl.ts）を使う。依頼のときに、その仕事の Buffer のポインタを書く。これで、同じ kernel を別々の
//     Buffer の組で（置き場ごと、スレッドごとに）ばらばらの順に投げても、どの待機分でも受けられる（tools/metal/spin_bindless.m）。
//     待機命令は、その形が使ったことのある Buffer（um_ap_res_*）を全部 useResource で知らせてから出す。初めての Buffer が
//     来たら、控えに足して、出してある待機分をまとめて空振りさせる（出し直す）。
//   - 待機分は形ごとに作る（um_mq_shape）。依頼の形が先頭の待機分と違えば、合う待機分まで（なければ列の後ろに 1 つ出して
//     そこまで）を個数 0 のまま空振りさせる（1 つ 約 5 µs。dispatch 1 つの空振りが 約 1.4 µs）。依頼を出した形と
//     空振りさせた形は列の後ろに補充する（um_mq_refill）。
//     今までの経路には回さない（ほかの列の命令は見張りに塞がれるので、どかすために待機分を全部空振りさせることになる）。
//   - 引数と個数は、番号を書く前に書く（見張りが抜けてから読まれる。その命令がまだ読んでいない行なので新しい値が見える）。
//     個数の領域は命令を出す前に 0 にしておく（依頼を書かれなかった待機分は、続きの dispatch が何もしない）。
//   - 見張りは枠を使い切ると抜ける（寿命は 1 秒強。長く走り続ける kernel は Metal に打ち切られるため）。寿命より前に
//     片付けるために、番号を UM_MB_IDLE_STOP 秒書いていなければ待機分をまとめて空振りさせ（um_ap_idle）、先頭の待機分が
//     UM_MB_LIVE 秒より古ければ、依頼の前にまとめて空振りさせる（寿命で抜けているかもしれないので、依頼は新しく出す
//     待機分に書く）。依頼を出した命令の見張りが「見つからず」で終わっていたら（ありえないはず）失敗にする。
//   - 見張りが回っているあいだ、このプロセスのほかの列の命令は走れない（GPU は列を命令の切れ目でしか切り替えない）。
//     そこで、ほかの列に命令を出す場所（um_async_commit_open、um_gpu_flush の提出、常駐の合図）では、出したあとに
//     待機分をまとめて空振りさせて見張りをどかす（um_mq_kick）。今までの経路に回した依頼もここを通る。
//     GPU を起こしておくスレッドの命令だけは、どかさずに待たせる（um_awake_main）。
//   - 空振りが多い使い方（スレッドごとに別の形を同時に投げる、など）では役に立たないので、郵便受けをやめて合図の経路に
//     戻す（um_mq_judge / um_mq_disable。戻したら、その実行のあいだは使わない）。
//   - 結果が CPU から見えるのは、今までどおり命令の区切り。ただし GPU が終えてから見えるまでは、続けて投げていれば 約 20 µs、
//     仕事のあいだが 40 µs 以上空くと 45〜85 µs かかる（合図の経路は 30〜40 µs。GPU を管理している側が、隙間なく続く命令の
//     切れ目では働かないため。別の列の命令や合図で起こしておく形は効かなかった。tools/metal/spin_bindless.m）。
//   - GPU が埋まっている使い方（投げた仕事が何件も順番を待っている）でも、命令の切れ目の手間（合図の経路で 約 8 µs）が
//     なくなる分だけ処理量が増える（探索の評価、10 スレッドで GPU の稼働 84% → 97%）。
//   - 鍵は um_gpu_lock → um_ap_lock → um_mq_lock → um_async_lock の順。列の状態（um_mq_*）は um_mq_lock を持って触る。
#define UM_MB_SLOTS 2496     // 郵便受けの枠の数（1 枠 64 バイト）
#define UM_MB_FAST 2048      // 間を空けずに見る枠の数（1 枠 約 0.4 µs）
#define UM_MB_IDLE_STOP 0.25
#define UM_MB_LIVE 0.5
#define UM_MQ_RING 64        // 待機分の輪（番号 c の待機分は [c % UM_MQ_RING]）
#define UM_MQ_AHEAD 2        // 形ごとに待機させておく数
#define UM_MQ_QID (2 + UM_AP_SLOTS)  // 列をまたぐ安全網（um_async_cross_wait）での、郵便受けの列の番号
static id<MTLCommandQueue> um_mq_queue;
static id<MTLSharedEvent> um_mq_event;  // 待機命令の最後に出す合図（誰も待たない。GPU を管理している側を命令の切れ目で働かせる）
static id<MTLBuffer> um_mq_mb;
static uint32_t *um_mq_mb_mem;
static id<MTLBuffer> um_mq_ring[UM_MQ_RING];
static unsigned char *um_mq_ring_mem[UM_MQ_RING];
static id<MTLCommandBuffer> um_mq_cmd[UM_MQ_RING];
static um_ap_fin *um_mq_fins[UM_MQ_RING];
static int um_mq_shape[UM_MQ_RING];       // その待機分の形（um_ap の枠の番号）
static double um_mq_commit_t[UM_MQ_RING];  // 出した時刻（um_gpu_clock）
static uint64_t um_mq_committed, um_mq_fired;
static double um_mq_last_post;             // 最後に番号を書いた時刻（um_gpu_clock）
static int um_mq_parked[UM_AP_SLOTS];      // 形ごとの、まだ走らせていない待機分の数
// 形ごとの、待機させておきたい数。依頼が来たら UM_MQ_AHEAD、空振りさせられるたびに 1 減らす（使われなくなった形の
// 待機分を補充し続けると、列の先頭に来るたびに空振りになる）。
static int um_mq_want[UM_AP_SLOTS];
static _Atomic int um_mq_handling[UM_MQ_RING];  // その領域を使った命令の完了の処理がまだ終わっていない（領域の状態を読むため）
static pthread_mutex_t um_mq_lock = PTHREAD_MUTEX_INITIALIZER;

// 郵便受けの列を用意する（um_gpu_lock を持って呼ぶ。最初の 1 回だけ）。使えるなら 1。
static int um_mq_init(void) {
  if (um_mq_on) return 1;
  if (um_mb_off || !um_mb_pipeline) return 0;
  id<MTLCommandQueue> q = [um_device newCommandQueueWithMaxCommandBufferCount:UM_MQ_RING * 2];
  id<MTLBuffer> mb = [um_device newBufferWithLength:UM_MB_SLOTS * 64 options:MTLResourceStorageModeShared];
  if (!q || !mb) return 0;
  memset([mb contents], 0, UM_MB_SLOTS * 64);
  pthread_mutex_lock(&um_mq_lock);
  for (int r = 0; r < UM_MQ_RING; r++) {
    id<MTLBuffer> ring = [um_device newBufferWithLength:UM_AP_RINGBYTES options:MTLResourceStorageModeShared];
    if (!ring) {
      for (int i = 0; i < r; i++) um_mq_ring[i] = nil;
      pthread_mutex_unlock(&um_mq_lock);
      return 0;
    }
    memset([ring contents], 0, UM_AP_RINGBYTES);
    um_mq_ring[r] = ring;
    um_mq_ring_mem[r] = (unsigned char *)[ring contents];
  }
  um_mq_queue = q;
  um_mq_event = [um_device newSharedEvent];
  um_mq_mb = mb;
  um_mq_mb_mem = (uint32_t *)[mb contents];
  um_mq_on = 1;
  pthread_mutex_unlock(&um_mq_lock);
  return 1;
}

// 郵便受けの全部の枠に、いまの番号（um_mq_fired）を書く（um_mq_lock を持って呼ぶ）。
static void um_mq_post(void) {
  // 引数と個数の書き込みを先に確定させてから番号を書く。前の枠から順に書く（見張りは先頭から読んでいる）。
  atomic_thread_fence(memory_order_seq_cst);
  uint32_t v = (uint32_t)um_mq_fired;
  uint32_t *mb = um_mq_mb_mem;
  for (int i = 0; i < UM_MB_SLOTS; i++) __atomic_store_n(&mb[i * 16], v, __ATOMIC_RELAXED);
  um_mq_last_post = um_gpu_clock();
}

// 待機分をまとめて空振りさせる（um_mq_lock を持って呼ぶ）。wait なら、列の命令が全部終わるのを待つ。
static void um_mq_flush(int wait) {
  if (um_mq_fired < um_mq_committed) {
    // 個数は出すときに 0 にしてあり、依頼を書いたのは um_mq_fired より前の待機分だけ。
    um_mq_fired = um_mq_committed;
    um_mq_post();
    um_gs_add(&um_gs_mq_flushes, 1);
  }
  memset(um_mq_parked, 0, sizeof um_mq_parked);
  if (wait) {
    for (int r = 0; r < UM_MQ_RING; r++) {
      id<MTLCommandBuffer> cmd = um_mq_cmd[r];
      if (cmd && [cmd status] < MTLCommandBufferStatusCompleted) [cmd waitUntilCompleted];
    }
  }
}

// 先頭の待機分の見張りが動き出した時刻の下限（um_mq_lock を持って呼ぶ）。前の命令が終わってから動き出すので、
// 出した時刻と、最後に番号を書いた時刻の遅いほうより後。
static double um_mq_head_since(void) {
  double t = um_mq_commit_t[um_mq_fired % UM_MQ_RING];
  return t > um_mq_last_post ? t : um_mq_last_post;
}

// 形 k の待機分を列の後ろに 1 つ足す（枠 k の鍵と um_mq_lock を持って呼ぶ）。見張り → 並びの dispatch。
static void um_mq_commit_one(int k) {
  uint64_t c = um_mq_committed;
  int r = (int)(c % UM_MQ_RING);
  // 輪を一周して同じ引数の領域を使う。前に使った命令がまだ走っていれば待つ（待機分は輪の半分までなので、走っている最中のもの）。
  id<MTLCommandBuffer> old = um_mq_cmd[r];
  if (old && [old status] < MTLCommandBufferStatusCompleted) [old waitUntilCompleted];
  old = nil;
  // 完了の処理は領域の状態（UM_AP_STATUS）を読むので、それが済むまで領域を書き換えない。
  while (atomic_load(&um_mq_handling[r])) sched_yield();
  atomic_store(&um_mq_handling[r], 1);
  id<MTLCommandBuffer> cmd = [um_mq_queue commandBuffer];
  if (!cmd) um_metal_fail("creating the command buffer", nil);
  id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
  if (!enc) um_metal_fail("creating the command buffer", nil);
  id<MTLBuffer> ring = um_mq_ring[r];
  // 個数を 0 にしてから出す（依頼を書かれないまま見張りが抜けたら、続きの dispatch は何もしない）。
  unsigned char *mem = um_mq_ring_mem[r];
  memset(mem + UM_AP_INDIRECT, 0, UM_PARK_SEQ * 16);
  for (int d = 0; d < UM_PARK_SEQ; d++) ((uint32_t *)(mem + (size_t)d * UM_AP_PSTRIDE))[0] = 0;
  memset(mem + UM_AP_STATUS, 0, 16);
  uint32_t m[4] = {UM_MB_SLOTS, UM_MB_FAST, (uint32_t)(c + 1), 0};
  [enc setComputePipelineState:um_mb_pipeline];
  [enc setBuffer:um_mq_mb offset:0 atIndex:0];
  [enc setBytes:m length:sizeof m atIndex:1];
  [enc setBuffer:ring offset:UM_AP_STATUS atIndex:2];
  [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
  // kernel は Buffer をポインタの表から読む。使うかもしれない Buffer を全部知らせておく。
  if (um_ap_res_n[k] > 0) {
    __unsafe_unretained id<MTLResource> res[UM_AP_RES];
    for (int i = 0; i < um_ap_res_n[k]; i++) res[i] = um_ap_res_buf[k][i];
    [enc useResources:res count:(NSUInteger)um_ap_res_n[k] usage:MTLResourceUsageRead | MTLResourceUsageWrite];
  }
  for (int d = 0; d < um_ap[k].n; d++) {
    // 追跡しない Buffer があれば、並びの前の kernel の書き込みを待たせる（um_untracked を参照）。
    if (d > 0 && um_ap_res_untracked[k]) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [enc setComputePipelineState:um_ap_pipeline[k][d]];
    [enc setBuffer:ring offset:(NSUInteger)d * UM_AP_PSTRIDE atIndex:0];
    [enc setBuffer:ring offset:(NSUInteger)d * UM_AP_PSTRIDE + UM_AP_PTRS atIndex:1];
    [enc dispatchThreadgroupsWithIndirectBuffer:ring
                           indirectBufferOffset:UM_AP_INDIRECT + (NSUInteger)d * 16
                          threadsPerThreadgroup:MTLSizeMake(um_ap[k].d[d].width, 1, 1)];
  }
  [enc endEncoding];
  // UM_GPU_MAILBOX_SIGNAL=1: 命令の最後に合図を出す操作を入れる。同じ列の命令は隙間なく続けて走るので、そのままでは
  // GPU を管理している側が命令の切れ目で働かず、結果が CPU から見えるのが遅れる（20〜84 µs。間が空くほど遅い）。合図の
  // 操作があると切れ目で働き、結果は 約 11 µs で見える（tools/metal/spin_bindless.m）。ただし次の命令の開始が管理している側の
  // 処理を待つようになり（16 µs〜）、続けて投げる使い方では往復も処理量も悪くなったので、既定では入れない。
  if (um_mb_signal && um_mq_event) [cmd encodeSignalEvent:um_mq_event value:c + 1];
  um_ap_fin *fins = (um_ap_fin *)calloc(1, sizeof(um_ap_fin));
  if (!fins) um_metal_fail("allocating the completion list", nil);
  [cmd addCompletedHandler:^(id<MTLCommandBuffer> done) {
    @autoreleasepool {  // um_gpu_wait の説明を参照
      if ([done status] != MTLCommandBufferStatusCompleted) atomic_store(&um_async_failed, 1);
      int n = atomic_load_explicit(&fins->n, memory_order_acquire);
      if (n > 0) {
        um_gs_gpu_time(done);
        um_gs_ap_lat(done, fins->fire_t);
        // 依頼を出したのに見張りが見つけずに抜けていたら、仕事は走っていない（寿命の確認は um_mq_take）。
        if (((const uint32_t *)((const unsigned char *)[ring contents] + UM_AP_STATUS))[0] != 1) {
          fprintf(stderr, "gpu: the mailbox watcher expired before a submitted job (kernel %d)\n", um_ap[k].d[0].kernel);
          atomic_store(&um_async_failed, 1);
        }
      }
      for (int j = 0; j < n; j++) um_fin_mark(fins->slot[j], fins->ticket[j]);
      free(fins);
      atomic_store(&um_mq_handling[r], 0);
    }
  }];
  [cmd commit];
  um_mq_commit_t[r] = um_gpu_clock();
  um_mq_shape[r] = k;
  um_mq_cmd[r] = cmd;
  um_mq_fins[r] = fins;
  um_mq_committed++;
  um_mq_parked[k]++;
}

// 形 k の依頼を書く待機分を選び、輪の番号を返す（枠 k の鍵と um_mq_lock を持って呼ぶ）。それより前の待機分は空振りになる。
// このあと呼ぶ側が引数と個数を書き、um_mq_go で番号を進める。
static uint64_t um_mq_take(int k) {
  // 先頭の見張りが寿命で抜けているかもしれないなら、いまある待機分は使わない。
  if (um_mq_fired < um_mq_committed && um_gpu_clock() - um_mq_head_since() > UM_MB_LIVE) um_mq_flush(0);
  uint64_t c = um_mq_fired;
  while (c < um_mq_committed && um_mq_shape[c % UM_MQ_RING] != k) c++;
  int cold = c == um_mq_committed;
  if (cold) {
    // この形の待機分がない（始めたばかり、まとめて空振りさせたあと、補充が間に合っていない）。列の後ろに 1 つ出して、
    // そこまでを空振りさせる。
    um_mq_commit_one(k);
    um_gs_add(&um_gs_mq_cold, 1);
  }
  for (uint64_t s = um_mq_fired; s < c; s++) {
    int other = um_mq_shape[s % UM_MQ_RING];
    um_mq_parked[other]--;
    if (um_mq_want[other] > 0) um_mq_want[other]--;
  }
  um_mq_want[k] = UM_MQ_AHEAD;
  if (c > um_mq_fired) um_gs_add(&um_gs_mq_skips, c - um_mq_fired);
  um_mq_judge(c - um_mq_fired, cold);
  return c;
}

// um_mq_take で選んだ待機分 c まで走らせる（引数と個数を書いたあとで呼ぶ。um_mq_lock を持って呼ぶ）。
static void um_mq_go(int k, uint64_t c) {
  um_mq_parked[k]--;
  um_mq_fired = c + 1;
  um_mq_post();
}

// ほかの列に命令を出したあとで呼ぶ。待機分をまとめて空振りさせて、回っている見張りをどかす（どの鍵を持っていてもよい。
// um_mq_lock を持ったまま um_async_lock などを取る場所はない）。
static void um_mq_kick(void) {
  if (!um_mq_on) return;
  pthread_mutex_lock(&um_mq_lock);
  if (um_mq_fired < um_mq_committed) {
    um_mq_flush(0);
    um_gs_add(&um_gs_mq_kicks, 1);
  }
  pthread_mutex_unlock(&um_mq_lock);
}

// 郵便受けが役に立っているかを数える（um_mq_lock を持って呼ぶ。依頼 1 つごと）。skipped は空振りさせた待機分の数、
// cold はその形の待機分がなかったか（いまは数えるだけ）。UM_MQ_WINDOW 回ごとに見て、空振りが依頼の 3 倍を超えていたら、
// やめる印を付ける（um_chain_run が um_mq_disable を呼ぶ）。たくさんの形（kernel の並び）を入り混じった順で投げる使い方。
#define UM_MQ_WINDOW 256
static int um_mq_win_n, um_mq_win_skipped, um_mq_win_cold;
static _Atomic int um_mq_disable_req;
static void um_mq_judge(uint64_t skipped, int cold) {
  um_mq_win_skipped += (int)skipped;
  um_mq_win_cold += cold;
  if (++um_mq_win_n < UM_MQ_WINDOW) return;
  if (um_mq_win_skipped > 3 * um_mq_win_n) atomic_store(&um_mq_disable_req, 1);
  um_mq_win_n = um_mq_win_skipped = um_mq_win_cold = 0;
}

// 郵便受けをやめて、合図の経路に戻す（um_gpu_lock だけを持って呼ぶ）。待機中の形は全部やめる（また続けば合図で待機し直す）。
static void um_ap_stop(int k);
static void um_mq_disable(void) {
  atomic_store(&um_mq_disable_req, 0);
  if (!um_mq_on) return;
  for (int k = 0; k < UM_AP_SLOTS; k++) {
    if (um_ap[k].state != 0) um_ap_stop(k);  // 数えている最中の形も空ける（合図の経路の形として数え直す）
  }
  um_mq_on = 0;
  um_mb_enabled = 0;
  um_mb_off = 1;
  if (um_trace) fprintf(stderr, "gpu: the mailbox is turned off (too many skipped commands)\n");
}

// 待機分の足りない形に補充する（鍵を持たずに呼ぶ。submit が um_gpu_lock を放したあと）。
static void um_mq_refill(void) {
  for (int k = 0; k < UM_AP_SLOTS; k++) {
    if (um_ap[k].state != 2 || !um_ap[k].generic || um_mq_parked[k] >= um_mq_want[k]) continue;  // 鍵なしの下見（下で確かめ直す）
    pthread_mutex_lock(&um_ap_lock[k]);
    if (um_ap[k].state == 2 && um_ap[k].generic) {
      pthread_mutex_lock(&um_mq_lock);
      while (um_mq_parked[k] < um_mq_want[k] && um_mq_committed - um_mq_fired < UM_MQ_RING / 2) um_mq_commit_one(k);
      pthread_mutex_unlock(&um_mq_lock);
    }
    pthread_mutex_unlock(&um_ap_lock[k]);
  }
}

// 枠 k の待機分を1つ足す（枠の鍵を持って呼ぶ）。並びの dispatch を順に1つの命令に入れ、合図 committed + 1 を待たせる。
static void um_ap_commit_one(int k) {
  uint64_t c = um_ap[k].committed;
  int r = (int)(c % UM_AP_RING);
  // 輪を一周して同じ引数の領域を使う。前に使った命令がまだ走っていれば待つ（待機分は UM_AP_AHEAD 個なので、まれ）。
  id<MTLCommandBuffer> old = um_ap_cmd[k][r];
  if (old && [old status] < MTLCommandBufferStatusCompleted) [old waitUntilCompleted];
  old = nil;
  id<MTLCommandBuffer> cmd = [um_ap_queue[k] commandBuffer];
  if (!cmd) um_metal_fail("creating the command buffer", nil);
  [cmd encodeWaitForEvent:um_ap_event[k] value:c + 1];
  id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
  if (!enc) um_metal_fail("creating the command buffer", nil);
  id<MTLBuffer> ring = um_ap_ring[k][r];
  for (int d = 0; d < um_ap[k].n; d++) {
    // 追跡しない Buffer があれば、並びの前の kernel の書き込みを待たせる（um_untracked を参照）。
    if (d > 0) {
      int untracked = 0;
      for (int32_t i = 0; i < um_ap[k].d[d].nbufs; i++) {
        untracked |= [um_ap_buf[k][d * UM_PARK_BUFS + i] hazardTrackingMode] == MTLHazardTrackingModeUntracked;
      }
      if (untracked) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
    [enc setComputePipelineState:um_ap_pipeline[k][d]];
    [enc setBuffer:ring offset:(NSUInteger)d * UM_AP_PSTRIDE atIndex:0];
    for (int32_t i = 0; i < um_ap[k].d[d].nbufs; i++) {
      [enc setBuffer:um_ap_buf[k][d * UM_PARK_BUFS + i] offset:0 atIndex:(NSUInteger)(i + 1)];
    }
    [enc dispatchThreadgroupsWithIndirectBuffer:ring
                           indirectBufferOffset:UM_AP_INDIRECT + (NSUInteger)d * 16
                          threadsPerThreadgroup:MTLSizeMake(um_ap[k].d[d].width, 1, 1)];
  }
  [enc endEncoding];
  um_ap_fin *fins = (um_ap_fin *)calloc(1, sizeof(um_ap_fin));
  if (!fins) um_metal_fail("allocating the completion list", nil);
  [cmd addCompletedHandler:^(id<MTLCommandBuffer> done) {
    // 終わったら、この待機分で走らせた券ごとに完了の印を書く（空振りで終わったものは n が 0）。
    @autoreleasepool {  // um_gpu_wait の説明を参照
      if ([done status] != MTLCommandBufferStatusCompleted) atomic_store(&um_async_failed, 1);
      int n = atomic_load_explicit(&fins->n, memory_order_acquire);
      if (n > 0) {
        um_gs_gpu_time(done);
        um_gs_ap_lat(done, fins->fire_t);
      }
      for (int j = 0; j < n; j++) um_fin_mark(fins->slot[j], fins->ticket[j]);
      free(fins);
    }
  }];
  [cmd commit];
  um_ap_cmd[k][r] = cmd;
  um_ap_fins[k][r] = fins;
  um_ap[k].committed++;
}

// 枠 k の形で待機を始める（um_gpu_lock と枠の鍵を持って呼ぶ）。うまくいけば 1（state が 2 になる）。
static int um_ap_start(int k) {
  if (!um_awake_start()) return 0;
  int mailbox = um_ap[k].generic;
  if (mailbox) {
    // 郵便受けの列の形: パイプラインは、Buffer をポインタの表から受け取る入口のもの。Buffer は依頼のたびに控える。
    if (!um_mq_init()) return 0;
    for (int d = 0; d < um_ap[k].n; d++) {
      id<MTLComputePipelineState> pipeline = um_pipeline_bl(um_ap[k].d[d].kernel, um_ap[k].d[d].src, um_ap[k].d[d].src_len);
      NSUInteger width = um_resident_width(pipeline, um_ap[k].d[d].group);
      if (width == 0 || um_ap[k].d[d].nbufs > UM_PARK_BUFS ||
          (size_t)(2 + um_ap[k].d[d].nbufs) * 4 + (size_t)um_ap[k].d[d].block_bytes > UM_AP_PTRS) {
        um_ap_clear(k);
        return 0;
      }
      um_ap[k].d[d].width = width;
      um_ap_pipeline[k][d] = pipeline;
    }
    um_ap[k].committed = 0;
    um_ap[k].fired = 0;
    um_ap[k].last_fire = CFAbsoluteTimeGetCurrent();
    um_ap[k].epoch++;
    um_ap[k].state = 2;
    atomic_fetch_add(&um_ap_parked, 1);
    pthread_mutex_lock(&um_mq_lock);
    um_mq_parked[k] = 0;
    um_mq_want[k] = 0;
    pthread_mutex_unlock(&um_mq_lock);
    um_gs_add(&um_gs_ap_starts, 1);
    if (um_trace) fprintf(stderr, "gpu: async resident started for kernel %d (%d in a row, slot %d, mailbox)\n", um_ap[k].d[0].kernel, um_ap[k].n, k);
    return 1;
  }
  um_ap_queue[k] = [um_device newCommandQueueWithMaxCommandBufferCount:64];
  um_ap_event[k] = [um_device newSharedEvent];
  if (!um_ap_queue[k] || !um_ap_event[k]) {
    um_ap_clear(k);
    return 0;
  }
  size_t page = (size_t)getpagesize();
  if (!um_async_wrap) um_async_wrap = [NSMutableDictionary dictionary];
  for (int d = 0; d < um_ap[k].n; d++) {
    id<MTLComputePipelineState> pipeline = um_pipeline(um_ap[k].d[d].kernel, um_ap[k].d[d].src, um_ap[k].d[d].src_len);
    NSUInteger width = um_resident_width(pipeline, um_ap[k].d[d].group);
    if (width == 0) {
      um_ap_clear(k);
      return 0;
    }
    um_ap[k].d[d].width = width;
    um_ap_pipeline[k][d] = pipeline;
    for (int32_t i = 0; i < um_ap[k].d[d].nbufs; i++) {
      const uint64_t *e = &um_ap[k].d[d].sig[i * 4];
      void *data = (void *)(uintptr_t)e[0];
      size_t len = (size_t)e[1];
      size_t size = (len > 0 ? len : 1) * (size_t)e[2];
      size_t rounded = (size + page - 1) / page * page;
      // 今までの経路と同じ GPU 側のバッファを使う（解放のときに um_on_free が控えを見つけられるように）。
      NSNumber *key = @((uintptr_t)data);
      id<MTLBuffer> b = um_async_wrap[key];
      if (!um_wrap_ok(b, rounded, data)) {
        b = [um_device newBufferWithBytesNoCopy:data length:rounded options:um_wrap_options(data) deallocator:nil];
        if (!b) {
          um_ap_clear(k);
          return 0;
        }
        if (!um_async_wrap[key]) atomic_fetch_add(&um_gpu_regions, 1);
        um_async_wrap[key] = b;
      }
      um_ap_buf[k][d * UM_PARK_BUFS + i] = b;
    }
  }
  for (int r = 0; r < UM_AP_RING; r++) {
    id<MTLBuffer> ring = [um_device newBufferWithLength:UM_AP_RINGBYTES options:MTLResourceStorageModeShared];
    if (!ring) {
      um_ap_clear(k);
      return 0;
    }
    memset([ring contents], 0, UM_AP_RINGBYTES);
    um_ap_ring[k][r] = ring;
    um_ap_ring_mem[k][r] = (unsigned char *)[ring contents];
    um_ap_cmd[k][r] = nil;
    um_ap_fins[k][r] = NULL;
  }
  um_ap[k].committed = 0;
  um_ap[k].fired = 0;
  um_ap[k].last_fire = CFAbsoluteTimeGetCurrent();
  um_ap[k].epoch++;
  um_ap[k].state = 2;
  atomic_fetch_add(&um_ap_parked, 1);
  for (int i = 0; i < UM_AP_AHEAD; i++) um_ap_commit_one(k);
  um_gs_add(&um_gs_ap_starts, 1);
  if (um_trace) fprintf(stderr, "gpu: async resident started for kernel %d (%d in a row, slot %d)\n", um_ap[k].d[0].kernel, um_ap[k].n, k);
  return 1;
}

// 枠 k を空ける（um_gpu_lock を持って呼ぶ）。待機中なら、待機分を個数 0 で空振りさせて、全部終わるのを待つ
// （提出した命令は取り消せないため）。
static void um_ap_stop(int k) {
  pthread_mutex_lock(&um_ap_lock[k]);
  if (um_ap[k].state == 2 && um_ap[k].generic) {
    // 郵便受けの列: 列の待機分をまとめて空振りさせ、全部終わるのを待つ（この形の Buffer を使う命令を残さない）。
    // ほかの形の待機分は、次の依頼のときに出し直される。
    @autoreleasepool {
      pthread_mutex_lock(&um_mq_lock);
      um_mq_flush(1);
      pthread_mutex_unlock(&um_mq_lock);
      um_ap_clear(k);
    }
    atomic_fetch_sub(&um_ap_parked, 1);
    um_gs_add(&um_gs_ap_stops, 1);
    if (um_trace) fprintf(stderr, "gpu: async resident stopped for kernel %d (slot %d)\n", um_ap[k].d[0].kernel, k);
  } else if (um_ap[k].state == 2) {
    @autoreleasepool {
      for (uint64_t c = um_ap[k].fired; c < um_ap[k].committed; c++) {
        unsigned char *mem = um_ap_ring_mem[k][c % UM_AP_RING];
        memset(mem + UM_AP_INDIRECT, 0, UM_PARK_SEQ * 16);
        for (int d = 0; d < um_ap[k].n; d++) ((uint32_t *)(mem + (size_t)d * UM_AP_PSTRIDE))[0] = 0;  // count = 0
      }
      um_ap[k].fired = um_ap[k].committed;
      um_ap_event[k].signaledValue = um_ap[k].fired;
      for (int r = 0; r < UM_AP_RING; r++) {
        id<MTLCommandBuffer> cmd = um_ap_cmd[k][r];
        if (cmd && [cmd status] < MTLCommandBufferStatusCompleted) [cmd waitUntilCompleted];
      }
      um_ap_clear(k);
    }
    atomic_fetch_sub(&um_ap_parked, 1);
    um_gs_add(&um_gs_ap_stops, 1);
    if (um_trace) fprintf(stderr, "gpu: async resident stopped for kernel %d (slot %d)\n", um_ap[k].d[0].kernel, k);
  }
  um_ap[k].state = 0;
  um_ap[k].epoch++;
  pthread_mutex_unlock(&um_ap_lock[k]);
}

// 解放されようとしている領域を使う形の枠を空ける（um_on_free から。um_gpu_lock を持って呼ぶ）。
static void um_ap_release(void *p) {
  for (int k = 0; k < UM_AP_SLOTS; k++) {
    if (um_ap[k].state == 0) continue;
    int used = 0;
    for (int i = 0; i < um_ap_res_n[k] && !used; i++) used = um_ap_res_data[k][i] == p;
    for (int d = 0; d < um_ap[k].n && !used && !um_ap[k].generic; d++) {
      for (int32_t i = 0; i < um_ap[k].d[d].nbufs; i++) {
        if ((void *)(uintptr_t)um_ap[k].d[d].sig[i * 4] == p) {
          used = 1;
          break;
        }
      }
    }
    if (used) um_ap_stop(k);
  }
}

// UM_PARK_IDLE_STOP 秒合図の出ていない待機を片付ける（GPU を起こしておくスレッドが呼ぶ。um_park_idle を参照）。
static void um_ap_idle(void) {
  if (atomic_load(&um_ap_parked) == 0) return;
  @autoreleasepool {
    pthread_mutex_lock(&um_gpu_lock);
    CFAbsoluteTime now = CFAbsoluteTimeGetCurrent();
    for (int k = 0; k < UM_AP_SLOTS; k++) {
      if (um_ap[k].state == 2 && now - um_ap[k].last_fire > UM_PARK_IDLE_STOP) um_ap_stop(k);
    }
    if (um_mq_on) {
      // 郵便受けの列: 番号をしばらく書いていなければ、見張りの寿命（1 秒強）より十分前に待機分を空振りさせる。
      pthread_mutex_lock(&um_mq_lock);
      if (um_mq_fired < um_mq_committed && um_gpu_clock() - um_mq_head_since() > UM_MB_IDLE_STOP) um_mq_flush(0);
      pthread_mutex_unlock(&um_mq_lock);
    }
    pthread_mutex_unlock(&um_gpu_lock);
  }
}

// 並び jobs（n 個）の形の枠を探す（um_gpu_lock を持って呼ぶ）。待機中ならその番号を返す。数えている最中なら
// 回数を数え、UM_PARK_AFTER 回に達したら待機を始める（始められたら番号を返す）。それ以外は -1。
// mailbox: 郵便受けの列の形として探す（Buffer の場所を形に含めない）。合図の経路の形とは別の枠になる。
static int um_ap_lookup(const um_job *jobs, int n, int mailbox) {
  um_ap_tick++;
  int k = -1;
  if (mailbox) {
    // 郵便受けの列: もっと長い並びの形が待機していて、この並びがその一部なら、相乗りする（um_ap_embed）。
    for (int i = 0; i < UM_AP_SLOTS; i++) {
      if (um_ap[i].state == 2 && um_ap[i].n > n && um_ap_embed(i, jobs, n, NULL)) {
        um_ap[i].last_use = um_ap_tick;
        return i;
      }
    }
  }
  for (int i = 0; i < UM_AP_SLOTS; i++) {
    if (um_ap[i].generic == mailbox && um_ap_same(i, jobs, n)) {
      k = i;
      break;
    }
  }
  if (k < 0) {
    // 初めて見る形。空きの枠か、数えている最中でいちばん長く使われていない枠に入れる。待機中の枠は追い出さない
    // （使われなくなれば um_ap_idle が空ける）。
    for (int i = 0; i < UM_AP_SLOTS; i++) {
      if (um_ap[i].state == 0) {
        k = i;
        break;
      }
      if (um_ap[i].state == 1 && (k < 0 || um_ap[i].last_use < um_ap[k].last_use)) k = i;
    }
    if (k < 0) return -1;
    pthread_mutex_lock(&um_ap_lock[k]);
    um_ap[k].state = 1;
    um_ap[k].hits = 0;
    um_ap[k].generic = mailbox;
    um_ap[k].n = n;
    for (int d = 0; d < n; d++) {
      um_ap[k].d[d].kernel = jobs[d].kernel;
      um_ap[k].d[d].src = jobs[d].src;
      um_ap[k].d[d].src_len = jobs[d].src_len;
      um_ap[k].d[d].group = jobs[d].group;
      um_ap[k].d[d].nbufs = jobs[d].nbufs;
      um_ap[k].d[d].block_bytes = jobs[d].block_bytes;
      memcpy(um_ap[k].d[d].sig, jobs[d].table, (size_t)jobs[d].nbufs * 4 * sizeof(uint64_t));
    }
    pthread_mutex_unlock(&um_ap_lock[k]);
  }
  um_ap[k].last_use = um_ap_tick;
  if (um_ap[k].state == 2) return k;
  if (++um_ap[k].hits < UM_PARK_AFTER) return -1;
  pthread_mutex_lock(&um_ap_lock[k]);
  int ok = um_ap_start(k);
  pthread_mutex_unlock(&um_ap_lock[k]);
  if (!ok) {
    um_ap[k].hits = -(1 << 30);  // この形は待機できない（数え直さない。枠は数えている最中のまま、追い出されうる）
    return -1;
  }
  return k;
}

// 並び jobs（n 個、券は slots / tickets）を、待機中の枠 k の命令で走らせる。um_gpu_lock を持って呼ぶ
// （別の列の書き込みを待つあいだは放す）。走らせたら 1。待つあいだに枠がやめられていたら 0（呼ぶ側が普通に積む）。
static int um_ap_fire(int k, const um_job *jobs, const int *slots, const uint64_t *tickets, int n) {
  int qid = um_ap[k].generic ? UM_MQ_QID : 2 + k;
  for (int j = 0; j < n; j++) um_async_cross_wait(jobs[j].table, jobs[j].nbufs, qid);
  int map[UM_PARK_SEQ];
  for (int j = 0; j < UM_PARK_SEQ; j++) map[j] = j;
  if (um_ap[k].state != 2 || !(um_ap[k].generic ? um_ap_embed(k, jobs, n, map) : um_ap_same(k, jobs, n))) return 0;
  pthread_mutex_lock(&um_ap_lock[k]);
  int mailbox = um_ap[k].generic;
  uint64_t c;
  int r;
  unsigned char *mem;
  uint64_t gpus[UM_PARK_SEQ][UM_PARK_BUFS];
  if (mailbox) {
    // この仕事の Buffer のポインタを求める。初めての Buffer は控えに足す（出してある待機分は、それを知らないので出し直す）。
    int added = 0;
    for (int d = 0; d < n; d++) {
      for (int32_t i = 0; i < jobs[d].nbufs; i++) {
        void *data = (void *)(uintptr_t)jobs[d].table[i * 4];
        int at = -1;
        for (int x = 0; x < um_ap_res_n[k]; x++) {
          if (um_ap_res_data[k][x] == data) {
            at = x;
            break;
          }
        }
        if (at < 0) {
          id<MTLBuffer> b = um_ap_res_n[k] < UM_AP_RES ? um_ap_wrap(&jobs[d].table[i * 4]) : nil;
          if (!b) {
            pthread_mutex_unlock(&um_ap_lock[k]);
            return 0;
          }
          at = um_ap_res_n[k]++;
          um_ap_res_buf[k][at] = b;
          um_ap_res_data[k][at] = data;
          um_ap_res_gpu[k][at] = [b gpuAddress];
          if ([b hazardTrackingMode] == MTLHazardTrackingModeUntracked) um_ap_res_untracked[k] = 1;
          added = 1;
        }
        gpus[d][i] = um_ap_res_gpu[k][at];
      }
    }
    pthread_mutex_lock(&um_mq_lock);
    if (added) um_mq_flush(0);
    c = um_mq_take(k);
    r = (int)(c % UM_MQ_RING);
    mem = um_mq_ring_mem[r];
    for (int d = 0; d < n; d++) memcpy(mem + (size_t)map[d] * UM_AP_PSTRIDE + UM_AP_PTRS, gpus[d], (size_t)jobs[d].nbufs * 8);
  } else {
    if (um_ap[k].fired == um_ap[k].committed) um_ap_commit_one(k);  // 補充が追いついていない（ふつうは起きない）
    c = um_ap[k].fired;
    r = (int)(c % UM_AP_RING);
    mem = um_ap_ring_mem[k][r];
  }
  for (int d = 0; d < n; d++) {
    const um_job *j = &jobs[d];
    int at = map[d];  // 形の並びの中の位置（相乗りでなければ d）
    uint32_t *words = (uint32_t *)(mem + (size_t)at * UM_AP_PSTRIDE);
    um_resident_grid(j->count, um_ap[k].d[at].group, um_ap[k].d[at].width, words,
                     (uint32_t *)(mem + UM_AP_INDIRECT + (size_t)at * 16));
    for (int32_t i = 0; i < j->nbufs; i++) words[2 + i] = (uint32_t)j->table[i * 4 + 1];
    if (j->block_bytes > 0) memcpy((char *)words + (size_t)(2 + j->nbufs) * 4, j->block, (size_t)j->block_bytes);
  }
  um_ap_fin *fins = mailbox ? um_mq_fins[r] : um_ap_fins[k][r];
  for (int j = 0; j < n; j++) {
    fins->slot[j] = slots[j];
    fins->ticket[j] = tickets[j];
  }
  if (um_gs_on) fins->fire_t = um_gpu_clock();
  atomic_store_explicit(&fins->n, n, memory_order_release);
  id<MTLCommandBuffer> cmd = mailbox ? um_mq_cmd[r] : um_ap_cmd[k][r];
  for (int j = 0; j < n; j++) um_async_cmd[slots[j]] = cmd;
  um_ap[k].last_fire = CFAbsoluteTimeGetCurrent();
  // 引数を書いてから合図（郵便受けなら番号）を進める。待機していた命令は走り出すときに引数を読む。
  if (mailbox) {
    um_mq_go(k, c);
    pthread_mutex_unlock(&um_mq_lock);
  } else {
    um_ap[k].fired++;
    um_ap_event[k].signaledValue = um_ap[k].fired;
    um_mq_kick();  // 郵便受けの列の見張りが回っていれば、どかす（合図の経路の形と混ぜて使っているとき）
  }
  um_ap_refill_k = k;
  um_ap_refill_epoch = um_ap[k].epoch;
  pthread_mutex_unlock(&um_ap_lock[k]);
  um_gs_add(&um_gs_ap_jobs, (uint64_t)n);
  um_gs_add(&um_gs_ap_fires, 1);
  if (um_allow_urgent || um_ap_enabled) {
    if (!um_async_lastwrite) um_async_lastwrite = [NSMutableDictionary dictionary];
    for (int j = 0; j < n; j++) {
      for (int32_t i = 0; i < jobs[j].nbufs; i++) {
        if (jobs[j].table[i * 4 + 3] != 0) {
          um_async_lastwrite[@((uintptr_t)jobs[j].table[i * 4])] = @((tickets[j] << 8) | (uint64_t)qid);
        }
      }
    }
  }
  return 1;
}

// 合図を出した枠 k に待機分を1つ補充する（um_gpu_lock は持たずに、枠の鍵だけで）。
static void um_ap_refill(int k, uint64_t epoch) {
  if (um_ap[k].generic) {
    um_mq_refill();
    return;
  }
  pthread_mutex_lock(&um_ap_lock[k]);
  if (um_ap[k].state == 2 && um_ap[k].epoch == epoch && um_ap[k].committed - um_ap[k].fired < UM_AP_AHEAD) {
    um_ap_commit_one(k);
  }
  pthread_mutex_unlock(&um_ap_lock[k]);
}

// ---- 並び（HOLD でつないだ submit。gpu.submit の常駐）----
// スレッドごとに1つ。中身は um_gpu_lock で守る（どのスレッドからも、積むために触る）。
typedef struct um_chain {
  int n;
  int busy;  // 並びから外して、まだ命令に入れ終えていない回数（そのあいだ、次の並びは走らせない。順番を守るため）
  int used;  // スレッドが持っている
  um_job jobs[UM_PARK_SEQ];
  int slot[UM_PARK_SEQ];
  uint64_t ticket[UM_PARK_SEQ];
  int q[UM_PARK_SEQ];
  int mailbox;  // 並びの仕事のどれかに UM_SUBMIT_MAILBOX が付いていた
  double t_first;  // 並びの最初の仕事を足した時刻（UM_GPU_STATS の、並びを作り始めてから合図までの時間）
} um_chain;
#define UM_CHAINS 1024
static um_chain *um_chains[UM_CHAINS];
static int um_nchains;
static _Thread_local um_chain *um_my_chain;
static pthread_key_t um_chain_key;
static pthread_once_t um_chain_once = PTHREAD_ONCE_INIT;

static void um_job_copy(um_job *dst, const um_job *src) {
  dst->kernel = src->kernel;
  dst->src = src->src;
  dst->src_len = src->src_len;
  dst->count = src->count;
  dst->group = src->group;
  dst->nbufs = src->nbufs;
  dst->block_bytes = src->block_bytes;
  memcpy(dst->table, src->table, (size_t)src->nbufs * 4 * sizeof(uint64_t));
  if (src->block_bytes > 0) memcpy(dst->block, src->block, (size_t)src->block_bytes);
}

// 並び c を走らせる（um_gpu_lock を持って呼ぶ。待つあいだは放す）。try_park なら、形が待機していれば合図で、
// していなければ（数えて）今までの経路で積む。try_park でなければ今までの経路。今までの経路では、最後の仕事以外は
// HOLD として積み、最後の仕事は last_hold に従う。
static void um_chain_run(um_chain *c, int try_park, int last_hold) {
  while (c->busy > 0) {
    pthread_mutex_unlock(&um_gpu_lock);
    sched_yield();
    pthread_mutex_lock(&um_gpu_lock);
  }
  int n = c->n;
  if (n == 0) return;
  um_job jobs[UM_PARK_SEQ];
  int slots[UM_PARK_SEQ], qs[UM_PARK_SEQ];
  uint64_t tickets[UM_PARK_SEQ];
  for (int j = 0; j < n; j++) {
    um_job_copy(&jobs[j], &c->jobs[j]);
    slots[j] = c->slot[j];
    tickets[j] = c->ticket[j];
    qs[j] = c->q[j];
    um_async[slots[j]].chain = NULL;  // ここから命令が決まるまで、待つ人は um_async_cmd_of で待つ
  }
  c->n = 0;
  c->busy++;
  int fired = 0;
  if (try_park) {
    int k = um_ap_lookup(jobs, n, (um_mb_enabled || c->mailbox) && !um_mb_off);
    c->mailbox = 0;
    if (k >= 0) fired = um_ap_fire(k, jobs, slots, tickets, n);
    if (atomic_load(&um_mq_disable_req)) um_mq_disable();
    if (!fired) um_gs_add(&um_gs_ap_normal, (uint64_t)n);
    if (fired && um_gs_on && c->t_first > 0) {
      um_gs_add(&um_gs_chain_n, 1);
      um_gs_add(&um_gs_chain_ns, (uint64_t)((um_gpu_clock() - c->t_first) * 1e9));
    }
  }
  if (!fired) {
    // 追跡しない Buffer を使う並びは、1つの命令に収める（別々の命令に分かれると、Metal は並びの前の仕事の書き込みを
    // 待たない。um_untracked）。ためている命令に入りきらなければ、先にそれを提出する。
    int untracked = 0, sameq = 1;
    for (int j = 0; j < n; j++) {
      sameq &= qs[j] == qs[0];
      for (int32_t i = 0; i < jobs[j].nbufs; i++) untracked |= um_is_untracked((const void *)(uintptr_t)jobs[j].table[i * 4]);
    }
    if (untracked && sameq) {
      pthread_mutex_lock(&um_async_lock);
      if (um_open_n[qs[0]] > 0 && um_open_n[qs[0]] + n > UM_OPEN_MAX) um_async_commit_open(qs[0]);
      pthread_mutex_unlock(&um_async_lock);
    }
    for (int j = 0; j < n; j++) {
      um_async_encode(jobs[j].kernel, jobs[j].src, jobs[j].src_len, jobs[j].count, jobs[j].group, jobs[j].block, jobs[j].block_bytes,
                      jobs[j].table, jobs[j].nbufs, slots[j], tickets[j], qs[j], j < n - 1 ? 1 : last_hold, 0);
    }
  }
  c->busy--;
}

// スレッドが終わるとき: 並びに残っている仕事を積んで（提出して）、並びを空ける。
static void um_chain_exit(void *p) {
  um_chain *c = (um_chain *)p;
  @autoreleasepool {
    pthread_mutex_lock(&um_gpu_lock);
    um_chain_run(c, 0, 0);
    while (c->busy > 0) {
      pthread_mutex_unlock(&um_gpu_lock);
      sched_yield();
      pthread_mutex_lock(&um_gpu_lock);
    }
    c->used = 0;
    pthread_mutex_unlock(&um_gpu_lock);
  }
}
static void um_chain_once_init(void) { pthread_key_create(&um_chain_key, um_chain_exit); }

// 呼んだスレッドの並び（なければ作る）。作れなければ NULL。um_gpu_lock を持って呼ぶ。
static um_chain *um_chain_mine(void) {
  if (um_my_chain) return um_my_chain;
  pthread_once(&um_chain_once, um_chain_once_init);
  um_chain *c = NULL;
  for (int i = 0; i < um_nchains && !c; i++) {
    if (!um_chains[i]->used) c = um_chains[i];
  }
  if (!c) {
    if (um_nchains == UM_CHAINS) return NULL;
    c = (um_chain *)calloc(1, sizeof(um_chain));
    if (!c) return NULL;
    um_chains[um_nchains++] = c;
  }
  c->used = 1;
  c->n = 0;
  c->busy = 0;
  um_my_chain = c;
  pthread_setspecific(um_chain_key, c);
  return c;
}

// 券 t（表の枠 i。呼ぶ側が waiting を付けてある）の命令を返す。終わっていれば nil。um_gpu_lock を持って呼ぶ
// （待つあいだは放す）。まだ並び（HOLD）に入っている券なら、その並びを今までの経路でいま積んで提出する。
// 並びから外して命令に入れている最中（別のスレッドが鍵を放して待っている）なら、決まるまで待つ。
static id<MTLCommandBuffer> um_async_cmd_of(int i, uint64_t t) {
  for (;;) {
    if (um_async[i].id != t || um_async[i].done) return nil;
    if (um_async_cmd[i]) return um_async_cmd[i];
    um_chain *c = um_async[i].chain;
    if (c) {
      @autoreleasepool {
        um_chain_run(c, 0, 0);
      }
      continue;
    }
    pthread_mutex_unlock(&um_gpu_lock);
    sched_yield();
    pthread_mutex_lock(&um_gpu_lock);
  }
}

// どのスレッドの並びも、今までの経路で積んで提出する（gpu.commit）。
static void um_chains_flush_all(void) {
  pthread_mutex_lock(&um_gpu_lock);
  for (int i = 0; i < um_nchains; i++) {
    if (um_chains[i]->used && um_chains[i]->n > 0) um_chain_run(um_chains[i], 0, 0);
  }
  pthread_mutex_unlock(&um_gpu_lock);
}

static uint64_t um_gpu_submit_inner(int32_t kernel, const char *src, int32_t src_len, uint32_t count, uint32_t group,
                                    const void *block, int32_t block_bytes, const uint64_t *table, int32_t nbufs) {
  if (count == 0) return 0;
  if (!um_metal_init()) um_metal_fail("no GPU", nil);
  um_gpu_flush();  // 先に積まれていた仕事は、順番どおり先に終わらせる（このスレッドの分）
  size_t page = (size_t)getpagesize();
  int copies = 0;
  for (int32_t i = 0; i < nbufs; i++) {
    void *data = (void *)(uintptr_t)table[i * 4];
    if (!um_zero_copy || !data || ((uintptr_t)data % page) != 0) copies++;
  }
  if (copies > 0 || nbufs > UM_BATCH_BUFS) {
    // HOLD でためてある自分の並びを先に積む。
    if (um_my_chain) {
      pthread_mutex_lock(&um_gpu_lock);
      um_chain_run(um_my_chain, 0, 0);
      pthread_mutex_unlock(&um_gpu_lock);
    }
    // コピーなしで渡せない場合は、待たずに戻ると写し戻す時点がなくなる。その場で最後まで実行する。
    um_run_alone(kernel, src, src_len, count, group, block, block_bytes, table, nbufs);
    return 0;
  }
  // 実際に使う印。環境変数が印より優先する（エンジンを作り直さずに切り替えて比べるため）。
  uint32_t eff = um_submit_flags;
  if (um_force_check == 0) eff |= UM_SUBMIT_NOCHECK;
  if (um_force_check == 1) eff &= ~UM_SUBMIT_NOCHECK;
  if (!um_allow_urgent) eff &= ~UM_SUBMIT_URGENT;
  if (!um_allow_hold) eff &= ~UM_SUBMIT_HOLD;
  int hold = (eff & UM_SUBMIT_HOLD) != 0;
  int checking = !(eff & UM_SUBMIT_NOCHECK) && um_check_enabled && um_awake_start() && um_check_pipeline && um_check_width > 0;
  if (eff & UM_SUBMIT_NOCHECK) um_gs_add(&um_gs_nocheck, 1);
  if (hold) um_gs_add(&um_gs_hold, 1);
  // 待機命令（常駐）の経路の対象か（「gpu.submit の常駐」）。
  // 既定では急ぎの列の submit だけ。今までの列（0）は、GPU が忙しいと何件も 1 つの命令にまとめて出すので、
  // 投げ続ける使い方（tests/metal/threads_flood.um）では 1 件ずつ待機命令で走らせるより速い（0.56 秒対 1.36 秒）。
  // グループの submit も対象（形にグループの大きさを含める。um_resident_grid）。
  // 郵便受けの印（MAILBOX）の付いた submit は、急ぎでなくても対象（郵便受けの列は命令が隙間なく続くので、今までの列の
  // 「まとめて出す」利点がない）。
  int eligible = um_ap_enabled && !checking && block_bytes <= UM_BATCH_BLOCK &&
                 (um_ap_enabled == 2 || (eff & UM_SUBMIT_URGENT) || ((eff & UM_SUBMIT_MAILBOX) && !um_mb_off));
  um_ap_refill_k = -1;

  pthread_mutex_lock(&um_gpu_lock);
  int q = um_async_queue_for(eff);
  um_chain *c = eligible ? um_chain_mine() : um_my_chain;
  if (!c) eligible = 0;
  // 対象外の submit の前に、ためてある自分の並びを（HOLD のまま）積む。順番を守るため。
  if (!eligible && c && c->n > 0) um_chain_run(c, 0, 1);
  int slot = um_async_take();
  uint64_t ticket = (um_async_seq++ << UM_ASYNC_BITS) | (uint64_t)slot;
  um_async_open_slot(slot, ticket, table, nbufs);
  int held = 0;
  if (eligible) {
    // 並びに足す。並びの終わり（HOLD なし）か、並びがいっぱいなら走らせる。
    if (c->n == UM_PARK_SEQ) um_chain_run(c, 0, 1);  // um_async_take で待つあいだに足された、などの念のため
    int at = c->n;
    if (at == 0 && um_gs_on) c->t_first = um_gpu_clock();
    um_job *j = &c->jobs[at];
    j->kernel = kernel;
    j->src = src;
    j->src_len = src_len;
    j->count = count;
    j->group = group;
    j->nbufs = nbufs;
    j->block_bytes = block_bytes;
    memcpy(j->table, table, (size_t)nbufs * 4 * sizeof(uint64_t));
    if (block_bytes > 0) memcpy(j->block, block, (size_t)block_bytes);
    c->slot[at] = slot;
    c->ticket[at] = ticket;
    c->q[at] = q;
    if (at == 0) c->mailbox = 0;
    if (eff & UM_SUBMIT_MAILBOX) c->mailbox = 1;
    c->n = at + 1;
    um_async[slot].chain = c;
    if (!hold || c->n == UM_PARK_SEQ) um_chain_run(c, 1, hold);
    else held = 1;
  } else {
    um_async_encode(kernel, src, src_len, count, group, block, block_bytes, table, nbufs, slot, ticket, q, hold, checking);
  }
  pthread_mutex_unlock(&um_gpu_lock);
  int fired_k = um_ap_refill_k;
  if (fired_k >= 0) um_ap_refill(fired_k, um_ap_refill_epoch);
  // 続けて投げられることが多いので、GPU を起こしたままにしておく（用意できなければ、そのまま進む）。
  if (um_awake_start()) atomic_store(&um_awake_until, CFAbsoluteTimeGetCurrent() + 0.05);
  if (um_trace) {
    fprintf(stderr, "gpu: submitted kernel %d count %u without waiting (ticket %llu)%s", kernel, count,
            (unsigned long long)ticket, held ? " held in the chain" : "");
    if (fired_k >= 0) fprintf(stderr, " by the async resident slot %d", fired_k);
    fprintf(stderr, "\n");
  }
  return ticket;
}

// `gpu.ready(t)`: 券 t の仕事が終わっていれば 1。待たない。終わっていたら枠を空ける。
// 見つからない券は、もう終わって片付けたもの（または券 0）なので 1。
// 完了の印が付いていれば、Metal にも検算にも聞かずに枠を空けて 1。um_gpu_lock を持って呼ぶ。
static int um_async_fin_fast(int i, uint64_t ticket) {
  if (atomic_load(&um_async_failed)) um_metal_fail("running the kernel", nil);
  if (atomic_load(&um_async_fin[i]) != ticket) return 0;
  um_async_release(i);
  um_gs_add(&um_gs_by_fin, 1);
  return 1;
}

int32_t um_gpu_ready(uint64_t ticket) {
  pthread_mutex_lock(&um_gpu_lock);
  int i = um_async_find(ticket);
  if (i < 0 || um_async[i].done || um_async_fin_fast(i, ticket)) {
    pthread_mutex_unlock(&um_gpu_lock);
    return 1;
  }
  um_check_snap snap;
  um_async_snap(i, &snap);
  um_async[i].waiting++;
  id<MTLCommandBuffer> cmd = um_async_cmd_of(i, ticket);
  pthread_mutex_unlock(&um_gpu_lock);
  // 完了の知らせより前でも、検算が合えば結果は読める（検算は鍵の外で計算する）。
  int finished = !cmd || um_snap_verified(&snap);
  if (!cmd) {
  } else if (finished) {
    um_gs_add(&um_gs_by_check, 1);
  } else {
    @autoreleasepool {  // um_gpu_wait の説明を参照
      if ([cmd status] == MTLCommandBufferStatusNotEnqueued) um_async_flush_held(cmd);
      if ([cmd status] >= MTLCommandBufferStatusCompleted) {
        if ([cmd status] != MTLCommandBufferStatusCompleted) um_metal_fail("running the kernel", [cmd error]);
        finished = 1;
        um_gs_add(&um_gs_by_status, 1);
      }
    }
  }
  pthread_mutex_lock(&um_gpu_lock);
  um_async[i].waiting--;
  if (finished || um_async[i].done) {
    finished = 1;
    um_async_release(i);
  }
  pthread_mutex_unlock(&um_gpu_lock);
  return finished;
}

// `gpu.wait(t)`: 券 t の仕事が終わるまで待つ。短い仕事が多いので、しばらくは寝ずに見続ける（鍵は持たない）。
//
// Metal の呼び出し（命令の status を見る、提出する、完了を待つ）は、オブジェクトを自動解放の枠に入れることがある。
// 生成コードの main には枠がないので、枠の外で呼ぶと、入れたものが二度と解放されず、物理メモリが増え続けた
// （gpu.submit / gpu.wait を回すと、1回あたり数十バイト。leaks では「まだ参照されている」ので見つからない）。
// そこで、Metal を呼ぶ入口（gpu.submit / gpu.ready / gpu.wait / um_gpu_flush / um_on_free と完了の処理）は、
// どれも @autoreleasepool の中で呼ぶ。待つあいだ status を見続けるループは、1回ごとに枠を空にする。
void um_gpu_wait(uint64_t ticket) {
  pthread_mutex_lock(&um_gpu_lock);
  int i = um_async_find(ticket);
  if (i < 0 || um_async[i].done || um_async_fin_fast(i, ticket)) {
    pthread_mutex_unlock(&um_gpu_lock);
    return;
  }
  um_check_snap snap;
  um_async_snap(i, &snap);
  um_async[i].waiting++;
  id<MTLCommandBuffer> cmd = um_async_cmd_of(i, ticket);
  pthread_mutex_unlock(&um_gpu_lock);
  @autoreleasepool {
    if (cmd) um_async_await(cmd, &snap);
    cmd = nil;
  }
  pthread_mutex_lock(&um_gpu_lock);
  um_async[i].waiting--;
  um_async_release(i);
  pthread_mutex_unlock(&um_gpu_lock);
}
