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
//   um_async_lock: ためている命令（um_open_cmd）。Metal の完了の処理（別のスレッド）からも取る。
//   um_table_lock: パイプラインの表と自動判断の実測値（um_stats）。
// まだ提出していない dispatch（um_pending）は文の並びの話なので、スレッドごとに持つ。
static pthread_mutex_t um_table_lock = PTHREAD_MUTEX_INITIALIZER;
// GPU が覚えている領域があるか（gpu.submit の控えの数 + 常駐の枠の数）。0 なら um_on_free は鍵を取らずに戻る。
static _Atomic int um_gpu_regions;
typedef void (*um_free_fn)(void *);
extern _Atomic(um_free_fn) um_free_hook;
extern _Noreturn void um_exit_failure(void);
static void um_on_free(void *p);

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

// 環境変数を読む（1回だけ）。GPU の準備（um_metal_init）とは分けてある。CPU だけで済むプログラムでは、
// GPU の準備（数十ミリ秒かかる）をしないで済ませるため。
static pthread_once_t um_env_once = PTHREAD_ONCE_INIT;
static int um_gpu_off = 0;  // UM_GPU=0 で GPU を使わない
static void um_env_init_once(void);
static void um_env_init(void) { pthread_once(&um_env_once, um_env_init_once); }
static void um_env_init_once(void) {
  const char *off = getenv("UM_GPU");
  if (off && strcmp(off, "0") == 0) um_gpu_off = 1;
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
  uint64_t table[UM_BATCH_BUFS * 4];
  unsigned char block[UM_BATCH_BLOCK];
} um_job;
static _Thread_local um_job um_pending[UM_MAX_BATCH];
static _Thread_local int um_pending_len;

void um_gpu_flush(void);
static id<MTLComputePipelineState> um_pipeline(int32_t kernel, const char *src, int32_t src_len);

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
    NSUInteger width;
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

static void um_park_idle(void);
static void *um_awake_main(void *arg) {
  (void)arg;
  id<MTLBuffer> mine = um_awake_buf;
  id<MTLCommandQueue> q = [um_device newCommandQueue];
  CFAbsoluteTime next_idle = 0;
  for (;;) {
    CFAbsoluteTime now = CFAbsoluteTimeGetCurrent();
    if (now > atomic_load(&um_awake_until)) {
      usleep(2000);
      um_park_idle();
      continue;
    }
    // 起きているあいだも待機の片付けを見る。gpu.submit やほかの形の合図が続くと um_awake_until が延び続け、
    // 休む側の分岐に来ないので、使われなくなった形の待機が残って Metal に打ち切られていた。
    if (now >= next_idle) {
      next_idle = now + 0.1;
      um_park_idle();
    }
    @autoreleasepool {
      id<MTLCommandBuffer> cmd = [q commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
      [enc setComputePipelineState:um_awake_pipeline];
      [enc setBuffer:mine offset:0 atIndex:0];
      [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      [enc endEncoding];
      [cmd commit];
      [cmd waitUntilCompleted];
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
options:nil error:&error];
  id<MTLFunction> fn = lib ? [lib newFunctionWithName:@"um_awake"] : nil;
  id<MTLFunction> check = lib ? [lib newFunctionWithName:@"um_check"] : nil;
  um_check_pipeline = check ? [um_device newComputePipelineStateWithFunction:check error:&error] : nil;
  um_check_width = um_check_pipeline ? [um_check_pipeline threadExecutionWidth] : 0;
  um_awake_pipeline = fn ? [um_device newComputePipelineStateWithFunction:fn error:&error] : nil;
  um_awake_buf = [um_device newBufferWithLength:64 options:MTLResourceStorageModeShared];
  if (!um_awake_pipeline || !um_awake_buf) return;
  memset([um_awake_buf contents], 0, 64);
  ((uint32_t *)[um_awake_buf contents])[2] = um_awake_lengths[0];
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
static id<MTLBuffer> um_async_params[UM_ASYNC_MAX];
// 投げられた仕事は、GPU に余裕があればすぐ提出する。先に出した命令が UM_INFLIGHT_MAX 個たまっているあいだに
// 投げられた分は、1つの命令（um_open_cmd）にためておき、先の命令が1つ終わった時点でまとめて提出する。書く側は1つずつ
// 投げるだけで、提出の手間（1回 約 0.1 ms）が何件かで分け合われる。GPU が空いているときは待たせないので、
// それで遅くなることはない。
// 前の提出が終わったことは、Metal が別のスレッドで呼ぶ完了の処理で知る。そこからも um_open_cmd を
// 触るので、um_async_lock で守る。
static pthread_mutex_t um_async_lock = PTHREAD_MUTEX_INITIALIZER;
static id<MTLCommandBuffer> um_open_cmd;       // ためている最中の命令（まだ提出していない）
static id<MTLComputeCommandEncoder> um_open_enc;
static int um_open_n;                          // そこに入っている仕事の数
static int um_inflight;                        // 提出して、まだ終わっていない命令の数
// 命令をこの数まで出しているあいだは、次をためる。GPU は命令を数個まで重ねて処理できるので、1 にすると
// かえって遅い。実測（32 局面の評価を 8 件預けて回す形、Apple M4）: 1 → 84、2〜3 → 34、4 → 30、
// 6 → 29、ためない → 38 マイクロ秒/件。
#define UM_INFLIGHT_MAX 4
// 1つの命令にためる仕事の上限（これに達したら、GPU が忙しくても提出する）。
#define UM_OPEN_MAX 32

// ためてある命令を提出する。um_async_lock を持った状態で呼ぶ。
static void um_async_commit_open(void) {
  if (!um_open_cmd || um_open_n == 0) return;
  id<MTLCommandBuffer> cmd = um_open_cmd;
  [um_open_enc endEncoding];
  [cmd addCompletedHandler:^(id<MTLCommandBuffer> done) {
    // 終わったら、そのあいだにたまった分を続けて提出する。Metal のスレッドで呼ばれるので、ここにも
    // 自動解放の枠を置く（um_gpu_wait の説明を参照）。
    (void)done;
    @autoreleasepool {
      pthread_mutex_lock(&um_async_lock);
      um_inflight--;
      if (um_inflight < UM_INFLIGHT_MAX) um_async_commit_open();
      pthread_mutex_unlock(&um_async_lock);
    }
  }];
  um_inflight++;
  um_open_cmd = nil;
  um_open_enc = nil;
  um_open_n = 0;
  [cmd commit];
}

// cmd がまだためている最中の命令なら、いま提出する（待つ前に必ず呼ぶ）。どの鍵を持っていてもよい（um_async_lock 以外）。
static void um_async_flush_cmd(id<MTLCommandBuffer> cmd) {
  pthread_mutex_lock(&um_async_lock);
  if (cmd == um_open_cmd) um_async_commit_open();
  pthread_mutex_unlock(&um_async_lock);
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
  um_async_free[um_async_nfree++] = i;
  um_async_live--;
}

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
    if (s && um_snap_verified(s)) return 1;
    if (CFAbsoluteTimeGetCurrent() > until) break;
    // 見張るあいだは CPU を譲る。コアの数より多いスレッドが GPU を待つとき、待っているだけのスレッドが
    // 計算するスレッドの CPU を奪わないように（ほかに動けるスレッドがなければ、すぐ戻ってくる）。
    sched_yield();
  }
  @autoreleasepool {
    if ([cmd status] < MTLCommandBufferStatusCompleted) [cmd waitUntilCompleted];
    if ([cmd status] != MTLCommandBufferStatusCompleted) um_metal_fail("running the kernel", [cmd error]);
  }
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
    id<MTLCommandBuffer> cmd = um_async_cmd[oldest];
    um_async[oldest].waiting++;
    pthread_mutex_unlock(&um_gpu_lock);
    um_async_await(cmd, NULL);
    pthread_mutex_lock(&um_gpu_lock);
    um_async[oldest].waiting--;
    if (um_async[oldest].id == t) um_async_release(oldest);
  }
}

// 領域が解放される前に呼ばれる（um_free から。ページ境界の領域だけ）。その領域を使っている GPU の仕事を片付ける。
static void um_on_free_inner(void *p);
static void um_on_free(void *p) {
  if (atomic_load(&um_gpu_regions) == 0) return;
  @autoreleasepool {  // um_gpu_wait の説明を参照
    um_on_free_inner(p);
  }
}
static void um_on_free_inner(void *p) {
  um_park_release(p);
  pthread_mutex_lock(&um_gpu_lock);
  NSNumber *key = @((uintptr_t)p);
  if (um_async_wrap && um_async_wrap[key]) {
    // この領域を使う提出が終わるまで待つ（鍵は放して待つ）。取っておいた GPU 側のバッファは、そのあと捨てる。
    for (;;) {
      int found = -1;
      for (int i = 0; i < UM_ASYNC_MAX && um_async_live > 0 && found < 0; i++) {
        if (um_async[i].id == 0 || um_async[i].done) continue;
        for (int b = 0; b < um_async[i].nbufs; b++) {
          if (um_async[i].data[b] == p) {
            found = i;
            break;
          }
        }
      }
      if (found < 0) break;
      uint64_t t = um_async[found].id;
      id<MTLCommandBuffer> cmd = um_async_cmd[found];
      um_async[found].waiting++;
      pthread_mutex_unlock(&um_gpu_lock);
      um_async_await(cmd, NULL);
      pthread_mutex_lock(&um_gpu_lock);
      um_async[found].waiting--;
      if (um_async[found].id == t) um_async_release(found);
    }
    [um_async_wrap removeObjectForKey:key];
    atomic_fetch_sub(&um_gpu_regions, 1);
  }
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
      NSUInteger width = [pipeline threadExecutionWidth];
      size_t header = (size_t)(2 + nbufs) * 4;
      id<MTLBuffer> params = [um_device newBufferWithLength:header + (size_t)block_bytes options:MTLResourceStorageModeShared];
      id<MTLBuffer> indirect = [um_device newBufferWithLength:12 options:MTLResourceStorageModeShared];
      // 1行の幅をスレッドのまとまりで割り切れないと、個数を実行時に決められない。
      if (width == 0 || UM_ROW % width != 0 || !params || !indirect) {
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
        um_check[k].c[c].whole = e[3] == 2;
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
    ((uint32_t *)[um_awake_buf contents])[2] = um_awake_lengths[which];
    for (int d = 0; d < um_slot[k].n; d++) {
      const um_job *j = &jobs[d];
      NSUInteger width = um_slot[k].d[d].width;
      uint32_t count = j->count;
      uint32_t row = count < UM_ROW ? (uint32_t)((count + width - 1) / width * width) : UM_ROW;
      uint32_t rows = (count + row - 1) / row;
      uint32_t *words = (uint32_t *)[um_slot_params[k][d] contents];
      words[0] = count;
      words[1] = row;
      for (int32_t i = 0; i < j->nbufs; i++) words[2 + i] = (uint32_t)j->table[i * 4 + 1];
      if (j->block_bytes > 0) memcpy((char *)words + (size_t)(2 + j->nbufs) * 4, j->block, (size_t)j->block_bytes);
      uint32_t *groups = (uint32_t *)[um_slot_indirect[k][d] contents];
      groups[0] = row / (uint32_t)width;
      groups[1] = rows;
      groups[2] = 1;
      um_park_counts[d] = count;
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
    // GPU が走っているあいだに、待機分を1つ補充する。
    um_park_commit_one(k);
  }
}

static int um_park_same(int k, const um_job *jobs, int n) {
  if (um_slot[k].state == 0 || um_slot[k].n != n) return 0;
  for (int d = 0; d < n; d++) {
    if (um_slot[k].d[d].kernel != jobs[d].kernel || um_slot[k].d[d].src != jobs[d].src ||
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
      uint32_t count = j->count;
      uint32_t row = count < UM_ROW ? count : UM_ROW;
      uint32_t rows = (count + row - 1) / row;

      // 引数の塊。
      size_t header = (size_t)(2 + j->nbufs) * 4;
      id<MTLBuffer> params = [um_device newBufferWithLength:header + (size_t)j->block_bytes options:MTLResourceStorageModeShared];
      if (!params) um_metal_fail("allocating the parameter block", nil);
      uint32_t *words = (uint32_t *)[params contents];
      words[0] = count;
      words[1] = row;
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
      NSUInteger width = [pipeline threadExecutionWidth];
      if (width > row) width = row;
      [enc dispatchThreads:MTLSizeMake(row, rows, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    }
    [enc endEncoding];
    [cmd commit];
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
static void um_run_alone(int32_t kernel, const char *src, int32_t src_len, uint32_t count, const void *block,
                         int32_t block_bytes, const uint64_t *table, int32_t nbufs) {
  @autoreleasepool {
    CFAbsoluteTime started = CFAbsoluteTimeGetCurrent();
    size_t page = (size_t)getpagesize();
    id<MTLComputePipelineState> pipeline = um_pipeline(kernel, src, src_len);
    uint32_t row = count < UM_ROW ? count : UM_ROW;
    uint32_t rows = (count + row - 1) / row;
    size_t header = (size_t)(2 + nbufs) * 4;
    id<MTLBuffer> params = [um_device newBufferWithLength:header + (size_t)block_bytes options:MTLResourceStorageModeShared];
    if (!params) um_metal_fail("allocating the parameter block", nil);
    uint32_t *words = (uint32_t *)[params contents];
    words[0] = count;
    words[1] = row;
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
    NSUInteger width = [pipeline threadExecutionWidth];
    if (width > row) width = row;
    [enc dispatchThreads:MTLSizeMake(row, rows, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    [enc endEncoding];
    [cmd commit];
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
    if (um_resident && n <= UM_PARK_SEQ && um_park_try(um_pending, n)) return;
    um_submit(um_pending, n);
  }
}

// dispatch を1つ積む。実行と完了待ちは um_gpu_flush で行う（生成コードが、次の文が dispatch でなければすぐ呼ぶ）。
void um_gpu_dispatch(int32_t kernel, const char *src, int32_t src_len, uint32_t count, const void *block,
                     int32_t block_bytes, const uint64_t *table, int32_t nbufs) {
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
    um_run_alone(kernel, src, src_len, count, block, block_bytes, table, nbufs);
    return;
  }
  if (um_pending_len == UM_MAX_BATCH) um_gpu_flush();
  um_job *j = &um_pending[um_pending_len++];
  j->kernel = kernel;
  j->src = src;
  j->src_len = src_len;
  j->count = count;
  j->nbufs = nbufs;
  j->block_bytes = block_bytes;
  memcpy(j->table, table, (size_t)nbufs * 4 * sizeof(uint64_t));
  if (block_bytes > 0) memcpy(j->block, block, (size_t)block_bytes);
}

// `gpu.submit`: kernel の実行を GPU に頼み、終わるのを待たずに戻る。券を返す（0 は「もう終わっている」）。
static uint64_t um_gpu_submit_inner(int32_t kernel, const char *src, int32_t src_len, uint32_t count, const void *block,
                                    int32_t block_bytes, const uint64_t *table, int32_t nbufs);
uint64_t um_gpu_submit(int32_t kernel, const char *src, int32_t src_len, uint32_t count, const void *block,
                       int32_t block_bytes, const uint64_t *table, int32_t nbufs) {
  @autoreleasepool {  // um_gpu_wait の説明を参照
    return um_gpu_submit_inner(kernel, src, src_len, count, block, block_bytes, table, nbufs);
  }
}
static uint64_t um_gpu_submit_inner(int32_t kernel, const char *src, int32_t src_len, uint32_t count, const void *block,
                                    int32_t block_bytes, const uint64_t *table, int32_t nbufs) {
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
    // コピーなしで渡せない場合は、待たずに戻ると写し戻す時点がなくなる。その場で最後まで実行する。
    um_run_alone(kernel, src, src_len, count, block, block_bytes, table, nbufs);
    return 0;
  }
  // 鍵を取る前に済ませられるもの。
  id<MTLComputePipelineState> pipeline = um_pipeline(kernel, src, src_len);
  int checking = um_check_enabled && um_awake_start() && um_check_pipeline && um_check_width > 0;
  uint32_t row = count < UM_ROW ? count : UM_ROW;
  uint32_t rows = (count + row - 1) / row;
  size_t header = (size_t)(2 + nbufs) * 4;
  NSUInteger width = [pipeline threadExecutionWidth];
  if (width > row) width = row;

  pthread_mutex_lock(&um_gpu_lock);
  int slot = um_async_take();
  uint64_t ticket = (um_async_seq++ << UM_ASYNC_BITS) | (uint64_t)slot;
  // この枠の前の提出は終わっているので、引数の領域は使い回せる。
  id<MTLBuffer> params = um_async_params[slot];
  if (!params || [params length] < header + (size_t)block_bytes) {
    size_t want = header + (size_t)block_bytes;
    params = [um_device newBufferWithLength:(want < 256 ? 256 : want) options:MTLResourceStorageModeShared];
    if (!params) um_metal_fail("allocating the parameter block", nil);
    um_async_params[slot] = params;
  }
  uint32_t *words = (uint32_t *)[params contents];
  words[0] = count;
  words[1] = row;
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
    if (!b || [b length] != rounded) {
      b = [um_device newBufferWithBytesNoCopy:data length:rounded options:MTLResourceStorageModeShared deallocator:nil];
      if (!b) um_metal_fail("sharing a buffer with the GPU", nil);
      if (!um_async_wrap[key]) atomic_fetch_add(&um_gpu_regions, 1);
      um_async_wrap[key] = b;
    }
    made[i] = b;
    um_async[slot].data[i] = data;
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
  while (!um_open_cmd) {
    // 命令は um_async_lock の外で作る。キューの「終わっていない命令」が上限（64 個）に達していると、作る呼び出しは
    // 先の命令が終わるまで待つ。終わりの処理（um_async_commit_open の完了の処理）は um_async_lock を取るので、
    // 鍵を持ったまま待つと、どちらも進まなくなる（多くのスレッドが投げ続けて、ためた命令が UM_OPEN_MAX ごとに
    // 提出され続けたときに起きた）。命令を作るのは um_gpu_lock を持つこの経路だけなので、鍵を取り直したあとも
    // ふつうは空のまま。完了の処理が提出して空にした場合も、ここで作り直す。
    pthread_mutex_unlock(&um_async_lock);
    id<MTLCommandBuffer> fresh = [um_queue commandBuffer];
    if (!fresh) um_metal_fail("creating the command buffer", nil);
    pthread_mutex_lock(&um_async_lock);
    if (!um_open_cmd) {
      um_open_cmd = fresh;
      um_open_enc = [um_open_cmd computeCommandEncoder];
      if (!um_open_enc) um_metal_fail("creating the command buffer", nil);
      um_open_n = 0;
    }
  }
  [um_open_enc setComputePipelineState:pipeline];
  [um_open_enc setBuffer:params offset:0 atIndex:0];
  for (int32_t i = 0; i < nbufs; i++) [um_open_enc setBuffer:made[i] offset:0 atIndex:(NSUInteger)(i + 1)];
  [um_open_enc dispatchThreads:MTLSizeMake(row, rows, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
  // 検算。仕事が書く Buffer と同じ GPU 側のバッファを読むので、仕事のあとに実行される。
  for (int c = 0; c < um_async[slot].nchecks; c++) {
    NSUInteger groups = (um_async[slot].check_words[c] + 63) / 64;
    [um_open_enc setComputePipelineState:um_check_pipeline];
    [um_open_enc setBuffer:um_async_sums[slot][c] offset:0 atIndex:0];
    [um_open_enc setBuffer:check_data[c] offset:0 atIndex:1];
    [um_open_enc setBuffer:um_async_check_params[slot][c] offset:0 atIndex:2];
    [um_open_enc dispatchThreads:MTLSizeMake(groups, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(um_check_width < groups ? um_check_width : groups, 1, 1)];
  }
  um_open_n++;
  um_async_cmd[slot] = um_open_cmd;
  // GPU が空いていれば（前の提出が終わっていれば）すぐ提出する。忙しければ、ためておく。
  if (um_inflight < UM_INFLIGHT_MAX || um_open_n >= UM_OPEN_MAX) um_async_commit_open();
  pthread_mutex_unlock(&um_async_lock);
  um_async[slot].id = ticket;
  um_async[slot].nbufs = nbufs;
  um_async[slot].waiting = 0;
  um_async[slot].done = 0;
  um_async_live++;
  pthread_mutex_unlock(&um_gpu_lock);
  // 続けて投げられることが多いので、GPU を起こしたままにしておく（用意できなければ、そのまま進む）。
  if (um_awake_start()) atomic_store(&um_awake_until, CFAbsoluteTimeGetCurrent() + 0.05);
  if (um_trace) fprintf(stderr, "gpu: submitted kernel %d count %u without waiting (ticket %llu)\n", kernel, count,
                        (unsigned long long)ticket);
  return ticket;
}

// `gpu.ready(t)`: 券 t の仕事が終わっていれば 1。待たない。終わっていたら枠を空ける。
// 見つからない券は、もう終わって片付けたもの（または券 0）なので 1。
int32_t um_gpu_ready(uint64_t ticket) {
  pthread_mutex_lock(&um_gpu_lock);
  int i = um_async_find(ticket);
  if (i < 0 || um_async[i].done) {
    pthread_mutex_unlock(&um_gpu_lock);
    return 1;
  }
  um_check_snap snap;
  um_async_snap(i, &snap);
  id<MTLCommandBuffer> cmd = um_async_cmd[i];
  um_async[i].waiting++;
  pthread_mutex_unlock(&um_gpu_lock);
  // 完了の知らせより前でも、検算が合えば結果は読める（検算は鍵の外で計算する）。
  int finished = um_snap_verified(&snap);
  if (!finished) {
    @autoreleasepool {  // um_gpu_wait の説明を参照
      if ([cmd status] >= MTLCommandBufferStatusCompleted) {
        if ([cmd status] != MTLCommandBufferStatusCompleted) um_metal_fail("running the kernel", [cmd error]);
        finished = 1;
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
  if (i < 0 || um_async[i].done) {
    pthread_mutex_unlock(&um_gpu_lock);
    return;
  }
  um_check_snap snap;
  um_async_snap(i, &snap);
  id<MTLCommandBuffer> cmd = um_async_cmd[i];
  um_async[i].waiting++;
  pthread_mutex_unlock(&um_gpu_lock);
  @autoreleasepool {
    um_async_await(cmd, &snap);
    cmd = nil;
  }
  pthread_mutex_lock(&um_gpu_lock);
  um_async[i].waiting--;
  um_async_release(i);
  pthread_mutex_unlock(&um_gpu_lock);
}
