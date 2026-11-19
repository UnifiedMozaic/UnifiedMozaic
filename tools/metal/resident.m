// 常駐カーネルの前提を実機で確かめる（調査用。テストではない）。
//
// GPU 上で走り続ける kernel が、CPU と共有しているメモリの「依頼票」を見張り、依頼が来たら答えを書く。
// 調べること:
//   1. 実行中の kernel が、CPU があとから書いた値を読めるか。CPU は kernel の書いた値を実行中に読めるか。
//   2. 依頼を書いてから答えが見えるまでの時間（コマンドを毎回出す方式と比べるため）。
//   3. 走り続ける kernel を OS が打ち切らないか（何秒まで走れるか）。
//
// 使い方:
//   clang -fobjc-arc -O2 tools/metal/resident.m -framework Metal -framework Foundation -o out/mac-check/resident
//   out/mac-check/resident ping 2000     依頼を 2000 回出して、答えが返るまでの時間を測る
//   out/mac-check/resident hold 30       依頼を出さずに 30 秒走らせ続け、打ち切られないかを見る
//   out/mac-check/resident trace 5       止まって見える時間のあいだ、kernel が走っていたのかどうかを切り分ける
//   out/mac-check/resident gaps 10       10 秒のあいだ、kernel が止まっていた時間の割合といちばん長い停止を測る
//
// 共有メモリの語: [0] 依頼の番号（CPU が書く）、[1] 答えた番号（GPU が書く）、[2] 答えの値（GPU が書く）、
//                [3] 止める合図（CPU が書く）、[4] 見張りの回った回数（GPU が書く）。

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <libkern/OSCacheControl.h>
#include <mach/mach_time.h>
#include <sys/mman.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static NSString *const kSource =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "kernel void resident(device atomic_uint *w [[buffer(0)]], const device uint *scratch [[buffer(1)]], device atomic_uint *rq [[buffer(2)]], uint3 gid [[thread_position_in_grid]]) {\n"
     "  if (gid.x != 0) return;\n"
     "  uint seen = 0;\n"
     "  uint spins = 0;\n"
     "  uint acc = 0;\n"
     "  uint sweep = atomic_load_explicit(&w[6], memory_order_relaxed);\n"
     "  uint stride = atomic_load_explicit(&w[8], memory_order_relaxed);\n"
     "  uint delay = atomic_load_explicit(&w[9], memory_order_relaxed);\n"
     "  uint quiet = atomic_load_explicit(&w[10], memory_order_relaxed);\n"
     "  // 止める合図が来なくても、回数の上限で必ず終わる（実験が GPU を占有し続けないように）。\n"
     // 読む場所は必ず rq[0] と rq[3] だが、添字を毎回メモリから作る（scratch の中身は 0x01010101 なので at は 0）。
     // コンパイラに「毎回同じ場所だから読み出しを減らせる」と判断させないため。w[11] が 0 なら従来どおり直接読む。
     "  uint opaque = atomic_load_explicit(&w[11], memory_order_relaxed);\n"
     "  uint at = 0u;\n"
     "  while (spins != 0xffffffffu && atomic_load_explicit(&rq[at + 3u], memory_order_relaxed) == 0) {\n"
     "    if (opaque != 0u) at = scratch[spins & 1023u] & 0xfeu;\n"
     "    uint req = atomic_load_explicit(&rq[at], memory_order_relaxed);\n"
     "    if (req != seen) {\n"
     "      seen = req;\n"
     "      atomic_store_explicit(&w[2], req * 3u + 1u, memory_order_relaxed);\n"
     "      atomic_store_explicit(&w[1], req, memory_order_relaxed);\n"
     "    }\n"
     "    spins += 1;\n"
     "    if (quiet == 0u || (spins & 65535u) == 1u) atomic_store_explicit(&w[4], spins, memory_order_relaxed);\n"
     // 追い出し: 見張りのたびに別の大きな領域を 64 バイトおきに読む（w[6] 語ぶん）。
     "    for (uint j = 0; j < sweep; j += stride) acc += scratch[j];\n"
     "    if (sweep != 0u) atomic_store_explicit(&w[7], acc, memory_order_relaxed);\n"
     // 待ち: メモリを読まずに計算だけで時間をつぶす（w[9] 回）。追い出しの効果と、時間の効果を分けるため。
     "    for (uint j = 0; j < delay; j++) { acc ^= acc << 13; acc ^= acc >> 17; acc ^= acc << 5; acc += j; }\n"
     "  }\n"
     "}\n"
     // 切り分け用: CPU が w[0] に書き続ける時刻（マイクロ秒）を読み、値が変わるたびに
     // （読んだ時刻、そのときの見張りの回数）を w[16] 以降に記録する。w[5] は記録した組の数。
     "kernel void tracer(device atomic_uint *w [[buffer(0)]], const device uint *scratch [[buffer(1)]], device atomic_uint *rq [[buffer(2)]], uint3 gid [[thread_position_in_grid]]) {\n"
     "  if (gid.x != 0) return;\n"
     "  uint seen = 0;\n"
     "  uint spins = 0;\n"
     "  uint acc = 0;\n"
     "  uint sweep = atomic_load_explicit(&w[6], memory_order_relaxed);\n"
     "  uint stride = atomic_load_explicit(&w[8], memory_order_relaxed);\n"
     "  uint delay = atomic_load_explicit(&w[9], memory_order_relaxed);\n"
     "  uint quiet = atomic_load_explicit(&w[10], memory_order_relaxed);\n"
     "  uint n = 0;\n"
     // 読む場所は必ず rq[0] と rq[3] だが、添字を毎回メモリから作る（scratch の中身は 0x01010101 なので at は 0）。
     // コンパイラに「毎回同じ場所だから読み出しを減らせる」と判断させないため。w[11] が 0 なら従来どおり直接読む。
     "  uint opaque = atomic_load_explicit(&w[11], memory_order_relaxed);\n"
     "  uint at = 0u;\n"
     "  while (spins != 0xffffffffu && atomic_load_explicit(&rq[at + 3u], memory_order_relaxed) == 0) {\n"
     "    if (opaque != 0u) at = scratch[spins & 1023u] & 0xfeu;\n"
     "    uint t = atomic_load_explicit(&rq[at], memory_order_relaxed);\n"
     "    if (t != seen && n < 60000u) {\n"
     "      seen = t;\n"
     "      atomic_store_explicit(&w[16u + n * 2u], t, memory_order_relaxed);\n"
     "      atomic_store_explicit(&w[17u + n * 2u], spins, memory_order_relaxed);\n"
     "      n += 1;\n"
     "      atomic_store_explicit(&w[5], n, memory_order_relaxed);\n"
     "    }\n"
     "    spins += 1;\n"
     "    if (quiet == 0u || (spins & 65535u) == 1u) atomic_store_explicit(&w[4], spins, memory_order_relaxed);\n"
     // 追い出し: 見張りのたびに別の大きな領域を 64 バイトおきに読む（w[6] 語ぶん）。
     "    for (uint j = 0; j < sweep; j += stride) acc += scratch[j];\n"
     "    if (sweep != 0u) atomic_store_explicit(&w[7], acc, memory_order_relaxed);\n"
     // 待ち: メモリを読まずに計算だけで時間をつぶす（w[9] 回）。追い出しの効果と、時間の効果を分けるため。
     "    for (uint j = 0; j < delay; j++) { acc ^= acc << 13; acc ^= acc >> 17; acc ^= acc << 5; acc += j; }\n"
     "  }\n"
     "}\n"
     // 流れ方式: 1つの kernel が回り続けるのではなく、とても多くのスレッドがそれぞれ1回だけ依頼票を読む。
     // 各スレッドは、読んだ時刻を out[自分の番号] に書く。あとで、時刻が細かく進んでいるか（新しい値が読めているか）を見る。
     "kernel void stream(device atomic_uint *rq [[buffer(2)]], device uint *out [[buffer(1)]], uint3 gid [[thread_position_in_grid]]) {\n"
     // out は 0 で埋めてあるので、読む場所は必ず rq[0]。ただしコンパイラには分からないので、
     // 「どのスレッドでも同じ値」とみなして読み出しを1回にまとめることができない。
     "  uint at = out[gid.x] & 1023u;\n"
     "  out[gid.x] = atomic_load_explicit(&rq[at], memory_order_relaxed);\n"
     "}\n";

static double now_ms(void) {
  static mach_timebase_info_data_t tb;
  if (tb.denom == 0) mach_timebase_info(&tb);
  return (double)mach_absolute_time() * tb.numer / tb.denom / 1e6;
}

static int cmp(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return x < y ? -1 : x > y;
}

int main(int argc, char **argv) {
  @autoreleasepool {
    const char *mode = argc > 1 ? argv[1] : "ping";
    if (strcmp(mode, "stream") == 0) {
      // out/mac-check/resident stream 64   → 64 * 2^20 個のスレッドを1回の dispatch で流す
      long millions = argc > 2 ? strtol(argv[2], NULL, 10) : 16;
      id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
      NSError *err = nil;
      id<MTLLibrary> l = [dev newLibraryWithSource:kSource options:nil error:&err];
      id<MTLComputePipelineState> p = l ? [dev newComputePipelineStateWithFunction:[l newFunctionWithName:@"stream"] error:&err] : nil;
      if (!p) {
        fprintf(stderr, "setup: %s\n", [[err localizedDescription] UTF8String]);
        return 1;
      }
      size_t n = (size_t)millions << 20;
      id<MTLBuffer> rqb = [dev newBufferWithLength:(size_t)getpagesize() options:MTLResourceStorageModeShared];
      id<MTLBuffer> outb = [dev newBufferWithLength:n * 4 options:MTLResourceStorageModeShared];
      _Atomic uint32_t *rq = (_Atomic uint32_t *)[rqb contents];
      uint32_t *out = (uint32_t *)[outb contents];
      memset(out, 0, n * 4);
      atomic_store(&rq[0], 1);
      id<MTLCommandBuffer> c = [[dev newCommandQueue] commandBuffer];
      id<MTLComputeCommandEncoder> e = [c computeCommandEncoder];
      [e setComputePipelineState:p];
      [e setBuffer:outb offset:0 atIndex:1];
      [e setBuffer:rqb offset:0 atIndex:2];
      [e dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake([p threadExecutionWidth], 1, 1)];
      [e endEncoding];
      double t0 = now_ms();
      [c commit];
      // 走っているあいだ、CPU は時刻（マイクロ秒）を書き続ける。あわせて、kernel の書いた値が CPU から
      // どこまで見えているかを 0.2 ms ごとに調べ、見えている先頭の進み方を記録する。
      size_t front = 0;
      long front_moves = 0, checks = 0;
      double next_check = 0.2;
      while ([c status] < MTLCommandBufferStatusCompleted) {
        double t = now_ms() - t0;
        atomic_store(&rq[0], (uint32_t)(t * 1000.0) + 2);
        // 3つ目の引数: civac なら、書くたびに CPU のキャッシュをメモリ本体へ書き戻して捨てる命令を実行する。
        //             yield なら、書くたびにスレッドを譲る。
        if (argc > 3 && strcmp(argv[3], "civac") == 0) __asm__ volatile("dc civac, %0" : : "r"(rq));
        if (argc > 3 && strcmp(argv[3], "yield") == 0) sched_yield();
        if (t >= next_check) {
          next_check = t + 0.2;
          checks++;
          size_t old = front;
          while (front < n && out[front] != 0) front += 4096;
          if (front != old) front_moves++;
        }
      }
      double ran = now_ms() - t0;
      size_t distinct = 0, longest = 0, run = 0;
      uint32_t prev = 0, lo = 0xffffffffu, hi = 0;
      for (size_t i = 0; i < n; i++) {
        uint32_t v = out[i];
        if (v < lo) lo = v;
        if (v > hi) hi = v;
        if (v != prev) {
          distinct++;
          run = 0;
        }
        run++;
        if (run > longest) longest = run;
        prev = v;
      }
      printf("threads: %zu, ran %.1f ms, values read span %.1f..%.1f ms\n", n, ran, lo / 1000.0, hi / 1000.0);
      printf("changes of value along the threads: %zu (one per %.1f us of run time), longest run of one value: %zu threads\n",
             distinct, ran * 1000.0 / distinct, longest);
      printf("CPU saw the kernel's output advance in %ld of %ld checks (every 0.2 ms)\n", front_moves, checks);
      return 0;
    }
    long amount = argc > 2 ? strtol(argv[2], NULL, 10) : 1000;
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      fprintf(stderr, "no GPU\n");
      return 1;
    }
    printf("device: %s\n", [[device name] UTF8String]);
    NSError *error = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:kSource options:nil error:&error];
    if (!lib) {
      fprintf(stderr, "compile: %s\n", [[error localizedDescription] UTF8String]);
      return 1;
    }
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:[lib newFunctionWithName:strcmp(mode, "trace") == 0 ? @"tracer" : @"resident"]
                                              error:&error];
    if (!pipeline) {
      fprintf(stderr, "pipeline: %s\n", [[error localizedDescription] UTF8String]);
      return 1;
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];

    // ランタイムと同じやり方: ページ境界の領域をコピーなしで GPU に渡す。
    size_t page = (size_t)getpagesize();
    void *mem = NULL;
    size_t bytes = page * 64;  // 記録用の領域を含む
    // 3つ目の引数: 領域の確保のしかた。nocopy = posix_memalign した領域をそのまま渡す（ランタイムと同じ）、
    // mmap = mmap した領域をそのまま渡す、metal = Metal に確保させる。
    // 4つ目の引数に flush を付けると、CPU は書いたあとと読む前に、その場所のキャッシュを書き戻す。
    const char *alloc = argc > 3 ? argv[3] : "nocopy";
    int flush = argc > 4 && strcmp(argv[4], "flush") == 0;
    id<MTLBuffer> buf = nil;
    if (strcmp(alloc, "metal") == 0) {
      buf = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
      mem = [buf contents];
    } else if (strcmp(alloc, "mmap") == 0) {
      mem = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
      if (mem == MAP_FAILED) return 1;
    } else {
      if (posix_memalign(&mem, page, bytes) != 0) return 1;
    }
    memset(mem, 0, bytes);
    // 5つ目の引数: kernel が見張りのたびに読む「追い出し用」の領域の大きさ（KB）。0 なら読まない。
    long sweep_kb = argc > 5 ? strtol(argv[5], NULL, 10) : 0;
    size_t scratch_bytes = (size_t)(sweep_kb > 0 ? sweep_kb : 16) * 1024;
    id<MTLBuffer> scratch = [device newBufferWithLength:scratch_bytes options:MTLResourceStorageModeShared];
    memset([scratch contents], 1, scratch_bytes);
    ((uint32_t *)mem)[6] = sweep_kb > 0 ? (uint32_t)(scratch_bytes / 4) : 0;
    // 6つ目の引数: 追い出し用の領域を読む間隔（バイト。既定は 64）。
    long stride_bytes = argc > 6 ? strtol(argv[6], NULL, 10) : 64;
    ((uint32_t *)mem)[8] = (uint32_t)(stride_bytes / 4);
    // 7つ目の引数: kernel が見張りのたびに、メモリを読まずに計算だけで待つ回数。
    long delay = argc > 7 ? strtol(argv[7], NULL, 10) : 0;
    ((uint32_t *)mem)[9] = (uint32_t)delay;
    if (delay > 0) printf("delay: %ld iterations\n", delay);
    // 8つ目の引数: split なら、CPU が書く語（依頼 rq[0] と止める合図 rq[3]）を、GPU が書く語とは別の領域に置く。
    //             quiet なら、それに加えて kernel は見張りの回数を毎回は書かない（65536 回に1回）。
    const char *layout = argc > 8 ? argv[8] : "same";
    int split = strcmp(layout, "same") != 0;
    ((uint32_t *)mem)[10] = strcmp(layout, "quiet") == 0;
    id<MTLBuffer> rqbuf = split ? [device newBufferWithLength:page options:MTLResourceStorageModeShared] : nil;
    _Atomic uint32_t *rq = split ? (_Atomic uint32_t *)[rqbuf contents] : (_Atomic uint32_t *)mem;
    if (split) memset((void *)rq, 0, page);
    printf("layout: %s\n", layout);
    // 9つ目の引数: opaque なら、kernel は読む場所の添字を毎回メモリから作る。
    ((uint32_t *)mem)[11] = argc > 9 && strcmp(argv[9], "opaque") == 0;
    printf("alloc: %s, cpu cache flush: %s, sweep: %ld KB, stride: %ld bytes\n", alloc, flush ? "yes" : "no", sweep_kb,
           stride_bytes);
    _Atomic uint32_t *w = (_Atomic uint32_t *)mem;
    if (!buf) buf = [device newBufferWithBytesNoCopy:mem length:bytes options:MTLResourceStorageModeShared deallocator:nil];
    if (!buf) {
      fprintf(stderr, "newBufferWithBytesNoCopy failed\n");
      return 1;
    }

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:buf offset:0 atIndex:0];
    [enc setBuffer:scratch offset:0 atIndex:1];
    [enc setBuffer:(rqbuf ? rqbuf : buf) offset:0 atIndex:2];
    [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [enc endEncoding];
    double t0 = now_ms();
    [cmd commit];

    // kernel が動き出すのを待つ（見張りの回数が増え始める）。
    while (atomic_load(&w[4]) == 0 && now_ms() - t0 < 5000 && [cmd status] < MTLCommandBufferStatusCompleted) usleep(100);
    printf("kernel started after %.3f ms (spins visible to the CPU while running: %s)\n", now_ms() - t0,
           atomic_load(&w[4]) != 0 ? "yes" : "NO");

    if (strcmp(mode, "hold") == 0) {
      // 依頼を出さずに走らせ続ける。1秒ごとに、まだ走っているか（見張りの回数が進んでいるか）を見る。
      uint32_t last = atomic_load(&w[4]);
      for (long s = 1; s <= amount; s++) {
        sleep(1);
        uint32_t cur = atomic_load(&w[4]);
        MTLCommandBufferStatus st = [cmd status];
        printf("t=%lds status=%lu spins/s=%u%s\n", s, (unsigned long)st, cur - last,
               cur == last ? "  (stopped)" : "");
        fflush(stdout);
        last = cur;
        if (st >= MTLCommandBufferStatusCompleted) break;
      }
    } else if (strcmp(mode, "trace") == 0) {
      // CPU は時刻（開始からのマイクロ秒）を書き続ける。kernel は値が変わるたびに、読んだ時刻と見張りの回数を記録する。
      // あとで記録を見て、kernel が「読めなかった時間」のあいだに見張りが回っていたかどうかを調べる。
      //   回っていた → kernel は走っていたが、CPU の書き込みが見えていなかった。
      //   回っていない → kernel そのものが GPU から外されていた。
      double start = now_ms();
      while (now_ms() - start < amount * 1000.0) {
        atomic_store(&rq[0], (uint32_t)((now_ms() - start) * 1000.0) + 1);
        if (flush) sys_dcache_flush((void *)rq, 64);
      }
      atomic_store(&rq[3], 1);
      [cmd waitUntilCompleted];
      uint32_t n = atomic_load(&w[5]);
      // 続く2つの記録のあいだの時間で分けて、時間の合計と、そのあいだに回った見張りの回数を集計する。
      const double edges[] = {100, 1000, 10000, 100000, 1e12};  // マイクロ秒
      const char *names[] = {"< 0.1 ms", "0.1-1 ms", "1-10 ms", "10-100 ms", ">= 100 ms"};
      double time_sum[5] = {0}, spin_sum[5] = {0};
      long count[5] = {0};
      for (uint32_t i = 1; i < n; i++) {
        double dt = (double)(w[16 + i * 2] - w[16 + (i - 1) * 2]);
        double ds = (double)(w[17 + i * 2] - w[17 + (i - 1) * 2]);
        int k = 0;
        while (dt >= edges[k]) k++;
        count[k]++;
        time_sum[k] += dt;
        spin_sum[k] += ds;
      }
      printf("records: %u, first at %.1f ms, last at %.1f ms, total spins %u\n", n, n ? w[16] / 1000.0 : 0,
             n ? w[16 + (n - 1) * 2] / 1000.0 : 0, atomic_load(&w[4]));
      printf("%-10s %8s %12s %14s %16s\n", "gap", "count", "time (ms)", "spins", "spins per ms");
      for (int k = 0; k < 5; k++) {
        printf("%-10s %8ld %12.1f %14.0f %16.0f\n", names[k], count[k], time_sum[k] / 1000.0, spin_sum[k],
               time_sum[k] > 0 ? spin_sum[k] / (time_sum[k] / 1000.0) : 0);
      }
    } else if (strcmp(mode, "gaps") == 0) {
      // 1 ms ごとに見張りの回数を読み、kernel が進んでいなかった区間（GPU から外されていた時間）を数える。
      long stalled = 0, samples = 0;
      double longest = 0, run_start = now_ms(), stall_start = 0;
      uint32_t last = atomic_load(&w[4]);
      while (now_ms() - run_start < amount * 1000.0) {
        usleep(1000);
        uint32_t cur = atomic_load(&w[4]);
        double t = now_ms();
        samples++;
        if (cur == last) {
          if (stall_start == 0) stall_start = t;
          stalled++;
          if (t - stall_start > longest) longest = t - stall_start;
        } else {
          stall_start = 0;
        }
        last = cur;
      }
      printf("samples: %ld, with no progress: %ld (%.1f%%), longest stall: %.1f ms, spins: %u\n", samples, stalled,
             100.0 * stalled / samples, longest, atomic_load(&w[4]));
    } else {
      double *lat = malloc(sizeof(double) * (size_t)amount);
      long wrong = 0, lost = 0;
      for (long i = 0; i < amount; i++) {
        uint32_t req = (uint32_t)i + 1;
        double a = now_ms();
        atomic_store(&rq[0], req);
        if (flush) sys_dcache_flush((void *)rq, 64);
        while (atomic_load(&w[1]) != req) {
          if (flush) sys_dcache_flush((void *)rq, 64);
          if (now_ms() - a > 2000) {
            lost++;
            break;
          }
        }
        lat[i] = now_ms() - a;
        if (atomic_load(&w[2]) != req * 3u + 1u) wrong++;
        usleep(200);  // 依頼の間を空ける（kernel は待機中も回り続ける）
      }
      qsort(lat, (size_t)amount, sizeof(double), cmp);
      printf("requests: %ld, wrong answers: %ld, no answer within 2 s: %ld\n", amount, wrong, lost);
      printf("latency ms: min %.4f  median %.4f  p99 %.4f  max %.4f\n", lat[0], lat[amount / 2],
             lat[amount * 99 / 100], lat[amount - 1]);
      free(lat);
    }

    // 止める合図を書き、kernel が自分で終わるのを待つ。
    double t1 = now_ms();
    atomic_store(&rq[3], 1);
    [cmd waitUntilCompleted];
    printf("stop: status=%lu after %.3f ms, total run %.1f ms, error: %s\n", (unsigned long)[cmd status], now_ms() - t1,
           now_ms() - t0, [cmd error] ? [[[cmd error] localizedDescription] UTF8String] : "none");
  }
  return 0;
}
