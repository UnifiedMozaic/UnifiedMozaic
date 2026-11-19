// 完了を早く、かつ安全に知る方法の実験（調査用。テストではない）。
//
// Metal の完了の知らせは、GPU が終わってから数十マイクロ秒遅れて届く。一方、GPU が書いた値は、
// それより前から CPU に見え始めるが、場所ごとに見えるタイミングが違う（終わりの印だけ先に見えることがある）。
// そこで、仕事のあとに GPU に出力の検算値（64 語ごとの和）と終わりの印を書かせ、CPU は印が見えたら
// 自分に見えている出力から同じ検算をして比べる。全部一致したら、出力は確かに見えている。
// 一致しなければ待ち直し、一定時間で合わなければ Metal の完了の知らせを待つ。
//
// 測るもの: 合図を出してから「検算が一致する」まで、「Metal が完了を知らせる」まで、
//           印が見えた時点で検算がまだ合わなかった回数、検算で通したのに実際は違っていた回数（0 のはず）。
//
// 使い方:
//   clang -fobjc-arc -O2 tools/metal/verify.m -framework Metal -framework Foundation -o out/mac-check/verify
//   out/mac-check/verify 1024      出力の語数（既定 1024）

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach_time.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static NSString *const kSource =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     // 仕事: 出力の各語を、依頼の番号から決まる値にする。少し計算して時間をかける。
     "kernel void job(device uint *out [[buffer(0)]], const device uint *req [[buffer(1)]], uint3 gid [[thread_position_in_grid]]) {\n"
     "  uint h = req[0] * 2654435761u + gid.x;\n"
     "  for (uint k = 0; k < req[2]; k++) h = (h * 1664525u + 1013904223u) ^ (h >> 13);\n"
     "  if (gid.x < req[1]) out[gid.x] = h;\n"
     "}\n"
     // 検算: スレッド g が、出力の 64 語ぶんの和（語の位置も混ぜる）を sums[g] に書く。
     "kernel void check(device uint *sums [[buffer(0)]], const device uint *out [[buffer(1)]], const device uint *req [[buffer(2)]], uint3 gid [[thread_position_in_grid]]) {\n"
     "  uint words = req[1];\n"
     "  uint base = gid.x * 64u;\n"
     "  if (base >= words) return;\n"
     // 和には依頼の番号とスレッドの番号を混ぜる。前回の検算値が古いまま見えていても、今回の検算とは一致しない。
     "  uint s = req[0] * 2654435761u + gid.x * 40503u;\n"
     "  for (uint j = base; j < min(base + 64u, words); j++) s += (out[j] ^ (j * 2654435761u)) * 2246822519u;\n"
     "  sums[gid.x] = s;\n"
     "}\n"
     // 終わりの印: 依頼の番号を写す。sums に依存させて、検算のあとに実行させる。
     "kernel void mark(device uint *m [[buffer(0)]], const device uint *req [[buffer(1)]], device uint *sums [[buffer(2)]], uint3 gid [[thread_position_in_grid]]) {\n"
     "  if (gid.x == 0u) m[0] = req[0];\n"
     "}\n"
     "kernel void busy(device uint *s [[buffer(0)]], uint3 gid [[thread_position_in_grid]]) {\n"
     "  uint acc = gid.x;\n"
     "  for (uint j = 0; j < s[0]; j++) { acc ^= acc << 13; acc ^= acc >> 17; acc ^= acc << 5; acc += j; }\n"
     "  s[1] = acc;\n"
     "}\n";

static double now_s(void) {
  static mach_timebase_info_data_t tb;
  if (tb.denom == 0) mach_timebase_info(&tb);
  return (double)mach_absolute_time() * tb.numer / tb.denom / 1e9;
}

static int cmp(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return x < y ? -1 : x > y;
}

static id<MTLDevice> device;
static id<MTLComputePipelineState> busy_p;
static _Atomic int warm_stop;
static id<MTLBuffer> warm_buf;

static void *warm_thread(void *arg) {
  (void)arg;
  id<MTLCommandQueue> q = [device newCommandQueue];
  while (!atomic_load(&warm_stop)) {
    @autoreleasepool {
      id<MTLCommandBuffer> cmd = [q commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
      [enc setComputePipelineState:busy_p];
      [enc setBuffer:warm_buf offset:0 atIndex:0];
      [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      [enc endEncoding];
      [cmd commit];
      [cmd waitUntilCompleted];
    }
  }
  return NULL;
}

static uint32_t expected(uint32_t req, uint32_t i, uint32_t work) {
  uint32_t h = req * 2654435761u + i;
  for (uint32_t k = 0; k < work; k++) h = (h * 1664525u + 1013904223u) ^ (h >> 13);
  return h;
}

int main(int argc, char **argv) {
  @autoreleasepool {
    uint32_t words = argc > 1 ? (uint32_t)atoi(argv[1]) : 1024;
    uint32_t work = argc > 2 ? (uint32_t)atoi(argv[2]) : 8;
    // 3つ目の引数に mark を付けると、検算のあとに終わりの印も書かせ、印が見えてから検算する。
    // 付けなければ、印なしで検算だけを見続ける（検算値に依頼の番号が混ざっているので、それだけで足りる）。
    int use_mark = argc > 3 && strcmp(argv[3], "mark") == 0;
    device = MTLCreateSystemDefaultDevice();
    NSError *error = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:kSource options:nil error:&error];
    if (!lib) {
      fprintf(stderr, "compile: %s\n", [[error localizedDescription] UTF8String]);
      return 1;
    }
    id<MTLComputePipelineState> job_p = [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"job"] error:&error];
    id<MTLComputePipelineState> check_p = [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"check"] error:&error];
    id<MTLComputePipelineState> mark_p = [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mark"] error:&error];
    busy_p = [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"busy"] error:&error];
    size_t page = (size_t)getpagesize();
    size_t out_bytes = ((size_t)words * 4 + page - 1) / page * page;
    uint32_t groups = (words + 63) / 64;
    void *out_mem = NULL;
    if (posix_memalign(&out_mem, page, out_bytes) != 0) return 1;
    memset(out_mem, 0, out_bytes);
    uint32_t *out = out_mem;
    id<MTLBuffer> out_buf = [device newBufferWithBytesNoCopy:out_mem length:out_bytes options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> req_buf = [device newBufferWithLength:page options:MTLResourceStorageModeShared];
    id<MTLBuffer> sums_buf = [device newBufferWithLength:((size_t)groups * 4 + page - 1) / page * page options:MTLResourceStorageModeShared];
    id<MTLBuffer> mark_buf = [device newBufferWithLength:page options:MTLResourceStorageModeShared];
    warm_buf = [device newBufferWithLength:page options:MTLResourceStorageModeShared];
    memset([req_buf contents], 0, page);
    memset([sums_buf contents], 0, [sums_buf length]);
    memset([mark_buf contents], 0, page);
    memset([warm_buf contents], 0, page);
    ((uint32_t *)[warm_buf contents])[0] = 100000;
    uint32_t *req = (uint32_t *)[req_buf contents];
    _Atomic uint32_t *sums = (_Atomic uint32_t *)[sums_buf contents];
    _Atomic uint32_t *mark = (_Atomic uint32_t *)[mark_buf contents];
    req[1] = words;
    req[2] = work;

    id<MTLCommandQueue> park_q = [device newCommandQueueWithMaxCommandBufferCount:256];
    id<MTLSharedEvent> ev = [device newSharedEvent];
    NSMutableArray<id<MTLCommandBuffer>> *parked = [NSMutableArray array];
    uint64_t committed = 0, fired = 0;
    NSUInteger jw = [job_p threadExecutionWidth], cw = [check_p threadExecutionWidth];

    pthread_t th;
    pthread_create(&th, NULL, warm_thread, NULL);
    usleep(50000);

    const int rounds = 20000;
    double *t_verified = malloc(sizeof(double) * rounds), *t_status = malloc(sizeof(double) * rounds);
    long not_yet = 0, fell_back = 0, wrong_after_verify = 0, wrong_after_status = 0;
    for (int r = 0; r < rounds; r++) {
      @autoreleasepool {
        while (committed < fired + 32) {
          id<MTLCommandBuffer> cmd = [park_q commandBuffer];
          [cmd encodeWaitForEvent:ev value:committed + 1];
          id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
          [enc setComputePipelineState:job_p];
          [enc setBuffer:out_buf offset:0 atIndex:0];
          [enc setBuffer:req_buf offset:0 atIndex:1];
          [enc dispatchThreads:MTLSizeMake(words, 1, 1) threadsPerThreadgroup:MTLSizeMake(jw < words ? jw : words, 1, 1)];
          [enc setComputePipelineState:check_p];
          [enc setBuffer:sums_buf offset:0 atIndex:0];
          [enc setBuffer:out_buf offset:0 atIndex:1];
          [enc setBuffer:req_buf offset:0 atIndex:2];
          [enc dispatchThreads:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(cw < groups ? cw : groups, 1, 1)];
          if (use_mark) {
            [enc setComputePipelineState:mark_p];
            [enc setBuffer:mark_buf offset:0 atIndex:0];
            [enc setBuffer:req_buf offset:0 atIndex:1];
            [enc setBuffer:sums_buf offset:0 atIndex:2];
            [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
          }
          [enc endEncoding];
          [cmd commit];
          [parked addObject:cmd];
          committed++;
        }
        uint32_t ticket = (uint32_t)r + 1;
        double a = now_s();
        req[0] = ticket;
        id<MTLCommandBuffer> mine = parked[0];
        [parked removeObjectAtIndex:0];
        fired++;
        ev.signaledValue = fired;

        // 印が見えたら検算する。合わなければ見直す。0.5 ms たっても合わなければ、Metal の完了の知らせを待つ。
        int verified = 0, first_look = 1;
        while (!verified) {
          if (!use_mark || atomic_load_explicit(&mark[0], memory_order_relaxed) == ticket) {
            int ok = 1;
            for (uint32_t g = 0; g < groups && ok; g++) {
              uint32_t s = ticket * 2654435761u + g * 40503u, end = g * 64 + 64 < words ? g * 64 + 64 : words;
              for (uint32_t j = g * 64; j < end; j++) s += (out[j] ^ (j * 2654435761u)) * 2246822519u;
              if (s != atomic_load_explicit(&sums[g], memory_order_relaxed)) ok = 0;
            }
            if (ok) {
              verified = 1;
              break;
            }
            if (first_look) not_yet++;
            first_look = 0;
          }
          if (now_s() - a > 0.0005) break;
        }
        t_verified[r] = (now_s() - a) * 1e6;
        if (verified) {
          // 検算が通った時点の出力が、本当に正しいか（正しくない出力を通していないか）を全部確かめる。
          for (uint32_t i = 0; i < words; i++) {
            if (out[i] != expected(ticket, i, work)) {
              wrong_after_verify++;
              break;
            }
          }
        } else {
          fell_back++;
        }
        while ([mine status] < MTLCommandBufferStatusCompleted) {
        }
        t_status[r] = (now_s() - a) * 1e6;
        for (uint32_t i = 0; i < words; i++) {
          if (out[i] != expected(ticket, i, work)) {
            wrong_after_status++;
            break;
          }
        }
      }
    }
    atomic_store(&warm_stop, 1);
    pthread_join(th, NULL);
    qsort(t_verified, rounds, sizeof(double), cmp);
    qsort(t_status, rounds, sizeof(double), cmp);
    printf("output words: %u, work per element: %u, rounds: %d\n", words, work, rounds);
    printf("until the checksum matches:   median %6.1f  p10 %6.1f  p90 %6.1f  p99 %7.1f us\n", t_verified[rounds / 2],
           t_verified[rounds / 10], t_verified[rounds * 9 / 10], t_verified[rounds * 99 / 100]);
    printf("until Metal reports complete: median %6.1f  p10 %6.1f  p90 %6.1f  p99 %7.1f us\n", t_status[rounds / 2],
           t_status[rounds / 10], t_status[rounds * 9 / 10], t_status[rounds * 99 / 100]);
    printf("checksum looked at but not matching yet: %ld rounds, gave up and waited for Metal: %ld\n", not_yet, fell_back);
    printf("WRONG output accepted by the checksum: %ld, wrong output after Metal reported complete: %ld\n",
           wrong_after_verify, wrong_after_status);
  }
  return 0;
}
