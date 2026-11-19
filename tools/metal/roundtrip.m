// GPU に小さい仕事を1つ渡して答えを受け取るまでの時間の内訳を測る（調査用。テストではない）。
//
// 内訳: 提出 → GPU が動き出す → GPU の計算 → GPU が終わる → CPU が答えを見る。
// いくつかの条件で比べる:
//   plain: 何もしない
//   warm:  別の待ち行列で、何もしない kernel を回し続けて GPU を起こしたままにする
//   qos:   呼び出すスレッドの優先度を上げる
// そのあと、渡し方を交互に実行して1回の往復の時間を比べる（同じ条件で比べるため、1周ごとに方式を変える）:
//   commit+wait:   提出して waitUntilCompleted で待つ（いまのランタイム）
//   commit+status: 提出して、寝ずに完了の状態を見続ける
//   commit+memory: 提出して、寝ずに答えがメモリに見えるのを待つ
//   parked+memory: 先に提出して合図を待たせておいた kernel に合図を出し、答えがメモリに見えるのを待つ。
//                  要素数は実行時に共有メモリから読む（間接 dispatch）。
//
// 使い方:
//   clang -fobjc-arc -O2 tools/metal/roundtrip.m -framework Metal -framework Foundation -o out/mac-check/roundtrip
//   out/mac-check/roundtrip

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
     "kernel void answer(device uint *w [[buffer(0)]], uint3 gid [[thread_position_in_grid]]) {\n"
     "  if (gid.x == 0) { w[2] = w[0] * 3u + 1u; w[1] = w[0]; }\n"
     "}\n"
     // GPU を起こしたままにするための kernel。count 回だけ計算して終わる。
     "kernel void busy(device atomic_uint *s [[buffer(0)]], uint3 gid [[thread_position_in_grid]]) {\n"
     "  uint acc = gid.x;\n"
     "  uint n = atomic_load_explicit(&s[0], memory_order_relaxed);\n"
     "  for (uint j = 0; j < n; j++) { acc ^= acc << 13; acc ^= acc >> 17; acc ^= acc << 5; acc += j; }\n"
     "  atomic_store_explicit(&s[1], acc, memory_order_relaxed);\n"
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

static double median(double *v, int n) {
  qsort(v, (size_t)n, sizeof(double), cmp);
  return v[n / 2];
}

static id<MTLDevice> device;
static id<MTLComputePipelineState> answer_p, busy_p;
static _Atomic int warm_stop;
static id<MTLBuffer> warm_buf;

// GPU を起こしたままにする: 短い kernel を間を空けずに出し続ける。
// 環境変数 PIPE=n: 起こしておく kernel を、終わるのを待たずに n 個まで重ねて出し続ける
// （GPU への指示が途切れないようにする）。付けなければ、1つ出しては終わるのを待つ。
static void *warm_thread(void *arg) {
  (void)arg;
  id<MTLCommandQueue> q = [device newCommandQueue];
  int pipe = getenv("PIPE") ? atoi(getenv("PIPE")) : 0;
  if (pipe > 0) {
    NSMutableArray<id<MTLCommandBuffer>> *flight = [NSMutableArray array];
    while (!atomic_load(&warm_stop)) {
      @autoreleasepool {
        id<MTLCommandBuffer> cmd = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:busy_p];
        [enc setBuffer:warm_buf offset:0 atIndex:0];
        [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [flight addObject:cmd];
        if ((int)[flight count] >= pipe) {
          [flight[0] waitUntilCompleted];
          [flight removeObjectAtIndex:0];
        }
      }
    }
    return NULL;
  }
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

static void measure(const char *name, id<MTLCommandQueue> queue, id<MTLBuffer> buf, int rounds) {
  _Atomic uint32_t *w = (_Atomic uint32_t *)[buf contents];
  double *total = malloc(sizeof(double) * (size_t)rounds), *to_start = malloc(sizeof(double) * (size_t)rounds);
  double *run = malloc(sizeof(double) * (size_t)rounds), *to_seen = malloc(sizeof(double) * (size_t)rounds);
  double *encode = malloc(sizeof(double) * (size_t)rounds);
  for (int r = 0; r < rounds; r++) {
    @autoreleasepool {
      uint32_t req = atomic_load(&w[1]) + 1;
      double a = now_s();
      atomic_store(&w[0], req);
      id<MTLCommandBuffer> cmd = [queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
      [enc setComputePipelineState:answer_p];
      [enc setBuffer:buf offset:0 atIndex:0];
      [enc dispatchThreads:MTLSizeMake(64, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      [enc endEncoding];
      double b = now_s();
      [cmd commit];
      while (atomic_load(&w[1]) != req) {
      }
      double seen = now_s();
      [cmd waitUntilCompleted];
      encode[r] = (b - a) * 1e6;
      total[r] = (seen - a) * 1e6;
      to_start[r] = ([cmd GPUStartTime] - b) * 1e6;
      run[r] = ([cmd GPUEndTime] - [cmd GPUStartTime]) * 1e6;
      to_seen[r] = (seen - [cmd GPUEndTime]) * 1e6;
    }
  }
  printf("%-8s total %7.1f | encode %6.1f | commit->GPU start %7.1f | GPU run %6.1f | GPU end->seen %7.1f   (us, medians)\n", name,
         median(total, rounds), median(encode, rounds), median(to_start, rounds), median(run, rounds), median(to_seen, rounds));
  free(total);
  free(to_start);
  free(run);
  free(to_seen);
  free(encode);
}

// 先に提出して待機させておく kernel。要素数は走り出すときに indirect から読む。
static id<MTLCommandQueue> park_q;
static id<MTLSharedEvent> park_ev;
static NSMutableArray<id<MTLCommandBuffer>> *parked;
static uint64_t park_committed, park_fired;
static id<MTLBuffer> indirect;

static void park_refill(id<MTLBuffer> buf, uint64_t ahead) {
  while (park_committed < park_fired + ahead) {
    id<MTLCommandBuffer> cmd = [park_q commandBuffer];
    [cmd encodeWaitForEvent:park_ev value:park_committed + 1];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:answer_p];
    [enc setBuffer:buf offset:0 atIndex:0];
    [enc dispatchThreadgroupsWithIndirectBuffer:indirect indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [parked addObject:cmd];
    park_committed++;
  }
}

static double one(int method, id<MTLCommandQueue> queue, id<MTLBuffer> buf) {
  _Atomic uint32_t *w = (_Atomic uint32_t *)[buf contents];
  uint32_t req = atomic_load(&w[1]) + 1;
  double a = now_s();
  atomic_store(&w[0], req);
  if (method >= 3) {
    ((uint32_t *)[indirect contents])[0] = 1;  // 64 スレッドのグループを1つ
    // method 4 は、続けてもう1つ待機 kernel を走らせる（答えは同じ値を書き直すだけ）。
    // GPU の仕事の区切りを早く作って、答えが見えるまでを縮められるかを試す。
    park_fired += method == 4 ? 2 : 1;
    id<MTLCommandBuffer> mine = parked[(NSUInteger)(park_fired - 1)];
    park_ev.signaledValue = park_fired;
    if (method == 5) {
      // 完了の状態を見続けて待つ（メモリに見えるかどうかではなく、Metal が完了を知らせるまで）。
      while ([mine status] < MTLCommandBufferStatusCompleted) {
      }
    } else {
      while (atomic_load(&w[1]) != req) {
      }
    }
  } else {
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:answer_p];
    [enc setBuffer:buf offset:0 atIndex:0];
    [enc dispatchThreads:MTLSizeMake(64, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    if (method == 0) [cmd waitUntilCompleted];
    else if (method == 1) {
      while ([cmd status] < MTLCommandBufferStatusCompleted) {
      }
    } else {
      while (atomic_load(&w[1]) != req) {
      }
    }
  }
  double t = (now_s() - a) * 1e6;
  if (atomic_load(&w[1]) != req || atomic_load(&w[2]) != req * 3u + 1u) {
    printf("wrong answer (method %d)\n", method);
    exit(1);
  }
  return t;
}

static void compare(const char *title, id<MTLCommandQueue> queue, id<MTLBuffer> buf, int rounds) {
  const char *names[] = {"commit+wait", "commit+status", "commit+memory", "parked+memory", "parked x2", "parked+status"};
  double *t[6];
  for (int m = 0; m < 6; m++) t[m] = malloc(sizeof(double) * (size_t)rounds);
  for (int r = 0; r < rounds; r++) {
    @autoreleasepool {
      park_refill(buf, 32);
      // 環境変数 ONLY に方式の番号を入れると、その方式だけを実行する（ほかの方式が GPU を温めないように）。
      const char *only = getenv("ONLY");
      // 環境変数 GAP（マイクロ秒）: 1回ごとに、次の合図まで CPU 側で間を空ける（呼び出しの合間の計算のつもり）。
      if (getenv("GAP")) {
        double until = now_s() + atof(getenv("GAP")) * 1e-6;
        while (now_s() < until) {
        }
      }
      for (int m = 0; m < 6; m++) {
        if (only && atoi(only) != m) {
          t[m][r] = 0;
          continue;
        }
        t[m][r] = one(m, queue, buf);
      }
    }
  }
  printf("-- %s (us) --\n", title);
  for (int m = 0; m < 6; m++) {
    qsort(t[m], (size_t)rounds, sizeof(double), cmp);
    printf("%-14s median %7.1f  p10 %7.1f  p90 %7.1f  p99 %8.1f\n", names[m], t[m][rounds / 2], t[m][rounds / 10],
           t[m][rounds * 9 / 10], t[m][rounds * 99 / 100]);
    free(t[m]);
  }
}

int main(void) {
  @autoreleasepool {
    device = MTLCreateSystemDefaultDevice();
    NSError *error = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:kSource options:nil error:&error];
    answer_p = lib ? [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"answer"] error:&error] : nil;
    busy_p = lib ? [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"busy"] error:&error] : nil;
    if (!answer_p || !busy_p) {
      fprintf(stderr, "setup: %s\n", [[error localizedDescription] UTF8String]);
      return 1;
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLBuffer> buf = [device newBufferWithLength:(NSUInteger)getpagesize() options:MTLResourceStorageModeShared];
    memset([buf contents], 0, (size_t)getpagesize());
    warm_buf = [device newBufferWithLength:(NSUInteger)getpagesize() options:MTLResourceStorageModeShared];
    memset([warm_buf contents], 0, (size_t)getpagesize());
    park_q = [device newCommandQueueWithMaxCommandBufferCount:256];
    park_ev = [device newSharedEvent];
    parked = [NSMutableArray array];
    indirect = [device newBufferWithLength:(NSUInteger)getpagesize() options:MTLResourceStorageModeShared];
    uint32_t *ind = (uint32_t *)[indirect contents];
    ind[0] = 1;
    ind[1] = 1;
    ind[2] = 1;

    const int rounds = 3000;
    measure("warmup", queue, buf, 300);
    measure("plain", queue, buf, rounds);

    compare("GPU otherwise idle", queue, buf, rounds);

    // GPU を起こしたままにして比べる。起こしておく kernel の1回の長さ（計算の回数）とスレッド数を変える。
    const uint32_t lengths[] = {getenv("WARM") ? (uint32_t)atoi(getenv("WARM")) : 100000};
    for (int i = 0; i < 1; i++) {
      ((uint32_t *)[warm_buf contents])[0] = lengths[i];
      atomic_store(&warm_stop, 0);
      pthread_t th;
      pthread_create(&th, NULL, warm_thread, NULL);
      usleep(50000);
      char title[96];
      snprintf(title, sizeof title, "GPU kept awake by a background kernel of %u iterations", lengths[i]);
      compare(title, queue, buf, 1500);
      atomic_store(&warm_stop, 1);
      pthread_join(th, NULL);
    }
  }
  return 0;
}
