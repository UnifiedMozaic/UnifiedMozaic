// Metal 4（macOS 26 以降の、命令の出し方が新しい API）で、小さい仕事1つの往復が縮むかを測る（調査用。テストではない）。
//
// 方式を1周ごとに交互に実行して比べる。GPU は別スレッドの軽い kernel で起こしたままにする。
//   old commit+status:   従来の API で提出し、完了の状態を寝ずに見続ける
//   old parked+status:   従来の API で先に提出して合図を待たせ、合図を出して完了の状態を見続ける（いまのランタイムの常駐）
//   mtl4 commit+feedback: Metal 4 で提出し、完了の知らせ（feedback handler）を寝ずに待つ
//   mtl4 commit+memory:   Metal 4 で提出し、答えがメモリに見えるのを待つ（参考。これを完了の判断に使うのは安全ではない）
//
// 使い方:
//   clang -fobjc-arc -O2 tools/metal/roundtrip4.m -framework Metal -framework Foundation -o out/mac-check/roundtrip4
//   out/mac-check/roundtrip4

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
static id<MTLComputePipelineState> answer_p, busy_p;
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
    size_t page = (size_t)getpagesize();
    id<MTLBuffer> buf = [device newBufferWithLength:page options:MTLResourceStorageModeShared];
    memset([buf contents], 0, page);
    _Atomic uint32_t *w = (_Atomic uint32_t *)[buf contents];
    warm_buf = [device newBufferWithLength:page options:MTLResourceStorageModeShared];
    memset([warm_buf contents], 0, page);
    ((uint32_t *)[warm_buf contents])[0] = getenv("WARM") ? (uint32_t)atoi(getenv("WARM")) : 100000;

    // 従来の API。
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandQueue> park_q = [device newCommandQueueWithMaxCommandBufferCount:256];
    id<MTLSharedEvent> park_ev = [device newSharedEvent];
    NSMutableArray<id<MTLCommandBuffer>> *parked = [NSMutableArray array];
    uint64_t park_committed = 0, park_fired = 0;

    // Metal 4。
    id<MTL4CommandQueue> q4 = nil;
    id<MTL4CommandAllocator> alloc4 = nil;
    id<MTL4CommandBuffer> cmd4 = nil;
    id<MTL4ArgumentTable> table4 = nil;
    int have4 = 0;
    if (@available(macOS 26.0, *)) {
      q4 = [device newMTL4CommandQueue];
      alloc4 = [device newCommandAllocator];
      cmd4 = [device newCommandBuffer];
      MTL4ArgumentTableDescriptor *td = [MTL4ArgumentTableDescriptor new];
      td.maxBufferBindCount = 4;
      table4 = [device newArgumentTableWithDescriptor:td error:&error];
      MTLResidencySetDescriptor *rd = [MTLResidencySetDescriptor new];
      id<MTLResidencySet> rs = [device newResidencySetWithDescriptor:rd error:&error];
      if (q4 && alloc4 && cmd4 && table4 && rs) {
        [table4 setAddress:buf.gpuAddress atIndex:0];
        [rs addAllocation:buf];
        [rs commit];
        [q4 addResidencySet:rs];
        have4 = 1;
      } else {
        fprintf(stderr, "Metal 4 setup failed: %s\n", error ? [[error localizedDescription] UTF8String] : "unknown");
      }
    }
    printf("device: %s, Metal 4: %s\n", [[device name] UTF8String], have4 ? "yes" : "no");

    pthread_t th;
    pthread_create(&th, NULL, warm_thread, NULL);
    usleep(50000);

    enum { M = 4 };
    const char *names[M] = {"old commit+status", "old parked+status", "mtl4 commit+feedback", "mtl4 commit+memory"};
    const int rounds = 3000;
    double *t[M];
    for (int m = 0; m < M; m++) t[m] = calloc((size_t)rounds, sizeof(double));
    static _Atomic uint32_t fed;  // Metal 4 の完了の知らせが届いた依頼の番号
    long wrong = 0;
    const char *only = getenv("ONLY");
    for (int r = 0; r < rounds; r++) {
      @autoreleasepool {
        while (park_committed < park_fired + 32) {
          id<MTLCommandBuffer> cmd = [park_q commandBuffer];
          [cmd encodeWaitForEvent:park_ev value:park_committed + 1];
          id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
          [enc setComputePipelineState:answer_p];
          [enc setBuffer:buf offset:0 atIndex:0];
          [enc dispatchThreads:MTLSizeMake(64, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
          [enc endEncoding];
          [cmd commit];
          [parked addObject:cmd];
          park_committed++;
        }
        for (int m = 0; m < M; m++) {
          if ((only && atoi(only) != m) || (m >= 2 && !have4)) continue;
          uint32_t req = atomic_load(&w[1]) + 1;
          double a = now_s();
          atomic_store(&w[0], req);
          if (m == 0) {
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:answer_p];
            [enc setBuffer:buf offset:0 atIndex:0];
            [enc dispatchThreads:MTLSizeMake(64, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            while ([cmd status] < MTLCommandBufferStatusCompleted) {
            }
          } else if (m == 1) {
            id<MTLCommandBuffer> mine = parked[(NSUInteger)park_fired];
            park_fired++;
            park_ev.signaledValue = park_fired;
            while ([mine status] < MTLCommandBufferStatusCompleted) {
            }
          } else if (@available(macOS 26.0, *)) {
            [alloc4 reset];
            [cmd4 beginCommandBufferWithAllocator:alloc4];
            id<MTL4ComputeCommandEncoder> enc = [cmd4 computeCommandEncoder];
            [enc setComputePipelineState:answer_p];
            [enc setArgumentTable:table4];
            [enc dispatchThreads:MTLSizeMake(64, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            [enc endEncoding];
            [cmd4 endCommandBuffer];
            MTL4CommitOptions *options = [MTL4CommitOptions new];
            [options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
              (void)feedback;
              atomic_store(&fed, req);
            }];
            id<MTL4CommandBuffer> list[1] = {cmd4};
            [q4 commit:list count:1 options:options];
            if (m == 2) {
              while (atomic_load(&fed) != req) {
              }
            } else {
              while (atomic_load(&w[1]) != req) {
              }
            }
          }
          t[m][r] = (now_s() - a) * 1e6;
          // Metal 4 でメモリを見て待った場合は、次へ進む前に完了の知らせも待つ（命令の置き場を使い回すため）。
          if (m == 3) {
            while (atomic_load(&fed) != req) {
            }
          }
          if (atomic_load(&w[1]) != req || atomic_load(&w[2]) != req * 3u + 1u) wrong++;
        }
      }
    }
    atomic_store(&warm_stop, 1);
    pthread_join(th, NULL);
    printf("%-22s %9s %9s %9s %9s   (us)\n", "method", "median", "p10", "p90", "p99");
    for (int m = 0; m < M; m++) {
      qsort(t[m], (size_t)rounds, sizeof(double), cmp);
      printf("%-22s %9.1f %9.1f %9.1f %9.1f\n", names[m], t[m][rounds / 2], t[m][rounds / 10], t[m][rounds * 9 / 10],
             t[m][rounds * 99 / 100]);
    }
    printf("wrong answers: %ld\n", wrong);
  }
  return 0;
}
