// GPU に仕事を1回渡すのにかかる最小の時間を測る（調査用。テストではない）。
//
// ほとんど何もしない kernel を続けて何度も提出し、1回あたりの時間を出す。
//   wait: 提出のたびに waitUntilCompleted で待つ（いまのランタイムと同じ）
//   spin: 提出のたびに、寝ずに状態を見続けて待つ
//   batch K: K 回ぶんの dispatch を1回の提出にまとめる
//   event: 先に提出しておいた kernel を GPU 側で待機させ、CPU が合図（共有イベント）を出したら走らせる。
//          kernel は走り出すときに依頼票（共有メモリ）を読む。合図を出してから結果が見えるまでを測る。
//
// 使い方:
//   clang -fobjc-arc -O2 tools/metal/submit_cost.m -framework Metal -framework Foundation -o out/mac-check/submit_cost
//   out/mac-check/submit_cost

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach_time.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static NSString *const kSource =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "kernel void bump(device uint *w [[buffer(0)]], uint3 gid [[thread_position_in_grid]]) {\n"
     "  w[gid.x] = w[gid.x] + 1u;\n"
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

int main(void) {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return 1;
    NSError *error = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:kSource options:nil error:&error];
    id<MTLComputePipelineState> pipeline =
        lib ? [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"bump"] error:&error] : nil;
    if (!pipeline) {
      fprintf(stderr, "setup: %s\n", [[error localizedDescription] UTF8String]);
      return 1;
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    size_t page = (size_t)getpagesize();
    void *mem = NULL;
    if (posix_memalign(&mem, page, page) != 0) return 1;
    memset(mem, 0, page);
    uint32_t *w = mem;
    id<MTLBuffer> buf = [device newBufferWithBytesNoCopy:mem length:page options:MTLResourceStorageModeShared deallocator:nil];

    const int rounds = 3000;
    double *t = malloc(sizeof(double) * rounds);
    struct {
      const char *name;
      int spin, batch;
    } modes[] = {{"wait", 0, 1}, {"spin", 1, 1}, {"batch 8 (wait)", 0, 8}, {"batch 64 (wait)", 0, 64}};
    uint32_t expect = 0;
    printf("device: %s, %d submissions per mode, 64 elements per dispatch\n", [[device name] UTF8String], rounds);
    printf("%-18s %12s %12s %12s %18s\n", "mode", "median ms", "p99 ms", "min ms", "per dispatch (us)");
    for (size_t m = 0; m < sizeof(modes) / sizeof(modes[0]); m++) {
      for (int r = 0; r < rounds; r++) {
        @autoreleasepool {
          double a = now_ms();
          id<MTLCommandBuffer> cmd = [queue commandBuffer];
          id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
          [enc setComputePipelineState:pipeline];
          [enc setBuffer:buf offset:0 atIndex:0];
          for (int k = 0; k < modes[m].batch; k++) {
            [enc dispatchThreads:MTLSizeMake(64, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
          }
          [enc endEncoding];
          [cmd commit];
          if (modes[m].spin) {
            while ([cmd status] < MTLCommandBufferStatusCompleted) {
            }
          } else {
            [cmd waitUntilCompleted];
          }
          t[r] = now_ms() - a;
          expect += (uint32_t)modes[m].batch;
          if (w[0] != expect) {
            printf("wrong result: %u, expected %u\n", w[0], expect);
            return 1;
          }
        }
      }
      qsort(t, rounds, sizeof(double), cmp);
      printf("%-18s %12.4f %12.4f %12.4f %18.1f\n", modes[m].name, t[rounds / 2], t[rounds * 99 / 100], t[0],
             t[rounds / 2] * 1000.0 / modes[m].batch);
    }

    // ---- 待機させておく方式 ----
    {
      const int ahead = 256;  // 先に提出しておく数
      id<MTLCommandQueue> q2 = [device newCommandQueueWithMaxCommandBufferCount:1024];
      id<MTLSharedEvent> ev = [device newSharedEvent];
      void *mem2 = NULL;
      if (posix_memalign(&mem2, page, page) != 0) return 1;
      memset(mem2, 0, page);
      _Atomic uint32_t *w2 = mem2;
      id<MTLBuffer> buf2 = [device newBufferWithBytesNoCopy:mem2 length:page options:MTLResourceStorageModeShared deallocator:nil];
      id<MTLLibrary> lib2 = [device newLibraryWithSource:
          @"#include <metal_stdlib>\nusing namespace metal;\n"
           "kernel void answer(device uint *w [[buffer(0)]], uint3 gid [[thread_position_in_grid]]) {\n"
           "  if (gid.x == 0) { w[2] = w[0] * 3u + 1u; w[1] = w[0]; }\n"
           "}\n" options:nil error:&error];
      id<MTLComputePipelineState> p2 = lib2 ? [device newComputePipelineStateWithFunction:[lib2 newFunctionWithName:@"answer"] error:&error] : nil;
      if (!p2) {
        fprintf(stderr, "setup: %s\n", [[error localizedDescription] UTF8String]);
        return 1;
      }
      NSMutableArray<id<MTLCommandBuffer>> *cmds = [NSMutableArray array];
      uint64_t committed = 0;
      double *lat = malloc(sizeof(double) * rounds);
      long wrong = 0, stuck = 0;
      for (int r = 0; r < rounds; r++) {
        @autoreleasepool {
          // 待機中の kernel が ahead 個になるまで補充する（この時間は測らない。実際には手の空いたときにやる）。
          while (committed < (uint64_t)r + ahead + (r == rounds - 1 ? 256 : 0)) {
            id<MTLCommandBuffer> cmd = [q2 commandBuffer];
            [cmd encodeWaitForEvent:ev value:committed + 1];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:p2];
            [enc setBuffer:buf2 offset:0 atIndex:0];
            [enc dispatchThreads:MTLSizeMake(64, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            [cmds addObject:cmd];
            committed++;
          }
          uint32_t req = (uint32_t)r + 1;
          double a = now_ms();
          atomic_store(&w2[0], req);       // 依頼票を書く
          ev.signaledValue = (uint64_t)r + 1;  // 合図
          id<MTLCommandBuffer> mine = cmds[(NSUInteger)r];
          while (atomic_load(&w2[1]) != req) {  // 答えがメモリに見えるまで待つ
            if (now_ms() - a > 2000) {
              stuck++;
              break;
            }
          }
          lat[r] = now_ms() - a;
          if (atomic_load(&w2[2]) != req * 3u + 1u) wrong++;
          (void)mine;
        }
      }
      qsort(lat, rounds, sizeof(double), cmp);
      printf("%-18s %12.4f %12.4f %12.4f %18.1f   (wrong answers %ld, no answer %ld)\n", "event", lat[rounds / 2],
             lat[rounds * 99 / 100], lat[0], lat[rounds / 2] * 1000.0, wrong, stuck);
      free(lat);

      // 答えを待たない場合: CPU は依頼票を書いて合図を出すだけで先へ進む。
      // CPU が1回渡すのにかかる時間と、GPU が全部を片付け終わるまでの時間を測る。
      // 依頼票は1つしかないので、GPU が走り出す前に次の依頼で上書きされることがある（何回ぶん実行されたかを数える）。
      const int fire = 256;  // 待機させてある数の範囲で出す
      uint32_t before = atomic_load(&w2[1]);
      double a = now_ms();
      for (int i = 0; i < fire; i++) {
        atomic_store(&w2[0], 1000000u + (uint32_t)i);
        ev.signaledValue = (uint64_t)rounds + (uint64_t)i + 1;
      }
      double handed = now_ms() - a;
      id<MTLCommandBuffer> last = cmds[(NSUInteger)(rounds + fire - 1)];
      [last waitUntilCompleted];
      double done = now_ms() - a;
      printf("event, not waiting: %d hand-offs took the CPU %.3f ms (%.1f us each); the GPU finished all of them after %.3f ms"
             " (%.1f us each); last request seen by the GPU: %u (before: %u)\n",
             fire, handed, handed * 1000.0 / fire, done, done * 1000.0 / fire, atomic_load(&w2[1]), before);
    }
    free(t);
  }
  return 0;
}
