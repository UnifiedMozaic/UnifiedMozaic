// effect.c をネイティブで実行したときの時間（ミリ秒 / コマ）を測る。bench.um と同じ画像・同じ回数。
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
void invert(uint8_t *p, int n);
void saturate(uint8_t *p, int n, float s);

static double now(void) {
  struct timespec t;
  timespec_get(&t, TIME_UTC);
  return t.tv_sec + t.tv_nsec / 1e9;
}

int main(void) {
  const int w = 1280, h = 720, frames = 200;
  uint8_t *px = malloc((size_t)w * h * 4);
  for (int i = 0; i < w * h * 4; i++) px[i] = (uint8_t)(i * 7);
  double t0 = now();
  for (int f = 0; f < frames; f++) invert(px, w * h);
  double t1 = now();
  for (int f = 0; f < frames; f++) saturate(px, w * h, 1.2f);
  double t2 = now();
  unsigned sum = 0;
  for (int i = 0; i < w * h * 4; i++) sum += px[i];
  printf("native invert:   %.3f ms/frame\n", (t1 - t0) * 1000 / frames);
  printf("native saturate: %.3f ms/frame\n", (t2 - t1) * 1000 / frames);
  printf("checksum %u\n", sum);
  return 0;
}
