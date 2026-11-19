#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
static void blur(float *out, const float *src, uint32_t n) {
  for (uint32_t i = 0; i < n; i++) {
    uint32_t left = i > 0 ? i - 1 : i;
    uint32_t right = i + 1 < n ? i + 1 : n - 1;
    out[i] = (src[left] + src[i] + src[right]) * 0.33333334f;
  }
}
static void saxpy(float *y, const float *x, float a, uint32_t n) {
  for (uint32_t i = 0; i < n; i++) y[i] = a * x[i] + y[i];
}
int main(void) {
  const uint32_t n = 4194304;
  float *a = calloc(n, sizeof(float));
  float *b = calloc(n, sizeof(float));
  for (uint32_t i = 0; i < n; i++) a[i] = (float)((i * 7919u) % 1000u);
  for (int t = 0; t < 60; t++) {
    blur(b, a, n);
    blur(a, b, n);
    saxpy(a, b, 0.001f, n);
  }
  double sum = 0;
  for (uint32_t i = 0; i < n; i++) sum += (double)a[i];
  printf("%llu\n", (unsigned long long)sum);
  free(a); free(b);
  return 0;
}
