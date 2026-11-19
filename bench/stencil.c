#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
static void step(const double *src, double *dst, uint64_t n) {
  for (uint64_t i = 1; i < n - 1; i++)
    dst[i] = (src[i - 1] + 2.0 * src[i] + src[i + 1]) * 0.25;
}
int main(void) {
  const uint64_t n = 1048576;
  double *a = calloc(n, sizeof(double));
  double *b = calloc(n, sizeof(double));
  for (uint64_t i = 0; i < n; i++) a[i] = (double)((i * 7919) % 1000);
  for (int t = 0; t < 200; t++) {
    step(a, b, n);
    step(b, a, n);
  }
  double sum = 0;
  for (uint64_t i = 0; i < n; i++) sum += a[i];
  printf("%llu\n", (unsigned long long)sum);
  free(a); free(b);
  return 0;
}
