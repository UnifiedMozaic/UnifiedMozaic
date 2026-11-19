#include <stdint.h>
#include <stdio.h>
static int32_t fib(int32_t n) {
  if (n < 2) return n;
  return fib(n - 1) + fib(n - 2);
}
int main(void) {
  int64_t total = 0;
  for (int32_t k = 0; k < 4; k++) total += (int64_t)fib(40 + k % 2);
  printf("%lld\n", (long long)total);
  return 0;
}
