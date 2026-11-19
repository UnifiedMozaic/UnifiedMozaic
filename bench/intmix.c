#include <stdint.h>
#include <stdio.h>
int main(void) {
  uint64_t x = 88172645463325252ULL;
  uint32_t acc = 2166136261u;
  uint64_t sum = 0;
  for (uint32_t i = 0; i < 400000000u; i++) {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    acc = (acc ^ (uint32_t)(x >> 32)) * 16777619u;
    acc = (acc << 5) | (acc >> 27);
    sum += (uint64_t)(acc & 255u);
  }
  printf("%llu\n%u\n%llu\n", (unsigned long long)x, (unsigned)acc, (unsigned long long)sum);
  return 0;
}
