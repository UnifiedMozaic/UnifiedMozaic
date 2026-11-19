#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
int main(void) {
  uint32_t *xs = NULL; uint64_t len = 0, cap = 0;
  uint64_t total = 0;
  for (uint32_t round = 0; round < 200; round++) {
    len = 0;
    for (uint32_t i = 0; i < 5000000u; i++) {
      if (len == cap) { cap = cap ? cap * 2 : 4; xs = realloc(xs, cap * sizeof(uint32_t)); if (!xs) abort(); }
      xs[len++] = i * 2654435761u + round;
    }
    for (uint64_t i = 0; i < len; i++) total += (uint64_t)(xs[i] >> 3);
  }
  printf("%llu\n%llu\n", (unsigned long long)total, (unsigned long long)len);
  free(xs);
  return 0;
}
