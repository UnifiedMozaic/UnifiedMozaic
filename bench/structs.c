#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
typedef struct { double x, y, vx, vy; uint32_t hits; } Particle;
static void update(Particle *ps, uint64_t n, double dt) {
  for (uint64_t i = 0; i < n; i++) {
    ps[i].x += ps[i].vx * dt;
    ps[i].y += ps[i].vy * dt;
    if (ps[i].x < 0.0 || ps[i].x > 1000.0) { ps[i].vx = -ps[i].vx; ps[i].hits += 1; }
    if (ps[i].y < 0.0 || ps[i].y > 1000.0) { ps[i].vy = -ps[i].vy; ps[i].hits += 1; }
  }
}
int main(void) {
  const uint64_t n = 100000;
  Particle *ps = malloc(n * sizeof(Particle));
  for (uint64_t i = 0; i < n; i++) {
    double k = (double)(i % 1000);
    Particle p = { k, (double)(999 - i % 1000), (double)(i % 7) - 3.0, (double)(i % 11) - 5.0, 0 };
    ps[i] = p;
  }
  for (int t = 0; t < 2000; t++) update(ps, n, 0.5);
  uint64_t hits = 0; double sx = 0;
  for (uint64_t i = 0; i < n; i++) { hits += ps[i].hits; sx += ps[i].x + ps[i].y; }
  printf("%llu\n%llu\n", (unsigned long long)hits, (unsigned long long)sx);
  free(ps);
  return 0;
}
