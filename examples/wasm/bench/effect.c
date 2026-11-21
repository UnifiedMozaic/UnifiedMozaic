// 速さの比較に使う画像の効果（C）。同じソースを wasm32 とネイティブにコンパイルして比べる（docs/wasm.ja.md の「速さ」）。
//   invert:   RGBA の RGB を反転する（メモリの読み書きが主）。
//   saturate: 彩度を変える（画素ごとに小数の計算と切り詰め）。
// p は画素の先頭、n は画素の数。
#include <stdint.h>

#ifdef __wasm__
#define EXPORT(name) __attribute__((export_name(name)))
#else
#define EXPORT(name)
#endif

EXPORT("invert") void invert(uint8_t *p, int n) {
  uint32_t *q = (uint32_t *)p;
  for (int i = 0; i < n; i++) q[i] ^= 0x00ffffffu;
}

static inline uint8_t clamp8(float v) {
  if (v < 0.0f) v = 0.0f;
  if (v > 255.0f) v = 255.0f;
  return (uint8_t)(v + 0.5f);
}

EXPORT("saturate") void saturate(uint8_t *p, int n, float s) {
  for (int i = 0; i < n; i++) {
    float r = p[4 * i], g = p[4 * i + 1], b = p[4 * i + 2];
    float l = 0.299f * r + 0.587f * g + 0.114f * b;
    p[4 * i] = clamp8(l + (r - l) * s);
    p[4 * i + 1] = clamp8(l + (g - l) * s);
    p[4 * i + 2] = clamp8(l + (b - l) * s);
  }
}
