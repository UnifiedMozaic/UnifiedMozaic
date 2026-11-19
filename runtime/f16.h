// f16 (IEEE 754 binary16) と他の浮動小数点型の変換。整数のビット操作だけで行う。
// コンパイラが出す変換の呼び出し（__extendhfsf2 など）は、生成したモジュール側の薄い関数からここへ来る。
// f16 をCPUが直接扱えるターゲット（arm64）では呼ばれない。
#include <stdint.h>

// f16 (IEEE 754 binary16) のビット列を f32 のビット列に変換する。すべての値が正確に表せる。
uint32_t um_h2f(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t man = h & 0x3ffu;
  if (exp == 0) {
    if (man == 0) return sign;
    // 非正規化数: 最上位ビットが 10 ビット目に来るまでずらす。
    uint32_t shift = (uint32_t)__builtin_clz(man) - 21u;
    man = (man << shift) & 0x3ffu;
    return sign | ((113u - shift) << 23) | (man << 13);
  }
  if (exp == 31) return sign | 0x7f800000u | (man << 13);
  return sign | ((exp + 112u) << 23) | (man << 13);
}
// f64 のビット列を、最も近い f16 に丸める（偶数丸め）。
uint16_t um_d2h(uint64_t d) {
  uint16_t sign = (uint16_t)((d >> 48) & 0x8000u);
  uint32_t exp = (uint32_t)(d >> 52) & 0x7ffu;
  uint64_t man = d & 0xfffffffffffffull;
  if (exp == 0x7ffu) {
    if (man == 0) return sign | 0x7c00u;
    return sign | 0x7e00u | (uint16_t)(man >> 42); // NaN
  }
  if (exp == 0) {
    // f64 の非正規化数は f16 の最小値の半分よりずっと小さい。
    return sign;
  }
  int32_t e = (int32_t)exp - 1023 + 15; // f16 の指数（バイアスつき）
  man |= 1ull << 52;                    // 53 ビットの仮数
  uint32_t drop;                         // 捨てる下位ビット数
  if (e >= 31) return sign | 0x7c00u;    // 丸める前から範囲外
  if (e <= 0) {
    if (e < -10) return sign;            // 最小の非正規化数の半分より小さい
    drop = 42u + (uint32_t)(1 - e);
    e = 0;
  } else {
    drop = 42u;
  }
  uint64_t kept = man >> drop;
  uint64_t rest = man & ((1ull << drop) - 1);
  uint64_t half = 1ull << (drop - 1);
  if (rest > half || (rest == half && (kept & 1))) kept++;
  // kept は、正規化数なら 1.xxx の 11 ビット、非正規化数なら 10 ビット以下。繰り上がりは指数へそのまま伝わる。
  uint32_t out;
  if (e == 0) out = (uint32_t)kept;
  else out = ((uint32_t)e << 10) + ((uint32_t)kept - 0x400u);
  if (out >= 0x7c00u) out = 0x7c00u;
  return sign | (uint16_t)out;
}
