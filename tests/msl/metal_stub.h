// Metal Shading Language（MSL）の出力を、Metal なしで C++ としてコンパイル・実行するための最小のスタブ。
// 生成した MSL が使う範囲だけを、Metal の意味で定義する。clang++ -std=c++14 で使う。
// 先に標準ヘッダを include してから、このヘッダを include すること（kernel などのマクロが標準ヘッダに影響しないように）。
// sqrt / floor / ceil / fabs / fmod / INFINITY / NAN は <cmath> のものをそのまま使う。
#pragma once

#include <cmath>
#include <cstdint>

// アドレス空間と kernel は、C++ では何もしない。
#define kernel
#define device
#define constant
#define thread

typedef unsigned int uint;

namespace metal {

// as_type<T>(x): 同じ大きさの型へのビットの読み替え。
template <class T, class F>
inline T as_type(F x) {
  static_assert(sizeof(T) == sizeof(F), "as_type needs types of the same size");
  T r;
  __builtin_memcpy(&r, &x, sizeof(T));
  return r;
}

struct uint3 {
  uint x, y, z;
};

template <class T>
inline T min(T a, T b) {
  return b < a ? b : a;
}

template <class T>
inline T max(T a, T b) {
  return a < b ? b : a;
}

template <class T>
inline T clamp(T x, T lo, T hi) {
  return min(max(x, lo), hi);
}

// Metal の select(a, b, c) は c が真なら b
template <class T>
inline T select(T a, T b, bool c) {
  return c ? b : a;
}

// 整数の popcount / clz / ctz は、引数と同じ型を返す。0 の clz / ctz は 32。
template <class T>
inline T popcount(T x) {
  return T(__builtin_popcount(uint(x)));
}

template <class T>
inline T clz(T x) {
  return uint(x) == 0u ? T(32) : T(__builtin_clz(uint(x)));
}

template <class T>
inline T ctz(T x) {
  return uint(x) == 0u ? T(32) : T(__builtin_ctz(uint(x)));
}

}  // namespace metal
