// Metal Shading Language（MSL）の出力を、Metal なしで C++ としてコンパイル・実行するための最小のスタブ。
// 生成した MSL が使う範囲だけを、Metal の意味で定義する。clang++ -std=c++14 で使う。
// 先に標準ヘッダを include してから、このヘッダを include すること（kernel などのマクロが標準ヘッダに影響しないように）。
// sqrt / floor / ceil / fabs / fmod / INFINITY / NAN は <cmath> のものをそのまま使う。
// Metal と同じく float を渡せば float を返すよう、std の多重定義を名前空間なしで見えるようにする
// （<cmath> だけでは、名前空間なしの floor(float) が C の floor(double) になり、結果が double になることがある）。
#pragma once

#include <cmath>
#include <cstdint>

using std::ceil;
using std::fabs;
using std::floor;
using std::fmod;
using std::sqrt;

// アドレス空間と kernel は、C++ では何もしない。
#define kernel
#define device
#define constant
#define thread
#define threadgroup

typedef unsigned int uint;
// MSL の long / ulong は 64 ビット（この stub を使う LP64 の C++ の long も 64 ビット）。
#ifdef _WIN32
// Windows の C++ の long は 32 ビット（LLP64）。tests/run.ts が MSL の long を um_long に置き換えてから読み込む。
typedef long long um_long;
typedef unsigned long long ulong;
#else
typedef unsigned long ulong;
#endif

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

// 整数の popcount / clz / ctz は、引数と同じ型を返す。0 の clz / ctz はビット幅（32 / 64）。
template <class T>
inline T popcount(T x) {
  if (sizeof(T) == 8) return T(__builtin_popcountll((unsigned long long)x));
  return T(__builtin_popcount(uint(x)));
}

template <class T>
inline T clz(T x) {
  if (sizeof(T) == 8) return (unsigned long long)x == 0ull ? T(64) : T(__builtin_clzll((unsigned long long)x));
  return uint(x) == 0u ? T(32) : T(__builtin_clz(uint(x)));
}

template <class T>
inline T ctz(T x) {
  if (sizeof(T) == 8) return (unsigned long long)x == 0ull ? T(64) : T(__builtin_ctzll((unsigned long long)x));
  return uint(x) == 0u ? T(32) : T(__builtin_ctz(uint(x)));
}

// グループの barrier。C++ では1スレッドずつ実行するので何もしない（構文の確認用）。
enum class mem_flags { mem_none = 0, mem_device = 1, mem_threadgroup = 2 };
inline void threadgroup_barrier(mem_flags) {}

// アトミック（relaxed だけ）。C++ では1スレッドずつ実行するので普通の読み書きにする。
enum memory_order { memory_order_relaxed };
struct atomic_int {
  int v;
};
struct atomic_uint {
  uint v;
};
inline int atomic_load_explicit(atomic_int *p, memory_order) { return p->v; }
inline uint atomic_load_explicit(atomic_uint *p, memory_order) { return p->v; }
inline void atomic_store_explicit(atomic_int *p, int x, memory_order) { p->v = x; }
inline void atomic_store_explicit(atomic_uint *p, uint x, memory_order) { p->v = x; }
inline int atomic_fetch_add_explicit(atomic_int *p, int x, memory_order) {
  int old = p->v;
  p->v = int(uint(old) + uint(x));
  return old;
}
inline uint atomic_fetch_add_explicit(atomic_uint *p, uint x, memory_order) {
  uint old = p->v;
  p->v = old + x;
  return old;
}
template <class A, class T>
inline bool atomic_compare_exchange_weak_explicit(A *p, T *expected, T desired, memory_order, memory_order) {
  if (p->v == *expected) {
    p->v = desired;
    return true;
  }
  *expected = p->v;
  return false;
}

}  // namespace metal
