// 生成した MSL（スタブでC++として）を、全添字について実行し、tests/msl/kernels.um と同じチェックサムを出す。
// 入力の作り方は kernels.um の main と同じ。コンパイル時に -I で生成物（<kernel>.metal）の場所を渡す。
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "metal_stub.h"

namespace k_arith {
#include "arith.metal"
}
namespace k_ints {
#include "ints.metal"
}
namespace k_uints {
#include "uints.metal"
}
namespace k_casts {
#include "casts.metal"
}
namespace k_clamps {
#include "clamps.metal"
}
namespace k_cells {
#include "cells.metal"
}
namespace k_collatz {
#include "collatz.metal"
}
namespace k_sums {
#include "sums.metal"
}
namespace k_loops {
#include "loops.metal"
}
namespace k_misc {
#include "misc.metal"
}

typedef uint64_t u64;

// 添字 0..count を、2次元の格子（幅 row）として1つずつ実行する。
static const uint ROW = 13;

static metal::uint3 grid(uint idx) {
  return metal::uint3{idx % ROW, idx / ROW, 0};
}

// ---- 乱数（kernels.um の Rng と同じ）----
struct Rng {
  uint32_t state;
  explicit Rng(uint32_t seed) : state(seed) {}
  uint32_t next() {
    state = state * 1664525u + 1013904223u;
    return state;
  }
};

// ---- チェックサム（kernels.um の hash* と同じ）----
static const u64 M = 1000003;

static u64 hi32(const std::vector<int>& a, size_t n) {
  u64 h = 0;
  for (size_t i = 0; i < n; i++) h = h * M + (u64)(int64_t)a[i];
  return h;
}

static u64 hu32(const std::vector<uint>& a, size_t n) {
  u64 h = 0;
  for (size_t i = 0; i < n; i++) h = h * M + (u64)a[i];
  return h;
}

// 浮動小数点から i64 へ（範囲外は飽和、NaN は 0）。
static int64_t f2i64(float v) {
  if (v != v) return 0;
  if (v >= 9223372036854775808.0f) return INT64_MAX;
  if (v <= -9223372036854775808.0f) return INT64_MIN;
  return (int64_t)v;
}

static u64 scaled(float v) {
  return (u64)f2i64(v * 64.0f);
}

static u64 hf32(const std::vector<float>& a, size_t n) {
  u64 h = 0;
  for (size_t i = 0; i < n; i++) h = h * M + scaled(a[i]);
  return h;
}

int main() {
  const uint n = 1000;
  Rng rng(12345);
  std::vector<float> xs(n), ys(n), fs(n);
  std::vector<int> ia(n), ib(n);
  std::vector<uint> ua(n + 64);
  for (uint i = 0; i < n; i++) {
    xs[i] = float(rng.next() % 2000) * 0.01f - 5.0f;
    ys[i] = float(rng.next() % 3000) * 0.02f - 30.0f;
    ia[i] = int(rng.next()) / 1000;
    ib[i] = int(rng.next() % 41) - 20;
    ua[i] = rng.next();
    fs[i] = float(rng.next() % 100000) * 97.5f - 4000000.0f;
  }
  for (uint i = n; i < n + 64; i++) ua[i] = rng.next();
  ia[0] = INT32_MIN;
  ib[0] = -1;
  ia[1] = 7;
  ib[1] = 0;
  fs[0] = 3000000000.0f;
  fs[1] = -3000000000.0f;
  fs[2] = 5000000000.0f;
  fs[3] = 0.0f;

  {
    using namespace k_arith;
    std::vector<float> out(n);
    UmParams p{};
    p.count = n; p.row = ROW; p.len_out = n; p.len_x = n; p.len_y = n; p.p_k = 1.75f;
    for (uint i = 0; i < n; i++) um_main(p, out.data(), xs.data(), ys.data(), grid(i));
    printf("%llu\n", (unsigned long long)hf32(out, n));
  }
  {
    using namespace k_ints;
    std::vector<int> out(n);
    UmParams p{};
    p.count = n; p.row = ROW; p.len_out = n; p.len_a = n; p.len_b = n;
    for (uint i = 0; i < n; i++) um_main(p, out.data(), ia.data(), ib.data(), grid(i));
    printf("%llu\n", (unsigned long long)hi32(out, n));
  }
  {
    using namespace k_uints;
    std::vector<uint> out(n);
    UmParams p{};
    p.count = n; p.row = ROW; p.len_out = n; p.len_a = n + 64;
    for (uint i = 0; i < n; i++) um_main(p, out.data(), ua.data(), grid(i));
    printf("%llu\n", (unsigned long long)hu32(out, n));
  }
  {
    using namespace k_casts;
    std::vector<int> out(n);
    UmParams p{};
    p.count = n; p.row = ROW; p.len_out = n; p.len_f = n;
    for (uint i = 0; i < n; i++) um_main(p, out.data(), fs.data(), grid(i));
    printf("%llu\n", (unsigned long long)hi32(out, n));
  }
  {
    using namespace k_clamps;
    std::vector<int> out(64);
    std::vector<int> small = {10, 20, 30, 40, 50};
    std::vector<int> empty(1);  // 長さ 0 の Buffer も1要素ぶん確保する
    UmParams p{};
    p.count = 64; p.row = ROW; p.len_out = 64; p.len_src = 5; p.len_empty = 0;
    for (uint i = 0; i < 64; i++) um_main(p, out.data(), small.data(), empty.data(), grid(i));
    printf("%llu\n", (unsigned long long)hi32(out, 64));
  }
  {
    using namespace k_cells;
    std::vector<S_Cell> cells(200);
    for (uint i = 0; i < 200; i++) {
      cells[i] = S_Cell{float(i) * 0.5f, int(i) - 100, A_uint_4{{i, 1, 2, 3}}};
    }
    UmParams p{};
    p.count = 200; p.row = ROW; p.len_out = 200;
    p.p_cfg = S_Config{0.5f, 3, 2, A_int_4{{1, -2, 3, -4}}};
    p.p_on = 1;
    for (uint i = 0; i < 200; i++) um_main(p, cells.data(), grid(i));
    u64 h = 0;
    for (const S_Cell& c : cells) {
      h = h * M + scaled(c.f_a);
      h = h * M + (u64)(int64_t)c.f_b;
      for (int k = 0; k < 4; k++) h = h * M + (u64)c.f_hist.v[k];
    }
    printf("%llu\n", (unsigned long long)h);
  }
  {
    using namespace k_collatz;
    std::vector<uint> out(300);
    UmParams p{};
    p.count = 300; p.row = ROW; p.len_steps = 300;
    for (uint i = 0; i < 300; i++) um_main(p, out.data(), grid(i));
    printf("%llu\n", (unsigned long long)hu32(out, 300));
  }
  {
    using namespace k_sums;
    std::vector<int> out(100);
    UmParams p{};
    p.count = 100; p.row = ROW; p.len_out = 100; p.len_src = n;
    for (uint i = 0; i < 100; i++) um_main(p, out.data(), ib.data(), grid(i));
    printf("%llu\n", (unsigned long long)hi32(out, 100));
  }
  {
    using namespace k_loops;
    std::vector<uint> out(n);
    UmParams p{};
    p.count = n; p.row = ROW; p.len_out = n; p.len_src = n + 64;
    for (uint i = 0; i < n; i++) um_main(p, out.data(), ua.data(), grid(i));
    printf("%llu\n", (unsigned long long)hu32(out, n));
  }
  {
    using namespace k_misc;
    std::vector<int> out(n);
    UmParams p{};
    p.count = n; p.row = ROW; p.len_out = n; p.len_a = n; p.len_fl = n;
    for (uint i = 0; i < n; i++) um_main(p, out.data(), ia.data(), fs.data(), grid(i));
    printf("%llu\n", (unsigned long long)hi32(out, n));
  }
  return 0;
}
