#!/bin/sh
# Mac（Apple Silicon）での確認をまとめて行い、結果を1つのレポートに書く。
#
# これまで Linux でしか確かめていない点を調べる:
#   1. テスト一式（ネイティブバイナリを実際に実行する）
#   2. ベンチマーク（C との比）
#   3. 生成した MSL を Metal のコンパイラに通す（実行はしない）
#
# 使い方:  sh tools/mac-check.sh        （リポジトリの一番上で）
# 必要なもの: Node.js 22.18 以上、Xcode のコマンドラインツール（clang）。
#            3 には Xcode 本体（xcrun metal）が要る。なければ 3 は飛ばす。
# 外部と通信せず、リポジトリの out/ の下にだけ書き込む。

set -u
cd "$(dirname "$0")/.." || exit 1
OUT=out/mac-check
REPORT=$OUT/report.txt
mkdir -p "$OUT/msl"
: > "$REPORT"

say() { printf '%s\n' "$*" | tee -a "$REPORT"; }

say "== 環境 =="
say "date: $(date)"
say "uname: $(uname -a)"
say "node: $(node --version 2>&1)"
CLANG_BIN=${UM_CLANG:-clang}
say "clang: $($CLANG_BIN --version 2>&1 | head -1) ($CLANG_BIN)"
say "cpu: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
say "cores: $(sysctl -n hw.ncpu 2>/dev/null || echo unknown)"
say ""

say "== 1. テスト =="
if node tests/run.ts > "$OUT/tests.log" 2>&1; then
  say "結果: 成功"
else
  say "結果: 失敗あり（下に失敗した項目）"
fi
grep -E 'passed, |飛ばし|^note:' "$OUT/tests.log" | tee -a "$REPORT"
say "-- 失敗した項目の名前 --"
grep '^FAIL' "$OUT/tests.log" | head -60 >> "$REPORT"
say "-- 最初の失敗5件の詳細 --"
grep -A 14 '^FAIL' "$OUT/tests.log" | head -90 >> "$REPORT"
say "-- Metal での実行（GPU と CPU の結果の比較） --"
grep '\[metal\]' "$OUT/tests.log" | tee -a "$REPORT"
say "-- 浮動小数点の演算ごとの CPU と GPU の差（番号: ずれた個数 最大の相対差） --"
if UM_METAL=1 node src/cli.ts build tools/metal/float_ops.um --target host -o "$OUT/float_ops" >> "$REPORT" 2>&1; then
  "$OUT/float_ops" 2>&1 | tee -a "$REPORT"
  say "-- 同じものを、コピーする方式（UM_GPU_COPY=1）で --"
  UM_GPU_COPY=1 "$OUT/float_ops" 2>&1 | tee -a "$REPORT"
fi
say ""

say "== 2. ベンチマーク（ms、um/c は C に対する比） =="
node bench/run.ts 2>&1 | tee -a "$REPORT"
say ""
say "-- kernel のスレッド数ごとの時間 (kernelblur) --"
if node src/cli.ts build bench/kernelblur.um --target host -o "$OUT/kernelblur" >> "$REPORT" 2>&1; then
  for t in 1 2 4 8; do
    # 1つの node の中で3回実行し、いちばん短い時間を取る（node の起動時間を含めないため）。
    ms=$(UM_THREADS=$t node -e '
      const { spawnSync } = require("node:child_process");
      let best = Infinity;
      for (let i = 0; i < 3; i++) {
        const t0 = process.hrtime.bigint();
        spawnSync(process.argv[1], [], { stdio: "ignore" });
        best = Math.min(best, Number(process.hrtime.bigint() - t0) / 1e6);
      }
      console.log(best.toFixed(1));' "$OUT/kernelblur")
    say "UM_THREADS=$t: $ms ms"
  done
fi
say ""

say "== 3. MSL を Metal のコンパイラに通す =="
if xcrun -sdk macosx metal --version > /dev/null 2>&1; then
  say "metal: $(xcrun -sdk macosx metal --version 2>&1 | head -1)"
  ok=0
  bad=0
  for src in examples/*.um tests/gpu/*.um tests/msl/kernels.um; do
    name=$(basename "$src" .um)
    node src/cli.ts build "$src" --target host --emit-msl -o "$OUT/msl/$name" > /dev/null 2>&1 || continue
    for m in "$OUT/msl/$name".*.metal; do
      [ -f "$m" ] || continue
      if xcrun -sdk macosx metal -c "$m" -o "$m.air" > "$m.log" 2>&1; then
        ok=$((ok + 1))
      else
        bad=$((bad + 1))
        say "NG: $m"
        head -12 "$m.log" >> "$REPORT"
      fi
    done
  done
  say "コンパイルできた kernel: $ok / 失敗: $bad"
else
  say "xcrun metal が見つからないので飛ばした（Xcode 本体が必要）"
fi
say ""
say "レポート: $REPORT"
say "このファイルの中身をそのまま貼ってください。"
