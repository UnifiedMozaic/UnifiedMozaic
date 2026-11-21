#!/usr/bin/env bash
# 失敗した CI の中身を注釈（::error::）にする。注釈は実行の画面の一覧に出て、ログを開かなくても読める。
#   test.log があれば、FAIL の行とその下の説明（字下げした行）を、まとめて注釈にする。
#   どのログも、最後の 40 行を注釈にする（準備の段階で止まったときのため）。
# 使い方: report-failures.sh <名前> <ログ>...
set -uo pipefail

name=$1
shift

# 注釈の本文に入れられない文字（%・改行）を置き換える。
escape() {
  local s=$1
  s=${s//'%'/'%25'}
  s=${s//$'\r'/}
  s=${s//$'\n'/'%0A'}
  printf '%s' "$s"
}

annotate() {
  printf '::error title=%s::%s\n' "$1" "$(escape "$2")"
}

for log in "$@"; do
  [ -f "$log" ] || continue
  if [ "$(basename "$log")" = test.log ]; then
    # FAIL の塊（FAIL の行と、続く字下げした行のうち 12 行まで）を集め、4000 文字ごとに 1 つの注釈にする
    # （1 つの段階に付けられる注釈は 10 個まで）。
    chunk=''
    count=0
    inblock=0
    detail=0
    while IFS= read -r line; do
      line=${line%$'\r'}
      if [[ $line == 'FAIL '* ]]; then
        if [ ${#chunk} -gt 4000 ]; then
          annotate "failed tests ($name)" "$chunk"
          chunk=''
          count=$((count + 1))
          [ $count -ge 8 ] && break
        fi
        chunk+="$line"$'\n'
        inblock=1
        detail=0
      elif [ $inblock = 1 ] && [[ $line == '     '* ]]; then
        detail=$((detail + 1))
        [ $detail -le 12 ] && chunk+="$line"$'\n'
      else
        inblock=0
      fi
    done < "$log"
    [ -n "$chunk" ] && annotate "failed tests ($name)" "$chunk"
  fi
  annotate "last lines of $(basename "$log") ($name)" "$(tail -n 40 "$log")"
done
exit 0
