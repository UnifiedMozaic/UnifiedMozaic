// ベンチマーク実行: node bench/run.ts
// 各 bench/<name>.um を host 向け -O2 でビルドし、bench/<name>.c を clang -O2 -fwrapv でビルドして
// 出力の一致を確認し、5回実行した最短の実時間を表にする。出力が違えば終了コード 1。

import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, readdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { build, CLANG, compileToLL } from '../src/driver.ts';

const dir = import.meta.dirname;
const tmp = mkdtempSync(join(tmpdir(), 'um-bench-'));
const RUNS = 5;

function runOnce(exe: string): { ms: number; out: string } {
  const t0 = process.hrtime.bigint();
  const r = spawnSync(exe, [], { encoding: 'utf8', maxBuffer: 1 << 24 });
  const ms = Number(process.hrtime.bigint() - t0) / 1e6;
  if (r.status !== 0) throw new Error(`${exe} exited with ${r.status}: ${r.stderr}`);
  return { ms, out: r.stdout };
}

function best(exe: string): { ms: number; out: string } {
  let ms = Infinity;
  let out = '';
  for (let i = 0; i < RUNS; i++) {
    const r = runOnce(exe);
    ms = Math.min(ms, r.ms);
    out = r.out;
  }
  return { ms, out };
}

const names = readdirSync(dir).filter((f) => f.endsWith('.um')).map((f) => f.slice(0, -3)).sort();
const rows: string[][] = [];
let bad = 0;

try {
  for (const name of names) {
    const umExe = join(tmp, `${name}.um.bin`);
    const cExe = join(tmp, `${name}.c.bin`);
    build(compileToLL(readFileSync(join(dir, `${name}.um`), 'utf8'), 'host'), 'host', umExe, '2');
    const cc = spawnSync(CLANG, ['-O2', '-fwrapv', '-ffp-contract=off', join(dir, `${name}.c`), '-o', cExe, '-lm'], { encoding: 'utf8' });
    if (cc.status !== 0) throw new Error(`clang failed for ${name}:\n${cc.stderr}`);
    const c = best(cExe);
    const um = best(umExe);
    const same = c.out === um.out;
    if (!same) {
      bad++;
      console.error(`MISMATCH ${name}\n  C : ${JSON.stringify(c.out)}\n  UM: ${JSON.stringify(um.out)}`);
    }
    rows.push([name + (same ? '' : ' (MISMATCH)'), c.ms.toFixed(1), um.ms.toFixed(1), (um.ms / c.ms).toFixed(2)]);
  }
} finally {
  rmSync(tmp, { recursive: true, force: true });
}

const header = ['name', 'C (ms)', 'UM (ms)', 'um/c'];
const all = [header, ...rows];
const w = header.map((_, i) => Math.max(...all.map((r) => r[i].length)));
for (const [i, r] of all.entries()) {
  console.log(r.map((c, j) => (j === 0 ? c.padEnd(w[j]) : c.padStart(w[j]))).join('  '));
  if (i === 0) console.log(w.map((n) => '-'.repeat(n)).join('  '));
}
process.exit(bad === 0 ? 0 : 1);
