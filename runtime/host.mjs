// UnifiedMozaic の wasm 用ランタイム。ブラウザと Node の両方で動く。
// 依存なし。eval や動的なコード生成は使わない。
//
// 入出力（readLine / readFile / writeFile）は、instantiate に渡された口（options）だけを通す。
// 口が渡されなければ、標準入力は「終わり」、ファイルは「読めない・書けない」として動く。
//
// GPU（WebGPU）での kernel 実行について:
//   - WebGPU の読み戻しは非同期でしか行えない。wasm の中の dispatch は同期的な呼び出しなので、
//     JSPI（JavaScript Promise Integration）で wasm の実行を一時停止して待つ。
//   - JSPI か WebGPU が使えない環境では GPU なしとして動き、dispatch は CPU で実行される。
//   - WebGPU では CPU と GPU のメモリが別なので、dispatch のたびに Buffer を送り、
//     書き込み先の Buffer を読み戻す。

// 原因コードは src/codegen.ts の PANIC_* と揃えること。
const PANIC_MESSAGES = {
  1: 'index out of bounds',
  2: 'division by zero',
  3: 'shift amount too large',
  4: 'pop from an empty array',
  5: 'out of memory',
  6: 'dispatch count exceeds the length of a written Buffer',
  7: 'the GPU is not available',
  8: 'could not start a thread',
  9: 'readLine and inputReady can only be called from the main thread',
  10: 'invalid group size (it must be 1 to 1024, and groups times size must fit in u32)',
  16: 'invalid file call (an unknown or closed file, an unknown open mode, or a range outside the array)',
};

// src/wgsl.ts の WORKGROUP_SIZE と揃えること。
const WORKGROUP_SIZE = 64;
// WebGPU が既定で保証する、1次元あたりのワークグループ数の上限。
const MAX_GROUPS = 65535;

export class UmPanic extends Error {
  constructor(code, line) {
    super(`panic: ${PANIC_MESSAGES[code] ?? 'unknown'} (line ${line})`);
    this.name = 'UmPanic';
    this.code = code;
    this.line = line;
  }
}

const jspi = typeof WebAssembly.Suspending === 'function' && typeof WebAssembly.promising === 'function';

/** WebGPU のデバイスを取得する。使えなければ null。 */
async function requestDevice() {
  const gpu = globalThis.navigator?.gpu;
  if (!gpu || !jspi) return null;
  try {
    const adapter = await gpu.requestAdapter();
    if (!adapter) return null;
    // dispatchGroups の大きなグループ（既定の上限は 256 スレッド）のために、アダプタが許す上限まで求める。
    const want = ['maxComputeInvocationsPerWorkgroup', 'maxComputeWorkgroupSizeX', 'maxComputeWorkgroupStorageSize'];
    const requiredLimits = {};
    for (const k of want) if (typeof adapter.limits?.[k] === 'number') requiredLimits[k] = adapter.limits[k];
    try {
      return await adapter.requestDevice({ requiredLimits });
    } catch {
      return await adapter.requestDevice();
    }
  } catch {
    return null;
  }
}

/** kernel を GPU で実行する仕組み。1つの wasm インスタンスにつき1つ作る。 */
class GpuRunner {
  constructor(device, memory) {
    this.device = device;
    this.memory = memory;
    /** kernel の番号 → コンパイル済みのパイプライン。 */
    this.pipelines = new Map();
    /** 実行した回数（デバッグ用）。 */
    this.dispatches = 0;
    /** 次の dispatch のグループの大きさ（um_gpu_group が設定し、dispatch が使って 0 に戻す）。0 は普通の dispatch。 */
    this.group = 0;
  }

  bytes() {
    // メモリが伸びると buffer が差し替わるので、使うたびに取り直す。
    return new Uint8Array(this.memory.buffer);
  }

  async pipeline(id, wgslPtr, wgslLen, descs, group) {
    // dispatchGroups の kernel は、グループの大きさ（シェーダの定数 um_wg）ごとに別のパイプラインになる。
    const key = group ? `${id}:${group}` : id;
    let p = this.pipelines.get(key);
    if (p) return p;
    const code = new TextDecoder().decode(this.bytes().slice(wgslPtr, wgslPtr + wgslLen));
    const module = this.device.createShaderModule({ code });
    // 使っていない Buffer があっても並びがずれないよう、バインドの配置は明示する。
    const entries = [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }];
    descs.forEach((d, i) => {
      entries.push({ binding: i + 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: d.writable ? 'storage' : 'read-only-storage' } });
    });
    const groupLayout = this.device.createBindGroupLayout({ entries });
    const pipeline = await this.device.createComputePipelineAsync({
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [groupLayout] }),
      compute: group ? { module, entryPoint: 'um_main', constants: { um_wg: group } } : { module, entryPoint: 'um_main' },
    });
    p = { pipeline, groupLayout };
    this.pipelines.set(key, p);
    return p;
  }

  async dispatch(id, wgslPtr, wgslLen, count, paramsPtr, paramBytes, descPtr, nbufs) {
    const group = this.group;
    this.group = 0;
    count >>>= 0;
    if (count === 0) return;
    const device = this.device;
    if (group > device.limits.maxComputeInvocationsPerWorkgroup || group > device.limits.maxComputeWorkgroupSizeX) {
      throw new Error(`UnifiedMozaic: the group size ${group} is larger than this GPU allows (${device.limits.maxComputeInvocationsPerWorkgroup})`);
    }
    const table = new Uint32Array(this.memory.buffer, descPtr, nbufs * 4);
    const descs = [];
    for (let i = 0; i < nbufs; i++) {
      descs.push({ ptr: table[i * 4], len: table[i * 4 + 1], elemBytes: table[i * 4 + 2], writable: table[i * 4 + 3] !== 0 });
    }

    device.pushErrorScope('validation');
    let made = [];
    try {
      const { pipeline, groupLayout } = await this.pipeline(id, wgslPtr, wgslLen, descs, group);

      // 要素数をワークグループの2次元の格子に割り当てる。dispatchGroups では、グループがそのままワークグループ。
      const groups = group ? count / group : Math.ceil(count / WORKGROUP_SIZE);
      const x = Math.min(groups, MAX_GROUPS);
      const y = Math.ceil(groups / x);

      // 引数の塊: 個数、1行の要素数、各 Buffer の長さ、値の引数（wasm 側が詰めた並びのまま）。
      const header = 2 + nbufs;
      const params = new Uint8Array(header * 4 + paramBytes);
      const words = new Uint32Array(params.buffer, 0, header);
      // dispatchGroups の形（src/wgsl.ts）では、個数はグループの数、1行はグループの数で数える。
      words[0] = group ? groups : count;
      words[1] = group ? x : x * WORKGROUP_SIZE;
      descs.forEach((d, i) => (words[2 + i] = d.len));
      params.set(this.bytes().subarray(paramsPtr, paramsPtr + paramBytes), header * 4);

      const make = (size, usage) => {
        const b = device.createBuffer({ size, usage });
        made.push(b);
        return b;
      };
      const paramBuffer = make(params.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
      device.queue.writeBuffer(paramBuffer, 0, params);

      const entries = [{ binding: 0, resource: { buffer: paramBuffer } }];
      const outputs = [];
      descs.forEach((d, i) => {
        // 長さ 0 でも1要素ぶん確保する（CPU 側と同じ。添字の切り詰め先を常に有効にするため）。
        // 8 / 16 ビットの要素の Buffer は、4 バイトの倍数に切り上げて写す（WebGPU の写しは 4 バイト単位。
        // CPU 側の確保も切り上げてあるので、はみ出しはない）。
        const used = Math.ceil((d.len * d.elemBytes) / 4) * 4;
        const size = Math.ceil((Math.max(d.len, 1) * d.elemBytes) / 4) * 4;
        const b = make(size, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
        if (used > 0) device.queue.writeBuffer(b, 0, this.bytes(), d.ptr, used);
        entries.push({ binding: i + 1, resource: { buffer: b } });
        if (d.writable && used > 0) {
          outputs.push({ d, source: b, staging: make(used, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST), used });
        }
      });

      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, device.createBindGroup({ layout: groupLayout, entries }));
      pass.dispatchWorkgroups(x, y);
      pass.end();
      for (const o of outputs) encoder.copyBufferToBuffer(o.source, 0, o.staging, 0, o.used);
      device.queue.submit([encoder.finish()]);

      const error = await device.popErrorScope();
      if (error) throw new Error(`UnifiedMozaic: GPU dispatch failed: ${error.message}`);

      await Promise.all(outputs.map((o) => o.staging.mapAsync(GPUMapMode.READ)));
      for (const o of outputs) {
        this.bytes().set(new Uint8Array(o.staging.getMappedRange()), o.d.ptr);
        o.staging.unmap();
      }
      this.dispatches++;
    } finally {
      for (const b of made) b.destroy();
      made = [];
    }
  }
}

/**
 * コンパイル済みの wasm を読み込む。
 *
 * 返す関数は、GPU が使える環境では Promise を返す（wasm の中で GPU の完了を待つため）。
 * どちらの環境でも動くよう、呼ぶ側は常に `await exports.main()` と書くこと。
 *
 * @param {BufferSource} bytes wasm のバイト列
 * @param {((line: string) => void) | {
 *   print?: (line: string) => void,
 *   gpu?: boolean,
 *   gpuThreshold?: number,
 *   readLine?: () => string | Uint8Array | null,
 *   inputReady?: () => boolean,
 *   readFile?: (path: string) => Uint8Array | null,
 *   writeFile?: (path: string, bytes: Uint8Array) => boolean,
 *   fileSize?: (path: string) => number | bigint | null,
 *   readFileRange?: (path: string, offset: number, length: number) => Uint8Array | null,
 *   appendFile?: (path: string, bytes: Uint8Array) => boolean,
 *   args?: string[],
 *   eprint?: (line: string) => void,
 *   fileOpen?: (path: string, mode: 'r' | 'w' | 'a' | 'r+') => unknown,
 *   fileRead?: (file: unknown, length: number, offset: number | null) => Uint8Array | null,
 *   fileWrite?: (file: unknown, bytes: Uint8Array) => boolean,
 *   fileSeek?: (file: unknown, offset: number) => boolean,
 *   fileLength?: (file: unknown) => number | bigint | null,
 *   fileClose?: (file: unknown) => void,
 * }} [options] print の出力先（関数を直接渡してもよい）と GPU の設定、入出力の口。
 *   gpu: false で GPU を使わない。gpuThreshold: 自動判断で GPU に回す最小の要素数（既定 4096）。
 *   readLine: プログラムの readLine() に渡す次の1行（改行を除く）を返す。もうなければ null。同期的に返すこと。
 *     渡さなければ、readLine() は常に null（入力の終わり）になる。
 *   inputReady: readLine が待たずに返せるなら true。渡さなければ常に true
 *     （readLine の口は待たずに返すもので、それもなければ入力は終わっているため）。
 *   readFile / writeFile: プログラムの readFile() / writeFile() の実体。渡さなければ、ファイルは読めず（null）、書けない（false）。
 *     名前は UTF-8 として文字列にして渡す（0 のバイトを含む名前と、UTF-8 として読めない名前は、口を呼ばずに失敗にする）。
 *   fileSize / readFileRange / appendFile: プログラムの fileSize() / readFileRange() / appendFile() の実体。
 *     渡さなければ readFile（と writeFile）の口で代わりに行う（ファイル全体を読むので遅い）。どちらもなければ失敗（null / false）。
 *     readFileRange が length より長いデータを返したときは、length バイトに切り詰める。
 *   args: プログラムの args() が返す引数（プログラム名を含めない）。渡さなければ空。
 *   eprint: プログラムの eprint() の出力先。渡さなければ console.error。
 *   readFileInto() は readFileRange の口（なければ readFile の口）を通り、データを Buffer へ直接写す。
 *   fileOpen / fileRead / fileWrite / fileSeek / fileLength / fileClose: 開いたままのファイル。fileOpen が返した値（null 以外）を
 *     ほかの口に渡す（プログラムに見せる番号は、ここで別に作る）。fileRead の offset が null なら今の位置から読み、
 *     数なら offset から読んで位置は動かさない。fileOpen の口がなければ、readFile / writeFile / appendFile の口で代わりに行う
 *     （"r" と "r+" は開くときに全体を読む。"w" と "a" は書くたびに書き足す。"w" / "a" の fileSeek は false）。
 * @returns {Promise<WebAssembly.Exports & { um_gpu?: { available: boolean, dispatches: () => number } }>}
 */
export async function instantiate(bytes, options = {}) {
  if (typeof options === 'function') options = { print: options };
  const print = options.print ?? ((s) => console.log(s));
  const threshold = options.gpuThreshold ?? 4096;
  const device = options.gpu === false ? null : await requestDevice();
  let runner = null;

  // 入出力。ホストが用意したデータ（1行、ファイルの中身）は、wasm 側が領域を確保したあとで um_host_take が写す。
  const mem = (ptr, len) => new Uint8Array(instance.exports.memory.buffer, Number(ptr) >>> 0, Number(len) >>> 0);
  const origin = globalThis.performance.now();
  let pending = null;
  const prepare = (data) => {
    if (data === null || data === undefined) return -1;
    const bytes = typeof data === 'string' ? new TextEncoder().encode(data) : data;
    if (!(bytes instanceof Uint8Array)) return -1;
    // wasm 側は長さを符号つき 32 ビットで受け取る。
    if (bytes.length > 0x7fffffff) return -1;
    pending = bytes;
    return bytes.length;
  };
  // 名前を文字列にする。0 のバイトを含む名前は受け付けない（ネイティブと同じ）。
  const pathOf = (ptr, len) => {
    const bytes = mem(ptr, len).slice();
    if (bytes.length === 0 || bytes.includes(0)) return null;
    // UTF-8 として読めない名前も受け付けない。読めないバイトを別の文字に置き換えると、
    // プログラムが指定したのとは別の（ほかの名前と同じになりうる）ファイルを指してしまう。
    try {
      return new TextDecoder('utf-8', { fatal: true, ignoreBOM: true }).decode(bytes);
    } catch {
      return null;
    }
  };

  // 書き足す。appendFile の口がなければ、readFile / writeFile の口で全体を書き直す。
  const appendBytes = (path, bytes) => {
    try {
      if (options.appendFile) return !!options.appendFile(path, bytes);
      if (!options.writeFile) return false;
      const old = options.readFile ? options.readFile(path) : null;
      if (!(old instanceof Uint8Array)) return !!options.writeFile(path, bytes);
      const joined = new Uint8Array(old.length + bytes.length);
      joined.set(old);
      joined.set(bytes, old.length);
      return !!options.writeFile(path, joined);
    } catch {
      return false;
    }
  };
  // 開いたままのファイル（プログラムに見せる番号 → 口の値か、代わりに行うときの状態）。
  const files = new Map();
  let nextFile = 1;

  const env = {
    um_print_i64: (v) => print(BigInt.asIntN(64, v).toString()),
    um_print_u64: (v) => print(BigInt.asUintN(64, v).toString()),
    um_print_f64: (v) => print(Object.is(v, -0) ? '0' : String(v)),
    um_print_bool: (v) => print(v ? 'true' : 'false'),
    um_print_str: (ptr, len) => {
      // 不正な UTF-8 は置換文字になる（停止しない）。
      const bytes = new Uint8Array(instance.exports.memory.buffer, Number(ptr), Number(len));
      print(new TextDecoder().decode(bytes.slice()));
    },
    um_fmt_f64: (v, ptr) => {
      const text = Object.is(v, -0) ? '0' : String(v);
      new Uint8Array(instance.exports.memory.buffer, ptr, text.length).set([...text].map((c) => c.charCodeAt(0)));
      return text.length;
    },
    um_panic: (code, line) => {
      throw new UmPanic(code, line);
    },
    um_host_line: () => {
      if (!options.readLine) return -1;
      let line = options.readLine();
      if (line === null || line === undefined) return -1;
      if (typeof line === 'string') line = new TextEncoder().encode(line);
      if (!(line instanceof Uint8Array)) return -1;
      // 改行と、その前の "\r" は含めない（口が付けて返した場合も取り除く）。
      let n = line.length;
      if (n > 0 && line[n - 1] === 10) n--;
      if (n > 0 && line[n - 1] === 13) n--;
      return prepare(line.subarray(0, n));
    },
    um_host_take: (dst) => {
      if (pending) mem(dst, pending.length).set(pending);
      pending = null;
    },
    // 口がなければ 1: readLine の口は同期的に返すので待たないし、それもなければ入力は終わっている
    // （ネイティブでも、入力の終わりでは inputReady() は true）。
    um_host_input_ready: () => (options.inputReady ? (options.inputReady() ? 1 : 0) : 1),
    um_host_clock: () => (globalThis.performance.now() - origin) / 1000,
    um_host_read_file: (ptr, len) => {
      const path = pathOf(ptr, len);
      if (path === null || !options.readFile) return -1;
      try {
        return prepare(options.readFile(path));
      } catch {
        return -1;
      }
    },
    um_host_write_file: (ptr, len, dataPtr, dataLen) => {
      const path = pathOf(ptr, len);
      if (path === null || !options.writeFile) return 0;
      try {
        // 口には写しを渡す（wasm のメモリはあとで書き換わる）。
        return options.writeFile(path, mem(dataPtr, dataLen).slice()) ? 1 : 0;
      } catch {
        return 0;
      }
    },
    um_host_file_size: (ptr, len) => {
      const path = pathOf(ptr, len);
      if (path === null) return -1n;
      try {
        if (options.fileSize) {
          const n = options.fileSize(path);
          if (n === null || n === undefined) return -1n;
          const b = BigInt(n);
          return b < 0n || b > 0x7fffffffffffffffn ? -1n : b;
        }
        if (!options.readFile) return -1n;
        const all = options.readFile(path);
        return all instanceof Uint8Array ? BigInt(all.length) : -1n;
      } catch {
        return -1n;
      }
    },
    um_host_read_file_range: (ptr, len, offset, length) => {
      const path = pathOf(ptr, len);
      if (path === null || offset < 0n || length < 0n) return -1;
      // wasm 側は長さを符号つき 32 ビットで受け取る。それより長く頼まれても、それ以上は渡せない。
      const want = Number(length > 0x7fffffffn ? 0x7fffffffn : length);
      try {
        let data;
        if (options.readFileRange) {
          if (offset > BigInt(Number.MAX_SAFE_INTEGER)) return -1;
          data = options.readFileRange(path, Number(offset), want);
        } else if (options.readFile) {
          const all = options.readFile(path);
          if (!(all instanceof Uint8Array)) return -1;
          const from = offset > BigInt(all.length) ? all.length : Number(offset);
          data = all.subarray(from, from + want);
        } else {
          return -1;
        }
        if (data instanceof Uint8Array && data.length > want) data = data.subarray(0, want);
        return prepare(data);
      } catch {
        return -1;
      }
    },
    um_host_append_file: (ptr, len, dataPtr, dataLen) => {
      const path = pathOf(ptr, len);
      if (path === null) return 0;
      const bytes = mem(dataPtr, dataLen).slice();
      return appendBytes(path, bytes) ? 1 : 0;
    },
    um_host_argc: () => (Array.isArray(options.args) ? options.args.length : 0),
    um_host_arg: (i) => prepare(new TextEncoder().encode(String(options.args[i]))),
    um_host_eprint: (ptr, len) => {
      const text = new TextDecoder().decode(mem(ptr, len).slice());
      (options.eprint ?? ((s) => console.error(s)))(text);
    },
    // 開いたままのファイル。知らない・閉じた番号には -2 を返す（wasm 側が停止する）。
    um_host_file_open: (ptr, len, how) => {
      const path = pathOf(ptr, len);
      const mode = ['r', 'w', 'a', 'r+'][how];
      if (path === null || !mode) return 0;
      let entry;
      try {
        if (options.fileOpen) {
          const h = options.fileOpen(path, mode);
          if (h === null || h === undefined || h === false) return 0;
          entry = { h };
        } else if (mode === 'r' || mode === 'r+') {
          const data = options.readFile ? options.readFile(path) : null;
          if (!(data instanceof Uint8Array)) return 0;
          entry = { emu: mode, path, data: data.slice(), pos: 0 };
        } else {
          const ok = mode === 'w' ? options.writeFile && options.writeFile(path, new Uint8Array(0)) : appendBytes(path, new Uint8Array(0));
          if (!ok) return 0;
          entry = { emu: mode, path };
        }
      } catch {
        return 0;
      }
      if (nextFile > 0x7fffffff) return 0;
      files.set(nextFile, entry);
      return nextFile++;
    },
    um_host_file_read: (id, at, length) => {
      const e = files.get(id);
      if (!e) return -2;
      try {
        let data;
        if (e.emu) {
          if (e.emu !== 'r' && e.emu !== 'r+') return -1;
          const size = e.data.length;
          const from = at < 0n ? Math.min(e.pos, size) : Number(at > BigInt(size) ? BigInt(size) : at);
          data = e.data.subarray(from, from + length);
          if (at < 0n) e.pos = from + data.length;
        } else {
          if (!options.fileRead) return -1;
          data = options.fileRead(e.h, length, at < 0n ? null : Number(at));
        }
        if (data instanceof Uint8Array && data.length > length) data = data.subarray(0, length);
        return prepare(data);
      } catch {
        return -1;
      }
    },
    um_host_file_write: (id, ptr, len) => {
      const e = files.get(id);
      if (!e) return -2;
      const bytes = mem(ptr, len).slice();
      try {
        if (!e.emu) return options.fileWrite && options.fileWrite(e.h, bytes) ? 1 : 0;
        if (e.emu === 'r') return 0;
        if (e.emu !== 'r+') return appendBytes(e.path, bytes) ? 1 : 0;
        // 読み書き: 今の位置に上書きし（足りなければ伸ばし）、全体を書き戻す。
        const end = e.pos + bytes.length;
        if (end > e.data.length) {
          const grown = new Uint8Array(end);
          grown.set(e.data);
          e.data = grown;
        }
        e.data.set(bytes, e.pos);
        e.pos = end;
        return options.writeFile && options.writeFile(e.path, e.data.slice()) ? 1 : 0;
      } catch {
        return 0;
      }
    },
    um_host_file_seek: (id, offset) => {
      const e = files.get(id);
      if (!e) return -2;
      if (offset < 0n || offset > BigInt(Number.MAX_SAFE_INTEGER)) return 0;
      try {
        if (!e.emu) return options.fileSeek && options.fileSeek(e.h, Number(offset)) ? 1 : 0;
        if (e.emu !== 'r' && e.emu !== 'r+') return 0;
        e.pos = Number(offset);
        return 1;
      } catch {
        return 0;
      }
    },
    um_host_file_length: (id) => {
      const e = files.get(id);
      if (!e) return -2n;
      try {
        let n;
        if (!e.emu) n = options.fileLength ? options.fileLength(e.h) : null;
        else if (e.data) n = e.data.length;
        else {
          const all = options.readFile ? options.readFile(e.path) : null;
          n = all instanceof Uint8Array ? all.length : null;
        }
        if (n === null || n === undefined) return -1n;
        const b = BigInt(n);
        return b < 0n || b > 0x7fffffffffffffffn ? -1n : b;
      } catch {
        return -1n;
      }
    },
    um_host_file_close: (id) => {
      const e = files.get(id);
      if (!e) return -2;
      files.delete(id);
      try {
        if (!e.emu && options.fileClose) options.fileClose(e.h);
      } catch {
        // 閉じられなくても、番号はもう使えない。
      }
      return 0;
    },
    um_gpu_available: () => (runner ? 1 : 0),
    // 自動判断: いまは「GPU があり、要素数がしきい値以上」なら GPU。実測による補正は今後入れる。
    um_gpu_pick: (_id, count) => (runner && count >>> 0 >= threshold ? 1 : 0),
    // GPU がない環境では呼ばれない（生成コードが先に um_gpu_available / um_gpu_pick を確かめる）。
    um_gpu_group: (size) => {
      if (runner) runner.group = size >>> 0;
    },
    um_gpu_dispatch: device
      ? new WebAssembly.Suspending((...args) => runner.dispatch(...args))
      : () => {
          throw new UmPanic(7, 0);
        },
  };

  const { instance } = await WebAssembly.instantiate(bytes, { env });
  if (!device) return instance.exports;

  runner = new GpuRunner(device, instance.exports.memory);
  const exports = { um_gpu: { available: true, dispatches: () => runner.dispatches } };
  for (const [name, value] of Object.entries(instance.exports)) {
    // GPU の完了を待てるよう、export された関数は Promise を返す形に包む。
    exports[name] = typeof value === 'function' ? WebAssembly.promising(value) : value;
  }
  return exports;
}
