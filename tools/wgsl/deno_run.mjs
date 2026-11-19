// WGSL の kernel を Deno の WebGPU（wgpu）で実行する、確認用の道具。ブラウザなしで WGSL の出力を検証できる。
// 使い方: node src/cli.ts build x.um --emit-wgsl -o out （out.<kernel>.wgsl ができる。リンカがなくても WGSL は出る）
//         deno run -A tools/wgsl/deno_run.mjs spec.json
// spec: { "wgsl": "out.k.wgsl", "count": n, "buffers": [{ "type": "i8", "data": [...], "writable": true }], "params": [{ "type": "u32", "value": 3 }] }
//   buffers は kernel の Buffer の引数の順、params は値の引数の順（4 バイトずつ）。type は i8 / u8 / i16 / u16 / i32 / u32 / f32。
// 出力: 書き込む Buffer の中身（{ "番号": [...] } の JSON）。CPU（cpu.dispatch）の結果と比べて使う。
const spec = JSON.parse(Deno.readTextFileSync(Deno.args[0]));
const src = Deno.readTextFileSync(spec.wgsl);
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();
const mod = device.createShaderModule({ code: src });
const info = await mod.getCompilationInfo();
for (const m of info.messages) console.error(m.type, m.lineNum, m.message);
const T = { i8: Int8Array, u8: Uint8Array, i16: Int16Array, u16: Uint16Array, i32: Int32Array, u32: Uint32Array, f32: Float32Array };
const WG = 64;
const groups = Math.ceil(spec.count / WG);
const nb = spec.buffers.length;
const header = 2 + nb;
const pbytes = new ArrayBuffer((header + spec.params.length) * 4);
const pw = new Uint32Array(pbytes);
pw[0] = spec.count; pw[1] = groups * WG;
spec.buffers.forEach((b, i) => (pw[2 + i] = b.data.length));
spec.params.forEach((p, i) => { new T[p.type](pbytes, (header + i) * 4, 1)[0] = p.value; });
const mk = (size, usage) => device.createBuffer({ size: Math.max(4, Math.ceil(size / 4) * 4), usage });
const pb = mk(pbytes.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
device.queue.writeBuffer(pb, 0, pbytes);
const entries = [{ binding: 0, resource: { buffer: pb } }];
const outs = [];
spec.buffers.forEach((b, i) => {
  const arr = new T[b.type](Math.max(b.data.length, 1));
  arr.set(b.data);
  const bytes = Math.ceil(arr.byteLength / 4) * 4;
  const padded = new Uint8Array(bytes); padded.set(new Uint8Array(arr.buffer));
  const g = mk(bytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
  device.queue.writeBuffer(g, 0, padded);
  entries.push({ binding: i + 1, resource: { buffer: g } });
  if (b.writable) outs.push({ i, g, bytes, b });
});
const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module: mod, entryPoint: 'um_main' } });
const enc = device.createCommandEncoder();
const pass = enc.beginComputePass();
pass.setPipeline(pipeline);
pass.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries }));
pass.dispatchWorkgroups(groups, 1);
pass.end();
const stages = outs.map((o) => { const s = mk(o.bytes, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST); enc.copyBufferToBuffer(o.g, 0, s, 0, o.bytes); return s; });
device.queue.submit([enc.finish()]);
const result = {};
for (let k = 0; k < outs.length; k++) {
  await stages[k].mapAsync(GPUMapMode.READ);
  const a = new T[outs[k].b.type](stages[k].getMappedRange().slice(0));
  result[outs[k].i] = Array.from(a.subarray(0, outs[k].b.data.length));
}
console.log(JSON.stringify(result));
