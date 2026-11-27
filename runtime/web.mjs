// ブラウザでのウィンドウ（canvas）と、WebGPU での描画。runtime/host.mjs が使う。依存なし。
//
// ウィンドウ（windowOpen など。ネイティブは runtime/window.c）:
//   - ウィンドウは canvas。最初のウィンドウは options.canvas（なければ作って options.container か document.body に足す）。
//   - イベントは DOM のイベントを、ネイティブと同じ 6 つの整数 [種類, ウィンドウ, a, b, c, d] にする（キーは SDL のキーコード）。
//   - windowPresent は canvas の 2D に描き、次の描画の時機（requestAnimationFrame）まで wasm を一時停止する（JSPI）。
//     そのあいだにブラウザが画面を描き、入力を届けるので、`while (running) { ... }` のループのまま動く。
//   - 待つ pollWindowEvent（timeoutMs が 0 でない）も、イベントが来るまで一時停止する。
//
// 描画（gpu.beginImage / gpu.draw / gpu.beginFrame など。ネイティブは runtime/wgpu.c）:
//   - 画像は色（bgra8unorm）と深度（depth32float）のテクスチャ。フレームは canvas の webgpu の面に描く。
//   - 1 つの canvas は 2D か webgpu のどちらかしか使えないので、GPU で描き始めたウィンドウに windowPresent は使えない。

// イベントの種類（runtime/window.c と揃える）。
const EV = { CLOSE: 1, RESIZE: 2, MOUSE_MOVE: 3, MOUSE_DOWN: 4, MOUSE_UP: 5, WHEEL: 6, KEY_DOWN: 7, KEY_UP: 8, TEXT: 9, PIXEL_SIZE: 18, SCALE: 19, FOCUS: 20, MOUSE_ENTER: 21 };
// windowCtl の操作番号（runtime/window.c と揃える）。
const WC = {
  SET_SIZE: 1, GET_SIZE: 2, GET_POS: 4, STATE: 10, PIXEL_SIZE: 11, DISPLAY: 12, DISPLAY_COUNT: 13, DISPLAY_BOUNDS: 14, DISPLAY_USABLE: 15,
  START_TEXT: 16, STOP_TEXT: 17, TEXT_ACTIVE: 18, CURSOR: 20, SHOW_CURSOR: 21, SET_FULLSCREEN: 9, TICKS_NS: 30, SLEEP_NS: 31,
};
const PANIC_WINDOW_ARGS = 12;

// SDL のキーコード（runtime/window.um の keyOf と揃える）。
const KEYS = {
  Enter: 13, Escape: 27, Backspace: 8, Tab: 9, ' ': 32, Delete: 127,
  ArrowLeft: 0x40000050, ArrowRight: 0x4000004f, ArrowUp: 0x40000052, ArrowDown: 0x40000051,
  Home: 0x4000004a, End: 0x4000004d, PageUp: 0x4000004b, PageDown: 0x4000004e,
};

function keyCode(e) {
  if (e.key in KEYS) return KEYS[e.key];
  const f = /^F([1-9]|1[0-2])$/.exec(e.key);
  if (f) return 0x40000039 + Number(f[1]);
  // 文字のキーは Shift によらない小文字（SDL と同じ）。配列の違いは e.key で拾う。
  if (/^Key[A-Z]$/.test(e.code)) return e.code.charCodeAt(3) + 32;
  if (/^Digit[0-9]$/.test(e.code)) return e.code.charCodeAt(5);
  if (e.key.length === 1) return e.key.toLowerCase().codePointAt(0);
  return 0x40000000 | (e.keyCode & 0xffff);
}

function modsOf(e) {
  return (e.shiftKey ? 1 : 0) | (e.ctrlKey ? 2 : 0) | (e.altKey ? 4 : 0) | (e.metaKey ? 8 : 0);
}

// SDL_SystemCursor の番号 → CSS の cursor（windowSetCursor）。
const CURSORS = ['default', 'text', 'wait', 'crosshair', 'progress', 'nwse-resize', 'nesw-resize', 'ew-resize', 'ns-resize', 'move', 'not-allowed', 'pointer'];

const nextFrame = () => new Promise((resolve) => (globalThis.requestAnimationFrame ? requestAnimationFrame(() => resolve()) : setTimeout(resolve, 16)));

/**
 * ウィンドウの口（um_host_window_*）を作る。
 * mem(): wasm のメモリの ArrayBuffer、prepare(data): 文字列を wasm に渡す準備（host.mjs の prepare）、Panic: UmPanic。
 */
export function windowHost(options, mem, prepare, Panic) {
  const doc = globalThis.document;
  const wins = [];  // 番号 n は wins[n - 1]
  const queue = [];
  let waiter = null;
  let textActive = true;
  let lastError = doc ? '' : 'windows need a browser (no document)';
  const dpr = () => globalThis.devicePixelRatio || 1;

  const push = (kind, id, a = 0, b = 0, c = 0, d = 0) => {
    queue.push([kind, id, a | 0, b | 0, c | 0, d | 0]);
    if (waiter) {
      const w = waiter;
      waiter = null;
      w();
    }
  };
  const lookup = (id, line) => {
    const w = wins[id - 1];
    if (!w) throw new Panic(PANIC_WINDOW_ARGS, line);
    return w;
  };
  // canvas の表示の大きさ（CSS の画素）と描画の面（画素）を合わせる。
  const fit = (w) => {
    const r = w.canvas.getBoundingClientRect();
    const cw = Math.max(1, Math.round(r.width || w.width));
    const ch = Math.max(1, Math.round(r.height || w.height));
    if (cw !== w.width || ch !== w.height) {
      w.width = cw;
      w.height = ch;
      push(EV.RESIZE, w.id, cw, ch);
    }
    const pw = Math.max(1, Math.round(cw * dpr()));
    const ph = Math.max(1, Math.round(ch * dpr()));
    if (pw !== w.canvas.width || ph !== w.canvas.height) {
      w.canvas.width = pw;
      w.canvas.height = ph;
      push(EV.PIXEL_SIZE, w.id, pw, ph);
    }
  };
  const pos = (w, e) => {
    const r = w.canvas.getBoundingClientRect();
    return [Math.round(e.clientX - r.left), Math.round(e.clientY - r.top)];
  };

  const env = {
    um_host_window_open(ptr, len, width, height, line) {
      if (!doc) return 0;
      const title = new TextDecoder().decode(new Uint8Array(mem(), ptr >>> 0, len >>> 0));
      const canvas = wins.length === 0 && options.canvas ? options.canvas : doc.createElement('canvas');
      if (!canvas.isConnected) (options.container ?? doc.body).appendChild(canvas);
      if (!canvas.style.width) canvas.style.width = `${width}px`;
      if (!canvas.style.height) canvas.style.height = `${height}px`;
      canvas.tabIndex = 0;
      canvas.style.outline = 'none';
      if (wins.length === 0 && title) doc.title = title;
      const id = wins.length + 1;
      const w = { id, canvas, title, width: 0, height: 0, ctx2d: null, gpu: null, image: null, scratch: null };
      wins.push(w);
      fit(w);
      // 開いた直後の大きさの知らせは要らない（ネイティブと同じく、変わったときだけ）。
      queue.length = 0;
      const on = (target, type, f, opt) => target.addEventListener(type, f, opt);
      on(canvas, 'mousemove', (e) => push(EV.MOUSE_MOVE, id, ...pos(w, e)));
      on(canvas, 'mousedown', (e) => {
        canvas.focus();
        push(EV.MOUSE_DOWN, id, ...pos(w, e), [1, 2, 3][e.button] ?? e.button + 1);
      });
      on(canvas, 'mouseup', (e) => push(EV.MOUSE_UP, id, ...pos(w, e), [1, 2, 3][e.button] ?? e.button + 1));
      on(canvas, 'contextmenu', (e) => e.preventDefault());
      on(canvas, 'mouseenter', () => push(EV.MOUSE_ENTER, id, 1));
      on(canvas, 'mouseleave', () => push(EV.MOUSE_ENTER, id, 0));
      on(canvas, 'wheel', (e) => {
        e.preventDefault();
        // 上・右が正（SDL と同じ）。画素の量は 100 画素で 1 ノッチとみなす。
        const scale = e.deltaMode === 1 ? 1 / 3 : e.deltaMode === 2 ? 1 : 1 / 100;
        const fx = -e.deltaX * scale;
        const fy = -e.deltaY * scale;
        push(EV.WHEEL, id, Math.trunc(fx), Math.trunc(fy), Math.round(fx * 1000), Math.round(fy * 1000));
      }, { passive: false });
      on(canvas, 'keydown', (e) => {
        push(EV.KEY_DOWN, id, keyCode(e), modsOf(e), e.repeat ? 1 : 0);
        if (textActive && e.key.length === 1 && !e.ctrlKey && !e.metaKey) push(EV.TEXT, id, e.key.codePointAt(0));
        // ページを動かすキー（矢印・Space など）とブラウザの既定の動作は止める（F5 と F12 は残す）。
        if (e.key !== 'F5' && e.key !== 'F12') e.preventDefault();
      });
      on(canvas, 'keyup', (e) => push(EV.KEY_UP, id, keyCode(e), modsOf(e)));
      on(canvas, 'focus', () => push(EV.FOCUS, id, 1));
      on(canvas, 'blur', () => push(EV.FOCUS, id, 0));
      if (globalThis.ResizeObserver) new ResizeObserver(() => fit(w)).observe(canvas);
      canvas.focus();
      return id;
    },

    um_host_window_close(id, line) {
      const w = lookup(id, line);
      if (w.canvas !== options.canvas) w.canvas.remove();
      wins[id - 1] = null;
    },

    um_host_window_poll(ptr, len, timeout, line) {
      if ((len >>> 0) < 6) throw new Panic(PANIC_WINDOW_ARGS, line);
      const take = () => {
        if (queue.length === 0) return 0;
        new Int32Array(mem(), ptr >>> 0, 6).set(queue.shift());
        return 1;
      };
      if (queue.length > 0 || timeout === 0) return take();
      // 来るまで（または timeout ミリ秒まで）待つ。JSPI で wasm を一時停止する。
      return new Promise((resolve) => {
        let timer = null;
        waiter = () => {
          if (timer !== null) clearTimeout(timer);
          resolve(take());
        };
        if (timeout > 0) {
          timer = setTimeout(() => {
            waiter = null;
            resolve(take());
          }, timeout);
        }
      });
    },

    um_host_window_present(id, ptr, len, width, height, x, y, rw, rh, show, line) {
      const w = lookup(id, line);
      if (width <= 0 || height <= 0 || width * height > (len >>> 0)) throw new Panic(PANIC_WINDOW_ARGS, line);
      if (w.gpu) {
        console.error('windowPresent: this window is drawn by the GPU (gpu.beginFrame); keep using gpu.beginFrame / gpu.endFrame for it');
        throw new Panic(PANIC_WINDOW_ARGS, line);
      }
      if (!w.ctx2d) w.ctx2d = w.canvas.getContext('2d');
      // 画素（0xAARRGGBB の u32）を ImageData（R, G, B, A のバイト）にする。大きさが変わったら全体を写す。
      if (!w.image || w.image.width !== width || w.image.height !== height) {
        w.image = new ImageData(width, height);
        x = 0;
        y = 0;
        rw = width;
        rh = height;
      }
      const src = new Uint32Array(mem(), ptr >>> 0, width * height);
      const dst = w.image.data;
      for (let j = y; j < y + rh; j++) {
        for (let i = x; i < x + rw; i++) {
          const v = src[j * width + i];
          const o = (j * width + i) * 4;
          dst[o] = (v >>> 16) & 255;
          dst[o + 1] = (v >>> 8) & 255;
          dst[o + 2] = v & 255;
          dst[o + 3] = v >>> 24;
        }
      }
      if (!show) return undefined;
      // canvas 全体に引き伸ばす（ネイティブと同じ）。
      if (width === w.canvas.width && height === w.canvas.height) {
        w.ctx2d.putImageData(w.image, 0, 0);
      } else {
        if (!w.scratch || w.scratch.width !== width || w.scratch.height !== height) {
          w.scratch = doc.createElement('canvas');
          w.scratch.width = width;
          w.scratch.height = height;
        }
        w.scratch.getContext('2d').putImageData(w.image, 0, 0);
        w.ctx2d.imageSmoothingEnabled = false;
        w.ctx2d.drawImage(w.scratch, 0, 0, w.canvas.width, w.canvas.height);
      }
      return nextFrame();
    },

    um_host_window_ctl(op, id, ioPtr, ioLen, line) {
      const io = new BigInt64Array(mem(), ioPtr >>> 0, Math.min(ioLen >>> 0, 6));
      const win = () => lookup(id, line);
      switch (op) {
        case WC.GET_SIZE: {
          const w = win();
          io[0] = BigInt(w.width);
          io[1] = BigInt(w.height);
          return 1n;
        }
        case WC.PIXEL_SIZE: {
          const w = win();
          io[0] = BigInt(w.canvas.width);
          io[1] = BigInt(w.canvas.height);
          return 1n;
        }
        case WC.SET_SIZE: {
          const w = win();
          w.canvas.style.width = `${Number(io[0])}px`;
          w.canvas.style.height = `${Number(io[1])}px`;
          fit(w);
          return 1n;
        }
        case WC.GET_POS:
          io[0] = 0n;
          io[1] = 0n;
          return 1n;
        case WC.STATE:
          return BigInt((doc?.fullscreenElement === win().canvas ? 1 : 0) | (doc?.activeElement === win().canvas ? 8 : 0));
        case WC.SET_FULLSCREEN: {
          const w = win();
          if (io[0] !== 0n) w.canvas.requestFullscreen?.().catch(() => {});
          else if (doc?.fullscreenElement) doc.exitFullscreen?.().catch(() => {});
          return 1n;
        }
        case WC.DISPLAY:
          return 0n;
        case WC.DISPLAY_COUNT:
          return 1n;
        case WC.DISPLAY_BOUNDS:
        case WC.DISPLAY_USABLE: {
          if (io[0] !== 0n) return 0n;
          const s = globalThis.screen;
          io[0] = 0n;
          io[1] = 0n;
          io[2] = BigInt(op === WC.DISPLAY_USABLE ? s?.availWidth ?? 0 : s?.width ?? 0);
          io[3] = BigInt(op === WC.DISPLAY_USABLE ? s?.availHeight ?? 0 : s?.height ?? 0);
          return 1n;
        }
        case WC.START_TEXT:
          textActive = true;
          return 1n;
        case WC.STOP_TEXT:
          textActive = false;
          return 1n;
        case WC.TEXT_ACTIVE:
          return textActive ? 1n : 0n;
        case WC.CURSOR:
          for (const w of wins) if (w) w.canvas.style.cursor = CURSORS[Number(io[0])] ?? 'default';
          return 1n;
        case WC.SHOW_CURSOR:
          for (const w of wins) if (w) w.canvas.style.cursor = io[0] !== 0n ? '' : 'none';
          return 1n;
        case WC.TICKS_NS:
          return BigInt(Math.round(globalThis.performance.now() * 1e6));
        case WC.SLEEP_NS:
          return new Promise((resolve) => setTimeout(() => resolve(1n), Number(io[0]) / 1e6));
        default:
          // ブラウザにないもの（OS のダイアログ・メニューバー・音・ウィンドウの位置など）は失敗（0）。
          return 0n;
      }
    },

    um_host_window_ctl_f(op, id, _a, _line) {
      switch (op) {
        case 1:  // 表示の倍率
        case 2:  // 画素の密度
        case 3:
          return dpr();
        case 4:  // 画面の更新の頻度（ブラウザは教えないので 0 = 不明）
          return 0;
        case 5:
          return globalThis.performance.now() / 1000;
        default:
          return 0;
      }
    },

    um_host_window_set_text(op, id, _a, ptr, len, line) {
      const text = new TextDecoder().decode(new Uint8Array(mem(), ptr >>> 0, len >>> 0));
      switch (op) {
        case 1: {  // 題名
          const w = lookup(id, line);
          w.title = text;
          if (doc && wins.find((x) => x) === w) doc.title = text;
          return 1n;
        }
        case 2:  // クリップボード（書くのは非同期。終わりを待たない）
          globalThis.navigator?.clipboard?.writeText(text).catch(() => {});
          return 1n;
        default:
          return 0n;
      }
    },

    um_host_window_get_text(op, id, _a, line) {
      switch (op) {
        case 4:
          return prepare(lookup(id, line).title);
        case 5:
          return prepare(lastError);
        case 6:
          return prepare('web');
        default:
          return prepare('');
      }
    },
  };
  return { env, windows: wins, nextFrame };
}

// ---- WebGPU での描画（gpu.beginImage / gpu.draw / gpu.beginFrame など。runtime/wgpu.c と同じ口）----

export class RenderRunner {
  constructor(device, memory, windows) {
    this.device = device;
    this.memory = memory;
    this.windows = windows;
    this.pipelines = new Map();
    this.images = [];  // 番号 n は images[n - 1]
  }

  bytes() {
    return new Uint8Array(this.memory.buffer);
  }

  slot(im) {
    let i = this.images.indexOf(null);
    if (i < 0) i = this.images.push(null) - 1;
    this.images[i] = im;
    return i + 1;
  }

  image(handle, what) {
    const im = this.images[handle - 1];
    if (!im) throw new Error(`UnifiedMozaic: ${what}: ${handle} is not an image or a frame`);
    return im;
  }

  depth(w, h) {
    return this.device.createTexture({ size: [w, h], format: 'depth32float', usage: GPUTextureUsage.RENDER_ATTACHMENT });
  }

  imageBegin(width, height, clear) {
    if (width <= 0 || height <= 0) return 0;
    const color = this.device.createTexture({ size: [width, height], format: 'bgra8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    return this.slot({ width, height, clear: clear >>> 0, format: 'bgra8unorm', color, view: color.createView(), depth: this.depth(width, height), drawn: false, window: 0 });
  }

  frameBegin(window, width, height, clear) {
    const w = this.windows[window - 1];
    if (!w || width <= 0 || height <= 0) return 0;
    if (!w.gpu) {
      if (w.ctx2d) {
        console.error('gpu.beginFrame: this window already shows pixels with windowPresent; a canvas cannot switch to the GPU');
        return 0;
      }
      const ctx = w.canvas.getContext('webgpu');
      if (!ctx) return 0;
      const format = globalThis.navigator.gpu.getPreferredCanvasFormat();
      w.gpu = { ctx, format, width: 0, height: 0, depth: null };
    }
    const g = w.gpu;
    if (w.canvas.width !== width || w.canvas.height !== height) {
      w.canvas.width = width;
      w.canvas.height = height;
    }
    if (g.width !== width || g.height !== height) {
      g.ctx.configure({ device: this.device, format: g.format, alphaMode: 'opaque' });
      g.depth?.destroy();
      g.depth = this.depth(width, height);
      g.width = width;
      g.height = height;
    }
    const color = g.ctx.getCurrentTexture();
    return this.slot({ width, height, clear: clear >>> 0, format: g.format, color, view: color.createView(), depth: g.depth, drawn: false, window });
  }

  pipeline(key, srcPtr, srcLen, nbufs, format) {
    const id = `${key}:${nbufs}:${format}`;
    let p = this.pipelines.get(id);
    if (p) return p;
    const code = new TextDecoder().decode(this.bytes().slice(srcPtr, srcPtr + srcLen));
    const module = this.device.createShaderModule({ code });
    const entries = [];
    for (let i = 0; i <= nbufs; i++) entries.push({ binding: i, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } });
    const layout = this.device.createBindGroupLayout({ entries });
    const pipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      vertex: { module, entryPoint: 'um_vs' },
      fragment: { module, entryPoint: 'um_fs', targets: [{ format }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
    });
    p = { pipeline, layout };
    this.pipelines.set(id, p);
    return p;
  }

  pass(encoder, im) {
    const c = im.clear;
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: im.view,
        loadOp: im.drawn ? 'load' : 'clear',
        storeOp: 'store',
        clearValue: { r: ((c >>> 16) & 255) / 255, g: ((c >>> 8) & 255) / 255, b: (c & 255) / 255, a: (c >>> 24) / 255 },
      }],
      depthStencilAttachment: { view: im.depth.createView(), depthLoadOp: im.drawn ? 'load' : 'clear', depthStoreOp: 'store', depthClearValue: 1 },
    });
    im.drawn = true;
    return pass;
  }

  draw(handle, key, srcPtr, srcLen, count, paramsPtr, paramBytes, descPtr, nbufs) {
    const im = this.image(handle, 'gpu.draw');
    const device = this.device;
    const { pipeline, layout } = this.pipeline(key, srcPtr, srcLen, nbufs, im.format);
    const table = new Uint32Array(this.memory.buffer, descPtr, nbufs * 4);
    // 引数の塊: 頂点の数、0、各 Buffer の長さ、値の引数（dispatch と同じ並び）。
    const header = 2 + nbufs;
    const params = new Uint8Array(header * 4 + paramBytes);
    const words = new Uint32Array(params.buffer, 0, header);
    words[0] = count >>> 0;
    for (let i = 0; i < nbufs; i++) words[2 + i] = table[i * 4 + 1];
    params.set(this.bytes().subarray(paramsPtr, paramsPtr + paramBytes), header * 4);
    const made = [];
    const make = (size, usage) => {
      const b = device.createBuffer({ size: Math.max(4, Math.ceil(size / 4) * 4), usage });
      made.push(b);
      return b;
    };
    const pb = make(params.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(pb, 0, params);
    const entries = [{ binding: 0, resource: { buffer: pb } }];
    for (let i = 0; i < nbufs; i++) {
      const ptr = table[i * 4];
      const used = Math.ceil((table[i * 4 + 1] * table[i * 4 + 2]) / 4) * 4;
      const b = make(used, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
      if (used > 0) device.queue.writeBuffer(b, 0, this.bytes(), ptr, used);
      entries.push({ binding: i + 1, resource: { buffer: b } });
    }
    const encoder = device.createCommandEncoder();
    const pass = this.pass(encoder, im);
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, device.createBindGroup({ layout, entries }));
    if (count > 0) pass.draw(count >>> 0);
    pass.end();
    device.queue.submit([encoder.finish()]);
    for (const b of made) b.destroy();
  }

  // 描き終えた画像の色を pixels（0xAARRGGBB の u32）に読み戻して、片づける。
  async imageEnd(handle, ptr, len) {
    const im = this.image(handle, 'gpu.endImage');
    if (im.window) throw new Error('UnifiedMozaic: gpu.endImage: this is a frame of a window; end it with gpu.endFrame');
    if ((len >>> 0) < im.width * im.height) throw new Error(`UnifiedMozaic: gpu.endImage: the Buffer holds ${len} elements but the image has ${im.width} x ${im.height} pixels`);
    const device = this.device;
    const encoder = device.createCommandEncoder();
    if (!im.drawn) this.pass(encoder, im).end();
    const row = Math.ceil((im.width * 4) / 256) * 256;
    const staging = device.createBuffer({ size: row * im.height, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    encoder.copyTextureToBuffer({ texture: im.color }, { buffer: staging, bytesPerRow: row, rowsPerImage: im.height }, [im.width, im.height]);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const src = new Uint8Array(staging.getMappedRange());
    const dst = this.bytes();
    for (let y = 0; y < im.height; y++) dst.set(src.subarray(y * row, y * row + im.width * 4), (ptr >>> 0) + y * im.width * 4);
    staging.unmap();
    staging.destroy();
    im.color.destroy();
    im.depth.destroy();
    this.images[handle - 1] = null;
  }

  // フレームを表示する（描かなければ埋めるだけ）。ブラウザは次の描画の時機に画面へ出すので、そこまで待つ。
  frameEnd(handle, nextFrame) {
    const im = this.image(handle, 'gpu.endFrame');
    if (!im.window) throw new Error('UnifiedMozaic: gpu.endFrame: this is an image; end it with gpu.endImage');
    if (!im.drawn) {
      const encoder = this.device.createCommandEncoder();
      this.pass(encoder, im).end();
      this.device.queue.submit([encoder.finish()]);
    }
    this.images[handle - 1] = null;
    return nextFrame();
  }
}
