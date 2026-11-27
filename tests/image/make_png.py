#!/usr/bin/env python3
# tests/image/png.um を作る（PNG のいろいろな形式を作り、バイト列と、画素の正しい値のハッシュを埋め込む）。
# python3 tests/image/make_png.py > tests/image/png.um
import struct, zlib, random

def chunk(t, d):
    return struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)

ADAM7 = [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)]

def paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc: return a
    if pb <= pc: return b
    return c

def filt(rows, bpp, rnd):
    out = b''
    prev = bytes(len(rows[0])) if rows else b''
    for row in rows:
        f = rnd.randrange(5)
        o = bytearray()
        for i, v in enumerate(row):
            a = row[i - bpp] if i >= bpp else 0
            b = prev[i]
            c = prev[i - bpp] if i >= bpp else 0
            pred = [0, a, b, (a + b) // 2, paeth(a, b, c)][f]
            o.append((v - pred) & 255)
        out += bytes([f]) + bytes(o)
        prev = row
    return out

def pack(samples, depth):
    if depth == 8: return bytes(samples)
    if depth == 16: return b''.join(struct.pack('>H', s) for s in samples)
    bits = ''.join(format(s, '0%db' % depth) for s in samples)
    bits += '0' * (-len(bits) % 8)
    return bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))

def make(w, h, ctype, depth, interlace, level, strategy, seed, trns=False):
    rnd = random.Random(seed)
    ch = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[ctype]
    maxv = (1 << depth) - 1
    npal = min(1 << depth, 13) if ctype == 3 else 0
    pal = [(rnd.randrange(256), rnd.randrange(256), rnd.randrange(256)) for _ in range(npal)]
    palA = [rnd.randrange(256) for _ in range(npal)]
    # 画素の標本（なめらかな部分とでたらめな部分を混ぜる）。
    px = {}
    for y in range(h):
        for x in range(w):
            if ctype == 3:
                px[x, y] = [rnd.randrange(npal) if (x + y) % 3 else (x * y) % npal]
            else:
                px[x, y] = [(rnd.randrange(maxv + 1) if (x ^ y) & 1 else ((x * 37 + y * 11 + c * 5) % (maxv + 1))) for c in range(ch)]
    key = None
    if trns and ctype in (0, 2):
        key = list(px[1, 0])
    bpp = max(1, ch * depth // 8)
    raw = b''
    passes = ADAM7 if interlace else [(0, 0, 1, 1)]
    for (x0, y0, dx, dy) in passes:
        xs = list(range(x0, w, dx)); ys = list(range(y0, h, dy))
        if not xs or not ys: continue
        rows = [pack([s for x in xs for s in px[x, y]], depth) for y in ys]
        raw += filt(rows, bpp, rnd)
    co = zlib.compressobj(level, zlib.DEFLATED, 15, 9, strategy)
    z = co.compress(raw) + co.flush()
    png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, depth, ctype, 0, 0, interlace))
    if ctype == 3:
        png += chunk(b'PLTE', bytes(v for c in pal for v in c))
        if trns: png += chunk(b'tRNS', bytes(palA))
    if key is not None:
        png += chunk(b'tRNS', b''.join(struct.pack('>H', v) for v in key))
    # IDAT を 2 つに分ける（つなげて読むことを確かめる）。
    half = len(z) // 2
    png += chunk(b'IDAT', z[:half]) + chunk(b'IDAT', z[half:]) + chunk(b'IEND', b'')
    def to8(v): return v >> 8 if depth == 16 else v if depth == 8 else v * 255 // maxv
    h32 = 2166136261
    for y in range(h):
        for x in range(w):
            s = px[x, y]
            if ctype == 3:
                r, g, b = pal[s[0]]; a = palA[s[0]] if trns else 255
            elif ctype == 0:
                r = g = b = to8(s[0]); a = 0 if key is not None and s == key else 255
            elif ctype == 4:
                r = g = b = to8(s[0]); a = to8(s[1])
            elif ctype == 2:
                r, g, b = map(to8, s); a = 0 if key is not None and s == key else 255
            else:
                r, g, b, a = map(to8, s)
            argb = (a << 24) | (r << 16) | (g << 8) | b
            h32 = ((h32 ^ argb) * 16777619) & 0xffffffff
    return png, h32

CASES = [
    ('rgba8', 13, 9, 6, 8, 0, 9, zlib.Z_DEFAULT_STRATEGY, False),
    ('rgb8-fixed', 7, 5, 2, 8, 0, 9, zlib.Z_FIXED, False),
    ('rgb8-stored', 5, 4, 2, 8, 0, 0, zlib.Z_DEFAULT_STRATEGY, False),
    ('rgb16-key', 6, 6, 2, 16, 0, 6, zlib.Z_DEFAULT_STRATEGY, True),
    ('gray1', 19, 3, 0, 1, 0, 9, zlib.Z_DEFAULT_STRATEGY, False),
    ('gray2', 9, 4, 0, 2, 0, 9, zlib.Z_DEFAULT_STRATEGY, False),
    ('gray4-key', 11, 5, 0, 4, 0, 9, zlib.Z_DEFAULT_STRATEGY, True),
    ('gray16', 4, 7, 0, 16, 0, 9, zlib.Z_DEFAULT_STRATEGY, False),
    ('graya8', 8, 8, 4, 8, 0, 9, zlib.Z_DEFAULT_STRATEGY, False),
    ('graya16', 5, 3, 4, 16, 0, 9, zlib.Z_DEFAULT_STRATEGY, False),
    ('pal1', 17, 2, 3, 1, 0, 9, zlib.Z_DEFAULT_STRATEGY, False),
    ('pal4-trns', 10, 6, 3, 4, 0, 9, zlib.Z_DEFAULT_STRATEGY, True),
    ('pal8-trns', 12, 12, 3, 8, 0, 9, zlib.Z_DEFAULT_STRATEGY, True),
    ('rgba8-adam7', 13, 11, 6, 8, 1, 9, zlib.Z_DEFAULT_STRATEGY, False),
    ('pal2-adam7', 3, 2, 3, 2, 1, 9, zlib.Z_DEFAULT_STRATEGY, False),
    ('rgba16-adam7', 9, 9, 6, 16, 1, 9, zlib.Z_DEFAULT_STRATEGY, False),
    ('rgb8-big', 32, 24, 2, 8, 0, 9, zlib.Z_DEFAULT_STRATEGY, False),
]

print('// PNG を読む（imageDecodePng。lib/png.um）。色の形式・ビットの深さ・インターレース・透明の色・圧縮の方式')
print('// （圧縮しない・決まった符号・ブロックの頭の符号）の組み合わせを、Python の zlib で作った PNG で確かめる。')
print('// 画素の正しい値は、同じ Python の中で作った値（のハッシュ）。tests/image/make_png.py が作ったファイル（手で直さない）。')
for i, c in enumerate(CASES):
    png, h = make(c[1], c[2], c[3], c[4], c[5], c[6], c[7], i, c[8])
    print(f'// expect: {c[0]} {c[1]} {c[2]} {h}')
print('// expect: broken broken PNG')
print('// expect: not png not a PNG file')
print()
print('import { imageDecodePng } from "um:png";')
print()
for i, c in enumerate(CASES):
    png, h = make(c[1], c[2], c[3], c[4], c[5], c[6], c[7], i, c[8])
    name = 'PNG_' + c[0].upper().replace('-', '_')
    print(f'const {name}: u8[{len(png)}] = [{", ".join(str(b) for b in png)}];')
print('''
function show(label: string, data: u8[]): void {
  const img = imageDecodePng(data);
  if (img.error != "") {
    print(`${label} ${img.error}`);
    return;
  }
  let h: u32 = 2166136261;
  for (let i: usize = 0; i < (img.pixels.length as usize); i++) {
    h = (h ^ img.pixels[i]) * 16777619;
  }
  print(`${label} ${img.width} ${img.height} ${h}`);
}

export function main(): i32 {''')
for c in CASES:
    name = 'PNG_' + c[0].upper().replace('-', '_')
    print(f'''  {{
    let d: u8[] = [];
    for (let i: usize = 0; i < {name}.length; i++) {{
      d.push({name}[i]);
    }}
    show("{c[0]}", d);
  }}''')
print('''  // 途中で切れた PNG と、PNG でないもの。
  let cut: u8[] = [];
  for (let i: usize = 0; i + 30 < PNG_RGBA8.length; i++) {
    cut.push(PNG_RGBA8[i]);
  }
  show("broken", cut);
  let text: u8[] = [72, 101, 108, 108, 111, 33, 33, 33, 33, 33];
  show("not png", text);
  return 0;
}''')
