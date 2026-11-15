
#!/usr/bin/env python3
r"""
ko_dxt_png.py - Knight Online .dxt <-> .png converter (encrypted files too)
 
  DXT -> PNG   python ko_dxt_png.py box.dxt                 -> box.png
  PNG -> DXT   python ko_dxt_png.py box.png                 -> box_new.dxt
               (uses box.dxt in the same folder as template)
  PNG -> DXT   python ko_dxt_png.py edited.png box.dxt out.dxt
 
  Many files   python ko_dxt_png.py C:\textures\*.dxt      (all to PNG)
               python ko_dxt_png.py C:\textures              (folder = all .dxt)
               python ko_dxt_png.py C:\textures\*.png      (all back to _new.dxt)
 
You can also drag a .dxt or .png onto this script in Explorer.
 
The PNG -> DXT step needs the ORIGINAL .dxt as a template. It copies the
header (name, flags, size, format, mipmap) from it, so the new file matches
what the game expects.
 
Encryption (NTF version 7): RC4 with the N3Base CWinCrypt key (built in).
The engine restarts the keystream for every read: once per row for
uncompressed images, once per mip level for DXT data. Encrypted files are
decrypted on export and re-encrypted on import, so the result stays v7.
 
DXT1/3/5 files also carry a half-size uncompressed copy; it is rebuilt
from your PNG too.
 
--plain   write an UNENCRYPTED file instead (version byte 7 -> 3)
 
Needs: pip install pillow numpy
"""
import os
import struct
import sys
 
import numpy as np
from PIL import Image
 
RAW_BPP = {20: 3, 21: 4, 22: 4, 23: 2, 24: 2, 25: 2, 26: 2}
DXT = (b'DXT1', b'DXT3', b'DXT5')
COMPANION = {b'DXT1': 25, b'DXT3': 26, b'DXT5': 26}
 
 
# ---------------------------------------------------------------- header
 
def read_dxt(path):
    data = open(path, 'rb').read()
    namelen = struct.unpack_from('<i', data, 0)[0]
    off = 4 + namelen
    if data[off:off + 3] != b'NTF':
        raise ValueError(f"{path}: not a Knight Online texture (no NTF signature)")
    flags = data[off + 3]
    w, h = struct.unpack_from('<ii', data, off + 4)
    fourcc = data[off + 12:off + 16]
    mip = struct.unpack_from('<i', data, off + 16)[0]
    hdr_end = off + 20
    return {'header': data[:hdr_end], 'flags': flags, 'w': w, 'h': h, 'mip': mip,
            'fourcc': fourcc, 'payload': data[hdr_end:]}
 
 
def fmt_code(fourcc):
    return None if fourcc in DXT else struct.unpack('<i', fourcc)[0]
 
 
def check_format(fourcc):
    code = fmt_code(fourcc)
    if code is not None and code not in RAW_BPP:
        raise ValueError(f"Unsupported pixel format code {code}")
 
 
def row_layout(w, h, fourcc):
    """(bytes per row, number of rows) of one mip level. DXT rows are block rows."""
    if fourcc in DXT:
        bs = 8 if fourcc == b'DXT1' else 16
        return max(1, (w + 3) // 4) * bs, max(1, (h + 3) // 4)
    return w * RAW_BPP[fmt_code(fourcc)], h
 
 
def chain(w, h, has_mips):
    """Mip levels the game stores: halve until a side would drop below 4."""
    levels = [(w, h)]
    while has_mips and min(w, h) // 2 >= 4:
        w, h = w // 2, h // 2
        levels.append((w, h))
    return levels
 
 
def chain_size(levels, fourcc):
    return sum(row_layout(w, h, fourcc)[0] * row_layout(w, h, fourcc)[1] for w, h in levels)
 
 
def layout(info):
    """Segments [(fourcc, levels)] that make up the payload.
    DXT textures carry a second, uncompressed half-size copy after the DXT
    mip chain (A1R5G5B5 for DXT1, A4R4G4B4 for DXT3/5)."""
    fc, w, h, n = info['fourcc'], info['w'], info['h'], len(info['payload'])
    main = chain(w, h, info['mip'])
    segs, size = [(fc, main)], chain_size(main, fc)
    if fc in DXT and n > size:
        cfc = struct.pack('<i', COMPANION[fc])
        comp = chain(max(1, w // 2), max(1, h // 2), info['mip'])
        segs.append((cfc, comp))
        size += chain_size(comp, cfc)
    if size != n:
        raise ValueError(f"Unexpected data layout: payload is {n} bytes, expected {size}")
    return segs
 
 
# ---------------------------------------------------------------- encryption
# One global key, the same in every file seen so far. Each row of each mip
# level of an uncompressed image is XOR'd with key[:row length].
# The key for the DXT-compressed part is NOT known yet.
 
GLOBAL_KEY = np.frombuffer(bytes.fromhex(
    "904d4f1f13a9e5f3f8de11d1f2981a33cade2324524f0325d68c25b3719f0a2c6be8d3af79a4ae564dce47893b9f7cc6d09ab82093c894f7d2cf192378316304280286e2b6256bd5307713e9f37d3436efb5d7f782daec0959b82ad57fe32e789ba88a9e79421a3fc67c4bb615626a31e3fcff12ef49f9ddf6912149d86d868d295abdef84f9fe07d5cc4a0e3d89d44bd5d3133806f62e3e46854110ef66e29379fd4dd89a348c0f11cf5f3b23586b3c51162a33b95842b9146c4944175d31a51da200c386352713f52a4cf864b9612e377165935e85e55ca191ae77f718be235b0b7149d1dfb76d9a294d80200fb3274f523695c419a285bbc085b93d0c8633"), np.uint8)
 
 
# RC4 key: SHA-1 of the N3Base CWinCrypt password, first 16 bytes
# (CryptoAPI MS_ENHANCED_PROV, CryptDeriveKey CALG_RC4 128-bit).
# Verified: its keystream matches the 256 bytes recovered from secret_box.dxt.
_CIPHER = b"owsd9012%$1as!wpow1033b%!@%12"
_KEYSTREAM = np.zeros(0, np.uint8)
 
 
def keystream(length):
    """First `length` bytes of the RC4 keystream (cached, grows on demand)."""
    global _KEYSTREAM
    if len(_KEYSTREAM) >= length:
        return _KEYSTREAM
    import hashlib
    key = hashlib.sha1(_CIPHER).digest()[:16]
    S = list(range(256))
    j = 0
    for i in range(256):
        j = (j + S[i] + key[i % 16]) & 255
        S[i], S[j] = S[j], S[i]
    n = max(length, 65536)
    out = bytearray(n)
    i = j = 0
    for k in range(n):
        i = (i + 1) & 255
        j = (j + S[i]) & 255
        S[i], S[j] = S[j], S[i]
        out[k] = S[(S[i] + S[j]) & 255]
    _KEYSTREAM = np.frombuffer(bytes(out), np.uint8)
    return _KEYSTREAM
 
 
def key_for(length, info=None):
    """Keystream bytes for one decrypt region of `length` bytes."""
    return keystream(length)
 
 
def has_key(length):
    return True
 
 
def region_sizes(levels, fourcc):
    """Sizes of the regions the engine decrypts separately: one region per
    DXT mip level, one region per row for uncompressed formats."""
    out = []
    for w, h in levels:
        rl, rows = row_layout(w, h, fourcc)
        out += [rl * rows] if fourcc in DXT else [rl] * rows
    return out
 
 
def xor_segment(payload, off, levels, fourcc, info=None):
    out = bytearray(payload)
    for n in region_sizes(levels, fourcc):
        key = key_for(n, info)
        blk = np.frombuffer(payload, np.uint8, n, off)
        out[off:off + n] = (blk ^ key[:n]).tobytes()
        off += n
    return bytes(out)
 
 
# ---------------------------------------------------------------- raw pixel formats
 
def decode_raw(pix, w, h, code):
    n = w * h
    if code in (21, 22):
        a = np.frombuffer(pix, np.uint8, n * 4).reshape(h, w, 4)
        out = a[..., [2, 1, 0, 3]].copy()
        if code == 22:
            out[..., 3] = 255
        return out
    if code == 20:
        a = np.frombuffer(pix, np.uint8, n * 3).reshape(h, w, 3)
        return np.dstack([a[..., [2, 1, 0]], np.full((h, w), 255, np.uint8)])
    v = np.frombuffer(pix, '<u2', n).reshape(h, w).astype(np.uint32)
    if code == 23:
        r, g, b = (v >> 11) & 31, (v >> 5) & 63, v & 31
        rgba = [(r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2), np.full_like(v, 255)]
    elif code in (24, 25):
        r, g, b = (v >> 10) & 31, (v >> 5) & 31, v & 31
        a = np.full_like(v, 255) if code == 24 else ((v >> 15) & 1) * 255
        rgba = [(r << 3) | (r >> 2), (g << 3) | (g >> 2), (b << 3) | (b >> 2), a]
    else:  # 26 A4R4G4B4
        rgba = [((v >> s) & 15) * 17 for s in (8, 4, 0, 12)]
    return np.dstack(rgba).astype(np.uint8)
 
 
def encode_raw(arr, code):
    a = arr.astype(np.uint32)
    r, g, b, al = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    if code in (21, 22):
        out = np.dstack([b, g, r, al if code == 21 else np.full_like(al, 255)])
        return out.astype(np.uint8).tobytes()
    if code == 20:
        return np.dstack([b, g, r]).astype(np.uint8).tobytes()
    if code == 23:
        v = ((r * 31 + 127) // 255 << 11) | ((g * 63 + 127) // 255 << 5) | ((b * 31 + 127) // 255)
    elif code in (24, 25):
        v = ((r * 31 + 127) // 255 << 10) | ((g * 31 + 127) // 255 << 5) | ((b * 31 + 127) // 255)
        if code == 25:
            v |= (al >= 128).astype(np.uint32) << 15
        else:
            v |= 1 << 15
    else:  # 26 A4R4G4B4
        q = lambda c: (c * 15 + 127) // 255
        v = (q(al) << 12) | (q(r) << 8) | (q(g) << 4) | q(b)
    return v.astype('<u2').tobytes()
 
 
# ---------------------------------------------------------------- DXT1/3/5
 
def _565(c):
    c = np.clip(np.round(c), 0, 255).astype(np.int64)
    return int(((c[0] >> 3) << 11) | ((c[1] >> 2) << 5) | (c[2] >> 3))
 
 
def _un565(v):
    r, g, b = (v >> 11) & 31, (v >> 5) & 63, v & 31
    return np.array([(r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)], float)
 
 
def _color_block(rgb, alpha=None, dxt1_alpha=False):
    px = rgb.reshape(16, 3).astype(float)
    trans = (alpha.reshape(16) < 128) if (dxt1_alpha and alpha is not None) else np.zeros(16, bool)
    use = px[~trans] if (~trans).any() else px
    lum = use @ [0.299, 0.587, 0.114]
    c0, c1 = _565(use[lum.argmax()]), _565(use[lum.argmin()])
    if trans.any():  # 3-colour mode: c0 <= c1, index 3 = transparent
        if c0 > c1:
            c0, c1 = c1, c0
        e0, e1 = _un565(c0), _un565(c1)
        pal = np.array([e0, e1, (e0 + e1) / 2])
    else:
        if c0 == c1:
            c1 = c0 - 1 if c0 > 0 else 1
        if c0 < c1:
            c0, c1 = c1, c0
        e0, e1 = _un565(c0), _un565(c1)
        pal = np.array([e0, e1, (2 * e0 + e1) / 3, (e0 + 2 * e1) / 3])
    idx = ((px[:, None] - pal[None]) ** 2).sum(2).argmin(1)
    idx[trans] = 3
    bits = 0
    for i, v in enumerate(idx):
        bits |= int(v) << (2 * i)
    return struct.pack('<HHI', c0, c1, bits)
 
 
def _bc3_alpha(alpha):
    a = alpha.reshape(16).astype(int)
    a0, a1 = int(a.max()), int(a.min())
    if a0 == a1:
        return struct.pack('<BB', a0, a1) + b'\0' * 6
    pal = [a0, a1] + [((7 - i) * a0 + i * a1) // 7 for i in range(1, 7)]
    idx = np.abs(a[:, None] - np.array(pal)[None]).argmin(1)
    bits = 0
    for i, v in enumerate(idx):
        bits |= int(v) << (3 * i)
    return struct.pack('<BB', a0, a1) + bits.to_bytes(6, 'little')
 
 
def _bc2_alpha(alpha):
    a = np.clip((alpha.reshape(16).astype(int) + 8) // 17, 0, 15)
    return sum(int(v) << (4 * i) for i, v in enumerate(a)).to_bytes(8, 'little')
 
 
def encode_dxt(arr, fourcc):
    h, w = arr.shape[:2]
    ph, pw = (-h) % 4, (-w) % 4
    if ph or pw:
        arr = np.pad(arr, ((0, ph), (0, pw), (0, 0)), mode='edge')
    out = bytearray()
    for by in range(0, arr.shape[0], 4):
        for bx in range(0, arr.shape[1], 4):
            blk = arr[by:by + 4, bx:bx + 4]
            if fourcc == b'DXT1':
                out += _color_block(blk[..., :3], blk[..., 3], dxt1_alpha=True)
            else:
                out += (_bc2_alpha if fourcc == b'DXT3' else _bc3_alpha)(blk[..., 3])
                out += _color_block(blk[..., :3])
    return bytes(out)
 
 
def decode_dxt(pix, w, h, fourcc):
    bw, bh = max(1, (w + 3) // 4), max(1, (h + 3) // 4)
    img = np.zeros((bh * 4, bw * 4, 4), np.uint8)
    bs = 8 if fourcc == b'DXT1' else 16
    off = 0
    for by in range(bh):
        for bx in range(bw):
            blk = pix[off:off + bs]
            off += bs
            cb = blk if bs == 8 else blk[8:]
            c0, c1, bits = struct.unpack('<HHI', cb)
            e0, e1 = _un565(c0), _un565(c1)
            if c0 > c1 or bs == 16:
                pal = [e0, e1, (2 * e0 + e1) / 3, (e0 + 2 * e1) / 3]
                pa = [255] * 4
            else:
                pal = [e0, e1, (e0 + e1) / 2, np.zeros(3)]
                pa = [255, 255, 255, 0]
            if fourcc == b'DXT3':
                av = int.from_bytes(blk[:8], 'little')
                alpha = [((av >> (4 * i)) & 15) * 17 for i in range(16)]
            elif fourcc == b'DXT5':
                a0, a1 = blk[0], blk[1]
                ab = int.from_bytes(blk[2:8], 'little')
                if a0 > a1:
                    ap = [a0, a1] + [((7 - i) * a0 + i * a1) // 7 for i in range(1, 7)]
                else:
                    ap = [a0, a1] + [((5 - i) * a0 + i * a1) // 5 for i in range(1, 5)] + [0, 255]
                alpha = [ap[(ab >> (3 * i)) & 7] for i in range(16)]
            else:
                alpha = None
            for i in range(16):
                s = (bits >> (2 * i)) & 3
                y, x = by * 4 + i // 4, bx * 4 + i % 4
                img[y, x, :3] = np.clip(pal[s], 0, 255)
                img[y, x, 3] = alpha[i] if alpha is not None else pa[s]
    return img[:h, :w]
 
 
# ---------------------------------------------------------------- top level
 
def decode_any(pix, w, h, fourcc):
    if fourcc in DXT:
        return decode_dxt(pix, w, h, fourcc)
    return decode_raw(pix, w, h, fmt_code(fourcc))
 
 
def encode_any(arr, fourcc):
    return encode_dxt(arr, fourcc) if fourcc in DXT else encode_raw(arr, fmt_code(fourcc))
 
 
def dxt_to_png(dxt_path, png_path):
    info = read_dxt(dxt_path)
    check_format(info['fourcc'])
    segs = layout(info)
    enc = info['flags'] & 0x04
    note = ''
    fc, levels = segs[0]
    off = 0
    if enc and fc in DXT and not has_key(max(region_sizes(levels[:1], fc))):
        if len(segs) < 2:
            raise ValueError("Encrypted DXT texture could not be decrypted")
        off = chain_size(segs[0][1], fc)
        fc, levels = segs[1]
        note = (f"exported the built-in {levels[0][0]}x{levels[0][1]} uncompressed copy "
                "(full-size DXT could not be decrypted)")
    payload = info['payload']
    if enc:
        payload = xor_segment(payload, off, levels[:1], fc, info)
    w, h = levels[0]
    rl, rows = row_layout(w, h, fc)
    arr = decode_any(payload[off:off + rl * rows], w, h, fc)
    Image.fromarray(arr, 'RGBA').save(png_path)
    return info, segs, note
 
 
PLAIN = False  # set by --plain: write encrypted templates as unencrypted files
 
 
def png_to_dxt(png_path, template_path, out_path):
    info = read_dxt(template_path)
    check_format(info['fourcc'])
    segs = layout(info)
    enc = info['flags'] & 0x04
    header = info['header']
    plain_note = ''
    if enc:
        need = max(max(region_sizes(lv, fc)) for fc, lv in segs)
        if PLAIN:
            # Clear the encryption bit in the NTF version byte (7 -> 3).
            pos = len(header) - 17
            header = header[:pos] + bytes([header[pos] & ~0x04 & 0xFF]) + header[pos + 1:]
            enc = 0
            info = dict(info, flags=info['flags'] & ~0x04)
            plain_note = "written UNENCRYPTED (--plain)"
    im = Image.open(png_path).convert('RGBA')
    note = ''
    if im.size != (info['w'], info['h']):
        note = f"PNG was {im.size[0]}x{im.size[1]}, resized to {info['w']}x{info['h']}"
        im = im.resize((info['w'], info['h']), Image.LANCZOS)
    payload, offsets = bytearray(), []
    for fc, levels in segs:
        offsets.append(len(payload))
        for w, h in levels:
            arr = np.array(im if (w, h) == im.size else im.resize((w, h), Image.BOX))
            payload += encode_any(arr, fc)
    payload = bytes(payload)
    if enc:
        for (fc, levels), off in zip(segs, offsets):
            payload = xor_segment(payload, off, levels, fc, info)
    if len(payload) != len(info['payload']):
        raise RuntimeError(f"Size mismatch: {len(payload)} vs template {len(info['payload'])}")
    with open(out_path, 'wb') as f:
        f.write(header + payload)
    note = '; '.join(x for x in (note, plain_note) if x)
    return info, segs, note
 
 
def _describe(info, segs, note=''):
    name = lambda fc: fc.decode() if fc in DXT else {20: 'R8G8B8', 21: 'A8R8G8B8', 22: 'X8R8G8B8',
        23: 'R5G6B5', 24: 'X1R5G5B5', 25: 'A1R5G5B5', 26: 'A4R4G4B4'}.get(fmt_code(fc), fmt_code(fc))
    parts = [f"{name(fc)} x{len(lv)} mips" for fc, lv in segs]
    enc = 'encrypted' if info['flags'] & 0x04 else 'not encrypted'
    s = f"{info['w']}x{info['h']}, {' + '.join(parts)}, {enc}"
    return s + (f"\n       note: {note}" if note else '')
 
 
def _find_template(png_path):
    base = os.path.splitext(png_path)[0]
    folder, stem = os.path.split(base)
    for f in os.listdir(folder or '.'):
        if os.path.splitext(f)[0].lower() == stem.lower() and f.lower().endswith('.dxt'):
            return os.path.join(folder, f)
    return None
 
 
def _expand(argv):
    """Windows cmd/PowerShell don't expand *.dxt, so do it here. A folder
    argument means every .dxt in it."""
    import glob
    files = []
    for a in argv:
        if os.path.isdir(a):
            files += sorted(os.path.join(a, f) for f in os.listdir(a) if f.lower().endswith('.dxt'))
        elif any(c in a for c in '*?['):
            files += sorted(glob.glob(a))
        else:
            files.append(a)
    return files
 
 
def _batch(files):
    ok = failed = 0
    for src in files:
        try:
            if src.lower().endswith('.dxt'):
                out = os.path.splitext(src)[0] + '.png'
                info, levels, note = dxt_to_png(src, out)
                print(f"OK   {os.path.basename(src)} -> {os.path.basename(out)}  ({_describe(info, levels, note)})")
            else:
                tpl = _find_template(src)
                if not tpl:
                    raise ValueError("no .dxt with the same name next to it")
                out = os.path.splitext(tpl)[0] + '_new.dxt'
                info, levels, note = png_to_dxt(src, tpl, out)
                print(f"OK   {os.path.basename(src)} -> {os.path.basename(out)}  ({_describe(info, levels, note)})")
            ok += 1
        except Exception as e:
            print(f"FAIL {os.path.basename(src)}: {e}")
            failed += 1
    print(f"\nDone: {ok} converted, {failed} failed.")
 
 
def main(argv):
    global PLAIN
    if '--plain' in argv:
        PLAIN = True
        argv = [a for a in argv if a != '--plain']
    if not argv:
        print(__doc__)
        return
    argv = _expand(argv)
    if not argv:
        print("No matching files found.")
        return
    if all(a.lower().endswith('.dxt') for a in argv) or (
            len(argv) > 1 and all(a.lower().endswith('.png') for a in argv)
            and not argv[1].lower().endswith('.dxt')):
        _batch(argv)
    elif argv and argv[0].lower().endswith('.png') and len(argv) in (1, 2, 3):
        src = argv[0]
        tpl = argv[1] if len(argv) > 1 else _find_template(src)
        if not tpl:
            raise SystemExit("No template .dxt found. Usage: ko_dxt_png.py edited.png original.dxt [out.dxt]")
        out = argv[2] if len(argv) > 2 else os.path.splitext(tpl)[0] + '_new.dxt'
        if os.path.abspath(out) == os.path.abspath(tpl):
            raise SystemExit("Refusing to overwrite the template; give a different output name.")
        info, levels, note = png_to_dxt(src, tpl, out)
        print(f"PNG -> DXT: {out}  (template: {tpl})\n  {_describe(info, levels, note)}")
    else:
        print(__doc__)
 
 
if __name__ == '__main__':
    try:
        main(sys.argv[1:])
    except Exception as e:
        print(f"ERROR: {e}")
    if len(sys.argv) == 2 and sys.stdin.isatty():  # drag-and-drop: keep the window open
        try:
            input("\nPress Enter to close...")
        except EOFError:
            pass
 
