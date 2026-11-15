
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
what the game expects. For encrypted files it also takes the XOR key from it.
 
Encryption (flag bit 0x04): every row of every mip level is XOR'd with the
first <row length> bytes of one key. The key is recovered from a fully
transparent row of the original texture, so the original must have at least
two transparent rows. Verified for raw formats (A4R4G4B4 etc.); encrypted
DXT1/3/5 files are refused with an error.
 
Needs: pip install pillow numpy
"""
import os
import struct
import sys
 
import numpy as np
from PIL import Image
 
RAW_BPP = {20: 3, 21: 4, 22: 4, 23: 2, 24: 2, 25: 2, 26: 2}
DXT = (b'DXT1', b'DXT3', b'DXT5')
 
 
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
    hdr_end = off + 20
    return {'header': data[:hdr_end], 'flags': flags, 'w': w, 'h': h,
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
 
 
def mip_levels(w, h, fourcc, payload_len):
    """List of (w, h) levels that exactly fill the payload."""
    levels, used = [], 0
    while True:
        rl, rows = row_layout(w, h, fourcc)
        if used + rl * rows > payload_len:
            break
        levels.append((w, h))
        used += rl * rows
        if w == 1 and h == 1:
            break
        w, h = max(1, w // 2), max(1, h // 2)
    if not levels:
        raise ValueError("Payload is smaller than the first mip level")
    return levels
 
 
# ---------------------------------------------------------------- encryption
 
def recover_key(info):
    if info['fourcc'] in DXT:
        # A transparent DXT block is not all zero bytes, and the exact bytes the
        # game's tool wrote are unknown, so the key cannot be recovered safely.
        raise ValueError("Encrypted DXT1/3/5 textures are not supported yet "
                         "(only encrypted raw formats like A4R4G4B4 were verified)")
    rl, rows = row_layout(info['w'], info['h'], info['fourcc'])
    lvl = np.frombuffer(info['payload'][:rl * rows], np.uint8).reshape(rows, rl)
    _, idx, counts = np.unique(lvl, axis=0, return_index=True, return_counts=True)
    if counts.max() < 2:
        raise ValueError("Encrypted texture has no fully transparent row; cannot recover the key")
    return lvl[idx[counts.argmax()]].copy()
 
 
def xor_levels(payload, levels, fourcc, key):
    out, off = bytearray(payload), 0
    for w, h in levels:
        rl, rows = row_layout(w, h, fourcc)
        blk = np.frombuffer(payload, np.uint8, rl * rows, off).reshape(rows, rl)
        out[off:off + rl * rows] = (blk ^ key[:rl]).tobytes()
        off += rl * rows
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
 
def dxt_to_png(dxt_path, png_path):
    info = read_dxt(dxt_path)
    check_format(info['fourcc'])
    levels = mip_levels(info['w'], info['h'], info['fourcc'], len(info['payload']))
    payload = info['payload']
    if info['flags'] & 0x04:
        payload = xor_levels(payload, levels, info['fourcc'], recover_key(info))
    w, h = info['w'], info['h']
    if info['fourcc'] in DXT:
        arr = decode_dxt(payload, w, h, info['fourcc'])
    else:
        arr = decode_raw(payload, w, h, fmt_code(info['fourcc']))
    Image.fromarray(arr, 'RGBA').save(png_path)
    return info, levels
 
 
def png_to_dxt(png_path, template_path, out_path):
    info = read_dxt(template_path)
    check_format(info['fourcc'])
    fourcc = info['fourcc']
    levels = mip_levels(info['w'], info['h'], fourcc, len(info['payload']))
    im = Image.open(png_path).convert('RGBA')
    if im.size != (info['w'], info['h']):
        print(f"  note: PNG is {im.size[0]}x{im.size[1]}, resized to {info['w']}x{info['h']} to match the template")
        im = im.resize((info['w'], info['h']), Image.LANCZOS)
    payload = bytearray()
    for w, h in levels:
        arr = np.array(im if (w, h) == im.size else im.resize((w, h), Image.BOX))
        payload += encode_dxt(arr, fourcc) if fourcc in DXT else encode_raw(arr, fmt_code(fourcc))
    payload = bytes(payload)
    if info['flags'] & 0x04:
        payload = xor_levels(payload, levels, fourcc, recover_key(info))
    if len(payload) != len(info['payload']):
        raise RuntimeError(f"Size mismatch: {len(payload)} vs template {len(info['payload'])}")
    with open(out_path, 'wb') as f:
        f.write(info['header'] + payload)
    return info, levels
 
 
def _describe(info, levels):
    fmt = info['fourcc'].decode() if info['fourcc'] in DXT else f"raw code {fmt_code(info['fourcc'])}"
    enc = 'encrypted' if info['flags'] & 0x04 else 'not encrypted'
    return f"{info['w']}x{info['h']}, {fmt}, {len(levels)} mip level(s), {enc}"
 
 
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
                info, levels = dxt_to_png(src, out)
                print(f"OK   {os.path.basename(src)} -> {os.path.basename(out)}  ({_describe(info, levels)})")
            else:
                tpl = _find_template(src)
                if not tpl:
                    raise ValueError("no .dxt with the same name next to it")
                out = os.path.splitext(tpl)[0] + '_new.dxt'
                info, levels = png_to_dxt(src, tpl, out)
                print(f"OK   {os.path.basename(src)} -> {os.path.basename(out)}  ({_describe(info, levels)})")
            ok += 1
        except Exception as e:
            print(f"FAIL {os.path.basename(src)}: {e}")
            failed += 1
    print(f"\nDone: {ok} converted, {failed} failed.")
 
 
def main(argv):
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
        info, levels = png_to_dxt(src, tpl, out)
        print(f"PNG -> DXT: {out}  (template: {tpl})\n  {_describe(info, levels)}")
    else:
        print(__doc__)
 
 
if __name__ == '__main__':
    try:
        main(sys.argv[1:])
    except Exception as e:
        print(f"ERROR: {e}")
    if len(sys.argv) == 2:  # drag-and-drop: keep the window open
        input("\nPress Enter to close...")
 
