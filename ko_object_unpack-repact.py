#!/usr/bin/env python3
r"""
ko_unpack.py - unpack and repack Knight Online .src archives

  UNPACK   python ko_object_unpack-repact.py object.src                 -> object_unpacked\
           python ko_object_unpack-repact.py object.src object.hdr out\ (explicit index / output)

  REPACK   python ko_object_unpack-repact.py --repack object_unpacked   -> object_new.src (+ object_new.hdr)

Unpacking writes _ko_index.txt into the output folder; repack reads it, so
keep that file. Edit files in the unpacked folder, or leave the converter's
"name_new.dxt" next to "name.dxt": repack uses the _new file automatically.
The original .src/.hdr are never overwritten. Rename the _new files when
you are happy with them (keep a backup of the originals).

With the .hdr index (same name, same folder as the .src) every file is
extracted with its real name and path, and repack rebuilds both files, so
edited files may change size and new files can be added.

Without a .hdr the script falls back to CARVE mode: it finds the textures
(.dxt) by their NTF headers and cuts them out using their exact size.
Repack then patches textures IN PLACE, so every offset stays the same and
the existing .hdr stays valid. That only works when the edited file has the
same size as the original, which ko_dxt_png.py guarantees (same template).

Needs: ko_dxt_png.py in the same folder (for carve mode), pip install numpy pillow
"""
import mmap
import os
import re
import struct
import sys


def find_hdr(src_path):
    folder, name = os.path.split(os.path.abspath(src_path))
    stem = os.path.splitext(name)[0].lower()
    for f in os.listdir(folder):
        if os.path.splitext(f)[0].lower() == stem and f.lower().endswith('.hdr'):
            return os.path.join(folder, f)
    return None


def safe_rel(name):
    """Archive name -> safe relative path (no drive letters, no '..')."""
    parts = [p for p in re.split(r'[\\/]+', name) if p not in ('', '.', '..')]
    parts = [re.sub(r'[<>:"|?*]', '_', p) for p in parts]
    return os.path.join(*parts) if parts else 'unnamed'


def unique(path, used):
    base, ext = os.path.splitext(path)
    n, out = 1, path
    while out.lower() in used:
        n += 1
        out = f"{base}_{n}{ext}"
    used.add(out.lower())
    return out


def read_hdr(hdr_path):
    """Same layout ko_vfs.cpp parses: u32 version, u32 timestamp, then
    entries of (u16 name length, name, u32 offset, u32 size)."""
    data = open(hdr_path, 'rb').read()
    entries, off = [], 8
    while off + 2 <= len(data):
        nlen = struct.unpack_from('<H', data, off)[0]
        off += 2
        if nlen == 0 or nlen > 512 or off + nlen + 8 > len(data):
            break
        name = data[off:off + nlen]
        off += nlen
        pos, size = struct.unpack_from('<II', data, off)
        off += 8
        if all(32 <= b < 127 for b in name):
            entries.append((name.decode('ascii'), pos, size))
    return entries


def unpack_with_hdr(src_path, hdr_path, out_dir):
    entries = read_hdr(hdr_path)
    if not entries:
        raise SystemExit(f"No entries found in {hdr_path} (unknown or encrypted index?)")
    src_len = os.path.getsize(src_path)
    used, ok, bad, manifest = set(), 0, 0, []
    with open(src_path, 'rb') as f:
        for name, pos, size in entries:
            rel = unique(safe_rel(name), used)
            manifest.append((rel, name, pos, size))
            if pos + size > src_len:
                print(f"SKIP {name}: points past the end of the .src")
                bad += 1
                continue
            f.seek(pos)
            out = os.path.join(out_dir, rel)
            os.makedirs(os.path.dirname(out) or out_dir, exist_ok=True)
            with open(out, 'wb') as o:
                o.write(f.read(size))
            ok += 1
    write_manifest(out_dir, 'hdr', src_path, hdr_path, manifest)
    print(f"Done: {ok} files extracted to {out_dir}" + (f", {bad} skipped" if bad else ""))


def carve_textures(src_path, out_dir):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        import ko_dxt_png as k
    except ImportError:
        raise SystemExit("Carve mode needs ko_dxt_png.py in the same folder as this script.")

    def plausible_start(m, o):
        if o == len(m):
            return True
        if o + 4 > len(m):
            return False
        n = struct.unpack_from('<i', m, o)[0]
        return 0 <= n <= 260 and all(32 <= b < 127 for b in m[o + 4:o + 4 + n])

    used, ok, unsure, manifest = set(), 0, 0, []
    os.makedirs(out_dir, exist_ok=True)
    with open(src_path, 'rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
        for mo in re.finditer(rb'NTF[\x00-\x0f]', m):
            p = mo.start()
            start = None
            for n in range(0, 261):          # look back for: int32 name length + name
                q = p - n - 4
                if q < 0:
                    break
                if struct.unpack_from('<i', m, q)[0] == n and all(32 <= b < 127 for b in m[q + 4:p]):
                    start, name = q, m[q + 4:p].decode('ascii')
                    break
            if start is None:
                continue
            hdr = p
            w, h = struct.unpack_from('<ii', m, hdr + 4)
            fc = m[hdr + 12:hdr + 16]
            mip = struct.unpack_from('<i', m, hdr + 16)[0]
            try:
                k.check_format(fc)
                main = k.chain(w, h, mip)
                size_main = k.chain_size(main, fc)
            except Exception:
                continue
            size_full = size_main
            if fc in k.DXT:
                cfc = struct.pack('<i', k.COMPANION[fc])
                size_full += k.chain_size(k.chain(max(1, w // 2), max(1, h // 2), mip), cfc)
            end_full, end_main = hdr + 20 + size_full, hdr + 20 + size_main
            if plausible_start(m, end_full):
                end = end_full
            elif plausible_start(m, end_main):
                end = end_main
            else:
                end = end_full
                unsure += 1
            fname = os.path.basename(name.replace('\\', '/')) or f"texture_{start:010d}"
            if not fname.lower().endswith('.dxt'):
                fname += '.dxt'
            rel = unique(safe_rel(fname), used)
            end = min(end, len(m))
            with open(os.path.join(out_dir, rel), 'wb') as o:
                o.write(m[start:end])
            manifest.append((rel, name, start, end - start))
            ok += 1
    write_manifest(out_dir, 'carve', src_path, None, manifest)
    print(f"Done (carve mode): {ok} textures extracted to {out_dir}"
          + (f"; {unsure} had an unclear end and may contain extra bytes" if unsure else ""))


MANIFEST = '_ko_index.txt'


def write_manifest(out_dir, mode, src, hdr, rows):
    with open(os.path.join(out_dir, MANIFEST), 'w', encoding='utf-8') as f:
        f.write(f"#mode\t{mode}\n#src\t{os.path.abspath(src)}\n#hdr\t{os.path.abspath(hdr) if hdr else ''}\n")
        for rel, name, pos, size in rows:
            f.write(f"{rel}\t{name}\t{pos}\t{size}\n")


def read_manifest(folder):
    path = os.path.join(folder, MANIFEST)
    if not os.path.isfile(path):
        raise SystemExit(f"{path} not found. Repack needs a folder made by this script's unpack.")
    meta, rows = {}, []
    for line in open(path, encoding='utf-8'):
        line = line.rstrip('\n')
        if line.startswith('#'):
            k, _, v = line[1:].partition('\t')
            meta[k] = v
        elif line:
            rel, name, pos, size = line.split('\t')
            rows.append((rel, name, int(pos), int(size)))
    return meta, rows


def edited_file(folder, rel):
    """The file to pack for this entry: name_new.ext beats name.ext."""
    path = os.path.join(folder, rel)
    base, ext = os.path.splitext(path)
    new = base + '_new' + ext
    return new if os.path.isfile(new) else path


def repack(folder):
    meta, rows = read_manifest(folder)
    src = meta['src']
    if not os.path.isfile(src):
        raise SystemExit(f"Original archive not found: {src}")
    out_src = os.path.splitext(src)[0] + '_new' + os.path.splitext(src)[1]

    if meta['mode'] == 'carve':
        import shutil
        shutil.copyfile(src, out_src)
        changed = skipped = 0
        with open(src, 'rb') as orig, open(out_src, 'r+b') as out:
            for rel, name, pos, size in rows:
                path = edited_file(folder, rel)
                if not os.path.isfile(path):
                    continue
                data = open(path, 'rb').read()
                orig.seek(pos)
                if data == orig.read(size):
                    continue
                if len(data) != size:
                    print(f"SKIP {rel}: {len(data)} bytes, original is {size}. "
                          "In-place patching needs the same size (re-make it with ko_dxt_png.py "
                          "using the original as template).")
                    skipped += 1
                    continue
                out.seek(pos)
                out.write(data)
                print(f"PATCH {rel}" + ("  (from _new file)" if path.endswith('_new' + os.path.splitext(rel)[1]) else ""))
                changed += 1
        print(f"\nDone: {out_src}\n  {changed} texture(s) patched in place, {skipped} skipped. "
              "Offsets are unchanged, so the existing .hdr stays valid.")
        return

    hdr = meta['hdr']
    if not os.path.isfile(hdr):
        raise SystemExit(f"Original index not found: {hdr}")
    out_hdr = os.path.splitext(hdr)[0] + '_new' + os.path.splitext(hdr)[1]
    header8 = open(hdr, 'rb').read(8)         # version + timestamp, kept as is

    # Files in the folder that are not in the index -> new entries.
    known = {r.lower() for r, *_ in rows}
    known |= {os.path.splitext(r)[0].lower() + '_new' + os.path.splitext(r)[1].lower() for r, *_ in rows}
    style_dir = any('\\' in n or '/' in n for _, n, _, _ in rows)
    extra = []
    for dp, _, fs in os.walk(folder):
        for fn in fs:
            rel = os.path.relpath(os.path.join(dp, fn), folder)
            if fn == MANIFEST or rel.lower() in known:
                continue
            name = rel.replace(os.sep, '\\') if style_dir else fn
            extra.append((rel, name, None, None))

    entries, changed, added = [], 0, 0
    written = {}                               # (pos, size) of unmodified data -> new offset
    with open(src, 'rb') as orig, open(out_src, 'wb') as out:
        for rel, name, pos, size in rows + extra:
            path = edited_file(folder, rel)
            data = None
            if pos is not None:
                orig.seek(pos)
                original = orig.read(size)
                if os.path.isfile(path):
                    data = open(path, 'rb').read()
                    if data == original:
                        data = None
                if data is None:
                    if (pos, size) in written:   # shared data: write once
                        entries.append((name, written[(pos, size)], size))
                        continue
                    data = original
                    written[(pos, size)] = out.tell()
                else:
                    changed += 1
                    print(f"UPDATE {rel}" + (f" ({size} -> {len(data)} bytes)" if len(data) != size else ""))
            else:
                data = open(path, 'rb').read()
                added += 1
                print(f"ADD    {rel} as '{name}'")
            new_pos = out.tell()
            if new_pos + len(data) > 0xFFFFFFFF:
                raise SystemExit("Archive would exceed 4 GB, which the .hdr format cannot address.")
            out.write(data)
            entries.append((name, new_pos, len(data)))

    with open(out_hdr, 'wb') as h:
        h.write(header8)
        for name, pos, size in entries:
            nb = name.encode('ascii')
            h.write(struct.pack('<H', len(nb)) + nb + struct.pack('<II', pos, size))
    print(f"\nDone: {out_src}\n      {out_hdr}\n  {changed} updated, {added} added, {len(entries)} entries total.")


def main(argv):
    if not argv:
        print(__doc__)
        return
    if argv[0] == '--repack':
        if len(argv) < 2:
            raise SystemExit("Usage: ko_unpack.py --repack <unpacked folder>")
        repack(argv[1])
        return
    src = argv[0]
    hdr = argv[1] if len(argv) > 1 and argv[1].lower().endswith('.hdr') else find_hdr(src)
    rest = [a for a in argv[1:] if not a.lower().endswith('.hdr')]
    out_dir = rest[0] if rest else os.path.splitext(src)[0] + '_unpacked'
    if hdr:
        print(f"Index: {hdr}")
        unpack_with_hdr(src, hdr, out_dir)
    else:
        print("No .hdr index found next to the .src -> carve mode (textures only).")
        carve_textures(src, out_dir)


if __name__ == '__main__':
    try:
        main(sys.argv[1:])
    except SystemExit:
        raise
    except Exception as e:
        print(f"ERROR: {e}")
    if len(sys.argv) == 2 and sys.stdin.isatty():
        try:
            input("\nPress Enter to close...")
        except EOFError:
            pass
