"""
DNG Camera Profile (.dcp) reader/writer + look-table injector.

A .dcp is a little-endian TIFF-structured file whose single IFD holds only
camera-profile tags. CRITICAL: real .dcp files use the magic number 0x4352
(not the standard TIFF 42) -- Lightroom requires this to accept the file as a
camera profile. (A generic TIFF reader like tifffile is lenient and will read a
42-magic file, which is how an earlier bug slipped through; Lightroom is not.)

Preferred path is `inject_look`: take Adobe's known-good Standard profile for the
camera and add our creative ProfileLookTable to it. That guarantees correct magic,
color matrices, HueSatMap and every tag LR needs -- we only graft the look on top.
`write_dcp` (from-scratch) is kept as a fallback and now also emits 0x4352.

We bake the look into ProfileLookTableData: a HSV delta table (hueShiftDeg,
satScale, valScale) that LR applies as a creative "look".
"""

import struct

DCP_MAGIC = 0x4352

# DNG / TIFF tag ids
T_UNIQUE_CAMERA_MODEL = 50708
T_COLOR_MATRIX_1 = 50721
T_CALIBRATION_ILLUMINANT_1 = 50778
T_PROFILE_NAME = 50936
T_PROFILE_EMBED_POLICY = 50941
T_PROFILE_COPYRIGHT = 50942
T_LOOK_TABLE_DIMS = 50981
T_LOOK_TABLE_DATA = 50982
T_LOOK_TABLE_ENCODING = 51108

ILLUMINANT_D65 = 21

# TIFF field-type -> byte size
_TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}


def _ascii_bytes(s):
    return s.encode("ascii") + b"\x00"


def _srational_bytes(values, den=1000000):
    out = b""
    for v in values:
        out += struct.pack("<ii", int(round(v * den)), den)
    return out


# --------------------------------------------------------------------------
# generic IFD read / write (accepts any magic on read, writes 0x4352)
# --------------------------------------------------------------------------
def read_ifd(path):
    """Return list of [tag, type, count, raw_value_bytes] for IFD0."""
    data = open(path, "rb").read()
    if data[:2] != b"II":
        raise ValueError("only little-endian (II) DCP files supported")
    _magic, ifd_off = struct.unpack("<HI", data[2:8])
    count = struct.unpack("<H", data[ifd_off:ifd_off + 2])[0]
    entries = []
    p = ifd_off + 2
    for _ in range(count):
        tag, typ, cnt = struct.unpack("<HHI", data[p:p + 8])
        valoff = data[p + 8:p + 12]
        size = _TYPE_SIZE.get(typ, 1) * cnt
        if size <= 4:
            raw = valoff[:size]
        else:
            off = struct.unpack("<I", valoff)[0]
            raw = data[off:off + size]
        entries.append([tag, typ, cnt, raw])
        p += 12
    return entries


def write_ifd(path, entries, magic=DCP_MAGIC):
    norm = sorted(entries, key=lambda e: e[0])
    n = len(norm)
    ifd_offset = 8
    ifd_size = 2 + 12 * n + 4
    ext_start = ifd_offset + ifd_size
    if ext_start & 1:
        ext_start += 1
    ext = b""
    cur = ext_start
    ifd = []
    for tag, typ, cnt, raw in norm:
        if len(raw) <= 4:
            valfield = raw + b"\x00" * (4 - len(raw))
        else:
            valfield = struct.pack("<I", cur)
            chunk = raw + (b"\x00" if len(raw) & 1 else b"")
            ext += chunk
            cur += len(chunk)
        ifd.append(struct.pack("<HHI", tag, typ, cnt) + valfield)
    out = b"II" + struct.pack("<HI", magic, ifd_offset)
    out += struct.pack("<H", n) + b"".join(ifd) + struct.pack("<I", 0)
    if len(out) < ext_start:
        out += b"\x00" * (ext_start - len(out))
    out += ext
    open(path, "wb").write(out)
    return path


# --------------------------------------------------------------------------
# the two public builders
# --------------------------------------------------------------------------
def _look_entries(profile_name, look_dims, look_data, encoding,
                  copyright="Aerochrome LUT toolkit"):
    return [
        [T_PROFILE_NAME, 2, len(profile_name) + 1, _ascii_bytes(profile_name)],
        [T_PROFILE_EMBED_POLICY, 4, 1, struct.pack("<I", 1)],
        [T_PROFILE_COPYRIGHT, 2, len(copyright) + 1, _ascii_bytes(copyright)],
        [T_LOOK_TABLE_DIMS, 3, 3, struct.pack("<3H", *look_dims)],
        [T_LOOK_TABLE_DATA, 11, len(look_data),
         struct.pack("<%df" % len(look_data), *look_data)],
        [T_LOOK_TABLE_ENCODING, 4, 1, struct.pack("<I", encoding)],
    ]


def inject_look(base_path, out_path, profile_name, look_dims, look_data, encoding=1):
    """Graft our look table onto a known-good base profile (preferred)."""
    new = _look_entries(profile_name, look_dims, look_data, encoding)
    replace = {e[0] for e in new}
    entries = [e for e in read_ifd(base_path) if e[0] not in replace]
    entries += new
    return write_ifd(out_path, entries, magic=DCP_MAGIC)


def write_dcp(path, profile_name, unique_model, color_matrix,
              look_dims, look_data, encoding=1):
    """From-scratch minimal profile (fallback when no base is available)."""
    entries = [
        [T_UNIQUE_CAMERA_MODEL, 2, len(unique_model) + 1, _ascii_bytes(unique_model)],
        [T_COLOR_MATRIX_1, 10, 9, _srational_bytes(color_matrix)],
        [T_CALIBRATION_ILLUMINANT_1, 3, 1, struct.pack("<H", ILLUMINANT_D65)],
    ] + _look_entries(profile_name, look_dims, look_data, encoding)
    return write_ifd(path, entries, magic=DCP_MAGIC)
