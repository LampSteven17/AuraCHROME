"""
Read / write Adobe/Resolve `.cube` 3D LUTs.

Format:
    TITLE "..."            (optional)
    LUT_3D_SIZE N
    <N**3 rows of "R G B" floats in [0,1]>

Ordering: RED varies FASTEST, then GREEN, then BLUE. i.e. the row index is
    idx = r + g*N + b*N*N
This matches the Resolve / Adobe convention. build_grid() below emits sample
coordinates in exactly this order so a transform sampled over it can be written
straight out.
"""

import numpy as np


def build_grid(n):
    """Return (n**3, 3) array of input coordinates in canonical .cube order
    (red fastest). Coordinates are evenly spaced on [0,1]."""
    axis = np.linspace(0.0, 1.0, n)
    # meshgrid with 'ij' then ordering so red is fastest:
    r, g, b = np.meshgrid(axis, axis, axis, indexing='ij')
    # We want idx = r + g*N + b*N*N  => red fastest. Build by iterating b slowest.
    grid = np.stack([
        np.tile(axis, n * n),                          # red   fastest
        np.tile(np.repeat(axis, n), n),                # green middle
        np.repeat(axis, n * n),                        # blue  slowest
    ], axis=-1)
    return grid


def write(path, table, size, title=None, decimals=6):
    """table: (size**3, 3) array in canonical (red-fastest) order, values [0,1]."""
    table = np.clip(np.asarray(table, dtype=np.float64), 0.0, 1.0)
    if table.shape != (size ** 3, 3):
        raise ValueError(f"table must be ({size**3}, 3), got {table.shape}")
    fmt = f"%.{decimals}f"
    lines = []
    if title:
        lines.append(f'TITLE "{title}"')
    lines.append(f"LUT_3D_SIZE {size}")
    lines.append("DOMAIN_MIN 0.0 0.0 0.0")
    lines.append("DOMAIN_MAX 1.0 1.0 1.0")
    for row in table:
        lines.append(f"{fmt % row[0]} {fmt % row[1]} {fmt % row[2]}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def read(path):
    """Return (size, table) where table is (size**3, 3) in canonical order."""
    size = None
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key = line.split()[0].upper()
            if key == "LUT_3D_SIZE":
                size = int(line.split()[1])
            elif key in ("TITLE", "DOMAIN_MIN", "DOMAIN_MAX", "LUT_3D_INPUT_RANGE"):
                continue
            else:
                parts = line.split()
                if len(parts) == 3:
                    try:
                        rows.append([float(x) for x in parts])
                    except ValueError:
                        continue
    if size is None:
        raise ValueError("no LUT_3D_SIZE found")
    table = np.array(rows, dtype=np.float64)
    if table.shape != (size ** 3, 3):
        raise ValueError(f"expected {size**3} rows, got {len(rows)}")
    return size, table


def apply_trilinear(img, size, table):
    """Apply a 3D LUT to img (...,3) in [0,1] via trilinear interpolation.
    table is canonical (red-fastest) order."""
    img = np.clip(np.asarray(img, dtype=np.float64), 0.0, 1.0)
    grid = table.reshape(size, size, size, 3)  # [b, g, r, 3] given red-fastest
    coord = img * (size - 1)
    i0 = np.floor(coord).astype(int)
    i1 = np.clip(i0 + 1, 0, size - 1)
    fr = coord - i0
    r0, g0, b0 = i0[..., 0], i0[..., 1], i0[..., 2]
    r1, g1, b1 = i1[..., 0], i1[..., 1], i1[..., 2]
    fx = fr[..., 0:1]; fy = fr[..., 1:2]; fz = fr[..., 2:3]

    def at(rr, gg, bb):
        return grid[bb, gg, rr]

    c00 = at(r0, g0, b0) * (1 - fx) + at(r1, g0, b0) * fx
    c01 = at(r0, g0, b1) * (1 - fx) + at(r1, g0, b1) * fx
    c10 = at(r0, g1, b0) * (1 - fx) + at(r1, g1, b0) * fx
    c11 = at(r0, g1, b1) * (1 - fx) + at(r1, g1, b1) * fx
    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy
    return c0 * (1 - fz) + c1 * fz
