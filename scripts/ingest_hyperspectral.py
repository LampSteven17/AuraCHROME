#!/usr/bin/env python3
"""
Hyperspectral cube -> (RGB, NIR-band) training pairs.

The data strategy (from the spectral-reconstruction research): NIR is in the
camera's "metameric black" null space, so you cannot recover it from RGB by
physics -- you train a learned RGB->NIR-band PRIOR. The right ground truth for
that is a hyperspectral cube spanning 400-1000nm (ICVL, bianlab): we SIMULATE the
RGB input by integrating the cube against CIE color-matching functions, and SLICE
the NIR target band(s) directly. Perfectly registered, no invented NIR.

We deliberately slice DISCRETE target bands (default 740/850/900nm = SFX / Aero /
HIE) rather than reconstruct a full spectrum -- the look only needs one value per
band, and full-spectrum reconstruction past 700nm is unconstrained anyway.

Reads ENVI (.hdr + raw cube). No heavy deps: numpy + tifffile only.

    python scripts/ingest_hyperspectral.py CUBE.hdr -o out/ --bands 740 850 900
    python scripts/ingest_hyperspectral.py icvl_dir/ -o out/        # all .hdr in a dir
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aerochrome import spectral  # noqa: E402

# ENVI data-type code -> numpy dtype
_ENVI_DTYPE = {1: "u1", 2: "i2", 3: "i4", 4: "f4", 5: "f8",
               12: "u2", 13: "u4", 14: "i8", 15: "u8"}


# --------------------------------------------------------------------------
# ENVI reader
# --------------------------------------------------------------------------
def read_envi(hdr_path):
    """Return (cube[H,W,B] float64, wavelengths_nm[B])."""
    with open(hdr_path) as f:
        text = f.read()
    # collapse {...} blocks (may span lines) then parse key = value
    meta = {}
    for m in re.finditer(r"^\s*([\w ]+?)\s*=\s*(\{.*?\}|.+?)\s*$",
                         text, re.MULTILINE | re.DOTALL):
        meta[m.group(1).strip().lower()] = m.group(2).strip()

    def _int(k):
        return int(meta[k])

    samples, lines, bands = _int("samples"), _int("lines"), _int("bands")
    dtype_code = _int("data type")
    interleave = meta.get("interleave", "bsq").lower()
    byte_order = int(meta.get("byte order", 0))
    offset = int(meta.get("header offset", 0))
    dt = np.dtype(_ENVI_DTYPE[dtype_code]).newbyteorder(">" if byte_order else "<")

    # locate the raw cube file (common extensions / same stem)
    stem = os.path.splitext(hdr_path)[0]
    raw = next((stem + e for e in ("", ".raw", ".img", ".dat", ".bin", ".cube")
                if os.path.isfile(stem + e)), None)
    if raw is None:
        raise FileNotFoundError(f"no raw cube next to {hdr_path}")

    data = np.fromfile(raw, dtype=dt, offset=offset, count=samples * lines * bands)
    data = data.astype(np.float64)
    if interleave == "bsq":      # band, row, col
        cube = data.reshape(bands, lines, samples).transpose(1, 2, 0)
    elif interleave == "bil":    # row, band, col
        cube = data.reshape(lines, bands, samples).transpose(0, 2, 1)
    else:                        # bip: row, col, band
        cube = data.reshape(lines, samples, bands)

    wl = None
    if "wavelength" in meta:
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
                          meta["wavelength"].strip("{}"))
        wl = np.array([float(n) for n in nums], dtype=np.float64)
        units = meta.get("wavelength units", "").lower()
        if "micro" in units or (wl.size and wl.max() < 100):  # microns -> nm
            wl *= 1000.0
    if wl is None or wl.size != bands:
        raise ValueError(f"{hdr_path}: need {bands} wavelengths, got "
                         f"{0 if wl is None else wl.size}")
    return cube, wl


# --------------------------------------------------------------------------
# render RGB + slice bands
# --------------------------------------------------------------------------
def render_rgb(refl, wl):
    """Simulate sRGB from a reflectance cube under equal-energy illuminant E."""
    vis = (wl >= 380) & (wl <= 730)
    wlv = wl[vis]
    xb, yb, zb = spectral.cie_xyz_bars(wlv)
    dlam = float(np.mean(np.diff(wlv))) if wlv.size > 1 else 1.0
    norm = np.sum(yb) * dlam  # so a perfect (refl=1) white -> Y=1
    r = refl[..., vis]
    X = (r * xb).sum(-1) * dlam / norm
    Y = (r * yb).sum(-1) * dlam / norm
    Z = (r * zb).sum(-1) * dlam / norm
    return spectral.xyz_to_srgb(np.stack([X, Y, Z], -1))


def slice_band(refl, wl, target, window):
    """Mean reflectance in [target-window, target+window]; nearest if empty."""
    sel = np.abs(wl - target) <= window
    if not sel.any():
        sel = np.abs(wl - target) == np.abs(wl - target).min()
    return refl[..., sel].mean(-1)


def _to_u16(a):
    return np.round(np.clip(a, 0.0, 1.0) * 65535.0).astype(np.uint16)


def ingest(hdr_path, outdir, bands, window, pct):
    cube, wl = read_envi(hdr_path)
    hi = np.percentile(cube, pct)
    refl = np.clip(cube / (hi + 1e-9), 0.0, 1.0)  # cube -> approx reflectance [0,1]

    stem = os.path.splitext(os.path.basename(hdr_path))[0]
    os.makedirs(outdir, exist_ok=True)
    rgb = render_rgb(refl, wl)
    tifffile.imwrite(os.path.join(outdir, f"{stem}_rgb.tif"), _to_u16(rgb))

    have = [b for b in bands if wl.min() - window <= b <= wl.max() + window]
    skipped = [b for b in bands if b not in have]
    for i, b in enumerate(have):
        nir = slice_band(refl, wl, b, window)
        tifffile.imwrite(os.path.join(outdir, f"{stem}_nir_{int(b)}.tif"), _to_u16(nir))
        if i == 0:  # primary band -> the trainer's `_nir.<ext>` convention
            tifffile.imwrite(os.path.join(outdir, f"{stem}_nir.tif"), _to_u16(nir))
    return stem, wl.min(), wl.max(), have, skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="an ENVI .hdr file, or a directory of them")
    ap.add_argument("-o", "--outdir", required=True)
    ap.add_argument("--bands", type=float, nargs="+", default=[740, 850, 900],
                    help="target NIR band centers in nm (default: 740 850 900)")
    ap.add_argument("--window", type=float, default=15.0,
                    help="nm half-window averaged around each band center")
    ap.add_argument("--pct", type=float, default=99.0,
                    help="percentile used to normalize the cube to [0,1] reflectance")
    args = ap.parse_args()

    if os.path.isdir(args.input):
        hdrs = sorted(glob.glob(os.path.join(args.input, "**", "*.hdr"), recursive=True))
    else:
        hdrs = [args.input]
    if not hdrs:
        sys.exit(f"no .hdr files at {args.input}")

    for hdr in hdrs:
        try:
            stem, lo, hi, have, skip = ingest(hdr, args.outdir, args.bands,
                                              args.window, args.pct)
        except Exception as e:  # keep going across a batch
            print(f"  SKIP {os.path.basename(hdr)}: {e}", file=sys.stderr)
            continue
        msg = f"  {stem}: cube {lo:.0f}-{hi:.0f}nm -> bands {[int(b) for b in have]}"
        if skip:
            msg += f"  (out of range, skipped: {[int(b) for b in skip]})"
        print(msg)
    print(f"done -> {args.outdir}")


if __name__ == "__main__":
    main()
