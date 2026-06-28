#!/usr/bin/env python3
"""
Apply a .cube LUT to one image, many images, or a whole folder.

Input must be sRGB / Rec.709 display-referred (a normal exported still). 8- and
16-bit TIFF/PNG and 8-bit JPEG are supported; 16-bit output is preserved for
TIFF/PNG so you keep headroom for further editing.

This doubles as:
  * the test rig for real photos (drop an A7C II raw exported as TIFF/JPEG), and
  * the engine behind the Lightroom "Export post-process" workflow (see README):
    point LR's export at a folder, run this over that folder, done.

Usage:
    python scripts/apply_lut.py in.jpg out.jpg
    python scripts/apply_lut.py in.tif out.tif --preset punchy
    python scripts/apply_lut.py --indir exported/ --outdir aerochrome/   # batch
    python scripts/apply_lut.py in.jpg out.jpg --direct   # full-precision, no LUT
"""
import argparse
import glob
import os
import sys

import numpy as np
import tifffile
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aerochrome import cube, params, transform  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LUT = os.path.join(ROOT, "luts", "Aerochrome_Classic_Display_65.cube")
EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def _load(path):
    """Return (float array (...,3) in [0,1], maxval). TIFF goes through tifffile
    so 16-bit RGB survives (PIL silently downconverts multichannel 16-bit)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        arr = tifffile.imread(path)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, -1)
        arr = arr[..., :3]
        maxval = 65535 if arr.dtype == np.uint16 else 255
        return arr.astype(np.float64) / maxval, maxval
    im = Image.open(path).convert("RGB")
    return np.asarray(im, dtype=np.float64) / 255.0, 255


def _save(path, out01, maxval):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        if maxval == 65535:
            tifffile.imwrite(path, np.clip(out01 * 65535.0 + 0.5, 0, 65535).astype(np.uint16))
        else:
            tifffile.imwrite(path, np.clip(out01 * 255.0 + 0.5, 0, 255).astype(np.uint8))
    else:
        data = np.clip(out01 * 255.0 + 0.5, 0, 255).astype(np.uint8)
        Image.fromarray(data).save(path)


def process(infile, outfile, size_table, direct, preset):
    arr, maxval = _load(infile)
    if direct:
        out = transform.aerochrome(arr, params.get(preset))
    else:
        size, table = size_table
        out = cube.apply_trilinear(arr, size, table)
    _save(outfile, out, maxval)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile", nargs="?")
    ap.add_argument("outfile", nargs="?")
    ap.add_argument("--indir", help="batch: process every image in this folder")
    ap.add_argument("--outdir", help="batch: write results here (mirrors filenames)")
    ap.add_argument("--suffix", default="_aerochrome",
                    help="filename suffix in batch mode (default: _aerochrome)")
    ap.add_argument("--lut", default=DEFAULT_LUT)
    ap.add_argument("--direct", action="store_true",
                    help="apply the transform directly instead of via the .cube")
    ap.add_argument("--preset", default="classic", choices=list(params.PRESETS))
    args = ap.parse_args()

    size_table = None if args.direct else cube.read(args.lut)
    tag = "direct" if args.direct else os.path.basename(args.lut)

    if args.indir:
        outdir = args.outdir or os.path.join(args.indir, "aerochrome")
        os.makedirs(outdir, exist_ok=True)
        files = [f for f in sorted(glob.glob(os.path.join(args.indir, "*")))
                 if f.lower().endswith(EXTS)]
        if not files:
            print(f"no images ({', '.join(EXTS)}) found in {args.indir}")
            return
        for f in files:
            base, ext = os.path.splitext(os.path.basename(f))
            out = os.path.join(outdir, f"{base}{args.suffix}{ext}")
            process(f, out, size_table, args.direct, args.preset)
            print(f"  {os.path.basename(f)} -> {os.path.basename(out)}")
        print(f"done: {len(files)} image(s) via {tag} -> {outdir}")
        return

    if not (args.infile and args.outfile):
        ap.error("give infile + outfile, or use --indir/--outdir for batch")
    process(args.infile, args.outfile, size_table, args.direct, args.preset)
    print(f"wrote {args.outfile}  ({tag})")


if __name__ == "__main__":
    main()
