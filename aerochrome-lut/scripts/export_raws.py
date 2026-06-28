#!/usr/bin/env python3
"""
Batch-apply the Aerochrome transform directly to camera RAW files -- the
"export route" (outside Lightroom). This is the exact display-space color the
.cube produces: shadow-protect included, and NO profile look-table banding,
because nothing is approximated through an HSV table.

For each RAW it decodes with rawpy (camera white balance), runs the transform
for each requested preset, and writes a full-resolution JPEG named
  <preset>-<n>.jpg
where <n> is the 1-based position of the RAW in sorted order -- matching the
Lightroom export naming (classic-1.jpg ...) so the two sets line up for A/B.

The transform is pure per-pixel, so the image is processed in pixel chunks to
keep memory bounded regardless of resolution (A7C II files are ~33 MP).

Note: grain is NOT added here. Grain lives in Lightroom's Grain panel (it is a
spatial effect, not part of the color map). This keeps the comparison about
color / banding / shadows. Add grain in LR, or ask for a separate grain pass.

Usage:
    python scripts/export_raws.py --indir DIR --outdir DIR
    python scripts/export_raws.py --indir DIR --outdir DIR --presets classic muted
    python scripts/export_raws.py ... --longedge 4000   # downscale for speed
"""
import argparse
import glob
import os
import sys

import numpy as np
import rawpy
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aerochrome import params, transform  # noqa: E402


def process(arr, p, chunk_px=2_000_000):
    """Apply the transform in flat pixel chunks to bound memory."""
    h, w, _ = arr.shape
    flat = arr.reshape(-1, 3)
    out = np.empty_like(flat)
    for i in range(0, flat.shape[0], chunk_px):
        out[i:i + chunk_px] = transform.aerochrome(flat[i:i + chunk_px], p)
    return out.reshape(h, w, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, help="folder of RAW (.ARW) files")
    ap.add_argument("--outdir", required=True, help="output folder for JPEGs")
    ap.add_argument("--presets", nargs="+", default=list(params.PRESETS),
                    help="which looks to render (default: all four)")
    ap.add_argument("--longedge", type=int, default=0,
                    help="downscale so the long edge is this many px (0 = full res)")
    ap.add_argument("--ext", default="ARW", help="RAW extension to glob (default ARW)")
    ap.add_argument("--quality", type=int, default=95)
    args = ap.parse_args()

    raws = sorted(glob.glob(os.path.join(args.indir, f"*.{args.ext}")))
    if not raws:
        sys.exit(f"no *.{args.ext} files in {args.indir}")
    os.makedirs(args.outdir, exist_ok=True)

    print(f"{len(raws)} RAW(s), presets={args.presets} -> {args.outdir}")
    for n, raw_path in enumerate(raws, start=1):
        with rawpy.imread(raw_path) as raw:
            rgb16 = raw.postprocess(use_camera_wb=True, output_bps=16,
                                    no_auto_bright=False)
        arr = rgb16.astype(np.float64) / 65535.0
        if args.longedge:
            im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
            w, h = im.size
            s = args.longedge / max(w, h)
            if s < 1.0:
                im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
            arr = np.asarray(im).astype(np.float64) / 255.0
        base = os.path.splitext(os.path.basename(raw_path))[0]
        print(f"  [{n}/{len(raws)}] {base}  {arr.shape[1]}x{arr.shape[0]}")
        for pre in args.presets:
            out = process(arr, params.get(pre))
            out8 = (np.clip(out, 0, 1) * 255.0 + 0.5).astype(np.uint8)
            dst = os.path.join(args.outdir, f"{pre}-{n}.jpg")
            Image.fromarray(out8).save(dst, quality=args.quality)
            print(f"        {pre:9s} -> {os.path.basename(dst)}")
    print("done.")


if __name__ == "__main__":
    main()
