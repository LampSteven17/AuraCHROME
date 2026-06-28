#!/usr/bin/env python3
"""
Parameter sweep -> contact sheet, so tuning is visual.

Sweeps one or two params across a small range and renders the procedural scene
(from preview.py) for each combination into a single contact-sheet PNG.

Usage:
    python scripts/tune.py --param ir_veg --values 1.1,1.55,1.9
    python scripts/tune.py --param skin_red_lift --values 0.5,0.8,1.1 \
                           --param2 skin_green_cut --values2 0.2,0.3,0.4
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aerochrome import params, transform  # noqa: E402
from preview import make_scene, _font  # noqa: E402

PREVIEW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "luts", "preview")


def render(scene, p):
    out = transform.aerochrome(scene, p)
    return (np.clip(out, 0, 1) * 255 + 0.5).astype("uint8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="classic", choices=list(params.PRESETS))
    ap.add_argument("--param", required=True)
    ap.add_argument("--values", required=True, help="comma-separated floats")
    ap.add_argument("--param2", default=None)
    ap.add_argument("--values2", default=None)
    ap.add_argument("--out", default=os.path.join(PREVIEW_DIR, "tune.png"))
    args = ap.parse_args()

    vals1 = [float(x) for x in args.values.split(",")]
    vals2 = [float(x) for x in args.values2.split(",")] if args.param2 else [None]

    scene = make_scene(360, 240)
    th, tw = scene.shape[0], scene.shape[1]
    pad, lab = 6, 20
    cols, rows = len(vals1), len(vals2)
    sheet = Image.new("RGB", (cols * (tw + pad) + pad, rows * (th + lab + pad) + pad),
                      (20, 20, 22))
    d = ImageDraw.Draw(sheet)
    f = _font(13)

    for j, v2 in enumerate(vals2):
        for i, v1 in enumerate(vals1):
            p = params.get(args.preset)
            p[args.param] = v1
            if args.param2:
                p[args.param2] = v2
            tile = render(scene, p)
            x = pad + i * (tw + pad)
            y = pad + j * (th + lab + pad)
            sheet.paste(Image.fromarray(tile), (x, y + lab))
            cap = f"{args.param}={v1}"
            if args.param2:
                cap += f" {args.param2}={v2}"
            d.text((x + 2, y + 2), cap, font=f, fill=(230, 230, 230))

    os.makedirs(PREVIEW_DIR, exist_ok=True)
    sheet.save(args.out)
    print(f"wrote {args.out}  ({cols}x{rows} grid)")


if __name__ == "__main__":
    main()
