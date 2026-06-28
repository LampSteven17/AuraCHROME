#!/usr/bin/env python3
"""
Generate Aerochrome .cube LUTs by sampling the transform over the N**3 grid.

Display variant (primary): sRGB / Rec.709 display-referred input.
Outputs land in luts/.

Usage:
    python scripts/make_cube.py                 # classic preset, all sizes
    python scripts/make_cube.py --preset punchy
    python scripts/make_cube.py --size 33
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aerochrome import cube, params, transform  # noqa: E402

LUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "luts")


def make(preset_name, size, outdir=LUTS_DIR, camera_safe=False):
    p = params.get(preset_name)
    grid = cube.build_grid(size)
    table = transform.aerochrome(grid, p)
    os.makedirs(outdir, exist_ok=True)
    if camera_safe:
        # 8.3-safe name for the A7C II card import
        fname = "AEROCHR.cube"
        title = "Aerochrome"
    else:
        fname = f"Aerochrome_{preset_name.capitalize()}_Display_{size}.cube"
        title = f"Aerochrome {preset_name} display {size}"
    path = os.path.join(outdir, fname)
    cube.write(path, table, size, title=title)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="classic", choices=list(params.PRESETS))
    ap.add_argument("--size", type=int, default=None, choices=[17, 33, 65])
    args = ap.parse_args()

    sizes = [args.size] if args.size else [33, 65]
    for sz in sizes:
        path = make(args.preset, sz)
        print(f"wrote {path}")
    # 8.3-named camera copy (33-point) of the chosen preset
    cam = make(args.preset, 33, camera_safe=True)
    print(f"wrote {cam}  (camera / A7C II import)")


if __name__ == "__main__":
    main()
