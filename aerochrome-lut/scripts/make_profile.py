#!/usr/bin/env python3
"""
Bake the Aerochrome transform into a Lightroom camera profile (.dcp).

The look is sampled into a ProfileLookTable (HSV hue-shift / sat-scale /
val-scale deltas). Drop the resulting .dcp into Lightroom's CameraProfiles
folder and it appears in the Profile Browser for the camera, where it can be
applied with one click and saved into a preset.

Caveats (see README): a profile applies its look in LR's reference space, so it
*approximates* the display-space .cube rather than matching it exactly; and the
embedded ColorMatrix is a representative Sony value, not the exact ILCE-7CM2
calibration. Both are fine for a stylised look and refinable once you eyeball it.

Usage:
    python scripts/make_profile.py
    python scripts/make_profile.py --preset punchy --model ILCE-7CM2
"""
import argparse
import colorsys
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aerochrome import dcp, encodings as enc, params, transform  # noqa: E402

# Lightroom applies camera-profile look tables in a HSV space built on the
# ProPhoto RGB primaries -- NOT sRGB. Sampling the look in sRGB makes hue-targeted
# corrections (cyan suppression, skin) land on the wrong colors in LR. These let
# us sample each node in ProPhoto space so the corrections line up.
try:
    import colour
    _PP2S = colour.matrix_RGB_to_RGB(colour.models.RGB_COLOURSPACE_PROPHOTO_RGB,
                                     colour.models.RGB_COLOURSPACE_sRGB)
    _S2PP = colour.matrix_RGB_to_RGB(colour.models.RGB_COLOURSPACE_sRGB,
                                     colour.models.RGB_COLOURSPACE_PROPHOTO_RGB)
except Exception:
    _PP2S = _S2PP = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LUTS_DIR = os.path.join(ROOT, "luts")

# Representative Sony E-mount ColorMatrix1 (D65, XYZ->camera). Stand-in for the
# exact ILCE-7CM2 values; the creative look table dominates the result.
SONY_COLOR_MATRIX = [
    0.7374, -0.2389, -0.0551,
    -0.5435, 1.3162, 0.2519,
    -0.1006, 0.1795, 0.6552,
]

EPS = 1e-6

# Adobe ships a known-good Standard profile per camera; injecting our look into
# it is far more robust than building a profile from scratch.
ADOBE_STD_DIRS = [
    "/mnt/c/Program Files/Adobe/Adobe Lightroom Classic/Resources/CameraProfiles/Adobe Standard",
    "/mnt/c/ProgramData/Adobe/CameraRaw/CameraProfiles/Adobe Standard",
]


def default_base(model):
    for d in ADOBE_STD_DIRS:
        cand = os.path.join(d, f"Sony {model} Adobe Standard.dcp")
        if os.path.exists(cand):
            return cand
    return None


def _node_to_srgb_display(rgb_enc, reference):
    """A look-table node's HSV->RGB, interpreted in the reference space, returned
    as sRGB display values ready for the transform."""
    if reference == "prophoto" and _PP2S is not None:
        lin_pp = enc.srgb_to_linear(np.array(rgb_enc))      # value uses sRGB encoding
        lin_srgb = np.clip(lin_pp @ _PP2S.T, 0.0, 1.0)
        return enc.linear_to_srgb(lin_srgb)
    return np.array(rgb_enc)


def _srgb_display_to_node(out_srgb, reference):
    """Inverse: transform output (sRGB display) back to reference-space RGB."""
    if reference == "prophoto" and _S2PP is not None:
        lin_srgb = enc.srgb_to_linear(out_srgb)
        lin_pp = np.clip(lin_srgb @ _S2PP.T, 0.0, 1.0)
        return enc.linear_to_srgb(lin_pp)
    return np.clip(out_srgb, 0, 1)


def build_look_table(p, hue_div=36, sat_div=8, val_div=8, max_scale=6.0,
                     reference="prophoto"):
    """Sample the transform into a HSV delta table, in Lightroom's reference space.
    Order: value (slow) -> hue -> saturation (fast), each entry
    (hueShiftDeg, satScale, valScale)."""
    data = []
    for vi in range(val_div):
        val = vi / (val_div - 1)
        for hi in range(hue_div):
            hue = hi / hue_div            # colorsys hue is 0..1
            for si in range(sat_div):
                sat = si / (sat_div - 1)
                rgb_enc = colorsys.hsv_to_rgb(hue, sat, val)
                disp = _node_to_srgb_display(rgb_enc, reference)
                out = transform.aerochrome(disp[None, :], p)[0]
                node_out = _srgb_display_to_node(out, reference)
                h2, s2, v2 = colorsys.rgb_to_hsv(*node_out)
                # hue shift in degrees, wrapped to (-180, 180]
                dh = ((h2 - hue) * 360.0 + 180.0) % 360.0 - 180.0
                ssc = (s2 / sat) if sat > EPS else 1.0
                vsc = (v2 / val) if val > EPS else 1.0
                # Lightroom applies the look table in a WIDER color space than
                # sRGB, so satScale > 1 over-saturates (neon cyan). Cap the boost
                # for cyan/green OUTPUT hues; leave magenta (foliage) free to pop.
                out_hue = (h2 * 360.0) % 360.0
                if 120.0 <= out_hue <= 240.0:           # cyan-green-teal arc
                    ssc = min(ssc, p.get('lut_cyan_satcap', 1.0))
                data.append(dh)
                data.append(float(np.clip(ssc, 0.0, max_scale)))
                data.append(float(np.clip(vsc, 0.0, max_scale)))
    return (hue_div, sat_div, val_div), data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="classic", choices=list(params.PRESETS))
    ap.add_argument("--model", default="ILCE-7CM2",
                    help="UniqueCameraModel the profile attaches to (A7C II = ILCE-7CM2)")
    ap.add_argument("--base", default=None,
                    help="base .dcp to inject the look into (defaults to Adobe Standard for the model)")
    ap.add_argument("--version", default="v5",
                    help="version tag in the profile name (avoids LR caching a stale profile)")
    ap.add_argument("--reference", default="prophoto", choices=["prophoto", "srgb"],
                    help="color space to sample the look table in (LR uses ProPhoto)")
    # Higher sat/val resolution than the original 8x8: a coarse table contours
    # visible ripples in smooth low-chroma gradients (hazy skies). 12x16 roughly
    # quarters the contour spacing along the tonal axis where banding shows most.
    ap.add_argument("--hue", type=int, default=36)
    ap.add_argument("--sat", type=int, default=12)
    ap.add_argument("--val", type=int, default=16)
    args = ap.parse_args()

    p = params.get(args.preset)
    dims, data = build_look_table(p, args.hue, args.sat, args.val, reference=args.reference)
    name = f"Aerochrome {args.preset.capitalize()} {args.version}"
    out = os.path.join(LUTS_DIR, f"Aerochrome_{args.preset.capitalize()}_{args.version}.dcp")
    os.makedirs(LUTS_DIR, exist_ok=True)

    base = args.base or default_base(args.model)
    if base and os.path.exists(base):
        dcp.inject_look(base, out, name, dims, data)
        how = f"injected into base: {os.path.basename(base)}"
    else:
        dcp.write_dcp(out, name, args.model, SONY_COLOR_MATRIX, dims, data)
        how = "from scratch (no Adobe base found; color may be approximate)"

    print(f"wrote {out}")
    print(f"  profile name : {name}")
    print(f"  camera model : {args.model}")
    print(f"  look table   : {dims[0]}x{dims[1]}x{dims[2]} ({len(data)//3} nodes)")
    print(f"  method       : {how}")


if __name__ == "__main__":
    main()
