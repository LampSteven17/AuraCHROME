#!/usr/bin/env python3
"""
Visual sanity check for the transform.

Produces, in luts/preview/:
  * swatches.png  -- labeled before/after grid of the §6 validation surfaces,
                     each annotated with its output hex and hue family.
  * scene.png     -- a small procedural scene (sky gradient + foliage blobs +
                     a red object + skin patch + cloud) before/after.

No external test image required. Run after editing transform/params to eyeball
whether it still reads as Aerochrome.

Usage:
    python scripts/preview.py [--preset classic]
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aerochrome import params, transform  # noqa: E402

PREVIEW_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "luts", "preview"
)

# §6 validation surfaces: (label, input hex, expected family)
SWATCHES = [
    ("healthy foliage", "#45752e", "magenta/red"),
    ("dark pine",       "#1f381c", "deep crimson"),
    ("sunlit grass",    "#739e38", "hot magenta"),
    ("blue sky",        "#598cd9", "blue/cyan"),
    ("deep zenith sky", "#2e57b8", "deep blue"),
    ("red flower",      "#c71f1f", "green"),
    ("orange",          "#e0731a", "green"),
    ("white cloud",     "#f2f2f5", "neutral/white"),
    ("water (lake)",    "#243847", "dark blue/black"),
    ("caucasian skin",  "#d19e85", "pale yellow-green"),
    ("concrete",        "#8c8c8a", "near-neutral"),
    ("asphalt",         "#38383b", "dark neutral"),
    ("yellow",          "#e6d133", "cyan-green"),
]


def hex2rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float64) / 255.0


def rgb2hex(rgb):
    v = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(int)
    return "#{:02x}{:02x}{:02x}".format(*v)


def _font(sz=13):
    for cand in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(cand, sz)
        except OSError:
            continue
    return ImageFont.load_default()


def render_swatches(p, path):
    rowh, sw = 56, 120
    W = 230 + sw * 2 + 20
    H = 40 + rowh * len(SWATCHES)
    img = Image.new("RGB", (W, H), (24, 24, 26))
    d = ImageDraw.Draw(img)
    f = _font(13); fb = _font(15)
    d.text((12, 12), "surface", font=fb, fill=(220, 220, 220))
    d.text((242, 12), "INPUT", font=fb, fill=(220, 220, 220))
    d.text((242 + sw + 20, 12), "AEROCHROME", font=fb, fill=(220, 220, 220))

    inputs = np.array([hex2rgb(hx) for _, hx, _ in SWATCHES])
    outs = transform.aerochrome(inputs, p)
    for i, (label, hx, fam) in enumerate(SWATCHES):
        y = 40 + i * rowh
        d.text((12, y + 6), label, font=f, fill=(230, 230, 230))
        d.text((12, y + 24), fam, font=_font(11), fill=(150, 150, 150))
        ic = tuple(int(c * 255) for c in inputs[i])
        oc = tuple(int(c * 255) for c in np.clip(outs[i], 0, 1))
        d.rectangle([242, y + 2, 242 + sw, y + rowh - 8], fill=ic)
        d.rectangle([242 + sw + 20, y + 2, 242 + sw * 2 + 20, y + rowh - 8], fill=oc)
        d.text((242 + 4, y + rowh - 22), hx, font=_font(11), fill=(255, 255, 255))
        d.text((242 + sw + 24, y + rowh - 22), rgb2hex(np.clip(outs[i], 0, 1)),
               font=_font(11), fill=(255, 255, 255))
    img.save(path)
    return outs


def make_scene(w=480, h=320):
    """Procedural scene as sRGB float (h,w,3)."""
    img = np.zeros((h, w, 3))
    yy = np.linspace(0, 1, h)[:, None]
    # sky gradient: deep blue (top) -> pale blue (horizon)
    sky_top = hex2rgb("#2e57b8"); sky_bot = hex2rgb("#9cc0ec")
    img[:] = sky_top[None, None] * (1 - yy[..., None]) + sky_bot[None, None] * yy[..., None]
    # ground band
    ground = hex2rgb("#5a5a55")
    horizon = int(h * 0.62)
    img[horizon:] = ground
    xx = np.arange(w)[None, :]; yg = np.arange(h)[:, None]
    # cloud (soft white blob, upper area)
    cd = np.exp(-(((xx - w * 0.74) / 70.0) ** 2 + ((yg - h * 0.22) / 26.0) ** 2))
    img = img * (1 - cd[..., None] * 0.9) + hex2rgb("#f2f2f5")[None, None] * (cd[..., None] * 0.9)
    # foliage blobs (varied greens)
    for cx, cy, rad, hx in [(0.18, 0.78, 60, "#45752e"), (0.34, 0.86, 46, "#739e38"),
                            (0.5, 0.8, 52, "#1f381c"), (0.66, 0.88, 40, "#5c8a3a")]:
        m = ((xx - w * cx) ** 2 + (yg - h * cy) ** 2) < rad ** 2
        img[m] = hex2rgb(hx)
    # red object
    m = ((xx - w * 0.82) ** 2 + (yg - h * 0.8) ** 2) < 26 ** 2
    img[m] = hex2rgb("#c71f1f")
    # skin patch (a face-ish rectangle)
    img[int(h*0.66):int(h*0.78), int(w*0.04):int(w*0.13)] = hex2rgb("#d19e85")
    # concrete strip
    img[int(h*0.92):, :] = hex2rgb("#8c8c8a")
    return np.clip(img, 0, 1)


def render_scene(p, path):
    scene = make_scene()
    out = transform.aerochrome(scene, p)
    combo = np.concatenate([scene, np.ones((scene.shape[0], 8, 3)) * 0.1, out], axis=1)
    Image.fromarray((np.clip(combo, 0, 1) * 255 + 0.5).astype("uint8")).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="classic", choices=list(params.PRESETS))
    args = ap.parse_args()
    os.makedirs(PREVIEW_DIR, exist_ok=True)
    p = params.get(args.preset)
    outs = render_swatches(p, os.path.join(PREVIEW_DIR, "swatches.png"))
    render_scene(p, os.path.join(PREVIEW_DIR, "scene.png"))
    print(f"wrote {PREVIEW_DIR}/swatches.png and scene.png  (preset={args.preset})")
    # also dump the table to stdout for quick diffing against §6
    print(f"\n{'surface':18} {'input':9} {'output':9} family")
    for (label, hx, fam), o in zip(SWATCHES, outs):
        print(f"{label:18} {hx:9} {rgb2hex(np.clip(o,0,1)):9} {fam}")


if __name__ == "__main__":
    main()
