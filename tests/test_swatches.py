"""
Regression tests: lock in the expected hue FAMILY per surface (handoff §6).

We assert the hue family (a coarse arc on the OKLab hue wheel) plus, where it
matters, a chroma/neutrality bound -- not exact hex, since those shift with
tuning. Run: pytest -q   (or: python tests/test_swatches.py)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aerochrome import encodings as enc  # noqa: E402
from aerochrome import params, transform  # noqa: E402


def hx(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=float) / 255.0


# family -> (hue_center_deg, half_width_deg) on the OKLab wheel
FAMILY = {
    "magenta":  (340, 45),
    "crimson":  (5,   45),
    "green":    (140, 45),
    "blue":     (255, 50),
    "cyan":     (200, 45),
    "yellow_green": (110, 35),
}

# surface -> (input hex, allowed families OR 'neutral')
CASES = [
    ("healthy foliage", "#45752e", ["magenta", "crimson"]),
    ("dark pine",       "#1f381c", ["magenta", "crimson"]),
    ("sunlit grass",    "#739e38", ["magenta", "crimson"]),
    ("blue sky",        "#598cd9", ["blue", "cyan"]),
    ("deep zenith sky", "#2e57b8", ["blue", "cyan"]),
    ("red flower",      "#c71f1f", ["green"]),
    ("orange",          "#e0731a", ["green", "yellow_green"]),
    ("water (lake)",    "#243847", ["blue", "cyan"]),
    ("caucasian skin",  "#d19e85", ["yellow_green"]),
]

NEUTRAL_CASES = [
    ("white cloud", "#f2f2f5"),
    ("concrete",    "#8c8c8a"),
    ("asphalt",     "#38383b"),
]


def out_lch(hex_in):
    o = transform.aerochrome(hx(hex_in)[None, :], params.CLASSIC)[0]
    return enc.oklab_to_oklch(enc.linear_rgb_to_oklab(enc.srgb_to_linear(o)))


def in_family(h_deg, fam):
    c, w = FAMILY[fam]
    d = abs((h_deg - c + 180) % 360 - 180)
    return d <= w


def test_hue_families():
    failures = []
    for name, hexin, fams in CASES:
        lch = out_lch(hexin)
        h = np.degrees(lch[2]) % 360
        if not any(in_family(h, f) for f in fams):
            failures.append(f"{name}: hue {h:.0f} not in {fams}")
    assert not failures, "\n".join(failures)


def test_neutrals_stay_neutral():
    # neutral inputs must not gain chroma (the §7 de-teal guarantee)
    failures = []
    for name, hexin in NEUTRAL_CASES:
        lch = out_lch(hexin)
        if lch[1] > 0.035:  # OKLCh chroma
            failures.append(f"{name}: chroma {lch[1]:.3f} too high (teal drift)")
    assert not failures, "\n".join(failures)


def test_skin_not_emerald():
    # skin must be yellow-green (hue < 130) and desaturated, never emerald
    lch = out_lch("#d19e85")
    h = np.degrees(lch[2]) % 360
    assert 80 <= h <= 130, f"skin hue {h:.0f} not yellow-green"
    assert lch[1] < 0.075, f"skin chroma {lch[1]:.3f} too saturated (emerald)"


if __name__ == "__main__":
    test_hue_families()
    test_neutrals_stay_neutral()
    test_skin_not_emerald()
    print("all swatch regression tests passed")
