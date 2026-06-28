"""
Color encodings and working-space conversions.

Two reasons this module exists:

1. The shipped baseline did its channel rotation and selective grading in
   *nonlinear sRGB*, with hue/lightness entangled. That is the root cause of
   "skin too emerald" and "neutrals drift teal": a ratio tweak meant for one
   surface bleeds into another because the axes aren't perceptually separable.
   We fix that by working in LINEAR light for the mechanical transform and in
   OKLab/OKLCh (perceptually uniform, hue == angle) for the corrective grade.

2. The in-camera / Log target needs S-Log3 / S-Gamut3.Cine decode+encode. Those
   live here too so the secondary pipeline can reuse one tested set of curves.

Everything is vectorized over arrays of shape (..., 3).
"""

import numpy as np

from .backend import xp_for

EPS = 1e-12


# --------------------------------------------------------------------------
# sRGB  <->  linear  (IEC 61966-2-1)
# --------------------------------------------------------------------------
def srgb_to_linear(c):
    xp = xp_for(c)
    c = xp.asarray(c, dtype=xp.float64)
    return xp.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    xp = xp_for(c)
    c = xp.asarray(c, dtype=xp.float64)
    c = xp.clip(c, 0.0, None)
    return xp.where(c <= 0.0031308, c * 12.92, 1.055 * (c ** (1.0 / 2.4)) - 0.055)


# --------------------------------------------------------------------------
# linear sRGB / Rec.709  <->  OKLab  (Björn Ottosson)
# Input is LINEAR rgb (not sRGB-encoded). Rec.709 primaries == sRGB primaries.
# --------------------------------------------------------------------------
_M1 = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
])
_M2 = np.array([
    [0.2104542553,  0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050,  0.4505937099],
    [0.0259040371,  0.7827717662, -0.8086757660],
])
_M1_INV = np.linalg.inv(_M1)
_M2_INV = np.linalg.inv(_M2)


def linear_rgb_to_oklab(rgb):
    xp = xp_for(rgb)
    lms = rgb @ xp.asarray(_M1).T
    lms_ = xp.cbrt(xp.clip(lms, 0.0, None))
    return lms_ @ xp.asarray(_M2).T


def oklab_to_linear_rgb(lab):
    xp = xp_for(lab)
    lms_ = lab @ xp.asarray(_M2_INV).T
    lms = lms_ ** 3
    return lms @ xp.asarray(_M1_INV).T


# --------------------------------------------------------------------------
# OKLab  <->  OKLCh  (cylindrical: L, Chroma, hue-radians)
# --------------------------------------------------------------------------
def oklab_to_oklch(lab):
    xp = xp_for(lab)
    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]
    C = xp.sqrt(a * a + b * b)
    h = xp.arctan2(b, a)  # radians, (-pi, pi]
    return xp.stack([L, C, h], axis=-1)


def oklch_to_oklab(lch):
    xp = xp_for(lch)
    L = lch[..., 0]
    C = lch[..., 1]
    h = lch[..., 2]
    return xp.stack([L, C * xp.cos(h), C * xp.sin(h)], axis=-1)


def srgb_to_oklch(rgb_srgb):
    return oklab_to_oklch(linear_rgb_to_oklab(srgb_to_linear(rgb_srgb)))


# Convenience: hue helpers operating in degrees on the OKLab a/b plane.
# Reference hue angles (deg, approx): red ~29, orange/skin ~50-70, yellow ~110,
# green ~142, cyan ~195, blue ~264, magenta ~328.
def hue_deg(lch):
    xp = xp_for(lch)
    return xp.degrees(lch[..., 2]) % 360.0


def hue_window(h_deg, center, width):
    """Smooth 0..1 membership of hue `h_deg` in an arc [center +- width],
    wrapping at 360. Triangular falloff."""
    xp = xp_for(h_deg)
    d = xp.abs((h_deg - center + 180.0) % 360.0 - 180.0)
    return xp.clip(1.0 - d / (width + EPS), 0.0, 1.0)


# --------------------------------------------------------------------------
# S-Log3 / S-Gamut3.Cine  (secondary, in-camera / Log target)
# Curves per Sony "Technical Summary for S-Gamut3.Cine/S-Log3 and
# S-Gamut3/S-Log3". Code values are normalized 0..1 (i.e. CV/1023).
# --------------------------------------------------------------------------
def slog3_to_linear(x):
    """S-Log3 code value (0..1) -> scene-linear reflection (1.0 == 90% gray*... ).
    Returns linear with 0.18 mapping to mid-gray per Sony spec."""
    x = np.asarray(x, dtype=np.float64)
    below = (x * 1023.0 - 95.0) * 0.01125000 / (171.2102946929 - 95.0)
    above = (10.0 ** ((x * 1023.0 - 420.0) / 261.5)) * (0.18 + 0.01) - 0.01
    return np.where(x >= 171.2102946929 / 1023.0, above, below)


def linear_to_slog3(y):
    """Scene-linear reflection -> S-Log3 code value (0..1)."""
    y = np.asarray(y, dtype=np.float64)
    below = (y * (171.2102946929 - 95.0) / 0.01125000 + 95.0) / 1023.0
    above = (420.0 + np.log10((np.clip(y, -0.01 + EPS, None) + 0.01) / (0.18 + 0.01)) * 261.5) / 1023.0
    return np.where(y >= 0.01125000, above, below)


# S-Gamut3.Cine -> Rec.709/sRGB linear primaries (3x3, normalized rows).
# Source: Sony color gamut conversion matrices.
_SGAMUT3CINE_TO_REC709 = np.array([
    [ 1.6269474097, -0.3884858150, -0.2384615947],
    [-0.1051645940,  1.1817439768, -0.0765793828],
    [-0.0250474090, -0.0913476291,  1.1163950382],
])


def sgamut3cine_to_rec709_linear(rgb_lin):
    return rgb_lin @ _SGAMUT3CINE_TO_REC709.T


def rec709_linear_to_sgamut3cine(rgb_lin):
    return rgb_lin @ np.linalg.inv(_SGAMUT3CINE_TO_REC709).T
