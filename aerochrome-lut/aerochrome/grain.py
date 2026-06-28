"""
Chromatic film grain -- a spatial pass for the export route (NOT bakeable into a
.cube or .dcp, which are per-pixel color maps).

Aerochrome / EIR is a three-layer false-color reversal stock, and its layers
grain semi-independently, so the grain is faintly *chromatic* (a subtle colored
shimmer in flat areas), not the pure luminance grain Lightroom's Grain panel can
make. We model that with a shared luminance noise field plus a smaller
independent per-channel field, mixed by `chroma`:

    chroma = 0  -> pure luminance grain (every channel identical)
    chroma = 1  -> fully independent per-channel grain (very colored)

Grain is weighted toward the midtones (it fades to nothing in clean blacks and
blown highlights, like real film), and its clump size is set by a Gaussian blur.
"""

import numpy as np

from .backend import gaussian, xp_for


def _field(rng, xp, h, w, sigma):
    """A unit-variance, blurred (grain-sized) noise field on `xp`'s device."""
    n = rng.standard_normal((h, w)).astype(xp.float64)
    if sigma > 0:
        n = gaussian(n, sigma)
    return n / (n.std() + 1e-9)


def chromatic_grain(rgb, strength, size, chroma=0.35, seed=0):
    """Add Aerochrome-style chromatic grain to a display-referred image.

    rgb: float array (...,3) in [0,1] (NumPy or CuPy -- runs on whichever device
    owns it).  strength: midtone noise sigma (display units, e.g. 0.03).  size:
    Gaussian sigma in px (grain clump size).  Returns the grained image, same
    shape/device, clipped to [0,1].
    """
    xp = xp_for(rgb)
    rgb = xp.asarray(rgb, dtype=xp.float64)
    if strength <= 0:
        return xp.clip(rgb, 0.0, 1.0)
    h, w = rgb.shape[0], rgb.shape[1]
    rng = xp.random.RandomState(int(seed) & 0x7FFFFFFF)

    base = _field(rng, xp, h, w, size)             # shared luminance grain
    lum = rgb @ xp.asarray([0.2126, 0.7152, 0.0722], dtype=xp.float64)
    mid = xp.clip(4.0 * lum * (1.0 - lum), 0.0, 1.0) ** 0.6   # fade at black/white

    out = rgb.copy()
    for c in range(3):
        indep = _field(rng, xp, h, w, size)        # per-layer component
        n = base * (1.0 - chroma) + indep * chroma
        out[..., c] = out[..., c] + strength * mid * n
    return xp.clip(out, 0.0, 1.0)
