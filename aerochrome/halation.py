"""
Halation / highlight bloom -- the Kodak HIE and Efke/Maco IR820-AURA signature.

Those stocks omit the anti-halation backing, so bright light scatters inside the
emulsion/base and bounces back out, blooming a soft (classically reddish) glow
around highlights. Like grain, this is a SPATIAL export pass -- it cannot live in
a per-pixel .cube/.dcp. Aerochrome itself has a normal base (no halation), so the
default for the existing looks is strength=0 (off).

Runs on numpy or cupy via the shared backend (one Gaussian on the highlight
field, so it's cheap even at full resolution).
"""

from .backend import gaussian, xp_for

# Warm red-orange glow, the look of HIE halation. (1,1,1) gives a neutral bloom.
DEFAULT_TINT = (1.0, 0.45, 0.28)


def halation(rgb, strength=0.0, size=6.0, threshold=0.72, tint=DEFAULT_TINT):
    """Bloom highlights of a display-referred image.

    rgb: float (..,3) in [0,1].  strength: glow amount (0 = off).  size: Gaussian
    sigma in px (spread).  threshold: luma above which highlights bloom.  tint:
    glow colour. Returns the same shape/device, clipped to [0,1]."""
    xp = xp_for(rgb)
    rgb = xp.asarray(rgb, dtype=xp.float64)
    if strength <= 0:
        return xp.clip(rgb, 0.0, 1.0)

    luma = rgb @ xp.asarray([0.2126, 0.7152, 0.0722], dtype=xp.float64)
    # soft, eased highlight mask so only genuine highlights glow
    hi = xp.clip((luma - threshold) / (1.0 - threshold + 1e-9), 0.0, 1.0)
    hi = hi * hi * luma                      # white highlight field
    if size > 0:
        hi = gaussian(hi, size)              # one blur of the scalar field

    t = xp.asarray(tint, dtype=xp.float64)
    glow = xp.clip(strength * hi[..., None] * t, 0.0, 1.0)
    # screen-blend the glow over the image (adds light without harsh clipping)
    out = 1.0 - (1.0 - rgb) * (1.0 - glow)
    return xp.clip(out, 0.0, 1.0)
