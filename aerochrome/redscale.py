"""
Redscale -- a PARAMETRIC, visible-light film effect (no infrared, no neural net).

Mechanism (per the datasheets): colour-negative film shot through its BASE, so
light hits the emulsion stack in reverse order. The red-sensitive layer (normally
the bottom) is now on top and over-exposed; the green layer is partially exposed;
the blue layer (now deepest, and still behind the film's yellow filter layer) is
strongly suppressed. Net look: a warm red -> orange -> yellow gradient with the
blue/cyan crushed, and the amount of green/yellow that survives is driven by
exposure (over-expose to drive light through the base).

This is the counter-example that motivates the StockProfile split: redscale lives
ENTIRELY in parametric channel/tone space -- it needs none of the IR machinery.
Runs on numpy or cupy (dispatched per-array, like the rest of the engine).
"""

from .backend import xp_for
from .encodings import linear_to_srgb, srgb_to_linear

# Defaults tuned for a believable mid-exposure redscale (deep but not black
# shadows, orange midtones, crushed blue). `exposure` is the main creative knob.
DEFAULT = dict(
    exposure=1.0,        # overall light through the base; >1 lifts green/yellow survival
    red_gain=1.18,       # the now-frontmost red layer is over-exposed
    green_survive=0.55,  # green layer partially exposed
    blue_survive=0.12,   # blue layer crushed (deepest + behind the yellow filter)
    cross_rg=0.22,       # red bleeds into green -> warms midtones toward orange
    base_warm=0.12,      # orange base cast, strongest in the shadows
    contrast=0.10,       # gentle S-curve
    black_lift=0.012,
    # shared chromatic grain (export pass; see grain.py)
    grain_strength=0.030,
    grain_size=1.2,
    grain_chroma=0.25,
)


def _contrast(x, amt):
    """Gentle S-curve toward smoothstep, pivoting at mid-grey."""
    s = x * x * (3.0 - 2.0 * x)
    return (1.0 - amt) * x + amt * s


def apply(rgb_srgb, p=None):
    """Apply redscale to a display-referred sRGB image (..,3) in [0,1].

    Returns the graded sRGB image, same shape/device, clipped to [0,1]."""
    if p is None:
        p = DEFAULT
    xp = xp_for(rgb_srgb)
    lin = srgb_to_linear(rgb_srgb)
    r = lin[..., 0]
    g = lin[..., 1]
    b = lin[..., 2]

    expo = p["exposure"]
    R = xp.clip(r * p["red_gain"] * expo, 0.0, None)
    G = xp.clip((g * p["green_survive"] + R * p["cross_rg"]) * expo, 0.0, None)
    B = xp.clip(b * p["blue_survive"] * expo, 0.0, None)

    # Orange base cast: the film base passes red, most visible where the scene is
    # dark, so shadows go warm rather than black. Weight by (1 - luma).
    luma = xp.clip(0.2126 * R + 0.7152 * G + 0.0722 * B, 0.0, 1.0)
    cast = p["base_warm"] * (1.0 - luma)
    R = R + cast
    G = G + cast * 0.35

    out = xp.stack([R, G, B], axis=-1)
    out = linear_to_srgb(out)
    out = _contrast(out, p["contrast"])
    bl = p["black_lift"]
    out = bl + (1.0 - bl) * out
    return xp.clip(out, 0.0, 1.0)
