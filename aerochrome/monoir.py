"""
Mono-IR family -- black & white infrared stocks (Kodak HIE and friends).

Unlike the Aerochrome false-color path, a B&W IR stock maps the infrared signal
to LUMINANCE: IR-bright foliage glows near-white, IR-dark skies/water go near
black (the "Wood effect"), skin gets an ethereal mid glow. We drive that from the
predicted NIR channel when available, with a GRVI vegetation proxy as the
fallback (same honest-fallback principle as the false-color path).

The HIE signature halation glow + coarse grain are SPATIAL passes applied by the
export route (halation.py / grain.py) via the stock's params -- not here.

Display-space, per-pixel, runs on numpy or cupy.
"""

from .backend import xp_for

EPS = 1e-12

# Strong contrast, dominant IR, heavy warm halation, coarse luminance grain.
DEFAULT = dict(
    ir_weight=0.82,        # how much the IR channel dominates the B&W luminance
    veg_gain=1.6,          # GRVI-fallback IR gain on vegetation (no neural NIR)
    ir_base=0.18,          # fallback IR floor (luminance reflects some IR)
    black_point=0.05,      # crush shadows -> inky skies
    contrast=0.38,         # high-contrast HIE tone
    gamma=0.95,
    tone_tint=(1.0, 1.0, 1.0),   # neutral B&W (warm it for a printed look)
    # spatial passes (consumed by convert): big halation, coarse mono grain
    halation_strength=0.55,
    halation_size=7.0,
    halation_threshold=0.60,
    halation_tint=(1.0, 0.5, 0.32),
    grain_strength=0.05,
    grain_size=1.7,
    grain_chroma=0.0,      # pure luminance grain (B&W)
)


def apply(rgb, p=None, nir=None):
    """Render a B&W infrared frame from display sRGB `rgb` (..,3) in [0,1].

    `nir` (..) in [0,1] is the predicted NIR channel; when None a GRVI proxy
    synthesizes the IR term. Returns a monochrome sRGB image, same shape/device."""
    if p is None:
        p = DEFAULT
    xp = xp_for(rgb)
    rgb = xp.asarray(rgb, dtype=xp.float64)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    vis = 0.2126 * r + 0.7152 * g + 0.0722 * b

    if nir is not None:
        ir = xp.clip(xp.asarray(nir, dtype=xp.float64), 0.0, 1.0)
    else:  # GRVI fallback: foliage (g>r) gets synthetic IR, sky/non-veg stays low
        grvi = (g - r) / (g + r + EPS)
        veg = xp.clip(grvi / 0.4, 0.0, 1.0) ** 0.85
        ir = xp.clip(vis * (p["ir_base"] + p["veg_gain"] * veg), 0.0, 1.0)

    L = p["ir_weight"] * ir + (1.0 - p["ir_weight"]) * vis
    bp = p["black_point"]
    L = xp.clip((L - bp) / (1.0 - bp + EPS), 0.0, 1.0)
    s = L * L * (3.0 - 2.0 * L)                         # S-curve
    L = (1.0 - p["contrast"]) * L + p["contrast"] * s
    L = xp.clip(L, 0.0, 1.0) ** p["gamma"]

    out = L[..., None] * xp.asarray(p["tone_tint"], dtype=xp.float64)
    return xp.clip(out, 0.0, 1.0)
