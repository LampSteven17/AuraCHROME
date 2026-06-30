"""
Spectral rendering primitives -- integrate reflectance spectra to a camera RGB
and to an infrared film band:

    value = integral( reflectance(lambda) * illuminant(lambda) * response(lambda) )

Used to SYNTHESIZE paired (RGB, NIR) training data from modeled/measured spectra
(scripts/synth_spectra.py) and to render hyperspectral cubes
(scripts/ingest_hyperspectral.py).

Physics notes from the research sweep:
 - RGB integrates the CIE color-matching functions -- defined on the VISIBLE band
   only (~360-830nm). RGB carries no measurement of NIR (metameric-black null
   space); that is why NIR must be predicted, not inverted.
 - The IR film band integrates a silicon-QE x longpass response (~720-1000nm).
 - The illuminant MUST reach NIR. CIE D65 stops at 830nm, so the default here is
   an analytic Planckian blackbody (valid at any wavelength).

Plain NumPy (these run on host spectra, not image tensors).
"""

import numpy as np

# --- CIE 1931 2-deg CMFs, Wyman/Sloan/Shirley (JCGT 2013) gaussian fit ---------
def _g(x, mu, s1, s2):
    s = np.where(x < mu, s1, s2)
    return np.exp(-0.5 * ((x - mu) / s) ** 2)


def cie_xyz_bars(wl):
    """CIE 1931 2-deg xbar/ybar/zbar at wavelengths `wl` (nm)."""
    x = (1.056 * _g(wl, 599.8, 37.9, 31.0)
         + 0.362 * _g(wl, 442.0, 16.0, 26.7)
         - 0.065 * _g(wl, 501.1, 20.4, 26.2))
    y = (0.821 * _g(wl, 568.8, 46.9, 40.5)
         + 0.286 * _g(wl, 530.9, 16.3, 31.1))
    z = (1.217 * _g(wl, 437.0, 11.8, 36.0)
         + 0.681 * _g(wl, 459.0, 26.0, 13.8))
    return x, y, z


_XYZ_TO_SRGB = np.array([
    [3.2406255, -1.5372080, -0.4986286],
    [-0.9689307, 1.8757561, 0.0415175],
    [0.0557101, -0.2040211, 1.0569959],
])


def linear_to_srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def xyz_to_srgb(xyz):
    return linear_to_srgb(xyz @ _XYZ_TO_SRGB.T)


# --- illuminants ---------------------------------------------------------------
def blackbody(wl, temp=5500.0):
    """Planckian SPD at `wl` (nm), peak-normalized. Analytic -> any range."""
    lam = wl * 1e-9
    c1, c2 = 3.7418e-16, 1.4388e-2
    m = c1 / (lam ** 5 * (np.exp(c2 / (lam * temp)) - 1.0))
    return m / (m.max() + 1e-30)


# --- infrared film band --------------------------------------------------------
def ir_band_response(wl, cut=720.0, rise=18.0, qe_lo=950.0, qe_hi=1100.0):
    """Silicon-QE x R72-style longpass: rises near `cut`, rolls off to ~0 by
    `qe_hi`. Approximates an Aerochrome/EIR-ish ~720-1000nm passband."""
    longpass = 1.0 / (1.0 + np.exp(-(wl - cut) / rise))
    rolloff = np.clip((qe_hi - wl) / (qe_hi - qe_lo), 0.0, 1.0)
    return longpass * rolloff


# --- render --------------------------------------------------------------------
def render(spectra, wl, illuminant=None, ir_kw=None):
    """Render reflectance spectra to (sRGB, NIR).

    spectra: (..., W) reflectance in [0,1] on grid `wl` (nm, ascending).
    illuminant: SPD (W,) or None -> blackbody(5500K). Use np.ones_like(wl) for a
    flat equal-energy illuminant E (e.g. rendering a radiance cube).
    Returns rgb (...,3) and nir (...), both normalized so reflectance==1 -> 1."""
    spectra = np.asarray(spectra, dtype=np.float64)
    wl = np.asarray(wl, dtype=np.float64)
    illum = blackbody(wl) if illuminant is None else np.asarray(illuminant, np.float64)
    dlam = float(np.mean(np.diff(wl))) if wl.size > 1 else 1.0

    vis = (wl >= 380) & (wl <= 730)
    xb, yb, zb = cie_xyz_bars(wl[vis])
    Ev = illum[vis]
    white = np.sum(yb * Ev) * dlam  # so a perfect white -> Y = 1
    sv = spectra[..., vis]
    X = (sv * (xb * Ev)).sum(-1) * dlam / white
    Y = (sv * (yb * Ev)).sum(-1) * dlam / white
    Z = (sv * (zb * Ev)).sum(-1) * dlam / white
    rgb = xyz_to_srgb(np.stack([X, Y, Z], -1))

    irb = ir_band_response(wl, **(ir_kw or {}))
    nir = (spectra * (irb * illum)).sum(-1) / ((irb * illum).sum() + 1e-30)
    return np.clip(rgb, 0.0, 1.0), np.clip(nir, 0.0, 1.0)
