#!/usr/bin/env python3
"""
Synthesize paired (RGB, NIR-band) training data from MODELED reflectance spectra.

Why this is sound: you cannot invent NIR from RGB (metameric-black null space), but
you CAN start from a reflectance curve and render BOTH the RGB and the IR band by
integrating against an illuminant + sensor response (aerochrome.spectral). The
spectra come from parametric physical models, so the vegetation red-edge / NIR
plateau and skin's smooth NIR glow are physically grounded -- exactly the priors a
learned RGB->NIR mapping needs.

Generators (analytic fallbacks run with zero extra deps; real models used when
installed):
  - vegetation: PROSPECT/PROSAIL via the `prosail` package if importable, else a
    physically-SHAPED analytic leaf (green bump, red absorption, red-edge, NIR
    plateau, ~975nm water dip). Param: chlorophyll, brown pigment, water, N.
  - skin: a physical two-chromophore Kubelka-Munk model (Jacques 2013 optics:
    melanin power-law absorption + oxy/deoxy haemoglobin + tissue scattering),
    parameterized by Fitzpatrick melanin fraction. (Meta `BioSkin` can be wired in
    here later if installed.)
  - soil: a simple monotonic brightening curve (background filler).

Outputs per-material pairs to an .npz (rgb[N,3], nir[N], class[N]); optionally
composes small spatial RGB/NIR tiles (--tiles) named for the trainer (_rgb/_nir).

    python scripts/synth_spectra.py -o data/synth --n 20000
    python scripts/synth_spectra.py -o data/synth --n 4000 --tiles 64 --tile-size 128
"""
import argparse
import os
import sys

import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aerochrome import spectral  # noqa: E402

WL = np.arange(400.0, 1001.0, 5.0)  # 400-1000nm @ 5nm working grid

try:
    import prosail  # noqa: F401
    _HAS_PROSAIL = True
except Exception:
    _HAS_PROSAIL = False


# --------------------------------------------------------------------------
# analytic reflectance models (fallbacks; physically shaped, not measured)
# --------------------------------------------------------------------------
def _smooth(x, lo, hi):
    t = np.clip((x - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def leaf_analytic(rng):
    """A green leaf: low visible w/ green bump, red-edge ~700-750, NIR plateau."""
    cab = rng.uniform(10, 75)          # chlorophyll -> red/blue absorption depth
    brown = rng.uniform(0.0, 0.9)      # senescence -> fills the red-edge, dims NIR
    nir_lvl = rng.uniform(0.35, 0.55) * (1.0 - 0.4 * brown)
    edge = 700.0 + 25.0 * (1.0 - brown)  # healthy edge sits redder
    r = np.full_like(WL, 0.05)
    r += 0.12 * np.exp(-0.5 * ((WL - 550) / 25) ** 2) * (1 - cab / 90)  # green bump
    r += (nir_lvl - 0.05) * _smooth(WL, edge - 40, edge + 40)           # red-edge jump
    r -= 0.04 * np.exp(-0.5 * ((WL - 975) / 25) ** 2)                   # water dip
    r += 0.20 * brown * _smooth(WL, 560, 680)                          # brown reflectance
    return np.clip(r, 0.0, 1.0)


def skin_analytic(rng):
    """Shape-only skin fallback (kept for reference / no-physics smoke use)."""
    mel = rng.uniform(0.02, 0.43)
    base = 0.55 - 0.45 * mel
    r = base * _smooth(WL, 480, 760)
    r += 0.12 * np.exp(-0.5 * ((WL - 560) / 40) ** 2)
    r = np.maximum(r, 0.10 + 0.5 * base * _smooth(WL, 700, 850))
    r -= 0.05 * np.exp(-0.5 * ((WL - 975) / 25) ** 2)
    return np.clip(r, 0.0, 1.0)


# --- physical skin: Kubelka-Munk over melanin + haemoglobin (Jacques 2013) -----
def _gauss(x, mu, amp, sig):
    return amp * np.exp(-0.5 * ((x - mu) / sig) ** 2)


def _mua_melanosome(wl):
    """Melanosome absorption (cm^-1): the 1.70e12 * lambda^-3.48 power law -- the
    steep falloff is why melanin barely absorbs in NIR (-> skin's NIR glow)."""
    return 1.70e12 * wl ** -3.48


def _mua_blood(wl, oxy):
    """Whole-blood absorption (cm^-1): oxy/deoxy haemoglobin as Soret + Q bands,
    near-zero in NIR (a faint deoxy bump ~758nm)."""
    hbo2 = _gauss(wl, 415, 1.0, 10) + _gauss(wl, 542, 0.28, 12) + _gauss(wl, 576, 0.32, 12)
    hb = _gauss(wl, 430, 0.90, 14) + _gauss(wl, 556, 0.30, 20) + _gauss(wl, 758, 0.05, 30)
    return 350.0 * (oxy * hbo2 + (1.0 - oxy) * hb)


def skin_km(rng):
    """Diffuse skin reflectance via semi-infinite Kubelka-Munk on an effective
    medium of melanin + blood + baseline, with lambda^-b tissue scattering.
    melanin fraction spans Fitzpatrick I-VI; NIR is bright + tone-independent."""
    mel = rng.uniform(0.013, 0.43)     # epidermal melanosome volume fraction
    blood = rng.uniform(0.002, 0.05)   # dermal blood volume
    oxy = rng.uniform(0.6, 0.95)
    mus500 = rng.uniform(35.0, 55.0)
    b = rng.uniform(0.8, 1.2)
    water = rng.uniform(0.5, 0.7)

    mua = (mel * _mua_melanosome(WL)
           + blood * _mua_blood(WL, oxy)
           + water * _gauss(WL, 970, 0.6, 30)   # ~975nm water feature
           + 0.3)                                # baseline tissue absorption
    musp = mus500 * (WL / 500.0) ** (-b)
    ks = mua / musp                              # K/S
    refl = 1.0 + ks - np.sqrt(ks * ks + 2.0 * ks)  # semi-infinite KM reflectance
    return np.clip(refl, 0.0, 1.0)


def skin(rng):
    # (BioSkin decoder could be wired here when installed; KM physics by default)
    return skin_km(rng)


def soil_analytic(rng):
    b = rng.uniform(0.12, 0.45)
    return np.clip(b * _smooth(WL, 400, 1000) + 0.5 * b, 0.0, 1.0)


def leaf(rng):
    if _HAS_PROSAIL:
        import prosail
        lam, refl, _ = prosail.run_prospect(
            rng.uniform(1.0, 2.2), rng.uniform(10, 75), rng.uniform(2, 12),
            rng.uniform(0.0, 0.9), rng.uniform(0.002, 0.04),
            rng.uniform(0.003, 0.012), ant=rng.uniform(0, 5), prospect_version="D")
        return np.interp(WL, lam, refl)
    return leaf_analytic(rng)


GENERATORS = {"leaf": leaf, "skin": skin, "soil": soil_analytic}


# --------------------------------------------------------------------------
# sampling + rendering
# --------------------------------------------------------------------------
def sample(n, weights, seed, temp_range=(4500, 6500)):
    rng = np.random.default_rng(seed)
    classes = list(weights)
    probs = np.array([weights[c] for c in classes], float)
    probs /= probs.sum()
    spectra = np.empty((n, WL.size))
    labels = np.empty(n, dtype="U8")
    for i in range(n):
        c = classes[rng.choice(len(classes), p=probs)]
        spectra[i] = GENERATORS[c](rng)
        labels[i] = c
    # render each under a randomized daylight temperature (domain randomization)
    rgb = np.empty((n, 3))
    nir = np.empty(n)
    for i in range(n):
        illum = spectral.blackbody(WL, rng.uniform(*temp_range))
        rgb[i], nir[i] = spectral.render(spectra[i], WL, illuminant=illum)
    return rgb, nir, labels, spectra


def _to_u16(a):
    return np.round(np.clip(a, 0.0, 1.0) * 65535.0).astype(np.uint16)


def make_tiles(n_tiles, size, weights, seed, outdir):
    """Compose spatial RGB/NIR tiles by painting sampled spectra into random
    blobs (so the U-Net sees structure, not flat fields)."""
    rng = np.random.default_rng(seed + 1)
    os.makedirs(outdir, exist_ok=True)
    yy, xx = np.mgrid[0:size, 0:size]
    for t in range(n_tiles):
        rgb_t = np.zeros((size, size, 3))
        nir_t = np.zeros((size, size))
        wsum = np.zeros((size, size)) + 1e-6
        for _ in range(rng.integers(4, 9)):  # a few overlapping material blobs
            cls = sample(1, weights, int(rng.integers(1 << 30)))
            rgb1, nir1 = cls[0][0], cls[1][0]
            cx, cy = rng.uniform(0, size, 2)
            rad = rng.uniform(size * 0.12, size * 0.4)
            w = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * rad ** 2))
            rgb_t += w[..., None] * rgb1
            nir_t += w * nir1
            wsum += w
        rgb_t /= wsum[..., None]
        nir_t /= wsum
        tifffile.imwrite(os.path.join(outdir, f"synth{t:05d}_rgb.tif"), _to_u16(rgb_t))
        tifffile.imwrite(os.path.join(outdir, f"synth{t:05d}_nir.tif"), _to_u16(nir_t))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--outdir", required=True)
    ap.add_argument("--n", type=int, default=20000, help="number of point-spectra pairs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--w-leaf", type=float, default=0.55)
    ap.add_argument("--w-skin", type=float, default=0.20)
    ap.add_argument("--w-soil", type=float, default=0.25)
    ap.add_argument("--tiles", type=int, default=0, help="also write N spatial tiles")
    ap.add_argument("--tile-size", type=int, default=128)
    args = ap.parse_args()

    weights = {"leaf": args.w_leaf, "skin": args.w_skin, "soil": args.w_soil}
    os.makedirs(args.outdir, exist_ok=True)
    rgb, nir, labels, _ = sample(args.n, weights, args.seed)
    np.savez(os.path.join(args.outdir, "pairs.npz"), rgb=rgb, nir=nir, label=labels)

    # quick sanity: vegetation should show positive NDVI; non-veg negative.
    # Linearize the sRGB red so NDVI is computed in the same (linear) space as NIR.
    veg = labels == "leaf"
    rlin = np.where(rgb[:, 0] <= 0.04045, rgb[:, 0] / 12.92,
                    ((rgb[:, 0] + 0.055) / 1.055) ** 2.4)
    ndvi = (nir - rlin) / (nir + rlin + 1e-9)
    print(f"  {args.n} pairs  (prosail={_HAS_PROSAIL})  -> {args.outdir}/pairs.npz")
    print(f"  leaf mean NDVI = {ndvi[veg].mean():.3f}  "
          f"(skin {ndvi[labels=='skin'].mean():.3f}, soil {ndvi[labels=='soil'].mean():.3f})")
    if args.tiles:
        make_tiles(args.tiles, args.tile_size, weights, args.seed, args.outdir)
        print(f"  {args.tiles} spatial tiles ({args.tile_size}px) -> {args.outdir}/synth*_rgb.tif")


if __name__ == "__main__":
    main()
