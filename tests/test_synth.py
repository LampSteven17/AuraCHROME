"""Spectral primitives + synthetic spectra generator."""
import importlib.util
import os

import numpy as np

from aerochrome import spectral


def _load_synth():
    spec = importlib.util.spec_from_file_location(
        "synth_spectra",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "synth_spectra.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_render_white_is_unit():
    wl = np.arange(400.0, 1001.0, 5.0)
    rgb, nir = spectral.render(np.ones_like(wl), wl)
    # a perfect reflector renders near white and full NIR
    assert rgb.min() > 0.9 and nir > 0.9


def test_ir_band_is_nir_only():
    wl = np.arange(400.0, 1001.0, 5.0)
    irb = spectral.ir_band_response(wl)
    assert irb[wl < 650].max() < 0.05      # blocks visible
    assert irb[(wl > 760) & (wl < 900)].min() > 0.5  # passes NIR


def test_vegetation_has_higher_ndvi_than_soil():
    # average many samples: any single leaf may be senescent (low NDVI, correct
    # physics), but the foliage population's red-edge must beat bare soil.
    synth = _load_synth()
    rng = np.random.default_rng(3)

    def mean_ndvi(gen, n=200):
        vals = []
        for _ in range(n):
            refl = gen(rng)
            rgb, nir = spectral.render(refl, synth.WL)
            vals.append((nir - rgb[0]) / (nir + rgb[0] + 1e-9))
        return float(np.mean(vals))

    ndvi_leaf = mean_ndvi(synth.leaf_analytic)
    ndvi_soil = mean_ndvi(synth.soil_analytic)
    assert ndvi_leaf > 0.0                    # foliage red-edge -> positive NDVI
    assert ndvi_soil < 0.0                    # bare soil -> negative
    assert ndvi_leaf > ndvi_soil + 0.15       # clearly separated


def test_skin_km_nir_glow_and_tone_convergence():
    synth = _load_synth()
    WL = synth.WL

    def km(mel):  # deterministic KM skin at a melanin fraction (others mid)
        mua = mel * synth._mua_melanosome(WL) + 0.02 * synth._mua_blood(WL, 0.8) + 0.3
        musp = 45.0 * (WL / 500.0) ** -1.0
        ks = mua / musp
        return np.clip(1 + ks - np.sqrt(ks * ks + 2 * ks), 0, 1)

    light, dark = km(0.025), km(0.35)
    nir = lambda r: r[(WL >= 800) & (WL <= 950)].mean()
    vis = lambda r: r[(WL >= 550) & (WL <= 650)].mean()
    # waxy glow: NIR brighter than visible for every skin tone
    assert nir(light) > vis(light) and nir(dark) > vis(dark)
    # skin tone separates strongly in visible but CONVERGES in NIR (relative spread)
    assert (vis(light) / vis(dark)) > (nir(light) / nir(dark))
    # default skin() generator is smooth across the NIR
    s = synth.skin(np.random.default_rng(4))
    assert s[(WL >= 800) & (WL <= 950)].std() < 0.12


def test_sample_shapes():
    synth = _load_synth()
    rgb, nir, labels, spectra = synth.sample(
        50, {"leaf": 0.5, "skin": 0.25, "soil": 0.25}, seed=1)
    assert rgb.shape == (50, 3) and nir.shape == (50,)
    assert spectra.shape == (50, synth.WL.size)
    assert set(labels) <= {"leaf", "skin", "soil"}
