"""HIE / mono-IR family."""
import numpy as np

from aerochrome import monoir, stocks


def test_registered_as_mono_ir():
    s = stocks.get("hie")
    assert s.family == "mono_ir" and s.needs_ir is True
    assert "hie" not in stocks.family_names("aerochrome")   # not in --preset all


def test_output_is_monochrome():
    img = np.random.default_rng(0).random((24, 32, 3))
    nir = np.random.default_rng(1).random((24, 32))
    out = monoir.apply(img, nir=nir)
    assert out.shape == img.shape
    # R == G == B everywhere (neutral B&W tone)
    assert np.allclose(out[..., 0], out[..., 1]) and np.allclose(out[..., 1], out[..., 2])


def test_wood_effect_high_nir_is_bright():
    img = np.full((8, 8, 3), 0.3)
    nir = np.zeros((8, 8))
    nir[:, :4] = 0.9   # IR-bright on the left (foliage), dark on the right (sky)
    out = monoir.apply(img, nir=nir)
    assert out[:, :4].mean() > out[:, 4:].mean() + 0.2   # left glows, right inky


def test_grvi_fallback_without_nir():
    # foliage-ish (g>r) should read brighter than sky-ish (b>g>r) with no NIR
    foliage = np.zeros((4, 4, 3)); foliage[..., 1] = 0.6; foliage[..., 0] = 0.2
    sky = np.zeros((4, 4, 3)); sky[..., 2] = 0.6; sky[..., 1] = 0.3
    of = monoir.apply(foliage, nir=None)
    os_ = monoir.apply(sky, nir=None)
    assert of.mean() > os_.mean()
