"""Stock-profile abstraction + redscale (parametric, no neural)."""
import numpy as np

from aerochrome import params, stocks, transform


def test_registry():
    names = stocks.names()
    assert set(["classic", "punchy", "muted", "portrait", "redscale"]) <= set(names)
    assert stocks.family_names("aerochrome") == ["classic", "punchy", "muted", "portrait"]
    assert "redscale" not in stocks.family_names("aerochrome")  # excluded from --preset all


def test_needs_ir():
    assert stocks.get("classic").needs_ir is True
    assert stocks.get("redscale").needs_ir is False


def test_aerochrome_dispatch_unchanged():
    # rendering through the stock layer must equal calling the transform directly
    img = np.random.default_rng(1).random((40, 60, 3))
    via_stock = stocks.render(img, "classic", nir=None)
    direct = transform.aerochrome(img, params.get("classic"), nir=None)
    assert np.allclose(via_stock, direct)


def test_redscale_is_red_dominant():
    img = np.random.default_rng(2).random((40, 60, 3))
    out = stocks.render(img, "redscale")
    assert out.shape == img.shape
    assert out.min() >= 0.0 and out.max() <= 1.0
    r, g, b = out[..., 0].mean(), out[..., 1].mean(), out[..., 2].mean()
    assert r > g > b  # warm red->orange, crushed blue


def test_redscale_exposure_lifts_green():
    img = np.full((8, 8, 3), 0.5)
    from aerochrome import redscale
    low = redscale.apply(img, {**redscale.DEFAULT, "exposure": 0.6})
    high = redscale.apply(img, {**redscale.DEFAULT, "exposure": 1.6})
    # more exposure drives more green/yellow survival
    assert high[..., 1].mean() > low[..., 1].mean()
