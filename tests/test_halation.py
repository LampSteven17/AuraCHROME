"""Halation/bloom spatial pass."""
import numpy as np

from aerochrome import halation


def test_off_is_identity():
    img = np.random.default_rng(0).random((16, 16, 3))
    out = halation.halation(img, strength=0.0)
    assert np.allclose(out, np.clip(img, 0, 1))


def test_bloom_spreads_from_highlight():
    # a dark field with one bright spot in the center
    img = np.full((33, 33, 3), 0.05)
    img[16, 16] = 1.0
    out = halation.halation(img, strength=1.0, size=4.0, threshold=0.5)
    # a neighbour that was dark should be lifted by the glow
    assert out[16, 19].mean() > img[16, 19].mean() + 1e-3
    # and the glow is warm (red >= blue) with the default tint
    assert out[16, 19, 0] >= out[16, 19, 2]


def test_no_bloom_below_threshold():
    img = np.full((16, 16, 3), 0.3)  # all below threshold
    out = halation.halation(img, strength=1.0, size=3.0, threshold=0.7)
    assert np.allclose(out, img, atol=1e-6)
