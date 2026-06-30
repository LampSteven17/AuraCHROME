"""ENVI hyperspectral ingestion -> (RGB, NIR band) pairs, on a synthetic cube."""
import importlib.util
import os

import numpy as np
import tifffile

_SPEC = importlib.util.spec_from_file_location(
    "ingest_hyperspectral",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "scripts", "ingest_hyperspectral.py"))
ing = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ing)


def _write_envi(path, cube, wl):
    """Write a tiny BSQ float32 ENVI cube + header (cube: H,W,B)."""
    h, w, b = cube.shape
    raw = cube.transpose(2, 0, 1).astype("<f4")  # band, row, col
    raw.tofile(path + ".raw")
    hdr = (f"ENVI\nsamples = {w}\nlines = {h}\nbands = {b}\n"
           "header offset = 0\ndata type = 4\ninterleave = bsq\nbyte order = 0\n"
           "wavelength units = Nanometers\n"
           "wavelength = {" + ", ".join(f"{x:.1f}" for x in wl) + "}\n")
    with open(path + ".hdr", "w") as f:
        f.write(hdr)


def test_envi_roundtrip_and_bands(tmp_path):
    # cube 400..1000nm @ 20nm; make 850nm band bright in the left half (veg-like)
    wl = np.arange(400, 1001, 20.0)
    H, W, B = 12, 16, wl.size
    cube = np.full((H, W, B), 0.2, np.float32)
    b850 = int(np.argmin(np.abs(wl - 850)))
    cube[:, : W // 2, b850] = 0.9              # bright NIR on the left
    bgreen = int(np.argmin(np.abs(wl - 550)))
    cube[:, :, bgreen] = 0.6                    # a little green so RGB isn't flat

    src = str(tmp_path / "scene")
    _write_envi(src, cube, wl)

    out = str(tmp_path / "out")
    stem, lo, hi, have, skip = ing.ingest(src + ".hdr", out, [740, 850, 900],
                                          window=15.0, pct=99.0)
    assert (lo, hi) == (400, 1000) and have == [740, 850, 900] and skip == []

    rgb = tifffile.imread(os.path.join(out, "scene_rgb.tif"))
    nir = tifffile.imread(os.path.join(out, "scene_nir.tif"))      # primary = 740
    nir850 = tifffile.imread(os.path.join(out, "scene_nir_850.tif"))
    assert rgb.shape == (H, W, 3) and nir.shape == (H, W)
    assert os.path.exists(os.path.join(out, "scene_nir_900.tif"))

    # the 850nm slice must be brighter on the left (where we set high NIR)
    assert nir850[:, : W // 2].mean() > nir850[:, W // 2:].mean()


def test_out_of_range_band_skipped(tmp_path):
    wl = np.arange(400, 701, 20.0)  # visible-only cube: 900nm is unreachable
    cube = np.full((8, 8, wl.size), 0.3, np.float32)
    src = str(tmp_path / "vis")
    _write_envi(src, cube, wl)
    _, _, _, have, skip = ing.ingest(src + ".hdr", str(tmp_path / "o"),
                                     [740, 900], window=15.0, pct=99.0)
    assert have == [] and skip == [740, 900]  # nothing within 15nm of the cube
