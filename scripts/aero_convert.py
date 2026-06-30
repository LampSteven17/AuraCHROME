#!/usr/bin/env python3
"""
Aerochrome converter -- the fidelity-first export route.

    shot RAW  ->  [this]  ->  16-bit TIFF  ->  Lightroom  ->  Photoshop

Applies the exact Aerochrome transform (the real color science -- no Lightroom
profile approximation, no HSV-table banding) plus an optional chromatic film
grain pass, and writes lossless 16-bit TIFFs that flow straight into LR/PS.

Why 16-bit TIFF (and not "RAW"): a camera RAW is undemosaiced sensor data; once
the look is applied we have real RGB pixels, so there is no sensor mosaic to
write back. 16-bit TIFF is the highest-fidelity, non-lossy interchange format --
65k levels/channel, read by everything. (A "linear DNG" would just be this TIFF
in a RAW wrapper, with no benefit here.)

Run it two ways:
    python scripts/aero_convert.py                 # interactive TUI
    python scripts/aero_convert.py -i DIR -o OUT --preset classic   # scripted
"""
import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aerochrome import (  # noqa: E402
    backend, grain as _grain, halation as _halation, neural, params, stocks, transform)

RAW_EXTS = {".arw", ".cr2", ".cr3", ".nef", ".raf", ".rw2", ".dng", ".orf", ".raw"}
IMG_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
STOCK_NAMES = stocks.names()  # aerochrome looks + redscale

# Grain amount -> multiplier on each preset's tuned grain_strength. "standard"
# keeps the per-preset look exactly as authored; the others scale around it.
GRAIN_SCALE = {"off": 0.0, "subtle": 0.6, "standard": 1.0, "heavy": 1.6}

# ---- ANSI niceties (degrade gracefully if not a TTY) ----------------------
def _c(code, s):
    return f"\033[{code}m{s}\033[0m" if sys.stdout.isatty() else s

BOLD = lambda s: _c("1", s)
DIM = lambda s: _c("2", s)
CYAN = lambda s: _c("36", s)
GREEN = lambda s: _c("32", s)


# ---- IO -------------------------------------------------------------------
def load_image(path):
    """Return a display-referred sRGB float image in [0,1], (H,W,3)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in RAW_EXTS:
        import rawpy
        with rawpy.imread(path) as raw:
            rgb16 = raw.postprocess(use_camera_wb=True, output_bps=16,
                                    no_auto_bright=False)
        return rgb16.astype(np.float64) / 65535.0
    # already-rendered image (e.g. a TIFF exported from Lightroom)
    try:
        import tifffile
        if ext in (".tif", ".tiff"):
            a = tifffile.imread(path)
            a = np.asarray(a)
            if a.ndim == 2:
                a = np.stack([a] * 3, -1)
            a = a[..., :3]
            maxv = 65535.0 if a.dtype == np.uint16 else 255.0
            return a.astype(np.float64) / maxv
    except Exception:
        pass
    from PIL import Image
    im = Image.open(path).convert("RGB")
    return np.asarray(im).astype(np.float64) / 255.0


def save_tiff16(path, rgb01):
    """Write a lossless 16-bit RGB TIFF, tagged sRGB where possible."""
    import tifffile
    img = (np.clip(rgb01, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
    icc = None
    try:
        from PIL import ImageCms
        icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    except Exception:
        icc = None
    try:
        kw = dict(photometric="rgb", compression="deflate")
        if icc:
            kw["extratags"] = [(34675, 1, len(icc), icc, True)]  # ICCProfile
        tifffile.imwrite(path, img, **kw)
    except Exception:
        tifffile.imwrite(path, img, photometric="rgb")


def save_jpeg(path, rgb01, quality=95):
    """Write an 8-bit sRGB JPEG -- a quick-share / proof export beside the TIFF.
    Quality 95 + no chroma subsampling keeps the false-color edges clean."""
    from PIL import Image
    img = (np.clip(rgb01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    im = Image.fromarray(img, "RGB")
    icc = None
    try:
        from PIL import ImageCms
        icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    except Exception:
        icc = None
    im.save(path, "JPEG", quality=quality, subsampling=0, icc_profile=icc)


# format -> (extension, writer). "both" writes a TIFF and a JPEG per image.
FORMATS = {
    "tiff16": [(".tif", save_tiff16)],
    "jpeg":   [(".jpg", save_jpeg)],
    "both":   [(".tif", save_tiff16), (".jpg", save_jpeg)],
}


def write_outputs(outdir, name, rgb01, fmt):
    """Write `rgb01` under `outdir/name` in the requested format(s); return the
    primary path (TIFF when present, else the JPEG) for progress reporting."""
    paths = []
    for ext, writer in FORMATS[fmt]:
        dst = os.path.join(outdir, name + ext)
        writer(dst, rgb01)
        paths.append(dst)
    return paths[0]


# ---- core -----------------------------------------------------------------
def _transform_chunked(arr, p, nir=None, chunk_px=2_000_000):
    """Apply the transform in flat pixel chunks to bound (CPU or GPU) memory.

    Chunking matters on the GPU too: the transform allocates many temporaries, so
    a whole 33 MP frame at once could exhaust VRAM. `nir` (full-frame H×W) is
    flattened and sliced in lockstep so each chunk gets its matching IR pixels."""
    xp = backend.xp_for(arr)
    h, w, _ = arr.shape
    flat = arr.reshape(-1, 3)
    nflat = None if nir is None else xp.asarray(nir).reshape(-1)
    out = xp.empty_like(flat)
    for i in range(0, flat.shape[0], chunk_px):
        nchunk = None if nflat is None else nflat[i:i + chunk_px]
        out[i:i + chunk_px] = transform.aerochrome(flat[i:i + chunk_px], p, nir=nchunk)
    return out.reshape(h, w, 3)


def convert(arr, preset, add_grain=True, seed=0, use_neural=None, grain_scale=1.0):
    """Transform + optional grain. Runs on whichever device owns `arr`.

    When the learned RGB->NIR model is available (torch + CUDA + weights), it
    predicts a NIR channel that drives the look as the default IR source; absent
    that, the per-pixel GRVI index synthesizes IR (automatic fallback).

    `grain_scale` multiplies the preset's tuned grain strength (e.g. 0.6 subtle,
    1.0 standard, 1.6 heavy); add_grain=False disables the pass entirely.

    `preset` is a stock name (see aerochrome.stocks): the aerochrome family runs
    the IR transform (neural NIR / GRVI); non-IR families (redscale) render
    parametrically and skip the neural net entirely."""
    stock = stocks.get(preset)
    p = stock.params
    nir = None
    if stock.needs_ir:
        if use_neural is None:
            use_neural = neural.is_available()
        if use_neural:
            nir = neural.predict_nir(arr)
    if stock.family == "aerochrome":
        out = _transform_chunked(arr, p, nir=nir)        # memory-aware chunking
    else:  # mono_ir uses nir; redscale ignores it; both render per-pixel
        out = stocks.render(arr, stock, nir=nir)
    # halation/bloom (HIE/Efke-AURA); off for stocks with halation_strength 0
    hs = p.get("halation_strength", 0.0)
    if hs > 0:
        out = _halation.halation(out, hs, p.get("halation_size", 6.0),
                                 p.get("halation_threshold", 0.72),
                                 tuple(p.get("halation_tint", _halation.DEFAULT_TINT)))
    strength = p.get("grain_strength", 0) * grain_scale
    if add_grain and strength > 0:
        out = _grain.chromatic_grain(out, strength, p["grain_size"],
                                     p.get("grain_chroma", 0.35), seed=seed)
    return out


def gather_inputs(path):
    if os.path.isfile(path):
        return [path]
    files = []
    for f in sorted(glob.glob(os.path.join(path, "*"))):
        if os.path.splitext(f)[1].lower() in (RAW_EXTS | IMG_EXTS):
            files.append(f)
    return files


def _maybe_downscale(arr, longedge):
    if not longedge:
        return arr
    from PIL import Image
    im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    w, h = im.size
    s = longedge / max(w, h)
    if s < 1.0:
        im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
        arr = np.asarray(im).astype(np.float64) / 255.0
    return arr


def process_one(src, n, outdir, presets, add_grain, longedge, use_gpu,
                use_neural=False, fmt="tiff16", grain_scale=1.0):
    """Convert one input file across all requested presets. Returns
    (source_basename, dims, [(preset, dst_path), ...]). Top-level so it is
    picklable for the multiprocessing path. `use_neural` is forced False in
    forked CPU workers (CUDA + multiprocessing don't mix); the neural NIR runs
    only on the serial path. `fmt` selects the output format(s) (see FORMATS)."""
    arr = _maybe_downscale(load_image(src), longedge)
    dev = backend.to_device(arr, use_gpu)
    stem = os.path.splitext(os.path.basename(src))[0]
    results = []
    for idx, pre in enumerate(presets):
        out = backend.to_numpy(convert(dev, pre, add_grain=add_grain,
                                       seed=n * 101 + idx, use_neural=use_neural,
                                       grain_scale=grain_scale))
        name = stem if len(presets) == 1 else f"{pre}-{stem}"
        dst = write_outputs(outdir, name, out, fmt)
        results.append((pre, dst))
    return stem, (arr.shape[1], arr.shape[0]), results


def run(inputs, outdir, presets, add_grain, longedge=0, use_gpu=False, jobs=1,
        progress_json=False, fmt="tiff16", no_neural=False, gpu_requested=False,
        grain_scale=1.0):
    os.makedirs(outdir, exist_ok=True)
    total = len(inputs) * len(presets)
    done = 0
    serial = use_gpu or jobs <= 1 or len(inputs) <= 1
    mode = backend.device_name() if use_gpu else (
        f"cpu x{jobs}" if not serial else "cpu (serial)")
    any_ir = any(stocks.get(pre).needs_ir for pre in presets)
    neural_on = any_ir and serial and not no_neural and neural.is_available()
    ir_src = ("neural NIR" if neural_on else "GRVI index") if any_ir else "n/a"

    # Surface a silent GPU fallback so the front-end can explain it.
    warning = None
    if gpu_requested and not use_gpu:
        warning = ("GPU requested but the CUDA backend is unavailable — "
                   "rendering on CPU. Install it with: poetry install --with gpu")

    def emit(obj):
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()

    if progress_json:
        start = {"event": "start", "total": total, "files": len(inputs),
                 "looks": len(presets), "device": mode, "ir": ir_src,
                 "format": fmt, "outdir": outdir}
        if warning:
            start["warning"] = warning
        emit(start)
    else:
        print(DIM(f"device: {mode}   |   IR: {ir_src}   |   format: {fmt}   |   "
                  f"{len(inputs)} file(s) x {len(presets)} look(s)\n"))

    def report(stem, dims, results):
        nonlocal done
        if not progress_json:
            print(DIM(f"  {stem}  {dims[0]}x{dims[1]}"))
        for pre, dst in results:
            done += 1
            if progress_json:
                emit({"event": "image", "done": done, "total": total,
                      "file": os.path.basename(dst), "preset": pre,
                      "w": dims[0], "h": dims[1]})
            else:
                print(GREEN(f"    ✓ {pre:9s} -> {os.path.basename(dst)}"))
        if not progress_json:
            print(DIM(f"    ({done}/{total})"))

    if serial:
        for n, src in enumerate(inputs, start=1):
            stem, dims, results = process_one(
                src, n, outdir, presets, add_grain, longedge, use_gpu,
                neural_on, fmt, grain_scale)
            report(stem, dims, results)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(process_one, src, n, outdir, presets,
                              add_grain, longedge, False, False, fmt, grain_scale): n
                    for n, src in enumerate(inputs, start=1)}
            for fut in as_completed(futs):
                stem, dims, results = fut.result()
                report(stem, dims, results)

    if progress_json:
        emit({"event": "done", "total": total, "outdir": outdir})
    else:
        print(BOLD(GREEN(f"\nDone. {total} TIFF(s) written to {outdir}")))


# ---- interactive TUI ------------------------------------------------------
def _ask(prompt, default=None):
    sfx = f" [{default}]" if default is not None else ""
    val = input(CYAN(f"{prompt}{sfx}: ")).strip()
    return val or (default or "")


def _menu(prompt, options, default_idx=0):
    print(BOLD(prompt))
    for i, o in enumerate(options, 1):
        mark = DIM(" (default)") if i - 1 == default_idx else ""
        print(f"  {i}) {o}{mark}")
    raw = input(CYAN("choose #: ")).strip()
    if not raw:
        return default_idx
    try:
        k = int(raw) - 1
        return k if 0 <= k < len(options) else default_idx
    except ValueError:
        return default_idx


def tui():
    print(BOLD(CYAN("\n  AEROCHROME converter  ")) +
          DIM("RAW -> 16-bit TIFF -> Lightroom -> Photoshop\n"))
    while True:
        ipath = _ask("Input file or folder")
        ipath = os.path.expanduser(ipath.strip('"').strip("'"))
        inputs = gather_inputs(ipath) if ipath else []
        if inputs:
            break
        print(DIM("  no RAW/image files found there -- try again.\n"))
    print(DIM(f"  found {len(inputs)} file(s)\n"))

    aero_looks = stocks.family_names("aerochrome")
    pidx = _menu("Look:", aero_looks + ["all four"], default_idx=0)
    presets = aero_looks if pidx == len(aero_looks) else [aero_looks[pidx]]

    gidx = _menu("\nChromatic film grain:", ["on", "off"], default_idx=0)
    add_grain = (gidx == 0)

    default_out = os.path.join(
        ipath if os.path.isdir(ipath) else os.path.dirname(ipath), "aurachrome_out")
    outdir = os.path.expanduser(_ask("\nOutput folder", default_out).strip('"').strip("'"))

    print(BOLD("\nSummary:"))
    use_gpu = backend.gpu_available()
    jobs = 1 if use_gpu else max(1, min(os.cpu_count() or 1, 4))
    print(f"  inputs : {len(inputs)} file(s)")
    print(f"  looks  : {', '.join(presets)}")
    print(f"  grain  : {'on (chromatic)' if add_grain else 'off'}")
    print(f"  device : {backend.device_name() if use_gpu else f'cpu (x{jobs})'}")
    print(f"  output : {outdir}  (16-bit TIFF)")
    if _ask("\nProceed? (y/n)", "y").lower() not in ("y", "yes"):
        print(DIM("cancelled."))
        return
    print()
    run(inputs, outdir, presets, add_grain, use_gpu=use_gpu, jobs=jobs)


def main():
    ap = argparse.ArgumentParser(description="Aerochrome RAW -> 16-bit TIFF converter")
    ap.add_argument("-i", "--input", help="RAW/image file or a folder of them")
    ap.add_argument("-o", "--outdir", help="output folder")
    ap.add_argument("--preset", choices=STOCK_NAMES + ["all"], default="classic")
    ap.add_argument("--format", choices=list(FORMATS), default="tiff16",
                    help="output format: tiff16 (16-bit), jpeg, or both")
    ap.add_argument("--grain", choices=list(GRAIN_SCALE), default="standard",
                    help="grain amount: off | subtle | standard | heavy")
    ap.add_argument("--no-grain", action="store_true",
                    help="disable the grain pass (alias for --grain off)")
    ap.add_argument("--no-neural", action="store_true",
                    help="force the GRVI index even if the learned NIR model is available")
    ap.add_argument("--longedge", type=int, default=0,
                    help="downscale long edge to N px (0 = full resolution)")
    ap.add_argument("--gpu", action="store_true", help="force the CUDA (CuPy) backend")
    ap.add_argument("--cpu", action="store_true", help="force CPU even if a GPU exists")
    ap.add_argument("--jobs", type=int, default=0,
                    help="CPU worker processes for batches (0 = auto, ignored on GPU)")
    ap.add_argument("--progress-json", action="store_true",
                    help="emit machine-readable JSON progress (one object per line) for the TUI")
    ap.add_argument("--probe-gpu", action="store_true",
                    help="print {\"gpu\": bool, \"name\": str} and exit (used by the TUI)")
    args = ap.parse_args()

    if args.probe_gpu:
        print(json.dumps({"gpu": backend.gpu_available(), "name": backend.device_name()}))
        return

    if not args.input:
        tui()
        return
    inputs = gather_inputs(os.path.expanduser(args.input))
    if not inputs:
        sys.exit(f"no RAW/image files at {args.input}")
    presets = stocks.family_names("aerochrome") if args.preset == "all" else [args.preset]
    outdir = args.outdir or os.path.join(
        args.input if os.path.isdir(args.input) else os.path.dirname(args.input),
        "aurachrome_out")

    # device resolution: explicit flags win; otherwise auto-use a GPU if present
    if args.cpu:
        use_gpu = False
    elif args.gpu:
        use_gpu = backend.gpu_available()
        if not use_gpu and not args.progress_json:
            print(DIM("--gpu requested but the CUDA backend is unavailable; using CPU."))
    else:
        use_gpu = backend.gpu_available()
    jobs = args.jobs or (1 if use_gpu else max(1, min(os.cpu_count() or 1, 4)))

    grain_level = "off" if args.no_grain else args.grain
    grain_scale = GRAIN_SCALE[grain_level]

    run(inputs, outdir, presets, add_grain=grain_scale > 0, grain_scale=grain_scale,
        longedge=args.longedge, use_gpu=use_gpu, jobs=jobs,
        progress_json=args.progress_json, fmt=args.format,
        no_neural=args.no_neural, gpu_requested=args.gpu)


if __name__ == "__main__":
    main()
