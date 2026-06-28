# Aurachrome — false-color film engine

**Aurachrome** is a Python engine that renders the Kodak Aerochrome / EIR
color-infrared "false color" look from camera RAW to lossless 16-bit TIFF (CPU or
CUDA GPU), for a **Sony A7C II** stills workflow into Lightroom → Photoshop. It
also still emits a portable 3D `.cube` LUT and an experimental Lightroom profile
(both deprecated in favor of the RAW→TIFF engine).

This is a **perceptual approximation, not a physical one.** "Close, not perfect."

## Quick start

```bash
make setup      # install Poetry (if missing) + deps, build the TUI
make install    # put `aurachrome-tui` on your PATH (~/.local/bin)
make doctor     # verify python / poetry / go / GPU
make run        # launch the TUI

make convert ARGS="-i RAWS/ -o OUT/ --preset classic --gpu"   # headless batch
```

`make deps` auto-adds the CUDA group when an NVIDIA GPU is present (force with
`make gpu`, skip with `GPU=0`). Run `make help` for all targets. Manual
(Poetry/Go) details below.

## Install & run (Poetry)

Dependencies are managed with [Poetry](https://python-poetry.org/) in an isolated
virtualenv (keep this off your global/conda env):

```bash
poetry install                 # core CPU stack
poetry install --with gpu      # + CUDA acceleration (NVIDIA GPU; CUDA 12.x)
poetry install --with dev      # + pytest
poetry shell                   # activate the env (or prefix commands with `poetry run`)
```

The GPU group pulls CuPy plus the NVIDIA CUDA runtime wheels (incl. `libnvrtc`),
so **no system-wide CUDA toolkit is needed** — only an NVIDIA driver.

### The converter (primary workflow)

The fidelity-first pipeline is **shot RAW → converter → 16-bit TIFF → Lightroom →
Photoshop**. The converter applies the exact transform (no Lightroom-profile
approximation, no look-table banding) plus optional chromatic film grain, and
writes lossless 16-bit TIFFs (sRGB).

```bash
poetry run aero-convert                         # interactive TUI
poetry run aero-convert -i RAWS/ -o OUT/ --preset classic        # batch
poetry run aero-convert -i RAWS/ -o OUT/ --preset all --gpu      # all looks, GPU
poetry run aero-convert -i RAWS/ -o OUT/ --no-grain --jobs 8     # CPU, 8 workers
```

Why 16-bit TIFF and not "RAW out": a camera RAW is undemosaiced sensor data;
once the look is applied you have real RGB pixels, so there is no mosaic to write
back. 16-bit TIFF is the lossless, universal interchange for LR/PS.

### TUI (Go / Bubble Tea)

A terminal wizard front-end lives in `tui/` (its own Go module, Charm stack). It
collects input/output/look/grain/device, then drives the Python engine as a
subprocess and renders its JSON progress live. The engine and TUI stay decoupled:
the contract is the `--progress-json` line stream.

```bash
go build -o aurachrome-tui ./tui      # from the repo root (Go 1.25+)
./aurachrome-tui                       # needs `poetry` + the engine installed
```

Overrides: `AURACHROME_ENGINE="poetry run aurachrome"` (the command) and
`AURACHROME_REPO=/path/to/repo` (working dir). Tests: `go test ./tui` (headless
state-machine smoke test; a live bridge test runs under `AURA_ENGINE_TEST=1`).

## GPU acceleration

The per-pixel transform is ~77% of the runtime and is pure elementwise math, so
it maps almost perfectly to a GPU. `aerochrome/backend.py` dispatches each array
to NumPy (CPU) or CuPy (CUDA) via `cupy.get_array_module` — **one codepath**, and
it silently stays on CPU if no GPU/CuPy is present.

- `--gpu` / `--cpu` force the backend; default auto-detects.
- `--jobs N` fans a CPU batch across worker processes (auto on multi-core).
- Measured (RTX 3090 Ti, full-res 33 MP, float64): transform+grain **35.2s → 3.2s
  (~11×)**; full 6-image batch (incl. RAW decode + TIFF write) ~27s.
- float64 is kept for fidelity; consumer cards run it slower than float32, so the
  win is real but smaller than float32 would give.

## The one constraint that defines the project

A normal A7C II has an IR-cut filter, so the files contain **no infrared
channel** — only visible RGB. The Aerochrome look is *driven* by near-IR
reflectance. So the output is a pure per-pixel function `(R,G,B) → (R,G,B)`.

Two surfaces with identical visible RGB (a live leaf vs. a green-painted wall)
**cannot** be separated — that's information theory, not a bug. We synthesize a
fake IR channel from a visible vegetation index and lean on the fact that, in
real scenes, "looks like a plant" correlates with "IR-bright." Everything here
is bakeable per-pixel; no spatial ops (those would break the pure-LUT
requirement).

## How it works (the color science)

Real color-infrared film does a **channel rotation**, blue discarded by a yellow
filter:

```
RED_out   <- INFRARED   (IR-bright live vegetation -> red)
GREEN_out <- RED        (red objects -> green)
BLUE_out  <- GREEN      (green objects -> blue)
```

We have RGB but no IR, so IR is the only *synthesized* term — built from `GRVI =
(G-R)/(G+R)`, gated by `GBI = (G-B)/(G+B)` to reject blue/cyan sky.

This implementation is **re-architected** off the reference baseline for color
correctness:

- The mechanical rotation and foliage/sky shaping run in **linear light**.
- The corrective grade (skin / neutral / highlight) runs in **OKLab/OKLCh**,
  where "warm low-chroma skin" and "near-neutral concrete" are clean, separable
  windows (a hue arc + a chroma threshold) instead of overlapping sRGB ratio
  hacks. This is what fixed the original baseline's *skin-too-emerald* and
  *neutrals-drift-teal* problems at the root.

See `aerochrome/transform.py` for the annotated pipeline.

## Layout

```
aerochrome/
  encodings.py   sRGB<->linear, OKLab/OKLCh, S-Log3/S-Gamut3.Cine
  transform.py   the core (R,G,B)->(R,G,B) function
  params.py      presets: classic / punchy / muted
  cube.py        write/read .cube (17/33/65) + trilinear apply
scripts/
  make_cube.py   generate .cube files into luts/
  apply_lut.py   apply a .cube (or the transform) to an image
  preview.py     swatch grid + procedural scene, before/after PNG
  tune.py        parameter sweep -> contact sheet
tests/
  test_swatches.py  asserts expected hue family per surface (handoff §6)
luts/            generated .cube + preview/ PNGs
```

## Quickstart

```bash
pip install numpy pillow tifffile   # only deps (tifffile = true 16-bit TIFF)

python scripts/make_cube.py                  # -> luts/Aerochrome_Classic_Display_{33,65}.cube
python scripts/preview.py                     # -> luts/preview/swatches.png, scene.png
python scripts/apply_lut.py my.tif out.tif    # apply default 65pt LUT to one photo
python scripts/apply_lut.py --indir in/ --outdir out/   # batch a whole folder
python tests/test_swatches.py                 # regression check

# presets / sizes / tuning
python scripts/make_cube.py --preset punchy --size 33
python scripts/tune.py --param ir_veg --values 1.1,1.55,1.9 \
                       --param2 foliage_green_cut --values2 0.25,0.35,0.45
```

`apply_lut.py --direct` runs the transform without the LUT (full precision),
useful for confirming a `.cube` matches its source.

## Presets

| preset    | character                                            |
|-----------|------------------------------------------------------|
| `classic` | balanced, the default                                |
| `punchy`  | max foliage pop, low desat, high contrast            |
| `muted`   | lower saturation, closer to faded EIR scans          |

## Encoding variants

- **Display (primary).** Input is sRGB / Rec.709 display-referred — a normal
  developed still. Use for Lightroom / Photoshop / Capture One / Resolve on
  stills. These are the `Aerochrome_*_Display_*.cube` files.
- **S-Log3 (secondary, in-camera / Log video).** Decode S-Log3 → linear, run the
  transform, re-encode. Curves live in `encodings.py`; a dedicated S-Log3
  make-target is the next build step (display is shipped first).

## Loading per target app

**Sony A7C II (in-camera, Log/video).** Copy the 8.3-named `AEROCHR.cube` to the
card, then `MENU → Exposure/Color → Color/Tone → Manage User LUTs →
Import/Edit → User1–User16`. Applies in the Log shooting pipeline and previews
live in the EVF. **Stills note:** this is a Log/movie feature — for stills, apply
the LUT in post instead (below). *(Use the Display variant for stills; an S-Log3
variant for the in-camera Log path is the next build step.)*

**Lightroom Classic (stills — the primary workflow).** A `.cube` is a LUT, not a
develop preset, and LR can't load one directly. There is also **no in-camera path
for stills** on the A7C II (its User LUT is movie-only; Creative Look can't do the
channel rotation; RAW carries no baked look). So the look is applied on the way out
of Lightroom, exactly and automatically, via an **Export post-process**:

1. In LR, develop/cull normally. Export to a folder as **16-bit TIFF, sRGB**
   (or 8-bit JPEG). Bit depth is preserved for TIFF.
2. Run the cube over that folder:
   ```bash
   python scripts/apply_lut.py --indir /path/to/export --outdir /path/to/aerochrome
   ```
   Each file gets an `_aerochrome` copy with the look baked in, display-correct
   (identical to the Photoshop/Resolve result).
3. (Optional, fully hands-off) Put a one-line wrapper that calls step 2 into
   Lightroom's **Export Actions** folder so LR runs it automatically after every
   export.

**Lightroom profile (`.dcp`) — one-click in the Profile Browser.** For a native
preset experience, `scripts/make_profile.py` bakes the look into a Lightroom camera
profile:

```bash
python scripts/make_profile.py                    # -> luts/Aerochrome_Classic.dcp
```

Install it (Windows) into:
`C:\Users\<you>\AppData\Roaming\Adobe\CameraRaw\CameraProfiles\`
(macOS: `~/Library/Application Support/Adobe/CameraRaw/CameraProfiles/`), restart
Lightroom, then **Develop → Profile Browser → Aerochrome**. Save it into a develop
preset (Profile + any tweaks) for true one-click. Attaches to `ILCE-7CM2` by default
(`--model` to change).

Honest caveat: a profile applies its look mid-pipeline in LR's reference space (not
display space), and embeds a representative — not exact — Sony color matrix, so it
*approximates* the `.cube` rather than matching the export result byte-for-byte. For
a stylized false-color look that's fine; use the Export route above when you want the
exact cube. The look-table resolution / color matrix are easy to refine once eyeballed.

**Photoshop (exact, per-photo).** Layer → New Adjustment Layer → **Color Lookup** →
Load 3D LUT. Good for one-off hero edits; batchable via a PS action.

**Capture One.** Add as a LUT layer / ICC-style adjustment.

**DaVinci Resolve.** Drop the `.cube` into the LUT folder → apply as a node LUT.
Resolve is also the reference for verifying channel ordering: this writer uses
the standard **red-fastest** ordering (`idx = r + g·N + b·N²`); grid points round-
trip exactly through `cube.apply_trilinear`.

## Validation status (handoff §6 / §7)

All §6 surfaces land in their expected hue family, and the three §7 problems are
fixed vs. the baseline:

| surface          | baseline   | this build | status              |
|------------------|------------|------------|---------------------|
| caucasian skin   | emerald    | `#cabd96`  | pale yellow-green ✅ |
| concrete         | teal       | `#7e8280`  | near-neutral ✅      |
| asphalt          | teal       | `#3b3934`  | dark neutral ✅      |

Regression-locked in `tests/test_swatches.py`.

## Known limits / next steps

- **Leaf vs. paint** is unsolvable per-pixel (see constraint above). The chroma-
  aware veg term improves *separation* of organic vs. dull greens but cannot
  recover real IR.
- **S-Log3 make-target** for the in-camera Log path (encodings are ready).
- **ΔE2000 auto-tune** against a real Aerochrome scan + matched visible shot
  (handoff §9 stretch).
- **Spatial pass** (texture/edge cue for foliage) would help leaf-vs-paint but
  *breaks the pure-LUT requirement* — if built, ships as `apply_lut.py --spatial`
  for stills only, not exportable to the camera.
```
