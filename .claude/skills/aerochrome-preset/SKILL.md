---
name: aerochrome-preset
description: Develop or tune an Aurachrome (Aerochrome/EIR) false-color "look" (preset) in this engine — scaffold params, render before/after A-Bs on real RAWs, validate, and regenerate outputs. Use when asked to create a new look, adjust an existing one (classic/punchy/muted/portrait), fix a color artifact, or change grain/shadow behavior. Covers BOTH IR sources (the learned neural NIR default and the GRVI fallback).
---

# Developing an Aurachrome look (preset)

This repo (`aurachrome`; Python import package stays `aerochrome`) is a
RAW→16-bit-TIFF Aerochrome conversion **engine**. A "look" is a named parameter
dict in `aerochrome/params.py`. This skill is the loop for creating or tuning one
safely.

## Mental model (read first)

The pipeline (`aerochrome/transform.py`) maps `(R,G,B)→(R,G,B)`:

1. **IR source** → vegetation signal → **the IR channel**. There are TWO sources:
   - **Neural NIR (DEFAULT when torch + CUDA + weights are present)**: a compact
     U-Net (`aerochrome/neural/`) predicts a NIR channel from RGB; the engine
     derives a real `NDVI=(NIR-R)/(NIR+R)` for veg detection and uses the predicted
     NIR as the IR term. This path uses **spatial context** (the U-Net sees
     neighborhoods), so it is NOT pure per-pixel — it can spatially ID dry/pale
     grass as IR-bright (fixes the old cyan-grass artifact) and beats GRVI on
     foliage.
   - **GRVI fallback (no torch/CUDA/weights, or `--no-neural`)**: a per-pixel
     vegetation index `GRVI` gated by `GBI`, then synthetic IR. Pure per-pixel.
2. **channel rotation in LINEAR light**: `R←IR, G←R, B←G` (blue discarded), plus
   foliage/sky shaping and skin-protect.
3. **corrective grade in OKLCh**: neutral de-teal, cyan-green suppression,
   cyan→blue deepen.
4. back to sRGB → filmic desat/contrast/black-lift.
5. **shadow protect**: where INPUT luma is low, pull output luma back toward it
   and desaturate (kills washed/blotchy lightless shadows).
6. (export route only) **chromatic grain** (`aerochrome/grain.py`) — spatial, so
   it is NOT part of the color map.

**The honest limit:** the neural path is learned CORRELATION, not physical IR
recovery — there is still no real IR in the file. It does NOT solve material
ambiguity (a live leaf and green paint can share RGB *and* context → same NIR; the
basis of IR camouflage), and dry/senescing vegetation's NIR can genuinely collapse.
On the **GRVI fallback** the limit is stricter: two identical RGB pixels → identical
output, so dry grass / lightless shadow can't be made magenta. Either way, the
honest fix for a missing-signal artifact is graceful failure (mute/darken), never
invented detail. Don't oversell "recovery."

## Where to edit

`aerochrome/params.py`: `CLASSIC` is the base dict; `PUNCHY/MUTED/PORTRAIT` are
`deepcopy(CLASSIC)` + `.update(...)`. Param groups are commented inline:

- **GRVI-path veg detection**: `grvi_lo/hi`, `gbi_lo/hi`, `veg_gamma`, synthetic
  IR (`ir_base/ir_veg/ir_gamma`).
- **Neural-path look tuning (active only when a predicted NIR is supplied)**:
  `ndvi_lo/hi` (real-NDVI veg gate), `nir_red_base` (IR floor for non-veg — keep
  LOW so non-veg reds rotate cleanly to green, e.g. signage), `nir_red_veg` (IR
  gain on vegetation), `nir_desat` (veg-weighted output desat that calms the
  model's intense magenta). The same `gbi_lo/hi` gate rejects blue-dominant sky
  from the veg mask so bright low-red skies render blue, not magenta.
- shared: rotation shaping, skin protect, OKLCh grade (`cyan_green_*`,
  `green_keep_*`, `cyan_blue_*`, `blue_deepen`), filmic, `shadow_*`, `grain_*`.

When you change `CLASSIC`, it propagates to the others unless they override the
same key — if a change should NOT touch a derived look, pin that key explicitly in
that look's `.update(...)`.

## The loop

1. **Scaffold**: add or edit the look in `params.py`. New look → also register it
   in the `PRESETS` dict.
2. **Render an A/B on REAL RAWs** (never trust swatches alone). Know which IR path
   you are tuning: the neural-path params do nothing without a predicted NIR.
   Use a throwaway script in the scratchpad following this pattern:
   ```python
   import sys, numpy as np, rawpy; from PIL import Image
   sys.path.insert(0, "<repo>")
   from aerochrome import transform, params, neural
   # load a RAW (camera WB, no_auto_bright=False to match the engine), downscale.
   # GRVI path:   transform.aerochrome(arr, P)
   # neural path: nir = neural.predict_nir(arr); transform.aerochrome(arr, P, nir=nir)
   # np.hstack the variants (SOURCE | GRVI | neural), save a PNG, then VIEW it.
   ```
   Representative RAWs live in `Downloads/aero-tests/` (foliage, skin/apple-picker,
   red→green sign, sky/pier, and the dusk portrait `DSC04990` = the shadow stress
   test). Always include a daylight scene AND the dusk one; when tuning the neural
   look, also include a blue-sky scene (the GBI sky gate).
3. **Inspect the image** (Read the PNG). Check: foliage→magenta intact, red→green
   intact, skies a deep blue (not neon cyan, not magenta), neutrals not teal, skin
   per the look's intent, and shadows deep+clean (not washed/blotchy).
4. **Iterate** params; keep `float64` (fidelity). Surface real trade-offs to the
   user (e.g. wider cyan-green suppression deepens grass-cyan fix but costs sky
   blue — they are the same hue range).
5. **Validate**: `poetry run pytest -q` (the §6 hue-family swatch tests must pass;
   they exercise the GRVI `nir=None` path, so a neural-only change must not regress
   them). Confirm no daylight regression (diff vs the prior look ~0 outside the
   region you targeted).
6. **Regenerate outputs** as needed (console cmd `aurachrome`, alias `aero-convert`):
   - `poetry run aurachrome -i <raws> -o <out> --preset <name> [--gpu]`
   - export options: `--format tiff16|jpeg|both`, `--longedge N` (downscale long
     edge for fast previews), `--no-neural` (force the GRVI index for A/B or repro).
   - (legacy LUT/.dcp/.xmp path exists but is deprecated — only if explicitly asked.)
7. **Update memory** (`aerochrome-lut-project.md` / `aurachrome-neural-nir.md`) with
   what changed and why.

## Guardrails

- Verify a render before claiming a fix; measure luma/dark-fraction when the
  complaint is "washed out" (input vs output `meanL`, `dark(L<.12)`).
- Don't lift lightless shadows; don't invent color where there's no signal. The
  neural NIR is plausible, not measured — don't claim it "recovers" real IR.
- Grain is export-only and spatial — never claim it can go in a cube/profile.
- Keep daylight untouched when fixing a shadow/edge case (gate by luma/hue/chroma).
- GPU and CPU must stay numerically identical for the color transform (they
  dispatch one codepath via `aerochrome/backend.py`); don't add a numpy-only call
  to a hot path. (The neural U-Net itself is CUDA-only and runs on the serial path;
  CPU multiprocessing workers force `--no-neural`-equivalent GRVI.)
