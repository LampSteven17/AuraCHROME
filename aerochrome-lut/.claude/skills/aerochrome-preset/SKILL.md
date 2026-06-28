---
name: aerochrome-preset
description: Develop or tune an Aerochrome/EIR false-color "look" (preset) in this engine — scaffold params, render before/after A-Bs on real RAWs, validate, and regenerate outputs. Use when asked to create a new look, adjust an existing one (classic/punchy/muted/portrait), fix a color artifact, or change grain/shadow behavior.
---

# Developing an Aerochrome look (preset)

This repo is a RAW→16-bit-TIFF Aerochrome conversion **engine**. A "look" is a
named parameter dict in `aerochrome/params.py`. This skill is the loop for
creating or tuning one safely.

## Mental model (read first)

The pipeline (`aerochrome/transform.py`) is a pure per-pixel `(R,G,B)→(R,G,B)`:

1. **linearize** → vegetation signal (`GRVI` gated by `GBI`) → **synthesize IR**
2. **channel rotation in LINEAR light**: `R←IR, G←R, B←G` (blue discarded), plus
   foliage/sky shaping and skin-protect.
3. **corrective grade in OKLCh**: neutral de-teal, cyan-green suppression,
   cyan→blue deepen.
4. back to sRGB → filmic desat/contrast/black-lift.
5. **shadow protect**: where INPUT luma is low, pull output luma back toward it
   and desaturate (kills washed/blotchy lightless shadows).
6. (export route only) **chromatic grain** (`aerochrome/grain.py`) — spatial, so
   it is NOT part of the color map.

**The hard limit:** there is no IR in the file. Two identical RGB pixels → identical
output. Dry/pale grass and lightless shadows can't be made magenta; the honest fix
is graceful failure (mute/darken), never invented detail. Don't oversell "recovery."

## Where to edit

`aerochrome/params.py`: `CLASSIC` is the base dict; `PUNCHY/MUTED/PORTRAIT` are
`deepcopy(CLASSIC)` + `.update(...)`. Param groups are commented inline:
veg detection, synthetic IR, rotation shaping, skin protect, OKLCh grade
(cyan_green_*, green_keep_*, cyan_blue_*, blue_deepen), filmic, shadow_* , grain_*.

When you change `CLASSIC`, it propagates to the others unless they override the
same key — if a change should NOT touch a derived look, pin that key explicitly in
that look's `.update(...)`.

## The loop

1. **Scaffold**: add or edit the look in `params.py`. New look → also register it
   in the `PRESETS` dict.
2. **Render an A/B on REAL RAWs** (never trust swatches alone). Use a throwaway
   script in the scratchpad following this pattern:
   ```python
   import sys, numpy as np, rawpy; from PIL import Image
   sys.path.insert(0, "<repo>/aerochrome-lut")
   from aerochrome import transform, params
   # load a RAW (camera WB, no_auto_bright=False to match the engine), downscale
   # for speed, run transform.aerochrome(arr, P) for the variants, np.hstack them,
   # save a PNG, then VIEW it.
   ```
   Representative RAWs live in `Downloads/aero-tests/` (foliage, skin/apple-picker,
   red→green sign, sky/pier, and the dusk portrait `DSC04990` = the shadow stress
   test). Always include a daylight scene AND the dusk one.
3. **Inspect the image** (Read the PNG). Check: foliage→magenta intact, red→green
   intact, skies a deep blue (not neon cyan), neutrals not teal, skin per the
   look's intent, and shadows deep+clean (not washed/blotchy).
4. **Iterate** params; keep `float64` (fidelity). Surface real trade-offs to the
   user (e.g. wider cyan-green suppression deepens grass-cyan fix but costs sky
   blue — they are the same hue range).
5. **Validate**: `poetry run pytest -q` (the §6 hue-family swatch tests must pass).
   Confirm no daylight regression (diff vs the prior look should be ~0 outside the
   region you targeted).
6. **Regenerate outputs** as needed:
   - TIFFs (primary): `poetry run aero-convert -i <raws> -o <out> --preset <name> [--gpu]`
   - (legacy LUT/profile path exists but is deprecated — only if explicitly asked.)
7. **Update memory** (`aerochrome-lut-project.md`) with what changed and why.

## Guardrails

- Verify a render before claiming a fix; measure luma/dark-fraction when the
  complaint is "washed out" (input vs output `meanL`, `dark(L<.12)`).
- Don't lift lightless shadows; don't invent color where there's no signal.
- Grain is export-only and spatial — never claim it can go in a cube/profile.
- Keep daylight untouched when fixing a shadow/edge case (gate by luma/hue/chroma).
- GPU and CPU must stay numerically identical (they dispatch one codepath via
  `aerochrome/backend.py`); don't add a numpy-only call to a hot path.
