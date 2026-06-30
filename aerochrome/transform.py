"""
Aerochrome approximation -- re-architected core transform.

Pure per-pixel function  (R,G,B) -> (R,G,B), sRGB display-referred in/out.
No spatial ops (so it bakes cleanly into a .cube).

Why this differs from the shipped baseline (deliberately -- see handoff Q):
  * The mechanical channel rotation and the foliage/sky shaping run in LINEAR
    light, where mixing reflectances is physically meaningful.
  * The corrective grade (skin / neutral / highlight) runs in OKLCh, where
    "warm low-chroma skin" and "near-neutral concrete" are clean, separable
    windows (a hue arc + a chroma threshold) instead of overlapping sRGB ratio
    hacks. This is what fixes the §7 skin-too-emerald / neutrals-drift-teal
    problems at the root rather than patching them.

The honest limit is unchanged: there is no IR channel in the file, so IR is
*synthesized* from a visible-band vegetation index. Two pixels with identical
RGB get identical output. We lean on "looks like a plant" correlating with
"IR-bright" -- a correlation, not a recovery.

The real film channel rotation:
    RED_out   <- INFRARED   (live vegetation -> red)
    GREEN_out <- RED        (red objects -> green)
    BLUE_out  <- GREEN      (green objects -> blue; blue discarded by yellow filter)
"""

import numpy as np

from . import encodings as enc
from . import params as _params
from .backend import xp_for

EPS = 1e-9


def smoothstep(e0, e1, x):
    xp = xp_for(x)
    t = xp.clip((x - e0) / (e1 - e0 + EPS), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def aerochrome(rgb, p=None, nir=None):
    """rgb: float array (...,3) in [0,1], sRGB display-referred. Same shape out.

    Runs on whichever backend owns `rgb`: a NumPy array stays on CPU, a CuPy
    array runs on the GPU. Behavior is identical; only the device differs.

    nir: optional estimated near-infrared channel, shape == rgb[...,0], in [0,1]
    (e.g. from the learned RGB->NIR model). When given, vegetation is detected
    from a *real* NDVI = (NIR-R)/(NIR+R) and the NIR feeds the red channel
    directly — the actual Aerochrome computation with an estimated IR band.
    When None, the per-pixel GRVI index synthesizes the IR term as before.
    """
    if p is None:
        p = _params.CLASSIC
    xp = xp_for(rgb)
    rgb = xp.clip(xp.asarray(rgb, dtype=xp.float64), 0.0, 1.0)

    # ---- to linear working space -----------------------------------------
    lin = enc.srgb_to_linear(rgb)
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]
    L = 0.2126 * r + 0.7152 * g + 0.0722 * b

    # input perceptual coords (for hue/chroma-gated corrections later)
    in_lch = enc.oklab_to_oklch(enc.linear_rgb_to_oklab(lin))
    in_C = in_lch[..., 1]
    in_h = enc.hue_deg(in_lch)

    # ---- stage 1: vegetation signal + IR ---------------------------------
    cb = smoothstep(p['veg_chroma_lo'], p['veg_chroma_hi'], in_C)
    if nir is not None:
        # estimated true-IR path: real NDVI drives vegetation; NIR is the red src.
        nir_lin = xp.clip(xp.asarray(nir, dtype=xp.float64), 0.0, 1.0)
        ndvi = (nir_lin - r) / (nir_lin + r + EPS)
        # The model tends to predict high NIR on bright sky/cloud; combined with
        # sky's low red that yields a high NDVI, so sky gets mis-flagged as veg and
        # rendered magenta. Reject it the same way the GRVI path does -- sky is
        # blue-dominant (g<b), so gate veg by green-vs-blue.
        gbi = (g - b) / (g + b + EPS)
        veg = smoothstep(p['ndvi_lo'], p['ndvi_hi'], ndvi) \
            * smoothstep(p['gbi_lo'], p['gbi_hi'], gbi)
        veg = veg ** p['veg_gamma']
        veg = xp.clip(veg * (1.0 + p['veg_chroma_boost'] * cb), 0.0, 1.0)
        # gate IR by vegetation: non-veg (red signage, paint) gets little IR so the
        # red->green rotation stays clean; foliage gets the full predicted NIR.
        ir = xp.clip(nir_lin * (p['nir_red_base'] + p['nir_red_veg'] * veg),
                     0.0, 1.0) ** p['ir_gamma']
    else:
        # synthesized-IR path: visible-band GRVI index, gated by green-vs-blue.
        grvi = (g - r) / (g + r + EPS)          # > 0 for foliage
        gbi = (g - b) / (g + b + EPS)           # > 0 when green beats blue (reject sky)
        veg = smoothstep(p['grvi_lo'], p['grvi_hi'], grvi) * \
              smoothstep(p['gbi_lo'], p['gbi_hi'], gbi)
        veg = veg ** p['veg_gamma']
        veg = veg * (1.0 + p['veg_chroma_boost'] * cb)   # saturated greens read "alive"
        veg = xp.clip(veg, 0.0, 1.0)
        ir = L * (p['ir_base'] + p['ir_veg'] * veg)
        ir = xp.clip(ir, 0.0, 1.0) ** p['ir_gamma']

    # ---- stage 2: the channel rotation (in linear) -----------------------
    R = ir              # IR    -> red
    G = r.copy()        # red   -> green
    B = g.copy()        # green -> blue

    # deepen foliage toward magenta/crimson by pulling mapped-green down
    G = G * (1.0 - p['foliage_green_cut'] * veg)

    # lift cyan in sky / non-veg blue-dominant regions
    sky = (1.0 - veg) * smoothstep(0.0, 0.25, (b - r) / (b + r + EPS))
    G = G + p['sky_cyan_boost'] * sky * 0.6
    B = B + p['sky_cyan_boost'] * sky

    # ---- stage 2b: skin protect ------------------------------------------
    # detect in OKLCh (warm hue arc, moderate chroma band, not vegetation),
    # correct in linear. Skin's mapped-green (G<-R_in) is large and bright,
    # which is exactly what makes it read emerald; so lift red TOWARD that
    # green channel (hue -> yellow-green) and trim green/blue.
    skin = enc.hue_window(in_h, p['skin_hue_center'], p['skin_hue_width'])
    skin = skin * smoothstep(p['skin_chroma_lo'], p['skin_chroma_lo2'], in_C) \
                * (1.0 - smoothstep(p['skin_chroma_hi'], p['skin_chroma_hi2'], in_C)) \
                * (1.0 - veg) * p['skin_amount']
    R = R + p['skin_red_lift'] * skin * G
    G = G * (1.0 - p['skin_green_cut'] * skin)
    B = B * (1.0 - p['skin_blue_cut'] * skin)

    out_lin = xp.clip(xp.stack([R, G, B], axis=-1), 0.0, None)

    # ---- stage 3: corrective grade in OKLCh ------------------------------
    out_lch = enc.oklab_to_oklch(enc.linear_rgb_to_oklab(out_lin))
    oL, oC, oh = out_lch[..., 0], out_lch[..., 1], out_lch[..., 2]

    # neutral de-teal: low INPUT chroma -> pull OUTPUT chroma toward gray.
    # (concrete/asphalt/cloud were drifting teal because B<-G pushes neutrals
    # cyan; here we just refuse to let near-neutral inputs gain chroma.)
    neutral = (1.0 - smoothstep(p['neutral_chroma_lo'], p['neutral_chroma_hi'], in_C))
    oC = oC * (1.0 - p['neutral_strength'] * neutral)

    # cyan-green cast suppression. Two different jobs:
    #  * VIVID GREEN (red->green objects, ~140 deg) must be preserved.
    #  * CYAN (sky, water, warm tones flipping cyan, ~190 deg) is the big aesthetic
    #    problem -- real Aerochrome keeps non-veg roughly neutral/deep-blue, not
    #    bright turquoise -- so desaturate it across ALL chroma and nudge it toward
    #    blue. The dull green-teal cast (rock/shadow) is desaturated too.
    oh_deg = xp.degrees(oh) % 360.0
    cg = enc.hue_window(oh_deg, p['cyan_green_center'], p['cyan_green_width'])
    # protect only vivid GREEN (not cyan): narrow green arc, gated by chroma
    green_keep = enc.hue_window(oh_deg, p['green_keep_center'], p['green_keep_width']) \
                 * smoothstep(p['cyan_green_keep_lo'], p['cyan_green_keep_hi'], oC)
    cg = cg * (1.0 - green_keep)
    oC = oC * (1.0 - p['cyan_green_desat'] * cg)
    # push remaining cyan toward blue AND deepen it, so skies/water read as a
    # rich Aerochrome blue rather than washed-out grey-cyan.
    cyan = enc.hue_window(oh_deg, p['cyan_blue_center'], p['cyan_blue_width'])
    oh = oh + xp.radians(p['cyan_to_blue']) * cyan
    oC = oC * (1.0 + p['blue_deepen'] * cyan)

    # skin grade: pull skin OUTPUT chroma down (waxy pale, less green) and
    # optionally lean it cooler. `skin` is the stage-2b mask. 0 = no change.
    if p.get('skin_desat', 0.0) or p.get('skin_cool', 0.0):
        oC = oC * (1.0 - p.get('skin_desat', 0.0) * skin)
        oh = oh + xp.radians(p.get('skin_cool', 0.0)) * skin

    out_lch = xp.stack([oL, oC, oh], axis=-1)
    out_lin = enc.oklab_to_linear_rgb(enc.oklch_to_oklab(out_lch))
    # tiny warm bias so de-teal'd neutrals/clouds aren't dead gray
    out_lin = out_lin + (neutral * p['neutral_warm'])[..., None] * xp.array([1.0, 0.2, -0.6])

    out_lin = xp.clip(out_lin, 0.0, None)

    # ---- back to sRGB + global filmic ------------------------------------
    out = enc.linear_to_srgb(out_lin)

    # global desat toward output luma (takes the digital edge off)
    gL = (out[..., 0] * 0.2126 + out[..., 1] * 0.7152 + out[..., 2] * 0.0722)[..., None]
    out = out * (1.0 - p['desat']) + gL * p['desat']

    # neural path only: veg-weighted desat to calm the model's intense magenta
    # (foliage gets the most pull-back; sky/neutrals largely untouched).
    if nir is not None and p.get('nir_desat', 0) > 0:
        nL = (out[..., 0] * 0.2126 + out[..., 1] * 0.7152 + out[..., 2] * 0.0722)[..., None]
        w = (p['nir_desat'] * veg)[..., None]
        out = out * (1.0 - w) + nL * w

    # gentle S-curve + small black lift (in display space)
    out = xp.clip(out, 0.0, 1.0)
    out = out + p['contrast'] * (out - 0.5) * (1.0 - xp.abs(out - 0.5) * 2.0)
    out = p['black_lift'] + (1.0 - p['black_lift']) * out
    out = xp.clip(out, 0.0, 1.0)

    # Portrait look: blend the skin region back toward its natural color so people
    # render pleasingly while the rest of the frame stays Aerochrome. (skin_preserve
    # is 0 for the authentic profiles, >0 only for Portrait.)
    if p['skin_preserve'] > 0:
        psk = enc.hue_window(in_h, p['skin_preserve_hue'], p['skin_preserve_hue_width']) \
            * smoothstep(p['skin_preserve_clo'], p['skin_preserve_clo2'], in_C) \
            * (1.0 - smoothstep(p['skin_preserve_chi'], p['skin_preserve_chi2'], in_C)) \
            * (1.0 - veg)
        w = xp.clip(p['skin_preserve'] * psk, 0.0, 1.0)[..., None]
        out = out * (1.0 - w) + rgb * w

    # ---- shadow protect --------------------------------------------------
    # Lightless shadows carry no real color, and the look above tends to lift +
    # noise-amplify them (washed muddy grey + blotchy magenta/cyan speckle). Where
    # the INPUT is deep shadow, pull output luma back down toward the input's (undo
    # the wash, restore depth) and desaturate (kill the color noise). Daylight is
    # unaffected because the gate is 0 above shadow_hi.
    if p.get('shadow_recover', 0) > 0 or p.get('shadow_desat', 0) > 0:
        in_luma = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
        sh = 1.0 - smoothstep(p['shadow_lo'], p['shadow_hi'], in_luma)
        out_luma = out[..., 0] * 0.2126 + out[..., 1] * 0.7152 + out[..., 2] * 0.0722
        # 1) recover depth: only pull DOWN (never brighten) toward the input luma
        target = xp.minimum(out_luma, in_luma)
        rw = sh * p['shadow_recover']
        new_luma = out_luma * (1.0 - rw) + target * rw
        out = out * (new_luma / (out_luma + EPS))[..., None]
        # 2) desaturate the protected shadows toward their (now darkened) luma
        g = out[..., 0] * 0.2126 + out[..., 1] * 0.7152 + out[..., 2] * 0.0722
        dw = (sh * p['shadow_desat'])[..., None]
        out = out * (1.0 - dw) + g[..., None] * dw

    return xp.clip(out, 0.0, 1.0)
