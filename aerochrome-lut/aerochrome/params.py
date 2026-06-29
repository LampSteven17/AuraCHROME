"""
Named parameter presets for the re-architected (linear + OKLCh) transform.

The transform runs in three stages; params group accordingly:
  1. veg detection  -> synthesize the missing IR channel
  2. channel rotation + foliage/sky shaping (done in LINEAR light)
  3. corrective grade in OKLCh (skin protect, neutral de-teal, highlight protect)
     + global filmic desat / contrast

Thresholds for the veg gates are tuned for LINEAR values, which run larger than
the sRGB-encoded numbers the original baseline used (linear GRVI of foliage is
~0.5 vs ~0.26 sRGB) -- don't port the old gate numbers verbatim.
"""

import copy

CLASSIC = dict(
    # --- stage 1: vegetation detection (computed in linear light) ---
    grvi_lo=0.05, grvi_hi=0.45,      # green-vs-red index gate; foliage > 0
    gbi_lo=-0.05, gbi_hi=0.45,       # green-vs-blue gate; rejects blue/cyan sky
    veg_gamma=0.85,
    # NDVI gate used ONLY on the learned-NIR path (real NDVI=(NIR-R)/(NIR+R)).
    ndvi_lo=0.10, ndvi_hi=0.50,      # >~0.2 = vegetation; soft ramp 0.10..0.50
    # neural-path look tuning (active only when a predicted NIR is supplied):
    # gate the IR->red channel by vegetation so non-veg reds keep the clean
    # red->green rotation (fixes red signage going amber) while foliage stays
    # magenta; plus a veg-weighted desat to calm the model's intense magenta.
    nir_red_base=0.18,               # IR floor for non-veg (low => reds rotate to green)
    nir_red_veg=0.95,                # IR gain on vegetation (keeps foliage hot)
    nir_desat=0.40,                  # veg-weighted output desaturation (calms the magenta)
    # chroma-aware boost: saturated organic greens get more synthetic IR than
    # dull yellow-green grass or a flat green wall (modest -- it is still a
    # per-pixel correlation, NOT a recovery of real IR).
    veg_chroma_lo=0.04, veg_chroma_hi=0.14, veg_chroma_boost=0.35,

    # --- synthetic IR ---
    ir_base=0.16,                    # IR floor (~ luminance reflects some IR)
    ir_veg=1.55,                     # extra IR for vegetation
    ir_gamma=0.90,

    # --- stage 2: rotation shaping (linear) ---
    foliage_green_cut=0.35,          # deepen foliage red -> magenta/crimson
    sky_cyan_boost=0.18,             # lift cyan in non-veg blue-dominant areas

    # --- stage 2b: skin protect (detect in OKLCh, correct in linear) ---
    # warm, low-chroma, non-veg pixels -> pale waxy yellow-green, not emerald.
    # chroma window is tight on the high side so it does NOT catch orange
    # (C~0.16), which must stay green per the §6 table.
    skin_hue_center=50.0,            # OKLab hue (deg) of caucasian skin (warm)
    skin_hue_width=38.0,
    skin_chroma_lo=0.020, skin_chroma_lo2=0.045,
    skin_chroma_hi=0.100, skin_chroma_hi2=0.150,
    skin_red_lift=0.80,              # lift mapped-red TOWARD the green channel
    skin_green_cut=0.28,             # pull mapped-green down (kills emerald)
    skin_blue_cut=0.25,              # pull mapped-blue down (toward yellow)
    skin_amount=1.0,

    # --- stage 3: corrective grade in OKLCh ---
    # neutral de-teal: low-INPUT-chroma surfaces (concrete/asphalt/cloud) get
    # their OUTPUT chroma pulled toward gray, proportional to (1 - input chroma).
    neutral_chroma_lo=0.015, neutral_chroma_hi=0.060,
    neutral_strength=0.90,
    neutral_warm=0.010,              # tiny warm bias so neutrals/clouds aren't dead

    # cyan-green cast suppression: water / wet rock / shadows pick up a teal tint
    # (B<-G_in spill). Desaturate the cyan-green output arc while leaving the
    # magenta/red foliage signal untouched -- the "non-veg goes neutral" rule.
    # Suppress the dull GREEN-TEAL cast (rock/shadow/wet) -- narrow band that does
    # NOT reach blue, so sky/water can stay blue rather than being greyed out.
    # Widened + strengthened (Classic v5) so stray BRIGHT cyan on dry/pale grass
    # -- blades too washed-out to register as vegetation, which fall through to the
    # cyan path instead of going magenta -- is muted toward a calm grey-blue rather
    # than glowing turquoise. This is the residual missing-IR artifact the user
    # flagged ("weird edges around the orbs/lights"); we can dull it, not remove it.
    cyan_green_center=168.0,         # OKLab hue (deg): green-teal to suppress
    cyan_green_width=56.0,           # ~140..196 (stops just short of cyan/blue 200)
    cyan_green_desat=0.80,
    cyan_green_keep_lo=0.11,         # chroma over which vivid GREEN is preserved
    cyan_green_keep_hi=0.19,
    green_keep_center=138.0,         # only protect this narrow green arc (red->green)
    green_keep_width=26.0,
    # Push cyan -> blue and DEEPEN it, so skies/water read as a rich Aerochrome
    # blue instead of a washed-out grey-cyan.
    cyan_blue_center=200.0,
    cyan_blue_width=42.0,            # ~179..221
    cyan_to_blue=30.0,               # degrees of hue nudge toward blue
    blue_deepen=0.26,               # chroma boost applied to the now-blue pixels
                                     # (eased a touch with the wider cyan-green cut so
                                     #  the grass fix doesn't flatten genuine sky/water)
    lut_cyan_satcap=1.05,            # profile-only satScale cap on cyan/green nodes
                                     # (raised a touch now that blue is intentional)

    # skin handling: 0 = true Aerochrome (authentic CIR waxy skin),
    # >0 = blend skin back toward its natural color (the Portrait look).
    skin_preserve=0.0,
    skin_preserve_hue=52.0,          # warm skin arc for the portrait blend
    skin_preserve_hue_width=46.0,
    skin_preserve_clo=0.020, skin_preserve_clo2=0.045,
    skin_preserve_chi=0.150, skin_preserve_chi2=0.230,

    # global filmic
    desat=0.12,                      # pull toward output luma
    contrast=0.18,                   # S-curve strength (applied in sRGB output)
    black_lift=0.015,

    # --- shadow protect (final stage) ---------------------------------------
    # In low-key images most of the frame is lightless shadow. The IR synthesis +
    # tone curve were LIFTING those shadows into muddy mid-grey, and the veg test
    # ran on noisy near-black values -> blotchy magenta/cyan speckle. Both read as
    # "fake". There is no real color in lightless shadow to recover, so the honest
    # move is to let shadows fall gracefully: pull output luma back toward the
    # INPUT's (restores depth/mood, kills the wash) and desaturate (kills the color
    # noise). Gated to deep shadow only via input luma, so daylight is untouched.
    shadow_lo=0.05,                  # below this input luma = full protect
    shadow_hi=0.26,                  # above this = no effect (daylight)
    shadow_recover=0.85,             # how hard to pull output luma back down to input
    shadow_desat=0.65,               # how hard to desaturate the protected shadows

    # --- chromatic film grain (export route only; spatial, see grain.py) ----
    # strength = midtone noise sigma in display units; size = grain clump sigma
    # (px, sized for full-res ~33 MP files); chroma = colored-shimmer amount.
    grain_strength=0.028,
    grain_size=1.2,
    grain_chroma=0.35,
)

PUNCHY = copy.deepcopy(CLASSIC)
PUNCHY.update(
    ir_veg=1.9,
    veg_chroma_boost=0.5,
    foliage_green_cut=0.45,
    sky_cyan_boost=0.24,
    desat=0.06,
    contrast=0.24,
    # Pinned to the original (narrower) cyan-green geometry so Punchy is byte-for-byte
    # the look the user already likes -- the Classic v5 grass-cut does NOT touch it.
    cyan_green_center=158.0,
    cyan_green_width=48.0,
    cyan_green_desat=0.66,           # keep a touch more vibrance in the punchy look
    blue_deepen=0.45,                # dramatic, saturated blue skies/water
    grain_strength=0.038,            # a touch heavier
    grain_size=1.3,
)

MUTED = copy.deepcopy(CLASSIC)
MUTED.update(
    ir_veg=1.25,
    veg_chroma_boost=0.2,
    foliage_green_cut=0.25,
    sky_cyan_boost=0.12,
    desat=0.22,
    contrast=0.12,
    cyan_green_desat=0.80,           # cleanest neutrals for the faded-scan look
    blue_deepen=0.16,                # more neutral: gentle blue, not dramatic
    grain_strength=0.046,            # grainier, coarser -- faded-scan feel
    grain_size=1.5,
    grain_chroma=0.40,
)

# Portrait: the authentic Classic look everywhere EXCEPT skin, which is blended
# back toward its natural color so people render pleasingly.
PORTRAIT = copy.deepcopy(CLASSIC)
PORTRAIT.update(
    skin_preserve=0.92,
    # SKIN-ONLY mask: narrow warm hue + a tight MODERATE-chroma band. Saturated
    # warm things (sunsets, neon-warm signage) sit above the chroma ceiling and
    # are excluded, so they still go full Aerochrome; near-neutral warm shadows
    # sit below the floor and are excluded too.
    skin_preserve_hue=52.0,
    skin_preserve_hue_width=34.0,
    skin_preserve_clo=0.030, skin_preserve_clo2=0.055,
    skin_preserve_chi=0.110, skin_preserve_chi2=0.150,
    grain_strength=0.018,            # gentle so skin stays clean
    grain_size=1.0,
)

PRESETS = {"classic": CLASSIC, "punchy": PUNCHY, "muted": MUTED, "portrait": PORTRAIT}


def get(name="classic"):
    return copy.deepcopy(PRESETS[name])
