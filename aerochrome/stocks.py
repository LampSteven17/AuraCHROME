"""
Stock profiles -- the abstraction that lets one engine emulate many film stocks.

A "stock" decomposes into: (1) spectral capture (which IR band, if any, the stock
sees -> drives whether the neural NIR is needed), (2) a render FAMILY (which
transform produces the look), (3) that family's parameters, and (4) shared spatial
artifacts (grain today; halation/bloom later). Redscale is the proof the split is
right: it is family="redscale", ir_band_nm=None -> it skips the IR machinery
entirely and renders purely parametrically.

Families:
  - "aerochrome": false-color reversal IR (classic/punchy/muted/portrait). Uses
    transform.aerochrome with a predicted/derived NIR. ir_band_nm set.
  - "redscale":   visible-light layer-order effect. No NIR, no neural.

The Aerochrome looks reuse the tuned dicts in params.py verbatim, so this layer is
additive -- the existing color science is untouched.
"""

import copy
from dataclasses import dataclass, field

from . import monoir, params, redscale, transform

# The deep-NIR band Aerochrome/EIR's IR layer effectively integrates (~700-900nm,
# cutoff ~900). Documentation + a hook for future spectral-band selection; the
# current neural backend predicts a single broadband NIR regardless.
AEROCHROME_IR_NM = 850.0


@dataclass
class StockProfile:
    name: str
    family: str                 # "aerochrome" | "redscale"
    params: dict = field(default_factory=dict)
    ir_band_nm: float = None    # target IR band; None => no NIR needed (skip neural)
    halation: float = 0.0       # future spatial bloom strength (HIE/Efke-AURA); 0 = none
    description: str = ""

    @property
    def needs_ir(self):
        """Whether this stock requires a NIR channel (and thus the neural net)."""
        return self.ir_band_nm is not None


_STOCKS = {}


def _register(profile):
    _STOCKS[profile.name] = profile


# Aerochrome family -- reuse the tuned param dicts from params.py.
for _name, _p in params.PRESETS.items():
    _register(StockProfile(
        name=_name, family="aerochrome", params=_p,
        ir_band_nm=AEROCHROME_IR_NM,
        description="Aerochrome / EIR false-color infrared reversal look",
    ))

# Redscale -- visible-light only, parametric, no IR.
_register(StockProfile(
    name="redscale", family="redscale", params=redscale.DEFAULT,
    ir_band_nm=None,
    description="Redscale: colour negative shot through the base (warm red->yellow)",
))

# HIE -- black & white high-speed infrared (Wood effect + halation glow). Uses
# the predicted NIR (deep band ~880nm), and the halation spatial pass.
_register(StockProfile(
    name="hie", family="mono_ir", params=monoir.DEFAULT,
    ir_band_nm=880.0, halation=monoir.DEFAULT["halation_strength"],
    description="Kodak HIE: B&W high-speed infrared, Wood effect + halation",
))


def get(name):
    """Return a StockProfile with a deep-copied params dict (safe to mutate)."""
    p = _STOCKS[name]
    return StockProfile(p.name, p.family, copy.deepcopy(p.params),
                        p.ir_band_nm, p.halation, p.description)


def names():
    """All registered stock names."""
    return list(_STOCKS)


def family_names(family):
    """Stock names in a given family (e.g. 'aerochrome' for the --preset all set)."""
    return [n for n, p in _STOCKS.items() if p.family == family]


def render(rgb, stock, nir=None):
    """Render `rgb` (display sRGB, (..,3) in [0,1]) through a stock profile.

    This is the simple, un-chunked dispatch used by tests/previews. The export
    path (aero_convert.convert) keeps its own memory-aware chunking for the
    aerochrome family; both go through the same family switch."""
    if isinstance(stock, str):
        stock = get(stock)
    if stock.family == "aerochrome":
        return transform.aerochrome(rgb, stock.params, nir=nir)
    if stock.family == "mono_ir":
        return monoir.apply(rgb, stock.params, nir=nir)
    if stock.family == "redscale":
        return redscale.apply(rgb, stock.params)
    raise ValueError(f"unknown stock family {stock.family!r}")
