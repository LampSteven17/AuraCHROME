"""
Learned RGB->NIR backend (optional).

The engine predicts a near-infrared channel from visible RGB with a small paired
U-Net, then feeds it to the transform as the infrared term (and as a real NDVI
source for vegetation detection). This is a *learned statistical correlation*,
not physical recovery: it renders convincingly on real foliage and will lie on
materials that share RGB but differ in NIR (green paint, plastic plants). See the
README "References" section.

Everything here is import-safe without torch: `is_available()` returns False and
the engine falls back to the per-pixel GRVI path.
"""

from .infer import default_weights_path, is_available, predict_nir

__all__ = ["is_available", "predict_nir", "default_weights_path"]
