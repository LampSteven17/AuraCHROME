"""
RGB->NIR inference: tiled, GPU, with CuPy<->torch zero-copy interop and a clean
availability gate so the engine can fall back to the per-pixel index.

The public surface is import-safe without torch.
"""

import os

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:
    torch = None
    _HAS_TORCH = False

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MODEL_CACHE = {}


def default_weights_path():
    """Where a trained NIR model lives. Override with $AURACHROME_NIR_WEIGHTS."""
    env = os.environ.get("AURACHROME_NIR_WEIGHTS")
    if env:
        return env
    return os.path.join(ROOT, "models", "nir_unet.pt")


def is_available(weights_path=None):
    """True iff torch + a CUDA device + a weights file are all present."""
    if not _HAS_TORCH:
        return False
    try:
        if not torch.cuda.is_available():
            return False
    except Exception:
        return False
    return os.path.exists(weights_path or default_weights_path())


def load_model(weights_path=None, device="cuda"):
    path = weights_path or default_weights_path()
    key = (path, device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    from .unet import CompactUNet
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    base = ckpt.get("base", 48) if isinstance(ckpt, dict) else 48
    model = CompactUNet(base=base).to(device).eval()
    model.load_state_dict(state)
    _MODEL_CACHE[key] = model
    return model


def _cosine_window(h, w, device):
    """2-D raised-cosine weights for seamless tile blending."""
    wy = torch.hann_window(h, periodic=False, device=device).clamp_min(1e-3)
    wx = torch.hann_window(w, periodic=False, device=device).clamp_min(1e-3)
    return (wy[:, None] * wx[None, :])[None, None]


def predict_nir(rgb, weights_path=None, tile=1024, overlap=128, device="cuda",
                smooth_sigma=1.2):
    """Predict a NIR channel from a display-referred sRGB image.

    rgb: float array (H,W,3) in [0,1] — NumPy or CuPy. Returns a (H,W) float array
    (same array module as the input) in [0,1]. Runs tiled with cosine blending to
    bound VRAM on full-resolution (~33 MP) frames.
    """
    if not _HAS_TORCH:
        raise RuntimeError("torch not installed (poetry install --with neural)")
    model = load_model(weights_path, device)

    # accept CuPy without a host round-trip (DLPack), else NumPy
    is_cupy = type(rgb).__module__.startswith("cupy")
    if is_cupy:
        import cupy as cp
        chw = cp.ascontiguousarray(rgb.astype(cp.float32).transpose(2, 0, 1)[None])
        t = torch.from_dlpack(chw)
    else:
        a = np.ascontiguousarray(np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1)[None])
        t = torch.from_numpy(a).to(device)

    _, _, H, W = t.shape
    stride = max(1, tile - overlap)
    # reflect-pad so every output pixel sees full context and edges stay full-size
    pad = overlap // 2
    t = torch.nn.functional.pad(t, (pad, pad, pad, pad), mode="reflect")
    _, _, Hp, Wp = t.shape

    acc = torch.zeros((1, 1, Hp, Wp), device=device)
    wsum = torch.zeros((1, 1, Hp, Wp), device=device)
    ys = list(range(0, max(1, Hp - tile + 1), stride)) + [max(0, Hp - tile)]
    xs = list(range(0, max(1, Wp - tile + 1), stride)) + [max(0, Wp - tile)]
    ys, xs = sorted(set(ys)), sorted(set(xs))

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for y in ys:
            for x in xs:
                th = min(tile, Hp - y)
                tw = min(tile, Wp - x)
                patch = t[:, :, y:y + th, x:x + tw]
                # U-Net (4 pools) needs H,W divisible by 16; pad up then crop back
                ph = (16 - th % 16) % 16
                pw = (16 - tw % 16) % 16
                if ph or pw:
                    patch = torch.nn.functional.pad(patch, (0, pw, 0, ph), mode="reflect")
                out = model(patch).float()[:, :, :th, :tw]
                win = _cosine_window(th, tw, device)
                acc[:, :, y:y + th, x:x + tw] += out * win
                wsum[:, :, y:y + th, x:x + tw] += win

    nir = (acc / wsum.clamp_min(1e-6))[:, :, pad:pad + H, pad:pad + W][0, 0]
    if is_cupy:
        import cupy as cp
        arr = cp.from_dlpack(nir.contiguous())
    else:
        arr = nir.detach().cpu().numpy()
    # the U-Net's transposed convs leave faint checkerboard texture; a light blur
    # removes it (NIR drives a slowly-varying veg signal, so this costs nothing).
    if smooth_sigma and smooth_sigma > 0:
        from ..backend import gaussian
        arr = gaussian(arr, smooth_sigma)
    return arr
