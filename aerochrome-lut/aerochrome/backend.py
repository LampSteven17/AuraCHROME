"""
Array-backend abstraction: run the exact same color science on NumPy (CPU) or
CuPy (CUDA GPU), chosen per-array so there is only ever one codepath.

The transform is pure elementwise math (OKLab round-trips, smoothsteps, hue
windows) -- ~77% of conversion time -- which is the ideal GPU workload. CuPy is a
near drop-in for NumPy, but `np.<fn>(cupy_array)` does NOT dispatch to the GPU;
you must call `cp.<fn>`. So every hot function asks `xp_for(arr)` for the right
module and uses that. With no GPU (or no CuPy installed) everything transparently
stays on NumPy -- the import never fails, nothing else changes.

float64 is kept for fidelity. Note: consumer GeForce cards run float64 at a
fraction of float32 throughput, so the GPU win is real but smaller than float32
would give.
"""

import numpy as np

try:
    import cupy as _cp
    try:
        _HAS_CUDA = _cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        _HAS_CUDA = False
except Exception:  # CuPy not installed
    _cp = None
    _HAS_CUDA = False


def gpu_available():
    return _HAS_CUDA


def device_name():
    if not _HAS_CUDA:
        return "cpu (numpy)"
    try:
        props = _cp.cuda.runtime.getDeviceProperties(0)
        return "gpu: " + props["name"].decode()
    except Exception:
        return "gpu (cupy)"


def xp_for(arr):
    """Return the array module (numpy or cupy) that owns `arr`."""
    if _cp is not None:
        return _cp.get_array_module(arr)
    return np


def to_device(arr, use_gpu):
    """Move a host array onto the GPU if requested and possible, else NumPy."""
    if use_gpu and _cp is not None and _HAS_CUDA:
        return _cp.asarray(arr)
    return np.asarray(arr)


def to_numpy(arr):
    """Bring an array back to host NumPy (no-op if already NumPy)."""
    if _cp is not None and isinstance(arr, _cp.ndarray):
        return _cp.asnumpy(arr)
    return np.asarray(arr)


def gaussian(arr, sigma):
    """Gaussian blur on the array's own device (scipy on CPU, cupyx on GPU)."""
    xp = xp_for(arr)
    if _cp is not None and xp is _cp:
        from cupyx.scipy.ndimage import gaussian_filter as _gf
        return _gf(arr, sigma)
    from scipy.ndimage import gaussian_filter as _gf
    return _gf(arr, sigma)
