"""
save_load_gaussians.py — Save/load Gaussian parameters in native .dgs format.

File layout (little-endian):
    4 bytes   magic   b"DGS1"
    4 bytes   uint32  gaussian_count
    4 bytes   uint32  params_per_gaussian (11)
    N*11*4 bytes float32  raw parameter data [pos(3), scale(3), quat(4), weight(1)]
"""

import numpy as np
import struct
from pathlib import Path

MAGIC = b"DGS1"
HEADER_SIZE = 12  # 4 + 4 + 4


def save_gaussians(params: np.ndarray, path: str) -> int:
    """
    Save Gaussian parameters to a .dgs file.

    Args:
        params: float32 array of shape (N, 11) — raw Gaussian parameters.
        path: output file path.

    Returns:
        File size in bytes.
    """
    params = np.ascontiguousarray(params, dtype=np.float32)
    if params.ndim == 1:
        params = params.reshape(-1, 11)
    n, ppg = params.shape
    assert ppg == 11, f"Expected 11 params per Gaussian, got {ppg}"

    path = Path(path)
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<II", n, ppg))
        f.write(params.tobytes())

    size = path.stat().st_size
    print(f"[DGS] Saved {n:,} Gaussians -> {path}  ({size / 1024:.1f} KB)")
    return size


def load_gaussians(path: str) -> np.ndarray:
    """
    Load Gaussian parameters from a .dgs file.

    Returns:
        float32 array of shape (N, 11).
    """
    path = Path(path)
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError(f"Not a DGS file (magic={magic!r}, expected {MAGIC!r})")
        n, ppg = struct.unpack("<II", f.read(8))
        data = np.frombuffer(f.read(n * ppg * 4), dtype=np.float32).reshape(n, ppg)

    print(f"[DGS] Loaded {n:,} Gaussians from {path}  ({path.stat().st_size / 1024:.1f} KB)")
    return data.copy()
