"""
measure_resampled_vdb.py
========================
For each of the 5 source VDB shapes, build a dense float volume at every
training resolution that the chapter5 experiments use, write that volume
back as a sparse Blosc-compressed VDB, and record the on-disk byte count.

The output is the apples-to-apples compression baseline: "the same data the
trainer fits, stored as a sparse VDB at the same resolution."

Sampling strategy
-----------------
We use grid.copyToArray() to dump the entire active bbox into a dense numpy
array in one C++ call (fast), then do nearest-neighbour resampling into a
cubic NxNxN array using vectorised numpy index math. Cubic embedding
preserves the source aspect ratio (zero-padding outside the longest axis),
matching the convention used by app.convert_grid_to_dense_volume.

Output
------
results/_meta/resampled_vdb_bytes.json:
    {
      "cloud_white": {
         "128": {"bytes": 41234, "active_voxels": 12345, "native_size": 187},
         "196": {...},
         "256": {...},
         "native": {"size": 187, "bytes": ...}
      },
      ...
    }

Usage
-----
    python tools/measure_resampled_vdb.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import openvdb as vdb


# 5 shapes used throughout chapter5. Paths and up_axis taken verbatim from
# config/exp_01_baselines.yaml; up_axis only matters for the trainer's
# coordinate system, NOT for byte count, so we ignore it here and just sample
# in source-VDB index space (this is invariant under axis permutation).
SHAPES = {
    "cloud_white": r"C:\Users\ade0n\Projects\VDBaussian\cloud_04_variant_0000.vdb",
    "tornado":     r"C:\Users\ade0n\Downloads\TornadoLoopingVDB\TornadoLooping\TornadoVDB\tornado_0109.vdb",
    "bunny_cloud": r"C:\Users\ade0n\Downloads\bunny_cloud.vdb",
    "smoke_plume": r"C:\Users\ade0n\Downloads\Smoke_Plume_01\Smoke_Plume_01\embergen_smoke_plume_a_80.vdb",
    "campfire":    r"C:\Users\ade0n\Downloads\SmallCampfireVDB\smallCampfire\smallCampfireVDB\smallCampfire_0122.vdb",
    "smoke2":      r"C:\Users\ade0n\Downloads\smoke2.vdb",
    "explosion":   r"C:\Users\ade0n\Downloads\explosion.vdb",
}

# Training resolutions that appear in the experiments. "native" is per-shape.
FIXED_RESOLUTIONS = [128, 196, 256, 460]  # 460 = exp_07c_capstone_large

OUT_PATH = Path(r"C:\Users\ade0n\Projects\VDBaussian\results\_meta\resampled_vdb_bytes.json")


def load_grid(path: str):
    raw = vdb.readAll(path)
    grid = raw[0][0] if isinstance(raw, (list, tuple)) else raw
    return grid


def dense_native_array(grid) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Dump active bbox into a dense numpy array via copyToArray.
    Returns (array, native_dims). array shape == native_dims.
    """
    bbox = grid.evalActiveVoxelBoundingBox()
    bb_min = np.array(bbox[0], dtype=np.int64)
    bb_max = np.array(bbox[1], dtype=np.int64)
    dims = tuple(int(d) for d in (bb_max - bb_min + 1))
    arr = np.zeros(dims, dtype=np.float32)
    grid.copyToArray(arr, ijk=tuple(int(x) for x in bb_min))
    return arr, dims


def resample_cubic_nearest(native: np.ndarray, target_size: int) -> np.ndarray:
    """Resample a non-cubic dense array into a cubic NxNxN array using
    nearest-neighbour sampling. The source is centred inside a cube whose side
    equals max(native_dims), so the aspect ratio is preserved and the
    zero-padding regions outside the source extent become background voxels
    (matches app.convert_grid_to_dense_volume's "max-axis cube" embedding).
    """
    nd = np.array(native.shape, dtype=np.float64)  # (Dx, Dy, Dz)
    cube_side = float(nd.max())
    half = cube_side / 2.0
    centre = nd / 2.0  # source-array centre, in source index units

    # For each output axis, compute the source index of every output voxel.
    # Output voxel j ∈ [0, N) maps to local coord ((j+0.5)/N - 0.5) * cube_side
    # which lies in [-half, +half]. Add the source centre to get the source
    # index. Out-of-bounds samples become 0 via the mask below.
    coords = (np.arange(target_size, dtype=np.float64) + 0.5) / target_size
    local = (coords - 0.5) * cube_side  # (-half .. +half)

    out = np.zeros((target_size, target_size, target_size), dtype=np.float32)

    # Per-axis source indices (float, then int)
    sx_f = local + centre[0]
    sy_f = local + centre[1]
    sz_f = local + centre[2]

    sx = np.floor(sx_f).astype(np.int64)
    sy = np.floor(sy_f).astype(np.int64)
    sz = np.floor(sz_f).astype(np.int64)

    mask_x = (sx >= 0) & (sx < native.shape[0])
    mask_y = (sy >= 0) & (sy < native.shape[1])
    mask_z = (sz >= 0) & (sz < native.shape[2])

    sxc = np.clip(sx, 0, native.shape[0] - 1)
    syc = np.clip(sy, 0, native.shape[1] - 1)
    szc = np.clip(sz, 0, native.shape[2] - 1)

    # Vectorised gather: native[sxc[:,None,None], syc[None,:,None], szc[None,None,:]]
    gathered = native[sxc[:, None, None], syc[None, :, None], szc[None, None, :]]
    valid = mask_x[:, None, None] & mask_y[None, :, None] & mask_z[None, None, :]
    out = np.where(valid, gathered, 0.0).astype(np.float32)
    return out


def normalise(arr: np.ndarray) -> np.ndarray:
    m = float(arr.max())
    if m > 0.0:
        return (arr / m).astype(np.float32)
    return arr


def save_sparse_and_measure(arr: np.ndarray, label: str) -> tuple[int, int]:
    """Write `arr` as a sparse FloatGrid to a temp .vdb and return
    (file_bytes, active_voxel_count). Voxels equal to 0.0 within tolerance
    become inactive (background)."""
    grid = vdb.FloatGrid()
    grid.name = label
    # tolerance=0.0 → only exact zeros are dropped to background
    grid.copyFromArray(arr, ijk=(0, 0, 0), tolerance=0.0)
    grid.pruneInactive()
    active = int(grid.activeVoxelCount())

    # Use a dedicated tmp dir so the path is short and stable on Windows.
    with tempfile.TemporaryDirectory() as td:
        tmp = os.path.join(td, f"{label}.vdb")
        vdb.write(tmp, grid)
        size = os.path.getsize(tmp)
    return size, active


def native_size_for_grid(grid) -> int:
    """Match the convention in app._get_native_vdb_size: longest active-bbox
    axis."""
    bbox = grid.evalActiveVoxelBoundingBox()
    bb_min = np.array(bbox[0], dtype=np.int64)
    bb_max = np.array(bbox[1], dtype=np.int64)
    return int((bb_max - bb_min + 1).max())


def measure_shape(name: str, path: str) -> dict:
    print(f"\n=== {name} ===")
    print(f"  Source: {path}")
    if not Path(path).exists():
        print(f"  !! file not found, skipping")
        return {"error": "file not found", "path": path}

    src_bytes = os.path.getsize(path)
    grid = load_grid(path)
    if grid is None:
        return {"error": "load failed", "path": path}

    native_arr, native_dims = dense_native_array(grid)
    native_arr = normalise(native_arr)
    native_size = native_size_for_grid(grid)
    print(f"  Native dims (active bbox): {native_dims}, longest axis: {native_size}")
    print(f"  Source .vdb on disk: {src_bytes:,} bytes")

    out: dict = {
        "source_vdb_path":  path,
        "source_vdb_bytes": src_bytes,
        "native_dims":      list(native_dims),
        "native_size":      native_size,
    }

    # Fixed resolutions
    for res in FIXED_RESOLUTIONS:
        resampled = resample_cubic_nearest(native_arr, res)
        # No second normalisation: native_arr is already in [0,1].
        bytes_, active = save_sparse_and_measure(resampled, f"{name}_{res}")
        fill = 100.0 * active / (res ** 3)
        print(f"  {res:>4}^3: {bytes_:>10,} B   active {active:>10,} ({fill:5.2f}%)")
        out[str(res)] = {
            "size":          res,
            "bytes":         bytes_,
            "active_voxels": active,
            "fill_percent":  round(fill, 4),
        }

    # Native resolution (only if different from the fixed ones)
    if native_size not in FIXED_RESOLUTIONS:
        resampled = resample_cubic_nearest(native_arr, native_size)
        bytes_, active = save_sparse_and_measure(resampled, f"{name}_native")
        fill = 100.0 * active / (native_size ** 3)
        print(f"  nat^3 ({native_size}): {bytes_:>10,} B   active {active:>10,} ({fill:5.2f}%)")
    else:
        # native already covered by fixed pass
        existing = out[str(native_size)]
        bytes_  = existing["bytes"]
        active  = existing["active_voxels"]
    out["native"] = {
        "size":          native_size,
        "bytes":         bytes_,
        "active_voxels": active,
        "fill_percent":  round(100.0 * active / (native_size ** 3), 4),
    }

    return out


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, path in SHAPES.items():
        results[name] = measure_shape(name, path)

    # Also record metadata about the openvdb build so the numbers are reproducible.
    results["_meta"] = {
        "openvdb_version": vdb.LIBRARY_VERSION,
        "file_format_version": vdb.FILE_FORMAT_VERSION,
        "note": (
            "bytes are sizes of single-grid .vdb files written via the "
            "default vdb.write() compression (whatever the OpenVDB build "
            "uses by default — typically Blosc + active-mask). Active "
            "voxels are counted after pruneInactive() with tolerance 0."
        ),
    }

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
