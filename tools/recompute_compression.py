"""
recompute_compression.py
========================
Walks every results/exp_*/batch_summary.json (and the matching per-experiment
results.json files) and rewrites the compression block so that the ratio is
computed against a Blosc-compressed sparse VDB at the same resolution that
training actually used. Also stores the actual training resolution and the
canonical shape name as first-class fields, so the chapter5 tables can pull
them directly.

Source of truth for the resampled-VDB byte counts:
    results/_meta/resampled_vdb_bytes.json
    (produced by tools/measure_resampled_vdb.py)

What gets added to each experiment's `compression` block:
    shape:                          canonical shape name (cloud_white, ...)
    training_resolution:            int, the actual N for the NxNxN volume
    resampled_vdb_bytes:            int, sparse Blosc VDB at training_resolution
    ratio_vs_resampled_vdb:         float, resampled_vdb_bytes / gauss_bytes

The legacy ratio_vs_volume / ratio_vs_vdb fields are LEFT IN PLACE for
historical reference but the chapter5 prose now reads ratio_vs_resampled_vdb.

Top-level training_resolution and shape are also mirrored on the experiment
entry itself (one level up from compression) so the chapter generator does
not have to dig into the compression sub-dict.

Usage:
    python tools/recompute_compression.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import yaml


REPO     = Path(__file__).resolve().parent.parent
RESULTS  = REPO / "results"
META     = RESULTS / "_meta" / "resampled_vdb_bytes.json"


# Map a vdb_file path (as it appears in YAML) to a canonical shape name.
# We compare on the basename for robustness, since some configs use absolute
# Windows paths and some use bare filenames.
VDB_BASENAME_TO_SHAPE = {
    "cloud_04_variant_0000.vdb":   "cloud_white",
    "tornado_0109.vdb":            "tornado",
    "bunny_cloud.vdb":             "bunny_cloud",
    "embergen_smoke_plume_a_80.vdb":"smoke_plume",
    "smallCampfire_0122.vdb":      "campfire",
    "smoke2.vdb":                  "smoke2",
    "explosion.vdb":               "explosion",
}


def shape_from_vdb_path(vdb_path: str) -> str | None:
    if not vdb_path:
        return None
    base = os.path.basename(vdb_path).strip()
    return VDB_BASENAME_TO_SHAPE.get(base)


def load_meta() -> dict:
    if not META.exists():
        sys.exit(f"!! missing {META}; run tools/measure_resampled_vdb.py first")
    with open(META) as f:
        return json.load(f)


def lookup_resampled_bytes(meta: dict, shape: str, resolution: int) -> int | None:
    if shape not in meta:
        return None
    shape_meta = meta[shape]
    key = str(resolution)
    if key in shape_meta:
        return int(shape_meta[key]["bytes"])
    # Also handle the "native" entry by matching its size
    nat = shape_meta.get("native")
    if nat and int(nat["size"]) == resolution:
        return int(nat["bytes"])
    return None


def resolve_training_resolution(cfg: dict, shape: str, meta: dict) -> int | None:
    """Look at a config_used.yaml dict and figure out the actual NxNxN size
    that was trained against."""
    vol = cfg.get("volume") or {}
    if vol.get("use_native_size"):
        nat = meta.get(shape, {}).get("native")
        if nat:
            return int(nat["size"])
        return None
    vs = vol.get("vol_size")
    return int(vs) if vs is not None else None


def shape_from_config(cfg: dict) -> str | None:
    vol = cfg.get("volume") or {}
    return shape_from_vdb_path(vol.get("vdb_file", ""))


def update_compression_block(comp: dict, shape: str, resolution: int,
                             resampled_bytes: int) -> dict:
    gauss_bytes = comp.get("gaussian_data_bytes") or 0
    comp = dict(comp)  # don't mutate the input
    comp["shape"] = shape
    comp["training_resolution"] = resolution
    comp["resampled_vdb_bytes"] = resampled_bytes
    comp["ratio_vs_resampled_vdb"] = (
        round(resampled_bytes / gauss_bytes, 3) if gauss_bytes > 0 else 0.0
    )
    return comp


def patch_results_json(out_dir: Path, shape: str, resolution: int,
                       resampled_bytes: int, dry: bool):
    rj = out_dir / "results.json"
    if not rj.exists():
        return
    with open(rj) as f:
        data = json.load(f)
    comp = data.get("compression") or {}
    if not comp:
        return
    new_comp = update_compression_block(comp, shape, resolution, resampled_bytes)
    if new_comp == comp:
        return
    data["compression"] = new_comp
    data["shape"] = shape
    data["training_resolution"] = resolution
    if not dry:
        with open(rj, "w") as f:
            json.dump(data, f, indent=2)


def patch_batch_summary(bs_path: Path, meta: dict, dry: bool) -> tuple[int, int]:
    """Returns (n_updated, n_skipped)."""
    with open(bs_path) as f:
        summary = json.load(f)

    n_updated = 0
    n_skipped = 0

    for exp in summary.get("experiments", []):
        out_dir_str = exp.get("out_dir")
        if not out_dir_str:
            n_skipped += 1
            continue
        out_dir = Path(out_dir_str)
        cfg_path = out_dir / "config_used.yaml"
        if not cfg_path.exists():
            print(f"  ?? no config_used.yaml in {out_dir}")
            n_skipped += 1
            continue

        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        shape = shape_from_config(cfg)
        if shape is None:
            print(f"  ?? unknown shape in {cfg_path}")
            n_skipped += 1
            continue

        resolution = resolve_training_resolution(cfg, shape, meta)
        if resolution is None:
            print(f"  ?? no training resolution in {cfg_path}")
            n_skipped += 1
            continue

        resampled_bytes = lookup_resampled_bytes(meta, shape, resolution)
        if resampled_bytes is None:
            print(f"  !! no resampled_vdb_bytes for {shape}@{resolution} (run measure_resampled_vdb.py)")
            n_skipped += 1
            continue

        comp = exp.get("compression") or {}
        if not comp:
            n_skipped += 1
            continue

        new_comp = update_compression_block(comp, shape, resolution, resampled_bytes)
        exp["compression"] = new_comp
        exp["shape"] = shape
        exp["training_resolution"] = resolution
        n_updated += 1

        # Mirror into the per-experiment results.json
        patch_results_json(out_dir, shape, resolution, resampled_bytes, dry)

    if n_updated > 0 and not dry:
        with open(bs_path, "w") as f:
            json.dump(summary, f, indent=2)

    return n_updated, n_skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    meta = load_meta()

    total_updated = 0
    total_skipped = 0
    for bs_path in sorted(RESULTS.glob("exp_*/batch_summary.json")):
        print(f"\n[{bs_path.parent.name}]")
        u, s = patch_batch_summary(bs_path, meta, args.dry_run)
        print(f"  updated {u}  skipped {s}")
        total_updated += u
        total_skipped += s

    print(f"\nTOTAL: updated {total_updated}  skipped {total_skipped}"
          f"{'  (dry-run, nothing written)' if args.dry_run else ''}")


if __name__ == "__main__":
    main()
