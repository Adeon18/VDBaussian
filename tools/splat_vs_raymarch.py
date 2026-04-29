"""
splat_vs_raymarch.py
====================
Render a trained Gaussian checkpoint three ways on the same cameras and compare:

    A = raymarch(reference VDB volume)      — ground-truth reference
    B = raymarch(rasterized Gaussian vol)   — fitting quality through the raymarch path
    C = splatter(Gaussians)                 — the deployed fast path

Writes per-camera PNGs (A, B, C, |A-C|, |B-C|, montage) and a summary.json
with per-view and mean luma RMSE / PSNR for d(A, C) and d(B, C).

Usage:
    # Point at a result directory (auto-discovers .dgs, config_used.yaml, VDB)
    python tools/splat_vs_raymarch.py results/exp_09_compression_capstone/calm_bunny

    # Explicit paths
    python tools/splat_vs_raymarch.py \
        --dgs path/to/final_gaussians.dgs \
        --vdb path/to/reference.vdb \
        --config path/to/config_used.yaml \
        --output-dir path/to/out
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))


def resolve_inputs(result_dir, dgs_path, vdb_path, config_path):
    """Fill in missing paths from result_dir. Returns resolved (dgs, vdb, cfg_path, out_default, cfg_dict)."""
    if result_dir is not None:
        rd = Path(result_dir).resolve()
        if not rd.is_dir():
            sys.exit(f"Not a directory: {rd}")
        if dgs_path is None:
            dgs_path = rd / "final_gaussians.dgs"
        if config_path is None:
            config_path = rd / "config_used.yaml"
        default_output = rd / "splat_comparison"
    else:
        default_output = Path.cwd() / "splat_comparison"

    if dgs_path is None or config_path is None:
        sys.exit("Need --dgs and --config (or a result_dir that contains them).")

    dgs_path = Path(dgs_path).resolve()
    config_path = Path(config_path).resolve()
    for p in (dgs_path, config_path):
        if not p.exists():
            sys.exit(f"Missing: {p}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if vdb_path is None:
        vdb_path = cfg.get("volume", {}).get("vdb_file")
    if vdb_path is None:
        sys.exit("No VDB path in config; pass --vdb.")
    vdb_path = Path(vdb_path)
    if not vdb_path.exists():
        sys.exit(f"Missing VDB: {vdb_path}")

    return dgs_path, vdb_path, config_path, default_output, cfg


def cameras_from_config(cfg, fallback_aspect=1440.0 / 1080.0, fallback_radius=2.5, radius_override=None):
    """Pull snapshot_cameras from config and rescale positions to snapshot_radius
    (matches train_headless behavior). `radius_override` replaces the config value."""
    snaps = cfg.get("output", {}).get("snapshot_cameras")
    snap_radius = radius_override if radius_override is not None \
        else cfg.get("output", {}).get("snapshot_radius")

    if snaps:
        out = []
        for c in snaps:
            cam = {k: list(c[k]) for k in ("pos", "front", "right", "up")}
            if snap_radius is not None:
                pos = np.asarray(cam["pos"], dtype=np.float32)
                r = float(np.linalg.norm(pos))
                if r > 0:
                    cam["pos"] = list(pos * (snap_radius / r))
            out.append((c["name"], cam))
        return out

    r = radius_override if radius_override is not None else fallback_radius
    a = fallback_aspect * 0.5
    return [
        ("front",  dict(pos=(0, 0,  r), front=(0, 0, -1), right=( a, 0, 0),  up=(0, 0.5, 0))),
        ("back",   dict(pos=(0, 0, -r), front=(0, 0,  1), right=(-a, 0, 0),  up=(0, 0.5, 0))),
        ("right",  dict(pos=( r, 0, 0), front=(-1, 0, 0), right=(0, 0, -a),  up=(0, 0.5, 0))),
        ("left",   dict(pos=(-r, 0, 0), front=( 1, 0, 0), right=(0, 0,  a),  up=(0, 0.5, 0))),
        ("top",    dict(pos=(0,  r, 0), front=(0, -1, 0), right=( a, 0, 0),  up=(0, 0, -0.5))),
        ("bottom", dict(pos=(0, -r, 0), front=(0,  1, 0), right=( a, 0, 0),  up=(0, 0,  0.5))),
    ]


class StubCam:
    """Quacks like a Camera — only get_gpu_data is called by the renderer."""
    def __init__(self, cam_dict):
        d = np.zeros(16, dtype=np.float32)
        d[0:3]   = np.asarray(cam_dict["pos"],   dtype=np.float32)
        d[4:7]   = np.asarray(cam_dict["front"], dtype=np.float32)
        d[8:11]  = np.asarray(cam_dict["right"], dtype=np.float32)
        d[12:15] = np.asarray(cam_dict["up"],    dtype=np.float32)
        self._data = d

    def get_gpu_data(self, _aspect):
        return self._data


def luma_rmse_psnr(a_u8, b_u8):
    """Luma (BT.709) RMSE + PSNR between two HxWx{3,4} uint8 arrays. Returns (rmse, psnr_dB)."""
    af = a_u8[..., :3].astype(np.float32) / 255.0
    bf = b_u8[..., :3].astype(np.float32) / 255.0
    w  = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    la = (af * w).sum(-1)
    lb = (bf * w).sum(-1)
    mse = float(((la - lb) ** 2).mean())
    rmse = math.sqrt(mse)
    psnr = 10.0 * math.log10(1.0 / mse) if mse > 1e-12 else 100.0
    return rmse, psnr


def rgb_rmse(a_u8, b_u8):
    af = a_u8[..., :3].astype(np.float32) / 255.0
    bf = b_u8[..., :3].astype(np.float32) / 255.0
    return math.sqrt(float(((af - bf) ** 2).mean()))


def make_diff_heatmap(a_u8, b_u8, scale):
    """|luma(a)-luma(b)| * scale as a viridis heatmap, HxWx3 uint8."""
    af = a_u8[..., :3].astype(np.float32) / 255.0
    bf = b_u8[..., :3].astype(np.float32) / 255.0
    w  = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    d = np.abs((af * w).sum(-1) - (bf * w).sum(-1))
    d = np.clip(d * scale, 0.0, 1.0)
    try:
        from matplotlib import cm
        rgb = (cm.viridis(d)[..., :3] * 255).astype(np.uint8)
    except Exception:
        g = (d * 255).astype(np.uint8)
        rgb = np.stack([g, g, g], axis=-1)
    return rgb


def save_png(pixels, path):
    from PIL import Image
    arr = pixels
    if arr.shape[-1] == 4:
        Image.fromarray(arr, "RGBA").save(path)
    else:
        Image.fromarray(arr, "RGB").save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result_dir", nargs="?", help="Result directory with final_gaussians.dgs + config_used.yaml")
    ap.add_argument("--dgs",    type=str, default=None)
    ap.add_argument("--vdb",    type=str, default=None)
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--width",  type=int, default=None, help="Override render width (default from config metrics_2d)")
    ap.add_argument("--height", type=int, default=None, help="Override render height (default from config metrics_2d)")
    ap.add_argument("--diff-scale", type=float, default=10.0, help="Multiplier for |A-C| and |B-C| heatmaps")
    ap.add_argument("--radius", type=float, default=None,
                    help="Override camera radius (world units). Defaults to config's snapshot_radius. "
                         "Use ~3.5-4.5 if the default clips the volume.")
    args = ap.parse_args()

    dgs, vdb, cfg_path, default_out, cfg = resolve_inputs(
        args.result_dir,
        args.dgs,
        args.vdb,
        args.config,
    )
    out_dir = Path(args.output_dir) if args.output_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve resolution
    m2d = cfg.get("metrics_2d", {})
    width  = args.width  or int(m2d.get("render_width",  1440))
    height = args.height or int(m2d.get("render_height", 1080))

    print(f"[tool] dgs    = {dgs}")
    print(f"[tool] vdb    = {vdb}")
    print(f"[tool] config = {cfg_path}")
    print(f"[tool] output = {out_dir}")
    print(f"[tool] res    = {width}x{height}")

    # --- Patch app module constants BEFORE importing domain objects ---
    vol_cfg = cfg.get("volume", {})
    VOL_SIZE_CFG   = int(vol_cfg.get("vol_size", 196))
    TILE_SIZE_CFG  = int(vol_cfg.get("tile_size", 4))
    MAX_G_PER_TILE = int(vol_cfg.get("max_gaussians_per_tile", 256))
    USE_NATIVE     = bool(vol_cfg.get("use_native_size", False))

    import app as _app
    _app.VOL_SIZE               = VOL_SIZE_CFG
    _app.TILE_SIZE              = TILE_SIZE_CFG
    _app.MAX_GAUSSIANS_PER_TILE = MAX_G_PER_TILE

    import slangpy as spy
    device = spy.Device(
        enable_debug_layers=False,
        compiler_options={"include_paths": [str(HERE)]},
        type=spy.DeviceType.vulkan,
    )

    from app import (
        PARAMS_PER_GAUSSIAN,
        Renderer,
        Settings,
        convert_grid_to_dense_volume,
        load_vdb_grid,
    )
    from config_loader import apply_config_to_settings
    from save_load_gaussians import load_gaussians

    # --- Load VDB into dense volume ---
    print("[tool] Loading VDB...")
    grid = load_vdb_grid(str(vdb))
    if grid is None:
        sys.exit(f"Failed to load VDB: {vdb}")

    up_axis = vol_cfg.get("up_axis", "+Y")
    vol_min, vol_max, vol_data, _axis_remap, resolved_size = convert_grid_to_dense_volume(
        grid, VOL_SIZE_CFG, up_axis_name=up_axis, use_native_size=USE_NATIVE
    )
    if resolved_size != VOL_SIZE_CFG:
        _app.VOL_SIZE = resolved_size
        print(f"[tool] resolved VOL_SIZE = {resolved_size}")

    # --- Renderer + settings ---
    renderer = Renderer(device, vol_data)
    renderer.resize(width, height)

    settings = Settings()
    apply_config_to_settings(cfg, settings)

    # --- Load Gaussians ---
    gaussians = load_gaussians(str(dgs))
    renderer.gaussian_count  = len(gaussians)
    renderer.gaussian_buffer = device.create_buffer(
        element_count=len(gaussians),
        struct_size=PARAMS_PER_GAUSSIAN * 4,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        memory_type=spy.MemoryType.device_local,
        data=gaussians,
    )
    renderer._vol_min = vol_min
    renderer._vol_max = vol_max
    renderer.init_training()

    # Renderer starts with a 1x1x1 dummy gaussian_volume_tex to save VRAM.
    # Reallocate to full size before rasterize writes into it.
    renderer._ensure_gaussian_volume_tex()

    cmd = device.create_command_encoder()
    renderer.rasterize_gaussians(cmd, vol_min, vol_max)
    device.submit_command_buffer(cmd.finish())

    # --- Cameras ---
    cameras = cameras_from_config(cfg, fallback_aspect=width / height, radius_override=args.radius)
    used_radius = args.radius if args.radius is not None else cfg.get("output", {}).get("snapshot_radius")
    print(f"[tool] {len(cameras)} camera(s), radius={used_radius}")

    per_view = []
    for name, cam_dict in cameras:
        cam = StubCam(cam_dict)
        cam_out = out_dir / f"cam_{name}"
        cam_out.mkdir(exist_ok=True)

        # A: raymarch reference volume
        renderer.use_gaussian_volume = False
        renderer.render_main(cam, settings)
        A = renderer.screen_tex.to_numpy()

        # B: raymarch rasterized Gaussian volume
        renderer.use_gaussian_volume = True
        renderer.render_main(cam, settings)
        B = renderer.screen_tex.to_numpy()
        renderer.use_gaussian_volume = False

        # C: splatter
        renderer.render_splat(cam, settings)
        C = renderer.splat_output_tex.to_numpy()

        save_png(A, cam_out / "A_ref_raymarch.png")
        save_png(B, cam_out / "B_gaussian_raymarch.png")
        save_png(C, cam_out / "C_splatter.png")

        rmse_ac, psnr_ac = luma_rmse_psnr(A, C)
        rmse_bc, psnr_bc = luma_rmse_psnr(B, C)
        rgb_ac = rgb_rmse(A, C)
        rgb_bc = rgb_rmse(B, C)

        diff_ac = make_diff_heatmap(A, C, args.diff_scale)
        diff_bc = make_diff_heatmap(B, C, args.diff_scale)
        save_png(diff_ac, cam_out / "diff_A_vs_C.png")
        save_png(diff_bc, cam_out / "diff_B_vs_C.png")

        # Montage: A | B | C | diff(A,C) | diff(B,C)
        mont = np.concatenate([A[..., :3], B[..., :3], C[..., :3], diff_ac, diff_bc], axis=1)
        save_png(mont, cam_out / "montage.png")

        print(f"[{name:6s}]  d(A,C) RMSE={rmse_ac:.5f} PSNR={psnr_ac:6.2f} dB  |  "
              f"d(B,C) RMSE={rmse_bc:.5f} PSNR={psnr_bc:6.2f} dB")

        per_view.append({
            "camera": name,
            "d_A_C": {"rmse_luma": rmse_ac, "psnr_luma": psnr_ac, "rmse_rgb": rgb_ac},
            "d_B_C": {"rmse_luma": rmse_bc, "psnr_luma": psnr_bc, "rmse_rgb": rgb_bc},
        })

    def mean(key, sub):
        return float(np.mean([r[key][sub] for r in per_view]))

    summary = {
        "inputs": {
            "dgs":            str(dgs),
            "vdb":            str(vdb),
            "config":         str(cfg_path),
            "gaussian_count": int(renderer.gaussian_count),
        },
        "resolution": [width, height],
        "diff_scale": args.diff_scale,
        "per_view":   per_view,
        "mean": {
            "d_A_C": {k: mean("d_A_C", k) for k in ("rmse_luma", "psnr_luma", "rmse_rgb")},
            "d_B_C": {k: mean("d_B_C", k) for k in ("rmse_luma", "psnr_luma", "rmse_rgb")},
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 70)
    print(f"Mean across {len(per_view)} views:")
    print(f"  d(A,C)  splatter vs reference raymarch       "
          f"RMSE={summary['mean']['d_A_C']['rmse_luma']:.5f}  "
          f"PSNR={summary['mean']['d_A_C']['psnr_luma']:6.2f} dB")
    print(f"  d(B,C)  splatter vs Gaussian-volume raymarch "
          f"RMSE={summary['mean']['d_B_C']['rmse_luma']:.5f}  "
          f"PSNR={summary['mean']['d_B_C']['psnr_luma']:6.2f} dB")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
