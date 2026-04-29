"""
dgs_to_ply_shaded.py
====================
Convert a .dgs Gaussian checkpoint to a 3DGS PLY with shading baked into SH:
  - Ambient colour in DC.
  - Henyey-Greenstein phase function in bands 1..3 via closed-form SH expansion.
  - Per-Gaussian self-shadow transmittance T_i from the splatter's own
    prefix-sum light pipeline (exactly matches splatter behaviour).
  - Physical opacity from the density-scale line-integral formula.

Sun direction defaults to config_used.yaml's rendering.sun_direction; override
with --sun-dir.  Since T_i depends on the sun direction at bake time, viewers
rotating the scene will see the shadows follow the rotation (not re-light).

Usage:
    python tools/dgs_to_ply_shaded.py results/exp_07b_capstone_native/aggressive_bunny
    python tools/dgs_to_ply_shaded.py --dgs path.dgs --vdb path.vdb \
        --config path.yaml --sun-dir "0.5,-1.0,0.5" --output shaded.ply
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))


# --- SH constants (Inria 3DGS convention; must match the PLY loader) -------
SH_C0 = 0.28209479177387814  # 1/(2*sqrt(pi))
SH_C1 = 0.4886025119029199   # sqrt(3/(4*pi))
SH_C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
         -1.0925484305920792, 0.5462742152960396]
SH_C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
         0.3731763325901154, -0.4570457994644658, 1.445305721320277,
         -0.5900435899266435]


def inria_sh_basis_16(d):
    """Evaluate the 16 Inria-convention SH basis functions at unit vector d.
    Returns array of length 16: [DC, band1 (3), band2 (5), band3 (7)].
    If d is unit, 2zz-xx-yy == 3z^2-1 etc.; we use the polynomial form that
    matches Inria's evaluator so the basis is exact regardless.
    """
    x, y, z = float(d[0]), float(d[1]), float(d[2])
    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z
    b = np.zeros(16, dtype=np.float64)
    b[0]  = SH_C0
    b[1]  = -SH_C1 * y
    b[2]  =  SH_C1 * z
    b[3]  = -SH_C1 * x
    b[4]  = SH_C2[0] * xy
    b[5]  = SH_C2[1] * yz
    b[6]  = SH_C2[2] * (2.0 * zz - xx - yy)
    b[7]  = SH_C2[3] * xz
    b[8]  = SH_C2[4] * (xx - yy)
    b[9]  = SH_C3[0] * y * (3.0 * xx - yy)
    b[10] = SH_C3[1] * xy * z
    b[11] = SH_C3[2] * y * (4.0 * zz - xx - yy)
    b[12] = SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy)
    b[13] = SH_C3[4] * x * (4.0 * zz - xx - yy)
    b[14] = SH_C3[5] * z * (xx - yy)
    b[15] = SH_C3[6] * x * (xx - 3.0 * yy)
    return b


def resolve_inputs(result_dir, dgs_path, vdb_path, config_path):
    if result_dir is not None:
        rd = Path(result_dir).resolve()
        if not rd.is_dir():
            sys.exit(f"Not a directory: {rd}")
        if dgs_path is None:
            dgs_path = rd / "final_gaussians.dgs"
        if config_path is None:
            config_path = rd / "config_used.yaml"
        default_output = rd / "shaded.ply"
    else:
        default_output = Path.cwd() / "shaded.ply"

    if dgs_path is None or config_path is None:
        sys.exit("Need --dgs and --config (or a result_dir).")

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


class StubCam:
    def get_gpu_data(self, _aspect):
        d = np.zeros(16, dtype=np.float32)
        d[0:3]   = (0.0, 0.0, 2.5)
        d[4:7]   = (0.0, 0.0, -1.0)
        d[8:11]  = (0.667, 0.0, 0.0)
        d[12:15] = (0.0, 0.5, 0.0)
        return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result_dir", nargs="?", help="Result directory with .dgs + config_used.yaml")
    ap.add_argument("--dgs", default=None)
    ap.add_argument("--vdb", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--output", default=None, help="Output .ply path")
    ap.add_argument("--sun-dir", default=None,
                    help="Sun direction override 'x,y,z'. Default: from config.")
    args = ap.parse_args()

    dgs, vdb, cfg_path, default_out, cfg = resolve_inputs(
        args.result_dir, args.dgs, args.vdb, args.config
    )
    out_path = Path(args.output) if args.output else default_out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Bake sun direction
    if args.sun_dir:
        bake_sun = np.asarray([float(x) for x in args.sun_dir.split(",")], dtype=np.float64)
    else:
        bake_sun = np.asarray(cfg["rendering"]["sun_direction"], dtype=np.float64)
    bake_sun = bake_sun / (np.linalg.norm(bake_sun) + 1e-8)

    print(f"[dgs->ply] dgs    = {dgs}")
    print(f"[dgs->ply] output = {out_path}")
    print(f"[dgs->ply] bake sun = [{bake_sun[0]:.3f}, {bake_sun[1]:.3f}, {bake_sun[2]:.3f}]")

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
    from save_ply import compute_physical_opacity, CloudLightingConfig, N_SH_REST

    # Load VDB (needed to get correct vol_min/vol_max; we still dense-sample
    # because convert_grid_to_dense_volume returns them together — a later
    # optimisation could skip the sampling loop since we don't actually use
    # the volume texture here).
    print("[dgs->ply] Loading VDB...")
    grid = load_vdb_grid(str(vdb))
    if grid is None:
        sys.exit(f"Failed to load VDB: {vdb}")
    up_axis = vol_cfg.get("up_axis", "+Y")
    vol_min, vol_max, vol_data, _, resolved_size = convert_grid_to_dense_volume(
        grid, VOL_SIZE_CFG, up_axis_name=up_axis, use_native_size=USE_NATIVE
    )
    if resolved_size != VOL_SIZE_CFG:
        _app.VOL_SIZE = resolved_size

    # --- Renderer (only for the light pipeline — we don't need actual rendering) ---
    renderer = Renderer(device, vol_data)
    renderer.resize(64, 64)  # output texture size irrelevant, we never sample it

    settings = Settings()
    apply_config_to_settings(cfg, settings)
    settings.sun_direction = bake_sun.tolist()

    gaussians = load_gaussians(str(dgs))
    N = len(gaussians)
    print(f"[dgs->ply] {N:,} Gaussians")

    renderer.gaussian_count  = N
    renderer.gaussian_buffer = device.create_buffer(
        element_count=N,
        struct_size=PARAMS_PER_GAUSSIAN * 4,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        memory_type=spy.MemoryType.device_local,
        data=gaussians,
    )
    renderer._vol_min = vol_min
    renderer._vol_max = vol_max
    renderer.init_training()

    # Invoke render_splat once to run the full light pipeline; cached
    # light_dirty=True on first call forces the sort + scan.
    print("[dgs->ply] Running splatter light pipeline...")
    renderer.render_splat(StubCam(), settings)

    # Read per-Gaussian transmittances back to CPU
    T = renderer.splat_light_transmittance_buf.to_numpy().view(np.float32)[:N].astype(np.float64)
    print(f"[dgs->ply] T_i: min={T.min():.4f} mean={T.mean():.4f} max={T.max():.4f}")

    # --- SH coefficient synthesis ---------------------------------------------
    g       = float(settings.phase_g)
    albedo  = np.asarray(settings.smoke_color,       dtype=np.float64)
    sun_c   = np.asarray(settings.sun_color_base,    dtype=np.float64) * float(settings.sun_intensity)
    amb_c   = np.asarray(settings.ambient_color_base, dtype=np.float64) * float(settings.ambient_intensity)

    print(f"[dgs->ply] g={g:.3f}  albedo={albedo.tolist()}")
    print(f"[dgs->ply] sun   ={sun_c.tolist()}")
    print(f"[dgs->ply] ambient={amb_c.tolist()}")

    # basis_n evaluated at bake_sun, one vector shared across all Gaussians
    basis = inria_sh_basis_16(bake_sun)  # (16,)

    # g^l per coefficient index (n=0..15): DC=1, band1=g, band2=g^2, band3=g^3
    g_exp = np.array(
        [1.0] + [g]*3 + [g*g]*5 + [g**3]*7, dtype=np.float64
    )  # (16,)

    # Per-channel scalars
    as_ch = albedo * sun_c   # (3,)  albedo ⊙ sunColor
    am_ch = albedo * amb_c   # (3,)  albedo ⊙ ambient

    # DC coefficient per Gaussian per channel (Inria convention: f_dc stores c_00,
    # viewer multiplies by Y_00 = SH_C0):
    #   c_00 = (a*s*T) * Y_00(sunDir) + (a*amb) / Y_00
    #        = (a*s*T) * SH_C0        + (a*amb) / SH_C0
    f_dc = np.empty((N, 3), dtype=np.float32)
    for ch in range(3):
        f_dc[:, ch] = (as_ch[ch] * T * SH_C0 + am_ch[ch] / SH_C0).astype(np.float32)

    # Rest coefficients (n=1..15), layout per Gaussian:
    #   [R_0..R_14,  G_0..G_14,  B_0..B_14]  (Inria flatten order)
    # c_{n,ch} = (a*s*T)_ch * g^{l_n} * basis_n(sunDir)
    coef_vec = (g_exp[1:16] * basis[1:16]).astype(np.float64)  # (15,)
    f_rest = np.zeros((N, N_SH_REST), dtype=np.float32)
    for ch in range(3):
        off = ch * 15
        f_rest[:, off:off + 15] = (as_ch[ch] * T[:, None] * coef_vec[None, :]).astype(np.float32)

    # --- Geometry + opacity (match save_ply.py) -------------------------------
    pos    = gaussians[:, 0:3].astype(np.float32)
    scale  = gaussians[:, 3:6].astype(np.float32)
    quat   = gaussians[:, 6:10].astype(np.float32)
    weight = gaussians[:, 10].astype(np.float32)

    lighting_cfg = CloudLightingConfig(density_scale=float(settings.density_scale))
    alpha = compute_physical_opacity(
        weight.astype(np.float64), scale.astype(np.float64),
        lighting_cfg.density_scale,
        lighting_cfg.opacity_floor,
        lighting_cfg.opacity_ceil,
    )
    opacity_logit = np.log(alpha / (1.0 - alpha)).astype(np.float32)

    log_scale = np.log(np.clip(scale, 1e-8, None)).astype(np.float32)

    # Quaternion (x,y,z,w) -> (w,x,y,z) per 3DGS convention, renormalised
    quat_wxyz = np.stack([quat[:, 3], quat[:, 0], quat[:, 1], quat[:, 2]], axis=1)
    norms = np.linalg.norm(quat_wxyz, axis=1, keepdims=True)
    quat_wxyz = (quat_wxyz / np.where(norms < 1e-8, 1.0, norms)).astype(np.float32)

    normals = np.zeros((N, 3), dtype=np.float32)

    prop_names = (
        ["x", "y", "z"]
        + ["nx", "ny", "nz"]
        + ["f_dc_0", "f_dc_1", "f_dc_2"]
        + [f"f_rest_{i}" for i in range(N_SH_REST)]
        + ["opacity"]
        + ["scale_0", "scale_1", "scale_2"]
        + ["rot_0", "rot_1", "rot_2", "rot_3"]
    )

    data = np.concatenate(
        [pos, normals, f_dc, f_rest, opacity_logit[:, None], log_scale, quat_wxyz],
        axis=1,
    ).astype(np.float32)

    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"comment VDBaussian shaded export {N} splats sun="
        f"[{bake_sun[0]:.3f},{bake_sun[1]:.3f},{bake_sun[2]:.3f}] g={g:.3f} "
        f"{datetime.now().isoformat(timespec='seconds')}\n"
        f"element vertex {N}\n"
        + "\n".join(f"property float {n}" for n in prop_names)
        + "\nend_header\n"
    )

    with open(out_path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[dgs->ply] wrote {N:,} Gaussians -> {out_path}  ({size_mb:.2f} MB)")
    print(f"[dgs->ply] alpha : min={alpha.min():.3f} mean={alpha.mean():.3f} max={alpha.max():.3f}")
    # Approximate visible colour = f_dc * SH_C0 (DC-only viewing)
    approx = f_dc * SH_C0
    print(f"[dgs->ply] DC rgb: "
          f"R min={approx[:,0].min():.3f} mean={approx[:,0].mean():.3f} max={approx[:,0].max():.3f}  "
          f"G mean={approx[:,1].mean():.3f}  B mean={approx[:,2].mean():.3f}")


if __name__ == "__main__":
    main()
