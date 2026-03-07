"""
save_ply.py  —  Export Gaussians to 3DGS PLY (SuperSplat compatible).

Opacity is derived PHYSICALLY from the volumetric density formula:
    alpha = 1 - exp(-weight * sigma_mean * sqrt(2*pi) * density_scale)

This is the transmittance a ray loses passing through each Gaussian,
exactly matching what your Slang raymarcher computes. The result is
a 3DGS scene that looks like the volumetric render.
"""

import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass


SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199


@dataclass
class CloudLightingConfig:
    # Must match your Settings.density_scale — this is the critical parameter
    density_scale: float = 40.0

    # Clamp computed alpha to this range so nothing is invisible or fully solid
    opacity_floor:  float = 0.02
    opacity_ceil:   float = 0.98

    # Color: depth-based self-shadowing
    bright_luminance:     float = 0.95
    dark_luminance:       float = 0.28
    depth_gamma:          float = 2.0
    top_light_strength:   float = 0.05
    shadow_warmth:        float = 0.10
    weight_influence:     float = 0.20


def compute_physical_opacity(
    weight: np.ndarray,
    scale:  np.ndarray,
    density_scale: float,
    floor: float,
    ceil:  float,
) -> np.ndarray:
    """
    alpha_i = 1 - exp(-weight_i * sigma_mean_i * sqrt(2*pi) * density_scale)

    This is the transmittance lost when a ray passes through the center of
    Gaussian i — the exact same quantity your volumetric raymarcher accumulates
    step by step. Exporting this as 3DGS opacity makes SuperSplat match your
    renderer without any fudge factors.
    """
    sigma_mean = np.mean(scale, axis=1)                          # isotropic approx
    line_integral = sigma_mean * np.sqrt(2.0 * np.pi)           # ∫ gaussian dx
    alpha = 1.0 - np.exp(-weight * line_integral * density_scale)
    return np.clip(alpha, floor, ceil)


def bake_cloud_colors(params: np.ndarray, cfg: CloudLightingConfig):
    N   = len(params)
    pos = params[:, 0:3].astype(np.float64)
    w   = params[:, 10].astype(np.float64)

    w_norm   = w / (w.sum() + 1e-12)
    centroid = (pos * w_norm[:, None]).sum(axis=0)
    diff     = pos - centroid
    var      = (w_norm[:, None] * diff ** 2).sum(axis=0)
    sigma    = np.sqrt(var) + 1e-8
    maha     = np.sqrt(np.sum((diff / sigma) ** 2, axis=1))

    p_inner = np.percentile(maha, 5)
    p_outer = np.percentile(maha, 95)
    depth_geo = 1.0 - np.clip(
        (maha - p_inner) / (p_outer - p_inner + 1e-8), 0.0, 1.0
    )

    p_w_lo = np.percentile(w, 10)
    p_w_hi = np.percentile(w, 90)
    depth_w = np.clip((w - p_w_lo) / (p_w_hi - p_w_lo + 1e-8), 0.0, 1.0)

    depth = np.clip(
        (1.0 - cfg.weight_influence) * depth_geo
        + cfg.weight_influence * (depth_geo * 0.5 + depth_w * 0.5),
        0.0, 1.0,
    )

    lum = cfg.bright_luminance - (depth ** cfg.depth_gamma) * (
        cfg.bright_luminance - cfg.dark_luminance
    )
    lum = np.clip(lum, 0.001, 0.999)

    shadow  = (1.0 - lum) * cfg.shadow_warmth
    lum_r   = np.clip(lum + shadow * 0.6, 0.001, 0.999)
    lum_g   = np.clip(lum + shadow * 0.3, 0.001, 0.999)
    lum_b   = lum.copy()
    lum_rgb = np.stack([lum_r, lum_g, lum_b], axis=1)
    dc_sh   = (lum_rgb / SH_C0).astype(np.float32)

    coeff  = cfg.top_light_strength / SH_C1
    band1  = np.zeros(9, dtype=np.float32)
    band1[0] = band1[3] = band1[6] = coeff
    rest_9 = np.tile(band1, (N, 1)).astype(np.float32)

    return dc_sh, rest_9


N_SH_REST = 45

def save_gaussians_to_ply(
    params: np.ndarray,
    output_path,
    lighting: CloudLightingConfig | None = None,
):
    if lighting is None:
        lighting = CloudLightingConfig()

    params = np.ascontiguousarray(params, dtype=np.float32)
    if params.ndim == 1:
        params = params.reshape(-1, 11)
    assert params.shape[1] == 11

    N = len(params)
    if N == 0:
        print("[PLY] Warning: 0 Gaussians, nothing to save.")
        return

    output_path = Path(output_path)

    pos    = params[:, 0:3]
    scale  = params[:, 3:6]
    quat   = params[:, 6:10]
    weight = params[:, 10]

    # Scale: export as-is (physical opacity already accounts for sigma)
    log_scale = np.log(np.clip(scale, 1e-8, None)).astype(np.float32)

    # Quaternion (x,y,z,w) -> (w,x,y,z)
    quat_wxyz = np.stack([quat[:, 3], quat[:, 0], quat[:, 1], quat[:, 2]], axis=1)
    norms = np.linalg.norm(quat_wxyz, axis=1, keepdims=True)
    quat_wxyz = (quat_wxyz / np.where(norms < 1e-8, 1.0, norms)).astype(np.float32)

    # Physical opacity from density formula
    alpha = compute_physical_opacity(
        weight.astype(np.float64),
        scale.astype(np.float64),
        lighting.density_scale,
        lighting.opacity_floor,
        lighting.opacity_ceil,
    )
    opacity_logit = np.log(alpha / (1.0 - alpha)).astype(np.float32)

    print(f"[PLY] Baking cloud shading for {N:,} Gaussians...")
    dc_sh, band1_sh = bake_cloud_colors(params, lighting)

    sh_rest = np.zeros((N, N_SH_REST), dtype=np.float32)
    sh_rest[:, :9] = band1_sh

    normals = np.zeros((N, 3), dtype=np.float32)

    prop_names = (
        ["x", "y", "z"] + ["nx", "ny", "nz"]
        + ["f_dc_0", "f_dc_1", "f_dc_2"]
        + [f"f_rest_{i}" for i in range(N_SH_REST)]
        + ["opacity"] + ["scale_0", "scale_1", "scale_2"]
        + ["rot_0", "rot_1", "rot_2", "rot_3"]
    )

    data = np.concatenate([
        pos, normals, dc_sh, sh_rest,
        opacity_logit[:, None], log_scale, quat_wxyz,
    ], axis=1).astype(np.float32)

    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"comment 3DGS cloud export {N} splats "
        f"{datetime.now().isoformat(timespec='seconds')}\n"
        f"element vertex {N}\n"
        + "\n".join(f"property float {n}" for n in prop_names)
        + "\nend_header\n"
    )

    with open(output_path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())

    size_mb = output_path.stat().st_size / 1_048_576
    approx_lum = dc_sh[:, 0] * SH_C0
    print(
        f"[PLY] Saved {N:,} Gaussians -> {output_path}  ({size_mb:.2f} MB)\n"
        f"[PLY] Opacity  : min={alpha.min():.4f}  mean={alpha.mean():.4f}  "
        f"max={alpha.max():.4f}\n"
        f"[PLY] Luminance: min={approx_lum.min():.2f}  "
        f"mean={approx_lum.mean():.2f}  max={approx_lum.max():.2f}  "
        f"white={(approx_lum>0.9).mean():.0%}  "
        f"grey={((approx_lum>=0.4)&(approx_lum<=0.9)).mean():.0%}  "
        f"dark={(approx_lum<0.4).mean():.0%}\n"
        f"[PLY] Tip: if too transparent -> raise density_scale; "
        f"if too solid -> lower it"
    )


def save_gaussians_dialog(
    params: np.ndarray, lighting: CloudLightingConfig | None = None
):
    default_name = f"splat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ply"
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.asksaveasfilename(
            title="Save Gaussians as PLY",
            initialfile=default_name,
            defaultextension=".ply",
            filetypes=[("PLY files", "*.ply"), ("All files", "*.*")],
        )
        root.destroy()
        if not path:
            print("[PLY] Save cancelled.")
            return
    except Exception as e:
        path = default_name
        print(f"[PLY] File dialog unavailable ({e}), saving to: {path}")
    save_gaussians_to_ply(params, path, lighting)
