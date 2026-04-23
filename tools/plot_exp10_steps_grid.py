"""Grid of bar charts for the step-budget sweep (exp_10 aggressive,
exp_10b calm) on cloud_white @ native resolution.

Same colour per step-count; calm = solid fill, aggressive = hatched fill.
Panels: PSNR_3D, PSNR_2D, SSIM_3D, SSIM_2D, IoU_3D, IoU_2D,
wall-time (s), compression ratio vs resampled VDB.
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

agg_dir  = "results/exp_10_steps_native_aggressive"
calm_dir = "results/exp_10b_steps_native_calm"

# Keep step-counts present in BOTH batches
steps = [500, 1000, 1500, 3000, 5000, 8000]
step_labels = [f"{s}" for s in steps]

# Distinct colour per step-count (tab10 palette, matches attached screenshot style)
step_color = {
    500:  "#1f77b4",
    1000: "#2ca02c",
    1500: "#ff7f0e",
    3000: "#d62728",
    5000: "#9467bd",
    8000: "#8c564b",
}

def load(dirpath):
    with open(os.path.join(dirpath, "batch_summary.json")) as f:
        b = json.load(f)
    return {e["name"]: e for e in b["experiments"]}

agg  = load(agg_dir)
calm = load(calm_dir)

def row(src, s):
    e = src.get(f"steps_{s}")
    if e is None:
        return None
    return dict(
        psnr3d = e["final_psnr_3d"],
        psnr2d = e["final_psnr_2d"],
        ssim3d = e["final_ssim_3d"],
        ssim2d = e["final_ssim_2d"],
        iou3d  = e["final_iou_3d"],
        iou2d  = e["final_iou_2d"],
        time_s = e["wall_time_seconds"],
        cr     = e["compression"]["ratio_vs_resampled_vdb"],
    )

calm_rows = {s: row(calm, s) for s in steps}
agg_rows  = {s: row(agg,  s) for s in steps}

panels = [
    ("psnr3d", r"PSNR$_{3D}$",   "dB",          "{:.2f}"),
    ("psnr2d", r"PSNR$_{2D}$",   "dB",          "{:.2f}"),
    ("time_s", r"Wall Time",     "seconds",     "{:.0f}"),
    ("cr",     r"Compression (vs resampled VDB)",
                                 r"$\times$",   "{:.2f}"),
]

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
axes = axes.flatten()

x = np.arange(len(steps))
w = 0.46
colors = [step_color[s] for s in steps]

for ax, (key, title, ylabel, fmt) in zip(axes, panels):
    calm_vals = [calm_rows[s][key] if calm_rows[s] else 0.0 for s in steps]
    agg_vals  = [agg_rows[s][key]  if agg_rows[s]  else 0.0 for s in steps]

    ax.bar(x - w/2, calm_vals, w,
           color=colors, edgecolor="black", linewidth=0.8)
    ax.bar(x + w/2, agg_vals,  w,
           color=colors, edgecolor="black", linewidth=0.8,
           hatch="//", alpha=0.85)

    # Per-panel legend, upper-left, matching the reference screenshot
    leg = ax.legend(
        handles=[
            Patch(facecolor="lightgray", edgecolor="black", linewidth=0.8,
                  label="calm"),
            Patch(facecolor="lightgray", edgecolor="black", linewidth=0.8,
                  hatch="//", alpha=0.85, label="aggressive"),
        ],
        loc="upper left", fontsize=12, frameon=True,
    )
    ax.add_artist(leg)

    if key == "cr":
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(step_labels, fontsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_title(title, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xlabel("Training Steps", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")

    all_vals = calm_vals + agg_vals
    vmax = max(all_vals)
    vmin = min(v for v in all_vals if v > 0) if any(all_vals) else 0.0

    # Zoom PSNR so differences are visible (values are all ~38-50 dB)
    if key in ("psnr3d", "psnr2d"):
        lo = max(0.0, vmin - 0.15 * (vmax - vmin + 1e-6))
        ax.set_ylim(lo, vmax + 0.55 * (vmax - lo))
    else:
        ax.set_ylim(0, vmax * 1.55)

    for i, v in enumerate(calm_vals):
        ax.text(i - w/2, v, fmt.format(v), ha="center", va="bottom",
                fontsize=11, rotation=0)
    for i, v in enumerate(agg_vals):
        ax.text(i + w/2, v, fmt.format(v), ha="center", va="bottom",
                fontsize=11, rotation=0)

plt.tight_layout()

out_dir = os.path.join(agg_dir, "plots")
os.makedirs(out_dir, exist_ok=True)
out_pdf = os.path.join(out_dir, "metrics_grid_steps.pdf")
out_png = os.path.join(out_dir, "metrics_grid_steps.png")
plt.savefig(out_pdf, bbox_inches="tight", dpi=300)
plt.savefig(out_png, bbox_inches="tight", dpi=200)
print(f"Saved -> {out_pdf}")
print(f"Saved -> {out_png}")
