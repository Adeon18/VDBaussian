"""Generate a 2x2 grid for exp_06_resolution: PSNR_3D, PSNR_2D, IoU_3D
training curves plus a compression-ratio bar chart.

Color = resolution; linestyle = ADC preset (calm solid, aggressive dashed).
Legend lives outside the grid (bottom centre) so panels stay clean.
"""

import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

exp_dir = "results/exp_06_resolution"

# Resolution -> color
res_order = ["128", "196", "256", "native"]
res_color = {
    "128":    "#1f77b4",
    "196":    "#2ca02c",
    "256":    "#ff7f0e",
    "native": "#d62728",
}
preset_style = {"calm": "-", "aggressive": "--"}
preset_marker = {"calm": "o", "aggressive": "s"}

# (resolution, preset, exp_name)
runs = []
for res in res_order:
    for preset in ("calm", "aggressive"):
        runs.append((res, preset, f"res_{res}_{preset}"))

with open(os.path.join(exp_dir, "batch_summary.json")) as f:
    batch = json.load(f)
final_by_name = {e["name"]: e for e in batch["experiments"]}

curves = {}  # name -> dict(steps, psnr3d, psnr2d, iou3d, cr)
for res, preset, name in runs:
    rpath = os.path.join(exp_dir, name, "results.json")
    if not os.path.exists(rpath):
        continue
    with open(rpath) as f:
        d = json.load(f)
    steps, psnr3d, psnr2d, iou3d = [], [], [], []
    for cp in d["checkpoints"]:
        m3 = cp.get("metrics_3d") or {}
        m2 = cp.get("metrics_2d") or {}
        if m3.get("psnr") is None:
            continue
        steps.append(cp["step"])
        psnr3d.append(m3["psnr"])
        psnr2d.append(m2.get("psnr"))
        iou3d.append(m3.get("iou"))
    cr = (final_by_name.get(name, {}).get("compression", {})
                       .get("ratio_vs_resampled_vdb", 0.0))
    curves[name] = dict(res=res, preset=preset, steps=steps,
                        psnr3d=psnr3d, psnr2d=psnr2d, iou3d=iou3d, cr=cr)

fig, axes = plt.subplots(2, 2, figsize=(11, 8))

# ---- 3 line panels --------------------------------------------------------
panels = [
    (axes[0, 0], "psnr3d", r"PSNR$_{3D}$", "dB"),
    (axes[0, 1], "psnr2d", r"PSNR$_{2D}$", "dB"),
    (axes[1, 0], "iou3d",  r"IoU$_{3D}$",  "IoU"),
]
for ax, key, title, ylabel in panels:
    for name, c in curves.items():
        ax.plot(c["steps"], c[key],
                color=res_color[c["res"]],
                linestyle=preset_style[c["preset"]],
                marker=preset_marker[c["preset"]],
                markersize=4, linewidth=1.6)
    ax.set_title(title, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel("Training Step", fontsize=12)
    ax.grid(True, alpha=0.3)

# ---- compression bar chart (grouped by resolution) ------------------------
ax = axes[1, 1]
import numpy as np
x = np.arange(len(res_order))
w = 0.38
calm_vals = [curves.get(f"res_{r}_calm",       {}).get("cr", 0) for r in res_order]
agg_vals  = [curves.get(f"res_{r}_aggressive", {}).get("cr", 0) for r in res_order]
ax.bar(x - w/2, calm_vals, w,
       color=[res_color[r] for r in res_order], edgecolor="black", linewidth=0.8)
ax.bar(x + w/2, agg_vals,  w,
       color=[res_color[r] for r in res_order], edgecolor="black",
       linewidth=0.8, hatch="//", alpha=0.85)
bar_legend = ax.legend(
    handles=[
        Patch(facecolor="lightgray", edgecolor="black", linewidth=0.8,
              label="calm"),
        Patch(facecolor="lightgray", edgecolor="black", linewidth=0.8,
              hatch="//", alpha=0.85, label="aggressive"),
    ],
    loc="upper left", fontsize=11, frameon=True,
)
ax.add_artist(bar_legend)
ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, zorder=0)
ax.set_xticks(x)
ax.set_xticklabels([f"{r}³" if r != "native" else "native" for r in res_order],
                   fontsize=12)
ax.set_title(r"Compression Ratio (vs resampled VDB)", fontsize=13)
ax.set_ylabel(r"$\times$", fontsize=12)
ax.set_xlabel("Resolution", fontsize=12)
ax.grid(True, alpha=0.3, axis="y")
_max_cr = max(max(calm_vals), max(agg_vals))
ax.set_ylim(0, _max_cr * 1.18)
for i, v in enumerate(calm_vals):
    ax.text(i - w/2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
for i, v in enumerate(agg_vals):
    ax.text(i + w/2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=10)

# ---- shared figure legend at bottom ---------------------------------------
res_handles = [Line2D([0], [0], color=res_color[r], linewidth=2.4,
                      label=(f"{r}³" if r != "native" else "native"))
               for r in res_order]
preset_handles = [
    Line2D([0], [0], color="black", linestyle="-",  marker="o",
           markersize=5, linewidth=1.6, label="calm ADC"),
    Line2D([0], [0], color="black", linestyle="--", marker="s",
           markersize=5, linewidth=1.6, label="aggressive ADC"),
]
fig.legend(
    handles=res_handles + preset_handles,
    loc="lower center", ncol=6, fontsize=12,
    bbox_to_anchor=(0.5, -0.02), frameon=True,
)

plt.tight_layout(rect=[0, 0.04, 1, 1])
out_pdf = os.path.join(exp_dir, "plots", "metrics_grid_06.pdf")
out_png = os.path.join(exp_dir, "plots", "metrics_grid_06.png")
os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
plt.savefig(out_pdf, bbox_inches="tight", dpi=300)
plt.savefig(out_png, bbox_inches="tight", dpi=200)
print(f"Saved -> {out_pdf}")
print(f"Saved -> {out_png}")
