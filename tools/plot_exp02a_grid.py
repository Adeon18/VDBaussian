"""Generate a 2x2 grid: PSNR_3D, PSNR_2D, SSIM_3D, IoU_3D training curves for exp_02a."""

import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

exp_dir = "results/exp_02a_init_count"
name_map = {
    "init_1k": (1000, "1k"),
    "init_2500": (2500, "2.5k"),
    "init_5k": (5000, "5k"),
    "init_10k": (10000, "10k"),
    "init_15k": (15000, "15k"),
    "init_20k": (20000, "20k"),
    "init_30k": (30000, "30k"),
    "init_40k": (40000, "40k"),
}

experiments = []
colors = plt.cm.tab10.colors

with open(os.path.join(exp_dir, "batch_summary.json")) as f:
    batch = json.load(f)
final_by_name = {e["name"]: e for e in batch["experiments"]}

for name, (count, label) in sorted(name_map.items(), key=lambda x: x[1][0]):
    path = os.path.join(exp_dir, name, "results.json")
    if not os.path.exists(path):
        continue
    with open(path) as f:
        d = json.load(f)
    steps, psnr3d, iou3d, ssim3d = [], [], [], []
    for cp in d["checkpoints"]:
        m3 = cp.get("metrics_3d", {})
        if m3.get("psnr") is None:
            continue
        steps.append(cp["step"])
        psnr3d.append(m3["psnr"])
        iou3d.append(m3.get("iou", 0))
        ssim3d.append(m3.get("ssim_mean", 0))
    cr = final_by_name.get(name, {}).get("compression", {}).get("ratio_vs_resampled_vdb", 0)
    experiments.append((label, count, steps, psnr3d, iou3d, ssim3d, cr))

fig, axes = plt.subplots(2, 2, figsize=(11, 8))

curve_panels = [
    (axes[0, 0], 3, r"PSNR$_{3D}$", "dB"),
    (axes[1, 0], 4, r"IoU$_{3D}$", "IoU"),
    (axes[1, 1], 5, r"SSIM$_{3D}$", "SSIM"),
]

for ax, idx, title, ylabel in curve_panels:
    for i, exp in enumerate(experiments):
        ax.plot(exp[2], exp[idx], marker="o", markersize=3, linewidth=1.8,
                color=colors[i % len(colors)], label=exp[0])
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xlabel("Training Step", fontsize=10)
    ax.grid(True, alpha=0.3)

ax = axes[0, 1]
cr_vals = [e[6] for e in experiments]
cr_labels = [e[0] for e in experiments]
bars = ax.bar(range(len(experiments)), cr_vals,
              color=[colors[i % len(colors)] for i in range(len(experiments))])
ax.set_xticks(range(len(experiments)))
ax.set_xticklabels(cr_labels, fontsize=9)
ax.set_title(r"Compression Ratio", fontsize=11)
ax.set_ylabel(r"$\times$", fontsize=10)
ax.set_xlabel("Init Count", fontsize=10)
ax.grid(True, alpha=0.3, axis="y")

axes[1, 1].legend(title="Init Count", fontsize=10, title_fontsize=11,
                   loc="lower right", ncol=2)

plt.tight_layout()
out = os.path.join(exp_dir, "plots", "metrics_grid.pdf")
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, bbox_inches="tight", dpi=300)
print(f"Saved to {out}")
