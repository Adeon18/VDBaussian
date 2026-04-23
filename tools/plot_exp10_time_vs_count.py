"""Total training wall-time vs final Gaussian count for the step-budget
sweep (exp_10 aggressive, exp_10b calm).
"""

import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

agg_dir  = "results/exp_10_steps_native_aggressive"
calm_dir = "results/exp_10b_steps_native_calm"

def load(dirpath):
    b = json.load(open(os.path.join(dirpath, "batch_summary.json")))
    rows = []
    for e in b["experiments"]:
        steps = int(e["name"].split("_")[1])
        rows.append((steps, e["final_gaussian_count"], e["wall_time_seconds"]))
    rows.sort(key=lambda r: r[1])
    return rows

calm_rows = load(calm_dir)
agg_rows  = load(agg_dir)

fig, ax = plt.subplots(figsize=(10, 7))

def plot_series(rows, color, marker, label):
    xs = [r[1] for r in rows]
    ys = [r[2] for r in rows]
    ax.plot(xs, ys, linestyle="-", color=color, linewidth=2.2,
            marker=marker, markersize=11, markeredgecolor="black",
            markeredgewidth=1.2, label=label, zorder=3)
    for steps, n, t in rows:
        ax.annotate(f"{steps}", xy=(n, t),
                    xytext=(9, 5), textcoords="offset points",
                    fontsize=11, color=color, fontweight="bold")

plot_series(calm_rows, "#1f77b4", "o", "calm (exp_10b)")
plot_series(agg_rows,  "#d62728", "s", "aggressive (exp_10)")

ax.set_xlabel("Final Gaussian Count", fontsize=14)
ax.set_ylabel("Total Training Wall Time (seconds)", fontsize=14)
ax.set_title("Total Training Time vs Gaussian Count (cloud_white, $460^3$)",
             fontsize=16)
ax.grid(True, alpha=0.3)
ax.tick_params(axis="both", labelsize=12)
ax.legend(loc="upper left", fontsize=13, frameon=True)

ax.text(0.98, 0.02, "Point labels = training-step budget",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=11, style="italic", color="gray")

plt.tight_layout()
out_dir = os.path.join(agg_dir, "plots")
os.makedirs(out_dir, exist_ok=True)
out_pdf = os.path.join(out_dir, "time_vs_count.pdf")
out_png = os.path.join(out_dir, "time_vs_count.png")
plt.savefig(out_pdf, bbox_inches="tight", dpi=300)
plt.savefig(out_png, bbox_inches="tight", dpi=200)
print(f"Saved -> {out_pdf}")
print(f"Saved -> {out_png}")
