"""
Plots Gaussian-count trajectories over training steps for the four ADC presets
discussed in chapter 5: ADC off, default, calm, aggressive.

Reads checkpoint data from results.json under the relevant experiment dirs and
emits a single PDF/PNG figure.

Usage:
    python tools/plot_gaussian_count.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams

# ---------------------------------------------------------------------------
# Source experiments
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

SOURCES = [
    ("ADC off",     ROOT / "results/exp_05a_adc_budget/adc_off/results.json",          "#888888", "--"),
    ("ADC default", ROOT / "results/exp_05a_adc_budget/adc_default/results.json",      "#1f77b4", "-"),
    ("Calm",        ROOT / "results/exp_05d_adc_combined/baseline/results.json",       "#2ca02c", "-"),
    ("Aggressive",  ROOT / "results/exp_05d_adc_combined/vfast_weight_high/results.json", "#d62728", "-"),
]

OUT_DIR = ROOT / "notes/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PDF = OUT_DIR / "gaussian_count_over_time.pdf"
OUT_PNG = OUT_DIR / "gaussian_count_over_time.png"

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "axes.linewidth": 1.0,
    "grid.linestyle": "--",
    "grid.alpha": 0.35,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def load_trajectory(path: Path) -> tuple[list[int], list[int]]:
    with open(path) as f:
        data = json.load(f)
    cps = data["checkpoints"]
    steps = [cp["step"] for cp in cps]
    counts = [cp["gaussian_count"] for cp in cps]
    return steps, counts


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    for label, path, colour, ls in SOURCES:
        if not path.exists():
            print(f"[warn] missing {path}")
            continue
        steps, counts = load_trajectory(path)
        ax.plot(
            steps, counts,
            label=label, color=colour, linestyle=ls,
            linewidth=2.0, marker="o", markersize=4.5,
        )
        # Annotate final count near the last point.
        ax.annotate(
            f"{counts[-1]:,}",
            xy=(steps[-1], counts[-1]),
            xytext=(6, 0), textcoords="offset points",
            fontsize=10, color=colour, va="center",
        )

    ax.set_xlabel("Training step")
    ax.set_ylabel("Number of Gaussians")
    ax.set_xlim(left=0, right=3000)
    ax.grid(True)
    ax.legend(loc="upper left", frameon=True, framealpha=0.92)

    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"[ok] wrote {OUT_PDF}")
    print(f"[ok] wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
