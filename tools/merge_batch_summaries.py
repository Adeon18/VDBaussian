"""
Merge a backup batch_summary (pre-ext) with the current one written by
experiment_runner, then regenerate the HTML/LaTeX report from the combined
experiment list.

Usage:
    python tools/merge_batch_summaries.py <results_dir>

Assumes <results_dir>/batch_summary_pre_ext.json exists (the backup written
before --only extension runs) and <results_dir>/batch_summary.json contains
the new --only results.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import report_generator


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python tools/merge_batch_summaries.py <results_dir>")
        sys.exit(1)

    results_dir = Path(sys.argv[1]).resolve()
    old_path = results_dir / "batch_summary_pre_ext.json"
    new_path = results_dir / "batch_summary.json"

    old = json.loads(old_path.read_text())
    new = json.loads(new_path.read_text())

    old_exps = old.get("experiments", [])
    new_exps = new.get("experiments", [])

    # Merge by name (new overrides old if same name re-run; shouldn't happen here)
    merged_by_name = {e["name"]: e for e in old_exps}
    for e in new_exps:
        merged_by_name[e["name"]] = e
    merged = list(merged_by_name.values())

    out = {
        "batch_file":         old.get("batch_file"),
        "started_at":         old.get("started_at"),
        "finished_at":        new.get("finished_at") or old.get("finished_at"),
        "total_wall_seconds": (old.get("total_wall_seconds", 0) or 0)
                              + (new.get("total_wall_seconds", 0) or 0),
        "experiments":        merged,
    }

    merged_path = results_dir / "batch_summary.json"
    merged_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"[merge] wrote {merged_path} -- {len(merged)} experiments total")
    for e in sorted(merged, key=lambda x: x["name"]):
        p3 = e.get("final_psnr_3d")
        print(f"  {e['name']:<15} {e['status']:<12} PSNR3D={p3:.2f}" if p3 else
              f"  {e['name']:<15} {e['status']:<12} PSNR3D=-")

    # Regenerate report from the merged experiment list
    print(f"[merge] regenerating report from {len(merged)} rows...")
    index_path = report_generator.generate(results_dir, merged)
    print(f"[merge] report -> {index_path}")


if __name__ == "__main__":
    main()
