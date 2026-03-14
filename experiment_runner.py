"""
Ostap: This is an LLM-generated wrapper and report provider for the purpose of just running experiments using already
existing code and features to do so.

experiment_runner.py  —  Sequential batch experiment orchestrator.

Usage:
    python experiment_runner.py --batch config/example_batch.yaml
    python experiment_runner.py --batch config/example_batch.yaml --only baseline_adam_l2 l1_loss
    python experiment_runner.py --batch config/example_batch.yaml --dry-run
    python experiment_runner.py --batch config/example_batch.yaml --no-report

Ctrl+C after the current experiment finishes gracefully.
Ctrl+C twice → hard kill.

After all experiments complete, generates results/report.html automatically.
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from config_loader import load_batch

# ---------------------------------------------------------------------------
# Graceful interrupt
# ---------------------------------------------------------------------------

_STOP_AFTER_CURRENT = False


def _sigint(sig, frame):
    global _STOP_AFTER_CURRENT
    print("\n[Runner] Ctrl+C received — will finish current experiment then stop.")
    print("[Runner] Ctrl+C again to force-quit immediately.")
    _STOP_AFTER_CURRENT = True
    signal.signal(signal.SIGINT, signal.SIG_DFL)


signal.signal(signal.SIGINT, _sigint)

# ---------------------------------------------------------------------------
# Exit code meanings (must match train_headless.py)
# ---------------------------------------------------------------------------

_STATUS_FROM_CODE = {
    0:  "completed",
    1:  "config_error",
    2:  "aborted_divergence",
    3:  "aborted_nan",
    99: "crashed",
}

# ---------------------------------------------------------------------------
# Run one experiment
# ---------------------------------------------------------------------------

def run_one(name: str, batch_path: Path, out_dir: Path) -> dict:
    """Launch train_headless.py as a subprocess. Returns summary dict."""
    exp_out = out_dir / name
    cmd = [
        sys.executable, str(Path(__file__).parent / "train_headless.py"),
        "--config",      str(batch_path),
        "--experiment",  name,
        "--output-dir",  str(exp_out),
    ]

    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  Experiment : {name}")
    print(f"  Output     : {exp_out}")
    print(f"  Started    : {datetime.now().strftime('%H:%M:%S')}")
    print(f"{sep}")

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=Path(__file__).parent)
    elapsed = time.time() - t0

    code   = proc.returncode
    status = _STATUS_FROM_CODE.get(code, f"unknown_exit_{code}")

    summary: dict = {
        "name":              name,
        "status":            status,
        "exit_code":         code,
        "wall_time_seconds": round(elapsed, 2),
        "out_dir":           str(exp_out),
        "timestamp":         datetime.now().isoformat(),
    }

    # Enrich from results.json if present
    results_json = exp_out / "results.json"
    if results_json.exists():
        try:
            with open(results_json) as f:
                r = json.load(f)
            summary["final_gaussian_count"] = r.get("final_gaussian_count", 0)
            summary["step_time_summary"]    = r.get("step_time_summary", {})

            cps = r.get("checkpoints", [])
            if cps:
                last = cps[-1]
                m3   = last.get("metrics_3d") or {}
                m2   = last.get("metrics_2d") or {}
                summary["final_psnr_3d"]  = m3.get("psnr")
                summary["final_l1_3d"]    = m3.get("l1")
                summary["final_iou_3d"]   = m3.get("iou")
                summary["final_ssim_3d"]  = m3.get("ssim_mean")
                summary["final_psnr_2d"]  = m2.get("psnr")
                summary["final_l1_2d"]    = m2.get("l1")
                summary["final_iou_2d"]   = m2.get("iou")
                summary["final_ssim_2d"]  = m2.get("ssim")
        except Exception as e:
            print(f"[Runner] Warning: could not parse results.json for '{name}': {e}")

    _print_exp_result(summary)
    return summary


def _print_exp_result(s: dict) -> None:
    ok   = s["status"] == "completed"
    icon = "✓" if ok else "✗"
    ms   = s.get("step_time_summary", {}).get("mean_ms", "?")
    p3   = s.get("final_psnr_3d")
    p3s  = f"{p3:.2f} dB" if p3 is not None else "—"
    t    = timedelta(seconds=int(s["wall_time_seconds"]))
    print(f"  {icon} {s['name']:<30}  {s['status']:<20}  "
          f"wall={t}  step={ms}ms  PSNR3D={p3s}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_batch_summary(summaries: list[dict]) -> None:
    print(f"\n{'='*90}")
    print(f"  BATCH SUMMARY — {len(summaries)} experiments")
    print(f"{'='*90}")
    hdr = (f"  {'Name':<30} {'Status':<22} {'Wall':>7}  "
           f"{'step ms':>8}  {'PSNR3D':>7}  {'L1-3D':>8}  {'PSNR2D':>7}")
    print(hdr)
    print(f"  {'-'*84}")

    for s in sorted(summaries,
                    key=lambda x: (x.get("final_psnr_3d") or 0),
                    reverse=True):
        wall = timedelta(seconds=int(s.get("wall_time_seconds", 0)))
        ms   = s.get("step_time_summary", {}).get("mean_ms", "?")
        p3   = s.get("final_psnr_3d")
        l3   = s.get("final_l1_3d")
        p2   = s.get("final_psnr_2d")
        print(f"  {s['name']:<30} {s['status']:<22} {str(wall):>7}  "
              f"{str(ms):>8}  "
              f"{f'{p3:.2f}' if p3 else '—':>7}  "
              f"{f'{l3:.5f}' if l3 else '—':>8}  "
              f"{f'{p2:.2f}' if p2 else '—':>7}")

    print(f"{'='*90}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Batch experiment orchestrator")
    p.add_argument("--batch",      required=True,
                   help="Batch YAML file (must contain an 'experiments' list)")
    p.add_argument("--output-dir", default=None,
                   help="Override base results directory")
    p.add_argument("--only",       nargs="*", default=None,
                   help="Run only these named experiments")
    p.add_argument("--dry-run",    action="store_true",
                   help="List experiments that would run, then exit")
    p.add_argument("--no-report",  action="store_true",
                   help="Skip HTML report generation at the end")
    return p.parse_args()


def main():
    args       = _parse_args()
    batch_path = Path(args.batch).resolve()

    all_cfgs = load_batch(batch_path)
    if not all_cfgs:
        print("[Runner] No experiments found in batch file.")
        sys.exit(0)

    # Filter by --only
    if args.only:
        wanted  = set(args.only)
        cfgs    = [c for c in all_cfgs if c.get("_name") in wanted]
        missing = wanted - {c.get("_name") for c in cfgs}
        if missing:
            print(f"[Runner] WARNING: unknown experiments: {missing}")
    else:
        cfgs = all_cfgs

    if not cfgs:
        print("[Runner] No matching experiments.")
        sys.exit(0)

    # Determine output dir from first config
    base_out = (
        Path(args.output_dir)
        if args.output_dir
        else Path(cfgs[0]["output"]["result_dir"])
    ).resolve()

    # Dry run
    if args.dry_run:
        print(f"[Dry run] {len(cfgs)} experiments → {base_out}")
        for c in cfgs:
            print(f"  - {c.get('_name','?'):<35} {c.get('_description','')}")
        return

    base_out.mkdir(parents=True, exist_ok=True)

    # Write initial batch manifest
    manifest_path = base_out / "batch_summary.json"
    manifest: dict = {
        "batch_file":  str(batch_path),
        "started_at":  datetime.now().isoformat(),
        "experiments": [],
    }

    summaries: list[dict] = []
    t_batch = time.time()

    for i, cfg in enumerate(cfgs, 1):
        name = cfg.get("_name", f"exp_{i}")
        print(f"\n[Runner] ({i}/{len(cfgs)}) Starting: {name}")

        summary = run_one(name, batch_path, base_out)
        summaries.append(summary)

        # Flush manifest after each experiment so Ctrl+C doesn't lose results
        manifest["experiments"] = summaries
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)

        if _STOP_AFTER_CURRENT:
            print("\n[Runner] Stopping batch after interrupt.")
            break

    manifest["finished_at"]       = datetime.now().isoformat()
    manifest["total_wall_seconds"] = round(time.time() - t_batch, 2)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print_batch_summary(summaries)
    print(f"[Runner] Total wall time: {timedelta(seconds=int(time.time()-t_batch))}")
    print(f"[Runner] Results directory: {base_out}")

    # HTML report
    if not args.no_report:
        print("[Runner] Generating HTML report...")
        try:
            import report_generator
            index_path = report_generator.generate(base_out, summaries)
            print(f"[Runner] Report ready → {index_path}")
            print(f"[Runner] Open in browser: file://{index_path.resolve()}")
        except Exception as e:
            import traceback
            print(f"[Runner] Report generation failed (non-fatal): {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
