"""
Ostap: This is an LLM-generated wrapper and report provider for the purpose of just running experiments using already
existing code and features to do so.


train_headless.py  —  Headless single-experiment trainer.

Usage (standalone):
    python train_headless.py --config config/example_batch.yaml --experiment baseline_adam_l2
    python train_headless.py --config config/my_single_exp.yaml

Called by experiment_runner.py as a subprocess — one fresh process per experiment,
which guarantees clean Vulkan/shader state and proper constant patching.

Exit codes:
    0  completed normally (or interrupted cleanly)
    1  setup / config error
    2  training aborted (abort_on_explosion=true triggered)
    3  NaN detected in gradients (abort_on_nan=true triggered)
   99  unexpected crash
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import signal
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Resolve project root so local imports work from any CWD
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(_HERE))

from config_loader import (
    LRScheduler,
    apply_config_to_adc,
    apply_config_to_metrics,
    apply_config_to_settings,
    apply_config_to_train_config,
    build_experiment_config,
    load_batch,
    load_single,
    resolve_checkpoints,
)



# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(name: str, log_path: Path) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# JSON serialisation helper
# ---------------------------------------------------------------------------

def _json_default(obj):
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return str(obj)


def _flush_json(data: dict, path: Path) -> None:
    """Atomic write: .tmp → rename."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)
    if path.exists():
        path.unlink()
    tmp.rename(path)


# ---------------------------------------------------------------------------
# PNG save (PIL preferred, raw PPM fallback)
# ---------------------------------------------------------------------------

def save_png(pixels: np.ndarray, path: Path) -> None:
    """pixels: H×W×4 uint8 RGBA array."""
    try:
        from PIL import Image
        Image.fromarray(pixels, "RGBA").save(path)
    except ImportError:
        rgb = pixels[:, :, :3]
        h, w = rgb.shape[:2]
        ppm = path.with_suffix(".ppm")
        with open(ppm, "wb") as f:
            f.write(f"P6\n{w} {h}\n255\n".encode())
            f.write(rgb.tobytes())
        path = ppm  # update path reference for logging
    return path


# ---------------------------------------------------------------------------
# Offscreen render helper
# ---------------------------------------------------------------------------

def render_to_array(renderer, cam_dict: dict, settings, w: int = 512, h: int = 384) -> np.ndarray:
    """
    Render one frame using a camera dict {pos, front, right, up}.
    Returns H×W×4 uint8.
    """
    class _Cam:
        def get_gpu_data(self, _aspect):
            data = np.zeros(16, dtype=np.float32)
            data[0:3]   = np.asarray(cam_dict["pos"],   dtype=np.float32)
            data[4:7]   = np.asarray(cam_dict["front"], dtype=np.float32)
            data[8:11]  = np.asarray(cam_dict["right"], dtype=np.float32)
            data[12:15] = np.asarray(cam_dict["up"],    dtype=np.float32)
            return data

    renderer.resize(w, h)
    renderer.render_main(_Cam(), settings)
    return renderer.screen_tex.to_numpy()


# ---------------------------------------------------------------------------
# Health monitor
# ---------------------------------------------------------------------------

class HealthMonitor:
    def __init__(self, cfg: dict):
        h = cfg["health"]
        self.explosion_thresh  = float(h["grad_explosion_threshold"])
        self.vanish_thresh     = float(h["grad_vanish_threshold"])
        self.streak_limit      = int(h["consecutive_bad_steps_before_flag"])
        self.abort_explosion   = bool(h["abort_on_explosion"])
        self.abort_nan         = bool(h["abort_on_nan"])
        self.check_interval    = int(h["check_interval"])
        self._bad_streak       = 0
        self.status            = "ok"
        self.events: list[dict]= []

    def check(self, step: int, renderer, logger: logging.Logger) -> str:
        """Returns 'abort' or 'continue'."""
        if not hasattr(renderer, "grad_buffer") or renderer.grad_buffer is None:
            return "continue"

        grads = renderer.grad_buffer.to_numpy().view(np.float32)

        # NaN — highest priority
        if np.any(np.isnan(grads)):
            logger.warning(f"Step {step}: NaN in gradients (streak {self._bad_streak+1})")
            self._bad_streak += 1
            if self._bad_streak >= self.streak_limit:
                self.status = "nan"
                self.events.append({"step": step, "type": "nan"})
                return "abort" if self.abort_nan else "continue"
            return "continue"

        nonzero = grads[np.abs(grads) > 1e-10]
        if len(nonzero) == 0:
            return "continue"

        grad_max  = float(np.max(np.abs(nonzero)))
        grad_mean = float(np.mean(np.abs(nonzero)))

        bad, etype = False, None
        if grad_max > self.explosion_thresh:
            bad, etype = True, "explosion"
            logger.warning(f"Step {step}: gradient explosion  max={grad_max:.3f}")
        elif grad_mean < self.vanish_thresh:
            bad, etype = True, "vanishing"
            logger.warning(f"Step {step}: gradient vanishing  mean={grad_mean:.2e}")

        if bad:
            self._bad_streak += 1
            if self._bad_streak >= self.streak_limit:
                self.status = etype
                self.events.append({"step": step, "type": etype,
                                    "grad_max": grad_max, "grad_mean": grad_mean})
                if etype == "explosion" and self.abort_explosion:
                    logger.error(f"Step {step}: aborting — sustained explosion")
                    return "abort"
        else:
            self._bad_streak = 0

        return "continue"


# ---------------------------------------------------------------------------
# Step-time tracker
# ---------------------------------------------------------------------------

class StepTimer:
    def __init__(self, sample_interval: int, rolling_window: int):
        self.sample_interval = sample_interval
        self._window         = deque(maxlen=rolling_window)
        self._t0             = None
        self.samples: list[tuple[int, float]] = []  # (step, ms)

    def start(self):
        self._t0 = time.perf_counter()

    def stop(self, step: int):
        if self._t0 is None:
            return
        ms = (time.perf_counter() - self._t0) * 1000.0
        self._window.append(ms)
        if step % self.sample_interval == 0:
            self.samples.append((step, round(ms, 3)))
        self._t0 = None

    @property
    def rolling_avg_ms(self) -> float:
        return float(np.mean(self._window)) if self._window else 0.0

    @property
    def p95_ms(self) -> float:
        return float(np.percentile(list(self._window), 95)) if len(self._window) >= 5 else 0.0

    def summary(self) -> dict:
        all_ms = [ms for _, ms in self.samples]
        if not all_ms:
            return {"mean_ms": None, "p95_ms": None, "min_ms": None, "max_ms": None}
        return {
            "mean_ms": round(float(np.mean(all_ms)),    3),
            "p95_ms":  round(float(np.percentile(all_ms, 95)), 3),
            "min_ms":  round(float(np.min(all_ms)),     3),
            "max_ms":  round(float(np.max(all_ms)),     3),
        }


# ---------------------------------------------------------------------------
# Checkpoint saver
# ---------------------------------------------------------------------------

def save_checkpoint(
    step:        int,
    renderer,
    settings,
    vol_min,
    vol_max,
    metrics_3d,
    metrics_2d,
    cfg:         dict,
    out_dir:     Path,
    logger:      logging.Logger,
    force_rasterize: bool = True,
    snap_w: int = 512, snap_h: int = 384
) -> dict:
    logger.info(f"Checkpoint step={step}  gaussians={renderer.gaussian_count:,}")

    snap: dict = {
        "step":             step,
        "gaussian_count":   renderer.gaussian_count,
        "metrics_3d":       {},
        "metrics_2d":       {},
        "renders":          [],
    }

    if cfg["metrics_3d"]["enabled"]:
        snap["metrics_3d"] = metrics_3d.get_latest()

    if cfg["metrics_2d"]["enabled"]:
        snap["metrics_2d"] = metrics_2d.get_latest()

    # Rasterize once for renders
    if force_rasterize:
        import slangpy as spy
        cmd = renderer.device.create_command_encoder()
        renderer.rasterize_gaussians(cmd, vol_min, vol_max)
        renderer.device.submit_command_buffer(cmd.finish())

    snap_cams = cfg["output"].get("snapshot_cameras", [])
    for cam_def in snap_cams:
        cam_name = cam_def.get("name", "cam")
        cam_dict = {k: np.array(v, dtype=np.float32)
                    for k, v in cam_def.items() if k != "name"}

        name_ref = f"step_{step:06d}_{cam_name}_ref.png"
        name_pred = f"step_{step:06d}_{cam_name}_pred.png"
        # Reference (raw VDB)
        renderer.use_gaussian_volume = False
        pix_ref  = render_to_array(renderer, cam_dict, settings, snap_w, snap_h)
        save_png(pix_ref, out_dir / name_ref)

        # Prediction (Gaussian volume)
        renderer.use_gaussian_volume = True
        pix_pred = render_to_array(renderer, cam_dict, settings, snap_w, snap_h)       
        save_png(pix_pred, out_dir / name_pred)

        renderer.use_gaussian_volume = False

        snap["renders"].append({
            "camera": cam_name,
            "ref":    name_ref,
            "pred":   name_pred,
        })

    return snap

from adc import CPUADCBackend

class _SyncCPUBackend(CPUADCBackend):
    @property
    def runs_async(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# HeadlessTrainer
# ---------------------------------------------------------------------------

class HeadlessTrainer:
    """
    Self-contained trainer that mirrors App but without GLFW window / ImGui.
    Exposes apply_densification() so ADCController works unchanged.
    """

    def __init__(self, cfg: dict, out_dir: Path, logger: logging.Logger):
        self.cfg     = cfg
        self.out_dir = out_dir
        self.logger  = logger
        self._interrupt = False

        vol = cfg["volume"]
        self.VOL_SIZE               = int(vol["vol_size"])
        self.TILE_SIZE              = int(vol["tile_size"])
        self.MAX_GAUSSIANS_PER_TILE = int(vol["max_gaussians_per_tile"])

    # ------------------------------------------------------------------ setup

    def _patch_app_constants(self):
        """
        Patch module-level constants in app.py before anything imports them.
        Since we're in a fresh subprocess this is safe.
        """
        import app as _app
        _app.VOL_SIZE               = self.VOL_SIZE
        _app.TILE_SIZE              = self.TILE_SIZE
        _app.MAX_GAUSSIANS_PER_TILE = self.MAX_GAUSSIANS_PER_TILE

    def setup(self):
        m2d_cfg = self.cfg.get("metrics_2d", {})
        self.snap_w = m2d_cfg.get("render_width",  512)
        self.snap_h = m2d_cfg.get("render_height", 384)
        self.logger.info(
            f"vol_size={self.VOL_SIZE}  tile_size={self.TILE_SIZE}  "
            f"max_g_per_tile={self.MAX_GAUSSIANS_PER_TILE}"
        )
        self._patch_app_constants()

        # ---- Invisible GLFW window for OpenGL blit ----
        import glfw
        if not glfw.init():
            raise RuntimeError("GLFW init failed")
        glfw.window_hint(glfw.VISIBLE, False)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        self._window = glfw.create_window(self.snap_w, self.snap_h, "headless", None, None)
        glfw.make_context_current(self._window)
        self._glfw = glfw

        # ---- Vulkan device ----
        import slangpy as spy
        self.device = spy.Device(
            enable_debug_layers=False,
            compiler_options={"include_paths": [str(_HERE)]},
            type=spy.DeviceType.vulkan,
        )
        self.logger.info("Vulkan device created")

        # ---- Import domain objects AFTER patching constants ----
        from app import (
            PARAMS_PER_GAUSSIAN,
            Renderer,
            Settings,
            TrainingConfig,
            convert_grid_to_dense_volume,
            convert_grid_to_gaussians,
            load_vdb_grid,
        )
        from adc import ADCConfig, ADCController, ADCMode
        from metrics import MetricsCollector
        from screen_metrics import ScreenMetricsCollector

        self._PARAMS_PER_GAUSSIAN = PARAMS_PER_GAUSSIAN

        # ---- VDB ----
        vdb_path = self.cfg["volume"]["vdb_file"]
        self.logger.info(f"Loading VDB: {vdb_path}")
        self.grid = load_vdb_grid(vdb_path)
        if self.grid is None:
            raise RuntimeError(f"Failed to load VDB: {vdb_path}")

        self.logger.info("Converting to dense volume...")
        self.vol_min, self.vol_max, vol_data = convert_grid_to_dense_volume(
            self.grid, self.VOL_SIZE
        )

        # Aliases expected by ADCController
        self.vol_min_world = self.vol_min
        self.vol_max_world = self.vol_max

        # ---- Settings ----
        self.settings = Settings()
        apply_config_to_settings(self.cfg, self.settings)

        # ---- TrainingConfig ----
        self.train_config = TrainingConfig()
        apply_config_to_train_config(self.cfg, self.train_config)

        # ---- Gaussians ----
        self.logger.info("Generating Gaussians...")
        self.gaussians = convert_grid_to_gaussians(self.grid, self.train_config)
        self.logger.info(f"Generated {len(self.gaussians):,} Gaussians")

        # ---- Renderer ----
        self.renderer = Renderer(self.device, vol_data)
        self.renderer.resize(self.snap_w, self.snap_h)
        self.renderer.gaussian_count = len(self.gaussians)
        self.renderer.gaussian_buffer = self.device.create_buffer(
            element_count=self.renderer.gaussian_count,
            struct_size=PARAMS_PER_GAUSSIAN * 4,
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            memory_type=spy.MemoryType.device_local,
            data=self.gaussians,
        )
        self.renderer._vol_min = self.vol_min
        self.renderer._vol_max = self.vol_max
        self.renderer.init_training()

        # Initial rasterize
        cmd = self.device.create_command_encoder()
        self.renderer.rasterize_gaussians(cmd, self.vol_min, self.vol_max)
        self.device.submit_command_buffer(cmd.finish())

        # ---- ADC ----
        adc_cfg = ADCConfig()
        apply_config_to_adc(self.cfg, adc_cfg)
        self.adc = ADCController(self, adc_cfg)
        self.adc.train_config = self.train_config
        self.adc._backends[ADCMode.CPU] = _SyncCPUBackend()

        # ---- Metrics ----
        self.metrics_3d = MetricsCollector(
            self.device, self.renderer,
            (self.VOL_SIZE, self.VOL_SIZE, self.VOL_SIZE),
        )
        m2d_cfg = self.cfg.get("metrics_2d", {})
        self.metrics_2d = ScreenMetricsCollector(
            self.device, self.renderer, self.settings,
            render_w=m2d_cfg.get("render_width", 512),
            render_h=m2d_cfg.get("render_height", 384)
        )
        apply_config_to_metrics(self.cfg, self.metrics_3d, self.metrics_2d)

        # ---- LR Scheduler ----
        self.lr_scheduler = LRScheduler(
            self.cfg["training"].get("lr_schedule", [])
        )

        # ---- Health + Perf ----
        self.health = HealthMonitor(self.cfg)
        p = self.cfg["perf"]
        self.timer = StepTimer(
            sample_interval=int(p["step_time_sample_interval"]),
            rolling_window=int(p["rolling_window"]),
        )

        self.logger.info("Setup complete.")

    # ------------------------------------------------------- apply_densification
    # Mirrors App.apply_densification so ADCController works unchanged.

    def apply_densification(self, new_params, surviving_indices=None):
        import slangpy as spy
        PPG = self._PARAMS_PER_GAUSSIAN
        new_params = np.ascontiguousarray(new_params, dtype=np.float32)
        if new_params.ndim == 2:
            new_params = new_params.reshape(-1, PPG)
        flat = new_params.flatten()

        if len(new_params) == 0:
            self.logger.warning("[ADC] apply_densification called with 0 Gaussians — skipping")
            return

        saved_iter = self.renderer.adam_iteration
        old_m1 = old_m2 = None

        if surviving_indices is not None:
            try:
                old_m1 = (self.renderer.adam_first_moment.to_numpy()
                          .view(np.float32).reshape(-1, PPG).copy())
                old_m2 = (self.renderer.adam_second_moment.to_numpy()
                          .view(np.float32).reshape(-1, PPG).copy())
            except Exception as e:
                self.logger.debug(f"[ADC] Could not read momentum: {e}")

        self.gaussians = new_params
        self.renderer.gaussian_count  = len(new_params)
        self.renderer.gaussian_buffer = self.device.create_buffer(
            element_count=len(new_params),
            struct_size=PPG * 4,
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            memory_type=spy.MemoryType.device_local,
            data=flat,
        )
        self.renderer.init_training()
        self.renderer.adam_iteration = saved_iter

        if surviving_indices is not None and old_m1 is not None:
            try:
                n = len(new_params)
                nm1 = np.zeros((n, PPG), dtype=np.float32)
                nm2 = np.zeros((n, PPG), dtype=np.float32)
                for i, src in enumerate(surviving_indices):
                    if i >= n:
                        break
                    if 0 <= src < len(old_m1):
                        nm1[i] = old_m1[src]
                        nm2[i] = old_m2[src]
                self.renderer.adam_first_moment.copy_from_numpy(nm1.flatten())
                self.renderer.adam_second_moment.copy_from_numpy(nm2.flatten())
            except Exception as e:
                self.logger.debug(f"[ADC] Momentum restore failed (non-fatal): {e}")

        self.renderer._needs_rebinning    = True
        self.renderer._needs_rasterization= True

    # ------------------------------------------------------------------ train

    def train(self) -> dict:
        cfg        = self.cfg
        max_steps  = cfg["training"]["max_steps"]
        checkpoints= resolve_checkpoints(cfg)
        cp_set     = set(checkpoints)

        results: dict = {
            "name":                cfg.get("_name", "unnamed"),
            "description":         cfg.get("_description", ""),
            "status":              "running",
            "checkpoints":         [],
            "health_events":       [],
            "step_time_samples":   [],
            "step_time_summary":   {},
            "final_gaussian_count":0,
            "wall_time_seconds":   0.0,
        }

        t_wall = time.time()
        self.logger.info(
            f"Training: max_steps={max_steps}  "
            f"checkpoints={checkpoints}"
        )

        # ---- step-0 checkpoint (pre-training) ----
        if 0 in cp_set:
            snap = save_checkpoint(
                0, self.renderer, self.settings,
                self.vol_min, self.vol_max,
                self.metrics_3d, self.metrics_2d,
                cfg, self.out_dir, self.logger,
                snap_w=self.snap_w, snap_h=self.snap_h,
            )
            results["checkpoints"].append(snap)
            _flush_json(results, self.out_dir / "results.json")

        # ---- main loop ----
        from tqdm import tqdm
        pbar = tqdm(
            range(1, max_steps + 1),
            desc=f"[{cfg.get('_name', 'train')}]",
            unit="step",
            dynamic_ncols=True,
        )

        for step in pbar:
            if self._interrupt:
                self.logger.info("Interrupted — exiting loop cleanly")
                results["status"] = "interrupted"
                break

            # LR schedule
            self.lr_scheduler.tick(step, self.train_config, self.logger)

            # Training step (timed)
            self.timer.start()
            self.renderer.run_training_step(
                self.vol_min, self.vol_max, self.train_config
            )
            self.timer.stop(step)

            # ADC
            self.adc.tick(step, True)
            self.adc.apply_pending()

            # 3D metrics
            if cfg["metrics_3d"]["enabled"]:
                self.metrics_3d.tick(step, self.vol_min, self.vol_max, True)  # force now
                snap["metrics_3d"] = self.metrics_3d.get_latest()

            if cfg["metrics_2d"]["enabled"]:
                self.metrics_2d.tick(step, True)  # force now
                snap["metrics_2d"] = self.metrics_2d.get_latest()

            # Health check
            if step % self.health.check_interval == 0:
                decision = self.health.check(step, self.renderer, self.logger)
                if decision == "abort":
                    results["status"] = f"aborted:{self.health.status}"
                    break

            # tqdm postfix
            if step % 50 == 0:
                snap3 = self.metrics_3d.get_latest()
                pbar.set_postfix({
                    "psnr": f"{snap3.get('psnr') or 0:.2f}",
                    "l1":   f"{snap3.get('l1')   or 0:.4f}",
                    "G":    f"{self.renderer.gaussian_count:,}",
                    "ms":   f"{self.timer.rolling_avg_ms:.1f}",
                    "hlth": self.health.status,
                }, refresh=False)

            # Checkpoint
            if step in cp_set:
                snap = save_checkpoint(
                    step, self.renderer, self.settings,
                    self.vol_min, self.vol_max,
                    self.metrics_3d, self.metrics_2d,
                    cfg, self.out_dir, self.logger,
                    snap_w=self.snap_w, snap_h=self.snap_h,
                )
                results["checkpoints"].append(snap)
                # Flush partial results to disk after every checkpoint
                results["step_time_samples"] = self.timer.samples
                _flush_json(results, self.out_dir / "results.json")

        pbar.close()

        # ---- PLY export ----
        if cfg["output"].get("save_ply_at_end", True):
            from save_ply import CloudLightingConfig, save_gaussians_to_ply
            ply_path = self.out_dir / "final_gaussians.ply"
            self.logger.info(f"Saving PLY → {ply_path}")
            gpu_params = (
                self.renderer.gaussian_buffer.to_numpy()
                .view(np.float32)
                .reshape(-1, self._PARAMS_PER_GAUSSIAN)
                .copy()
            )
            out_c = cfg["output"]
            lighting = CloudLightingConfig(
                density_scale    = float(out_c.get("ply_density_scale",     40.0)),
                bright_luminance = float(out_c.get("ply_bright_luminance",  0.95)),
                dark_luminance   = float(out_c.get("ply_dark_luminance",    0.28)),
                depth_gamma      = float(out_c.get("ply_depth_gamma",       2.0)),
                shadow_warmth    = float(out_c.get("ply_shadow_warmth",     0.10)),
            )
            save_gaussians_to_ply(gpu_params, ply_path, lighting)

        # ---- Finalise ----
        if results["status"] == "running":
            results["status"] = "completed"

        results["health_events"]        = self.health.events
        results["final_gaussian_count"] = self.renderer.gaussian_count
        results["wall_time_seconds"]    = time.time() - t_wall
        results["step_time_samples"]    = self.timer.samples
        results["step_time_summary"]    = self.timer.summary()

        _flush_json(results, self.out_dir / "results.json")

        self.logger.info(
            f"Done  status={results['status']}  "
            f"gaussians={results['final_gaussian_count']:,}  "
            f"wall={results['wall_time_seconds']:.1f}s  "
            f"mean_step={results['step_time_summary'].get('mean_ms','?')}ms"
        )
        return results

    def cleanup(self):
        try:
            self._glfw.terminate()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Headless Gaussian splat trainer")
    p.add_argument("--config",     required=True,
                   help="Path to experiment YAML or batch YAML")
    p.add_argument("--experiment", default=None,
                   help="Experiment name (required when --config is a batch YAML)")
    p.add_argument("--output-dir", default=None,
                   help="Override output directory")
    return p.parse_args()


def main():
    args = _parse_args()
    config_path = Path(args.config)

    # ---- Resolve config ----
    try:
        with open(config_path) as f:
            raw = yaml.safe_load(f)
    except Exception as e:
        print(f"[ERROR] Cannot load config: {e}", file=sys.stderr)
        sys.exit(1)

    if "experiments" in raw:
        if args.experiment is None:
            print("[ERROR] --experiment required for batch config", file=sys.stderr)
            sys.exit(1)
        all_cfgs = load_batch(config_path)
        matches  = [c for c in all_cfgs if c.get("_name") == args.experiment]
        if not matches:
            print(f"[ERROR] Experiment '{args.experiment}' not in batch", file=sys.stderr)
            sys.exit(1)
        cfg = matches[0]
    else:
        cfg = load_single(config_path)

    # ---- Output dir ----
    name    = cfg.get("_name", "experiment")
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(cfg["output"]["result_dir"]) / name
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Logger ----
    logger = setup_logger(name, out_dir / "train.log")
    logger.info(f"Experiment : {name}")
    logger.info(f"Description: {cfg.get('_description', '')}")
    logger.info(f"Output dir : {out_dir}")

    # Save exact config for reproducibility
    with open(out_dir / "config_used.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    # ---- Trainer ----
    trainer = HeadlessTrainer(cfg, out_dir, logger)

    def _sigint(sig, frame):
        logger.warning("SIGINT — will stop after current checkpoint")
        trainer._interrupt = True
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    signal.signal(signal.SIGINT, _sigint)

    exit_code = 0
    try:
        trainer.setup()
        results = trainer.train()
        status  = results.get("status", "")
        if status.startswith("aborted:nan"):
            exit_code = 3
        elif status.startswith("aborted"):
            exit_code = 2
    except Exception as e:
        logger.exception(f"Trainer crashed: {e}")
        _flush_json({"status": "crashed", "error": str(e)},
                    out_dir / "crash.json")
        exit_code = 99
    finally:
        trainer.cleanup()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
