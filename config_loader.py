"""
config_loader.py  —  Config loading, deep-merging, validation, LR scheduling.

Merge order (each layer deep-merges on top of the previous):
    defaults.yaml → global_overrides → per-experiment overrides

Lists are REPLACED not extended — intentional for lr_schedule, checkpoints,
snapshot_cameras, etc.  If you want to extend, copy the full list into your
override.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DEFAULTS_PATH = Path(__file__).parent / "config" / "defaults.yaml"


# ---------------------------------------------------------------------------
# Deep merge
# ---------------------------------------------------------------------------

def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into a copy of base. Lists are replaced."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(val, dict)
        ):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def load_defaults() -> dict:
    with open(_DEFAULTS_PATH) as f:
        return yaml.safe_load(f)


def _load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_experiment_config(
    experiment_dict: dict,
    global_overrides: dict | None = None,
    defaults: dict | None = None,
) -> dict:
    """
    Merge: defaults → global_overrides → experiment_dict.
    'name' and 'description' keys are preserved as _name/_description
    but not merged into the config tree.
    """
    if defaults is None:
        defaults = load_defaults()

    cfg = copy.deepcopy(defaults)

    if global_overrides:
        cfg = deep_merge(cfg, global_overrides)

    exp = copy.deepcopy(experiment_dict)
    name = exp.pop("name", "unnamed")
    description = exp.pop("description", "")

    cfg = deep_merge(cfg, exp)
    cfg["_name"] = name
    cfg["_description"] = description
    return cfg


def load_batch(batch_path: str | Path) -> list[dict]:
    """Load a batch YAML → list of fully-resolved experiment configs."""
    raw = _load_yaml(batch_path)
    defaults = load_defaults()
    global_overrides = raw.get("global_overrides", {})

    resolved = []
    for exp in raw.get("experiments", []):
        cfg = build_experiment_config(exp, global_overrides, defaults)
        resolved.append(cfg)
    return resolved


def load_single(config_path: str | Path) -> dict:
    """Load a single-experiment YAML (flat overrides, no 'experiments' list)."""
    raw = _load_yaml(config_path)
    defaults = load_defaults()
    name = raw.pop("name", Path(config_path).stem)
    description = raw.pop("description", "")
    cfg = deep_merge(defaults, raw)
    cfg["_name"] = name
    cfg["_description"] = description
    return cfg


# ---------------------------------------------------------------------------
# Apply config → domain objects
# ---------------------------------------------------------------------------

def apply_config_to_settings(cfg: dict, settings) -> None:
    r = cfg["rendering"]
    settings.step_size        = float(r["step_size"])
    settings.density_scale    = float(r["density_scale"])
    settings.density_curve    = float(r["density_curve"])
    settings.step_count       = int(r["step_count"])
    settings.smoke_color      = list(map(float, r["smoke_color"]))
    settings.light_penetration= float(r["light_penetration"])
    settings.phase_g          = float(r["phase_g"])
    settings.sun_direction    = list(map(float, r["sun_direction"]))
    settings.sun_color_base   = list(map(float, r["sun_color"]))
    settings.sun_intensity    = float(r["sun_intensity"])
    settings.ambient_color_base = list(map(float, r["ambient_color"]))
    settings.ambient_intensity= float(r["ambient_intensity"])
    settings.shadow_steps     = int(r["shadow_steps"])
    settings.shadow_step_mult = float(r["shadow_step_mult"])


def apply_config_to_train_config(cfg: dict, train_config) -> None:
    t = cfg["training"]
    train_config.use_adam             = bool(t["use_adam"])
    train_config.loss_mode            = int(t["loss_mode"])
    train_config.huber_delta          = float(t["huber_delta"])
    train_config.learning_rate_pos    = float(t["learning_rate_pos"])
    train_config.learning_rate_scale  = float(t["learning_rate_scale"])
    train_config.learning_rate_rotation = float(t["learning_rate_rotation"])
    train_config.learning_rate_weight = float(t["learning_rate_weight"])
    train_config.adam_beta1           = float(t["adam_beta1"])
    train_config.adam_beta2           = float(t["adam_beta2"])
    train_config.adam_epsilon         = float(t["adam_epsilon"])

    g = cfg["gaussian_init"]
    train_config.probability_scale    = float(g["probability_scale"])
    train_config.sigma_scale          = float(g["sigma_scale"])
    train_config.jitter_scale         = float(g["jitter_scale"])

    s = cfg["sgld"]
    train_config.sgld.enabled           = bool(s["enabled"])
    train_config.sgld.temperature_start = float(s["temperature_start"])
    train_config.sgld.temperature_decay = float(s["temperature_decay"])
    train_config.sgld.temperature_min   = float(s["temperature_min"])
    train_config.sgld.sgld_k            = float(s["sgld_k"])
    train_config.sgld.reset()


def apply_config_to_adc(cfg: dict, adc_config) -> None:
    from adc import ADCMode
    a = cfg["adc"]

    if not a.get("enabled", True):
        adc_config.mode = ADCMode.DISABLED
        return

    adc_config.mode                  = ADCMode.CPU
    adc_config.start_frame           = int(a["start_frame"])
    adc_config.full_adc_every        = int(a["full_adc_every"])
    adc_config.prune_only_every      = int(a["prune_only_every"])
    adc_config.backoff_factor        = float(a["backoff_factor"])
    adc_config.max_interval          = int(a["max_interval"])
    adc_config.max_gaussians         = int(a["max_gaussians"])
    adc_config.prune_weight_min      = float(a["prune_weight_min"])
    adc_config.prune_scale_max_frac  = float(a["prune_scale_max_frac"])
    adc_config.prune_scale_min_frac  = float(a["prune_scale_min_frac"])
    adc_config.prune_aniso_max       = float(a["prune_aniso_max"])
    adc_config.aniso_thresh          = float(a["aniso_thresh"])
    adc_config.aniso_n_split         = int(a["aniso_n_split"])
    adc_config.aniso_child_overlap   = float(a["aniso_child_overlap"])
    adc_config.grad_n_split          = int(a["grad_n_split"])
    adc_config.grad_size_thresh_frac = float(a["grad_size_thresh_frac"])
    adc_config.grad_percentile       = int(a["grad_percentile"])
    adc_config.clone_n_dense         = int(a["clone_n_dense"])
    adc_config.clone_substance_thresh= float(a["clone_substance_thresh"])
    adc_config.clone_sigma_scale     = float(a["clone_sigma_scale"])
    adc_config.clone_n_sparse        = int(a["clone_n_sparse"])
    adc_config.clone_sparse_min      = float(a["clone_sparse_min"])
    adc_config.clone_sparse_sigma_scale = float(a["clone_sparse_sigma_scale"])
    adc_config.clone_weight_init_frac= float(a["clone_weight_init_frac"])
    adc_config.clone_weight_min      = float(a["clone_weight_min"])
    adc_config.clone_weight_max      = float(a["clone_weight_max"])
    adc_config.clone_error_percentile= int(a["clone_error_percentile"])


def apply_config_to_metrics(cfg: dict, metrics_3d, metrics_2d) -> None:
    m3 = cfg["metrics_3d"]
    c3 = metrics_3d.config
    c3.fast_interval        = int(m3["fast_interval"])
    c3.slow_interval        = int(m3["slow_interval"])
    c3.history_length       = int(m3["history_length"])
    c3.ssim_slices_per_axis = int(m3["ssim_slices_per_axis"])
    c3.iou_threshold        = float(m3["iou_threshold"])
    c3.ssim_window_half     = int(m3["ssim_window_half"])
    c3.ssim_c1              = float(m3["ssim_c1"])
    c3.ssim_c2              = float(m3["ssim_c2"])
    c3.enable_psnr          = bool(m3["enable_psnr"])
    c3.enable_l1            = bool(m3["enable_l1"])
    c3.enable_iou           = bool(m3["enable_iou"])
    c3.enable_ssim          = bool(m3["enable_ssim"])

    m2 = cfg["metrics_2d"]
    c2 = metrics_2d.config
    c2.interval             = int(m2["interval"])
    c2.history_length       = int(m2["history_length"])
    c2.iou_threshold        = float(m2["iou_threshold"])
    c2.ssim_window_half     = int(m2["ssim_window_half"])
    c2.ssim_c1              = float(m2["ssim_c1"])
    c2.ssim_c2              = float(m2["ssim_c2"])
    c2.enable_psnr          = bool(m2["enable_psnr"])
    c2.enable_l1            = bool(m2["enable_l1"])
    c2.enable_iou           = bool(m2["enable_iou"])
    c2.enable_ssim          = bool(m2["enable_ssim"])

    from screen_metrics import CAMERA_PRESETS
    preset = m2.get("camera_preset", "orbit 6")
    if preset in CAMERA_PRESETS:
        metrics_2d._cameras = CAMERA_PRESETS[preset]
        metrics_2d._preset_name = preset
        metrics_2d._reset_per_view_history()


# ---------------------------------------------------------------------------
# LR Scheduler
# ---------------------------------------------------------------------------

class LRScheduler:
    """
    Applies one-shot step-triggered patches to TrainingConfig.
    Call tick() every training step.
    """

    def __init__(self, schedule: list[dict]):
        self._events = sorted(schedule, key=lambda e: int(e["at_step"]))
        self._fired  = [False] * len(self._events)

    def tick(self, step: int, train_config, logger=None) -> bool:
        """Returns True if any patch was applied."""
        applied = False
        for i, event in enumerate(self._events):
            if self._fired[i]:
                continue
            if step >= int(event["at_step"]):
                self._fired[i] = True
                patch = {k: v for k, v in event.items() if k != "at_step"}
                self._apply(patch, train_config)
                msg = f"[LRSchedule] step={step} patch={patch}"
                print(msg)
                if logger:
                    logger.info(msg)
                applied = True
        return applied

    @staticmethod
    def _apply(patch: dict[str, Any], train_config) -> None:
        for key, val in patch.items():
            if hasattr(train_config, key):
                setattr(train_config, key, val)
            elif hasattr(train_config.sgld, key):
                setattr(train_config.sgld, key, val)
            else:
                print(f"[LRSchedule] WARNING: unknown field '{key}' — skipped")


# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------

def resolve_checkpoints(cfg: dict) -> list[int]:
    """
    Expand checkpoint list: replace -1 with max_steps,
    always include 0 and max_steps, return sorted unique list.
    """
    max_steps = cfg["training"]["max_steps"]
    raw = cfg["output"]["checkpoints"]
    resolved = set()
    for c in raw:
        resolved.add(max_steps if c == -1 else int(c))
    resolved.add(0)
    resolved.add(max_steps)
    return sorted(resolved)
