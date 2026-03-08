"""
adc.py  Adaptive Density Control
"""

from __future__ import annotations

import threading
import traceback
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

PARAMS_PER_GAUSSIAN = 11

P_POS = slice(0, 3)
P_SCALE = slice(3, 6)
P_QUAT = slice(6, 10)
P_W = 10


class ADCMode:
    DISABLED = 0
    CPU = 1
    GPU = 2  # stub


@dataclass
class ADCConfig:
    mode: int = ADCMode.CPU

    # scheduling
    start_frame: int = 300
    full_adc_every: int = 400  # base interval; grows with back-off
    prune_only_every: int = 150
    backoff_factor: float = 1.05  # multiply full_adc_every after each run
    max_interval: int = 2000  # cap on the back-off growth
    max_gaussians: int = 50_000

    # pruning
    prune_weight_min: float = 0.005
    prune_scale_max_frac: float = 0.30  # fraction of vol_extent
    prune_scale_min_frac: float = 0.002  # remove dust-sized blobs
    prune_aniso_max: float = 50.0  # max(s)/min(s) before forced split
    prune_contribution_min: float = 1e-5

    # anisotropy split
    aniso_thresh: float = 4.0  # max(s)/min(s) triggers aniso split
    aniso_n_split: int = 60  # max aniso splits per full pass
    aniso_child_overlap: float = 0.55  # weight share per child (sum > 1 intentional)

    # gradient-driven split
    grad_n_split: int = 60
    grad_size_thresh_frac: float = 0.04  # min size (frac of extent) to be candidate
    grad_percentile: int = 80

    # clone
    clone_n_dense: int = 400
    clone_n_sparse: int = 200  # dedicated budget for low-density regions
    clone_substance_thresh: float = 0.02  # dense/sparse boundary
    clone_sparse_min: float = 0.001  # absolute floor for sparse sampling
    clone_sigma_scale: float = 2.0  # multiplied by local voxel size
    clone_sparse_sigma_scale: float = 0.8  # smaller blobs for wispy regions
    clone_weight_init_frac: float = 0.25  # init_weight = ref_val * this
    clone_weight_min: float = 0.04
    clone_weight_max: float = 0.30
    clone_error_percentile: int = 30  # only spawn into top-N% error voxels


class ADCBackend:
    """
    Abstract base class for ADC backends.
    Subclasses must implement run() and declare runs_async.
    """

    @property
    def runs_async(self) -> bool:
        """
        True  → controller spawns a daemon thread and calls run() on it.
        False → controller calls run() inline on the main thread (GPU path,
                which must own the Vulkan context).
        """
        raise NotImplementedError

    def run(
        self,
        params: np.ndarray,  # (N, 11) float32
        grads: np.ndarray,  # (N, 11) float32
        pred: np.ndarray,  # (Z, Y, X) float32
        ref: np.ndarray,  # (Z, Y, X) float32
        config: ADCConfig,
        vol_min: np.ndarray,
        vol_max: np.ndarray,
        do_full: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (new_params, surviving_indices).
        surviving_indices[i] is the index in the *nput params array that
        new_params[i] corresponds to, or -1 for newly spawned Gaussians.
        The controller uses this to transplant Adam moments.
        """
        raise NotImplementedError


class GPUADCBackend(ADCBackend):
    """
    Future GPU-native ADC via Slang kernels.
    Planned:
      1. prune_kernel: compact survivors with InterlockedAdd
      2. compact_kernel: write tight param buffer from survivor list
      3. spawn_kernel: reservoir-sample loss texture, append Gaussians
      4. indirect dispatch: wire new count into DispatchIndirectCommand
    """

    @property
    def runs_async(self) -> bool:
        return False

    def run(self, params, grads, pred, ref, config, vol_min, vol_max, do_full):
        raise NotImplementedError("GPU ADC not yet implemented")


class CPUADCBackend(ADCBackend):
    """Pure-numpy backend. Releases GIL during heavy operations → safe on a thread."""

    @property
    def runs_async(self) -> bool:
        return True

    def run(
        self,
        params: np.ndarray,  # (N, 11) float32
        grads: np.ndarray,  # (N, 11) float32
        pred: np.ndarray,  # (Z, Y, X) float32
        ref: np.ndarray,  # (Z, Y, X) float32
        config: ADCConfig,
        vol_min: np.ndarray,
        vol_max: np.ndarray,
        do_full: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (new_params, surviving_indices).
        surviving_indices maps new_params[:len(survivors)] → original param rows,
        so the caller can transplant Adam moments correctly.
        """
        vol_min = np.asarray(vol_min, dtype=np.float32)
        vol_max = np.asarray(vol_max, dtype=np.float32)
        vol_extent = float(np.max(vol_max - vol_min))

        params, grads, survivors = self._prune(params, grads, config, vol_extent)

        if do_full:
            params, grads, extra_survivors = self._split_anisotropic(
                params, grads, config, vol_extent
            )
            survivors = np.concatenate([survivors, extra_survivors])

            params, grads, extra_survivors = self._split_gradient(
                params, grads, config, vol_extent
            )
            survivors = np.concatenate([survivors, extra_survivors])

            params = self._clone(params, pred, ref, config, vol_min, vol_max)

        return params, survivors

    def _prune(
        self,
        params: np.ndarray,
        grads: np.ndarray,
        config: ADCConfig,
        vol_extent: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(params) == 0:
            return params, grads, np.arange(0, dtype=np.int64)

        weights = params[:, P_W]
        scales = params[:, P_SCALE]
        max_scale = scales.max(axis=1)
        min_scale = scales.min(axis=1)
        aniso = max_scale / np.clip(min_scale, 1e-8, None)

        scale_max_world = vol_extent * config.prune_scale_max_frac
        scale_min_world = vol_extent * config.prune_scale_min_frac

        keep = (
            (weights >= config.prune_weight_min)
            & (max_scale <= scale_max_world)
            & (max_scale >= scale_min_world)
            & (aniso <= config.prune_aniso_max)  # extreme needles get pruned
        )

        n_pruned = int((~keep).sum())
        if n_pruned:
            print(
                f"[ADC] Pruned {n_pruned}/{len(params)}  "
                f"(weight<{config.prune_weight_min:.4f}: "
                f"{int((weights < config.prune_weight_min).sum())}, "
                f"oversized: {int((max_scale > scale_max_world).sum())}, "
                f"dust: {int((max_scale < scale_min_world).sum())}, "
                f"extreme-aniso: {int((aniso > config.prune_aniso_max).sum())})"
            )

        surviving = np.where(keep)[0]
        if len(surviving) == 0:
            print("[ADC] Prune removed everything — keeping top-10 by weight")
            surviving = np.argsort(weights)[-10:]

        return params[surviving], grads[surviving], surviving

    def _split_anisotropic(self, params, grads, config, vol_extent):
        if len(params) == 0 or config.aniso_n_split == 0:
            return params, grads, np.arange(0, dtype=np.int64)

        scales = params[:, P_SCALE]
        max_scale = scales.max(axis=1)
        min_scale = scales.min(axis=1)
        aniso = max_scale / np.clip(min_scale, 1e-8, None)

        candidates = np.where(
            (aniso >= config.aniso_thresh) & (params[:, P_W] > config.prune_weight_min)
        )[0]

        if len(candidates) == 0:
            return params, grads, np.arange(0, dtype=np.int64)

        candidates = candidates[np.argsort(aniso[candidates])[::-1]]
        to_split = candidates[: config.aniso_n_split]

        keep_mask = np.ones(len(params), dtype=bool)
        new_rows = []
        extra_surv = []

        for idx in to_split:
            g = params[idx].copy()
            long_axis = int(np.argmax(g[P_SCALE]))
            long_scale = g[3 + long_axis]

            q = g[P_QUAT]
            R = _quat_to_matrix(q)
            axis_dir = R[:, long_axis]

            g1, g2 = g.copy(), g.copy()
            offset = axis_dir * long_scale * 0.5
            g1[P_POS] = g[P_POS] + offset
            g2[P_POS] = g[P_POS] - offset
            g1[3 + long_axis] = long_scale * 0.5
            g2[3 + long_axis] = long_scale * 0.5

            # Floor short axes relative to the new long axis
            child_long = long_scale * 0.5
            min_short = max(child_long * 0.25, vol_extent * 0.002)
            for ax in range(3):
                if ax != long_axis:
                    g1[3 + ax] = max(g1[3 + ax], min_short)
                    g2[3 + ax] = max(g2[3 + ax], min_short)

            g1[P_W] = g[P_W] * config.aniso_child_overlap
            g2[P_W] = g[P_W] * config.aniso_child_overlap

            keep_mask[idx] = False
            new_rows.extend([g1, g2])
            extra_surv.extend([-1, -1])

        survivors_kept = np.where(keep_mask)[0]
        survivor_indices = np.concatenate(
            [survivors_kept, np.full(len(extra_surv), -1, dtype=np.int64)]
        )

        result = params[keep_mask]
        if new_rows:
            result = np.vstack([result, np.array(new_rows, dtype=np.float32)])
            print(
                f"[ADC] Aniso-split {len(to_split)} → +{len(to_split)} children "
                f"(aniso thresh {config.aniso_thresh:.1f}x)"
            )

        new_grads = np.zeros((len(new_rows), PARAMS_PER_GAUSSIAN), dtype=np.float32)
        result_grads = (
            np.vstack([grads[keep_mask], new_grads]) if new_rows else grads[keep_mask]
        )
        return result, result_grads, survivor_indices

    def _split_gradient(self, params, grads, config, vol_extent):
        if len(params) == 0 or config.grad_n_split == 0:
            return params, grads, np.arange(0, dtype=np.int64)

        weights = params[:, P_W]
        scales = params[:, P_SCALE]
        max_scale = scales.max(axis=1)
        pos_grad = np.linalg.norm(grads[:, P_POS], axis=1)

        size_thresh = vol_extent * config.grad_size_thresh_frac
        nz_grads = pos_grad[pos_grad > 0]
        if len(nz_grads) == 0:
            return params, grads, np.arange(0, dtype=np.int64)

        grad_thresh = float(np.percentile(nz_grads, config.grad_percentile))
        candidates = np.where(
            (max_scale >= size_thresh)
            & (pos_grad >= grad_thresh)
            & (weights > config.prune_weight_min)
        )[0]

        if len(candidates) == 0:
            return params, grads, np.arange(0, dtype=np.int64)

        candidates = candidates[np.argsort(pos_grad[candidates])[::-1]]
        to_split = candidates[: config.grad_n_split]

        keep_mask = np.ones(len(params), dtype=bool)
        new_rows = []

        for idx in to_split:
            g = params[idx].copy()
            long_axis = int(np.argmax(g[P_SCALE]))
            long_scale = g[3 + long_axis]

            q = g[P_QUAT]
            R = _quat_to_matrix(q)
            axis_dir = R[:, long_axis]

            g1, g2 = g.copy(), g.copy()
            offset = axis_dir * long_scale * 0.5
            g1[P_POS] = g[P_POS] + offset
            g2[P_POS] = g[P_POS] - offset
            g1[3 + long_axis] = long_scale * 0.5
            g2[3 + long_axis] = long_scale * 0.5

            # Same short-axis floor as aniso split
            child_long = long_scale * 0.5
            min_short = max(child_long * 0.25, vol_extent * 0.002)
            for ax in range(3):
                if ax != long_axis:
                    g1[3 + ax] = max(g1[3 + ax], min_short)
                    g2[3 + ax] = max(g2[3 + ax], min_short)

            g1[P_W] = g[P_W] * 0.6
            g2[P_W] = g[P_W] * 0.6

            keep_mask[idx] = False
            new_rows.extend([g1, g2])

        survivors_kept = np.where(keep_mask)[0]
        survivor_indices = np.concatenate(
            [survivors_kept, np.full(len(new_rows), -1, dtype=np.int64)]
        )

        result = params[keep_mask]
        if new_rows:
            result = np.vstack([result, np.array(new_rows, dtype=np.float32)])
            print(f"[ADC] Grad-split {len(to_split)} → +{len(to_split)} children")

        new_grads = np.zeros((len(new_rows), PARAMS_PER_GAUSSIAN), dtype=np.float32)
        result_grads = (
            np.vstack([grads[keep_mask], new_grads]) if new_rows else grads[keep_mask]
        )
        return result, result_grads, survivor_indices

    def _clone(
        self,
        params: np.ndarray,
        pred: np.ndarray,
        ref: np.ndarray,
        config: ADCConfig,
        vol_min: np.ndarray,
        vol_max: np.ndarray,
    ) -> np.ndarray:
        """
        Two-budget clone:
          dense: voxels where ref > substance_thresh
          sparse: voxels where sparse_min < ref <= substance_thresh
        """
        if len(params) >= config.max_gaussians:
            print(f"[ADC] At cap ({config.max_gaussians}), skipping clone")
            return params

        Z, Y, X = ref.shape
        vol_extent = vol_max - vol_min
        voxel_size = vol_extent / np.array([X, Y, Z], dtype=np.float32)

        under = np.clip(ref.astype(np.float32) - pred.astype(np.float32), 0.0, None)

        budget_left = config.max_gaussians - len(params)
        n_dense = min(config.clone_n_dense, budget_left)
        n_sparse = min(config.clone_n_sparse, budget_left - n_dense)

        new_rows: list[list] = []

        # dense pass
        if n_dense > 0:
            dense_err = under * (ref > config.clone_substance_thresh).astype(np.float32)
            rows = self._sample_sites(
                dense_err,
                n_dense,
                ref,
                vol_min,
                vol_max,
                voxel_size,
                config.clone_sigma_scale,
                config.clone_weight_init_frac,
                config.clone_weight_min,
                config.clone_weight_max,
                config.clone_error_percentile,
                Z,
                Y,
                X,
            )
            new_rows.extend(rows)
            if rows:
                print(f"[ADC] Dense clone: +{len(rows)} Gaussians")

        # sparse pass
        if n_sparse > 0:
            sparse_mask = (ref > config.clone_sparse_min) & (
                ref <= config.clone_substance_thresh
            )
            sparse_err = under * sparse_mask.astype(np.float32)
            rows = self._sample_sites(
                sparse_err,
                n_sparse,
                ref,
                vol_min,
                vol_max,
                voxel_size,
                config.clone_sparse_sigma_scale,
                config.clone_weight_init_frac * 0.5,
                config.clone_weight_min * 0.5,
                config.clone_weight_max * 0.3,
                config.clone_error_percentile,
                Z,
                Y,
                X,
            )
            new_rows.extend(rows)
            if rows:
                print(f"[ADC] Sparse clone: +{len(rows)} Gaussians (wispy regions)")

        if not new_rows:
            return params

        new_arr = np.array(new_rows, dtype=np.float32)
        total = len(params) + len(new_arr)
        print(f"[ADC] Clone done — total Gaussians: {total:,}")
        return np.vstack([params, new_arr])

    def _sample_sites(
        self,
        error_map: np.ndarray,
        n_spawn: int,
        ref: np.ndarray,
        vol_min: np.ndarray,
        vol_max: np.ndarray,
        voxel_size: np.ndarray,
        sigma_scale: float,
        weight_init_frac: float,
        weight_min: float,
        weight_max: float,
        error_percentile: int,
        Z: int,
        Y: int,
        X: int,
    ) -> list:
        """
        Weighted-random sampling of spawn sites from error_map.
        Applies a percentile floor so we only spawn into the worst-error voxels.
        """
        total = float(error_map.sum())
        if total < 1e-8:
            return []

        nz = error_map[error_map > 0]
        if len(nz) == 0:
            return []

        thresh = float(np.percentile(nz, error_percentile))
        masked = error_map.copy()
        masked[masked < thresh] = 0.0

        flat = masked.flatten()
        s = flat.sum()
        if s < 1e-8:
            return []

        probs = flat / s
        n_nz = int(np.count_nonzero(flat))
        replace = n_nz < n_spawn
        chosen = np.random.choice(len(flat), size=n_spawn, replace=replace, p=probs)

        zs = chosen // (Y * X)
        ys = (chosen % (Y * X)) // X
        xs = chosen % X

        sigma_world = float(np.mean(voxel_size)) * sigma_scale

        rows = []
        for i in range(n_spawn):
            uvw = np.array([xs[i] + 0.5, ys[i] + 0.5, zs[i] + 0.5], np.float32)
            uvw /= np.array([X, Y, Z], np.float32)
            world_pos = vol_min + uvw * (vol_max - vol_min)
            jitter = (np.random.rand(3).astype(np.float32) - 0.5) * voxel_size * 0.3
            world_pos = world_pos + jitter

            local_ref = float(ref[zs[i], ys[i], xs[i]])
            init_weight = float(
                np.clip(local_ref * weight_init_frac, weight_min, weight_max)
            )

            rows.append(
                [
                    world_pos[0],
                    world_pos[1],
                    world_pos[2],
                    sigma_world,
                    sigma_world,
                    sigma_world,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    init_weight,
                ]
            )
        return rows


class ADCController:
    """
    Owns scheduling, threading, momentum transplant and imgui UI.
    """

    def __init__(self, app_ref, config: ADCConfig):
        self.app = app_ref
        self.config = config

        self._backends: dict[int, ADCBackend] = {
            ADCMode.CPU: CPUADCBackend(),
            ADCMode.GPU: GPUADCBackend(),
        }

        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._pending_params = None
        # Scheduling state
        self._current_interval = config.full_adc_every
        self._next_full_frame = config.start_frame + config.full_adc_every
        self._next_prune_frame = config.start_frame + config.prune_only_every
        # UI history
        self.gaussian_count_history: list[int] = []
        self.last_adc_frame = 0
        self.last_adc_type = "none"
        # Cached reference volume (never changes)
        self._ref_cache: Optional[np.ndarray] = None

    def tick(self, frame: int, is_training: bool):
        """Call every frame from the main loop."""
        # Keep sigma_scale in sync with training config
        self.config.clone_sigma_scale = self.app.train_config.sigma_scale

        if not is_training:
            return
        if self.config.mode == ADCMode.DISABLED:
            return
        if frame < self.config.start_frame:
            return

        # Adaptive prune threshold based on current count
        n = self.app.renderer.gaussian_count
        if n < 10_000:
            self.config.prune_weight_min = 0.004
        elif n < 15_000:
            self.config.prune_weight_min = 0.007
        elif n < 25_000:
            self.config.prune_weight_min = 0.010
        else:
            self.config.prune_weight_min = 0.013

        if frame >= self._next_full_frame:
            self._trigger(do_full=True)
            # Exponential back-off: interval grows as training matures
            self._current_interval = min(
                int(self._current_interval * self.config.backoff_factor),
                self.config.max_interval,
            )
            self._next_full_frame = frame + self._current_interval
            self._next_prune_frame = frame + self.config.prune_only_every
            self.last_adc_frame = frame

        elif frame >= self._next_prune_frame:
            self._trigger(do_full=False)
            self._next_prune_frame = frame + self.config.prune_only_every

    def apply_pending(self):
        """Zero-cost when nothing is ready. Call every frame."""
        if self.config.mode == ADCMode.DISABLED:
            return
        with self._lock:
            if self._pending_params is None:
                return
            params, surviving_indices = self._pending_params
            self._pending_params = None

        self.app.apply_densification(params, surviving_indices)
        self.gaussian_count_history.append(len(params))
        if len(self.gaussian_count_history) > 300:
            self.gaussian_count_history.pop(0)

    def _trigger(self, do_full: bool):
        if self.config.mode == ADCMode.DISABLED:
            return
        if self._running:
            print("[ADC] Previous pass still running, skipping")
            return

        backend = self._backends.get(self.config.mode)
        if backend is None:
            return

        if self.config.mode == ADCMode.GPU:
            print("[ADC] GPU backend not yet implemented, falling back to CPU")
            backend = self._backends[ADCMode.CPU]

        snapshots = self._take_snapshots()
        if snapshots is None:
            return

        self._running = True
        self.last_adc_type = "Full" if do_full else "Prune-only"
        params_snap, grads_snap, pred_snap, ref_snap = snapshots

        if backend.runs_async:
            self._thread = threading.Thread(
                target=self._worker,
                args=(backend, params_snap, grads_snap, pred_snap, ref_snap, do_full),
                daemon=True,
            )
            self._thread.start()
        else:
            self._worker(backend, params_snap, grads_snap, pred_snap, ref_snap, do_full)

        print(
            f"[ADC] {self.last_adc_type} pass started "
            f"({'async' if backend.runs_async else 'inline'}, "
            f"{len(params_snap):,} Gaussians, interval={self._current_interval})"
        )

    def _take_snapshots(self):
        """GPU readbacks must run on the main thread."""
        try:
            r = self.app.renderer

            params_snap = (
                r.gaussian_buffer.to_numpy()
                .view(np.float32)
                .reshape(-1, PARAMS_PER_GAUSSIAN)
                .copy()
            )
            grads_snap = (
                r.grad_buffer.to_numpy()
                .view(np.float32)
                .reshape(-1, PARAMS_PER_GAUSSIAN)
                .copy()
            )

            # Fresh rasterise so prediction matches current state
            cmd = self.app.device.create_command_encoder()
            r.rasterize_gaussians(cmd, self.app.vol_min_world, self.app.vol_max_world)
            self.app.device.submit_command_buffer(cmd.finish())
            pred_snap = r.gaussian_volume_tex.to_numpy().copy()

            if self._ref_cache is None:
                self._ref_cache = r.volume_tex.to_numpy().copy()
                print("[ADC] Reference volume cached")

            return params_snap, grads_snap, pred_snap, self._ref_cache

        except Exception as e:
            print(f"[ADC] Snapshot failed: {e}")
            traceback.print_exc()
            self._running = False
            return None

    def _worker(self, backend: ADCBackend, params, grads, pred, ref, do_full):
        try:
            result_params, surviving_indices = backend.run(
                params,
                grads,
                pred,
                ref,
                self.config,
                np.asarray(self.app.vol_min_world, np.float32),
                np.asarray(self.app.vol_max_world, np.float32),
                do_full,
            )
            with self._lock:
                self._pending_params = (result_params, surviving_indices)

        except Exception as e:
            print(f"[ADC] Worker error: {e}")
            traceback.print_exc()
        finally:
            self._running = False

    def draw_ui(self):
        from imgui_bundle import imgui

        if not imgui.begin("ADC — Adaptive Density Control"):
            imgui.end()
            return

        # ── mode ──────────────────────────────────────────────────────────────
        modes = ["Disabled", "CPU (Async)", "GPU (Stub)"]
        _, self.config.mode = imgui.combo("Backend", self.config.mode, modes)
        if self.config.mode == ADCMode.GPU:
            imgui.text_colored(
                imgui.ImVec4(1, 0.5, 0, 1),
                "GPU ADC not yet implemented — falls back to CPU",
            )

        imgui.separator()

        # ── status ────────────────────────────────────────────────────────────
        if self._running:
            imgui.text_colored(
                imgui.ImVec4(1, 1, 0, 1), f"Running: {self.last_adc_type}..."
            )
        elif self.config.mode == ADCMode.DISABLED:
            imgui.text_colored(imgui.ImVec4(0.5, 0.5, 0.5, 1), "Disabled")
        else:
            next_in = max(0, self._next_full_frame - self.last_adc_frame)
            imgui.text_colored(
                imgui.ImVec4(0.4, 1, 0.4, 1), f"Idle  (next full in ~{next_in} frames)"
            )

        n_current = self.app.renderer.gaussian_count
        imgui.text(f"Gaussians: {n_current:,} / {self.config.max_gaussians:,}")
        fill = float(n_current) / float(max(self.config.max_gaussians, 1))
        imgui.progress_bar(
            fill, imgui.ImVec2(-1, 0), f"{n_current:,} / {self.config.max_gaussians:,}"
        )

        if len(self.gaussian_count_history) > 1:
            arr = np.array(self.gaussian_count_history, dtype=np.float32)
            imgui.plot_lines(
                "##gh",
                arr,
                scale_min=0.0,
                scale_max=float(self.config.max_gaussians),
                graph_size=imgui.ImVec2(300, 50),
            )

        imgui.separator()

        # ── scheduling ────────────────────────────────────────────────────────
        imgui.text("Scheduling")
        _, self.config.start_frame = imgui.slider_int(
            "Start Frame", self.config.start_frame, 0, 2000
        )
        _, self.config.full_adc_every = imgui.slider_int(
            "Base Full Interval", self.config.full_adc_every, 50, 1000
        )
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Interval grows automatically via back-off as training matures"
            )
        _, self.config.prune_only_every = imgui.slider_int(
            "Prune Interval", self.config.prune_only_every, 20, 500
        )
        _, self.config.backoff_factor = imgui.slider_float(
            "Back-off Factor", self.config.backoff_factor, 1.0, 2.0, "%.2f"
        )
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "After each full ADC, interval *= this. 1.0 = no back-off."
            )
        _, self.config.max_interval = imgui.slider_int(
            "Max Interval", self.config.max_interval, 200, 5000
        )
        _, self.config.max_gaussians = imgui.slider_int(
            "Max Gaussians", self.config.max_gaussians, 1000, 200_000
        )
        imgui.text(f"Current interval: {self._current_interval}")

        imgui.separator()

        # ── pruning ───────────────────────────────────────────────────────────
        imgui.text("Pruning")
        _, self.config.prune_weight_min = imgui.slider_float(
            "Min Weight", self.config.prune_weight_min, 0.0001, 0.05, "%.4f"
        )
        _, self.config.prune_scale_max_frac = imgui.slider_float(
            "Max Scale (frac)", self.config.prune_scale_max_frac, 0.05, 1.0, "%.2f"
        )
        _, self.config.prune_scale_min_frac = imgui.slider_float(
            "Min Scale (frac)", self.config.prune_scale_min_frac, 1e-5, 0.01, "%.5f"
        )
        _, self.config.prune_aniso_max = imgui.slider_float(
            "Max Anisotropy", self.config.prune_aniso_max, 2.0, 100.0, "%.1f"
        )
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Gaussians more needle-like than this ratio are removed.\nUsually you want this >= aniso_thresh * 2."
            )

        imgui.separator()

        # ── anisotropy split ──────────────────────────────────────────────────
        imgui.text("Anisotropy Split")
        _, self.config.aniso_thresh = imgui.slider_float(
            "Aniso Threshold", self.config.aniso_thresh, 1.5, 20.0, "%.1f"
        )
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "max(s)/min(s) above which a Gaussian is split along its long axis"
            )
        _, self.config.aniso_n_split = imgui.slider_int(
            "Max Aniso Splits", self.config.aniso_n_split, 0, 300
        )
        _, self.config.aniso_child_overlap = imgui.slider_float(
            "Child Weight Share", self.config.aniso_child_overlap, 0.4, 0.8, "%.2f"
        )
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Each child gets parent_weight * this.\n>0.5 means slight overlap (usually desired)."
            )

        imgui.separator()

        # ── gradient split ────────────────────────────────────────────────────
        imgui.text("Gradient-driven Split")
        _, self.config.grad_n_split = imgui.slider_int(
            "Max Grad Splits", self.config.grad_n_split, 0, 200
        )
        _, self.config.grad_size_thresh_frac = imgui.slider_float(
            "Min Size (frac)", self.config.grad_size_thresh_frac, 0.01, 0.2, "%.3f"
        )
        _, self.config.grad_percentile = imgui.slider_int(
            "Grad Percentile", self.config.grad_percentile, 50, 99
        )
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Only split Gaussians whose pos-grad magnitude exceeds this percentile"
            )

        imgui.separator()

        # ── clone ─────────────────────────────────────────────────────────────
        imgui.text("Clone — Dense regions")
        _, self.config.clone_n_dense = imgui.slider_int(
            "Dense Count", self.config.clone_n_dense, 0, 500
        )
        _, self.config.clone_substance_thresh = imgui.slider_float(
            "Dense Threshold", self.config.clone_substance_thresh, 0.005, 0.2, "%.3f"
        )
        if imgui.is_item_hovered():
            imgui.set_tooltip("Voxels above this ref-density count as 'dense'")
        _, self.config.clone_sigma_scale = imgui.slider_float(
            "Sigma Scale", self.config.clone_sigma_scale, 0.5, 5.0, "%.2f"
        )

        imgui.dummy((0, 4))
        imgui.text("Clone — Sparse / wispy regions")
        _, self.config.clone_n_sparse = imgui.slider_int(
            "Sparse Count", self.config.clone_n_sparse, 0, 300
        )
        _, self.config.clone_sparse_min = imgui.slider_float(
            "Sparse Floor", self.config.clone_sparse_min, 0.0001, 0.02, "%.4f"
        )
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Absolute minimum ref-density to still be a valid sparse spawn site.\n"
                "Voxels below this are truly empty and ignored."
            )
        _, self.config.clone_sparse_sigma_scale = imgui.slider_float(
            "Sparse Sigma", self.config.clone_sparse_sigma_scale, 0.2, 2.0, "%.2f"
        )
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Smaller sigma keeps wispy Gaussians from bleeding into dense neighbours"
            )

        imgui.dummy((0, 4))
        imgui.text("Clone — Shared")
        _, self.config.clone_weight_init_frac = imgui.slider_float(
            "Weight Init Frac", self.config.clone_weight_init_frac, 0.05, 0.8, "%.2f"
        )
        _, self.config.clone_weight_min = imgui.slider_float(
            "Weight Min", self.config.clone_weight_min, 0.001, 0.1, "%.3f"
        )
        _, self.config.clone_weight_max = imgui.slider_float(
            "Weight Max", self.config.clone_weight_max, 0.05, 0.8, "%.2f"
        )
        _, self.config.clone_error_percentile = imgui.slider_int(
            "Error Percentile", self.config.clone_error_percentile, 0, 90
        )
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Only spawn into voxels whose under-prediction error exceeds this percentile.\n"
                "Lower = more permissive spawning.\n"
                "0 = spawn anywhere there is any under-prediction."
            )

        imgui.separator()

        # ── manual triggers ───────────────────────────────────────────────────
        disabled = self._running or self.config.mode == ADCMode.DISABLED
        if disabled:
            imgui.begin_disabled()
        if imgui.button("Run Full ADC Now"):
            self._trigger(do_full=True)
        imgui.same_line()
        if imgui.button("Prune Only"):
            self._trigger(do_full=False)
        if disabled:
            imgui.end_disabled()

        imgui.end()


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """q = [x, y, z, w]  →  3×3 rotation matrix  (column-major matches shader)"""
    x, y, z, w = q[0], q[1], q[2], q[3]
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
