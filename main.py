import slangpy as spy
import numpy as np
import glfw
from OpenGL.GL import *
from pathlib import Path
import math
import ctypes 
from imgui_bundle import imgui
from slangpy.math import uint3
import random
import openvdb as vdb
import itertools
import time
import threading
import traceback

from save_ply import save_gaussians_dialog, CloudLightingConfig


# ==========================================
# CONFIGURATION
# ==========================================

ENABLE_PROFILING = True  # Toggle detailed timing measurements
PROFILE_EVERY_N_FRAMES = 10  # Only profile every N iterations to reduce overhead
PROFILE_WAIT_FOR_GPU = False

# Debug Output
PRINT_TILE_STATS = True  # Print tiling statistics
PRINT_GRADIENT_STATS = True # Print gradient statistics

VDB_FILE = "cloud_04_variant_0000.vdb" 
VOL_SIZE = 196
SHADER_FILE = "shaders/hybrid.slang"
TILE_SIZE = 4
MAX_GAUSSIANS_PER_TILE = 256

# Window Settings
WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 768
WINDOW_TITLE = "VDB Editor"

# Camera Defaults
CAMERA_START_POS = [0.0, 0.0, 2.5]
CAMERA_SPEED = 2.0
CAMERA_SENSITIVITY = 0.1

PARAMS_PER_GAUSSIAN = 11  # pos(3) + scale(3) + quat(4) + weight(1)

# ==========================================
# ADC CONFIGURATION
# ==========================================
ADC_START_FRAME      = 300    # Don't densify until Adam has settled
ADC_EVERY            = 200    # Full ADC (prune + split + clone) interval
PRUNE_ONLY_EVERY     = 100    # Prune-only pass between full ADC cycles
ADC_MAX_GAUSSIANS    = 50000  # Hard cap — never exceed this
ADC_PRUNE_WEIGHT_MIN = 0.005  # Gaussians below this weight get pruned
ADC_PRUNE_SCALE_MAX  = 0.3    # Fraction of vol_extent — blobs larger than this get pruned
ADC_N_CLONE          = 40     # New Gaussians spawned per clone pass
ADC_N_SPLIT          = 20     # Gaussians split per split pass
ADC_SUBSTANCE_THRESH = 0.02   # Min reference density to be a valid spawn site

class DebugMode:
    NONE = 0
    LOSS = 1
    GRADIENT_MAGNITUDE = 2
    TILE_DENSITY = 3
    GAUSSIAN_OVERLAP = 4
    PREDICTION_VS_TARGET = 5


# ==========================================
# DATA & LOGIC CLASSES
# ==========================================

class ProfilingContext:
    """
    Lightweight profiling context manager.
    Zero overhead when ENABLE_PROFILING=False.
    """
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.timings = {}
        self.current_label = None
        self.start_time = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
    
    def start(self, label):
        """Start timing a labeled section"""
        if not self.enabled:
            return
        self.current_label = label
        self.start_time = time.perf_counter()
    
    def stop(self):
        """Stop timing current section"""
        if not self.enabled or self.current_label is None:
            return
        elapsed = time.perf_counter() - self.start_time
        self.timings[self.current_label] = elapsed
        self.current_label = None
    
    def print_summary(self, title="Profiling Results"):
        """Print timing summary"""
        if not self.enabled or not self.timings:
            return
        
        print(f"\n{'='*60}")
        print(f"{title:^60}")
        print(f"{'='*60}")
        
        total = sum(self.timings.values())
        
        # Sort by time (slowest first)
        sorted_timings = sorted(self.timings.items(), key=lambda x: x[1], reverse=True)
        
        for label, t in sorted_timings:
            percent = (t / total * 100) if total > 0 else 0
            print(f"  {label:30s}: {t*1000:7.2f} ms  ({percent:5.1f}%)")
        
        print(f"  {'-'*58}")
        print(f"  {'TOTAL':30s}: {total*1000:7.2f} ms")
        print(f"{'='*60}\n")

class Settings:
    def __init__(self):
        # Raymarcher
        self.step_size = 0.015
        self.density_scale = 40.0
        self.density_curve = 1.0
        self.step_count = 256
        self.smoke_color = [0.9, 0.95, 1.0]

        # Lighting
        self.light_penetration = 0.15
        self.phase_g = 0.4
        
        self.sun_direction = [1.0, -1.0, 0.0]
        self.sun_color_base = [1.0, 0.9, 0.7]
        self.sun_intensity = 8.0
        
        self.ambient_color_base = [0.6, 0.7, 0.8]
        self.ambient_intensity = 0.6
        
        self.shadow_steps = 6
        self.shadow_step_mult = 4.0
    
    def get_sun_dir(self):
        return self.sun_direction


class Camera:
    def __init__(self, w, h):
        self.pos = np.array(CAMERA_START_POS, dtype=np.float32)
        self.yaw = -90.0
        self.pitch = 0.0
        self.front = np.array([0, 0, -1], dtype=np.float32)
        self.right = np.array([1, 0, 0], dtype=np.float32)
        self.up = np.array([0, 1, 0], dtype=np.float32)
        self.speed = CAMERA_SPEED
        self.sensitivity = CAMERA_SENSITIVITY
        self.last_x, self.last_y = w / 2.0, h / 2.0
        self.first_mouse = True
        self.is_dragging = False
        self.update_vectors()

    def update_vectors(self):
        rad_yaw = math.radians(self.yaw)
        rad_pitch = math.radians(self.pitch)
        f = np.array([
            math.cos(rad_yaw) * math.cos(rad_pitch),
            math.sin(rad_pitch),
            math.sin(rad_yaw) * math.cos(rad_pitch)
        ], dtype=np.float32)
        self.front = f / np.linalg.norm(f)
        world_up = np.array([0, 1, 0], dtype=np.float32)
        self.right = np.cross(self.front, world_up)
        self.right /= np.linalg.norm(self.right)
        self.up = np.cross(self.right, self.front)
        self.up /= np.linalg.norm(self.up)

    def process_mouse(self, xpos, ypos):
        if self.first_mouse:
            self.last_x, self.last_y = xpos, ypos
            self.first_mouse = False
            return 
        
        xoffset = (xpos - self.last_x) * self.sensitivity
        yoffset = (self.last_y - ypos) * self.sensitivity 
        self.last_x, self.last_y = xpos, ypos
        self.yaw += xoffset
        self.pitch += yoffset 
        self.pitch = np.clip(self.pitch, -89.0, 89.0)
        self.update_vectors()

    def get_gpu_data(self, aspect_ratio):
        data = np.zeros(16, dtype=np.float32)
        data[0:3] = self.pos
        data[4:7] = self.front
        data[8:11] = self.right * aspect_ratio * 0.5 
        data[12:15] = self.up * 0.5
        return data


def load_vdb_grid(vdb_path):
    print(f"Loading VDB file: {vdb_path}...")
    try:
        if not Path(vdb_path).exists():
            raise FileNotFoundError("File missing")
        
        raw = vdb.readAll(vdb_path)
        grid = raw[0][0] if isinstance(raw, (list, tuple)) else raw
        return grid
    except Exception as e:
        print(f"Error loading VDB: {e}")
        return None


def convert_grid_to_dense_volume(grid, size):
    """Convert sparse VDB grid to dense 3D texture"""
    if grid is None:
        print("Error! Grid is None!")
        return None

    bbox = grid.evalActiveVoxelBoundingBox()
    min_i, max_i = np.array(bbox[0]), np.array(bbox[1])
    center = (min_i + max_i) / 2.0
    
    extent = np.max(max_i - min_i)
    r = extent / 2.0

    min_index_bound = center - r
    max_index_bound = center + r
    
    transform = grid.transform
    min_w = np.array(transform.indexToWorld(tuple(min_index_bound)))
    max_w = np.array(transform.indexToWorld(tuple(max_index_bound)))
    
    accessor = grid.getAccessor()
    data = np.zeros((size, size, size), dtype=np.float32)
    inv_size = 1.0 / size
    
    for z, y, x in itertools.product(range(size), range(size), range(size)):
        uvw = (np.array([x, y, z]) * inv_size) - 0.5
        pos = center + uvw * extent
        val = accessor.getValue(tuple(pos.astype(int)))
        data[z, y, x] = val
        
    m = np.max(data)
    if m > 0: 
        data /= m
    
    return min_w, max_w, np.ascontiguousarray(data, dtype=np.float32)


def convert_grid_to_gaussians(grid, config):
    """Stochastically sample VDB grid to create Gaussians"""
    if grid is None: 
        return np.array([], dtype=np.float32)

    print("Generating Gaussians from grid...")
    accessor = grid.getAccessor()
    bbox = grid.evalActiveVoxelBoundingBox()
    min_i, max_i = np.array(bbox[0]), np.array(bbox[1])
    
    transform = grid.transform
    p0 = np.array(transform.indexToWorld((0, 0, 0)))
    p1 = np.array(transform.indexToWorld((1, 0, 0)))
    voxel_size = np.linalg.norm(p1 - p0)

    gaussians = []
    
    for z in range(min_i[2], max_i[2] + 1):
        for y in range(min_i[1], max_i[1] + 1):
            for x in range(min_i[0], max_i[0] + 1):
                value = accessor.getValue((x, y, z))
                if value <= 0.0: 
                    continue

                if value * config.probability_scale < random.random():
                    continue

                center = np.array(transform.indexToWorld((x, y, z)), dtype=np.float32)
                jitter = (np.random.rand(3) - 0.5) * voxel_size * config.jitter_scale
                position = center + jitter
                
                sigma = voxel_size * config.sigma_scale
                weight = value

                gaussians.append((
                    position[0], position[1], position[2],   # pos
                    sigma, sigma, sigma,                     # scale (isotropic start)
                    0.0, 0.0, 0.0, 1.0,                      # quaternion (identity)
                    weight                                   # weight
                ))

    print(f"Generated {len(gaussians)} gaussians.")
    return np.array(gaussians, dtype=np.float32)

class ADCMode:
    DISABLED = 0
    CPU      = 1
    GPU      = 2  # stub, not yet implemented


class ADCConfig:
    def __init__(self):
        self.mode = ADCMode.CPU

        # Scheduling
        self.start_frame      = 300
        self.full_adc_every   = 400
        self.prune_only_every = 200
        self.max_gaussians    = 50000

        # Prune
        self.prune_weight_min = 0.006
        self.prune_scale_max  = 0.3    # fraction of vol_extent

        # Clone
        self.n_clone          = 350
        self.substance_thresh = 0.05

        # Split
        self.n_split               = 20
        self.split_scale_thresh    = 0.04  # fraction of vol_extent
        self.split_grad_percentile = 85

class ADCBackend:
    @property
    def runs_async(self) -> bool:
        return False

    def run(self, params, grads, pred, ref, config, vol_min, vol_max, do_full) -> np.ndarray:
        raise NotImplementedError

class CPUADCBackend(ADCBackend):
    """
    Pure numpy. Safe to run on a background thread because numpy
    releases the GIL during its C-level operations.
    """

    @property
    def runs_async(self):
        return True

    def _prune(self, params, grads, config, vol_min, vol_max):
        if len(params) == 0:
            return params, grads, np.arange(0, dtype=np.int64)

        weights = params[:, 10]
        max_scale = np.max(params[:, 3:6], axis=1)
        vol_extent = float(np.max(np.array(vol_max, dtype=np.float32)
                                - np.array(vol_min, dtype=np.float32)))

        keep = (
            (weights > config.prune_weight_min) &
            (max_scale < vol_extent * config.prune_scale_max)
        )

        n_pruned = int(np.sum(~keep))
        if n_pruned > 0:
            print(f"[ADC:CPU] Pruned {n_pruned} / {len(params)} Gaussians")

        surviving_indices = np.where(keep)[0]
        result_params = params[keep]
        result_grads = grads[keep]

        # fallback
        if len(result_params) == 0:
            print("[ADC:CPU] Warning: pruning removed everything, keeping top-10 by weight")
            top = np.argsort(weights)[-10:]
            result_params = params[top]
            result_grads = grads[top]
            surviving_indices = top

        return result_params, result_grads, surviving_indices


    def run(self, params, grads, pred, ref, config, vol_min, vol_max, do_full):
        params, grads, surviving_indices = self._prune(params, grads, config, vol_min, vol_max)
        if do_full:
            params = self._split(params, grads, config, vol_min, vol_max)
            params = self._clone(params, pred, ref, config, vol_min, vol_max)
        return params, surviving_indices

    def _split(self, params, grads, config, vol_min, vol_max):
        if len(params) == 0:
            return params

        weights    = params[:, 10]
        max_scale  = np.max(params[:, 3:6], axis=1)
        vol_extent = float(np.max(np.array(vol_max, dtype=np.float32)
                                - np.array(vol_min, dtype=np.float32)))

        pos_grad   = np.linalg.norm(grads[:, 0:3], axis=1)

        alive      = weights   > 0.01
        is_large   = max_scale > vol_extent * config.split_scale_thresh

        nonzero_pg = pos_grad[pos_grad > 0]
        if len(nonzero_pg) == 0:
            return params

        grad_thresh = float(np.percentile(nonzero_pg, config.split_grad_percentile))
        has_grad = pos_grad > grad_thresh
        candidates  = np.where(is_large & has_grad & alive)[0]

        if len(candidates) == 0:
            return params

        sorted_cands = candidates[np.argsort(pos_grad[candidates])[::-1]]
        to_split     = sorted_cands[:config.n_split]

        keep_mask = np.ones(len(params), dtype=bool)
        new_rows  = []

        for idx in to_split:
            g          = params[idx]
            keep_mask[idx] = False

            split_axis         = int(np.argmax(g[3:6]))
            offset             = np.zeros(3, dtype=np.float32)
            offset[split_axis] = g[3 + split_axis] * 0.5

            g1 = g.copy(); g2 = g.copy()
            g1[0:3] = g[0:3] + offset
            g2[0:3] = g[0:3] - offset
            g1[3:6] = g[3:6] * 0.7071   # scale / sqrt(2)
            g2[3:6] = g[3:6] * 0.7071
            g1[10]  = g[10]  * 0.6      # shared weight with overlap factor
            g2[10]  = g[10]  * 0.6
            new_rows.extend([g1, g2])

        survivors = params[keep_mask]
        print(f"[ADC:CPU] Split {len(to_split)} Gaussians → +{len(to_split)} children")

        if new_rows:
            return np.vstack([survivors, np.array(new_rows, dtype=np.float32)])
        return survivors

    def _clone(self, params, pred, ref, config, vol_min, vol_max):
        if len(params) >= config.max_gaussians:
            print(f"[ADC:CPU] At Gaussian cap ({config.max_gaussians}), skipping clone")
            return params

        # Positive where we are UNDER-predicting
        under_error = np.clip(
            ref.astype(np.float32) - pred.astype(np.float32), 0.0, None)

        # Only spawn where the reference has actual substance
        under_error *= (ref > config.substance_thresh).astype(np.float32)

        if np.any(under_error > 0):
            # Only spawn into top 40% worst voxels
            # As training matures, raise this to 60-70% but I did not do that yet
            thresh = float(np.percentile(under_error[under_error > 0], 60))
            under_error[under_error < thresh] = 0.0

        total_error = float(under_error.sum())
        if total_error < 1e-8:
            print("[ADC:CPU] No significant under-prediction, skipping clone")
            return params

        n_spawn = min(config.n_clone, config.max_gaussians - len(params))
        if n_spawn <= 0:
            return params

        flat      = under_error.flatten()
        probs     = flat / flat.sum()
        n_nonzero = int(np.count_nonzero(flat))
        replace   = n_nonzero < n_spawn
        chosen    = np.random.choice(len(flat), size=n_spawn,
                                     replace=replace, p=probs)

        Z, Y, X = ref.shape
        zs = chosen // (Y * X)
        ys = (chosen % (Y * X)) // X
        xs = chosen % X

        vol_min_a  = np.array(vol_min, dtype=np.float32)
        vol_max_a  = np.array(vol_max, dtype=np.float32)
        vol_extent = vol_max_a - vol_min_a
        voxel_size = vol_extent / np.array([X, Y, Z], dtype=np.float32)

        new_rows = []
        for i in range(n_spawn):
            uvw       = np.array([xs[i]+0.5, ys[i]+0.5, zs[i]+0.5], dtype=np.float32)
            uvw      /= np.array([X, Y, Z], dtype=np.float32)
            world_pos = vol_min_a + uvw * vol_extent
            jitter    = (np.random.rand(3).astype(np.float32) - 0.5) * voxel_size * 0.5
            world_pos = world_pos + jitter

            voxel_size_world = vol_extent / np.array([X, Y, Z], dtype=np.float32)
            sigma = float(voxel_size_world.mean() * config.clone_sigma_scale)
            local_ref   = float(ref[zs[i], ys[i], xs[i]])
            init_weight = float(np.clip(local_ref * 0.3, 0.05, 0.3))

            new_rows.append([
                world_pos[0], world_pos[1], world_pos[2],
                sigma, sigma, sigma,
                0.0, 0.0, 0.0, 1.0,
                init_weight
            ])

        new_arr = np.array(new_rows, dtype=np.float32)
        print(f"[ADC:CPU] Cloned {n_spawn} Gaussians "
              f"(total → {len(params) + n_spawn})")
        return np.vstack([params, new_arr])


class GPUADCBackend(ADCBackend):
    """
    Future GPU-native ADC via Slang kernels.
    Planned:
      1. prune_kernel   - compact survivors with InterlockedAdd
      2. compact_kernel - write tight param buffer from survivor list
      3. spawn_kernel   - reservoir-sample loss texture, append Gaussians
      4. indirect dispatch - wire new count into DispatchIndirectCommand
    """
    @property
    def runs_async(self):
        # GPU backend runs inline on main thread (slangpy requires the
        # Vulkan context owner thread). GPU itself is the async engine.
        return False

    def run(self, params, grads, pred, ref, config, vol_min, vol_max, do_full):
        raise NotImplementedError("GPU ADC not yet implemented")

class ADCController:
    """
    Owns all ADC scheduling, threading, and UI.

    App calls:
        controller.tick(frame, is_training): every frame, handles scheduling
        controller.apply_pending(): every frame, zero-cost if nothing ready
        controller.draw_ui(): self-contained imgui panel
    """

    def __init__(self, app_ref, config: ADCConfig):
        self.app    = app_ref
        self.config = config

        self._backends = {
            ADCMode.CPU: CPUADCBackend(),
            ADCMode.GPU: GPUADCBackend(),
        }

        # Threading state (CPU path only)
        self._lock           = threading.Lock()
        self._thread         = None
        self._running        = False
        self._pending_params = None

        # Stats for UI
        self.gaussian_count_history = []
        self.last_adc_frame         = 0
        self.last_adc_type          = "none"

        # Cached reference volume
        self._ref_cache = None

    def tick(self, frame: int, is_training: bool):
        self.config.clone_sigma_scale = self.app.train_config.sigma_scale
        if not is_training:
            return
        if self.config.mode == ADCMode.DISABLED:
            return
        if frame < self.config.start_frame:
            return

        # Update threshold BEFORE triggering so worker uses correct value
        n = self.app.renderer.gaussian_count
        if n < 10000:
            self.config.prune_weight_min = 0.005
        elif n < 12000:
            self.config.prune_weight_min = 0.008
        elif n < 15000:
            self.config.prune_weight_min = 0.010
        else:
            self.config.prune_weight_min = 0.012

        frames_since_full = frame % self.config.full_adc_every
        do_full           = frames_since_full == 0
        halfway           = self.config.full_adc_every // 2
        prune_window_open = frames_since_full >= halfway
        prune_only        = (prune_window_open and
                            frames_since_full % self.config.prune_only_every == 0 and
                            not do_full)

        if do_full:
            self._trigger(do_full=True)
        elif prune_only:
            self._trigger(do_full=False)

    def apply_pending(self):
        """
        Zero cost when nothing is ready.
        """
        if self.config.mode == ADCMode.DISABLED:
            return

        with self._lock:
            if self._pending_params is None:
                return
            params, surviving_indices = self._pending_params
            self._pending_params = None

        self.app.apply_densification(params, surviving_indices)

        self.gaussian_count_history.append(len(params))
        if len(self.gaussian_count_history) > 200:
            self.gaussian_count_history.pop(0)

    def _trigger(self, do_full: bool):
        backend = self._backends.get(self.config.mode)

        if backend is None or self.config.mode == ADCMode.DISABLED:
            return

        if self.config.mode == ADCMode.GPU:
            print("[ADC] GPU backend not yet implemented, falling back to CPU")
            backend = self._backends[ADCMode.CPU]

        # CPU path: async via thread
        if self._running:
            print("[ADC] Previous pass still running, skipping this cycle")
            return

        snapshots = self._take_snapshots()
        if snapshots is None:
            return

        params_snap, grads_snap, pred_snap, ref_snap = snapshots

        self._running = True
        self.last_adc_type = "Full" if do_full else "Prune-only"

        self._thread = threading.Thread(
            target=self._worker,
            args=(params_snap, grads_snap, pred_snap, ref_snap, do_full),
            daemon=True
        )
        self._thread.start()
        print(f"[ADC] {self.last_adc_type} pass started "
              f"({len(params_snap)} Gaussians in)")

    def _take_snapshots(self):
        """
        GPU readbacks — must run on the main thread.
        Returns (params, grads, pred, ref) numpy snapshots or None on error.
        """
        try:
            r = self.app.renderer

            params_snap = (r.gaussian_buffer
                           .to_numpy()
                           .view(np.float32)
                           .reshape(-1, PARAMS_PER_GAUSSIAN)
                           .copy())

            grads_snap = (r.grad_buffer
                          .to_numpy()
                          .view(np.float32)
                          .reshape(-1, PARAMS_PER_GAUSSIAN)
                          .copy())

            # Fresh rasterize so prediction matches current Gaussian state
            cmd = self.app.device.create_command_encoder()
            r.rasterize_gaussians(cmd, self.app.vol_min_world, self.app.vol_max_world)
            self.app.device.submit_command_buffer(cmd.finish())
            pred_snap = r.gaussian_volume_tex.to_numpy().copy()

            # Reference never changes — cache it after first read
            if self._ref_cache is None:
                self._ref_cache = r.volume_tex.to_numpy().copy()
                print("[ADC] Reference volume cached")

            return params_snap, grads_snap, pred_snap, self._ref_cache

        except Exception as e:
            print(f"[ADC] Snapshot failed: {e}")
            traceback.print_exc()
            self._running = False
            return None

    def _worker(self, params, grads, pred, ref, do_full):
        try:
            backend = self._backends[self.config.mode]
            if self.config.mode == ADCMode.GPU:
                backend = self._backends[ADCMode.CPU]

            result_params, surviving_indices = backend.run(
                params, grads, pred, ref,
                self.config,
                self.app.vol_min_world,
                self.app.vol_max_world,
                do_full
            )

            self.gaussian_count_history.append(len(result_params))
            if len(self.gaussian_count_history) > 200:
                self.gaussian_count_history.pop(0)

            with self._lock:
                self._pending_params = (result_params, surviving_indices)

        except Exception as e:
            print(f"[ADC] Worker error: {e}")
            traceback.print_exc()
        finally:
            self._running = False

    def draw_ui(self):
        """
        Self-contained imgui panel.
        Call from App.draw_ui() as: self.adc.draw_ui()
        Requires imgui_bundle to be imported in the calling module.
        """
        from imgui_bundle import imgui

        if not imgui.begin("ADC - Adaptive Density Control"):
            imgui.end()
            return

        # Mode selector
        modes = ["Disabled", "CPU (Async)", "GPU (Stub)"]
        changed, self.config.mode = imgui.combo("Backend", self.config.mode, modes)
        if self.config.mode == ADCMode.GPU:
            imgui.text_colored(imgui.ImVec4(1, 0.5, 0, 1),
                               "GPU ADC not yet implemented — falls back to CPU")

        imgui.separator()

        # Status
        if self._running:
            imgui.text_colored(imgui.ImVec4(1, 1, 0, 1),
                               f"Running: {self.last_adc_type}...")
        elif self.config.mode == ADCMode.DISABLED:
            imgui.text_colored(imgui.ImVec4(0.5, 0.5, 0.5, 1), "Disabled")
        else:
            imgui.text_colored(imgui.ImVec4(0.4, 1, 0.4, 1), "Idle")

        n_current = self.app.renderer.gaussian_count
        imgui.text(f"Gaussians: {n_current:,} / {self.config.max_gaussians:,}")

        # Progress bar toward cap
        fill = float(n_current) / float(self.config.max_gaussians)
        imgui.progress_bar(fill, imgui.ImVec2(-1, 0),
                           f"{n_current:,} / {self.config.max_gaussians:,}")

        # History plot
        if len(self.gaussian_count_history) > 1:
            arr = np.array(self.gaussian_count_history, dtype=np.float32)
            imgui.plot_lines(
                "##gauss_history", arr,
                scale_min=0.0,
                scale_max=float(self.config.max_gaussians),
                graph_size=imgui.ImVec2(300, 50)
            )

        imgui.separator()

        # Scheduling
        imgui.text("Scheduling")
        _, self.config.start_frame = imgui.slider_int(
            "Start Frame", self.config.start_frame, 0, 2000)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Don't run ADC until Adam has settled this many frames")

        _, self.config.full_adc_every = imgui.slider_int(
            "Full ADC Every N Frames", self.config.full_adc_every, 50, 1000)
        _, self.config.prune_only_every = imgui.slider_int(
            "Prune Only Every N Frames", self.config.prune_only_every, 20, 500)
        _, self.config.max_gaussians = imgui.slider_int(
            "Max Gaussians", self.config.max_gaussians, 1000, 200000)

        imgui.separator()

        # Prune
        imgui.text("Pruning")
        _, self.config.prune_weight_min = imgui.slider_float(
            "Min Weight Threshold", self.config.prune_weight_min,
            0.0001, 0.05, format="%.4f")
        if imgui.is_item_hovered():
            imgui.set_tooltip("Gaussians with weight below this are removed")

        _, self.config.prune_scale_max = imgui.slider_float(
            "Max Scale (fraction of extent)", self.config.prune_scale_max,
            0.05, 1.0, format="%.2f")
        if imgui.is_item_hovered():
            imgui.set_tooltip("Gaussians larger than this fraction of the volume extent are removed")

        imgui.separator()

        # Split
        imgui.text("Split")
        _, self.config.n_split = imgui.slider_int(
            "Split Count", self.config.n_split, 0, 200)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Max Gaussians to split per ADC pass")

        _, self.config.split_scale_thresh = imgui.slider_float(
            "Split Size Threshold", self.config.split_scale_thresh,
            0.01, 0.2, format="%.3f")
        if imgui.is_item_hovered():
            imgui.set_tooltip("Min scale (fraction of extent) to be a split candidate")

        _, self.config.split_grad_percentile = imgui.slider_int(
            "Grad Percentile Threshold", self.config.split_grad_percentile,
            50, 99)
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Only split Gaussians whose gradient magnitude exceeds this percentile")

        imgui.separator()

        # Clone
        imgui.text("Clone / Spawn")
        _, self.config.n_clone = imgui.slider_int(
            "Clone Count", self.config.n_clone, 0, 500)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Max new Gaussians to spawn per full ADC pass")

        _, self.config.substance_thresh = imgui.slider_float(
            "Substance Threshold", self.config.substance_thresh,
            0.001, 0.2, format="%.3f")
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Min reference density at a voxel to be a valid spawn site.\n"
                "Prevents spawning in empty boundary regions.")

        imgui.separator()

        # Manual triggers
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
    
class TrainingConfig:
    """Configurable training hyperparameters"""
    def __init__(self):
        # SGD Learning Rates
        self.learning_rate_pos = 0.1
        self.learning_rate_scale = 0.01
        self.learning_rate_rotation = 0.001
        self.learning_rate_weight = 0.1
        
        # Adam Hyperparameters
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_epsilon = 1e-8
        self.use_adam = True  # Toggle between SGD and Adam
        
        # Gaussian Generation
        self.probability_scale = 0.02
        self.sigma_scale = 2.0
        self.jitter_scale = 5.0

# ==========================================
# RENDERER SYSTEM
# ==========================================

class Renderer:
    def __init__(self, device, volume_data):
        self.device = device
        self.pipeline = None
        self.last_mod_time = 0
        self.error_msg = ""
        
        self.linear_sampler = device.create_sampler(
            min_filter=spy.TextureFilteringMode.linear,
            mag_filter=spy.TextureFilteringMode.linear,
            mip_filter=spy.TextureFilteringMode.linear,
            address_u=spy.TextureAddressingMode.wrap, 
            address_v=spy.TextureAddressingMode.wrap,
            address_w=spy.TextureAddressingMode.wrap
        )

        self.volume_tex = device.create_texture(
            type=spy.TextureType.texture_3d, format=spy.Format.r32_float,
            width=VOL_SIZE, height=VOL_SIZE, depth=VOL_SIZE,
            usage=spy.TextureUsage.shader_resource, label="VDBVolume"
        )

        self.tile_res = (
            (VOL_SIZE + TILE_SIZE - 1) // TILE_SIZE,
            (VOL_SIZE + TILE_SIZE - 1) // TILE_SIZE,
            (VOL_SIZE + TILE_SIZE - 1) // TILE_SIZE
        )
        
        cmd = device.create_command_encoder()
        cmd.upload_texture_data(self.volume_tex, [volume_data])
        device.submit_command_buffer(cmd.finish())

        self.cam_buffer = device.create_buffer(size=64, usage=spy.BufferUsage.shader_resource, memory_type=spy.MemoryType.upload)
        self.settings_buffer = device.create_buffer(size=128, usage=spy.BufferUsage.shader_resource, memory_type=spy.MemoryType.upload)

        self.screen_tex = None
        self.display_gl_tex = None
        self.width, self.height = 0, 0

        self.gaussian_volume_tex = device.create_texture(
            type=spy.TextureType.texture_3d,
            format=spy.Format.r32_float,
            width=VOL_SIZE, height=VOL_SIZE, depth=VOL_SIZE,
            usage=spy.TextureUsage.unordered_access | spy.TextureUsage.shader_resource,
            label="GaussianVolume"
        )

        self._needs_rebinning = True
        self._needs_rasterization = True
        prog_raster_tiled = self.device.load_program("shaders/3drasterizer.slang", ["rasterize_tiled"])
        self.pipe_raster_tiled = self.device.create_compute_pipeline(program=prog_raster_tiled)

        self.use_gaussian_volume = False

        self.gaussian_buffer = None
        self.gaussian_count = 0
        self.adam_iteration = 1  # Track iteration for bias correction
        
        # Adam state buffers
        self.adam_first_moment = None
        self.adam_second_moment = None

        self._training_initialized = False

        self.check_hot_reload()

        # Debug
        self.debug_tex = device.create_texture(
            type=spy.TextureType.texture_3d,
            format=spy.Format.rgba8_unorm,
            width=VOL_SIZE, height=VOL_SIZE, depth=VOL_SIZE,
            usage=spy.TextureUsage.unordered_access | spy.TextureUsage.shader_resource,
            label="DebugVolume"
        )
        
        self.debug_mode = 0
        self.debug_scale = 1.0
        
        # Load debug raymarcher
        prog_debug = device.load_program("shaders/debug_raymarch.slang", ["vertex_main", "fragment_main"])
        self.debug_pipeline = device.create_render_pipeline(
            program=prog_debug,
            input_layout=device.create_input_layout(input_elements=[], vertex_streams=[]),
            targets=[{"format": spy.Format.rgba8_unorm}]
        )

        self.gradient_health = {
            'status': 'unknown',  # 'healthy', 'exploding', 'vanishing', 'dead'
            'mean': 0.0,
            'max': 0.0,
            'min': 0.0,
            'active_ratio': 0.0,
            'recommendation': ''
        }
        
        self.loss_history = []
        self.grad_mean_history = []
    
    def init_training(self):

        total_tiles = self.tile_res[0] * self.tile_res[1] * self.tile_res[2]
        
        self.grad_buffer = self.device.create_buffer(
            element_count=self.gaussian_count * PARAMS_PER_GAUSSIAN,
            struct_size=4,
            usage=spy.BufferUsage.unordered_access | spy.BufferUsage.shader_resource,
            memory_type=spy.MemoryType.device_local
        )
        
        self.tile_content = self.device.create_buffer(
            element_count=total_tiles * MAX_GAUSSIANS_PER_TILE,
            struct_size=4,
            usage=spy.BufferUsage.unordered_access | spy.BufferUsage.shader_resource,
            memory_type=spy.MemoryType.device_local
        )
        
        self.tile_counts = self.device.create_buffer(
            element_count=total_tiles, 
            struct_size=4,
            usage=spy.BufferUsage.unordered_access | spy.BufferUsage.shader_resource,
            memory_type=spy.MemoryType.device_local
        )

        self.adam_first_moment = self.device.create_buffer(
            element_count=self.gaussian_count * PARAMS_PER_GAUSSIAN,
            struct_size=4,
            usage=spy.BufferUsage.unordered_access | spy.BufferUsage.shader_resource,
            memory_type=spy.MemoryType.device_local
        )
        
        self.adam_second_moment = self.device.create_buffer(
            element_count=self.gaussian_count * PARAMS_PER_GAUSSIAN,
            struct_size=4,
            usage=spy.BufferUsage.unordered_access | spy.BufferUsage.shader_resource,
            memory_type=spy.MemoryType.device_local
        )

        if not self._training_initialized:
            print("Compiling Kernels...")
            
            prog_clear_grads = self.device.load_program("shaders/training.slang", ["clear_gradients"])
            self.pipe_clear_grads = self.device.create_compute_pipeline(program=prog_clear_grads)
            
            prog_clear_tiles = self.device.load_program("shaders/training.slang", ["clear_tiles"])
            self.pipe_clear_tiles = self.device.create_compute_pipeline(program=prog_clear_tiles)

            prog_bin = self.device.load_program("shaders/training.slang", ["bin_gaussians"])
            self.pipe_bin = self.device.create_compute_pipeline(program=prog_bin)

            prog_train = self.device.load_program("shaders/training.slang", ["train_main"])
            self.pipe_train = self.device.create_compute_pipeline(program=prog_train)

            prog_optim_sgd = self.device.load_program("shaders/training.slang", ["optimizer_sgd"])
            self.pipe_optim_sgd = self.device.create_compute_pipeline(program=prog_optim_sgd)
            
            prog_optim_adam = self.device.load_program("shaders/training.slang", ["optimizer_adam"])
            self.pipe_optim_adam = self.device.create_compute_pipeline(program=prog_optim_adam)
            
            prog_init_adam = self.device.load_program("shaders/training.slang", ["init_adam_state"])
            self.pipe_init_adam = self.device.create_compute_pipeline(program=prog_init_adam)

            prog_debug = self.device.load_program("shaders/training.slang", ["compute_debug"])
            self.pipe_debug_only = self.device.create_compute_pipeline(program=prog_debug)

            self._training_initialized = True

        self.debug_needs_update = True
        # Initialize Adam state to zeros
        self._init_adam_state()
    
    def _init_adam_state(self):
        """Initialize Adam momentum buffers to zero"""
        cmd = self.device.create_command_encoder()
        
        with cmd.begin_compute_pass() as cp:
            root_object = cp.bind_pipeline(self.pipe_init_adam)
            cursor = spy.ShaderCursor(root_object)
            cursor["gAdamFirstMoment"] = self.adam_first_moment
            cursor["gAdamSecondMoment"] = self.adam_second_moment
            cursor["TrainParams"]["gaussianCount"] = self.gaussian_count
            
            threads = self.gaussian_count * PARAMS_PER_GAUSSIAN
            cp.dispatch(thread_count=(threads, 1, 1))
        
        self.device.submit_command_buffer(cmd.finish())


    def analyze_gradients(self):
        """Analyze gradient health - CALIBRATED FOR VOLUMETRIC FITTING. This function are just suggestions for me"""
        if not hasattr(self, 'grad_buffer') or self.grad_buffer is None:
            return
        
        grad_bytes = self.grad_buffer.to_numpy()
        grads = grad_bytes.view(dtype=np.float32)
        
        grad_abs = np.abs(grads)
        nonzero_grads = grad_abs[grad_abs > 1e-10]
        
        if len(nonzero_grads) == 0:
            self.gradient_health = {
                'status': 'dead',
                'mean': 0.0,
                'max': 0.0,
                'min': 0.0,
                'active_ratio': 0.0,
                'recommendation': 'No gradients! Check if training is running.'
            }
            return
        
        mean_grad = np.mean(nonzero_grads)
        max_grad = np.max(nonzero_grads)
        min_grad = np.min(nonzero_grads)
        active_ratio = len(nonzero_grads) / len(grads)
        
        self.grad_mean_history.append(mean_grad)
        if len(self.grad_mean_history) > 100:
            self.grad_mean_history.pop(0)
        
        status = 'healthy'
        recommendation = 'Training normally. Gradients reflect prediction error.'
        
        # Check for TRUE exploding gradients (much higher thresholds!)
        if max_grad > 50.0: 
            status = 'exploding'
            recommendation = 'CRITICAL: Gradients exploding! Reduce learning rates by 10×'
        elif max_grad > 20.0:
            status = 'high'
            recommendation = 'Gradients high but manageable. Consider reducing LRs by 2× if loss oscillates.'
        
        # Early training (high error = high gradients)
        elif mean_grad > 1.0:
            status = 'early'
            recommendation = 'Early training - high gradients are normal as error is large.'
        
        # Check for vanishing gradients
        elif mean_grad < 1e-5:
            status = 'vanishing'
            if mean_grad < 1e-7:
                recommendation = 'CRITICAL: Gradients vanishing! Increase learning rates by 10×'
            else:
                recommendation = 'Gradients getting small. May need higher learning rates or have converged.'
        
        # Check for convergence
        elif mean_grad < 0.001 and len(self.grad_mean_history) > 20:
            recent = self.grad_mean_history[-20:]
            std = np.std(recent)
            if std < mean_grad * 0.1:
                status = 'converged'
                recommendation = 'Gradients stable and small. Training likely converged!'
        
        # Check trend (is it improving?)
        if status == 'healthy' and len(self.grad_mean_history) > 20:
            recent = self.grad_mean_history[-20:]
            older = self.grad_mean_history[-40:-20] if len(self.grad_mean_history) >= 40 else self.grad_mean_history[:-20]
            
            if len(older) > 0:
                recent_mean = np.mean(recent)
                older_mean = np.mean(older)
                
                if recent_mean < older_mean * 0.8:
                    status = 'improving'
                    recommendation = 'Gradients decreasing - loss is improving! Keep current settings.'
                elif recent_mean > older_mean * 1.2:
                    status = 'diverging'
                    recommendation = 'Gradients increasing - may be diverging. Consider reducing learning rates.'
        
        # Check for sparse activation
        if active_ratio < 0.3:
            status = 'sparse'
            recommendation = f'Only {active_ratio*100:.1f}% Gaussians active. Increase sigma_scale or add more Gaussians.'
        
        self.gradient_health = {
            'status': status,
            'mean': mean_grad,
            'max': max_grad,
            'min': min_grad,
            'active_ratio': active_ratio,
            'recommendation': recommendation
        }

    def update_debug_visualization(self, cmd, vol_min, vol_max):
        """
        Update debug visualization WITHOUT running training.
        Call this when debug mode changes or user wants to refresh.
        """
        if self.debug_mode == 0:
            return
        
        # Just run the debug computation kernel (no training)
        with cmd.begin_compute_pass() as cp:
            root_object = cp.bind_pipeline(self.pipe_debug_only)  # New pipeline!
            cursor = spy.ShaderCursor(root_object)
            
            cursor["ReferenceVol"] = self.volume_tex 
            cursor["gGaussianParamsRaw"] = self.gaussian_buffer
            cursor["gGradientsRaw"] = self.grad_buffer
            cursor["gTileContent"] = self.tile_content
            cursor["gTileCounts"] = self.tile_counts
            cursor["gDebugVolume"] = self.debug_tex
            
            cursor["TrainParams"]["volumeResolution"] = (VOL_SIZE, VOL_SIZE, VOL_SIZE)
            cursor["TrainParams"]["tileResolution"] = self.tile_res
            cursor["TrainParams"]["minWorld"] = tuple(vol_min)
            cursor["TrainParams"]["maxWorld"] = tuple(vol_max)
            cursor["TrainParams"]["tileSize"] = TILE_SIZE
            cursor["TrainParams"]["debugMode"] = self.debug_mode
            cursor["TrainParams"]["debugScale"] = self.debug_scale
            cursor["TrainParams"]["gaussianCount"] = self.gaussian_count
            
            cp.dispatch(thread_count=(VOL_SIZE, VOL_SIZE, VOL_SIZE))
        
        self.debug_needs_update = False

    def run_training_step(self, vol_min, vol_max, train_config):
        cmd = self.device.create_command_encoder()
        
        total_tiles = self.tile_res[0] * self.tile_res[1] * self.tile_res[2]

        # 1. Clear Gradients
        with cmd.begin_compute_pass() as cp:
            root_object = cp.bind_pipeline(self.pipe_clear_grads)
            cursor = spy.ShaderCursor(root_object)
            cursor["gGradientsRaw"] = self.grad_buffer
            cursor["TrainParams"]["gaussianCount"] = self.gaussian_count
            cursor["TrainParams"]["tileSize"] = TILE_SIZE
            
            threads = self.gaussian_count * PARAMS_PER_GAUSSIAN
            cp.dispatch(thread_count=(threads, 1, 1))
        
        if self._needs_rebinning:
            # 2. Clear Tiles
            with cmd.begin_compute_pass() as cp:
                root_object = cp.bind_pipeline(self.pipe_clear_tiles)
                cursor = spy.ShaderCursor(root_object)
                cursor["gTileCounts"] = self.tile_counts
                cursor["TrainParams"]["tileResolution"] = self.tile_res
                cursor["TrainParams"]["totalTiles"] = total_tiles
                cursor["TrainParams"]["tileSize"] = TILE_SIZE
                cp.dispatch(thread_count=(total_tiles, 1, 1))

            # 3. Bin Gaussians
            with cmd.begin_compute_pass() as cp:
                root_object = cp.bind_pipeline(self.pipe_bin)
                cursor = spy.ShaderCursor(root_object)
                
                cursor["gGaussianParamsRaw"] = self.gaussian_buffer
                cursor["gTileContent"] = self.tile_content
                cursor["gTileCounts"] = self.tile_counts
                cursor["TrainParams"]["volumeResolution"] = (VOL_SIZE, VOL_SIZE, VOL_SIZE)
                cursor["TrainParams"]["tileResolution"] = self.tile_res
                cursor["TrainParams"]["gaussianCount"] = self.gaussian_count
                cursor["TrainParams"]["minWorld"] = tuple(vol_min)
                cursor["TrainParams"]["maxWorld"] = tuple(vol_max)
                cursor["TrainParams"]["tileSize"] = TILE_SIZE
                cp.dispatch(thread_count=(self.gaussian_count, 1, 1))
            self._needs_rebinning = False

        # 4. Train
        with cmd.begin_compute_pass() as cp:
            root_object = cp.bind_pipeline(self.pipe_train)
            cursor = spy.ShaderCursor(root_object)
            
            cursor["ReferenceVol"] = self.volume_tex 
            cursor["gGaussianParamsRaw"] = self.gaussian_buffer
            cursor["gGradientsRaw"] = self.grad_buffer
            cursor["gTileContent"] = self.tile_content
            cursor["gTileCounts"] = self.tile_counts

            cursor["TrainParams"]["volumeResolution"] = (VOL_SIZE, VOL_SIZE, VOL_SIZE)
            cursor["TrainParams"]["tileResolution"] = self.tile_res
            cursor["TrainParams"]["minWorld"] = tuple(vol_min)
            cursor["TrainParams"]["maxWorld"] = tuple(vol_max)
            cursor["TrainParams"]["tileSize"] = TILE_SIZE

            cp.dispatch(thread_count=(VOL_SIZE, VOL_SIZE, VOL_SIZE))

        # 5. Optimize
        pipeline = self.pipe_optim_adam if train_config.use_adam else self.pipe_optim_sgd
        
        with cmd.begin_compute_pass() as cp:
            root_object = cp.bind_pipeline(pipeline)
            cursor = spy.ShaderCursor(root_object)
            
            cursor["gGaussianParamsRaw"] = self.gaussian_buffer
            cursor["gGradientsRaw"] = self.grad_buffer
            cursor["TrainParams"]["gaussianCount"] = self.gaussian_count
            
            # Pack learning rates as array
            cursor["TrainParams"]["learningRates"] = [
                train_config.learning_rate_pos,
                train_config.learning_rate_scale,
                train_config.learning_rate_rotation,
                train_config.learning_rate_weight
            ]
            
            cursor["TrainParams"]["tileSize"] = TILE_SIZE
            
            if train_config.use_adam:
                cursor["gAdamFirstMoment"] = self.adam_first_moment
                cursor["gAdamSecondMoment"] = self.adam_second_moment
                cursor["TrainParams"]["adamBeta1"] = train_config.adam_beta1
                cursor["TrainParams"]["adamBeta2"] = train_config.adam_beta2
                cursor["TrainParams"]["adamEpsilon"] = train_config.adam_epsilon
                cursor["TrainParams"]["adamIteration"] = self.adam_iteration
            
            cp.dispatch(thread_count=(self.gaussian_count, 1, 1))

        if self.use_gaussian_volume:
            self.rasterize_gaussians(cmd, vol_min, vol_max)

        # Pass 7: Debug visualization
        if self.debug_needs_update and self.debug_mode > 0:
            self.update_debug_visualization(cmd, vol_min, vol_max)

        self.device.submit_command_buffer(cmd.finish())
        self._needs_rebinning = True
        self._needs_rasterization = True
        self.debug_needs_update = True

        if train_config.use_adam:
            self.adam_iteration += 1

    def resize(self, w, h):
        if w == self.width and h == self.height: 
            return
        self.width, self.height = w, h
        
        self.screen_tex = self.device.create_texture(
            format=spy.Format.rgba8_unorm,  # was rgba32_float
            width=w, height=h,
            usage=spy.TextureUsage.render_target, label="Screen"
        )
        
        if self.display_gl_tex is None: 
            self.display_gl_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.display_gl_tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)

    def check_hot_reload(self):
        try:
            curr_time = Path(SHADER_FILE).stat().st_mtime
            if curr_time > self.last_mod_time:
                self.last_mod_time = curr_time
                
                prog = self.device.load_program(SHADER_FILE, ["vertex_main", "fragment_main"])
                new_pipe = self.device.create_render_pipeline(
                    program=prog,
                    input_layout=self.device.create_input_layout(input_elements=[], vertex_streams=[]),
                    targets=[{"format": spy.Format.rgba8_unorm}]
                )
                
                self.pipeline = new_pipe
                self.error_msg = ""
                print("Shader Reloaded!")
        except Exception as e:
            self.error_msg = str(e)
            print("Shader Compile Error (Safe)")
    
    def render(self, camera, settings):
        if self.debug_mode > 0:
            self.render_debug(camera)
        else:
            self.render_main(camera, settings)

    def render_debug(self, camera):
        """Debug visualization using raymarcher"""
        self.cam_buffer.copy_from_numpy(camera.get_gpu_data(self.width / self.height))
        
        cmd = self.device.create_command_encoder()
        cmd.set_texture_state(self.debug_tex, spy.ResourceState.shader_resource)

        with cmd.begin_render_pass({"color_attachments": [{"view": self.screen_tex.create_view({})}]}) as rp:
            rp.bind_pipeline(self.debug_pipeline)
            cursor = spy.ShaderCursor(rp.bind_pipeline(self.debug_pipeline))
            
            cursor["debugVolume"] = self.debug_tex
            cursor["camera"] = self.cam_buffer
            cursor["linearSampler"] = self.linear_sampler
            
            rp.set_render_state({
                "viewports": [spy.Viewport.from_size(self.width, self.height)],
                "scissor_rects": [spy.ScissorRect.from_size(self.width, self.height)]
            })
            rp.draw({"vertex_count": 3})

        self.device.submit_command_buffer(cmd.finish())

    def render_main(self, camera, settings):
        if not self.pipeline: 
            return

        self.cam_buffer.copy_from_numpy(camera.get_gpu_data(self.width / self.height))
        
        s_data = np.zeros(24, dtype=np.float32)
        
        s_data[0] = settings.step_size
        s_data[1] = settings.density_scale
        s_data[2] = settings.density_curve
        s_data[3] = float(settings.step_count)
        s_data[4:7] = settings.smoke_color
        
        s_data[7] = settings.light_penetration
        s_data[8] = settings.phase_g
        
        s_data[9:12] = settings.get_sun_dir()
        
        s_data[12] = settings.sun_color_base[0] * settings.sun_intensity
        s_data[13] = settings.sun_color_base[1] * settings.sun_intensity
        s_data[14] = settings.sun_color_base[2] * settings.sun_intensity
        
        s_data[15] = settings.ambient_color_base[0] * settings.ambient_intensity
        s_data[16] = settings.ambient_color_base[1] * settings.ambient_intensity
        s_data[17] = settings.ambient_color_base[2] * settings.ambient_intensity
        
        s_data[18] = float(settings.shadow_steps)
        s_data[19] = settings.shadow_step_mult

        self.settings_buffer.copy_from_numpy(s_data)

        cmd = self.device.create_command_encoder()
        cmd.set_texture_state(self.volume_tex, spy.ResourceState.shader_resource)

        with cmd.begin_render_pass({"color_attachments": [{"view": self.screen_tex.create_view({})}]}) as rp:
            rp.bind_pipeline(self.pipeline)
            cursor = spy.ShaderCursor(rp.bind_pipeline(self.pipeline))
            
            cursor["inVolume"] = self.gaussian_volume_tex if self.use_gaussian_volume else self.volume_tex
            cursor["camera"] = self.cam_buffer
            cursor["linearSampler"] = self.linear_sampler
            cursor["settings"] = self.settings_buffer
            
            rp.set_render_state({
                "viewports": [spy.Viewport.from_size(self.width, self.height)],
                "scissor_rects": [spy.ScissorRect.from_size(self.width, self.height)]
            })
            rp.draw({"vertex_count": 3})

        self.device.submit_command_buffer(cmd.finish())

    def update_display(self):
        pixels = self.screen_tex.to_numpy()  # already RGBA8, no clip/cast needed
        glBindTexture(GL_TEXTURE_2D, self.display_gl_tex)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.width, self.height, 
                        GL_RGBA, GL_UNSIGNED_BYTE, pixels)
        
        fb = glGenFramebuffers(1)
        glBindFramebuffer(GL_READ_FRAMEBUFFER, fb)
        glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, 
                            GL_TEXTURE_2D, self.display_gl_tex, 0)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
        glBlitFramebuffer(0, 0, self.width, self.height, 
                        0, 0, self.width, self.height, 
                        GL_COLOR_BUFFER_BIT, GL_NEAREST)
        glDeleteFramebuffers(1, [fb])
    
    def rasterize_gaussians(self, cmd, vol_min, vol_max):

        cmd.clear_texture_float(self.gaussian_volume_tex)
        cmd.set_texture_state(self.gaussian_volume_tex, spy.ResourceState.unordered_access)

        total_tiles = self.tile_res[0] * self.tile_res[1] * self.tile_res[2]
        if self._needs_rebinning:
            # 2. Clear Tiles
            with cmd.begin_compute_pass() as cp:
                root_object = cp.bind_pipeline(self.pipe_clear_tiles)
                cursor = spy.ShaderCursor(root_object)
                cursor["gTileCounts"] = self.tile_counts
                cursor["TrainParams"]["tileResolution"] = self.tile_res
                cursor["TrainParams"]["totalTiles"] = total_tiles
                cursor["TrainParams"]["tileSize"] = TILE_SIZE
                cp.dispatch(thread_count=(total_tiles, 1, 1))

            # 3. Bin Gaussians
            with cmd.begin_compute_pass() as cp:
                root_object = cp.bind_pipeline(self.pipe_bin)
                cursor = spy.ShaderCursor(root_object)
                
                cursor["gGaussianParamsRaw"] = self.gaussian_buffer
                cursor["gTileContent"] = self.tile_content
                cursor["gTileCounts"] = self.tile_counts
                cursor["TrainParams"]["volumeResolution"] = (VOL_SIZE, VOL_SIZE, VOL_SIZE)
                cursor["TrainParams"]["tileResolution"] = self.tile_res
                cursor["TrainParams"]["gaussianCount"] = self.gaussian_count
                cursor["TrainParams"]["minWorld"] = tuple(vol_min)
                cursor["TrainParams"]["maxWorld"] = tuple(vol_max)
                cursor["TrainParams"]["tileSize"] = TILE_SIZE
                cp.dispatch(thread_count=(self.gaussian_count, 1, 1))
            self._needs_rebinning = False

        with cmd.begin_compute_pass() as cp:
            root_object = cp.bind_pipeline(self.pipe_raster_tiled)
            cursor = spy.ShaderCursor(root_object)
            
            cursor["gGaussians"] = self.gaussian_buffer
            cursor["gGaussianVolume"] = self.gaussian_volume_tex
            cursor["gTileContent"] = self.tile_content
            cursor["gTileCounts"] = self.tile_counts
            cursor["RasterParams"]["volumeResolution"] = (VOL_SIZE, VOL_SIZE, VOL_SIZE)
            cursor["RasterParams"]["tileResolution"] = self.tile_res
            cursor["RasterParams"]["tileSize"] = TILE_SIZE
            cursor["RasterParams"]["volumeMinWorld"] = tuple(vol_min)
            cursor["RasterParams"]["volumeMaxWorld"] = tuple(vol_max)
            
            cp.dispatch(thread_count=(VOL_SIZE, VOL_SIZE, VOL_SIZE))

        self._needs_rasterization = False


# ==========================================
# APP CLASS
# ==========================================

class App:
    def __init__(self):
        if not glfw.init(): 
            raise Exception("GLFW failed")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        
        self.width, self.height = WINDOW_WIDTH, WINDOW_HEIGHT
        self.window = glfw.create_window(self.width, self.height, WINDOW_TITLE, None, None)
        glfw.make_context_current(self.window)

        imgui.create_context()
        self.io = imgui.get_io()
        self.io.config_flags |= imgui.ConfigFlags_.docking_enable
        window_address = ctypes.cast(self.window, ctypes.c_void_p).value
        imgui.backends.glfw_init_for_opengl(window_address, True)
        imgui.backends.opengl3_init("#version 330")

        self.is_training = False
        self.settings = Settings()
        self.camera = Camera(self.width, self.height)
        self.train_config = TrainingConfig()
        
        example_dir = Path(__file__).parent
        self.device = spy.Device(enable_debug_layers=True, compiler_options={"include_paths": [example_dir]}, type=spy.DeviceType.vulkan)
        
        self.grid = load_vdb_grid(VDB_FILE)
        self.vol_min_world, self.vol_max_world, vol_data = convert_grid_to_dense_volume(self.grid, VOL_SIZE)
        self.gaussians = convert_grid_to_gaussians(self.grid, self.train_config)

        self.renderer = Renderer(self.device, vol_data)
        self.renderer.resize(self.width, self.height)
        self.renderer.gaussian_count = len(self.gaussians)
        self.renderer.gaussian_buffer = self.device.create_buffer(
                element_count=self.renderer.gaussian_count,
                struct_size=PARAMS_PER_GAUSSIAN*4,
                usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
                memory_type=spy.MemoryType.device_local,
                data=self.gaussians 
            )
        self.renderer.init_training()
        
        cmd = self.device.create_command_encoder()
        self.renderer.rasterize_gaussians(cmd, self.vol_min_world, self.vol_max_world)
        self.device.submit_command_buffer(cmd.finish())

        self.adc = ADCController(self, ADCConfig())
        self.adc.train_config = self.train_config

        self.ply_lighting = CloudLightingConfig()

    
    def apply_densification(self, new_params, surviving_indices=None):
        new_params = np.ascontiguousarray(new_params, dtype=np.float32)
        if new_params.ndim == 2:
            new_params = new_params.reshape(-1, PARAMS_PER_GAUSSIAN)
        flat_params = new_params.flatten()

        if len(new_params) == 0:
            print("[ADC] Warning: 0 Gaussians, skipping apply")
            return

        saved_iteration = self.renderer.adam_iteration

        if surviving_indices is not None:
            try:
                old_m1 = (self.renderer.adam_first_moment
                        .to_numpy().view(np.float32)
                        .reshape(-1, PARAMS_PER_GAUSSIAN).copy())
                old_m2 = (self.renderer.adam_second_moment
                        .to_numpy().view(np.float32)
                        .reshape(-1, PARAMS_PER_GAUSSIAN).copy())
            except Exception as e:
                print(f"[ADC] Could not read momentum buffers: {e}")
                old_m1 = old_m2 = None
        else:
            old_m1 = old_m2 = None

        self.gaussians                = new_params
        self.renderer.gaussian_count  = len(new_params)
        self.renderer.gaussian_buffer = self.device.create_buffer(
            element_count=len(new_params),
            struct_size=PARAMS_PER_GAUSSIAN * 4,
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            memory_type=spy.MemoryType.device_local,
            data=flat_params
        )
        self.renderer.init_training()
        self.renderer.adam_iteration = saved_iteration  # restore — Regenerate button overrides this to 1

        if surviving_indices is not None and old_m1 is not None:
            try:
                n_new  = len(new_params)
                new_m1 = np.zeros((n_new, PARAMS_PER_GAUSSIAN), dtype=np.float32)
                new_m2 = np.zeros((n_new, PARAMS_PER_GAUSSIAN), dtype=np.float32)
                valid  = surviving_indices[surviving_indices < len(old_m1)]
                new_m1[:len(valid)] = old_m1[valid]
                new_m2[:len(valid)] = old_m2[valid]
                self.renderer.adam_first_moment.copy_from_numpy(new_m1.flatten())
                self.renderer.adam_second_moment.copy_from_numpy(new_m2.flatten())
                print(f"[ADC] Momentum restored for {len(valid)} survivors")
            except Exception as e:
                print(f"[ADC] Momentum restore failed (non-fatal): {e}")

        self.renderer._needs_rebinning     = True
        self.renderer._needs_rasterization = True
        print(f"[ADC] Buffer rebuilt: {len(new_params):,} Gaussians (Adam iter {saved_iteration})")


    def run(self):
        last_time = glfw.get_time()
        frame_count = 0
        
        while not glfw.window_should_close(self.window):
            curr_time = glfw.get_time()
            dt = curr_time - last_time
            last_time = curr_time
            
            glfw.poll_events()
            
            w, h = glfw.get_window_size(self.window)
            if w == 0 or h == 0: 
                continue
            self.renderer.resize(w, h)

            self.renderer.check_hot_reload()

            imgui.backends.opengl3_new_frame()
            imgui.backends.glfw_new_frame()
            imgui.new_frame()
            imgui.dock_space_over_viewport(0, imgui.get_main_viewport(), imgui.DockNodeFlags_.passthru_central_node)

            self.draw_ui(dt)
            self.handle_input(dt)

            if self.is_training:
                self.renderer.run_training_step(self.vol_min_world, self.vol_max_world, self.train_config)

                if frame_count % 100 == 0:
                    tile_debug = self.renderer.tile_counts.to_numpy().view(dtype=np.uint32)
                    total_refs = np.sum(tile_debug)
                    
                    grad_bytes = self.renderer.grad_buffer.to_numpy()
                    grad_all   = grad_bytes.view(dtype=np.float32).reshape(-1, PARAMS_PER_GAUSSIAN)
                    
                    # Check weight gradients across ALL Gaussians (param index 10)
                    weight_grads = grad_all[:, 10]
                    grad_max     = np.max(np.abs(grad_all))
                    nonzero      = np.count_nonzero(weight_grads)
                    
                    if total_refs == 0:
                        print(f"[CRITICAL] Tiler is empty! (Total Refs: {total_refs})")
                    elif grad_max == 0:
                        print(f"[ALERT] Tiler works ({total_refs} refs) but ALL gradients are ZERO.")
                    else:
                        print(f"[OK] Training Running. Refs: {total_refs}, "
                            f"GradMax: {grad_max:.6f}, "
                            f"ActiveWeightGrads: {nonzero}/{len(weight_grads)}")
                    maxed_tiles = np.sum(tile_debug >= MAX_GAUSSIANS_PER_TILE)
                    if maxed_tiles > 0:
                        print(f"[WARN] {maxed_tiles} tiles at MAX_GAUSSIANS_PER_TILE cap!")
                
                                
                # Track loss every 50 frames
                if frame_count % 50 == 0:
                    self.renderer.analyze_gradients()
                    loss = self.compute_current_loss()
                    self.renderer.loss_history.append(loss)
                    if len(self.renderer.loss_history) > 200:
                        self.renderer.loss_history.pop(0)
            
            self.adc.tick(frame_count, self.is_training)
            self.adc.apply_pending()
                
            
            if not self.is_training:
                needs_compute = (
                    (self.renderer.debug_needs_update and self.renderer.debug_mode > 0) or
                    (self.renderer._needs_rasterization and self.renderer.use_gaussian_volume)
                )
                if needs_compute:
                    cmd = self.device.create_command_encoder()
                    if self.renderer._needs_rasterization and self.renderer.use_gaussian_volume:
                        self.renderer.rasterize_gaussians(cmd, self.vol_min_world, self.vol_max_world)
                    if self.renderer.debug_needs_update and self.renderer.debug_mode > 0:
                        self.renderer.update_debug_visualization(cmd, self.vol_min_world, self.vol_max_world)
                    self.device.submit_command_buffer(cmd.finish())
                
            self.renderer.render(self.camera, self.settings)
            self.renderer.update_display()

            frame_count += 1

            imgui.render()
            imgui.backends.opengl3_render_draw_data(imgui.get_draw_data())
            glfw.swap_buffers(self.window)

        self.cleanup()

    def handle_input(self, dt):
        if self.io.want_capture_mouse: 
            return
        
        right_down = glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        if right_down and not self.camera.is_dragging:
            self.camera.is_dragging = True
            self.camera.first_mouse = True
            glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_DISABLED)
        elif not right_down and self.camera.is_dragging:
            self.camera.is_dragging = False
            glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_NORMAL)
            
        if self.camera.is_dragging:
            x, y = glfw.get_cursor_pos(self.window)
            self.camera.process_mouse(x, y)

        speed = self.camera.speed * dt
        if glfw.get_key(self.window, glfw.KEY_W) == glfw.PRESS: self.camera.pos += self.camera.front * speed
        if glfw.get_key(self.window, glfw.KEY_S) == glfw.PRESS: self.camera.pos -= self.camera.front * speed
        if glfw.get_key(self.window, glfw.KEY_A) == glfw.PRESS: self.camera.pos -= self.camera.right * speed
        if glfw.get_key(self.window, glfw.KEY_D) == glfw.PRESS: self.camera.pos += self.camera.right * speed
        if glfw.get_key(self.window, glfw.KEY_Q) == glfw.PRESS: self.camera.pos += self.camera.up * speed
        if glfw.get_key(self.window, glfw.KEY_E) == glfw.PRESS: self.camera.pos -= self.camera.up * speed
        if glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS: glfw.set_window_should_close(self.window, True)

    def draw_ui(self, dt):
        if imgui.begin("Stats"):
            imgui.text(f"FPS: {1.0/(dt+0.0001):.1f}")
            if self.renderer.error_msg:
                imgui.text_colored(imgui.ImVec4(1, 0, 0, 1), "SHADER ERROR")
        imgui.end()

        if imgui.begin("Rendering"):
            if self.renderer.error_msg:
                imgui.text_colored(imgui.ImVec4(1, 0, 0, 1), f"{self.renderer.error_msg}")
            else:
                imgui.text_colored(imgui.ImVec4(0, 1, 0, 1), "Shader Active")
            
            imgui.text("Raymarcher")
            imgui.separator()
            _, self.settings.density_scale = imgui.slider_float("Density", self.settings.density_scale, 1.0, 200.0)
            _, self.settings.density_curve = imgui.slider_float("Gamma", self.settings.density_curve, 0.1, 2.0)
            _, self.settings.step_size = imgui.slider_float("Step Size", self.settings.step_size, 0.001, 0.05)
            _, self.settings.step_count = imgui.slider_int("Max Steps", self.settings.step_count, 10, 500)
            
            imgui.dummy((0, 10))
            imgui.text("Lighting")
            imgui.separator()
            
            _, self.settings.sun_direction = imgui.slider_float3("Sun Direction", self.settings.sun_direction, -1.0, 1.0)
            _, self.settings.sun_intensity = imgui.slider_float("Sun Intensity", self.settings.sun_intensity, 0.0, 20.0)
            _, self.settings.sun_color_base = imgui.color_edit3("Sun Color", self.settings.sun_color_base)
            
            _, self.settings.ambient_intensity = imgui.slider_float("Ambient Intensity", self.settings.ambient_intensity, 0.0, 2.0)
            _, self.settings.ambient_color_base = imgui.color_edit3("Ambient Color", self.settings.ambient_color_base)

            imgui.dummy((0, 10))
            imgui.text("Volumetric Look")
            imgui.separator()
            
            _, self.settings.light_penetration = imgui.slider_float("Fluffiness", self.settings.light_penetration, 0.01, 1.0)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Low = Light penetrates deep (fluffy clouds)\nHigh = Light blocked early (dark smoke)")
                
            _, self.settings.phase_g = imgui.slider_float("Phase G (Silver Lining)", self.settings.phase_g, -0.9, 0.9)
            
            _, self.settings.shadow_steps = imgui.slider_int("Shadow Steps", self.settings.shadow_steps, 1, 16)
            _, self.settings.shadow_step_mult = imgui.slider_float("Shadow Step Mult", self.settings.shadow_step_mult, 1.0, 10.0)

            imgui.dummy((0, 10))
            _, self.settings.smoke_color = imgui.color_edit3("Smoke Albedo", self.settings.smoke_color)
            
            _, self.renderer.use_gaussian_volume = imgui.checkbox("Render Gaussian Volume", self.renderer.use_gaussian_volume)
            
        imgui.end()

        if imgui.begin("Training"):
            imgui.text("Optimizer")
            imgui.separator()
            
            _, self.train_config.use_adam = imgui.checkbox("Use Adam Optimizer", self.train_config.use_adam)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Adam: Adaptive learning with momentum (faster, smoother)\nSGD: Simple gradient descent")
            
            imgui.dummy((0, 10))
            imgui.text("Learning Rates")
            imgui.separator()
            _, self.train_config.learning_rate_pos = imgui.slider_float(
                "Position", self.train_config.learning_rate_pos, 0.0001, 0.5, format="%.4f")
            _, self.train_config.learning_rate_scale = imgui.slider_float(
                "Scale", self.train_config.learning_rate_scale, 0.00001, 0.1, format="%.5f")
            _, self.train_config.learning_rate_rotation = imgui.slider_float(
                "Rotation", self.train_config.learning_rate_rotation, 0.00001, 0.1, format="%.5f")
            _, self.train_config.learning_rate_weight = imgui.slider_float(
                "Weight", self.train_config.learning_rate_weight, 0.001, 0.5, format="%.4f")
            
            if self.train_config.use_adam:
                imgui.dummy((0, 10))
                imgui.text("Adam Hyperparameters")
                imgui.separator()
                _, self.train_config.adam_beta1 = imgui.slider_float(
                    "Beta1 (Momentum)", self.train_config.adam_beta1, 0.8, 0.999, format="%.3f")
                _, self.train_config.adam_beta2 = imgui.slider_float(
                    "Beta2 (Variance)", self.train_config.adam_beta2, 0.9, 0.9999, format="%.4f")
                _, self.train_config.adam_epsilon = imgui.slider_float(
                    "Epsilon", self.train_config.adam_epsilon, 1e-10, 1e-6, format="%.2e")
                
                imgui.text(f"Iteration: {self.renderer.adam_iteration}")
            
            imgui.dummy((0, 10))
            imgui.text("Gaussian Generation")
            imgui.separator()
            _, self.train_config.probability_scale = imgui.slider_float(
                "Spawn Probability", self.train_config.probability_scale, 0.001, 0.1, format="%.4f")
            _, self.train_config.sigma_scale = imgui.slider_float(
                "Initial Sigma Scale", self.train_config.sigma_scale, 1.0, 20.0)
            _, self.train_config.jitter_scale = imgui.slider_float(
                "Position Jitter", self.train_config.jitter_scale, 0.0, 200.0)

            imgui.separator()
            if imgui.button("Regenerate Gaussians"):
                self.gaussians = convert_grid_to_gaussians(self.grid, self.train_config)
                self.apply_densification(self.gaussians)
                self.renderer.adam_iteration = 1
                self.adc._ref_cache = None
                cmd = self.device.create_command_encoder()
                self.renderer.rasterize_gaussians(cmd, self.vol_min_world, self.vol_max_world)
                self.device.submit_command_buffer(cmd.finish())
                print("Gaussians regenerated!")

            imgui.dummy((0, 10))
            if imgui.button("Start Training" if not self.is_training else "Stop Training"):
                self.is_training = not self.is_training
            
            if self.is_training:
                optimizer_name = "ADAM" if self.train_config.use_adam else "SGD"
                imgui.text_colored((0, 1, 0, 1), f"TRAINING ACTIVE ({optimizer_name})")
            
            imgui.dummy((0, 5))
            imgui.separator()
            imgui.text("Export to SuperSplat")

            # The ONLY knob you should normally need
            _, self.ply_lighting.density_scale = imgui.slider_float(
                "Density Scale##ply", self.ply_lighting.density_scale, 1.0, 200.0)
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Match this to your Rendering > Density slider.\n"
                    "Opacity is computed physically: alpha = 1 - exp(-w * sigma * sqrt(2pi) * density_scale)\n"
                    "Too transparent? Raise it. Too solid? Lower it.")

            imgui.text_colored(imgui.ImVec4(0.6,0.6,0.6,1), "Shading")
            _, self.ply_lighting.bright_luminance = imgui.slider_float(
                "Bright##ply", self.ply_lighting.bright_luminance, 0.7, 1.0, "%.2f")
            _, self.ply_lighting.dark_luminance = imgui.slider_float(
                "Dark##ply", self.ply_lighting.dark_luminance, 0.05, 0.6, "%.2f")
            _, self.ply_lighting.depth_gamma = imgui.slider_float(
                "Depth Gamma##ply", self.ply_lighting.depth_gamma, 0.5, 4.0, "%.1f")
            _, self.ply_lighting.shadow_warmth = imgui.slider_float(
                "Shadow Warmth##ply", self.ply_lighting.shadow_warmth, 0.0, 0.4, "%.2f")

            imgui.dummy((0, 3))
            if imgui.button("Sync Density from Renderer"):
                self.ply_lighting.density_scale = self.settings.density_scale
            if imgui.is_item_hovered():
                imgui.set_tooltip("Copy density_scale from your Rendering panel")

            imgui.dummy((0, 3))
            if imgui.button("Save to PLY (SuperSplat)"):
                gpu_params = (
                    self.renderer.gaussian_buffer
                    .to_numpy()
                    .view(np.float32)
                    .reshape(-1, PARAMS_PER_GAUSSIAN)
                    .copy()
                )
                save_gaussians_dialog(gpu_params, self.ply_lighting)
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Opacity is physically derived from weight * sigma * density_scale.\n"
                    "This matches what your volumetric renderer shows.\n"
                    "Check console for opacity stats after saving.")
            

            # Debug
            imgui.dummy((0, 10))
        imgui.end()

        if imgui.begin("Debug"):
            imgui.text("Debug Visualization")
            imgui.separator()
            
            debug_modes = [
                "None (Normal Render)",
                "Loss Heatmap",
                "Gradient Magnitude",
                "Tile Gaussian Count",
                "Gaussian Overlap",
                "Prediction Error"
            ]
            
            old_mode = self.renderer.debug_mode
            clicked, self.renderer.debug_mode = imgui.combo(
                "Debug Mode", 
                self.renderer.debug_mode, 
                debug_modes
            )
            
            # Refresh debug when mode changes
            if self.renderer.debug_mode != old_mode:
                self.renderer.debug_needs_update = True
            
            if self.renderer.debug_mode > 0:
                old_scale = self.renderer.debug_scale
                _, self.renderer.debug_scale = imgui.slider_float(
                    "Visualization Scale", 
                    self.renderer.debug_scale, 
                    0.1, 5.0
                )
                
                # Refresh debug when scale changes
                if abs(self.renderer.debug_scale - old_scale) > 0.01:
                    self.renderer.debug_needs_update = True
                
                # Mode-specific information
                imgui.dummy((0, 5))
                
                if self.renderer.debug_mode == 1:  # Loss
                    imgui.text_colored((0.7, 0.7, 0.7, 1), "Loss Heatmap")
                    imgui.text_wrapped(
                        "Green: Low loss (< 0.05) - well fitted\n"
                        "Yellow: Medium loss (0.05-0.1) - needs improvement\n"
                        "Red: High loss (> 0.1) - poor fit, add Gaussians"
                    )
                    imgui.separator()
                    imgui.text(f"Typical range: 0.0 - 0.05")
                    imgui.text(f"Scale multiplier: {self.renderer.debug_scale:.2f}x")
                    
                elif self.renderer.debug_mode == 2:  # Gradients
                    imgui.text_colored((0.7, 0.7, 0.7, 1), "Gradient Magnitude")
                    imgui.text_wrapped(
                        "Shows where optimizer is making changes.\n"
                        "Green: Small gradients (< 0.04) - converged\n"
                        "Yellow: Medium gradients (0.04-0.1)\n"
                        "Red: Large gradients (> 0.1) - active learning"
                    )
                    imgui.separator()
                    
                    # Show actual gradient stats
                    if hasattr(self.renderer, 'grad_buffer') and self.renderer.grad_buffer:
                        grad_bytes = self.renderer.grad_buffer.to_numpy()
                        grad_debug = grad_bytes.view(dtype=np.float32)
                        grad_nonzero = grad_debug[grad_debug != 0]
                        if len(grad_nonzero) > 0:
                            imgui.text(f"Current gradients:")
                            imgui.text(f"  Max: {np.max(np.abs(grad_nonzero)):.6f}")
                            imgui.text(f"  Mean: {np.mean(np.abs(grad_nonzero)):.6f}")
                            imgui.text(f"  Active: {len(grad_nonzero)}/{len(grad_debug)}")
                    
                elif self.renderer.debug_mode == 3:  # Tile density
                    imgui.text_colored((0.7, 0.7, 0.7, 1), "Tile Gaussian Count")
                    imgui.text_wrapped(
                        "Shows Gaussian distribution efficiency.\n"
                        "Green: Sparse (< 16 Gaussians/tile)\n"
                        "Yellow: Medium (16-32)\n"
                        "Red: Dense (> 32) - may need larger tiles"
                    )
                    imgui.separator()
                    
                    # Show tile stats
                    if hasattr(self.renderer, 'tile_counts') and self.renderer.tile_counts:
                        tile_debug = self.renderer.tile_counts.to_numpy().view(dtype=np.uint32)
                        total_refs = np.sum(tile_debug)
                        occupied = np.sum(tile_debug > 0)
                        if occupied > 0:
                            imgui.text(f"Tile statistics:")
                            imgui.text(f"  Total refs: {total_refs:,}")
                            imgui.text(f"  Avg/tile: {total_refs/occupied:.1f}")
                            imgui.text(f"  Max/tile: {np.max(tile_debug)}")
                    
                elif self.renderer.debug_mode == 4:  # Overlap
                    imgui.text_colored((0.7, 0.7, 0.7, 1), "Gaussian Overlap")
                    imgui.text_wrapped(
                        "Shows combined Gaussian density.\n"
                        "Green: Low density (< 0.25)\n"
                        "Yellow: Medium density (0.25-0.5)\n"
                        "Red: High density (> 0.5)"
                    )
                    imgui.separator()
                    imgui.text(f"Typical range: 0.0 - 1.0")
                    
                elif self.renderer.debug_mode == 5:  # Error
                    imgui.text_colored((0.7, 0.7, 0.7, 1), "Prediction Error")
                    imgui.text_wrapped(
                        "Compares prediction vs. ground truth.\n"
                        "Blue: Underpredicting (add Gaussians)\n"
                        "White: Accurate (< ±0.02 error)\n"
                        "Red: Overpredicting (reduce weights)"
                    )
                    imgui.separator()
                    imgui.text(f"Error range: -0.1 to +0.1")
                
                imgui.dummy((0, 5))
                if imgui.button("Refresh Debug View"):
                    self.renderer.debug_needs_update = True
            
        imgui.end()

        if imgui.begin("Training Health Monitor"):
            health = self.renderer.gradient_health
            
            # Status indicator with color
            imgui.text("Gradient Status:")
            imgui.same_line()
            
            status_colors = {
                'healthy': (0.0, 1.0, 0.0, 1.0),      # Green
                'converged': (0.0, 0.8, 1.0, 1.0),    # Cyan
                'exploding': (1.0, 0.0, 0.0, 1.0),    # Red
                'vanishing': (1.0, 0.5, 0.0, 1.0),    # Orange
                'dead': (0.5, 0.0, 0.0, 1.0),         # Dark red
                'sparse': (1.0, 1.0, 0.0, 1.0),       # Yellow
                'unknown': (0.5, 0.5, 0.5, 1.0)       # Gray
            }
            
            color = status_colors.get(health['status'], (1, 1, 1, 1))
            imgui.text_colored(color, health['status'].upper())
            
            # Gradient statistics
            imgui.separator()
            imgui.text("Gradient Statistics:")
            imgui.text(f"  Mean: {health['mean']:.6f}")
            imgui.text(f"  Max:  {health['max']:.6f}")
            imgui.text(f"  Min:  {health['min']:.6f}")
            imgui.text(f"  Active: {health['active_ratio']*100:.1f}%")
            
            # Recommendation box
            imgui.dummy((0, 5))
            imgui.separator()
            
            if health['status'] in ['exploding', 'vanishing', 'dead', 'sparse']:
                # Warning box
                imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(1, 1, 0, 1))
                imgui.text_wrapped(f"⚠️  {health['recommendation']}")
                imgui.pop_style_color()
                
                # Quick fix buttons
                imgui.dummy((0, 5))
                
                if health['status'] == 'exploding':
                    if imgui.button("Fix: Reduce LRs by 5×"):
                        self.train_config.learning_rate_scale /= 5.0
                        self.train_config.learning_rate_rotation /= 5.0
                        self.train_config.learning_rate_weight /= 5.0
                        print(f"Reduced learning rates by 5×")
                    
                    imgui.same_line()
                    if imgui.button("Reset Adam State"):
                        self.renderer._init_adam_state()
                        print("Adam state reset")
                
                elif health['status'] == 'vanishing':
                    if imgui.button("Fix: Increase LRs by 5×"):
                        self.train_config.learning_rate_pos *= 5.0
                        self.train_config.learning_rate_scale *= 5.0
                        self.train_config.learning_rate_rotation *= 5.0
                        self.train_config.learning_rate_weight *= 5.0
                        print(f"Increased learning rates by 5×")
                
                elif health['status'] == 'sparse':
                    if imgui.button("Fix: Increase Sigma Scale"):
                        self.train_config.sigma_scale *= 1.5
                        imgui.text_colored((1, 1, 0, 1), "Note: Regenerate Gaussians to apply!")
            
            else:
                # All good
                imgui.text_colored((0, 1, 0, 1), f"✓ {health['recommendation']}")
            
            # Loss trend
            imgui.dummy((0, 10))
            imgui.separator()
            imgui.text("Loss Trend:")
            
            if len(self.renderer.loss_history) > 2:
                loss_array = np.array(self.renderer.loss_history[-100:], dtype=np.float32)
                imgui.plot_lines(
                    "##loss_trend",
                    loss_array,
                    scale_min=0.0,
                    scale_max=np.max(loss_array) * 1.1 if len(loss_array) > 0 else 1.0,
                    graph_size=(300, 80)
                )
                
                current_loss = self.renderer.loss_history[-1]
                imgui.text(f"Current: {current_loss:.10f}")
                
                # Loss trend analysis
                if len(loss_array) > 10:
                    recent = loss_array[-10:]
                    prev = loss_array[-20:-10] if len(loss_array) >= 20 else loss_array[:-10]
                    
                    improvement = np.mean(prev) - np.mean(recent) if len(prev) > 0 else 0
                    
                    if improvement > 0.001:
                        imgui.text_colored((0, 1, 0, 1), f"↓ Improving ({improvement:.6f}/10 iter)")
                    elif improvement < -0.001:
                        imgui.text_colored((1, 0, 0, 1), f"↑ Worsening ({-improvement:.6f}/10 iter)")
                    else:
                        imgui.text_colored((1, 1, 0, 1), "→ Plateaued")
            
            # Gradient magnitude trend
            imgui.dummy((0, 10))
            imgui.text("Gradient Magnitude Trend:")
            
            if len(self.renderer.grad_mean_history) > 2:
                grad_array = np.array(self.renderer.grad_mean_history[-100:], dtype=np.float32)
                imgui.plot_lines(
                    "##grad_trend",
                    grad_array,
                    scale_min=0.0,
                    scale_max=np.max(grad_array) * 1.1 if len(grad_array) > 0 else 1.0,
                    graph_size=(300, 80)
                )
        
        imgui.end()

        self.adc.draw_ui()

    def cleanup(self):
        imgui.backends.opengl3_shutdown()
        imgui.backends.glfw_shutdown()
        imgui.destroy_context()
        glfw.terminate()
    
    def compute_current_loss(self):
        """Compute MSE loss between Gaussian volume and reference"""
        # Only rasterize if we're using Gaussian volume
        if self.renderer.use_gaussian_volume:
            vol = self.renderer.gaussian_volume_tex.to_numpy()
        else:
            # Need to rasterize temporarily
            cmd = self.device.create_command_encoder()
            self.renderer.rasterize_gaussians(cmd, self.vol_min_world, self.vol_max_world)
            self.device.submit_command_buffer(cmd.finish())
            vol = self.renderer.gaussian_volume_tex.to_numpy()
        
        ref = self.renderer.volume_tex.to_numpy()
        diff = vol - ref
        return np.mean(diff * diff)


# ==========================================
# ENTRY POINT
# ==========================================

if __name__ == "__main__":
    app = App()
    app.run()