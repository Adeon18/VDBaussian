import slangpy as spy
import numpy as np
from pathlib import Path
import math
import random
import openvdb as vdb
import itertools
import time

# ==========================================
# UI MODE: Set to False for slim slangpy-native window (no OpenGL/imgui)
# ==========================================
EXTENDED_UI = True

if EXTENDED_UI:
    import glfw
    from OpenGL.GL import *
    import ctypes
    from imgui_bundle import imgui
    from save_ply import save_gaussians_dialog, CloudLightingConfig
    from metrics import MetricsCollector
    from adc import ADCConfig, ADCController
    from screen_metrics import ScreenMetricsCollector
else:
    import slangpy.ui as sui
    from adc import ADCConfig, ADCController

# ==========================================
# CONFIGURATION
# ==========================================

ENABLE_PROFILING = True  # Toggle detailed timing measurements
PROFILE_EVERY_N_FRAMES = 10  # Only profile every N iterations to reduce overhead
PROFILE_WAIT_FOR_GPU = False

# Debug Output
PRINT_TILE_STATS = True  # Print tiling statistics
PRINT_GRADIENT_STATS = True  # Print gradient statistics

# VDB_FILE = "C:\\Users\\ade0n\\Downloads\\bunny_cloud.vdb"
VDB_FILE = "cloud_01_variant_0000.vdb"
# VDB_FILE = "C:\\Users\\ade0n\\Downloads\\TornadoLoopingVDB\\TornadoLooping\\TornadoVDB\\tornado_0109.vdb"
VOL_SIZE = 196
USE_NATIVE_VDB_SIZE = False  # When True, VOL_SIZE is ignored and the VDB's native resolution is used
VDB_UP_AXIS = "-Z"  # Source VDB up axis: "+Y" (Houdini/Maya), "+Z" (Blender/3dsMax), etc.
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

class SGLDConfig:
    def __init__(self):
        self.enabled = False
        self.temperature_start = 1e-4
        self.temperature_decay = 0.9999
        self.temperature_min   = 1e-7
        self.sgld_k = 10.0
        self._current_temperature = self.temperature_start

    def reset(self):
        self._current_temperature = self.temperature_start

    def step(self):
        self._current_temperature = max(
            self._current_temperature * self.temperature_decay,
            self.temperature_min,
        )

    @property
    def current_temperature(self):
        return self._current_temperature

    @property
    def gpu_enabled_flag(self):
        return 1 if self.enabled else 0


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

        # Splatting-specific
        self.shadow_strength = 1.0        # multiplier on self-shadow (1.0 = physical)
        self.splat_softness = 0.3         # low-pass filter on 2D covariance (higher = softer edges)
        self.enable_blur = False          # screen-space blur post-process
        self.blur_radius = 2             # blur kernel half-size (1-4 pixels, only used when enable_blur=True)
        self.use_depth_darkening = False   # view-dependent depth darkening

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
        f = np.array(
            [
                math.cos(rad_yaw) * math.cos(rad_pitch),
                math.sin(rad_pitch),
                math.sin(rad_yaw) * math.cos(rad_pitch),
            ],
            dtype=np.float32,
        )
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


# VDB files do not store an up-axis convention.  Different DCC tools
# bake data in their own coordinate system:
#   Houdini / Maya  → +Y up   (most common for VDB content)
#   Blender / 3ds Max → +Z up
#   EmberGen        → configurable at export
# Real engines (UE, Unity) let the user pick — we do the same.
VDB_UP_AXES = {
    "+Y": (1,  1),   # Houdini / Maya (default)
    "-Y": (1, -1),
    "+Z": (2,  1),   # Blender / 3ds Max
    "-Z": (2, -1),
    "+X": (0,  1),
    "-X": (0, -1),
}
VDB_UP_AXIS_NAMES = list(VDB_UP_AXES.keys())


def _build_axis_remap(up_axis, up_sign):
    """Build permutation and sign arrays that remap VDB world space so
    that the detected up axis becomes Y with the correct direction.

    Returns (perm, signs) where:
        our_pos[i] = signs[i] * vdb_pos[perm[i]]
        vdb_pos[perm[i]] = signs[i] * our_pos[i]   (inverse)
    """
    if up_axis == 0:       # VDB X is up → swap X↔Y
        perm = [1, 0, 2]
    elif up_axis == 2:     # VDB Z is up → swap Y↔Z
        perm = [0, 2, 1]
    else:                  # VDB Y is up → identity
        perm = [0, 1, 2]

    signs = [1, up_sign, 1]
    return perm, signs


def _remap_position(pos, perm, signs):
    """VDB world position → our coordinate system."""
    return np.array([signs[i] * pos[perm[i]] for i in range(3)])


def _remap_bounds(world_min, world_max, perm, signs):
    """Remap an axis-aligned bounding box from VDB world to our space."""
    our_min = np.zeros(3)
    our_max = np.zeros(3)
    for i in range(3):
        lo = world_min[perm[i]]
        hi = world_max[perm[i]]
        if signs[i] < 0:
            lo, hi = -hi, -lo
        our_min[i] = lo
        our_max[i] = hi
    return our_min, our_max



def _get_native_vdb_size(grid, up_axis_name="+Y"):
    """Compute the cubic resolution needed to represent the VDB at native resolution.

    Returns the max dimension of the active voxel bounding box (after axis remap),
    so the output cube matches the longest axis 1:1 with the VDB's own voxels.
    """
    bbox = grid.evalActiveVoxelBoundingBox()
    min_i, max_i = np.array(bbox[0]), np.array(bbox[1])
    vdb_dims = max_i - min_i + 1

    up_axis, up_sign = VDB_UP_AXES[up_axis_name]
    perm, _ = _build_axis_remap(up_axis, up_sign)
    remapped_dims = np.array([vdb_dims[perm[i]] for i in range(3)])

    return int(np.max(remapped_dims))


def convert_grid_to_dense_volume(grid, size, up_axis_name="+Y", use_native_size=False):
    """Convert sparse VDB grid to dense 3D texture.

    Remaps the VDB so the specified source up axis becomes +Y in our
    coordinate system, then samples uniformly in world space.

    Args:
        grid:             OpenVDB grid object.
        size:             Voxels per axis for the output cube (ignored when
                          use_native_size is True).
        up_axis_name:     Source up axis — one of "+Y", "-Y", "+Z", "-Z",
                          "+X", "-X".  Default "+Y" (Houdini/Maya).
        use_native_size:  When True, *size* is overridden by the VDB's native
                          resolution (max active-bbox axis).

    Returns (vol_min, vol_max, data, axis_remap, resolved_size) where
    axis_remap is a (perm, signs) tuple needed by convert_grid_to_gaussians,
    and resolved_size is the actual voxels-per-axis used (== size unless
    use_native_size was True).
    """
    if grid is None:
        print("Error! Grid is None!")
        return None

    bbox = grid.evalActiveVoxelBoundingBox()
    min_i, max_i = np.array(bbox[0]), np.array(bbox[1])

    vdb_dims = max_i - min_i + 1

    if use_native_size:
        size = _get_native_vdb_size(grid, up_axis_name)
        print(f"=== VDB Info (native resolution) ===")
    else:
        print(f"=== VDB Info ===")
    print(f"  Active bbox (index): {tuple(min_i)} -> {tuple(max_i)}")
    print(f"  VDB dimensions: {vdb_dims[0]} x {vdb_dims[1]} x {vdb_dims[2]} voxels")
    print(f"  Total active voxels: ~{int(np.prod(vdb_dims)):,}")
    print(f"  Output resolution: {size} x {size} x {size}")

    transform = grid.transform

    # Compute world-space AABB from all 8 corners of the index bbox
    corners_idx = np.array([
        [min_i[0], min_i[1], min_i[2]],
        [max_i[0], min_i[1], min_i[2]],
        [min_i[0], max_i[1], min_i[2]],
        [min_i[0], min_i[1], max_i[2]],
        [max_i[0], max_i[1], min_i[2]],
        [max_i[0], min_i[1], max_i[2]],
        [min_i[0], max_i[1], max_i[2]],
        [max_i[0], max_i[1], max_i[2]],
    ], dtype=np.float64)
    corners_world = np.array([
        transform.indexToWorld(tuple(c)) for c in corners_idx
    ])
    vdb_world_min = corners_world.min(axis=0)
    vdb_world_max = corners_world.max(axis=0)

    # Build axis remap from user-specified up axis
    up_axis, up_sign = VDB_UP_AXES[up_axis_name]
    perm, signs = _build_axis_remap(up_axis, up_sign)
    axis_remap = (perm, signs)

    print(f"  Source up axis: {up_axis_name} → remap to +Y")

    # Remap bounds to our coordinate system (Y-up)
    our_min, our_max = _remap_bounds(vdb_world_min, vdb_world_max, perm, signs)

    print(f"  VDB world bounds: {vdb_world_min} -> {vdb_world_max}")
    print(f"  Remapped bounds:  {our_min} -> {our_max}")
    print(f"================")

    # Make a uniform cube (preserves aspect ratio)
    center = (our_min + our_max) / 2.0
    half = np.max(our_max - our_min) / 2.0
    min_world_bound = center - half
    max_world_bound = center + half

    # Build a grid of sample positions (vectorized)
    coords = np.arange(size, dtype=np.float64) + 0.5
    xg, yg, zg = np.meshgrid(coords, coords, coords, indexing='ij')
    # xg[x,y,z] = x+0.5, yg[x,y,z] = y+0.5, zg[x,y,z] = z+0.5
    t_x = xg / size
    t_y = yg / size
    t_z = zg / size

    extent = max_world_bound - min_world_bound
    our_x = min_world_bound[0] + t_x * extent[0]
    our_y = min_world_bound[1] + t_y * extent[1]
    our_z = min_world_bound[2] + t_z * extent[2]

    # Inverse remap: our coords → VDB world coords (vectorized)
    signs_arr = np.array(signs, dtype=np.float64)
    our_components = [our_x, our_y, our_z]
    vdb_x = np.zeros_like(our_x)
    vdb_y = np.zeros_like(our_y)
    vdb_z = np.zeros_like(our_z)
    vdb_components = [vdb_x, vdb_y, vdb_z]
    for i in range(3):
        vdb_components[perm[i]] = signs_arr[i] * our_components[i]
    vdb_x, vdb_y, vdb_z = vdb_components

    # Sample the VDB using the accessor (must loop — accessor is not vectorizable)
    accessor = grid.getAccessor()
    data = np.zeros((size, size, size), dtype=np.float32)
    total_voxels = size * size * size
    print(f"  Sampling {total_voxels:,} voxels...")
    t_start = time.time()

    for x in range(size):
        for y in range(size):
            for z in range(size):
                vdb_pos = (float(vdb_x[x, y, z]),
                           float(vdb_y[x, y, z]),
                           float(vdb_z[x, y, z]))
                idx = transform.worldToIndex(vdb_pos)
                val = accessor.getValue((int(idx[0]), int(idx[1]), int(idx[2])))
                data[z, y, x] = val

    elapsed = time.time() - t_start
    print(f"  Sampling took {elapsed:.1f}s")

    m = np.max(data)
    if m > 0:
        data /= m
    fill_ratio = np.count_nonzero(data) / data.size * 100
    print(f"  Fill ratio: {fill_ratio:.1f}% of voxels non-zero")

    return min_world_bound, max_world_bound, np.ascontiguousarray(data, dtype=np.float32), axis_remap, size


def convert_grid_to_gaussians(grid, config, axis_remap=None):
    """Stochastically sample VDB grid to create Gaussians.

    Supports two modes (config.gaussian_mode):
      "percentage" — probability is config.probability_scale.
      "count"      — probability is auto-calculated from
                     config.target_count / n_active_voxels.

    Sampling (config.density_weighted):
      True  — acceptance = value * prob  (biased toward dense regions)
      False — acceptance = prob          (uniform across all active voxels)

    In both modes, config.min_count / config.max_count are enforced.
    If the result undershoots min_count the probability is bumped and
    generation retried.  If it overshoots max_count, excess Gaussians
    are randomly dropped.

    If axis_remap is provided (from convert_grid_to_dense_volume),
    Gaussian positions are remapped into the same Y-up coordinate
    system as the dense volume.
    """
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

    if axis_remap is not None:
        perm, signs = axis_remap
    else:
        perm, signs = [0, 1, 2], [1, 1, 1]

    # Collect all active voxels (index, value) once — reused across retries
    active_voxels = []
    for z in range(min_i[2], max_i[2] + 1):
        for y in range(min_i[1], max_i[1] + 1):
            for x in range(min_i[0], max_i[0] + 1):
                value = accessor.getValue((x, y, z))
                if value > 0.0:
                    active_voxels.append(((x, y, z), value))

    n_active = len(active_voxels)
    print(f"  Active voxels: {n_active:,}")

    if n_active == 0:
        print("  No active voxels — returning empty.")
        return np.array([], dtype=np.float32)

    # Determine probability
    mode = getattr(config, "gaussian_mode", "percentage")
    min_count = getattr(config, "min_count", 0)
    max_count = getattr(config, "max_count", 999_999)

    density_weighted = getattr(config, "density_weighted", True)
    sampling = "density-weighted" if density_weighted else "uniform"

    if mode == "count":
        target = getattr(config, "target_count", 8000)
        if density_weighted:
            mean_density = sum(v for _, v in active_voxels) / max(n_active, 1)
            prob = min(target / (max(n_active, 1) * mean_density), 1.0)
        else:
            prob = min(target / max(n_active, 1), 1.0)
        print(f"  Mode: count (target={target}, auto prob={prob:.5f}, {sampling})")
    else:
        prob = config.probability_scale
        print(f"  Mode: percentage (prob_scale={prob:.5f}, {sampling})")

    # Generate with retry if undershooting min_count
    max_attempts = 3
    for attempt in range(max_attempts):
        gaussians = []
        sigma = voxel_size * config.sigma_scale

        for (ix, iy, iz), value in active_voxels:
            acceptance = value * prob if density_weighted else prob
            if acceptance < random.random():
                continue

            center = np.array(transform.indexToWorld((ix, iy, iz)), dtype=np.float64)
            jitter = (np.random.rand(3) - 0.5) * voxel_size * config.jitter_scale
            position = _remap_position(center + jitter, perm, signs)

            gaussians.append((
                position[0], position[1], position[2],
                sigma, sigma, sigma,
                0.0, 0.0, 0.0, 1.0,
                value,
            ))

        count = len(gaussians)

        if count >= min_count or attempt == max_attempts - 1:
            break

        # Undershoot — bump probability for next attempt
        old_prob = prob
        prob = min(min_count / max(n_active, 1), 1.0)
        print(f"  Retry {attempt + 1}: {count} < min_count {min_count}, "
              f"bumping prob {old_prob:.5f} -> {prob:.5f}")

    # Clamp to max_count by random subsampling
    if len(gaussians) > max_count:
        print(f"  Clamping {len(gaussians):,} -> {max_count:,} (random subsample)")
        indices = random.sample(range(len(gaussians)), max_count)
        gaussians = [gaussians[i] for i in indices]

    print(f"  Generated {len(gaussians):,} Gaussians "
          f"(active={n_active:,}, mode={mode})")
    return np.array(gaussians, dtype=np.float32)


class SGLDDiagnostics:
    def __init__(self, app):
        self.app = app
        self._tracked_positions = None
        self._tracked_indices   = None
        self._tile_history      = []
        self._mobility_history  = []

    def snapshot(self):
        """Call once after ~100 training steps to establish baseline."""
        params  = self._get_params()
        weights = params[:, 10]
        mask    = weights < np.percentile(weights, 20)
        self._tracked_positions = params[mask, 0:3].copy()
        self._tracked_indices   = np.where(mask)[0]
        print(f"[SGLD Diag] Tracking {mask.sum()} low-weight Gaussians (n={len(params)})")

    def tick(self):
        """Call every 500 frames while training."""
        params  = self._get_params()
        weights = params[:, 10]
        n       = len(params)

        # Тoise sanity: displacement binned by weight
        if self._tracked_positions is not None and self._tracked_indices is not None:
            valid_mask      = self._tracked_indices < n
            valid_indices   = self._tracked_indices[valid_mask]
            valid_positions = self._tracked_positions[valid_mask]

            if len(valid_indices) > 0:
                current   = params[valid_indices, 0:3]
                disp      = np.linalg.norm(current - valid_positions, axis=1)
                mean_disp = np.mean(disp)
                self._mobility_history.append(mean_disp)
                print(f"[SGLD Diag] Low-weight displacement  mean={mean_disp:.5f}  max={np.max(disp):.5f}"
                    f"  ({len(valid_indices)}/{len(self._tracked_indices)} survivors)")
            else:
                print(f"[SGLD Diag] All tracked Gaussians pruned — re-snapshotting")
                self.snapshot()
                return

        # Gradient magnitude binned by weight band
        g = self.app.renderer.grad_buffer.to_numpy().view(np.float32).reshape(-1, 11)
        for lo, hi in [(0.0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.0)]:
            mask = (weights >= lo) & (weights < hi)
            if mask.sum() > 0:
                pos_grad_mag = np.linalg.norm(g[mask, 0:3], axis=1).mean()
                print(f"[SGLD Diag]   w[{lo:.1f}-{hi:.1f}]  n={mask.sum():4d}  "
                    f"pos_grad_mag={pos_grad_mag:.6f}")

        # Tile coverage
        tile_counts = self.app.renderer.tile_counts.to_numpy().view(np.uint32)
        occupied    = np.sum(tile_counts > 0)
        total       = len(tile_counts)
        self._tile_history.append(occupied)
        print(f"[SGLD Diag] Tile coverage  {occupied}/{total}  ({occupied/total*100:.1f}%)")

        if len(self._tile_history) >= 3:
            trend = self._tile_history[-1] - self._tile_history[-3]
            if trend > 0:
                print(f"[SGLD Diag]   ↑ Coverage growing (+{trend} tiles) — Gaussians migrating")
            elif trend == 0:
                print(f"[SGLD Diag]   → Coverage static — Gaussians not migrating")
            else:
                print(f"[SGLD Diag]   ↓ Coverage shrinking — ADC pruning faster than migration")

    def _get_params(self):
        return (
            self.app.renderer.gaussian_buffer.to_numpy()
            .view(np.float32)
            .reshape(-1, 11)
            .copy()
        )


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
        self.gaussian_mode = "percentage"  # "percentage" or "count"
        self.density_weighted = True       # True = bias toward dense voxels, False = uniform
        self.probability_scale = 0.02
        self.target_count = 8000
        self.min_count = 500
        self.max_count = 50000
        self.sigma_scale = 2.0
        self.jitter_scale = 5.0

        self.loss_mode = 0  # 0=L2, 1=L1(pseudo), 2=Huber
        self.huber_delta = 0.1

        # SSIM combined loss: L = (1-ssim_weight)*L_pervoxel + ssim_weight*D-SSIM
        self.ssim_weight = 0.0  # 0.0 = off, 0.2 = recommended
        self.ssim_window_radius = 2  # 1 = 3x3x3, 2 = 5x5x5

        self.sgld = SGLDConfig()


# ==========================================
# RENDERER SYSTEM
# ==========================================


class Renderer:
    def __init__(self, device, volume_data):
        self.device = device
        self.pipeline = None
        self.last_mod_time = 0
        self.error_msg = ""
        self.tile_size = TILE_SIZE
        self.max_gaussians_per_tile = MAX_GAUSSIANS_PER_TILE

        self.linear_sampler = device.create_sampler(
            min_filter=spy.TextureFilteringMode.linear,
            mag_filter=spy.TextureFilteringMode.linear,
            mip_filter=spy.TextureFilteringMode.linear,
            address_u=spy.TextureAddressingMode.wrap,
            address_v=spy.TextureAddressingMode.wrap,
            address_w=spy.TextureAddressingMode.wrap,
        )

        self.volume_tex = device.create_texture(
            type=spy.TextureType.texture_3d,
            format=spy.Format.r32_float,
            width=VOL_SIZE,
            height=VOL_SIZE,
            depth=VOL_SIZE,
            usage=spy.TextureUsage.shader_resource,
            label="VDBVolume",
        )

        self.tile_res = (
            (VOL_SIZE + TILE_SIZE - 1) // TILE_SIZE,
            (VOL_SIZE + TILE_SIZE - 1) // TILE_SIZE,
            (VOL_SIZE + TILE_SIZE - 1) // TILE_SIZE,
        )

        cmd = device.create_command_encoder()
        cmd.upload_texture_data(self.volume_tex, [volume_data])
        device.submit_command_buffer(cmd.finish())

        self.cam_buffer = device.create_buffer(
            size=64,
            usage=spy.BufferUsage.shader_resource,
            memory_type=spy.MemoryType.upload,
        )
        self.settings_buffer = device.create_buffer(
            size=128,
            usage=spy.BufferUsage.shader_resource,
            memory_type=spy.MemoryType.upload,
        )

        self.screen_tex = None
        self.display_gl_tex = None
        self.width, self.height = 0, 0

        self.gaussian_volume_tex = device.create_texture(
            type=spy.TextureType.texture_3d,
            format=spy.Format.r32_float,
            width=VOL_SIZE,
            height=VOL_SIZE,
            depth=VOL_SIZE,
            usage=spy.TextureUsage.unordered_access | spy.TextureUsage.shader_resource,
            label="GaussianVolume",
        )

        self.ssim_gradient_tex = device.create_texture(
            type=spy.TextureType.texture_3d,
            format=spy.Format.r32_float,
            width=VOL_SIZE,
            height=VOL_SIZE,
            depth=VOL_SIZE,
            usage=spy.TextureUsage.unordered_access | spy.TextureUsage.shader_resource,
            label="SSIMGradient",
        )

        self._needs_rebinning = True
        self._needs_rasterization = True
        prog_raster_tiled = self.device.load_program(
            "shaders/3drasterizer.slang", ["rasterize_tiled"]
        )
        self.pipe_raster_tiled = self.device.create_compute_pipeline(
            program=prog_raster_tiled
        )

        self.use_gaussian_volume = False
        self.use_splatting = False  # Splatting render mode

        self.gaussian_buffer = None
        self.gaussian_count = 0
        self.adam_iteration = 1  # Track iteration for bias correction

        # Adam state buffers
        self.adam_first_moment = None
        self.adam_second_moment = None

        self._training_initialized = False

        # Splatting renderer setup
        self._init_splatting_pipelines()

        self.check_hot_reload()

        # Debug
        self.debug_tex = device.create_texture(
            type=spy.TextureType.texture_3d,
            format=spy.Format.rgba8_unorm,
            width=VOL_SIZE,
            height=VOL_SIZE,
            depth=VOL_SIZE,
            usage=spy.TextureUsage.unordered_access | spy.TextureUsage.shader_resource,
            label="DebugVolume",
        )

        self.debug_mode = 0
        self.debug_scale = 1.0

        # Load debug raymarcher
        prog_debug = device.load_program(
            "shaders/debug_raymarch.slang", ["vertex_main", "fragment_main"]
        )
        self.debug_pipeline = device.create_render_pipeline(
            program=prog_debug,
            input_layout=device.create_input_layout(
                input_elements=[], vertex_streams=[]
            ),
            targets=[{"format": spy.Format.rgba8_unorm}],
        )

        self.gradient_health = {
            "status": "unknown",  # 'healthy', 'exploding', 'vanishing', 'dead'
            "mean": 0.0,
            "max": 0.0,
            "min": 0.0,
            "active_ratio": 0.0,
            "recommendation": "",
        }

        self.loss_history = []
        self.grad_mean_history = []

    def init_training(self):

        total_tiles = self.tile_res[0] * self.tile_res[1] * self.tile_res[2]

        self.grad_buffer = self.device.create_buffer(
            element_count=self.gaussian_count * PARAMS_PER_GAUSSIAN,
            struct_size=4,
            usage=spy.BufferUsage.unordered_access | spy.BufferUsage.shader_resource,
            memory_type=spy.MemoryType.device_local,
        )

        self.tile_content = self.device.create_buffer(
            element_count=total_tiles * self.max_gaussians_per_tile,
            struct_size=4,
            usage=spy.BufferUsage.unordered_access | spy.BufferUsage.shader_resource,
            memory_type=spy.MemoryType.device_local,
        )

        self.tile_counts = self.device.create_buffer(
            element_count=total_tiles,
            struct_size=4,
            usage=spy.BufferUsage.unordered_access | spy.BufferUsage.shader_resource,
            memory_type=spy.MemoryType.device_local,
        )

        self.adam_first_moment = self.device.create_buffer(
            element_count=self.gaussian_count * PARAMS_PER_GAUSSIAN,
            struct_size=4,
            usage=spy.BufferUsage.unordered_access | spy.BufferUsage.shader_resource,
            memory_type=spy.MemoryType.device_local,
        )

        self.adam_second_moment = self.device.create_buffer(
            element_count=self.gaussian_count * PARAMS_PER_GAUSSIAN,
            struct_size=4,
            usage=spy.BufferUsage.unordered_access | spy.BufferUsage.shader_resource,
            memory_type=spy.MemoryType.device_local,
        )

        if not self._training_initialized:
            print("Compiling Kernels...")

            prog_clear_grads = self.device.load_program(
                "shaders/training.slang", ["clear_gradients"]
            )
            self.pipe_clear_grads = self.device.create_compute_pipeline(
                program=prog_clear_grads
            )

            prog_clear_tiles = self.device.load_program(
                "shaders/training.slang", ["clear_tiles"]
            )
            self.pipe_clear_tiles = self.device.create_compute_pipeline(
                program=prog_clear_tiles
            )

            prog_bin = self.device.load_program(
                "shaders/training.slang", ["bin_gaussians"]
            )
            self.pipe_bin = self.device.create_compute_pipeline(program=prog_bin)

            prog_train = self.device.load_program(
                "shaders/training.slang", ["train_main"]
            )
            self.pipe_train = self.device.create_compute_pipeline(program=prog_train)

            prog_optim_sgd = self.device.load_program(
                "shaders/training.slang", ["optimizer_sgd"]
            )
            self.pipe_optim_sgd = self.device.create_compute_pipeline(
                program=prog_optim_sgd
            )

            prog_optim_adam = self.device.load_program(
                "shaders/training.slang", ["optimizer_adam"]
            )
            self.pipe_optim_adam = self.device.create_compute_pipeline(
                program=prog_optim_adam
            )

            prog_init_adam = self.device.load_program(
                "shaders/training.slang", ["init_adam_state"]
            )
            self.pipe_init_adam = self.device.create_compute_pipeline(
                program=prog_init_adam
            )

            prog_ssim = self.device.load_program(
                "shaders/training.slang", ["compute_ssim_gradient"]
            )
            self.pipe_ssim_gradient = self.device.create_compute_pipeline(
                program=prog_ssim
            )

            prog_debug = self.device.load_program(
                "shaders/training.slang", ["compute_debug"]
            )
            self.pipe_debug_only = self.device.create_compute_pipeline(
                program=prog_debug
            )

            self._training_initialized = True

        self.debug_needs_update = True
        # Initialize Adam state to zeros
        self._init_adam_state()
    
    def _bind_sgld_params(self, cursor, train_config):
        cursor["TrainParams"]["sgldEnabled"]    = train_config.sgld.gpu_enabled_flag
        cursor["TrainParams"]["sgldTemperature"] = train_config.sgld.current_temperature
        cursor["TrainParams"]["sgldK"]           = train_config.sgld.sgld_k

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

    def set_max_gaussians_per_tile(self, value):
        """Change max gaussians per tile and recreate the tile content buffer."""
        value = max(1, int(value))
        if value == self.max_gaussians_per_tile:
            return
        self.max_gaussians_per_tile = value
        total_tiles = self.tile_res[0] * self.tile_res[1] * self.tile_res[2]
        self.tile_content = self.device.create_buffer(
            element_count=total_tiles * self.max_gaussians_per_tile,
            struct_size=4,
            usage=spy.BufferUsage.unordered_access | spy.BufferUsage.shader_resource,
            memory_type=spy.MemoryType.device_local,
        )
        self._needs_rebinning = True
        print(f"Tile content buffer resized: max_gaussians_per_tile={value}")

    def analyze_gradients(self):
        """Analyze gradient health - CALIBRATED FOR VOLUMETRIC FITTING. This function are just suggestions for me"""
        if not hasattr(self, "grad_buffer") or self.grad_buffer is None:
            return

        grad_bytes = self.grad_buffer.to_numpy()
        grads = grad_bytes.view(dtype=np.float32)

        grad_abs = np.abs(grads)
        nonzero_grads = grad_abs[grad_abs > 1e-10]

        if len(nonzero_grads) == 0:
            self.gradient_health = {
                "status": "dead",
                "mean": 0.0,
                "max": 0.0,
                "min": 0.0,
                "active_ratio": 0.0,
                "recommendation": "No gradients! Check if training is running.",
            }
            return

        mean_grad = np.mean(nonzero_grads)
        max_grad = np.max(nonzero_grads)
        min_grad = np.min(nonzero_grads)
        active_ratio = len(nonzero_grads) / len(grads)

        self.grad_mean_history.append(mean_grad)
        if len(self.grad_mean_history) > 100:
            self.grad_mean_history.pop(0)

        status = "healthy"
        recommendation = "Training normally. Gradients reflect prediction error."

        # Check for TRUE exploding gradients (much higher thresholds!)
        if max_grad > 50.0:
            status = "exploding"
            recommendation = (
                "CRITICAL: Gradients exploding! Reduce learning rates by 10×"
            )
        elif max_grad > 20.0:
            status = "high"
            recommendation = "Gradients high but manageable. Consider reducing LRs by 2× if loss oscillates."

        # Early training (high error = high gradients)
        elif mean_grad > 1.0:
            status = "early"
            recommendation = (
                "Early training - high gradients are normal as error is large."
            )

        # Check for vanishing gradients
        elif mean_grad < 1e-5:
            status = "vanishing"
            if mean_grad < 1e-7:
                recommendation = (
                    "CRITICAL: Gradients vanishing! Increase learning rates by 10×"
                )
            else:
                recommendation = "Gradients getting small. May need higher learning rates or have converged."

        # Check for convergence
        elif mean_grad < 0.001 and len(self.grad_mean_history) > 20:
            recent = self.grad_mean_history[-20:]
            std = np.std(recent)
            if std < mean_grad * 0.1:
                status = "converged"
                recommendation = (
                    "Gradients stable and small. Training likely converged!"
                )

        # Check trend (is it improving?)
        if status == "healthy" and len(self.grad_mean_history) > 20:
            recent = self.grad_mean_history[-20:]
            older = (
                self.grad_mean_history[-40:-20]
                if len(self.grad_mean_history) >= 40
                else self.grad_mean_history[:-20]
            )

            if len(older) > 0:
                recent_mean = np.mean(recent)
                older_mean = np.mean(older)

                if recent_mean < older_mean * 0.8:
                    status = "improving"
                    recommendation = "Gradients decreasing - loss is improving! Keep current settings."
                elif recent_mean > older_mean * 1.2:
                    status = "diverging"
                    recommendation = "Gradients increasing - may be diverging. Consider reducing learning rates."

        # Check for sparse activation
        if active_ratio < 0.3:
            status = "sparse"
            recommendation = f"Only {active_ratio*100:.1f}% Gaussians active. Increase sigma_scale or add more Gaussians."

        self.gradient_health = {
            "status": status,
            "mean": mean_grad,
            "max": max_grad,
            "min": min_grad,
            "active_ratio": active_ratio,
            "recommendation": recommendation,
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
            cursor["TrainParams"]["maxGaussiansPerTile"] = self.max_gaussians_per_tile
            cursor["TrainParams"]["debugMode"] = self.debug_mode
            cursor["TrainParams"]["debugScale"] = self.debug_scale
            cursor["TrainParams"]["gaussianCount"] = self.gaussian_count

            cp.dispatch(thread_count=(VOL_SIZE, VOL_SIZE, VOL_SIZE))

        self.debug_needs_update = False

    def run_training_step(self, vol_min, vol_max, train_config):
        # Manage the SGLD temps
        if train_config.sgld.enabled:
            train_config.sgld.step()

        cmd = self.device.create_command_encoder()

        total_tiles = self.tile_res[0] * self.tile_res[1] * self.tile_res[2]

        # 1. Clear Gradients
        with cmd.begin_compute_pass() as cp:
            root_object = cp.bind_pipeline(self.pipe_clear_grads)
            cursor = spy.ShaderCursor(root_object)
            cursor["gGradientsRaw"] = self.grad_buffer
            cursor["TrainParams"]["gaussianCount"] = self.gaussian_count
            cursor["TrainParams"]["tileSize"] = TILE_SIZE
            cursor["TrainParams"]["maxGaussiansPerTile"] = self.max_gaussians_per_tile

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
                cursor["TrainParams"]["maxGaussiansPerTile"] = self.max_gaussians_per_tile
                cp.dispatch(thread_count=(total_tiles, 1, 1))

            # 3. Bin Gaussians
            with cmd.begin_compute_pass() as cp:
                root_object = cp.bind_pipeline(self.pipe_bin)
                cursor = spy.ShaderCursor(root_object)

                cursor["gGaussianParamsRaw"] = self.gaussian_buffer
                cursor["gTileContent"] = self.tile_content
                cursor["gTileCounts"] = self.tile_counts
                cursor["TrainParams"]["volumeResolution"] = (
                    VOL_SIZE,
                    VOL_SIZE,
                    VOL_SIZE,
                )
                cursor["TrainParams"]["tileResolution"] = self.tile_res
                cursor["TrainParams"]["gaussianCount"] = self.gaussian_count
                cursor["TrainParams"]["minWorld"] = tuple(vol_min)
                cursor["TrainParams"]["maxWorld"] = tuple(vol_max)
                cursor["TrainParams"]["tileSize"] = TILE_SIZE
                cursor["TrainParams"]["maxGaussiansPerTile"] = self.max_gaussians_per_tile
                cp.dispatch(thread_count=(self.gaussian_count, 1, 1))
            self._needs_rebinning = False

        # 3.5. SSIM gradient precomputation (uses predicted volume from previous step)
        ssim_active = train_config.ssim_weight > 0.0 and self.adam_iteration > 1
        if ssim_active:
            with cmd.begin_compute_pass() as cp:
                root_object = cp.bind_pipeline(self.pipe_ssim_gradient)
                cursor = spy.ShaderCursor(root_object)

                cursor["PredictedVol"] = self.gaussian_volume_tex
                cursor["ReferenceVol"] = self.volume_tex
                cursor["gSSIMGradientVol"] = self.ssim_gradient_tex

                cursor["TrainParams"]["volumeResolution"] = (VOL_SIZE, VOL_SIZE, VOL_SIZE)
                cursor["TrainParams"]["ssimWeight"] = train_config.ssim_weight
                cursor["TrainParams"]["ssimWindowRadius"] = train_config.ssim_window_radius

                cp.dispatch(thread_count=(VOL_SIZE, VOL_SIZE, VOL_SIZE))

        # 4. Train
        with cmd.begin_compute_pass() as cp:
            root_object = cp.bind_pipeline(self.pipe_train)
            cursor = spy.ShaderCursor(root_object)

            cursor["ReferenceVol"] = self.volume_tex
            cursor["gGaussianParamsRaw"] = self.gaussian_buffer
            cursor["gGradientsRaw"] = self.grad_buffer
            cursor["gTileContent"] = self.tile_content
            cursor["gTileCounts"] = self.tile_counts
            cursor["gSSIMGradientVol"] = self.ssim_gradient_tex

            cursor["TrainParams"]["volumeResolution"] = (VOL_SIZE, VOL_SIZE, VOL_SIZE)
            cursor["TrainParams"]["tileResolution"] = self.tile_res
            cursor["TrainParams"]["minWorld"] = tuple(vol_min)
            cursor["TrainParams"]["maxWorld"] = tuple(vol_max)
            cursor["TrainParams"]["tileSize"] = TILE_SIZE
            cursor["TrainParams"]["maxGaussiansPerTile"] = self.max_gaussians_per_tile

            cursor["TrainParams"]["lossMode"] = train_config.loss_mode
            cursor["TrainParams"]["huberDelta"] = train_config.huber_delta
            cursor["TrainParams"]["ssimWeight"] = train_config.ssim_weight if ssim_active else 0.0
            cursor["TrainParams"]["ssimWindowRadius"] = train_config.ssim_window_radius

            cp.dispatch(thread_count=(VOL_SIZE, VOL_SIZE, VOL_SIZE))

        # 5. Optimize
        pipeline = (
            self.pipe_optim_adam if train_config.use_adam else self.pipe_optim_sgd
        )

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
                train_config.learning_rate_weight,
            ]

            cursor["TrainParams"]["tileSize"] = TILE_SIZE
            cursor["TrainParams"]["maxGaussiansPerTile"] = self.max_gaussians_per_tile
            # Adam iter is not used as training iter because I said so
            cursor["TrainParams"]["adamIteration"] = self.adam_iteration

            if train_config.use_adam:
                cursor["gAdamFirstMoment"] = self.adam_first_moment
                cursor["gAdamSecondMoment"] = self.adam_second_moment
                cursor["TrainParams"]["adamBeta1"] = train_config.adam_beta1
                cursor["TrainParams"]["adamBeta2"] = train_config.adam_beta2
                cursor["TrainParams"]["adamEpsilon"] = train_config.adam_epsilon
            
            self._bind_sgld_params(cursor, train_config)

            cp.dispatch(thread_count=(self.gaussian_count, 1, 1))

        if self.use_gaussian_volume or train_config.ssim_weight > 0.0:
            self.rasterize_gaussians(cmd, vol_min, vol_max)

        # Pass 7: Debug visualization
        if self.debug_needs_update and self.debug_mode > 0:
            self.update_debug_visualization(cmd, vol_min, vol_max)

        self.device.submit_command_buffer(cmd.finish())
        self._needs_rebinning = True
        self._needs_rasterization = True
        self._cached_light_dir = None  # invalidate light cache after training step
        self.debug_needs_update = True

        # SGD uses this as welll
        self.adam_iteration += 1

    def resize(self, w, h):
        if w == self.width and h == self.height:
            return
        self.width, self.height = w, h

        self.screen_tex = self.device.create_texture(
            format=spy.Format.rgba8_unorm,
            width=w,
            height=h,
            usage=spy.TextureUsage.render_target | spy.TextureUsage.shader_resource,
            label="Screen",
        )

        if EXTENDED_UI:
            if self.display_gl_tex is None:
                self.display_gl_tex = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self.display_gl_tex)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexImage2D(
                GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, None
            )

    def blit_to_surface(self, cmd, surface_tex):
        """Blit the rendered output to a swapchain surface texture (slim mode)."""
        src = self.splat_output_tex if self.use_splatting else self.screen_tex
        if src is not None:
            cmd.blit(surface_tex, src)

    def check_hot_reload(self):
        try:
            curr_time = Path(SHADER_FILE).stat().st_mtime
            if curr_time > self.last_mod_time:
                self.last_mod_time = curr_time

                prog = self.device.load_program(
                    SHADER_FILE, ["vertex_main", "fragment_main"]
                )
                new_pipe = self.device.create_render_pipeline(
                    program=prog,
                    input_layout=self.device.create_input_layout(
                        input_elements=[], vertex_streams=[]
                    ),
                    targets=[{"format": spy.Format.rgba8_unorm}],
                )

                self.pipeline = new_pipe
                self.error_msg = ""
                print("Shader Reloaded!")
        except Exception as e:
            self.error_msg = str(e)
            print("Shader Compile Error (Safe)")

    def render(self, camera, settings, ext_cmd=None):
        """Render the scene. If ext_cmd is provided, record commands onto it
        instead of creating/submitting our own (used by slim app to avoid
        swapchain semaphore conflicts)."""
        if self.debug_mode > 0:
            self.render_debug(camera)
        elif self.use_splatting:
            self.render_splat(camera, settings, ext_cmd=ext_cmd)
        else:
            self.render_main(camera, settings, ext_cmd=ext_cmd)

    def render_debug(self, camera):
        """Debug visualization using raymarcher"""
        self.cam_buffer.copy_from_numpy(camera.get_gpu_data(self.width / self.height))

        cmd = self.device.create_command_encoder()
        cmd.set_texture_state(self.debug_tex, spy.ResourceState.shader_resource)

        with cmd.begin_render_pass(
            {"color_attachments": [{"view": self.screen_tex.create_view({})}]}
        ) as rp:
            rp.bind_pipeline(self.debug_pipeline)
            cursor = spy.ShaderCursor(rp.bind_pipeline(self.debug_pipeline))

            cursor["debugVolume"] = self.debug_tex
            cursor["camera"] = self.cam_buffer
            cursor["linearSampler"] = self.linear_sampler

            rp.set_render_state(
                {
                    "viewports": [spy.Viewport.from_size(self.width, self.height)],
                    "scissor_rects": [
                        spy.ScissorRect.from_size(self.width, self.height)
                    ],
                }
            )
            rp.draw({"vertex_count": 3})

        self.device.submit_command_buffer(cmd.finish())

    def render_main(self, camera, settings, ext_cmd=None):
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

        own_cmd = ext_cmd is None
        cmd = self.device.create_command_encoder() if own_cmd else ext_cmd
        cmd.set_texture_state(self.volume_tex, spy.ResourceState.shader_resource)

        with cmd.begin_render_pass(
            {"color_attachments": [{"view": self.screen_tex.create_view({})}]}
        ) as rp:
            rp.bind_pipeline(self.pipeline)
            cursor = spy.ShaderCursor(rp.bind_pipeline(self.pipeline))

            cursor["inVolume"] = (
                self.gaussian_volume_tex
                if self.use_gaussian_volume
                else self.volume_tex
            )
            cursor["camera"] = self.cam_buffer
            cursor["linearSampler"] = self.linear_sampler
            cursor["settings"] = self.settings_buffer

            rp.set_render_state(
                {
                    "viewports": [spy.Viewport.from_size(self.width, self.height)],
                    "scissor_rects": [
                        spy.ScissorRect.from_size(self.width, self.height)
                    ],
                }
            )
            rp.draw({"vertex_count": 3})

        if own_cmd:
            self.device.submit_command_buffer(cmd.finish())

    def update_display(self):
        pixels = self.screen_tex.to_numpy()  # already RGBA8, no clip/cast needed
        glBindTexture(GL_TEXTURE_2D, self.display_gl_tex)
        glTexSubImage2D(
            GL_TEXTURE_2D,
            0,
            0,
            0,
            self.width,
            self.height,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            pixels,
        )

        fb = glGenFramebuffers(1)
        glBindFramebuffer(GL_READ_FRAMEBUFFER, fb)
        glFramebufferTexture2D(
            GL_READ_FRAMEBUFFER,
            GL_COLOR_ATTACHMENT0,
            GL_TEXTURE_2D,
            self.display_gl_tex,
            0,
        )
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
        glBlitFramebuffer(
            0,
            0,
            self.width,
            self.height,
            0,
            0,
            self.width,
            self.height,
            GL_COLOR_BUFFER_BIT,
            GL_NEAREST,
        )
        glDeleteFramebuffers(1, [fb])

    def rasterize_gaussians(self, cmd, vol_min, vol_max):

        cmd.clear_texture_float(self.gaussian_volume_tex)
        cmd.set_texture_state(
            self.gaussian_volume_tex, spy.ResourceState.unordered_access
        )

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
                cursor["TrainParams"]["maxGaussiansPerTile"] = self.max_gaussians_per_tile
                cp.dispatch(thread_count=(total_tiles, 1, 1))

            # 3. Bin Gaussians
            with cmd.begin_compute_pass() as cp:
                root_object = cp.bind_pipeline(self.pipe_bin)
                cursor = spy.ShaderCursor(root_object)

                cursor["gGaussianParamsRaw"] = self.gaussian_buffer
                cursor["gTileContent"] = self.tile_content
                cursor["gTileCounts"] = self.tile_counts
                cursor["TrainParams"]["volumeResolution"] = (
                    VOL_SIZE,
                    VOL_SIZE,
                    VOL_SIZE,
                )
                cursor["TrainParams"]["tileResolution"] = self.tile_res
                cursor["TrainParams"]["gaussianCount"] = self.gaussian_count
                cursor["TrainParams"]["minWorld"] = tuple(vol_min)
                cursor["TrainParams"]["maxWorld"] = tuple(vol_max)
                cursor["TrainParams"]["tileSize"] = TILE_SIZE
                cursor["TrainParams"]["maxGaussiansPerTile"] = self.max_gaussians_per_tile
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
            cursor["RasterParams"]["maxGaussiansPerTile"] = self.max_gaussians_per_tile
            cursor["RasterParams"]["volumeMinWorld"] = tuple(vol_min)
            cursor["RasterParams"]["volumeMaxWorld"] = tuple(vol_max)

            cp.dispatch(thread_count=(VOL_SIZE, VOL_SIZE, VOL_SIZE))

        self._needs_rasterization = False

    # ==== Splatting Renderer ====

    SPLAT_TILE_SIZE = 16
    MAX_GAUSSIANS_PER_SCREEN_TILE = 2048

    def _init_splatting_pipelines(self):
        self.pipe_splat_project = self.device.create_compute_pipeline(
            program=self.device.load_program("shaders/splatting.slang", ["project_gaussians"])
        )
        self.pipe_splat_init_sort_light = self.device.create_compute_pipeline(
            program=self.device.load_program("shaders/splatting.slang", ["init_sort_keys_light"])
        )
        self.pipe_splat_local_sort = self.device.create_compute_pipeline(
            program=self.device.load_program("shaders/splatting.slang", ["local_bitonic_sort"])
        )
        self.pipe_splat_global_merge = self.device.create_compute_pipeline(
            program=self.device.load_program("shaders/splatting.slang", ["global_bitonic_merge"])
        )
        self.pipe_splat_local_merge_finish = self.device.create_compute_pipeline(
            program=self.device.load_program("shaders/splatting.slang", ["local_merge_finish"])
        )
        self.pipe_splat_compute_tau = self.device.create_compute_pipeline(
            program=self.device.load_program("shaders/splatting.slang", ["compute_gaussian_tau"])
        )
        self.pipe_splat_scan_transmittance = self.device.create_compute_pipeline(
            program=self.device.load_program("shaders/splatting.slang", ["scan_transmittance"])
        )
        self.pipe_splat_clear_tiles = self.device.create_compute_pipeline(
            program=self.device.load_program("shaders/splatting.slang", ["clear_screen_tiles"])
        )
        self.pipe_splat_tile_assign = self.device.create_compute_pipeline(
            program=self.device.load_program("shaders/splatting.slang", ["tile_assign_screen"])
        )
        self.pipe_splat_render = self.device.create_compute_pipeline(
            program=self.device.load_program("shaders/splatting.slang", ["render_splats"])
        )
        self.pipe_splat_blur = self.device.create_compute_pipeline(
            program=self.device.load_program("shaders/splatting.slang", ["blur_splats"])
        )

        self.splat_output_tex = None
        self.splat_projected_buf = None
        self.splat_sorted_indices_buf = None
        self.splat_sort_keys_buf = None
        self.splat_tile_content_buf = None
        self.splat_tile_counts_buf = None
        self.splat_light_transmittance_buf = None
        self.splat_blur_tex = None
        self.splat_tile_res = (0, 0)
        self._splat_sort_count = 0
        self._cached_light_dir = None
        self._cached_light_pen = None
        self._cached_density_scale = None
        self._cached_shadow_strength = None
        self._cached_gaussian_count = None

    def _ensure_splat_buffers(self):
        """Create/recreate splatting buffers when gaussian count or screen size changes."""
        if self.gaussian_count == 0:
            return

        # Projected gaussians buffer — flat float array (11 floats per gaussian)
        proj_floats = self.gaussian_count * 11
        if self.splat_projected_buf is None or self.splat_projected_buf.size < proj_floats * 4:
            self.splat_projected_buf = self.device.create_buffer(
                element_count=proj_floats,
                struct_size=4,
                usage=spy.BufferUsage.unordered_access | spy.BufferUsage.shader_resource,
                memory_type=spy.MemoryType.device_local,
            )

        # Sort buffers — padded to next power of 2 for bitonic sort
        sort_count = 1
        while sort_count < self.gaussian_count:
            sort_count <<= 1
        if self._splat_sort_count != sort_count:
            self._splat_sort_count = sort_count
            self.splat_sorted_indices_buf = self.device.create_buffer(
                element_count=sort_count,
                struct_size=4,
                usage=spy.BufferUsage.unordered_access | spy.BufferUsage.shader_resource,
                memory_type=spy.MemoryType.device_local,
            )
            self.splat_sort_keys_buf = self.device.create_buffer(
                element_count=sort_count,
                struct_size=4,
                usage=spy.BufferUsage.unordered_access | spy.BufferUsage.shader_resource,
                memory_type=spy.MemoryType.device_local,
            )
            self.splat_light_transmittance_buf = self.device.create_buffer(
                element_count=self.gaussian_count,
                struct_size=4,
                usage=spy.BufferUsage.unordered_access | spy.BufferUsage.shader_resource,
                memory_type=spy.MemoryType.device_local,
            )

        # Screen tile buffers
        tile_res_x = (self.width + self.SPLAT_TILE_SIZE - 1) // self.SPLAT_TILE_SIZE
        tile_res_y = (self.height + self.SPLAT_TILE_SIZE - 1) // self.SPLAT_TILE_SIZE
        total_tiles = tile_res_x * tile_res_y

        if self.splat_tile_res != (tile_res_x, tile_res_y):
            self.splat_tile_res = (tile_res_x, tile_res_y)
            self.splat_tile_content_buf = self.device.create_buffer(
                element_count=total_tiles * self.MAX_GAUSSIANS_PER_SCREEN_TILE,
                struct_size=4,
                usage=spy.BufferUsage.unordered_access | spy.BufferUsage.shader_resource,
                memory_type=spy.MemoryType.device_local,
            )
            self.splat_tile_counts_buf = self.device.create_buffer(
                element_count=total_tiles,
                struct_size=4,
                usage=spy.BufferUsage.unordered_access | spy.BufferUsage.shader_resource,
                memory_type=spy.MemoryType.device_local,
            )

        # Output texture
        if self.splat_output_tex is None or self.splat_output_tex.width != self.width or self.splat_output_tex.height != self.height:
            self.splat_output_tex = self.device.create_texture(
                format=spy.Format.rgba8_unorm,
                width=self.width,
                height=self.height,
                usage=spy.TextureUsage.unordered_access | spy.TextureUsage.shader_resource,
                label="SplatOutput",
            )
            self.splat_blur_tex = self.device.create_texture(
                format=spy.Format.rgba8_unorm,
                width=self.width,
                height=self.height,
                usage=spy.TextureUsage.unordered_access | spy.TextureUsage.shader_resource,
                label="SplatBlurTemp",
            )

    def _dispatch_sort(self, cmd, sort_count, num_sort_groups, LOCAL_SORT_SIZE):
        """Dispatch local bitonic sort + global merge passes (reused for light and camera sorts)."""
        with cmd.begin_compute_pass() as cp:
            cursor = spy.ShaderCursor(cp.bind_pipeline(self.pipe_splat_local_sort))
            cursor["gSortKeys"] = self.splat_sort_keys_buf
            cursor["gSortedIndices"] = self.splat_sorted_indices_buf
            cursor["SortParams"]["sortCount"] = sort_count
            cursor["SortParams"]["blockSize"] = 0
            cursor["SortParams"]["subBlockSize"] = 0
            cp.dispatch(thread_count=(num_sort_groups * 1024, 1, 1))

        if sort_count > LOCAL_SORT_SIZE:
            block_size = LOCAL_SORT_SIZE * 2
            while block_size <= sort_count:
                sub_block = block_size >> 1
                while sub_block >= LOCAL_SORT_SIZE:
                    with cmd.begin_compute_pass() as cp:
                        cursor = spy.ShaderCursor(cp.bind_pipeline(self.pipe_splat_global_merge))
                        cursor["gSortKeys"] = self.splat_sort_keys_buf
                        cursor["gSortedIndices"] = self.splat_sorted_indices_buf
                        cursor["SortParams"]["sortCount"] = sort_count
                        cursor["SortParams"]["blockSize"] = block_size
                        cursor["SortParams"]["subBlockSize"] = sub_block
                        cp.dispatch(thread_count=(sort_count, 1, 1))
                    sub_block >>= 1

                with cmd.begin_compute_pass() as cp:
                    cursor = spy.ShaderCursor(cp.bind_pipeline(self.pipe_splat_local_merge_finish))
                    cursor["gSortKeys"] = self.splat_sort_keys_buf
                    cursor["gSortedIndices"] = self.splat_sorted_indices_buf
                    cursor["SortParams"]["sortCount"] = sort_count
                    cursor["SortParams"]["blockSize"] = block_size
                    cursor["SortParams"]["subBlockSize"] = 0
                    cp.dispatch(thread_count=(num_sort_groups * 1024, 1, 1))

                block_size <<= 1

    def render_splat(self, camera, settings, ext_cmd=None):
        """Render gaussians via volumetric splatting — fully GPU, no CPU readback."""
        if self.gaussian_count == 0:
            return

        self._ensure_splat_buffers()

        aspect = self.width / self.height
        cam_data = camera.get_gpu_data(aspect)
        self.cam_buffer.copy_from_numpy(cam_data)

        # Pack settings for splatting (reuse same layout as raymarcher)
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

        tile_res = self.splat_tile_res
        total_tiles = tile_res[0] * tile_res[1]
        sort_count = self._splat_sort_count

        vol_min = tuple(self._vol_min)
        vol_max = tuple(self._vol_max)

        own_cmd = ext_cmd is None
        cmd = self.device.create_command_encoder() if own_cmd else ext_cmd

        # Pass 1: Project gaussians + init camera sort keys
        with cmd.begin_compute_pass() as cp:
            cursor = spy.ShaderCursor(cp.bind_pipeline(self.pipe_splat_project))
            cursor["gGaussians"] = self.gaussian_buffer
            cursor["gCamera"] = self.cam_buffer
            cursor["gSettings"] = self.settings_buffer
            cursor["gProjected"] = self.splat_projected_buf
            cursor["gSortKeys"] = self.splat_sort_keys_buf
            cursor["gSortedIndices"] = self.splat_sorted_indices_buf
            cursor["SplatParams"]["gaussianCount"] = self.gaussian_count
            cursor["SplatParams"]["paddedSortCount"] = sort_count
            cursor["SplatParams"]["screenWidth"] = self.width
            cursor["SplatParams"]["screenHeight"] = self.height
            cursor["SplatParams"]["paddedSortCount"] = sort_count
            cursor["SplatParams"]["tileResolution"] = tile_res
            cursor["SplatParams"]["densityScale"] = settings.density_scale
            cursor["SplatParams"]["lightPenetration"] = settings.light_penetration
            cursor["SplatParams"]["shadowStrength"] = settings.shadow_strength
            cursor["SplatParams"]["splatSoftness"] = settings.splat_softness
            cursor["SplatParams"]["useDepthDarkening"] = 1 if settings.use_depth_darkening else 0
            cursor["SplatParams"]["blurRadius"] = settings.blur_radius
            cursor["SplatParams"]["volumeMin"] = vol_min
            cursor["SplatParams"]["volumeMax"] = vol_max
            cp.dispatch(thread_count=(sort_count, 1, 1))

        LOCAL_SORT_SIZE = 4096
        num_sort_groups = (sort_count + LOCAL_SORT_SIZE - 1) // LOCAL_SORT_SIZE
        sun_dir = np.array(settings.get_sun_dir(), dtype=np.float32)
        sun_dir_norm = sun_dir / (np.linalg.norm(sun_dir) + 1e-8)
        light_dir = tuple(sun_dir_norm.astype(np.float32))

        # === Light-direction sort + transmittance sweep (cached) ===
        light_dirty = (
            self._cached_light_dir != light_dir
            or self._cached_light_pen != settings.light_penetration
            or self._cached_density_scale != settings.density_scale
            or self._cached_shadow_strength != settings.shadow_strength
            or self._cached_gaussian_count != self.gaussian_count
        )

        if light_dirty:
            # Init sort keys from light direction
            with cmd.begin_compute_pass() as cp:
                cursor = spy.ShaderCursor(cp.bind_pipeline(self.pipe_splat_init_sort_light))
                cursor["gGaussians"] = self.gaussian_buffer
                cursor["gSortKeys"] = self.splat_sort_keys_buf
                cursor["gSortedIndices"] = self.splat_sorted_indices_buf
                cursor["SplatParams"]["gaussianCount"] = self.gaussian_count
                cursor["SplatParams"]["paddedSortCount"] = sort_count
                cursor["SplatParams"]["volumeMin"] = vol_min
                cursor["SplatParams"]["volumeMax"] = vol_max
                cursor["SortParams"]["sortCount"] = sort_count
                cursor["SortParams"]["lightDirX"] = light_dir[0]
                cursor["SortParams"]["lightDirY"] = light_dir[1]
                cursor["SortParams"]["lightDirZ"] = light_dir[2]
                cp.dispatch(thread_count=(sort_count, 1, 1))

            # Local bitonic sort (light order)
            self._dispatch_sort(cmd, sort_count, num_sort_groups, LOCAL_SORT_SIZE)

            # Parallel: compute per-gaussian tau (one thread per gaussian)
            with cmd.begin_compute_pass() as cp:
                cursor = spy.ShaderCursor(cp.bind_pipeline(self.pipe_splat_compute_tau))
                cursor["gGaussians"] = self.gaussian_buffer
                cursor["gLightTransmittance"] = self.splat_light_transmittance_buf
                cursor["SplatParams"]["gaussianCount"] = self.gaussian_count
                cursor["SplatParams"]["paddedSortCount"] = sort_count
                cursor["SplatParams"]["densityScale"] = settings.density_scale
                cursor["SplatParams"]["lightPenetration"] = settings.light_penetration
                cursor["SplatParams"]["shadowStrength"] = settings.shadow_strength
                cursor["SplatParams"]["splatSoftness"] = settings.splat_softness
                cursor["SplatParams"]["useDepthDarkening"] = 1 if settings.use_depth_darkening else 0
                cursor["SplatParams"]["blurRadius"] = settings.blur_radius
                cursor["SplatParams"]["volumeMin"] = vol_min
                cursor["SplatParams"]["volumeMax"] = vol_max
                cursor["SortParams"]["lightDirX"] = light_dir[0]
                cursor["SortParams"]["lightDirY"] = light_dir[1]
                cursor["SortParams"]["lightDirZ"] = light_dir[2]
                cp.dispatch(thread_count=(self.gaussian_count, 1, 1))

            # Serial scan: prefix sum of taus + exp (lightweight, no matrix math)
            with cmd.begin_compute_pass() as cp:
                cursor = spy.ShaderCursor(cp.bind_pipeline(self.pipe_splat_scan_transmittance))
                cursor["gSortedIndices"] = self.splat_sorted_indices_buf
                cursor["gLightTransmittance"] = self.splat_light_transmittance_buf
                cursor["SplatParams"]["gaussianCount"] = self.gaussian_count
                cursor["SplatParams"]["paddedSortCount"] = sort_count
                cp.dispatch(thread_count=(1, 1, 1))

            self._cached_light_dir = light_dir
            self._cached_light_pen = settings.light_penetration
            self._cached_density_scale = settings.density_scale
            self._cached_shadow_strength = settings.shadow_strength
            self._cached_gaussian_count = self.gaussian_count

        # === Camera-depth sort (keys already initialized in projection) ===

        # Local bitonic sort (camera order)
        self._dispatch_sort(cmd, sort_count, num_sort_groups, LOCAL_SORT_SIZE)

        # Clear screen tiles
        with cmd.begin_compute_pass() as cp:
            cursor = spy.ShaderCursor(cp.bind_pipeline(self.pipe_splat_clear_tiles))
            cursor["gScreenTileCounts"] = self.splat_tile_counts_buf
            cursor["SplatParams"]["gaussianCount"] = self.gaussian_count
            cursor["SplatParams"]["paddedSortCount"] = sort_count
            cursor["SplatParams"]["screenWidth"] = self.width
            cursor["SplatParams"]["screenHeight"] = self.height
            cursor["SplatParams"]["tileResolution"] = tile_res
            cursor["SplatParams"]["densityScale"] = settings.density_scale
            cursor["SplatParams"]["lightPenetration"] = settings.light_penetration
            cursor["SplatParams"]["shadowStrength"] = settings.shadow_strength
            cursor["SplatParams"]["splatSoftness"] = settings.splat_softness
            cursor["SplatParams"]["useDepthDarkening"] = 1 if settings.use_depth_darkening else 0
            cursor["SplatParams"]["blurRadius"] = settings.blur_radius
            cursor["SplatParams"]["volumeMin"] = vol_min
            cursor["SplatParams"]["volumeMax"] = vol_max
            cp.dispatch(thread_count=(total_tiles, 1, 1))

        # Pass 3: Tile assignment (sorted order)
        with cmd.begin_compute_pass() as cp:
            cursor = spy.ShaderCursor(cp.bind_pipeline(self.pipe_splat_tile_assign))
            cursor["gProjected"] = self.splat_projected_buf
            cursor["gSortedIndices"] = self.splat_sorted_indices_buf
            cursor["gScreenTileContent"] = self.splat_tile_content_buf
            cursor["gScreenTileCounts"] = self.splat_tile_counts_buf
            cursor["SplatParams"]["gaussianCount"] = self.gaussian_count
            cursor["SplatParams"]["paddedSortCount"] = sort_count
            cursor["SplatParams"]["screenWidth"] = self.width
            cursor["SplatParams"]["screenHeight"] = self.height
            cursor["SplatParams"]["tileResolution"] = tile_res
            cursor["SplatParams"]["densityScale"] = settings.density_scale
            cursor["SplatParams"]["lightPenetration"] = settings.light_penetration
            cursor["SplatParams"]["shadowStrength"] = settings.shadow_strength
            cursor["SplatParams"]["splatSoftness"] = settings.splat_softness
            cursor["SplatParams"]["useDepthDarkening"] = 1 if settings.use_depth_darkening else 0
            cursor["SplatParams"]["blurRadius"] = settings.blur_radius
            cursor["SplatParams"]["volumeMin"] = vol_min
            cursor["SplatParams"]["volumeMax"] = vol_max
            cp.dispatch(thread_count=(self.gaussian_count, 1, 1))

        # Pass 4: Render splats
        cmd.clear_texture_float(self.splat_output_tex)
        cmd.set_texture_state(self.splat_output_tex, spy.ResourceState.unordered_access)
        with cmd.begin_compute_pass() as cp:
            cursor = spy.ShaderCursor(cp.bind_pipeline(self.pipe_splat_render))
            cursor["gProjected"] = self.splat_projected_buf
            cursor["gScreenTileContent"] = self.splat_tile_content_buf
            cursor["gScreenTileCounts"] = self.splat_tile_counts_buf
            cursor["gCamera"] = self.cam_buffer
            cursor["gSettings"] = self.settings_buffer
            cursor["gOutputTexture"] = self.splat_output_tex
            cursor["gLightTransmittance"] = self.splat_light_transmittance_buf
            cursor["SplatParams"]["gaussianCount"] = self.gaussian_count
            cursor["SplatParams"]["paddedSortCount"] = sort_count
            cursor["SplatParams"]["screenWidth"] = self.width
            cursor["SplatParams"]["screenHeight"] = self.height
            cursor["SplatParams"]["tileResolution"] = tile_res
            cursor["SplatParams"]["densityScale"] = settings.density_scale
            cursor["SplatParams"]["lightPenetration"] = settings.light_penetration
            cursor["SplatParams"]["shadowStrength"] = settings.shadow_strength
            cursor["SplatParams"]["splatSoftness"] = settings.splat_softness
            cursor["SplatParams"]["useDepthDarkening"] = 1 if settings.use_depth_darkening else 0
            cursor["SplatParams"]["blurRadius"] = settings.blur_radius
            cursor["SplatParams"]["volumeMin"] = vol_min
            cursor["SplatParams"]["volumeMax"] = vol_max
            cp.dispatch(thread_count=(self.width, self.height, 1))

        # Post-process blur (if enabled)
        blur_radius = settings.blur_radius if settings.enable_blur else 0
        if blur_radius > 0:
            cmd.set_texture_state(self.splat_output_tex, spy.ResourceState.unordered_access)
            cmd.set_texture_state(self.splat_blur_tex, spy.ResourceState.unordered_access)
            with cmd.begin_compute_pass() as cp:
                cursor = spy.ShaderCursor(cp.bind_pipeline(self.pipe_splat_blur))
                cursor["gOutputTexture"] = self.splat_output_tex
                cursor["gBlurOutput"] = self.splat_blur_tex
                cursor["SplatParams"]["screenWidth"] = self.width
                cursor["SplatParams"]["screenHeight"] = self.height
                cursor["SplatParams"]["blurRadius"] = blur_radius
                cp.dispatch(thread_count=(self.width, self.height, 1))
            # Swap: blur result becomes the output
            self.splat_output_tex, self.splat_blur_tex = self.splat_blur_tex, self.splat_output_tex

        if own_cmd:
            self.device.submit_command_buffer(cmd.finish())

    def update_display_splat(self):
        """Read splatting output and upload to GL for display."""
        pixels = self.splat_output_tex.to_numpy()
        glBindTexture(GL_TEXTURE_2D, self.display_gl_tex)
        glTexSubImage2D(
            GL_TEXTURE_2D, 0, 0, 0,
            self.width, self.height,
            GL_RGBA, GL_UNSIGNED_BYTE, pixels,
        )

        fb = glGenFramebuffers(1)
        glBindFramebuffer(GL_READ_FRAMEBUFFER, fb)
        glFramebufferTexture2D(
            GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
            GL_TEXTURE_2D, self.display_gl_tex, 0,
        )
        glBlitFramebuffer(
            0, 0, self.width, self.height,
            0, 0, self.width, self.height,
            GL_COLOR_BUFFER_BIT, GL_NEAREST,
        )
        glBindFramebuffer(GL_READ_FRAMEBUFFER, 0)
        glDeleteFramebuffers(1, [fb])


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
        self.window = glfw.create_window(
            self.width, self.height, WINDOW_TITLE, None, None
        )
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
        self.device = spy.Device(
            enable_debug_layers=True,
            compiler_options={"include_paths": [example_dir]},
            type=spy.DeviceType.vulkan,
        )

        global VOL_SIZE
        self.grid = load_vdb_grid(VDB_FILE)
        self.vdb_up_axis = VDB_UP_AXIS
        self.vol_min_world, self.vol_max_world, vol_data, self.axis_remap, resolved_size = convert_grid_to_dense_volume(
            self.grid, VOL_SIZE, up_axis_name=self.vdb_up_axis,
            use_native_size=USE_NATIVE_VDB_SIZE,
        )
        VOL_SIZE = resolved_size
        self.gaussians = convert_grid_to_gaussians(self.grid, self.train_config, self.axis_remap)

        self.renderer = Renderer(self.device, vol_data)
        self.renderer.resize(self.width, self.height)
        self.renderer.gaussian_count = len(self.gaussians)
        self.renderer.gaussian_buffer = self.device.create_buffer(
            element_count=self.renderer.gaussian_count,
            struct_size=PARAMS_PER_GAUSSIAN * 4,
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            memory_type=spy.MemoryType.device_local,
            data=self.gaussians,
        )
        self.renderer._vol_min = self.vol_min_world
        self.renderer._vol_max = self.vol_max_world
        self.renderer.init_training()

        cmd = self.device.create_command_encoder()
        self.renderer.rasterize_gaussians(cmd, self.vol_min_world, self.vol_max_world)
        self.device.submit_command_buffer(cmd.finish())

        self.adc = ADCController(self, ADCConfig())
        self.adc.train_config = self.train_config

        self.ply_lighting = CloudLightingConfig()

        self.metrics = MetricsCollector(
            self.device, self.renderer, (VOL_SIZE, VOL_SIZE, VOL_SIZE)
        )
        self.screen_metrics = ScreenMetricsCollector(
            self.device, self.renderer, self.settings
        )

        self.sgld_diag = SGLDDiagnostics(self)
        self.train_step = 0
    

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
                old_m1 = (
                    self.renderer.adam_first_moment.to_numpy()
                    .view(np.float32)
                    .reshape(-1, PARAMS_PER_GAUSSIAN)
                    .copy()
                )
                old_m2 = (
                    self.renderer.adam_second_moment.to_numpy()
                    .view(np.float32)
                    .reshape(-1, PARAMS_PER_GAUSSIAN)
                    .copy()
                )
            except Exception as e:
                print(f"[ADC] Could not read momentum buffers: {e}")
                old_m1 = old_m2 = None
        else:
            old_m1 = old_m2 = None

        self.gaussians = new_params
        self.renderer.gaussian_count = len(new_params)
        self.renderer.gaussian_buffer = self.device.create_buffer(
            element_count=len(new_params),
            struct_size=PARAMS_PER_GAUSSIAN * 4,
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            memory_type=spy.MemoryType.device_local,
            data=flat_params,
        )
        self.renderer.init_training()
        self.renderer.adam_iteration = saved_iteration

        if surviving_indices is not None and old_m1 is not None:
            try:
                n_new = len(new_params)
                new_m1 = np.zeros((n_new, PARAMS_PER_GAUSSIAN), dtype=np.float32)
                new_m2 = np.zeros((n_new, PARAMS_PER_GAUSSIAN), dtype=np.float32)

                # Only transfer momentum for true survivors (index != -1)
                # surviving_indices[i] = source row in old buffers, or -1 for new spawns
                for i, src in enumerate(surviving_indices):
                    if i >= n_new:
                        break
                    if src >= 0 and src < len(old_m1):
                        new_m1[i] = old_m1[src]
                        new_m2[i] = old_m2[src]
                    # src == -1 (split child or clone): leave as zeros — fresh Adam state

                self.renderer.adam_first_moment.copy_from_numpy(new_m1.flatten())
                self.renderer.adam_second_moment.copy_from_numpy(new_m2.flatten())
                n_restored = int(np.sum(np.array(surviving_indices[:n_new]) >= 0))
                print(
                    f"[ADC] Momentum restored for {n_restored} survivors "
                    f"({n_new - n_restored} new spawns start fresh)"
                )
            except Exception as e:
                print(f"[ADC] Momentum restore failed (non-fatal): {e}")

        self.renderer._needs_rebinning = True
        self.renderer._needs_rasterization = True
        print(
            f"[ADC] Buffer rebuilt: {len(new_params):,} Gaussians (Adam iter {saved_iteration})"
        )

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
            imgui.dock_space_over_viewport(
                0, imgui.get_main_viewport(), imgui.DockNodeFlags_.passthru_central_node
            )

            self.draw_ui(dt)
            self.handle_input(dt)

            if self.is_training:
                self.renderer.run_training_step(
                    self.vol_min_world, self.vol_max_world, self.train_config
                )
                self.train_step += 1

                if self.train_step == 100:
                    self.sgld_diag.snapshot()
                if self.train_step % 500 == 0 and self.train_step > 100:
                    self.sgld_diag.tick()

                if self.train_step % 100 == 0:
                    tile_debug = self.renderer.tile_counts.to_numpy().view(dtype=np.uint32)
                    total_refs = np.sum(tile_debug)
                    grad_bytes = self.renderer.grad_buffer.to_numpy()
                    grad_all = grad_bytes.view(dtype=np.float32).reshape(-1, PARAMS_PER_GAUSSIAN)
                    weight_grads = grad_all[:, 10]
                    grad_max = np.max(np.abs(grad_all))
                    nonzero = np.count_nonzero(weight_grads)
                    if total_refs == 0:
                        print(f"[CRITICAL] Tiler is empty! (Total Refs: {total_refs})")
                    elif grad_max == 0:
                        print(f"[ALERT] Tiler works ({total_refs} refs) but ALL gradients are ZERO.")
                    else:
                        print(f"[OK] Training Running. Refs: {total_refs}, GradMax: {grad_max:.6f}, ActiveWeightGrads: {nonzero}/{len(weight_grads)}")
                    maxed_tiles = np.sum(tile_debug >= self.renderer.max_gaussians_per_tile)
                    if maxed_tiles > 0:
                        print(f"[WARN] {maxed_tiles} tiles at max_gaussians_per_tile ({self.renderer.max_gaussians_per_tile}) cap!")

                if self.train_step % 50 == 0:
                    self.renderer.analyze_gradients()
                    loss = self.compute_current_loss()
                    self.renderer.loss_history.append(loss)
                    if len(self.renderer.loss_history) > 200:
                        self.renderer.loss_history.pop(0)

            self.adc.tick(frame_count, self.is_training)
            self.adc.apply_pending()

            # Collect regular (3D) metrics and screen space ones
            self.metrics.tick(
                frame_count, self.vol_min_world, self.vol_max_world, self.is_training
            )
            self.screen_metrics.tick(frame_count, self.is_training)
            if self.screen_metrics._wants_snapshot:
                self.screen_metrics.snapshot_live_camera(self.camera)
                self.screen_metrics._wants_snapshot = False

            if not self.is_training:
                needs_compute = (
                    self.renderer.debug_needs_update and self.renderer.debug_mode > 0
                ) or (
                    self.renderer._needs_rasterization
                    and self.renderer.use_gaussian_volume
                )
                if needs_compute:
                    cmd = self.device.create_command_encoder()
                    if (
                        self.renderer._needs_rasterization
                        and self.renderer.use_gaussian_volume
                    ):
                        self.renderer.rasterize_gaussians(
                            cmd, self.vol_min_world, self.vol_max_world
                        )
                    if (
                        self.renderer.debug_needs_update
                        and self.renderer.debug_mode > 0
                    ):
                        self.renderer.update_debug_visualization(
                            cmd, self.vol_min_world, self.vol_max_world
                        )
                    self.device.submit_command_buffer(cmd.finish())

            self.renderer.render(self.camera, self.settings)
            if self.renderer.use_splatting:
                self.renderer.update_display_splat()
            else:
                self.renderer.update_display()

            frame_count += 1

            imgui.render()
            imgui.backends.opengl3_render_draw_data(imgui.get_draw_data())
            glfw.swap_buffers(self.window)

        self.cleanup()

    def handle_input(self, dt):
        if self.io.want_capture_mouse:
            return

        right_down = (
            glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        )
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
        if glfw.get_key(self.window, glfw.KEY_W) == glfw.PRESS:
            self.camera.pos += self.camera.front * speed
        if glfw.get_key(self.window, glfw.KEY_S) == glfw.PRESS:
            self.camera.pos -= self.camera.front * speed
        if glfw.get_key(self.window, glfw.KEY_A) == glfw.PRESS:
            self.camera.pos -= self.camera.right * speed
        if glfw.get_key(self.window, glfw.KEY_D) == glfw.PRESS:
            self.camera.pos += self.camera.right * speed
        if glfw.get_key(self.window, glfw.KEY_Q) == glfw.PRESS:
            self.camera.pos += self.camera.up * speed
        if glfw.get_key(self.window, glfw.KEY_E) == glfw.PRESS:
            self.camera.pos -= self.camera.up * speed
        if glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(self.window, True)

    def draw_ui(self, dt):
        if imgui.begin("Stats"):
            imgui.text(f"FPS: {1.0/(dt+0.0001):.1f}")
            if self.renderer.error_msg:
                imgui.text_colored(imgui.ImVec4(1, 0, 0, 1), "SHADER ERROR")
        imgui.end()

        if imgui.begin("Rendering"):
            if self.renderer.error_msg:
                imgui.text_colored(
                    imgui.ImVec4(1, 0, 0, 1), f"{self.renderer.error_msg}"
                )
            else:
                imgui.text_colored(imgui.ImVec4(0, 1, 0, 1), "Shader Active")

            imgui.text("Raymarcher")
            imgui.separator()
            _, self.settings.density_scale = imgui.slider_float(
                "Density", self.settings.density_scale, 1.0, 200.0
            )
            _, self.settings.density_curve = imgui.slider_float(
                "Gamma", self.settings.density_curve, 0.1, 2.0
            )
            _, self.settings.step_size = imgui.slider_float(
                "Step Size", self.settings.step_size, 0.001, 0.05
            )
            _, self.settings.step_count = imgui.slider_int(
                "Max Steps", self.settings.step_count, 10, 500
            )

            imgui.dummy((0, 10))
            imgui.text("Lighting")
            imgui.separator()

            _, self.settings.sun_direction = imgui.slider_float3(
                "Sun Direction", self.settings.sun_direction, -1.0, 1.0
            )
            _, self.settings.sun_intensity = imgui.slider_float(
                "Sun Intensity", self.settings.sun_intensity, 0.0, 20.0
            )
            _, self.settings.sun_color_base = imgui.color_edit3(
                "Sun Color", self.settings.sun_color_base
            )

            _, self.settings.ambient_intensity = imgui.slider_float(
                "Ambient Intensity", self.settings.ambient_intensity, 0.0, 2.0
            )
            _, self.settings.ambient_color_base = imgui.color_edit3(
                "Ambient Color", self.settings.ambient_color_base
            )

            imgui.dummy((0, 10))
            imgui.text("Volumetric Look")
            imgui.separator()

            _, self.settings.light_penetration = imgui.slider_float(
                "Fluffiness", self.settings.light_penetration, 0.01, 1.0
            )
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Low = Light penetrates deep (fluffy clouds)\nHigh = Light blocked early (dark smoke)"
                )

            _, self.settings.shadow_strength = imgui.slider_float(
                "Shadow Strength", self.settings.shadow_strength, 0.0, 20.0
            )
            _, self.settings.splat_softness = imgui.slider_float(
                "Splat Softness", self.settings.splat_softness, 0.1, 10.0
            )
            _, self.settings.enable_blur = imgui.checkbox(
                "Enable Blur", self.settings.enable_blur
            )
            _, self.settings.blur_radius = imgui.slider_int(
                "Blur Radius", self.settings.blur_radius, 1, 4
            )
            _, self.settings.use_depth_darkening = imgui.checkbox(
                "Depth Darkening", self.settings.use_depth_darkening
            )

            _, self.settings.phase_g = imgui.slider_float(
                "Phase G (Silver Lining)", self.settings.phase_g, -0.9, 0.9
            )

            _, self.settings.shadow_steps = imgui.slider_int(
                "Shadow Steps", self.settings.shadow_steps, 1, 16
            )
            _, self.settings.shadow_step_mult = imgui.slider_float(
                "Shadow Step Mult", self.settings.shadow_step_mult, 1.0, 10.0
            )

            imgui.dummy((0, 10))
            _, self.settings.smoke_color = imgui.color_edit3(
                "Smoke Albedo", self.settings.smoke_color
            )

            _, self.renderer.use_gaussian_volume = imgui.checkbox(
                "Render Gaussian Volume", self.renderer.use_gaussian_volume
            )
            _, self.renderer.use_splatting = imgui.checkbox(
                "Splatting Mode (3DGS)", self.renderer.use_splatting
            )
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Render via volumetric gaussian splatting instead of raymarching.\n"
                    "Uses closed-form ray integral for physically-based volumetric opacity."
                )

        imgui.end()

        if imgui.begin("Training"):
            imgui.text("Optimizer")
            imgui.separator()

            _, self.train_config.use_adam = imgui.checkbox(
                "Use Adam Optimizer", self.train_config.use_adam
            )
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Adam: Adaptive learning with momentum (faster, smoother)\nSGD: Simple gradient descent"
                )
            
            sgld = self.train_config.sgld

            imgui.dummy((0, 10))
            imgui.text("SGLD Noise (Langevin Dynamics)")
            imgui.separator()

            _, sgld.enabled = imgui.checkbox("Enable SGLD Noise##sgld", sgld.enabled)
            imgui.same_line()
            if sgld.enabled:
                imgui.text_colored(imgui.ImVec4(0.4, 0.9, 0.4, 1.0), "ACTIVE")
            else:
                imgui.text_colored(imgui.ImVec4(0.5, 0.5, 0.5, 1.0), "off")

            if sgld.enabled:
                imgui.text(f"Current T: {sgld.current_temperature:.2e}")

                _, sgld.temperature_start = imgui.slider_float(
                    "T0 (Initial Temp)##sgld", sgld.temperature_start, 1e-7, 1e-2, format="%.2e"
                )
                _, sgld.temperature_decay = imgui.slider_float(
                    "Decay Rate##sgld", sgld.temperature_decay, 0.999, 0.99999, format="%.5f"
                )
                _, sgld.temperature_min = imgui.slider_float(
                    "T_min (Floor)##sgld", sgld.temperature_min, 1e-10, 1e-5, format="%.2e"
                )
                _, sgld.sgld_k = imgui.slider_float(
                    "Opacity Gate K##sgld", sgld.sgld_k, 1.0, 50.0, format="%.1f"
                )
                if imgui.is_item_hovered():
                    imgui.set_tooltip(
                        "Controls how strongly density suppresses noise.\n"
                        "exp(-weight * K): higher K = only very transparent\n"
                        "Gaussians swoosh. Lower K = even dense ones jitter.\n"
                        "Typical range: 5-20"
                    )

                t_progress = (sgld.current_temperature - sgld.temperature_min) / max(
                    sgld.temperature_start - sgld.temperature_min, 1e-12
                )
                imgui.progress_bar(t_progress, imgui.ImVec2(-1, 0), f"Temperature: {t_progress*100:.1f}%")
                
                if imgui.button("Reset Temperature##sgld"):
                    sgld.reset()

            imgui.dummy((0, 10))
            imgui.text("Learning Rates")
            imgui.separator()
            _, self.train_config.learning_rate_pos = imgui.slider_float(
                "Position",
                self.train_config.learning_rate_pos,
                0.0001,
                0.5,
                format="%.4f",
            )
            _, self.train_config.learning_rate_scale = imgui.slider_float(
                "Scale",
                self.train_config.learning_rate_scale,
                0.00001,
                0.1,
                format="%.5f",
            )
            _, self.train_config.learning_rate_rotation = imgui.slider_float(
                "Rotation",
                self.train_config.learning_rate_rotation,
                0.00001,
                0.1,
                format="%.5f",
            )
            _, self.train_config.learning_rate_weight = imgui.slider_float(
                "Weight",
                self.train_config.learning_rate_weight,
                0.001,
                0.5,
                format="%.4f",
            )

            if self.train_config.use_adam:
                imgui.dummy((0, 10))
                imgui.text("Adam Hyperparameters")
                imgui.separator()
                _, self.train_config.adam_beta1 = imgui.slider_float(
                    "Beta1 (Momentum)",
                    self.train_config.adam_beta1,
                    0.8,
                    0.999,
                    format="%.3f",
                )
                _, self.train_config.adam_beta2 = imgui.slider_float(
                    "Beta2 (Variance)",
                    self.train_config.adam_beta2,
                    0.9,
                    0.9999,
                    format="%.4f",
                )
                _, self.train_config.adam_epsilon = imgui.slider_float(
                    "Epsilon",
                    self.train_config.adam_epsilon,
                    1e-10,
                    1e-6,
                    format="%.2e",
                )

            imgui.text(f"Iteration: {self.renderer.adam_iteration}")

            imgui.dummy((0, 10))
            imgui.text("Gaussian Generation")
            imgui.separator()

            gauss_modes = ["percentage", "count"]
            cur_mode_idx = gauss_modes.index(self.train_config.gaussian_mode) \
                if self.train_config.gaussian_mode in gauss_modes else 0
            imgui.push_item_width(140)
            changed_gm, new_gm_idx = imgui.combo(
                "Init Mode##gm", cur_mode_idx, ["Percentage", "Target Count"]
            )
            imgui.pop_item_width()
            if changed_gm:
                self.train_config.gaussian_mode = gauss_modes[new_gm_idx]
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Percentage: user-specified spawn probability.\n"
                    "Target Count: auto-calculate probability to hit a target."
                )

            _, self.train_config.density_weighted = imgui.checkbox(
                "Density Weighted##dw", self.train_config.density_weighted
            )
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "On: acceptance = density * prob (more Gaussians in dense regions).\n"
                    "Off: acceptance = prob (uniform across all active voxels)."
                )

            if self.train_config.gaussian_mode == "percentage":
                _, self.train_config.probability_scale = imgui.slider_float(
                    "Spawn Probability",
                    self.train_config.probability_scale,
                    0.001,
                    0.2,
                    format="%.4f",
                )
            else:
                _, self.train_config.target_count = imgui.slider_int(
                    "Target Count##gc",
                    self.train_config.target_count,
                    100,
                    100000,
                )

            _, self.train_config.min_count = imgui.slider_int(
                "Min Gaussians", self.train_config.min_count, 0, 10000
            )
            _, self.train_config.max_count = imgui.slider_int(
                "Max Gaussians", self.train_config.max_count, 1000, 200000
            )

            _, self.train_config.sigma_scale = imgui.slider_float(
                "Initial Sigma Scale", self.train_config.sigma_scale, 1.0, 20.0
            )
            _, self.train_config.jitter_scale = imgui.slider_float(
                "Position Jitter", self.train_config.jitter_scale, 0.0, 200.0
            )

            imgui.dummy((0, 10))
            imgui.text("Loss Function")
            imgui.separator()
            loss_modes = ["L2 (MSE)", "L1 (Pseudo-Huber smooth)", "Huber"]
            _, self.train_config.loss_mode = imgui.combo(
                "Loss Mode", self.train_config.loss_mode, loss_modes
            )
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "L2: Standard MSE. Fast convergence but ignores sparse regions.\n"
                    "L1: Constant gradient everywhere. Helps fill low-density voids.\n"
                    "Huber: L2 for small errors, L1 for large. Best of both worlds."
                )

            if self.train_config.loss_mode == 2:  # Huber only
                _, self.train_config.huber_delta = imgui.slider_float(
                    "Huber Delta",
                    self.train_config.huber_delta,
                    0.01,
                    1.0,
                    format="%.3f",
                )
                if imgui.is_item_hovered():
                    imgui.set_tooltip(
                        "Errors below delta use L2, above use L1.\n"
                        "Set near your typical per-voxel error magnitude."
                    )

            imgui.dummy((0, 5))
            imgui.text("SSIM Combined Loss")
            _, self.train_config.ssim_weight = imgui.slider_float(
                "SSIM Weight",
                self.train_config.ssim_weight,
                0.0,
                1.0,
                format="%.2f",
            )
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Blend weight for structural similarity loss.\n"
                    "L = (1-w)*L_pervoxel + w*D-SSIM\n"
                    "0.0 = per-voxel only, 0.2 = recommended (3DGS standard).\n"
                    "Uses predicted volume from previous step."
                )

            if self.train_config.ssim_weight > 0.0:
                window_sizes = ["3x3x3 (R=1)", "5x5x5 (R=2)", "7x7x7 (R=3)"]
                cur_ws = self.train_config.ssim_window_radius - 1
                _, new_ws = imgui.combo("SSIM Window", cur_ws, window_sizes)
                self.train_config.ssim_window_radius = new_ws + 1
                if imgui.is_item_hovered():
                    imgui.set_tooltip(
                        "Local window size for SSIM computation.\n"
                        "Larger = more structural awareness but slower.\n"
                        "5x5x5 is a good default."
                    )

            imgui.separator()

            # VDB up axis selector
            imgui.text("VDB up axis")
            imgui.same_line()
            imgui.push_item_width(80)
            cur_idx = VDB_UP_AXIS_NAMES.index(self.vdb_up_axis) if self.vdb_up_axis in VDB_UP_AXIS_NAMES else 0
            changed_up, new_up_idx = imgui.combo("##vdb_up", cur_idx, VDB_UP_AXIS_NAMES)
            imgui.pop_item_width()
            if changed_up:
                self.vdb_up_axis = VDB_UP_AXIS_NAMES[new_up_idx]
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Source VDB coordinate convention.\n"
                    "+Y = Houdini / Maya (most common)\n"
                    "+Z = Blender / 3ds Max\n"
                    "Change and press Reload VDB to apply."
                )
            imgui.same_line()
            if imgui.button("Reload VDB"):
                self.vol_min_world, self.vol_max_world, vol_data, self.axis_remap, _ = convert_grid_to_dense_volume(
                    self.grid, VOL_SIZE, up_axis_name=self.vdb_up_axis
                )
                cmd = self.device.create_command_encoder()
                cmd.upload_texture_data(self.renderer.volume_tex, [vol_data])
                self.device.submit_command_buffer(cmd.finish())
                self.renderer._vol_min = self.vol_min_world
                self.renderer._vol_max = self.vol_max_world
                self.gaussians = convert_grid_to_gaussians(self.grid, self.train_config, self.axis_remap)
                self.apply_densification(self.gaussians)
                self.renderer.adam_iteration = 1
                self.train_step = 0
                self.adc._ref_cache = None
                cmd = self.device.create_command_encoder()
                self.renderer.rasterize_gaussians(cmd, self.vol_min_world, self.vol_max_world)
                self.device.submit_command_buffer(cmd.finish())
                self.train_config.sgld.reset()
                print(f"VDB reloaded with up axis: {self.vdb_up_axis}")

            if imgui.button("Regenerate Gaussians"):
                self.gaussians = convert_grid_to_gaussians(self.grid, self.train_config, self.axis_remap)
                self.apply_densification(self.gaussians)
                self.renderer.adam_iteration = 1
                self.train_step = 0
                self.adc._ref_cache = None
                cmd = self.device.create_command_encoder()
                self.renderer.rasterize_gaussians(
                    cmd, self.vol_min_world, self.vol_max_world
                )
                self.device.submit_command_buffer(cmd.finish())
                self.train_config.sgld.reset()
                print("Gaussians regenerated!")

            imgui.dummy((0, 5))
            changed_mgpt, new_mgpt = imgui.slider_int(
                "Max Gaussians/Tile", self.renderer.max_gaussians_per_tile, 32, 1024
            )
            if changed_mgpt:
                self.renderer.set_max_gaussians_per_tile(new_mgpt)

            imgui.dummy((0, 10))
            if imgui.button(
                "Start Training" if not self.is_training else "Stop Training"
            ):
                self.is_training = not self.is_training
                if self.is_training:
                    self.train_step = 0
                    self.train_config.sgld.reset()

            if self.is_training:
                optimizer_name = "ADAM" if self.train_config.use_adam else "SGD"
                sgld_suffix    = "+SGLD" if self.train_config.sgld.enabled else ""
                imgui.text_colored((0, 1, 0, 1), f"TRAINING ACTIVE ({optimizer_name}{sgld_suffix})")

            imgui.dummy((0, 5))
            imgui.separator()
            imgui.text("Export to SuperSplat")

            # The ONLY knob you should normally need
            _, self.ply_lighting.density_scale = imgui.slider_float(
                "Density Scale##ply", self.ply_lighting.density_scale, 1.0, 200.0
            )
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Match this to your Rendering > Density slider.\n"
                    "Opacity is computed physically: alpha = 1 - exp(-w * sigma * sqrt(2pi) * density_scale)\n"
                    "Too transparent? Raise it. Too solid? Lower it."
                )

            imgui.text_colored(imgui.ImVec4(0.6, 0.6, 0.6, 1), "Shading")
            _, self.ply_lighting.bright_luminance = imgui.slider_float(
                "Bright##ply", self.ply_lighting.bright_luminance, 0.7, 1.0, "%.2f"
            )
            _, self.ply_lighting.dark_luminance = imgui.slider_float(
                "Dark##ply", self.ply_lighting.dark_luminance, 0.05, 0.6, "%.2f"
            )
            _, self.ply_lighting.depth_gamma = imgui.slider_float(
                "Depth Gamma##ply", self.ply_lighting.depth_gamma, 0.5, 4.0, "%.1f"
            )
            _, self.ply_lighting.shadow_warmth = imgui.slider_float(
                "Shadow Warmth##ply", self.ply_lighting.shadow_warmth, 0.0, 0.4, "%.2f"
            )

            imgui.dummy((0, 3))
            if imgui.button("Sync Density from Renderer"):
                self.ply_lighting.density_scale = self.settings.density_scale
            if imgui.is_item_hovered():
                imgui.set_tooltip("Copy density_scale from your Rendering panel")

            imgui.dummy((0, 3))
            if imgui.button("Save to PLY (SuperSplat)"):
                gpu_params = (
                    self.renderer.gaussian_buffer.to_numpy()
                    .view(np.float32)
                    .reshape(-1, PARAMS_PER_GAUSSIAN)
                    .copy()
                )
                save_gaussians_dialog(gpu_params, self.ply_lighting)
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Opacity is physically derived from weight * sigma * density_scale.\n"
                    "This matches what your volumetric renderer shows.\n"
                    "Check console for opacity stats after saving."
                )

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
                "Prediction Error",
            ]

            old_mode = self.renderer.debug_mode
            clicked, self.renderer.debug_mode = imgui.combo(
                "Debug Mode", self.renderer.debug_mode, debug_modes
            )

            # Refresh debug when mode changes
            if self.renderer.debug_mode != old_mode:
                self.renderer.debug_needs_update = True

            if self.renderer.debug_mode > 0:
                old_scale = self.renderer.debug_scale
                _, self.renderer.debug_scale = imgui.slider_float(
                    "Visualization Scale", self.renderer.debug_scale, 0.1, 5.0
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
                    imgui.text("Typical range: 0.0 - 0.05")
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
                    if (
                        hasattr(self.renderer, "grad_buffer")
                        and self.renderer.grad_buffer
                    ):
                        grad_bytes = self.renderer.grad_buffer.to_numpy()
                        grad_debug = grad_bytes.view(dtype=np.float32)
                        grad_nonzero = grad_debug[grad_debug != 0]
                        if len(grad_nonzero) > 0:
                            imgui.text("Current gradients:")
                            imgui.text(f"  Max: {np.max(np.abs(grad_nonzero)):.6f}")
                            imgui.text(f"  Mean: {np.mean(np.abs(grad_nonzero)):.6f}")
                            imgui.text(
                                f"  Active: {len(grad_nonzero)}/{len(grad_debug)}"
                            )

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
                    if (
                        hasattr(self.renderer, "tile_counts")
                        and self.renderer.tile_counts
                    ):
                        tile_debug = self.renderer.tile_counts.to_numpy().view(
                            dtype=np.uint32
                        )
                        total_refs = np.sum(tile_debug)
                        occupied = np.sum(tile_debug > 0)
                        if occupied > 0:
                            imgui.text("Tile statistics:")
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
                    imgui.text("Typical range: 0.0 - 1.0")

                elif self.renderer.debug_mode == 5:  # Error
                    imgui.text_colored((0.7, 0.7, 0.7, 1), "Prediction Error")
                    imgui.text_wrapped(
                        "Compares prediction vs. ground truth.\n"
                        "Blue: Underpredicting (add Gaussians)\n"
                        "White: Accurate (< ±0.02 error)\n"
                        "Red: Overpredicting (reduce weights)"
                    )
                    imgui.separator()
                    imgui.text("Error range: -0.1 to +0.1")

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
                "healthy": (0.0, 1.0, 0.0, 1.0),  # Green
                "converged": (0.0, 0.8, 1.0, 1.0),  # Cyan
                "exploding": (1.0, 0.0, 0.0, 1.0),  # Red
                "vanishing": (1.0, 0.5, 0.0, 1.0),  # Orange
                "dead": (0.5, 0.0, 0.0, 1.0),  # Dark red
                "sparse": (1.0, 1.0, 0.0, 1.0),  # Yellow
                "unknown": (0.5, 0.5, 0.5, 1.0),  # Gray
            }

            color = status_colors.get(health["status"], (1, 1, 1, 1))
            imgui.text_colored(color, health["status"].upper())

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

            if health["status"] in ["exploding", "vanishing", "dead", "sparse"]:
                # Warning box
                imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(1, 1, 0, 1))
                imgui.text_wrapped(f"⚠️  {health['recommendation']}")
                imgui.pop_style_color()

                # Quick fix buttons
                imgui.dummy((0, 5))

                if health["status"] == "exploding":
                    if imgui.button("Fix: Reduce LRs by 5×"):
                        self.train_config.learning_rate_scale /= 5.0
                        self.train_config.learning_rate_rotation /= 5.0
                        self.train_config.learning_rate_weight /= 5.0
                        print("Reduced learning rates by 5×")

                    imgui.same_line()
                    if imgui.button("Reset Adam State"):
                        self.renderer._init_adam_state()
                        print("Adam state reset")

                elif health["status"] == "vanishing":
                    if imgui.button("Fix: Increase LRs by 5×"):
                        self.train_config.learning_rate_pos *= 5.0
                        self.train_config.learning_rate_scale *= 5.0
                        self.train_config.learning_rate_rotation *= 5.0
                        self.train_config.learning_rate_weight *= 5.0
                        print("Increased learning rates by 5×")

                elif health["status"] == "sparse":
                    if imgui.button("Fix: Increase Sigma Scale"):
                        self.train_config.sigma_scale *= 1.5
                        imgui.text_colored(
                            (1, 1, 0, 1), "Note: Regenerate Gaussians to apply!"
                        )

            else:
                # All good
                imgui.text_colored((0, 1, 0, 1), f"✓ {health['recommendation']}")

            # Loss trend
            imgui.dummy((0, 10))
            imgui.separator()
            imgui.text("Loss Trend:")

            if len(self.renderer.loss_history) > 2:
                loss_array = np.array(
                    self.renderer.loss_history[-100:], dtype=np.float32
                )
                imgui.plot_lines(
                    "##loss_trend",
                    loss_array,
                    scale_min=0.0,
                    scale_max=np.max(loss_array) * 1.1 if len(loss_array) > 0 else 1.0,
                    graph_size=(300, 80),
                )

                current_loss = self.renderer.loss_history[-1]
                imgui.text(f"Current: {current_loss:.10f}")

                # Loss trend analysis
                if len(loss_array) > 10:
                    recent = loss_array[-10:]
                    prev = (
                        loss_array[-20:-10]
                        if len(loss_array) >= 20
                        else loss_array[:-10]
                    )

                    improvement = (
                        np.mean(prev) - np.mean(recent) if len(prev) > 0 else 0
                    )

                    if improvement > 0.001:
                        imgui.text_colored(
                            (0, 1, 0, 1), f"↓ Improving ({improvement:.6f}/10 iter)"
                        )
                    elif improvement < -0.001:
                        imgui.text_colored(
                            (1, 0, 0, 1), f"↑ Worsening ({-improvement:.6f}/10 iter)"
                        )
                    else:
                        imgui.text_colored((1, 1, 0, 1), "→ Plateaued")

            # Gradient magnitude trend
            imgui.dummy((0, 10))
            imgui.text("Gradient Magnitude Trend:")

            if len(self.renderer.grad_mean_history) > 2:
                grad_array = np.array(
                    self.renderer.grad_mean_history[-100:], dtype=np.float32
                )
                imgui.plot_lines(
                    "##grad_trend",
                    grad_array,
                    scale_min=0.0,
                    scale_max=np.max(grad_array) * 1.1 if len(grad_array) > 0 else 1.0,
                    graph_size=(300, 80),
                )
            self.metrics.draw_ui_inline()
            self.screen_metrics.draw_ui_inline()

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
            self.renderer.rasterize_gaussians(
                cmd, self.vol_min_world, self.vol_max_world
            )
            self.device.submit_command_buffer(cmd.finish())
            vol = self.renderer.gaussian_volume_tex.to_numpy()

        ref = self.renderer.volume_tex.to_numpy()
        diff = vol - ref
        return np.mean(diff * diff)


# ==========================================
# SLIM APP (slangpy-native, no OpenGL/imgui)
# ==========================================


class SlimApp(spy.AppWindow):
    """Lightweight app using slangpy's native Vulkan window and UI."""

    def __init__(self, app):
        super().__init__(
            app,
            width=WINDOW_WIDTH,
            height=WINDOW_HEIGHT,
            title=WINDOW_TITLE,
            enable_vsync=False,
        )

        self._app = app
        self._width = WINDOW_WIDTH
        self._height = WINDOW_HEIGHT

        self.is_training = False
        self.settings = Settings()
        self.camera = Camera(self._width, self._height)
        self.train_config = TrainingConfig()

        # Use the device from AppWindow (created in entry point and passed via spy.App)
        self._device = self.device

        global VOL_SIZE
        self.grid = load_vdb_grid(VDB_FILE)
        self.vdb_up_axis = VDB_UP_AXIS
        self.vol_min_world, self.vol_max_world, vol_data, self.axis_remap, resolved_size = (
            convert_grid_to_dense_volume(
                self.grid, VOL_SIZE, up_axis_name=self.vdb_up_axis,
                use_native_size=USE_NATIVE_VDB_SIZE,
            )
        )
        VOL_SIZE = resolved_size
        self.gaussians = convert_grid_to_gaussians(
            self.grid, self.train_config, self.axis_remap
        )

        self.renderer = Renderer(self._device, vol_data)
        self.renderer.resize(self._width, self._height)
        self.renderer.gaussian_count = len(self.gaussians)
        self.renderer.gaussian_buffer = self._device.create_buffer(
            element_count=self.renderer.gaussian_count,
            struct_size=PARAMS_PER_GAUSSIAN * 4,
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            memory_type=spy.MemoryType.device_local,
            data=self.gaussians,
        )
        self.renderer._vol_min = self.vol_min_world
        self.renderer._vol_max = self.vol_max_world
        self.renderer.init_training()

        cmd = self._device.create_command_encoder()
        self.renderer.rasterize_gaussians(
            cmd, self.vol_min_world, self.vol_max_world
        )
        self._device.submit_command_buffer(cmd.finish())

        self.adc = ADCController(self, ADCConfig())
        self.adc.train_config = self.train_config

        self.sgld_diag = SGLDDiagnostics(self)
        self.train_step = 0

        # Input state
        self._keys_down = set()
        self._right_mouse_down = False
        self._last_mouse_pos = None

        # Timing
        self._last_time = time.perf_counter()
        self._frame_count = 0
        self._fps = 0.0
        self._fps_timer = 0.0
        self._fps_frames = 0

        # Build UI
        self._build_ui()

    def _build_ui(self):
        """Create spy.ui widgets for minimal controls."""
        screen = self.screen
        win = sui.Window(screen, title="Controls", size=spy.math.float2(300, 500))

        self._ui_fps = sui.Text(win, text="FPS: --")
        self._ui_gauss_count = sui.Text(
            win, text=f"Gaussians: {self.renderer.gaussian_count:,}"
        )

        sui.Text(win, text="--- Render ---")
        self._ui_splatting = sui.CheckBox(
            win, label="Splatting Mode", value=self.renderer.use_splatting,
            callback=self._on_splatting_toggle,
        )
        self._ui_gauss_vol = sui.CheckBox(
            win, label="Gaussian Volume", value=self.renderer.use_gaussian_volume,
            callback=self._on_gauss_vol_toggle,
        )
        self._ui_density = sui.SliderFloat(
            win, label="Density Scale", value=self.settings.density_scale,
            min=0.1, max=200.0,
            callback=self._on_density_change,
        )
        self._ui_light_pen = sui.SliderFloat(
            win, label="Light Penetration", value=self.settings.light_penetration,
            min=0.0, max=1.0,
            callback=self._on_light_pen_change,
        )
        self._ui_sun_x = sui.SliderFloat(
            win, label="Sun Dir X", value=self.settings.sun_direction[0],
            min=-1.0, max=1.0,
            callback=lambda v: self._on_sun_dir_change(0, v),
        )
        self._ui_sun_y = sui.SliderFloat(
            win, label="Sun Dir Y", value=self.settings.sun_direction[1],
            min=-1.0, max=1.0,
            callback=lambda v: self._on_sun_dir_change(1, v),
        )
        self._ui_sun_z = sui.SliderFloat(
            win, label="Sun Dir Z", value=self.settings.sun_direction[2],
            min=-1.0, max=1.0,
            callback=lambda v: self._on_sun_dir_change(2, v),
        )
        self._ui_sun_intensity = sui.SliderFloat(
            win, label="Sun Intensity", value=self.settings.sun_intensity,
            min=0.0, max=20.0,
            callback=lambda v: setattr(self.settings, 'sun_intensity', v),
        )
        self._ui_ambient_intensity = sui.SliderFloat(
            win, label="Ambient Intensity", value=self.settings.ambient_intensity,
            min=0.0, max=5.0,
            callback=lambda v: setattr(self.settings, 'ambient_intensity', v),
        )
        self._ui_shadow_strength = sui.SliderFloat(
            win, label="Shadow Strength", value=self.settings.shadow_strength,
            min=0.0, max=20.0,
            callback=lambda v: setattr(self.settings, 'shadow_strength', v),
        )
        self._ui_splat_softness = sui.SliderFloat(
            win, label="Splat Softness", value=self.settings.splat_softness,
            min=0.1, max=10.0,
            callback=lambda v: setattr(self.settings, 'splat_softness', v),
        )
        self._ui_enable_blur = sui.CheckBox(
            win, label="Enable Blur", value=self.settings.enable_blur,
            callback=lambda v: setattr(self.settings, 'enable_blur', v),
        )
        self._ui_blur_radius = sui.SliderInt(
            win, label="Blur Radius", value=self.settings.blur_radius,
            min=1, max=4,
            callback=lambda v: setattr(self.settings, 'blur_radius', v),
        )
        self._ui_depth_darkening = sui.CheckBox(
            win, label="Depth Darkening", value=self.settings.use_depth_darkening,
            callback=lambda v: setattr(self.settings, 'use_depth_darkening', v),
        )

        sui.Text(win, text="--- Training ---")
        self._ui_train_btn = sui.Button(
            win, label="Start Training", callback=self._on_train_toggle
        )
        self._ui_train_step = sui.Text(win, text="Step: 0")
        self._ui_lr_pos = sui.SliderFloat(
            win, label="LR Position", value=self.train_config.learning_rate_pos,
            min=0.001, max=1.0, format="%.4f",
            callback=lambda v: setattr(self.train_config, 'learning_rate_pos', v),
        )
        self._ui_lr_weight = sui.SliderFloat(
            win, label="LR Weight", value=self.train_config.learning_rate_weight,
            min=0.001, max=1.0, format="%.4f",
            callback=lambda v: setattr(self.train_config, 'learning_rate_weight', v),
        )
        self._ui_lr_scale = sui.SliderFloat(
            win, label="LR Scale", value=self.train_config.learning_rate_scale,
            min=0.0001, max=0.1, format="%.5f",
            callback=lambda v: setattr(self.train_config, 'learning_rate_scale', v),
        )

        sui.Text(win, text="--- Gaussians ---")
        self._ui_target_count = sui.SliderInt(
            win, label="Target Count", value=self.train_config.target_count,
            min=500, max=50000,
            callback=lambda v: setattr(self.train_config, 'target_count', v),
        )
        sui.Button(
            win, label="Regenerate Gaussians", callback=self._on_regenerate
        )

    # ---- UI callbacks ----

    def _on_splatting_toggle(self, val):
        self.renderer.use_splatting = val

    def _on_gauss_vol_toggle(self, val):
        self.renderer.use_gaussian_volume = val
        if val:
            self.renderer._needs_rasterization = True

    def _on_density_change(self, val):
        self.settings.density_scale = val

    def _on_light_pen_change(self, val):
        self.settings.light_penetration = val

    def _on_sun_dir_change(self, axis, val):
        self.settings.sun_direction[axis] = val

    def _on_train_toggle(self):
        self.is_training = not self.is_training
        self._ui_train_btn.label = (
            "Stop Training" if self.is_training else "Start Training"
        )
        if self.is_training:
            print("[SlimApp] Training started")
        else:
            print("[SlimApp] Training stopped")

    def _on_regenerate(self):
        was_training = self.is_training
        self.is_training = False
        self.train_config.gaussian_mode = "count"

        self.gaussians = convert_grid_to_gaussians(
            self.grid, self.train_config, self.axis_remap
        )
        self.renderer.gaussian_count = len(self.gaussians)
        self.renderer.gaussian_buffer = self._device.create_buffer(
            element_count=self.renderer.gaussian_count,
            struct_size=PARAMS_PER_GAUSSIAN * 4,
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            memory_type=spy.MemoryType.device_local,
            data=self.gaussians,
        )
        self.renderer.init_training()
        self.renderer._needs_rebinning = True
        self.renderer._needs_rasterization = True
        self.renderer.adam_iteration = 1
        self.train_step = 0

        cmd = self._device.create_command_encoder()
        self.renderer.rasterize_gaussians(
            cmd, self.vol_min_world, self.vol_max_world
        )
        self._device.submit_command_buffer(cmd.finish())

        self._ui_gauss_count.text = f"Gaussians: {self.renderer.gaussian_count:,}"
        self._ui_train_step.text = "Step: 0"
        print(f"[SlimApp] Regenerated {self.renderer.gaussian_count:,} gaussians")

        if was_training:
            self.is_training = True

    # ---- Densification support (needed by ADC) ----

    def apply_densification(self, new_params, surviving_indices=None):
        """Same as App.apply_densification — needed by ADCController."""
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
                old_m1 = (
                    self.renderer.adam_first_moment.to_numpy()
                    .view(np.float32).reshape(-1, PARAMS_PER_GAUSSIAN).copy()
                )
                old_m2 = (
                    self.renderer.adam_second_moment.to_numpy()
                    .view(np.float32).reshape(-1, PARAMS_PER_GAUSSIAN).copy()
                )
            except Exception as e:
                print(f"[ADC] Could not read momentum buffers: {e}")
                old_m1 = old_m2 = None
        else:
            old_m1 = old_m2 = None

        self.gaussians = new_params
        self.renderer.gaussian_count = len(new_params)
        self.renderer.gaussian_buffer = self._device.create_buffer(
            element_count=len(new_params),
            struct_size=PARAMS_PER_GAUSSIAN * 4,
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            memory_type=spy.MemoryType.device_local,
            data=flat_params,
        )
        self.renderer.init_training()
        self.renderer.adam_iteration = saved_iteration

        if surviving_indices is not None and old_m1 is not None:
            try:
                n_new = len(new_params)
                new_m1 = np.zeros((n_new, PARAMS_PER_GAUSSIAN), dtype=np.float32)
                new_m2 = np.zeros((n_new, PARAMS_PER_GAUSSIAN), dtype=np.float32)
                for i, src in enumerate(surviving_indices):
                    if i >= n_new:
                        break
                    if src >= 0 and src < len(old_m1):
                        new_m1[i] = old_m1[src]
                        new_m2[i] = old_m2[src]
                self.renderer.adam_first_moment.copy_from_numpy(new_m1.flatten())
                self.renderer.adam_second_moment.copy_from_numpy(new_m2.flatten())
            except Exception as e:
                print(f"[ADC] Momentum restore failed (non-fatal): {e}")

        self.renderer._needs_rebinning = True
        self.renderer._needs_rasterization = True
        self._ui_gauss_count.text = f"Gaussians: {len(new_params):,}"
        print(f"[ADC] Buffer rebuilt: {len(new_params):,} Gaussians")

    def compute_current_loss(self):
        if self.renderer.use_gaussian_volume:
            vol = self.renderer.gaussian_volume_tex.to_numpy()
        else:
            cmd = self._device.create_command_encoder()
            self.renderer.rasterize_gaussians(
                cmd, self.vol_min_world, self.vol_max_world
            )
            self._device.submit_command_buffer(cmd.finish())
            vol = self.renderer.gaussian_volume_tex.to_numpy()
        ref = self.renderer.volume_tex.to_numpy()
        diff = vol - ref
        return np.mean(diff * diff)

    # ---- Input ----

    def on_keyboard_event(self, event):
        if event.is_key_press() or event.is_key_repeat():
            self._keys_down.add(event.key)
            # Toggle shortcuts
            if event.is_key_press():
                if event.key == spy.KeyCode.space:
                    self._on_train_toggle()
                elif event.key == spy.KeyCode.t:
                    self.renderer.use_splatting = not self.renderer.use_splatting
                    self._ui_splatting.value = self.renderer.use_splatting
                    print(f"Splatting: {self.renderer.use_splatting}")
                elif event.key == spy.KeyCode.g:
                    self.renderer.use_gaussian_volume = not self.renderer.use_gaussian_volume
                    self._ui_gauss_vol.value = self.renderer.use_gaussian_volume
                    if self.renderer.use_gaussian_volume:
                        self.renderer._needs_rasterization = True
                    print(f"Gaussian Volume: {self.renderer.use_gaussian_volume}")
        elif event.is_key_release():
            self._keys_down.discard(event.key)

    def on_mouse_event(self, event):
        if event.is_button_down() and event.button == spy.MouseButton.right:
            self._right_mouse_down = True
            self.camera.first_mouse = True
            self.camera.is_dragging = True
        elif event.is_button_up() and event.button == spy.MouseButton.right:
            self._right_mouse_down = False
            self.camera.is_dragging = False
        elif event.is_move() and self._right_mouse_down:
            x, y = event.pos.x, event.pos.y
            self.camera.process_mouse(x, y)

    def _process_movement(self, dt):
        speed = self.camera.speed * dt
        if spy.KeyCode.w in self._keys_down:
            self.camera.pos += self.camera.front * speed
        if spy.KeyCode.s in self._keys_down:
            self.camera.pos -= self.camera.front * speed
        if spy.KeyCode.a in self._keys_down:
            self.camera.pos -= self.camera.right * speed
        if spy.KeyCode.d in self._keys_down:
            self.camera.pos += self.camera.right * speed
        if spy.KeyCode.q in self._keys_down:
            self.camera.pos += self.camera.up * speed
        if spy.KeyCode.e in self._keys_down:
            self.camera.pos -= self.camera.up * speed

    def on_resize(self, width, height):
        if width == 0 or height == 0:
            return
        self._device.wait_for_idle()
        self._width = width
        self._height = height
        self.renderer.resize(width, height)

    # ---- Main render callback ----

    def render(self, render_context):
        now = time.perf_counter()
        dt = now - self._last_time
        self._last_time = now

        # FPS counter
        self._fps_timer += dt
        self._fps_frames += 1
        if self._fps_timer >= 0.5:
            self._fps = self._fps_frames / self._fps_timer
            self._ui_fps.text = f"FPS: {self._fps:.1f}"
            self._fps_timer = 0.0
            self._fps_frames = 0

        self._process_movement(dt)
        self.renderer.check_hot_reload()

        # Training
        if self.is_training:
            self.renderer.run_training_step(
                self.vol_min_world, self.vol_max_world, self.train_config
            )
            self.train_step += 1

            if self.train_step == 100:
                self.sgld_diag.snapshot()
            if self.train_step % 500 == 0 and self.train_step > 100:
                self.sgld_diag.tick()

            if self.train_step % 100 == 0:
                self._ui_train_step.text = f"Step: {self.train_step}"
                tile_debug = self.renderer.tile_counts.to_numpy().view(dtype=np.uint32)
                total_refs = np.sum(tile_debug)
                grad_bytes = self.renderer.grad_buffer.to_numpy()
                grad_all = grad_bytes.view(dtype=np.float32).reshape(
                    -1, PARAMS_PER_GAUSSIAN
                )
                grad_max = np.max(np.abs(grad_all))
                if total_refs == 0:
                    print(f"[CRITICAL] Tiler is empty!")
                elif grad_max == 0:
                    print(f"[ALERT] ALL gradients are ZERO.")
                else:
                    print(
                        f"[OK] Step {self.train_step}, Refs: {total_refs}, "
                        f"GradMax: {grad_max:.6f}"
                    )

            if self.train_step % 50 == 0:
                self.renderer.analyze_gradients()
                loss = self.compute_current_loss()
                self.renderer.loss_history.append(loss)
                if len(self.renderer.loss_history) > 200:
                    self.renderer.loss_history.pop(0)

        self.adc.tick(self._frame_count, self.is_training)
        self.adc.apply_pending()

        # Rasterize if needed (non-training)
        if not self.is_training:
            if self.renderer._needs_rasterization and self.renderer.use_gaussian_volume:
                cmd = self._device.create_command_encoder()
                self.renderer.rasterize_gaussians(
                    cmd, self.vol_min_world, self.vol_max_world
                )
                self._device.submit_command_buffer(cmd.finish())

        # Render + blit on the framework's command encoder to avoid
        # swapchain semaphore conflicts (no extra submit_command_buffer).
        # Note: splatting pass 1 still does its own submit for CPU depth sort
        # readback, but passes 2-4 go on this encoder.
        cmd = render_context.command_encoder
        self.renderer.render(self.camera, self.settings, ext_cmd=cmd)
        self.renderer.blit_to_surface(cmd, render_context.surface_texture)

        self._frame_count += 1


# ==========================================
# ENTRY POINT
# ==========================================

if __name__ == "__main__":
    if EXTENDED_UI:
        app = App()
        app.run()
    else:
        example_dir = Path(__file__).parent
        device = spy.Device(
            enable_debug_layers=True,
            compiler_options={"include_paths": [example_dir]},
            type=spy.DeviceType.vulkan,
        )
        app = spy.App(device=device)
        window = SlimApp(app)
        app.run()
