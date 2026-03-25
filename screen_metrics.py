# screen_metrics.py
# Screen-space (2D) metrics: PSNR, SSIM, IoU, L1
# Renders reference and Gaussian volumes from multiple canonical cameras
# and compares the resulting images pixel-by-pixel on GPU.
#
# Public API:
#   sm = ScreenMetricsCollector(device, renderer, settings)
#   sm = ScreenMetricsCollector(device, renderer, settings, render_w=512, render_h=384)
#   sm.tick(frame, is_training)        # call every frame in run loop
#   sm.get_latest() -> dict            # JSON-safe snapshot
#   sm.get_config() -> dict            # serialise config
#   sm.set_config(d)                   # restore config
#   sm.get_cameras() -> list[dict]     # serialise camera set
#   sm.set_cameras_from_dicts(l)       # restore camera set
#   sm.export_csv(path)                # full history dump
#   sm.draw_ui_inline()                # inside any imgui begin/end block

from __future__ import annotations

import csv
import dataclasses
import math
import numpy as np
import slangpy as spy
from imgui_bundle import imgui

# Slot indices must match screen_metrics.slang
SLOT_MSE_SUM       = 0
SLOT_L1_SUM        = 1
SLOT_IOU_INTERSECT = 2
SLOT_IOU_UNION     = 3
SLOT_SSIM_SUM      = 4
SLOT_SSIM_COUNT    = 5
SLOT_PIXEL_COUNT   = 6
NUM_SLOTS          = 8

# Default off-screen render resolution — overridable per-instance via render_w/render_h
DEFAULT_METRICS_W = 512
DEFAULT_METRICS_H = 384


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ScreenMetricsSnapshot:
    """One timestep. NaN until computed. to_dict() is JSON/CSV safe."""
    frame:    int        = 0
    psnr:     float      = float("nan")
    l1:       float      = float("nan")
    iou:      float      = float("nan")
    ssim:     float      = float("nan")
    per_view: list[dict] = dataclasses.field(default_factory=list)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "ScreenMetricsSnapshot":
        pv = d.pop("per_view", [])
        snap = ScreenMetricsSnapshot(**d)
        snap.per_view = pv
        return snap


@dataclasses.dataclass
class CanonicalCamera:
    """
    Frozen camera pose.
    right and up already carry the aspect * 0.5 / 0.5 factors baked in,
    matching the layout produced by Camera.get_gpu_data() in main.py.
    """
    pos:   tuple = (0.0, 0.0, 2.5)
    front: tuple = (0.0, 0.0, -1.0)
    right: tuple = (0.667, 0.0, 0.0)
    up:    tuple = (0.0, 0.5, 0.0)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "CanonicalCamera":
        return CanonicalCamera(**{k: tuple(v) for k, v in d.items()})

    @staticmethod
    def from_live_camera(camera, aspect: float) -> "CanonicalCamera":
        """Snapshot a live Camera object from main.py."""
        gpu = camera.get_gpu_data(aspect)
        return CanonicalCamera(
            pos=tuple(map(float, gpu[0:3])),
            front=tuple(map(float, gpu[4:7])),
            right=tuple(map(float, gpu[8:11])),
            up=tuple(map(float, gpu[12:15])),
        )


@dataclasses.dataclass
class ScreenMetricsConfig:
    interval:         int   = 200     # compute every N frames
    history_length:   int   = 300
    iou_threshold:    float = 0.1
    ssim_window_half: int   = 5
    ssim_c1:          float = 0.0001
    ssim_c2:          float = 0.0009
    enable_psnr:      bool  = True
    enable_l1:        bool  = True
    enable_iou:       bool  = True
    enable_ssim:      bool  = True

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "ScreenMetricsConfig":
        return ScreenMetricsConfig(**d)


# ---------------------------------------------------------------------------
# Camera orbit helpers
# ---------------------------------------------------------------------------

def make_orbit_cameras(
    n: int = 6,
    radius: float = 2.5,
    elevation_deg: float = 15.0,
    aspect: float = DEFAULT_METRICS_W / DEFAULT_METRICS_H,
) -> list[CanonicalCamera]:
    """
    n evenly-spaced cameras orbiting the origin at a fixed elevation.
    aspect bakes into the right vector as aspect * 0.5, matching main.py.

    Note: preset cameras are built with the default aspect ratio. If you use
    a significantly different render_w/render_h ratio the right/up vectors
    will be slightly off — call make_orbit_cameras() with your own aspect
    and use set_cameras_from_dicts() to replace the preset if this matters.
    """
    cameras = []
    elev = math.radians(elevation_deg)

    for i in range(n):
        az = math.radians(360.0 * i / n)

        px = radius * math.cos(elev) * math.cos(az)
        py = radius * math.sin(elev)
        pz = radius * math.cos(elev) * math.sin(az)

        # Front: toward origin
        fl = radius
        fx, fy, fz = -px / fl, -py / fl, -pz / fl

        # Right: front × world_up, normalised, scaled by aspect * 0.5
        wx, wy, wz = 0.0, 1.0, 0.0
        rx = fy * wz - fz * wy
        ry = fz * wx - fx * wz
        rz = fx * wy - fy * wx
        rlen = math.sqrt(rx*rx + ry*ry + rz*rz) or 1.0
        rx, ry, rz = rx/rlen * aspect * 0.5, ry/rlen * aspect * 0.5, rz/rlen * aspect * 0.5

        # Up: right × front (after scaling — direction only), scaled by 0.5
        ux_r, uy_r, uz_r = rx / (aspect * 0.5), ry / (aspect * 0.5), rz / (aspect * 0.5)
        ux = uy_r * fz - uz_r * fy
        uy = uz_r * fx - ux_r * fz
        uz = ux_r * fy - uy_r * fx
        ulen = math.sqrt(ux*ux + uy*uy + uz*uz) or 1.0
        ux, uy, uz = ux/ulen * 0.5, uy/ulen * 0.5, uz/ulen * 0.5

        cameras.append(CanonicalCamera(
            pos=(px, py, pz),
            front=(fx, fy, fz),
            right=(rx, ry, rz),
            up=(ux, uy, uz),
        ))

    return cameras


def make_6sided_cameras(
    radius: float = 2.5,
    aspect: float = DEFAULT_METRICS_W / DEFAULT_METRICS_H,
) -> list[CanonicalCamera]:
    """Axis-aligned cameras: front, back, left, right, top, bottom."""
    a = aspect * 0.5
    return [
        CanonicalCamera(  # front (+Z looking -Z)
            pos=(0.0, 0.0, radius),
            front=(0.0, 0.0, -1.0),
            right=(a, 0.0, 0.0),
            up=(0.0, 0.5, 0.0),
        ),
        CanonicalCamera(  # back (-Z looking +Z)
            pos=(0.0, 0.0, -radius),
            front=(0.0, 0.0, 1.0),
            right=(-a, 0.0, 0.0),
            up=(0.0, 0.5, 0.0),
        ),
        CanonicalCamera(  # right (+X looking -X)
            pos=(radius, 0.0, 0.0),
            front=(-1.0, 0.0, 0.0),
            right=(0.0, 0.0, -a),
            up=(0.0, 0.5, 0.0),
        ),
        CanonicalCamera(  # left (-X looking +X)
            pos=(-radius, 0.0, 0.0),
            front=(1.0, 0.0, 0.0),
            right=(0.0, 0.0, a),
            up=(0.0, 0.5, 0.0),
        ),
        CanonicalCamera(  # top (+Y looking -Y)
            pos=(0.0, radius, 0.0),
            front=(0.0, -1.0, 0.0),
            right=(a, 0.0, 0.0),
            up=(0.0, 0.0, -0.5),
        ),
        CanonicalCamera(  # bottom (-Y looking +Y)
            pos=(0.0, -radius, 0.0),
            front=(0.0, 1.0, 0.0),
            right=(a, 0.0, 0.0),
            up=(0.0, 0.0, 0.5),
        ),
    ]


# Built-in presets
CAMERA_PRESETS: dict[str, list[CanonicalCamera]] = {
    "front only": [
        CanonicalCamera(
            pos=(0.0, 0.0, 2.5),
            front=(0.0, 0.0, -1.0),
            right=(0.667, 0.0, 0.0),
            up=(0.0, 0.5, 0.0),
        )
    ],
    "6-sided":  make_6sided_cameras(),
    "orbit 4":  make_orbit_cameras(n=4,  elevation_deg=0.0),
    "orbit 6":  make_orbit_cameras(n=6,  elevation_deg=15.0),
    "orbit 8":  make_orbit_cameras(n=8,  elevation_deg=20.0),
    "top + orbit 4": make_orbit_cameras(n=4, elevation_deg=0.0) + [
        CanonicalCamera(
            pos=(0.0, 2.5, 0.001),
            front=(0.0, -1.0, 0.0),
            right=(0.667, 0.0, 0.0),
            up=(0.0, 0.0, -0.5),
        )
    ],
}

_PRESET_NAMES  = list(CAMERA_PRESETS.keys())
_DEFAULT_PRESET = "6-sided"


# ---------------------------------------------------------------------------
# Main collector
# ---------------------------------------------------------------------------

class ScreenMetricsCollector:
    """
    GPU screen-space metrics collector.

    render_w / render_h control the off-screen resolution used for all metric
    computations. Set once at construction; consistent across all checkpoints
    so metrics are comparable within an experiment. Configure per-experiment
    via the metrics_2d.render_width / render_height config keys.

    Serialisation contract
    ----------------------
    All state that matters for reproducibility is exposed via to/from dict:
        sm.get_latest()               -> dict   (last snapshot)
        sm.get_config()               -> dict   (ScreenMetricsConfig)
        sm.set_config(d)
        sm.get_cameras()              -> list[dict]
        sm.set_cameras_from_dicts(l)
    """

    def __init__(self, device, renderer, settings,
                 render_w: int = DEFAULT_METRICS_W,
                 render_h: int = DEFAULT_METRICS_H):
        self.render_w = render_w
        self.render_h = render_h
        self.device   = device
        self.renderer = renderer
        self.settings = settings

        self.config  = ScreenMetricsConfig()
        self._snap   = ScreenMetricsSnapshot()

        # Camera set
        self._preset_name = _DEFAULT_PRESET
        self._cameras: list[CanonicalCamera] = CAMERA_PRESETS[_DEFAULT_PRESET]

        # History
        self._frames: list[int]              = []
        self._psnr:   list[float]            = []
        self._l1:     list[float]            = []
        self._iou:    list[float]            = []
        self._ssim:   list[float]            = []
        self._per_view_psnr: list[list[float]] = []
        self._reset_per_view_history()

        self._wants_snapshot = False
        self._compiled       = False

        self._build_gpu_resources()
        self._compile_pipelines()

    # -----------------------------------------------------------------------
    # GPU setup
    # -----------------------------------------------------------------------

    def _build_gpu_resources(self):
        d = self.device

        def make_rw_tex(label: str):
            return d.create_texture(
                format=spy.Format.rgba32_float,
                width=self.render_w,
                height=self.render_h,
                usage=spy.TextureUsage.unordered_access | spy.TextureUsage.shader_resource,
                label=label,
            )

        self._ref_img  = make_rw_tex("SM_RefImage")
        self._pred_img = make_rw_tex("SM_PredImage")

        self._result_buf = d.create_buffer(
            element_count=NUM_SLOTS,
            struct_size=4,
            usage=spy.BufferUsage.unordered_access | spy.BufferUsage.shader_resource,
            memory_type=spy.MemoryType.device_local,
        )

    def _compile_pipelines(self):
        print(f"[ScreenMetrics] Compiling kernels... (res {self.render_w}×{self.render_h})")
        load = self.device.load_program

        self._pipe_clear   = self.device.create_compute_pipeline(
            program=load("shaders/screen_metrics.slang", ["clear_screen_metrics"])
        )
        self._pipe_render  = self.device.create_compute_pipeline(
            program=load("shaders/screen_metrics.slang", ["render_both"])
        )
        self._pipe_metrics = self.device.create_compute_pipeline(
            program=load("shaders/screen_metrics.slang", ["compute_screen_metrics"])
        )

        self._compiled = True
        print("[ScreenMetrics] Done")

    # -----------------------------------------------------------------------
    # Public serialisation interface
    # -----------------------------------------------------------------------

    def get_latest(self) -> dict:
        return self._snap.to_dict()

    def get_config(self) -> dict:
        return self.config.to_dict()

    def set_config(self, d: dict):
        self.config = ScreenMetricsConfig.from_dict(d)

    def get_cameras(self) -> list[dict]:
        return [c.to_dict() for c in self._cameras]

    def set_cameras_from_dicts(self, lst: list[dict]):
        self._cameras = [CanonicalCamera.from_dict(d) for d in lst]
        self._preset_name = "custom"
        self._reset_per_view_history()

    def snapshot_live_camera(self, camera, aspect: float | None = None):
        """Add (or replace) the first camera with the current interactive camera."""
        if aspect is None:
            aspect = self.render_w / self.render_h
        cam = CanonicalCamera.from_live_camera(camera, aspect)
        if self._cameras:
            self._cameras[0] = cam
        else:
            self._cameras = [cam]
        self._preset_name = "custom"
        self._reset_per_view_history()
        print(f"[ScreenMetrics] Camera 0 updated: pos={cam.pos}")

    def export_csv(self, path: str):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "psnr", "l1", "iou", "ssim"])
            for row in zip(self._frames, self._psnr, self._l1, self._iou, self._ssim):
                w.writerow([
                    "" if isinstance(v, float) and math.isnan(v) else v
                    for v in row
                ])
        print(f"[ScreenMetrics] {len(self._frames)} rows → {path}")

    # -----------------------------------------------------------------------
    # Main loop entry point
    # -----------------------------------------------------------------------

    def tick(self, frame: int, is_training: bool, force: bool = False):
        """Call every frame. Near-zero cost on skipped frames."""
        if not self._compiled:
            return
        if not is_training:
            return
        if self.renderer.gaussian_buffer is None:
            return
        if not force and frame % max(1, self.config.interval) != 0:
            return
        self._run(frame)

    # -----------------------------------------------------------------------
    # GPU execution
    # -----------------------------------------------------------------------

    def _bind_params(self, cursor, cam: CanonicalCamera):
        s   = self.settings
        cfg = self.config

        p = cursor["ScreenMetricsParams"]

        p["camPos"]   = cam.pos
        p["camFront"] = cam.front
        p["camRight"] = cam.right
        p["camUp"]    = cam.up

        p["stepSize"]      = s.step_size
        p["densityScale"]  = s.density_scale
        p["densityCurve"]  = s.density_curve
        p["maxSteps"]      = s.step_count
        p["smokeAlbedo"]   = tuple(s.smoke_color)
        p["lightPenetration"] = s.light_penetration
        p["phaseG"]        = s.phase_g
        p["sunDir"]        = tuple(s.get_sun_dir())
        p["sunColor"]      = (
            s.sun_color_base[0] * s.sun_intensity,
            s.sun_color_base[1] * s.sun_intensity,
            s.sun_color_base[2] * s.sun_intensity,
        )
        p["ambientColor"]  = (
            s.ambient_color_base[0] * s.ambient_intensity,
            s.ambient_color_base[1] * s.ambient_intensity,
            s.ambient_color_base[2] * s.ambient_intensity,
        )
        p["shadowSteps"]    = s.shadow_steps
        p["shadowStepMult"] = s.shadow_step_mult

        p["screenW"]            = self.render_w
        p["screenH"]            = self.render_h
        p["iouLumaThreshold"]   = cfg.iou_threshold
        p["ssimC1"]             = cfg.ssim_c1
        p["ssimC2"]             = cfg.ssim_c2
        p["ssimHalf"]           = cfg.ssim_window_half

    def _run_one_camera(self, cam: CanonicalCamera) -> dict:
        """Render + measure one camera. Returns per-view metric dict."""
        cmd = self.device.create_command_encoder()

        # Clear result slots
        with cmd.begin_compute_pass() as cp:
            cursor = spy.ShaderCursor(cp.bind_pipeline(self._pipe_clear))
            cursor["gScreenMetrics"] = self._result_buf
            cp.dispatch(thread_count=(8, 1, 1))

        # Render ref + pred
        gx = (self.render_w + 7) // 8
        gy = (self.render_h + 7) // 8
        with cmd.begin_compute_pass() as cp:
            cursor = spy.ShaderCursor(cp.bind_pipeline(self._pipe_render))
            cursor["gRefVolume"]     = self.renderer.volume_tex
            cursor["gPredVolume"]    = self.renderer.gaussian_volume_tex
            cursor["gLinearSampler"] = self.renderer.linear_sampler
            cursor["gScreenMetrics"] = self._result_buf
            cursor["gRefImage"]      = self._ref_img
            cursor["gPredImage"]     = self._pred_img
            self._bind_params(cursor, cam)
            cp.dispatch(thread_count=(gx * 8, gy * 8, 1))

        # Pixel-wise metrics
        total_px = self.render_w * self.render_h
        groups   = (total_px + 255) // 256
        with cmd.begin_compute_pass() as cp:
            cursor = spy.ShaderCursor(cp.bind_pipeline(self._pipe_metrics))
            cursor["gRefImage"]      = self._ref_img
            cursor["gPredImage"]     = self._pred_img
            cursor["gScreenMetrics"] = self._result_buf
            self._bind_params(cursor, cam)
            cp.dispatch(thread_count=(groups * 256, 1, 1))

        self.device.submit_command_buffer(cmd.finish())

        # Readback
        buf   = self._result_buf.to_numpy().view(np.float32)
        px    = float(buf[SLOT_PIXEL_COUNT])
        mse   = float(buf[SLOT_MSE_SUM])      / max(px, 1.0)
        iou_i = float(buf[SLOT_IOU_INTERSECT])
        iou_u = float(buf[SLOT_IOU_UNION])

        return {
            "psnr": 10.0 * math.log10(1.0 / mse) if mse > 1e-12 else 100.0,
            "l1":   float(buf[SLOT_L1_SUM])   / max(px, 1.0),
            "iou":  iou_i / max(iou_u, 1e-12),
            "ssim": float(buf[SLOT_SSIM_SUM]) / max(px, 1.0),
        }

    def _run(self, frame: int):
        # Gaussian volume must be current — rasterize if needed
        if not self.renderer.use_gaussian_volume or self.renderer._needs_rasterization:
            cmd = self.device.create_command_encoder()
            self.renderer.rasterize_gaussians(
                cmd,
                self.renderer._vol_min if hasattr(self.renderer, "_vol_min") else (0, 0, 0),
                self.renderer._vol_max if hasattr(self.renderer, "_vol_max") else (1, 1, 1),
            )
            self.device.submit_command_buffer(cmd.finish())

        per_view = [self._run_one_camera(cam) for cam in self._cameras]

        def nanmean(key: str) -> float:
            vals = [v[key] for v in per_view if not math.isnan(v[key])]
            return sum(vals) / len(vals) if vals else float("nan")

        snap = ScreenMetricsSnapshot(
            frame=frame,
            psnr=nanmean("psnr"),
            l1=nanmean("l1"),
            iou=nanmean("iou"),
            ssim=nanmean("ssim"),
            per_view=per_view,
        )
        self._snap = snap

        # Append to history
        self._frames.append(frame)
        self._psnr.append(snap.psnr)
        self._l1.append(snap.l1)
        self._iou.append(snap.iou)
        self._ssim.append(snap.ssim)

        for i, vr in enumerate(per_view):
            if i < len(self._per_view_psnr):
                self._per_view_psnr[i].append(vr["psnr"])

        self._trim_history()

    # -----------------------------------------------------------------------
    # History helpers
    # -----------------------------------------------------------------------

    def _reset_per_view_history(self):
        self._per_view_psnr = [[] for _ in self._cameras]

    def _trim_history(self):
        cap = self.config.history_length
        for lst in (self._frames, self._psnr, self._l1, self._iou, self._ssim):
            if len(lst) > cap:
                lst[:] = lst[-cap:]
        for lst in self._per_view_psnr:
            if len(lst) > cap:
                lst[:] = lst[-cap:]

    # -----------------------------------------------------------------------
    # ImGui UI
    # -----------------------------------------------------------------------

    def draw_ui_inline(self):
        snap = self._snap
        cfg  = self.config

        imgui.dummy((0, 8))
        header_open, _ = imgui.collapsing_header("Screen-Space Metrics", True)
        if not header_open:
            return

        # --- camera controls ---
        imgui.text("Camera preset")
        imgui.same_line()
        imgui.push_item_width(160)
        current_idx = _PRESET_NAMES.index(self._preset_name) \
            if self._preset_name in _PRESET_NAMES else 0
        changed, new_idx = imgui.combo("##smpreset", current_idx, _PRESET_NAMES)
        imgui.pop_item_width()
        if changed:
            self._preset_name = _PRESET_NAMES[new_idx]
            self._cameras     = CAMERA_PRESETS[self._preset_name]
            self._reset_per_view_history()

        imgui.same_line()
        if imgui.button("+ Live Cam##sm"):
            self._wants_snapshot = True
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Replaces view 0 with the current interactive camera.\n"
                "Useful for adding a hero angle to the orbit set."
            )

        imgui.text_colored(
            imgui.ImVec4(0.45, 0.45, 0.45, 1.0),
            f"  {len(self._cameras)} view(s)   "
            f"render res {self.render_w}×{self.render_h}"
        )

        imgui.separator()

        # --- scheduling ---
        imgui.push_item_width(150)
        _, cfg.interval = imgui.slider_int(
            "Every N frames##sm", cfg.interval, 10, 1000
        )
        imgui.pop_item_width()
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Screen metrics require one full raymarch pair per camera.\n"
                "200–500 is comfortable during training."
            )

        # --- enable toggles ---
        _, cfg.enable_psnr = imgui.checkbox("PSNR##sme", cfg.enable_psnr)
        imgui.same_line()
        _, cfg.enable_l1   = imgui.checkbox("L1##sme",   cfg.enable_l1)
        imgui.same_line()
        _, cfg.enable_iou  = imgui.checkbox("IoU##sme",  cfg.enable_iou)
        imgui.same_line()
        _, cfg.enable_ssim = imgui.checkbox("SSIM##sme", cfg.enable_ssim)

        imgui.separator()

        # --- current mean values ---
        def metric_row(label: str, val: float, good_dir: str,
                       green_t: float, red_t: float):
            imgui.text(f"{label:<14}")
            imgui.same_line(165)
            if math.isnan(val):
                imgui.text_colored(imgui.ImVec4(0.4, 0.4, 0.4, 1), "  —")
                return
            t = (
                (val - red_t) / (green_t - red_t + 1e-9)
                if good_dir == "high"
                else (red_t - val) / (red_t - green_t + 1e-9)
            )
            t = float(np.clip(t, 0.0, 1.0))
            imgui.text_colored(
                imgui.ImVec4(1.0 - t, t, 0.0, 1.0),
                f"{val:>9.4f}"
            )

        imgui.text("Mean across views:")
        metric_row("PSNR (dB)",  snap.psnr, "high", 35.0, 20.0)
        metric_row("L1 / MAE",   snap.l1,   "low",  0.01, 0.15)
        metric_row("IoU",        snap.iou,  "high",  0.7,  0.3)
        metric_row("SSIM",       snap.ssim, "high",  0.9,  0.5)
        imgui.text_colored(
            imgui.ImVec4(0.35, 0.35, 0.35, 1),
            f"  frame {snap.frame}"
        )

        # --- per-view breakdown (collapsible) ---
        if len(snap.per_view) > 1:
            imgui.dummy((0, 3))
            breakdown_open, _ = imgui.collapsing_header(
                f"Per-View ({len(snap.per_view)} views)##smv", False
            )
            if breakdown_open:
                for i, vr in enumerate(snap.per_view):
                    psnr_v = vr.get("psnr", float("nan"))
                    ssim_v = vr.get("ssim", float("nan"))
                    l1_v   = vr.get("l1",   float("nan"))
                    imgui.text(f"  [{i:02d}]")
                    imgui.same_line(55)
                    if not math.isnan(psnr_v):
                        t = float(np.clip((psnr_v - 20.0) / 15.0, 0.0, 1.0))
                        imgui.text_colored(
                            imgui.ImVec4(1-t, t, 0, 1),
                            f"PSNR {psnr_v:6.2f} dB"
                        )
                    imgui.same_line(175)
                    if not math.isnan(ssim_v):
                        imgui.text_colored(
                            imgui.ImVec4(0.55, 0.55, 0.55, 1),
                            f"SSIM {ssim_v:.4f}"
                        )
                    imgui.same_line(280)
                    if not math.isnan(l1_v):
                        imgui.text_colored(
                            imgui.ImVec4(0.55, 0.55, 0.55, 1),
                            f"L1 {l1_v:.4f}"
                        )

        # --- trend plots ---
        imgui.dummy((0, 4))

        def safe_plot(label: str, data: list, uid: str):
            clean = [x for x in data if not math.isnan(x)]
            if len(clean) < 2:
                imgui.text_colored(
                    imgui.ImVec4(0.4, 0.4, 0.4, 1),
                    f"{label}: waiting..."
                )
                return
            arr = np.array(clean, dtype=np.float32)
            lo  = float(arr.min())
            hi  = float(arr.max())
            if hi - lo < 1e-9:
                hi = lo + 1e-3
            imgui.text(label)
            imgui.plot_lines(
                f"##smplot{uid}", arr,
                scale_min=lo, scale_max=hi,
                graph_size=imgui.ImVec2(-1, 48),
            )

        safe_plot("PSNR",  self._psnr, "psnr")
        safe_plot("L1",    self._l1,   "l1")
        safe_plot("IoU",   self._iou,  "iou")
        safe_plot("SSIM",  self._ssim, "ssim")

        imgui.dummy((0, 4))
        imgui.separator()

        if imgui.button("Export CSV##smexp"):
            self.export_csv("screen_metrics_export.csv")
        if imgui.is_item_hovered():
            imgui.set_tooltip("Writes full history to screen_metrics_export.csv")