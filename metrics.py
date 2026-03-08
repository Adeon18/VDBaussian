# metrics.py
# Two-tier GPU metrics using tile-accelerated Gaussian density evaluation.

# Public API:
#   collector = MetricsCollector(device, renderer, vol_resolution)
#   collector.tick(frame, vol_min, vol_max)   # every frame in run loop
#   collector.draw_ui_inline()                # inside Training Health Monitor begin/end
#   collector.export_csv(path)               # batch experiment hook
#   collector.get_latest() -> dict           # snapshot for experiments

import numpy as np
import slangpy as spy
from imgui_bundle import imgui

# Result buffer slot indices which must match metrics.slang
SLOT_MSE_SUM = 0
SLOT_L1_SUM = 1
SLOT_IOU_INTERSECT = 2
SLOT_IOU_UNION = 3
SLOT_SSIM_XY = 4
SLOT_SSIM_XZ = 5
SLOT_SSIM_YZ = 6
SLOT_VOXEL_COUNT = 7
NUM_SLOTS = 8

_FAST_WG = 512


class MetricsConfig:
    """All tuneable knobs. Safe to mutate between frames."""

    def __init__(self):
        self.fast_interval: int = 1  # Tier-1 every N frames (1 = every frame)
        self.slow_interval: int = 50  # Tier-2 every N frames
        self.history_length: int = 500  # entries kept in RAM
        self.iou_threshold: float = 0.05  # density threshold for binary IoU
        self.ssim_window_half: int = 5  # box window half-size → 11x11
        self.ssim_c1: float = 0.0001
        self.ssim_c2: float = 0.0009
        self.ssim_slices_per_axis: int = 16  # slices sampled per axis
        self.enable_psnr: bool = True
        self.enable_l1: bool = True
        self.enable_iou: bool = True
        self.enable_ssim: bool = True


class MetricsSnapshot:
    """One timestep of metric values. All float, NaN until computed."""

    __slots__ = (
        "frame",
        "psnr",
        "l1",
        "iou",
        "ssim_xy",
        "ssim_xz",
        "ssim_yz",
        "ssim_mean",
    )

    def __init__(self):
        for s in self.__slots__:
            setattr(self, s, float("nan") if s != "frame" else 0)

    def to_dict(self) -> dict:
        return {s: getattr(self, s) for s in self.__slots__}


class MetricsCollector:
    """
    GPU-accelerated metrics collector.

    Uses the tile data and Gaussian parameter buffer directly.

    Batch experiment usage:
        collector.tick(frame, vol_min, vol_max)
        ...
        collector.export_csv("run_001.csv")
        best = collector.get_latest()
    """

    def __init__(self, device, renderer, vol_resolution: tuple):
        self.device = device
        self.renderer = renderer
        self.vol_res = vol_resolution
        self.config = MetricsConfig()

        self._last_snap = MetricsSnapshot()
        self._compiled = False

        # History
        self._frames: list[int] = []
        self._psnr: list[float] = []
        self._l1: list[float] = []
        self._iou: list[float] = []
        self._ssim_mean: list[float] = []

        # 8-float GPU result buffer, tiny, readback is free
        self._result_buf = device.create_buffer(
            element_count=NUM_SLOTS,
            struct_size=4,
            usage=spy.BufferUsage.unordered_access | spy.BufferUsage.shader_resource,
            memory_type=spy.MemoryType.device_local,
        )

        self._compile_pipelines()

    # Compilation
    def _compile_pipelines(self):
        print("[Metrics] Compiling kernels")
        prog_clear = self.device.load_program(
            "shaders/metrics.slang", ["clear_metrics"]
        )
        self._pipe_clear = self.device.create_compute_pipeline(program=prog_clear)

        prog_fast = self.device.load_program("shaders/metrics.slang", ["fast_metrics"])
        self._pipe_fast = self.device.create_compute_pipeline(program=prog_fast)

        prog_ssim = self.device.load_program("shaders/metrics.slang", ["ssim_slice"])
        self._pipe_ssim = self.device.create_compute_pipeline(program=prog_ssim)

        self._compiled = True
        print("[Metrics] Done:D")

    # Main loop entry point
    def tick(self, frame: int, vol_min: tuple, vol_max: tuple, is_training: bool):
        """Zero cost on frames where nothing is scheduled."""
        if not is_training:
            return
        if not self._compiled:
            return

        do_fast = frame % max(1, self.config.fast_interval) == 0
        do_slow = frame % max(1, self.config.slow_interval) == 0

        if not (do_fast or do_slow):
            return

        # Tile data must be populated. If a training step ran this frame
        # it already is. If not (paused), we need a fresh bin pass.
        if self.renderer._needs_rebinning:
            self._run_bin_pass(vol_min, vol_max)

        cmd = self.device.create_command_encoder()
        self._clear(cmd)

        if do_fast:
            self._dispatch_fast(cmd, vol_min, vol_max)

        if do_slow and self.config.enable_ssim:
            self._dispatch_ssim(cmd, vol_min, vol_max)

        self.device.submit_command_buffer(cmd.finish())
        self._readback(frame, do_slow)

    # GPU passes
    def _run_bin_pass(self, vol_min: tuple, vol_max: tuple):
        """Re-bin Gaussians into tiles when training is paused."""
        r = self.renderer
        cmd = self.device.create_command_encoder()
        total_tiles = r.tile_res[0] * r.tile_res[1] * r.tile_res[2]

        with cmd.begin_compute_pass() as cp:
            cursor = spy.ShaderCursor(cp.bind_pipeline(r.pipe_clear_tiles))
            cursor["gTileCounts"] = r.tile_counts
            cursor["TrainParams"]["tileResolution"] = r.tile_res
            cursor["TrainParams"]["totalTiles"] = total_tiles
            cursor["TrainParams"][
                "tileSize"
            ] = r.tile_size  # expose as attribute if needed
            cp.dispatch(thread_count=(total_tiles, 1, 1))

        with cmd.begin_compute_pass() as cp:
            cursor = spy.ShaderCursor(cp.bind_pipeline(r.pipe_bin))
            cursor["gGaussianParamsRaw"] = r.gaussian_buffer
            cursor["gTileContent"] = r.tile_content
            cursor["gTileCounts"] = r.tile_counts
            cursor["TrainParams"]["volumeResolution"] = self.vol_res
            cursor["TrainParams"]["tileResolution"] = r.tile_res
            cursor["TrainParams"]["gaussianCount"] = r.gaussian_count
            cursor["TrainParams"]["minWorld"] = tuple(vol_min)
            cursor["TrainParams"]["maxWorld"] = tuple(vol_max)
            cursor["TrainParams"]["tileSize"] = r.tile_size
            cp.dispatch(thread_count=(r.gaussian_count, 1, 1))

        self.device.submit_command_buffer(cmd.finish())
        self.renderer._needs_rebinning = False

    def _fill_shared_params(self, cursor, vol_min, vol_max):
        """Bind all MetricsParams fields shared across passes."""
        cursor["MetricsParams"]["volumeResolution"] = self.vol_res
        cursor["MetricsParams"]["tileResolution"] = self.renderer.tile_res
        cursor["MetricsParams"]["tileSize"] = self.renderer.tile_size
        cursor["MetricsParams"]["iouThreshold"] = self.config.iou_threshold
        cursor["MetricsParams"]["minWorld"] = tuple(vol_min)
        cursor["MetricsParams"]["maxWorld"] = tuple(vol_max)

    def _clear(self, cmd):
        with cmd.begin_compute_pass() as cp:
            cursor = spy.ShaderCursor(cp.bind_pipeline(self._pipe_clear))
            cursor["gMetricsResult"] = self._result_buf
            cp.dispatch(thread_count=(8, 1, 1))

    def _dispatch_fast(self, cmd, vol_min, vol_max):
        total_voxels = self.vol_res[0] * self.vol_res[1] * self.vol_res[2]
        # Round up to workgroup boundary
        groups = (total_voxels + _FAST_WG - 1) // _FAST_WG

        with cmd.begin_compute_pass() as cp:
            cursor = spy.ShaderCursor(cp.bind_pipeline(self._pipe_fast))
            cursor["gReference"] = self.renderer.volume_tex
            cursor["gGaussianParamsRaw"] = self.renderer.gaussian_buffer
            cursor["gTileContent"] = self.renderer.tile_content
            cursor["gTileCounts"] = self.renderer.tile_counts
            cursor["gMetricsResult"] = self._result_buf
            self._fill_shared_params(cursor, vol_min, vol_max)
            cp.dispatch(thread_count=(groups * _FAST_WG, 1, 1))

    def _dispatch_ssim(self, cmd, vol_min, vol_max):
        wh = self.config.ssim_window_half
        # axis → (slice_depth_dim, sliceW, sliceH)
        axes = [
            (
                0,
                self.vol_res[2],
                self.vol_res[0],
                self.vol_res[1],
            ),  # XY slices, iterate Z
            (
                1,
                self.vol_res[1],
                self.vol_res[0],
                self.vol_res[2],
            ),  # XZ slices, iterate Y
            (
                2,
                self.vol_res[0],
                self.vol_res[1],
                self.vol_res[2],
            ),  # YZ slices, iterate X
        ]

        for axis, depth, sw, sh in axes:
            n_slices = min(self.config.ssim_slices_per_axis, depth)
            slice_step = max(1, depth // n_slices)
            gx = (sw + 15) // 16
            gy = (sh + 15) // 16

            for s in range(n_slices):
                with cmd.begin_compute_pass() as cp:
                    cursor = spy.ShaderCursor(cp.bind_pipeline(self._pipe_ssim))
                    cursor["gReference"] = self.renderer.volume_tex
                    cursor["gGaussianParamsRaw"] = self.renderer.gaussian_buffer
                    cursor["gTileContent"] = self.renderer.tile_content
                    cursor["gTileCounts"] = self.renderer.tile_counts
                    cursor["gMetricsResult"] = self._result_buf
                    self._fill_shared_params(cursor, vol_min, vol_max)
                    cursor["MetricsParams"]["ssimC1"] = self.config.ssim_c1
                    cursor["MetricsParams"]["ssimC2"] = self.config.ssim_c2
                    cursor["MetricsParams"]["ssimWindowHalf"] = wh
                    cursor["MetricsParams"]["ssimSliceAxis"] = axis
                    cursor["MetricsParams"]["ssimSliceIndex"] = s * slice_step
                    cp.dispatch(thread_count=(gx * 16, gy * 16, 1))

    # Readback and decode
    def _readback(self, frame: int, has_ssim: bool):
        buf = self._result_buf.to_numpy().view(np.float32)

        mse_sum = float(buf[SLOT_MSE_SUM])
        l1_sum = float(buf[SLOT_L1_SUM])
        iou_i = float(buf[SLOT_IOU_INTERSECT])
        iou_u = float(buf[SLOT_IOU_UNION])
        ssim_xy_sum = float(buf[SLOT_SSIM_XY])
        ssim_xz_sum = float(buf[SLOT_SSIM_XZ])
        ssim_yz_sum = float(buf[SLOT_SSIM_YZ])
        voxel_count = float(buf[SLOT_VOXEL_COUNT])

        # update frame always
        self._last_snap.frame = frame

        if voxel_count > 0:
            if self.config.enable_psnr:
                mse = mse_sum / voxel_count
                if mse > 1e-12:
                    self._last_snap.psnr = float(10.0 * np.log10(1.0 / mse))
                else:
                    self._last_snap.psnr = 100.0
            if self.config.enable_l1:
                self._last_snap.l1 = l1_sum / voxel_count

        if iou_u > 1e-12 and self.config.enable_iou:
            self._last_snap.iou = iou_i / iou_u

        # only touch SSIM fields when they were actually computed
        if has_ssim and self.config.enable_ssim:
            n = float(self.config.ssim_slices_per_axis)

            px_xy = float(self.vol_res[0] * self.vol_res[1])
            px_xz = float(self.vol_res[0] * self.vol_res[2])
            px_yz = float(self.vol_res[1] * self.vol_res[2])

            self._last_snap.ssim_xy = ssim_xy_sum / (n * px_xy)
            self._last_snap.ssim_xz = ssim_xz_sum / (n * px_xz)
            self._last_snap.ssim_yz = ssim_yz_sum / (n * px_yz)
            self._last_snap.ssim_mean = (
                self._last_snap.ssim_xy
                + self._last_snap.ssim_xz
                + self._last_snap.ssim_yz
            ) / 3.0

        self._frames.append(frame)
        self._psnr.append(self._last_snap.psnr)
        self._l1.append(self._last_snap.l1)
        self._iou.append(self._last_snap.iou)
        self._ssim_mean.append(self._last_snap.ssim_mean)

        cap = self.config.history_length
        if len(self._frames) > cap:
            self._frames = self._frames[-cap:]
            self._psnr = self._psnr[-cap:]
            self._l1 = self._l1[-cap:]
            self._iou = self._iou[-cap:]
            self._ssim_mean = self._ssim_mean[-cap:]

    # Public API
    def get_latest(self) -> dict:
        """Returns latest snapshot as a plain dict. Safe to pickle."""
        return self._last_snap.to_dict()

    def export_csv(self, path: str):
        """
        Dumps full history to CSV.
        Columns: frame, psnr, l1, iou, ssim_mean
        Call from your batch experiment script after training completes.
        """
        import csv

        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "psnr", "l1", "iou", "ssim_mean"])
            for row in zip(
                self._frames, self._psnr, self._l1, self._iou, self._ssim_mean
            ):
                w.writerow(
                    [
                        v if not (isinstance(v, float) and np.isnan(v)) else ""
                        for v in row
                    ]
                )
        print(f"[Metrics] {len(self._frames)} rows → {path}")

    # ImGui — inline, no begin/end
    def draw_ui_inline(self):
        snap = self._last_snap
        cfg = self.config

        imgui.dummy((0, 8))
        header_open, _ = imgui.collapsing_header("Quality Metrics", True)
        if not header_open:
            return

        # Scheduling controls on two rows
        imgui.push_item_width(150)
        _, cfg.fast_interval = imgui.slider_int(
            "Fast every N##mf", cfg.fast_interval, 1, 100
        )
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "PSNR / L1 / IoU run every N frames.\n"
                "Uses tile-based density eval — same path as training forward pass.\n"
                "1 is safe even at 196^3."
            )
        imgui.same_line()
        _, cfg.slow_interval = imgui.slider_int(
            "SSIM every N##ms", cfg.slow_interval, 1, 500
        )
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "SSIM dispatches 3 axes × ssim_slices_per_axis kernels.\n"
                "Each kernel re-evaluates tile density for a 2D slice.\n"
                "50 is a safe default."
            )
        imgui.pop_item_width()

        imgui.push_item_width(120)
        _, cfg.ssim_slices_per_axis = imgui.slider_int(
            "SSIM slices/axis##msl", cfg.ssim_slices_per_axis, 1, 64
        )
        if imgui.is_item_hovered():
            imgui.set_tooltip("More slices = more accurate, proportionally more work.")
        imgui.pop_item_width()

        imgui.dummy((0, 3))

        # Enable toggles
        _, cfg.enable_psnr = imgui.checkbox("PSNR##me", cfg.enable_psnr)
        imgui.same_line()
        _, cfg.enable_l1 = imgui.checkbox("L1##me", cfg.enable_l1)
        imgui.same_line()
        _, cfg.enable_iou = imgui.checkbox("IoU##me", cfg.enable_iou)
        imgui.same_line()
        _, cfg.enable_ssim = imgui.checkbox("SSIM##me", cfg.enable_ssim)

        imgui.separator()

        # Current values with red→green colouring
        def metric_row(label, val, good_dir, green_t, red_t):
            imgui.text(f"{label:<13}")
            imgui.same_line(150)
            if np.isnan(val):
                imgui.text_colored(imgui.ImVec4(0.4, 0.4, 0.4, 1), "  —")
                return
            t = (
                (val - red_t) / (green_t - red_t + 1e-9)
                if good_dir == "high"
                else (red_t - val) / (red_t - green_t + 1e-9)
            )
            t = float(np.clip(t, 0.0, 1.0))
            imgui.text_colored(imgui.ImVec4(1.0 - t, t, 0.0, 1.0), f"{val:>9.4f}")

        metric_row("PSNR (dB)", snap.psnr, "high", 35.0, 20.0)
        metric_row("L1 / MAE", snap.l1, "low", 0.01, 0.1)
        metric_row("IoU", snap.iou, "high", 0.7, 0.3)
        metric_row("SSIM mean", snap.ssim_mean, "high", 0.9, 0.5)

        if not np.isnan(snap.ssim_xy):
            imgui.text_colored(
                imgui.ImVec4(0.5, 0.5, 0.5, 1),
                f"  XY {snap.ssim_xy:.4f}   XZ {snap.ssim_xz:.4f}   YZ {snap.ssim_yz:.4f}",
            )

        imgui.text_colored(imgui.ImVec4(0.35, 0.35, 0.35, 1), f"  frame {snap.frame}")
        imgui.dummy((0, 5))

        # Trend plots — 55px tall, fills available width
        def safe_plot(label, data, uid):
            clean = [x for x in data if not (isinstance(x, float) and np.isnan(x))]
            if len(clean) < 2:
                imgui.text_colored(
                    imgui.ImVec4(0.4, 0.4, 0.4, 1), f"{label}: waiting..."
                )
                return
            arr = np.array(clean, dtype=np.float32)
            lo, hi = float(arr.min()), float(arr.max())
            if hi - lo < 1e-9:
                hi = lo + 1e-3
            imgui.text(label)
            imgui.plot_lines(
                f"##mp{uid}",
                arr,
                scale_min=lo,
                scale_max=hi,
                graph_size=imgui.ImVec2(-1, 55),
            )
            imgui.same_line(0, 4)
            imgui.text(f"{clean[-1]:.4f}")

        safe_plot("PSNR", self._psnr, "psnr")
        safe_plot("L1", self._l1, "l1")
        safe_plot("IoU", self._iou, "iou")
        safe_plot("SSIM mean", self._ssim_mean, "ssim")

        imgui.dummy((0, 3))
        imgui.separator()

        if imgui.button("Export CSV##mexp"):
            self.export_csv("metrics_export.csv")
        if imgui.is_item_hovered():
            imgui.set_tooltip("Writes full history to metrics_export.csv")
