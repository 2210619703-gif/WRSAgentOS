"""D405 采集。

开流参数跟 ``wrs/drivers/devices/realsense/realsense_d400s.py`` 里的
``RealSenseD405`` 一样：中等 848×480 或高 1280×720，深度 z16、彩色 bgr8，30fps。
取帧改成比赛 demo 的做法：``rs.align`` 把深度对齐到彩色，再按针孔模型反投影。
这样后面的检测框可以直接索引同一像素的三维点。

    python fafu_real/vision/watch.py --live
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import fafu_config as C
import numpy as np

RESOLUTIONS = {"mid": (848, 480), "high": (1280, 720)}
FPS = 30
WARMUP_FRAMES = 8


class CameraError(RuntimeError):
    """相机打不开或取帧失败。"""


@dataclass
class CameraFrame:
    """一帧对齐后的数据。``points`` 与 ``color`` 逐像素对应，无效深度为 0。"""

    color: np.ndarray
    depth_m: np.ndarray
    points: np.ndarray
    intrinsics: dict

    @property
    def width(self) -> int:
        return int(self.color.shape[1])

    @property
    def height(self) -> int:
        return int(self.color.shape[0])

    def valid_ratio(self) -> float:
        return float(np.count_nonzero(self.depth_m > 0) / self.depth_m.size)

    def save(self, out_dir: str) -> str:
        import cv2

        out_dir = os.path.abspath(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, "color.png"), self.color)
        np.save(os.path.join(out_dir, "depth_m.npy"), self.depth_m.astype(np.float32))
        with open(os.path.join(out_dir, "intrinsics.json"), "w", encoding="utf-8") as f:
            json.dump(self.intrinsics, f, ensure_ascii=False, indent=2)
        return out_dir


def deproject(depth_m: np.ndarray, intr: dict) -> np.ndarray:
    """对齐后的深度图 → HxWx3 相机系点（米）。"""
    h, w = depth_m.shape[:2]
    fx, fy = float(intr["fx"]), float(intr["fy"])
    cx, cy = float(intr["cx"]), float(intr["cy"])
    us, vs = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    z = depth_m.astype(np.float32)
    bad = (z <= 0) | (z > C.MAX_DEPTH_M) | ~np.isfinite(z)
    z = np.where(bad, 0.0, z)
    pts = np.stack([(us - cx) / fx * z, (vs - cy) / fy * z, z], axis=-1)
    pts[bad] = 0.0
    return pts


def _colorize_depth(depth_m: np.ndarray) -> np.ndarray:
    import cv2

    vis = np.clip(depth_m / max(C.MAX_DEPTH_M, 1e-6), 0.0, 1.0)
    vis = np.where(depth_m > 0, vis, 0.0)
    u8 = (vis * 255.0).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_JET)


class D405Camera:
    """用 ``with`` 打开，退出时关掉管道。"""

    def __init__(self, resolution: str = C.CAMERA_RESOLUTION, serial: str | None = C.CAMERA_SERIAL, *, logger=print):
        if resolution not in RESOLUTIONS:
            raise CameraError(f"分辨率只能是 {tuple(RESOLUTIONS)}，收到 {resolution!r}")
        self.resolution = resolution
        self.serial = serial or None
        self.log = logger
        self._pipe = None
        self._align = None
        self._depth_scale = 0.001
        self._intr: dict | None = None

    def __enter__(self) -> D405Camera:
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def open(self) -> None:
        try:
            import pyrealsense2 as rs
        except Exception as exc:
            raise CameraError(f"导入 pyrealsense2 失败：{exc}") from exc

        w, h = RESOLUTIONS[self.resolution]
        pipe = rs.pipeline()
        conf = rs.config()
        if self.serial:
            conf.enable_device(str(self.serial))
        conf.enable_stream(rs.stream.depth, w, h, rs.format.z16, FPS)
        conf.enable_stream(rs.stream.color, w, h, rs.format.bgr8, FPS)
        try:
            profile = pipe.start(conf)
        except Exception as exc:
            raise CameraError(
                f"D405 打不开（{w}x{h}）：{exc}\n检查：USB3 口、相机是否被别的程序占用。"
            ) from exc

        self._pipe = pipe
        self._align = rs.align(rs.stream.color)
        self._depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
        try:
            color_sensor = profile.get_device().first_color_sensor()
            if color_sensor.supports(rs.option.enable_auto_exposure):
                color_sensor.set_option(rs.option.enable_auto_exposure, 1.0)
        except Exception:
            pass
        for _ in range(WARMUP_FRAMES):
            try:
                pipe.wait_for_frames(timeout_ms=2000)
            except Exception:
                break
        frames = self._align.process(pipe.wait_for_frames(timeout_ms=5000))
        intr = frames.get_color_frame().profile.as_video_stream_profile().intrinsics
        self._intr = {"fx": float(intr.fx), "fy": float(intr.fy), "cx": float(intr.ppx), "cy": float(intr.ppy)}
        self.log(
            f"[cam] D405 就绪 {w}x{h}@{FPS}  depth_scale={self._depth_scale:.6f} "
            f"fx={intr.fx:.1f} fy={intr.fy:.1f}"
        )

    def close(self) -> None:
        if self._pipe is not None:
            try:
                self._pipe.stop()
            except Exception:
                pass
            self._pipe = None

    @property
    def intrinsics(self) -> dict:
        if self._intr is None:
            raise CameraError("相机还没 open()")
        return dict(self._intr)

    def capture(self, timeout_ms: int = 5000) -> CameraFrame:
        if self._pipe is None:
            raise CameraError("相机还没 open()")
        try:
            frames = self._align.process(self._pipe.wait_for_frames(timeout_ms=timeout_ms))
        except Exception as exc:
            raise CameraError(f"取帧失败：{exc}") from exc
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            raise CameraError("这一帧缺深度或彩色")
        color = np.asanyarray(color_frame.get_data()).copy()
        depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self._depth_scale
        intr = self.intrinsics
        return CameraFrame(color=color, depth_m=depth_m, points=deproject(depth_m, intr), intrinsics=intr)


def live_preview(cam: D405Camera) -> None:
    """彩色和深度并排。按 q 或 Esc 退出。不加载任何权重。"""
    import cv2

    win = "D405 live  (q 退出)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print("[cam] 实时预览已打开，按 q 或 Esc 关闭", flush=True)
    while True:
        frame = cam.capture(timeout_ms=2000)
        depth = _colorize_depth(frame.depth_m)
        view = np.hstack([frame.color, depth])
        cv2.putText(
            view,
            f"{frame.width}x{frame.height}  depth {frame.valid_ratio() * 100:.0f}%",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(win, view)
        if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
            break
    cv2.destroyWindow(win)
