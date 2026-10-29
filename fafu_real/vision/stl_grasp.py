"""用碗托和漏勺的 STL 把抓取点放在把手上，并判断哪只碗是带把手的生食碗。

``bowl_support.stl`` 是带把手的生食碗。检测框里的深度点对得上这只模型时，那只碗记为
``bowl_raw``，朝向跟着把手。对不上的碗记为 ``bowl_serve``，仍抓碗沿。
两只都对不上时，退回按世界坐标 Y 区分。

``skimmer.stl`` 是漏勺。对得上时朝向跟着把手；对不上就保持原来的朝向，抓取点仍在模型把手上。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_REAL = Path(__file__).resolve().parents[1]
BOWL_STL = _REAL / "bowl_support.stl"
SKIMMER_STL = _REAL / "skimmer.stl"


@dataclass(frozen=True)
class Template:
    size: tuple[float, float, float]
    grasp_offset: tuple[float, float, float]
    close_axis_deg: float
    body_radius: float
    r_inner: float
    r_outer: float
    handle_angle: float
    handle_reach: float
    z_band: float = 0.05
    min_outer: int = 12
    max_span_deg: float = 80.0
    reach_tol: float = 0.04

    def shape(self) -> dict:
        return {
            "size": self.size,
            "grasp_offset": self.grasp_offset,
            "close_axis_deg": self.close_axis_deg,
        }


def load_vertices(path: Path) -> np.ndarray:
    """二进制 STL 的三角顶点，米。"""
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size < 84:
        raise ValueError(f"STL 太小：{path}")
    count = int(np.frombuffer(raw, dtype=np.uint32, count=1, offset=80)[0])
    if 84 + count * 50 > raw.size:
        raise ValueError(f"STL 长度不对：{path}")
    face = np.ascontiguousarray(raw[84 : 84 + count * 50].reshape(count, 50)[:, 12:48])
    verts = face.view(np.float32).reshape(-1, 3).astype(np.float64)
    if len(verts) > 80_000:
        verts = verts[:: max(1, len(verts) // 80_000)]
    return verts


def _robust_center(xy: np.ndarray, radius: float) -> np.ndarray | None:
    """反复丢掉半径外的点，得到主体中心。把手在半径外，不会把中心拉偏。"""
    if len(xy) < 20:
        return None
    center = np.median(xy, axis=0)
    for _ in range(5):
        keep = np.linalg.norm(xy - center, axis=1) <= radius
        if int(keep.sum()) < 20:
            return None
        center = np.median(xy[keep], axis=0)
    return center


def _angle_span(xy: np.ndarray, center: np.ndarray) -> float:
    ang = np.sort(np.arctan2(xy[:, 1] - center[1], xy[:, 0] - center[0]))
    if len(ang) < 2:
        return 0.0
    gaps = np.diff(ang)
    wrap = ang[0] + 2.0 * math.pi - ang[-1]
    return float(2.0 * math.pi - np.append(gaps, wrap).max())


def _from_mesh(path: Path, **spec) -> Template:
    verts = load_vertices(path)
    center_xy = _robust_center(verts[:, :2], spec["body_radius"])
    if center_xy is None:
        raise ValueError(f"分不出主体：{path.name}")
    dist = np.linalg.norm(verts[:, :2] - center_xy, axis=1)
    outer = verts[(dist >= spec["r_inner"]) & (dist <= spec["r_outer"])]
    if len(outer) < int(spec.get("min_outer", 12)):
        raise ValueError(f"分不出把手：{path.name}，外侧点 {len(outer)}")
    grasp = (outer.min(axis=0) + outer.max(axis=0)) / 2.0
    origin_z = float(spec["size"][2]) / 2.0
    offset = grasp - np.array([center_xy[0], center_xy[1], origin_z])
    reach = float(np.hypot(offset[0], offset[1]))
    return Template(
        grasp_offset=(float(offset[0]), float(offset[1]), float(offset[2])),
        handle_angle=float(math.atan2(offset[1], offset[0])),
        handle_reach=reach,
        **spec,
    )


_TEMPLATES: dict[str, Template] | None = None


def templates() -> dict[str, Template]:
    """第一次调用时从 STL 算出主体尺寸和把手抓取点，之后复用。"""
    global _TEMPLATES
    if _TEMPLATES is None:
        _TEMPLATES = {
            "bowl_raw": _from_mesh(
                BOWL_STL,
                size=(0.168, 0.168, 0.085),
                close_axis_deg=90.0,
                body_radius=0.090,
                r_inner=0.095,
                r_outer=0.160,
            ),
            "strainer": _from_mesh(
                SKIMMER_STL,
                size=(0.135, 0.120, 0.135),
                close_axis_deg=0.0,
                body_radius=0.100,
                r_inner=0.160,
                r_outer=0.320,
                min_outer=8,
            ),
        }
    return _TEMPLATES


def mesh_shapes() -> dict[str, dict]:
    return {name: item.shape() for name, item in templates().items()}


def fit_handle(points: np.ndarray, template: Template) -> tuple[np.ndarray, float, float] | None:
    """深度点里若有一侧凸出，且距离和模型把手一致，返回主体平面中心、朝向、分数。

    朝向是绕竖直轴的弧度，使物体自身的把手方向转到看见的那一侧。分不出来返回 None。
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    pts = pts[np.linalg.norm(pts, axis=1) > 1e-4]
    if len(pts) < 40:
        return None
    center = _robust_center(pts[:, :2], template.body_radius)
    if center is None:
        return None
    dist = np.linalg.norm(pts[:, :2] - center, axis=1)
    body = dist <= template.body_radius
    if int(body.sum()) < 20:
        return None
    z0 = float(np.median(pts[body, 2]))
    outer = (dist >= template.r_inner) & (dist <= template.r_outer)
    outer &= np.abs(pts[:, 2] - z0) <= template.z_band
    if int(outer.sum()) < template.min_outer:
        return None
    if _angle_span(pts[outer, :2], center) > math.radians(template.max_span_deg):
        return None
    if abs(float(np.median(dist[outer])) - template.handle_reach) > template.reach_tol:
        return None
    direction = np.median(pts[outer, :2] - center, axis=0)
    yaw = math.atan2(float(direction[1]), float(direction[0])) - template.handle_angle
    yaw = (yaw + math.pi) % (2.0 * math.pi) - math.pi
    return center, yaw, float(outer.sum())
