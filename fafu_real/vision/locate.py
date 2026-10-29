"""用自己训练的检测权重看一帧，把手眼变到世界系的表面点交给场景。

不写 kitchen_scene.py。那边只保留示教关节角。这里看见的物体优先，没看见的不动。

类别对应：food → noodles，其余名字和场景物体相同（strainer、pot、spoon、plate）。
碗只有一个类别。检测框里的深度点对得上 bowl_support.stl 的是生食碗，抓它的把手；
对不上的是盛出碗。两只都对不上时，世界坐标 Y 较大的是生食碗，较小的是盛出碗。
漏勺对得上 skimmer.stl 时，朝向跟着把手。
"""

from __future__ import annotations

import glob
import math
import os

import fafu_config as C
import numpy as np

# 训练类别 → kitchen_scene 里的物体名。碗单独按位置分成两只，不写在这里。
CLASS_TO_OBJECT = {
    "food": "noodles",
    "noodles": "noodles",
    "strainer": "strainer",
    "pot": "pot",
    "spoon": "spoon",
    "plate": "plate",
}


class LocateError(RuntimeError):
    pass


def trained_weights() -> str:
    """优先 dataset/weights/best.pt，否则用 vision 目录下唯一的 best*.pt。"""
    if os.path.isfile(C.TRAINED_WEIGHTS):
        return C.TRAINED_WEIGHTS
    folder = os.path.join(C.HERE, "vision")
    found = sorted(glob.glob(os.path.join(folder, "best*.pt")))
    if len(found) == 1:
        return found[0]
    if not found:
        raise LocateError(
            f"还没有训练权重：{C.TRAINED_WEIGHTS}\n"
            "或者把 best.pt 放在 fafu_real/vision/ 下（只能有一份 best*.pt）。"
        )
    names = ", ".join(os.path.basename(p) for p in found)
    raise LocateError(f"fafu_real/vision/ 里有多份权重（{names}），请只留一份 best*.pt")


def surface_point(points: np.ndarray, xyxy) -> np.ndarray | None:
    """框中心附近有效深度的中位数，相机系，米。边缘上的背景点不用。"""
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw, hh = (x2 - x1) * 0.25, (y2 - y1) * 0.25
    h, w = points.shape[:2]
    xa = max(0, int(cx - hw))
    xb = min(w, int(cx + hw) + 1)
    ya = max(0, int(cy - hh))
    yb = min(h, int(cy + hh) + 1)
    if xb <= xa or yb <= ya:
        return None
    patch = points[ya:yb, xa:xb].reshape(-1, 3)
    good = patch[np.linalg.norm(patch, axis=1) > 1e-4]
    if len(good) < 15:
        return None
    return np.median(good, axis=0)


def world_point(flange_of, q, t_flange_cam, p_cam) -> np.ndarray:
    """世界点 = FK(法兰) @ T_法兰_相机 @ 相机点。"""
    homo = np.ones(4)
    homo[:3] = np.asarray(p_cam, dtype=float)
    return (flange_of(q) @ np.asarray(t_flange_cam, dtype=float) @ homo)[:3]


def _box_points(points: np.ndarray, xyxy) -> np.ndarray:
    """检测框里的有效相机点，含边缘。把手通常不在框的正中。"""
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    height, width = points.shape[:2]
    xa, xb = max(0, int(math.floor(x1))), min(width, int(math.ceil(x2)))
    ya, yb = max(0, int(math.floor(y1))), min(height, int(math.ceil(y2)))
    if xb <= xa or yb <= ya:
        return np.zeros((0, 3))
    patch = points[ya:yb, xa:xb].reshape(-1, 3)
    return patch[np.linalg.norm(patch, axis=1) > 1e-4]


def _world_points(flange_of, q, t_flange_cam, cam_pts: np.ndarray) -> np.ndarray:
    if len(cam_pts) == 0:
        return cam_pts
    posed = flange_of(q) @ np.asarray(t_flange_cam, dtype=float)
    homo = np.ones((len(cam_pts), 4))
    homo[:, :3] = cam_pts
    return (homo @ posed.T)[:, :3]


def _with_handle(name, surface, score, fit, template) -> tuple:
    """对上模型时，平面中心用主体，高度仍由框中心的表面点减去半个高度。"""
    center_xy, yaw, _fit_score = fit
    center = np.array(
        [float(center_xy[0]), float(center_xy[1]), float(surface[2]) - template.size[2] / 2.0]
    )
    return (name, center, score, float(yaw))


def _assign_bowls_by_y(bowls, logger) -> list[tuple]:
    """对不上带把手的模型时：Y 大的是生食碗，Y 小的是盛出碗。只看见一只时当作生食碗。"""
    ranked = sorted(bowls, key=lambda item: item[1], reverse=True)[:2]
    ranked.sort(key=lambda item: float(item[0][1]), reverse=True)
    roles = ("bowl_raw", "bowl_serve")
    hits = [
        (role, point, score, None) for role, (point, score) in zip(roles, ranked, strict=False)
    ]
    if len(ranked) == 1:
        logger("[yolo] 只看见一只碗，当作生食碗 bowl_raw；盛出碗仍用示教位置")
    elif len(ranked) == 2:
        logger(
            "[yolo] 两只碗按 Y 分开："
            f"生食碗 Y={ranked[0][0][1]:+.3f}，盛出碗 Y={ranked[1][0][1]:+.3f}"
        )
    return hits


def _assign_bowls(bowls, template, logger) -> list[tuple]:
    """每项是 (表面点, 置信度, 世界系点云)。对得上 STL 的最高分是生食碗。"""
    from vision.stl_grasp import fit_handle

    if template is None:
        return _assign_bowls_by_y([(point, score) for point, score, _cloud in bowls], logger)
    parsed = []
    for surface, score, cloud in bowls:
        parsed.append((surface, score, fit_handle(cloud, template)))
    matched = [item for item in parsed if item[2] is not None]
    if not matched:
        logger("[yolo] 没有碗对上带把手的模型，仍按 Y 区分")
        return _assign_bowls_by_y([(point, score) for point, score, _fit in parsed], logger)
    matched.sort(key=lambda item: item[2][2], reverse=True)
    best = matched[0]
    raw = _with_handle("bowl_raw", best[0], best[1], best[2], template)
    logger(
        "[yolo] 这只碗对上 bowl_support.stl，当作生食碗，抓把手 "
        f"Y={best[0][1]:+.3f}"
    )
    others = [item for item in parsed if item is not best]
    if not others:
        logger("[yolo] 只对上一只生食碗；盛出碗仍用示教位置")
        return [raw]
    other = max(others, key=lambda item: item[1])
    if other[2] is None:
        logger("[yolo] 另一只碗对不上这只模型，当作盛出碗，抓碗沿")
    else:
        logger("[yolo] 两只都像带把手的碗，取得分高的当生食碗，另一只当盛出碗")
    return [raw, ("bowl_serve", other[0], other[1], None)]


def _templates():
    try:
        from vision.stl_grasp import templates
    except ImportError:
        return {}
    try:
        return templates()
    except (OSError, ValueError) as exc:
        raise LocateError(f"读 STL 失败：{exc}") from exc


def hits_from_result(result, points, flange_of, q, t_flange_cam, logger=print):
    """每个场景物体只留置信度最高的一框。碗用 STL 区分生食碗和盛出碗。"""
    names = result.names or {}
    boxes = result.boxes
    models = _templates()
    best: dict[str, tuple] = {}
    bowls = []
    n = 0 if boxes is None else len(boxes)
    for i in range(n):
        cls_name = str(names.get(int(boxes[i].cls), int(boxes[i].cls)))
        score = float(boxes[i].conf)
        xyxy = boxes[i].xyxy[0].cpu().numpy()
        cam = surface_point(points, xyxy)
        if cam is None:
            logger(f"[yolo] {cls_name} 框里没有有效深度，这一框不用")
            continue
        world = world_point(flange_of, q, t_flange_cam, cam)
        cloud = _world_points(flange_of, q, t_flange_cam, _box_points(points, xyxy))
        if cls_name == "bowl":
            bowls.append((world, score, cloud))
            continue
        target = CLASS_TO_OBJECT.get(cls_name)
        if target is None:
            logger(f"[yolo] 看见 {cls_name}（{score:.2f}），场景里没有对应物体，忽略")
            continue
        hit = (target, world, score, None)
        model = models.get(target)
        if model is not None:
            from vision.stl_grasp import fit_handle

            fitted = fit_handle(cloud, model)
            if fitted is None:
                logger(f"[yolo] {target} 没对上 STL，把手方向仍用当前朝向")
            else:
                hit = _with_handle(target, world, score, fitted, model)
                logger(f"[yolo] {target} 对上 STL，抓把手")
        prev = best.get(target)
        if prev is None or score > prev[2]:
            best[target] = hit
    return [*best.values(), *_assign_bowls(bowls, models.get("bowl_raw"), logger)]


def detect_objects(q, flange_of, logger=print):
    """在当前关节角下拍一帧并检测。调用前手臂应已经停在看得见桌面的位置。"""
    import handeye
    from camera import CameraError, D405Camera
    from ultralytics import YOLO

    weights = trained_weights()
    t_fc = handeye.load()
    logger(f"[yolo] 权重 {os.path.basename(weights)}")
    try:
        with D405Camera() as cam:
            frame = cam.capture()
    except CameraError as exc:
        raise LocateError(str(exc)) from exc
    logger(f"[yolo] 拍到一帧，深度有效 {frame.valid_ratio() * 100:.0f}%")
    model = YOLO(weights)
    result = model.predict(
        frame.color,
        conf=C.YOLO_CONFIDENCE,
        iou=C.YOLO_IOU,
        imgsz=C.YOLO_IMGSZ,
        verbose=False,
        device=C.YOLO_DEVICE,
    )[0]
    return hits_from_result(result, frame.points, flange_of, q, t_fc, logger)
