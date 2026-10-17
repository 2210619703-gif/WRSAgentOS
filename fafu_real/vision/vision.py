"""观察位拍一帧。有训练权重时在彩色图上画框，并可用手眼把框中心变到相机系。

模仿 demo 的 ``--mode vision``：只识别、不抓。

    python fafu_real/vision/vision.py
    python fafu_real/vision/vision.py --yolo
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REAL = Path(__file__).resolve().parents[1]
ROOT = REAL.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REAL))

import fafu_config as C  # noqa: E402
from camera import CameraError, D405Camera  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="D405 拍一帧，可选 YOLO")
    parser.add_argument("--yolo", action="store_true")
    parser.add_argument("--weights", default=C.TRAINED_WEIGHTS)
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "dataset", "outputs", "frame"))
    args = parser.parse_args(argv)
    if args.yolo and not os.path.isfile(args.weights):
        raise SystemExit(f"还没有训练权重：{args.weights}")
    try:
        with D405Camera() as cam:
            frame = cam.capture()
    except CameraError as exc:
        print(exc)
        return 1
    saved = frame.save(args.out)
    print(f"[vision] 已保存 {saved}  深度有效 {frame.valid_ratio() * 100:.0f}%")
    if not args.yolo:
        return 0
    from ultralytics import YOLO

    model = YOLO(args.weights)
    result = model.predict(frame.color, conf=C.YOLO_CONFIDENCE, iou=C.YOLO_IOU, imgsz=C.YOLO_IMGSZ, verbose=False, device=C.YOLO_DEVICE)[0]
    names = model.names or {}
    boxes = result.boxes
    n = 0 if boxes is None else len(boxes)
    for i in range(n):
        x1, y1, x2, y2 = (float(v) for v in boxes[i].xyxy[0].cpu().numpy())
        u, v = int((x1 + x2) / 2), int((y1 + y2) / 2)
        point = frame.points[min(v, frame.height - 1), min(u, frame.width - 1)]
        name = str(names.get(int(boxes[i].cls), int(boxes[i].cls)))
        print(f"[vision] {name} {float(boxes[i].conf):.2f}  相机系 xyz {point.tolist()}")
    if n == 0:
        print("[vision] 没有检出目标")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
