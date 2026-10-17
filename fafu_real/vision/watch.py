"""D405 实时窗口，对齐比赛仓库 demo/fruit_yolo_realtime.py。

默认和 ``--live`` 一样，只看相机，不加载权重::

    python fafu_real/vision/watch.py --live
    python fafu_real/vision/watch.py --yolo

``--live`` 左边彩色、右边深度。``--yolo`` 在彩色上画框，使用
``fafu_real/vision/dataset/weights/best.pt``（自己训练出来的那份）。
q / Esc 退出。有权重时 ``[`` ``]`` 调阈值，``-`` ``=`` 调输入尺寸，``s`` 存图。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REAL = Path(__file__).resolve().parents[1]
ROOT = REAL.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REAL))

import fafu_config as C  # noqa: E402
from camera import CameraError, D405Camera, live_preview  # noqa: E402


def live_yolo(cam: D405Camera, weights: str, conf: float, iou: float, imgsz: int) -> None:
    import cv2
    from ultralytics import YOLO

    print(f"[cam] 加载 {os.path.basename(weights)}", flush=True)
    model = YOLO(weights)
    names = model.names or {}
    win = "D405 + YOLO"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    save_i = 0
    save_dir = os.path.join(os.path.dirname(__file__), "dataset", "outputs")
    print("[cam] q 退出；[ / ] 调阈值；- / = 调输入尺寸；s 存图", flush=True)
    while True:
        frame = cam.capture(timeout_ms=2000)
        t0 = time.perf_counter()
        result = model.predict(
            frame.color, conf=min(conf, 0.08), iou=iou, imgsz=imgsz, verbose=False, device=C.YOLO_DEVICE
        )[0]
        infer_ms = (time.perf_counter() - t0) * 1000.0
        view = frame.color.copy()
        boxes = result.boxes
        n_all = 0 if boxes is None else len(boxes)
        n_pass = 0
        for i in range(n_all):
            x1, y1, x2, y2 = (int(v) for v in boxes[i].xyxy[0].cpu().numpy())
            score = float(boxes[i].conf)
            name = str(names.get(int(boxes[i].cls), int(boxes[i].cls)))
            passed = score >= conf
            n_pass += int(passed)
            color = (0, 220, 0) if passed else (0, 180, 255)
            cv2.rectangle(view, (x1, y1), (x2, y2), color, 2 if passed else 1)
            cv2.putText(
                view, f"{name} {score:.2f}", (x1, max(16, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
            )
        cv2.putText(
            view,
            f"{frame.width}x{frame.height}  n={n_pass}/{n_all}  conf={conf:.2f}  imgsz={imgsz}  {infer_ms:.0f}ms",
            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.imshow(win, view)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("["):
            conf = max(0.05, round(conf - 0.05, 2))
        elif key == ord("]"):
            conf = min(0.95, round(conf + 0.05, 2))
        elif key == ord("-"):
            imgsz = max(320, imgsz - 64)
        elif key in (ord("="), ord("+")):
            imgsz = min(1280, imgsz + 64)
        elif key == ord("s"):
            os.makedirs(save_dir, exist_ok=True)
            out = os.path.join(save_dir, f"live_{save_i:04d}.jpg")
            cv2.imwrite(out, view)
            print(f"[cam] 已保存 {out}")
            save_i += 1
    cv2.destroyWindow(win)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="D405 实时预览")
    parser.add_argument("--live", action="store_true", help="彩色和深度并排，按 q 退出")
    parser.add_argument("--yolo", action="store_true", help="彩色上叠加训练好的检测框（会打开实时窗口）")
    parser.add_argument("--weights", default=C.TRAINED_WEIGHTS)
    parser.add_argument("--resolution", default=C.CAMERA_RESOLUTION, choices=tuple(("mid", "high")))
    parser.add_argument("--serial", default=C.CAMERA_SERIAL)
    parser.add_argument("--conf", type=float, default=C.YOLO_CONFIDENCE)
    parser.add_argument("--iou", type=float, default=C.YOLO_IOU)
    parser.add_argument("--imgsz", type=int, default=C.YOLO_IMGSZ)
    args = parser.parse_args(argv)
    if args.yolo and not os.path.isfile(args.weights):
        raise SystemExit(f"还没有训练权重：{args.weights}\n先按 fafu_real/vision/dataset/README.md 拍照并训练。")
    try:
        with D405Camera(args.resolution, args.serial) as cam:
            if args.yolo:
                live_yolo(cam, args.weights, args.conf, args.iou, args.imgsz)
            else:
                live_preview(cam)  # 不带参数，或只带 --live
    except CameraError as exc:
        print(exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
