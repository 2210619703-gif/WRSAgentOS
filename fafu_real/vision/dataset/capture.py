"""用 D405 拍训练图，并在图上拉框写成 YOLO 标签。

拍照（不加载权重）::

    python fafu_real/vision/dataset/capture.py

标注已经拍下的图::

    python fafu_real/vision/dataset/capture.py --label

类别在同目录 data.yaml。照片在 images/train，标签在 labels/train。
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REAL = HERE.parents[1]
ROOT = REAL.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REAL))

import yaml  # noqa: E402

from camera import CameraError, D405Camera  # noqa: E402

IMAGES = HERE / "images" / "train"
LABELS = HERE / "labels" / "train"


def class_names() -> list[str]:
    data = yaml.safe_load((HERE / "data.yaml").read_text(encoding="utf-8"))
    names = data["names"]
    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]
    return list(names)


def _next_image() -> Path:
    IMAGES.mkdir(parents=True, exist_ok=True)
    nums = []
    for path in IMAGES.glob("*.jpg"):
        try:
            nums.append(int(path.stem))
        except ValueError:
            pass
    return IMAGES / f"{(max(nums) + 1) if nums else 1:04d}.jpg"


def shoot() -> int:
    import cv2

    try:
        cam_cm = D405Camera()
        cam_cm.open()
    except CameraError as exc:
        print(exc)
        return 1
    win = "D405 capture  (s 保存, q 退出)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print("[capture] s 保存当前画面，q 退出。保存后用 --label 拉框。", flush=True)
    try:
        while True:
            frame = cam_cm.capture(timeout_ms=2000)
            view = frame.color.copy()
            cv2.putText(
                view, f"saved next {_next_image().name}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
            )
            cv2.imshow(win, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                path = _next_image()
                cv2.imwrite(str(path), frame.color)
                print(f"[capture] {path}")
    finally:
        cam_cm.close()
        cv2.destroyAllWindows()
    return 0


def _yolo_line(cls: int, box, width: int, height: int) -> str:
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2) / width
    cy = ((y1 + y2) / 2) / height
    w = abs(x2 - x1) / width
    h = abs(y2 - y1) / height
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def label_images() -> int:
    import cv2

    names = class_names()
    images = sorted(IMAGES.glob("*.jpg"))
    if not images:
        print(f"[label] {IMAGES} 里没有照片，先运行不带 --label 的拍照。")
        return 1
    LABELS.mkdir(parents=True, exist_ok=True)
    state = {"down": None, "boxes": [], "cls": 0, "image": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["down"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and state["down"] is not None:
            x1, y1 = state["down"]
            if abs(x - x1) > 4 and abs(y - y1) > 4:
                state["boxes"].append((state["cls"], (min(x1, x), min(y1, y), max(x1, x), max(y1, y))))
            state["down"] = None

    win = "label"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    print("[label] 鼠标拉框。数字键换类别，z 撤销，s 保存并下一张，q 退出。", flush=True)
    print("[label] " + "  ".join(f"{i}:{name}" for i, name in enumerate(names)), flush=True)
    index = 0
    while index < len(images):
        path = images[index]
        image = cv2.imread(str(path))
        state["image"] = image
        state["boxes"] = []
        label_path = LABELS / f"{path.stem}.txt"
        if label_path.is_file() and image is not None:
            h, w = image.shape[:2]
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
                x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
                x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
                state["boxes"].append((cls, (x1, y1, x2, y2)))
        while True:
            if state["image"] is None:
                print(f"[label] 读不了 {path}")
                index += 1
                break
            view = state["image"].copy()
            h, w = view.shape[:2]
            for cls, (x1, y1, x2, y2) in state["boxes"]:
                cv2.rectangle(view, (x1, y1), (x2, y2), (0, 220, 0), 2)
                title = names[cls] if cls < len(names) else str(cls)
                cv2.putText(view, title, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
            current = names[state["cls"]] if state["cls"] < len(names) else str(state["cls"])
            cv2.putText(
                view, f"{index + 1}/{len(images)}  {path.name}  class {state['cls']}:{current}",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
            )
            cv2.imshow(win, view)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                cv2.destroyAllWindows()
                return 0
            if ord("0") <= key <= ord("9"):
                state["cls"] = key - ord("0")
            elif key == ord("z") and state["boxes"]:
                state["boxes"].pop()
            elif key == ord("s"):
                lines = [_yolo_line(cls, box, w, h) for cls, box in state["boxes"]]
                label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                print(f"[label] {label_path}  {len(lines)} 个框")
                index += 1
                break
    cv2.destroyAllWindows()
    return 0


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else list(argv)
    if args == ["--label"]:
        return label_images()
    if args:
        raise SystemExit("用法：python fafu_real/vision/dataset/capture.py [--label]")
    return shoot()


if __name__ == "__main__":
    raise SystemExit(main())
