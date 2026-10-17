"""用 dataset/data.yaml 训练一份小检测权重，写出 dataset/weights/best.pt。

    python fafu_real/vision/dataset/train.py
    python fafu_real/vision/dataset/train.py --epochs 80 --device 0

基础模型是 yolov8n.pt（约 6MB）。第一次训练会从 Ultralytics 下载这一份，
不会加载 YOLO-World 和 CLIP。验证集 images/val 为空时，暂时用训练集做验证。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REAL = HERE.parents[1]
ROOT = REAL.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REAL))

import fafu_config as C  # noqa: E402
import yaml  # noqa: E402


def _count(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return len(list(folder.glob("*.jpg")) + list(folder.glob("*.png")))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="训练厨房检测权重")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=C.YOLO_IMGSZ)
    parser.add_argument("--device", default=C.YOLO_DEVICE)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args(argv)

    train_n = _count(HERE / "images" / "train")
    val_n = _count(HERE / "images" / "val")
    if train_n == 0:
        raise SystemExit("images/train 里没有照片。先运行 capture.py 拍照并 --label 拉框。")
    labeled = _count_labels(HERE / "labels" / "train")
    if labeled == 0:
        raise SystemExit("labels/train 里没有标签。运行 capture.py --label 给照片拉框。")

    data = yaml.safe_load((HERE / "data.yaml").read_text(encoding="utf-8"))
    data["path"] = str(HERE)
    if val_n == 0:
        print("[train] images/val 是空的，这次用训练集做验证。正式结果请留一部分图到 images/val。")
        data["val"] = data["train"]
    yaml_path = HERE / "data.resolved.yaml"
    yaml_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")

    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(HERE / "runs"),
        name="kitchen",
        exist_ok=True,
    )
    best = HERE / "runs" / "kitchen" / "weights" / "best.pt"
    if not best.is_file():
        raise SystemExit(f"训练结束但没有找到 {best}")
    out = Path(C.TRAINED_WEIGHTS)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, out)
    print(f"[train] 已复制到 {out}")
    print("[train] 预览：python fafu_real/vision/watch.py --yolo")
    return 0


def _count_labels(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return sum(1 for path in folder.glob("*.txt") if path.stat().st_size > 0)


if __name__ == "__main__":
    raise SystemExit(main())
