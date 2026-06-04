#!/usr/bin/env python3
"""Export trained YOLO weights to TFLite (INT8) for edge deployment."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

# 减少 TensorFlow 刷屏；不影响转换
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DEFAULT_DEVICE = "cpu"  # Mac 导出 TFLite 用 CPU 更稳；MPS 仅用于训练


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export screw-hole model to TFLite")
    p.add_argument(
        "--weights",
        type=str,
        default="runs/detect/screw_hole/weights/best.pt",
        help="Path to .pt weights",
    )
    p.add_argument("--imgsz", type=int, default=640, help="Export image size")
    p.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help="cpu recommended for TFLite export on Mac",
    )
    p.add_argument(
        "--int8",
        action="store_true",
        default=True,
        help="INT8 quantization (default: on, slower)",
    )
    p.add_argument(
        "--no-int8",
        action="store_false",
        dest="int8",
        help="float32 TFLite — faster, larger file",
    )
    p.add_argument(
        "--data",
        type=str,
        default="dataset.yaml",
        help="Dataset YAML (required for INT8 calibration)",
    )
    return p.parse_args()


def resolve_weights(path: str) -> Path:
    w = Path(path)
    if w.exists():
        return w.resolve()
    matches = list(ROOT.glob("**/screw_hole/weights/best.pt"))
    if matches:
        return max(matches, key=lambda p: p.stat().st_mtime).resolve()
    raise FileNotFoundError(f"Weights not found: {path}\nTrain first: python train.py")


def main() -> None:
    args = parse_args()
    weights = resolve_weights(args.weights)
    data = str(ROOT / args.data) if args.data else None

    print("=" * 60)
    print("YOLO → TFLite export")
    print(f"  weights: {weights}")
    print(f"  int8:    {args.int8}")
    print(f"  device:  {args.device}")
    print("  Mac 上 INT8 导出常见 5–20 分钟，日志停在 TensorFlow 属正常。")
    print("  若太久：python export.py --no-int8 --imgsz 320")
    print("=" * 60)

    t0 = time.time()
    model = YOLO(str(weights))
    out = model.export(
        format="tflite",
        imgsz=args.imgsz,
        device=args.device,
        int8=args.int8,
        data=data if args.int8 else None,
    )
    print(f"\nDone in {time.time() - t0:.0f}s")
    print(f"Exported TFLite: {Path(out).resolve()}")
    if args.int8:
        print("INT8 — smaller, for edge devices.")
    else:
        print("float32 TFLite — faster export, larger file.")


if __name__ == "__main__":
    main()
