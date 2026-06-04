#!/usr/bin/env python3
"""Train YOLOv8n for PCB screw-hole detection on Apple Silicon (MPS)."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DEFAULT_DEVICE = "mps"
DEFAULT_DATA = "dataset.yaml"
DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_PROJECT = str(ROOT / "runs" / "detect")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train screw-hole detector (YOLOv8n, MPS)")
    p.add_argument("--data", type=str, default=DEFAULT_DATA, help="Dataset YAML path")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Base weights")
    p.add_argument("--epochs", type=int, default=100, help="Training epochs")
    p.add_argument("--imgsz", type=int, default=640, help="Image size")
    p.add_argument("--batch", type=int, default=16, help="Batch size (reduce if OOM on 16GB)")
    p.add_argument("--device", type=str, default=DEFAULT_DEVICE, help='Device, e.g. "mps", "cpu"')
    p.add_argument("--project", type=str, default=DEFAULT_PROJECT, help="Save project dir (absolute)")
    p.add_argument("--name", type=str, default="screw_hole", help="Run name")
    p.add_argument("--patience", type=int, default=50, help="Early stopping patience")
    p.add_argument("--workers", type=int, default=4, help="Dataloader workers")
    p.add_argument("--resume", action="store_true", help="Resume last run")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset config not found: {data_path}\n"
            "Prepare data under data/screw_holes/ and edit dataset.yaml."
        )

    model = YOLO(args.model)
    results = model.train(
        data=str(data_path if data_path.is_absolute() else ROOT / data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        workers=args.workers,
        resume=args.resume,
        save=True,
        plots=True,
        val=True,
        pretrained=True,
    )

    best = Path(args.project) / args.name / "weights" / "best.pt"
    if not best.exists():
        nested = list(ROOT.glob(f"**/{args.name}/weights/best.pt"))
        if nested:
            best = max(nested, key=lambda p: p.stat().st_mtime)
    print("\n--- Training finished ---")
    print(f"Best weights: {best.resolve()}")
    print(f"TensorBoard:  tensorboard --logdir {Path(args.project).resolve()}")
    print(f"Metrics CSV:  {Path(args.project) / args.name / 'results.csv'}")
    if results is not None:
        print(f"Last metrics: {getattr(results, 'results_dict', results)}")


if __name__ == "__main__":
    main()
