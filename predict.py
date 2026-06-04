#!/usr/bin/env python3
"""Run screw-hole detection on a single image."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

DEFAULT_DEVICE = "mps"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict screw holes on one image")
    p.add_argument("--source", type=str, required=True, help="Image path")
    p.add_argument(
        "--weights",
        type=str,
        default="runs/detect/screw_hole/weights/best.pt",
        help=".pt or .tflite weights",
    )
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, help="Inference size")
    p.add_argument("--device", type=str, default=DEFAULT_DEVICE, help='Device ("mps" for .pt)')
    p.add_argument("--save", action="store_true", default=True, help="Save annotated image")
    p.add_argument("--project", type=str, default="runs/predict", help="Output directory")
    p.add_argument("--name", type=str, default="exp", help="Run subfolder")
    return p.parse_args()


def print_detections(result) -> None:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        print("No screw holes detected.")
        return
    names = result.names
    print(f"\nDetected {len(boxes)} screw hole(s):\n")
    for i, box in enumerate(boxes):
        xyxy = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = names.get(cls_id, "screw_hole")
        cx = (xyxy[0] + xyxy[2]) / 2
        cy = (xyxy[1] + xyxy[3]) / 2
        print(
            f"  [{i + 1}] {label} conf={conf:.3f} "
            f"bbox=[{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}] "
            f"center=({cx:.1f}, {cy:.1f})"
        )


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"Image not found: {source}")

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    is_tflite = weights.suffix.lower() == ".tflite"
    device = "cpu" if is_tflite else args.device

    model = YOLO(str(weights))
    results = model.predict(
        source=str(source),
        conf=args.conf,
        imgsz=args.imgsz,
        device=device,
        save=args.save,
        project=args.project,
        name=args.name,
        verbose=False,
    )

    for r in results:
        print_detections(r)

    if args.save:
        out_dir = Path(args.project) / args.name
        print(f"\nSaved visualization under: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
