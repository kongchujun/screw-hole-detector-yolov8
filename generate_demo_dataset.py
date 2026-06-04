#!/usr/bin/env python3
"""Generate synthetic PCB + screw-hole images for quick demo training.

No download required. For production, replace with real labeled PCB photos.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data" / "screw_holes"
CLASS_ID = 0


def yolo_line(cx: float, cy: float, w: float, h: float) -> str:
    return f"{CLASS_ID} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"


def draw_pcb_with_holes(
    size: int = 640,
    n_holes: int | None = None,
) -> tuple[Image.Image, list[tuple[float, float, float, float]]]:
    """Return RGB image and list of normalized YOLO boxes (cx,cy,w,h)."""
    if n_holes is None:
        n_holes = random.randint(4, 12)

    # PCB substrate (green) + subtle noise
    base = np.random.randint(30, 50, (size, size, 3), dtype=np.uint8)
    base[:, :, 1] = np.clip(base[:, :, 1] + random.randint(60, 100), 0, 255)
    base[:, :, 0] = np.clip(base[:, :, 0] - 10, 0, 255)
    base[:, :, 2] = np.clip(base[:, :, 2] - 10, 0, 255)
    img = Image.fromarray(base)
    draw = ImageDraw.Draw(img)

    # Copper traces (decorative)
    for _ in range(random.randint(3, 8)):
        x1, y1 = random.randint(0, size), random.randint(0, size)
        x2, y2 = random.randint(0, size), random.randint(0, size)
        draw.line((x1, y1, x2, y2), fill=(180, 140, 60), width=random.randint(2, 5))

    boxes: list[tuple[float, float, float, float]] = []
    margin = int(size * 0.12)
    min_dist = size * 0.09

    for _ in range(n_holes * 3):
        if len(boxes) >= n_holes:
            break
        r = random.randint(int(size * 0.018), int(size * 0.035))
        cx = random.randint(margin + r, size - margin - r)
        cy = random.randint(margin + r, size - margin - r)
        ok = True
        for bc_x, bc_y, bw, bh in boxes:
            px, py = bc_x * size, bc_y * size
            if (cx - px) ** 2 + (cy - py) ** 2 < (min_dist * size) ** 2:
                ok = False
                break
        if not ok:
            continue

        # Hole: dark ring + lighter center (through-hole look)
        outer = (25, 25, 30)
        inner = (70, 72, 78)
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=outer, outline=(10, 10, 12))
        ri = max(2, int(r * 0.45))
        draw.ellipse((cx - ri, cy - ri, cx + ri, cy + ri), fill=inner)

        w = (2 * r) / size
        h = (2 * r) / size
        boxes.append((cx / size, cy / size, w, h))

    while len(boxes) < max(4, n_holes):
        r = int(size * 0.025)
        cx = random.randint(margin, size - margin)
        cy = random.randint(margin, size - margin)
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(25, 25, 30))
        boxes.append((cx / size, cy / size, (2 * r) / size, (2 * r) / size))

    return img, boxes


def write_split(split: str, count: int, seed: int) -> None:
    img_dir = DATA_ROOT / "images" / split
    lbl_dir = DATA_ROOT / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    for i in range(count):
        rng.seed(seed + i)
        random.seed(seed + i)
        np.random.seed(seed + i)
        img, boxes = draw_pcb_with_holes(size=640, n_holes=rng.randint(5, 10))
        stem = f"{split}_{i:04d}"
        img.save(img_dir / f"{stem}.jpg", quality=92)
        with open(lbl_dir / f"{stem}.txt", "w") as f:
            for b in boxes:
                f.write(yolo_line(*b))


def main() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic screw-hole demo dataset")
    p.add_argument("--train", type=int, default=80, help="Number of train images")
    p.add_argument("--val", type=int, default=20, help="Number of val images")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    write_split("train", args.train, args.seed)
    write_split("val", args.val, args.seed + 10_000)

    print(f"Done. Dataset: {DATA_ROOT.resolve()}")
    print(f"  train: {args.train} images")
    print(f"  val:   {args.val} images")
    print("Re-run Notebook「检查数据集」单元格，然后即可训练。")
    print("\n正式项目请替换为真实 PCB 标注数据（见 README）。")


if __name__ == "__main__":
    main()
