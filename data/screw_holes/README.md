# Dataset directory

Training images are **not** in git (see root `.gitignore`). Regenerate locally:

```bash
python generate_demo_dataset.py --train 80 --val 20
```

YOLO layout:

```
images/train/  images/val/
labels/train/  labels/val/
```

Labels may be tracked in git; images are ignored.
