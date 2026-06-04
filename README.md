# screw-hole-detector-yolov8

**ML Vision-Guided Screw Fastening**

<p align="center">
  <strong>English</strong> · <a href="README_zh.md">中文</a>
</p>

---

### What is this project?

An **ML vision-guided screw fastening** demo: **YOLOv8** detects screw holes on PCB images and returns **bounding boxes, hole centers, and confidence scores** for a robot arm pipeline (**align → fasten**), similar to manufacturing visual guidance for automated screw driving.

| Aspect | Detail |
|--------|--------|
| Type | ML object detection (not classical rule-based vision) |
| Model | YOLOv8n (lightweight, MacBook M2 + MPS) |
| Training | **PyTorch / Ultralytics** (not TensorFlow training) |
| Deployment | Optional **TFLite** export for edge inference |
| UX | Jupyter Notebook + Gradio Web Demo |

### Result preview

After training / inference, you get annotated detections in Gradio or the Notebook. Example:

<p align="center">
  <img src="images/img.png" alt="Screw-hole detection result example" width="720"/>
</p>

<p align="center"><sub>Figure: PCB screw-hole detection example (<code>images/img.png</code>)</sub></p>

---

### End-to-end workflow

```mermaid
flowchart TB
    subgraph Data["① Data"]
        A1[Real PCB photos + labels<br/>or synthetic demo via generate_demo_dataset.py]
        A2[data/screw_holes + dataset.yaml]
        A1 --> A2
    end

    subgraph Train["② Train · YOLO / PyTorch / MPS"]
        B1[train.py or Notebook §3]
        B2[Fine-tune YOLOv8n on screw_hole]
        B3[best.pt under runs/]
        B1 --> B2 --> B3
    end

    subgraph Eval["③ Validate & demo"]
        C1[TensorBoard / results.png]
        C2[predict.py / Notebook §6]
        C3[demo.py Gradio / Notebook §7]
        B3 --> C1
        B3 --> C2
        B3 --> C3
    end

    subgraph Deploy["④ Edge deploy optional"]
        D1[export.py / Notebook §5]
        D2[TFLite .tflite INT8 or float32]
        D3[IPC / embedded runtime]
        B3 --> D1 --> D2 --> D3
    end

    subgraph Robot["⑤ Screw fastening business logic"]
        E1[Camera capture]
        E2[ML hole localization]
        E3[Hand-eye calibration + motion]
        E4[Tool fastening]
        D2 --> E2
        E1 --> E2 --> E3 --> E4
    end

    Data --> Train --> Eval
```

---

### Scripts & artifacts

```mermaid
flowchart LR
    NB[ScrewHole_YOLOv8.ipynb<br/>all-in-one]
    GEN[generate_demo_dataset.py]
    TR[train.py · YOLO MPS]
    EX[export.py → TFLite]
    PR[predict.py]
    DM[demo.py Gradio]

    GEN --> TR
    NB --> TR
    TR --> EX
    TR --> PR
    TR --> DM
```

| File | Role |
|------|------|
| `ScrewHole_YOLOv8.ipynb` | **Recommended** step-by-step pipeline |
| `generate_demo_dataset.py` | Synthetic PCB + holes for quick trials |
| `dataset.yaml` | Dataset paths & class `screw_hole` |
| `train.py` | Train YOLOv8n with `device=mps` → `best.pt` |
| `export.py` | Convert `best.pt` to `.tflite` |
| `predict.py` | Single-image CLI inference |
| `demo.py` | Gradio UI with boxes & coordinates |
| `images/img.png` | Example final result screenshot |

---

### Project layout

```
screw-hole-detector-yolov8/
├── README.md / README_zh.md
├── ScrewHole_YOLOv8.ipynb
├── images/img.png
├── dataset.yaml
├── train.py · export.py · predict.py · demo.py
├── generate_demo_dataset.py
├── data/screw_holes/
└── runs/                 # gitignored
```

---

### Quick start

```bash
cd ~/Projects/screw-hole-detector-yolov8
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python generate_demo_dataset.py --train 80 --val 20
python train.py --epochs 50 --batch 8
python export.py --no-int8 --device cpu --imgsz 320
python demo.py
```

Open `ScrewHole_YOLOv8.ipynb` and run cells top to bottom. For export on Mac, prefer `EXPORT_INT8 = False` and `device=cpu` in the Notebook to avoid long silent TensorFlow conversion.

---

### Dataset layout (YOLO)

```
data/screw_holes/
├── images/train/   images/val/
└── labels/train/   labels/val/
```

---

### Training vs TensorFlow

| Stage | Stack |
|-------|--------|
| Train / `.pt` inference | **YOLOv8 + PyTorch** (MPS on Apple Silicon) |
| `.tflite` export | **TensorFlow** via Ultralytics converter & quantization |

---

### FAQ

- **Missing `best.pt`**: Check nested path `runs/detect/runs/detect/...` or use `find_best_pt()` in the Notebook.
- **TFLite export seems stuck**: Usually CPU conversion; try `--no-int8` first.
- **Very high mAP on synthetic data**: Expected for demo data; retrain on real PCB labels for production.

---

### License

Demo / educational use. YOLOv8 follows [Ultralytics AGPL-3.0](https://github.com/ultralytics/ultralytics).

---

<p align="center"><a href="README_zh.md">中文文档 →</a></p>
