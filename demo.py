#!/usr/bin/env python3
"""Gradio Web Demo — PCB screw-hole detection (.pt / .tflite)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image
from ultralytics import YOLO

DEFAULT_PT = "runs/detect/screw_hole/weights/best.pt"
DEFAULT_TFLITE = "runs/detect/screw_hole/weights/best_int8.tflite"
DEFAULT_DEVICE = "mps"

_model_cache: dict[str, YOLO] = {}


def load_model(weights_path: str, backend: str) -> YOLO:
    path = Path(weights_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    key = f"{backend}:{path}"
    if key not in _model_cache:
        _model_cache[key] = YOLO(str(path))
    return _model_cache[key]


def run_inference(
    image: Image.Image | np.ndarray | None,
    backend: str,
    weights_path: str,
    conf: float,
    imgsz: int,
) -> tuple[Image.Image | None, str, str]:
    if image is None:
        return None, "请上传一张电路板图片。", ""

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    try:
        model = load_model(weights_path.strip(), backend)
    except FileNotFoundError as e:
        return None, str(e), ""

    is_tflite = backend == "TFLite" or str(weights_path).lower().endswith(".tflite")
    device = "cpu" if is_tflite else DEFAULT_DEVICE

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.convert("RGB").save(tmp.name)
        tmp_path = tmp.name

    try:
        results = model.predict(
            source=tmp_path,
            conf=conf,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    result = results[0]
    plotted = result.plot()
    out_img = Image.fromarray(plotted[:, :, ::-1])  # BGR -> RGB

    boxes = result.boxes
    count = 0 if boxes is None else len(boxes)
    names = result.names or {0: "screw_hole"}

    lines: list[str] = []
    table_rows: list[list] = []
    if boxes is not None and count > 0:
        for i, box in enumerate(boxes):
            xyxy = box.xyxy[0].tolist()
            conf_v = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = names.get(cls_id, "screw_hole")
            cx = (xyxy[0] + xyxy[2]) / 2
            cy = (xyxy[1] + xyxy[3]) / 2
            lines.append(
                f"#{i + 1} {label} | conf={conf_v:.3f} | "
                f"bbox=({xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}) | "
                f"center=({cx:.1f}, {cy:.1f})"
            )
            table_rows.append(
                [
                    i + 1,
                    label,
                    f"{conf_v:.3f}",
                    f"({xyxy[0]:.0f}, {xyxy[1]:.0f})",
                    f"({xyxy[2]:.0f}, {xyxy[3]:.0f})",
                    f"({cx:.0f}, {cy:.0f})",
                ]
            )

    summary = f"检测到 **{count}** 个螺丝孔。"
    detail = "\n".join(lines) if lines else "未检测到螺丝孔，可尝试降低置信度阈值或更换模型。"
    table_md = ""
    if table_rows:
        header = "| # | 类别 | 置信度 | 左上 | 右下 | 中心 |\n|---:|---|---:|---|---|---|\n"
        body = "\n".join(
            f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} |" for r in table_rows
        )
        table_md = header + body

    return out_img, summary + "\n\n" + detail, table_md


def on_backend_change(backend: str) -> str:
    return DEFAULT_TFLITE if backend == "TFLite" else DEFAULT_PT


def build_ui() -> gr.Blocks:
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
    )

    css = """
    .gradio-container { max-width: 1100px !important; margin: auto; }
    #title { text-align: center; margin-bottom: 0.25rem; }
    #subtitle { text-align: center; color: #64748b; margin-bottom: 1rem; }
    """

    with gr.Blocks(title="螺丝孔检测 Demo", theme=theme, css=css) as demo:
        gr.Markdown(
            "# 🔩 PCB 螺丝孔检测 · YOLOv8\n"
            "<p id='subtitle'>视觉引导机械臂锁付场景 Demo · 支持 PyTorch (.pt) 与 TFLite (.tflite)</p>",
            elem_id="title",
        )

        with gr.Row():
            with gr.Column(scale=1):
                backend = gr.Radio(
                    ["PyTorch (.pt)", "TFLite"],
                    value="PyTorch (.pt)",
                    label="推理后端",
                )
                weights = gr.Textbox(
                    label="模型路径",
                    value=DEFAULT_PT,
                    placeholder="runs/detect/screw_hole/weights/best.pt",
                )
                conf = gr.Slider(0.05, 0.95, value=0.35, step=0.05, label="置信度阈值")
                imgsz = gr.Slider(320, 1280, value=640, step=32, label="推理尺寸")
                image_in = gr.Image(type="pil", label="上传电路板图片")
                run_btn = gr.Button("开始检测", variant="primary")

            with gr.Column(scale=1):
                image_out = gr.Image(type="pil", label="检测结果")
                summary = gr.Markdown(label="摘要")
                table = gr.Markdown(label="坐标表")

        backend.change(on_backend_change, inputs=backend, outputs=weights)
        run_btn.click(
            run_inference,
            inputs=[image_in, backend, weights, conf, imgsz],
            outputs=[image_out, summary, table],
        )
        image_in.change(
            run_inference,
            inputs=[image_in, backend, weights, conf, imgsz],
            outputs=[image_out, summary, table],
        )

        gr.Markdown(
            "---\n"
            "**提示：** 首次使用请先 `python train.py` 训练模型，或 `python export.py` 导出 TFLite。"
            " TFLite 推理在 CPU 上运行；.pt 在 M2 上使用 MPS 加速。"
        )

    return demo


def main() -> None:
    app = build_ui()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()
