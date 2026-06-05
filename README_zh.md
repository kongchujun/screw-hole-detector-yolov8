# screw-hole-detector-yolov8

**ML 视觉引导打螺丝**

<p align="center">
  <a href="README.md">English</a> · <strong>中文</strong>
</p>

---

### 项目是什么？

这是一个 **机器学习（ML）打螺丝** 演示项目：用 **YOLOv8** 在电路板（PCB）图像上检测螺丝孔位置，输出每个孔位的 **边界框、中心坐标、置信度**，供视觉引导机械臂完成 **对准 → 锁付**（模拟产线视觉引导锁螺丝流程）。

| 项目定位 | 说明 |
|----------|------|
| 类型 | ML 目标检测（非传统规则视觉） |
| 模型 | YOLOv8n（轻量，适合 MacBook M2 + MPS） |
| 训练栈 | **PyTorch / Ultralytics**（不是 TensorFlow 训练） |
| 部署 | 可导出 **TFLite** 给边缘设备推理 |
| 交互 | Jupyter Notebook + Gradio Web Demo |

### ML 打螺丝产线总览

下图是 **整条打螺丝工作流**：工位相机对电路板（PLC）采图 → **边缘计算节点**用已训练模型算出孔位 → 经 **Kepware** 把孔位发给 **机械臂** → 机械臂在板上 **拧螺丝**；**云服务器**可向边缘节点 **更新模型**，以适配新型号电路板。

<p align="center">
  <img src="images/img2.png" alt="ML 打螺丝产线工作流" width="900"/>
</p>

<p align="center"><sub>图：产线工作流（<code>images/img2.png</code>）— Image → Edge calculation → Robot arm → Screw；Cloud server → Update the model</sub></p>

| 环节 | 说明 |
|------|------|
| **Image / PLC** | 工位相机拍摄电路板图像 |
| **Edge calculation** | 边缘节点运行 YOLO 模型，输出螺丝孔位置（左上/右下/中心、置信度） |
| **Kepware** | 工业通信中间件，将孔位坐标下发给机械臂控制器 |
| **Robot arm** | 按孔位走位，执行锁付 |
| **Screw** | 螺钉打入检测到的孔位 |
| **Cloud server → Update the model** | 云端训练新权重并下发边缘，换型时热替换模型，无需改机械臂主流程 |

---

### 检测结果示意

训练 / 推理完成后，可在 Gradio 或 Notebook 中得到带框的检测结果。示例截图：

<p align="center">
  <img src="images/img.png" alt="螺丝孔检测结果示例" width="720"/>
</p>

<p align="center"><sub>图：PCB 螺丝孔检测示例（<code>images/img.png</code>）</sub></p>

---

### 产线部署：相机 → 边缘节点 → 机械臂

与上图一致，数据流可概括为：**工位相机**拍摄电路板 → **边缘计算节点**推理 → **Kepware** 传孔位 → **机械臂**锁付。

```mermaid
flowchart LR
    subgraph Station["① 装配工位"]
        CAM["工业相机<br/>拍摄电路板图像"]
    end

    subgraph Edge["② 边缘计算节点"]
        RX["接收图像<br/>GigE / USB / 共享内存"]
        INF["已训练模型推理<br/>.tflite 或 .pt 运行时"]
        POS["输出孔位坐标<br/>像素 → 世界坐标 (x, y, θ)"]
        RX --> INF --> POS
    end

    subgraph Arm["③ 机械臂控制器"]
        CAL["手眼标定<br/>相机坐标系 → 机器人基座"]
        PLAN["运动规划<br/>逐孔接近"]
        FAST["锁付<br/>螺钉打入螺丝孔"]
        CAL --> PLAN --> FAST
    end

    CAM -->|"图像流 / 单帧"| RX
    POS -->|"位置 + 置信度"| CAL
```

| 步骤 | 环节 | 作用 |
|------|------|------|
| 1 | **工位相机** | 对到位后的 PCB 拍照（或锁付前复检） |
| 2 | **边缘节点** | 运行部署好的检测模型，输出每个螺丝孔中心与置信度 |
| 3 | **机械臂** | 坐标变换、走位、拧螺丝 / 电批锁付 |

边缘侧常见做法：`export.py` 导出 `.tflite` 装在工控机或 NPU 上；孔位列表再通过 PLC / 现场总线交给机械臂。

---

### 新电路板适配：替换模型

换 **新型号 PCB**（孔位布局不同）时，只需 **重新采集标注 → 训练 → 在边缘节点替换模型文件**，机械臂仍消费同一套「孔位坐标」接口，无需改整套锁付流程。

```mermaid
flowchart TB
    N["产线换型 / 新电路板"]
    D["采集 PCB 照片 + 标注螺丝孔<br/>（YOLO 格式，data/screw_holes/）"]
    T["训练新模型<br/>train.py / Notebook → best.pt"]
    E["边缘导出<br/>export.py → 新 .tflite"]
    S["边缘节点热替换<br/>推理服务加载新模型"]
    R["流程不变：相机 → 边缘推理 → 机械臂打螺丝"]

    N --> D --> T --> E --> S --> R
    S -.->|"下一款板 / 新布局"| N
```

| 变更内容 | 需要更新 | 通常不变 |
|----------|----------|----------|
| 新 PCB 布局 | 数据集、`dataset.yaml`、重训、边缘上新 `.tflite` | 相机安装、传图链路、机械臂锁付动作序列 |
| 精度不够 | 增标数据、加大 epoch 或换更大 YOLO | 上述整体架构 |

本仓库负责 **训练与导出**（图中 D → E）；相机 SDK、机械臂通信需在工厂侧对接。

---

### 研发流程（本仓库）

从数据到模型开发的完整链路如下。

```mermaid
flowchart TB
    subgraph Data["① 数据准备"]
        A1[真实 PCB 拍照 + 标注<br/>或 generate_demo_dataset.py 合成数据]
        A2[data/screw_holes/images + labels<br/>dataset.yaml]
        A1 --> A2
    end

    subgraph Train["② 模型训练 · YOLO / PyTorch / MPS"]
        B1[train.py 或 Notebook §3]
        B2[YOLOv8n 微调 screw_hole]
        B3[runs/.../weights/best.pt]
        B1 --> B2 --> B3
    end

    subgraph Eval["③ 验证与可视化"]
        C1[TensorBoard / results.png]
        C2[predict.py / Notebook §6]
        C3[demo.py Gradio / Notebook §7]
        B3 --> C1
        B3 --> C2
        B3 --> C3
    end

    subgraph Deploy["④ 边缘部署（可选）"]
        D1[export.py / Notebook §5]
        D2[TFLite .tflite INT8 或 float32]
        D3[工控机 / 嵌入式推理]
        B3 --> D1 --> D2 --> D3
    end

    subgraph Robot["⑤ 机械臂打螺丝（业务层）"]
        E1[相机采图]
        E2[ML 检测孔位中心]
        E3[手眼标定 + 运动规划]
        E4[锁付工具拧紧]
        D2 --> E2
        E1 --> E2 --> E3 --> E4
    end

    Data --> Train --> Eval
```

---

### 脚本 / 文件分工

```mermaid
flowchart LR
    NB[ScrewHole_YOLOv8.ipynb<br/>一站式推荐]
    GEN[generate_demo_dataset.py<br/>合成演示数据]
    TR[train.py<br/>YOLO 训练 MPS]
    EX[export.py<br/>→ TFLite]
    PR[predict.py<br/>单张 CLI 推理]
    DM[demo.py<br/>Gradio UI]

    GEN --> TR
    NB --> TR
    TR --> EX
    TR --> PR
    TR --> DM
```

| 文件 | 作用 |
|------|------|
| `ScrewHole_YOLOv8.ipynb` | **推荐**：按章节完成环境、数据、训练、导出、预测、Demo |
| `generate_demo_dataset.py` | 无真实数据时，本地生成合成 PCB 螺丝孔数据集 |
| `dataset.yaml` | YOLO 数据集路径与类别 `screw_hole` |
| `train.py` | YOLOv8n 训练，`device=mps`，输出 `best.pt` |
| `export.py` | 将 `best.pt` 转为 `.tflite`（INT8 或 float32） |
| `predict.py` | 命令行单图检测 |
| `demo.py` | Gradio：上传图 → 框 + 孔位坐标表 |
| `images/img2.png` | **产线打螺丝总览图**（相机→边缘→机械臂→云更新模型） |
| `images/img.png` | Gradio 检测结果示例截图 |

---

### 项目结构

```
screw-hole-detector-yolov8/
├── README.md / README_zh.md
├── ScrewHole_YOLOv8.ipynb
├── images/img2.png   # 产线工作流总览
├── images/img.png    # 检测 Demo 截图
├── dataset.yaml
├── train.py · export.py · predict.py · demo.py
├── generate_demo_dataset.py
├── data/screw_holes/
└── runs/                 # git 忽略
```

---

### 快速开始

```bash
cd ~/Projects/screw-hole-detector-yolov8
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 无标注数据时：生成演示集
python generate_demo_dataset.py --train 80 --val 20

# 训练（YOLO + MPS）
python train.py --epochs 50 --batch 8

# 快速导出 TFLite（float32，较快）
python export.py --no-int8 --device cpu --imgsz 320

# Gradio Demo
python demo.py
```

**Notebook：** 打开 `ScrewHole_YOLOv8.ipynb`，从上到下运行；导出章节默认 `EXPORT_INT8 = False`、`device=cpu`，避免 Mac 上 TensorFlow 长时间无输出。

---

### 数据集目录（YOLO 格式）

```
data/screw_holes/
├── images/train/   images/val/
└── labels/train/   labels/val/    # 与图片同名的 .txt
```

---

### 训练 vs TensorFlow

| 阶段 | 技术 |
|------|------|
| 训练 / `.pt` 推理 | **YOLOv8 + PyTorch**（M2 用 MPS） |
| `.tflite` 导出 | Ultralytics 调用 **TensorFlow** 做格式转换与量化 |

---

### 常见问题

- **`best.pt` 找不到**：权重可能在 `runs/detect/runs/detect/screw_hole/...`；Notebook 已用 `find_best_pt()` 自动查找，或见 `runs/detect/screw_hole/weights/best.pt`。
- **导出 TFLite 日志停住**：多为 CPU 转换中，非报错；可先 `export.py --no-int8`。
- **合成数据指标很高**：仅说明流程通畅；产线请换真实 PCB 标注重训。

---

### 许可证

Demo / 学习用途。YOLOv8 遵循 [Ultralytics AGPL-3.0](https://github.com/ultralytics/ultralytics)。

---

<p align="center"><a href="README.md">English documentation →</a></p>
