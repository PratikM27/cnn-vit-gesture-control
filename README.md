# Real-Time Cursor Control Using Hand Gestures with Vision Transformer

> CNN vs Vision Transformer — A Comparative Study for Gesture-Based HCI

---

## Overview

This project implements a **real-time gesture-controlled cursor system** using webcam input. Two deep learning models are trained and compared:

| Model | Architecture | Input Size | Parameters |
|-------|-------------|-----------|------------|
| **CNN** (Baseline) | Custom 4-layer CNN | 128×128 | ~3M |
| **ViT** (Proposed) | ViT-Base-Patch16-224 | 224×224 | ~86M |

### Gesture Classes (7)

| Gesture | Cursor Action |
|---------|--------------|
| Open Palm | Move cursor |
| Index Point | Left click |
| Two Fingers Up | Right click |
| Fist | Neutral / Stop |
| Pinch | Drag |
| Three Fingers Up | Scroll up |
| Three Fingers Down | Scroll down |

---

## Setup Instructions

### 1. Prerequisites

- Python 3.9+
- Webcam
- (Optional) NVIDIA GPU with CUDA for faster training

### 2. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install packages
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import timm; print(f'timm {timm.__version__}')"
python -c "import mediapipe; print(f'MediaPipe {mediapipe.__version__}')"
```

### 4. Download Model

- If you want to use pretrain model then
- Download the trained model from Hugging Face:
```bash
👉 https://huggingface.co/pratikm27/gesture-recognition-model/blob/main/best_cnn_model.pth
```

- Important:
📂 Place it here:

project/
│
├── checkpoints/
│   └── best_cnn_model.pth
│   └── best_vit_model.pth
│
├── src/
├── main.py
├── .gitignore
└── README.md

---

## Usage Guide

### Step 1: Collect Training Data

```bash
python data/collect_data.py
```

- Press keys **0–6** to select a gesture class
- Press **SPACE** to start/stop auto-capture
- Press **S** to save a single frame
- Press **Q** to quit
- Target: **200–500 images per class**

### Step 2: Prepare Dataset (Split into Train/Val/Test)

```bash
python data/prepare_dataset.py
```

This creates the `data/gesture_dataset/{train,val,test}/` directory structure.

### Step 3: Train Models

```bash
# Train CNN
python training/train.py --model cnn --epochs 50

# Train ViT
python training/train.py --model vit --epochs 30
```

Training curves and best checkpoints are saved automatically.

### Step 4: Evaluate Models

```bash
# Evaluate CNN
python training/evaluate.py --model cnn

# Evaluate ViT
python training/evaluate.py --model vit
```

Generates confusion matrices, classification reports, and latency benchmarks.

### Step 5: Run Real-Time Gesture Control

```bash
# Using CNN model
python realtime/gesture_control.py --model cnn

# Using ViT model
python realtime/gesture_control.py --model vit
```

Press **Q** to quit the real-time system.

### Step 6: Compare Models

```bash
python analysis/compare_models.py
```

Generates side-by-side comparison tables, bar charts, and radar plots.

### Step 7: (Optional) Optimize Models

```bash
# Quantize for faster CPU inference
python optimization/quantize_model.py --model cnn
python optimization/quantize_model.py --model vit

# Export to ONNX
python optimization/export_onnx.py --model cnn
python optimization/export_onnx.py --model vit
```

---

## Project Structure

```
├── config.py                  # Central configuration
├── requirements.txt           # Dependencies
├── README.md                  # This file
├── data/
│   ├── collect_data.py        # Webcam data collection
│   ├── prepare_dataset.py     # Train/val/test split
│   └── gesture_dataset/       # Generated dataset
├── models/
│   ├── cnn_model.py           # CNN architecture
│   └── vit_model.py           # ViT architecture
├── training/
│   ├── train.py               # Training script
│   ├── evaluate.py            # Evaluation & metrics
│   └── utils.py               # Helpers
├── realtime/
│   ├── gesture_control.py     # Main real-time loop
│   ├── hand_detector.py       # MediaPipe wrapper
│   ├── cursor_controller.py   # PyAutoGUI wrapper
│   └── gesture_smoother.py    # Smoothing & debounce
├── optimization/
│   ├── quantize_model.py      # Model quantization
│   └── export_onnx.py         # ONNX export
├── analysis/
│   ├── compare_models.py      # Model comparison
│   ├── generate_report.py     # Report generation
│   └── ppt_content.md         # PPT content
├── checkpoints/               # Saved model weights
└── results/                   # Output plots & metrics
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | PyTorch, timm |
| Hand Detection | MediaPipe |
| Video Capture | OpenCV |
| Cursor Control | PyAutoGUI |
| Data Science | NumPy, Matplotlib, scikit-learn |

---

## License

This project is developed for academic purposes as a Final Year Project.
