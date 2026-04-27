# Execution Guide — Real-Time Cursor Control Using Hand Gestures

> Follow these steps **in order** to get the full project running.

---

## Prerequisites

- **Python 3.9+** installed
- **Webcam** connected
- **NVIDIA GPU with CUDA** (recommended for ViT training, not mandatory)

---

### Download Model

- If you want to use pretrain model then
- Download the trained model from Hugging Face:
```bash
👉 https://huggingface.co/pratikm27/gesture-recognition-model/blob/main/best_cnn_model.pth

👉 https://huggingface.co/pratikm27/gesture-recognition-model/blob/main/best_vit_model.pth
```

- Important:
📂 Place it here:
```
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
```

## Step 1: Install Dependencies

Open a terminal in the project folder and run:

```bash
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import timm; print(f'timm {timm.__version__}')"
python -c "import mediapipe; print(f'MediaPipe {mediapipe.__version__}')"
python -c "import pyautogui; print(f'PyAutoGUI {pyautogui.__version__}')"
```

All four should print version numbers without errors.

---

## Step 2: Collect Gesture Data

```bash
python data/collect_data.py
```

A webcam window will open with a control panel overlay.

### Controls

| Key     | Action                        |
|---------|-------------------------------|
| 0–6     | Select gesture class          |
| SPACE   | Toggle auto-capture ON/OFF    |
| S       | Save a single frame           |
| Q       | Quit and save                 |

### What To Do

1. Press a **number key (0–6)** to select a gesture class
2. Perform that gesture in front of the webcam
3. Press **SPACE** to start auto-capturing (images save automatically)
4. Hold the gesture steady for a few seconds
5. Press **SPACE** again to stop auto-capture
6. Switch to the next class and repeat
7. **Target: 200–500 images per class** (progress bars shown on screen)

### Gesture Reference

| Key | Gesture              | What To Do With Your Hand                |
|-----|----------------------|------------------------------------------|
| 0   | Open Palm            | Spread all 5 fingers wide, palm facing camera |
| 1   | Index Point          | Only index finger extended, others curled |
| 2   | Two Fingers Up       | Index + middle finger extended (peace sign) |
| 3   | Fist                 | Close all fingers into a fist            |
| 4   | Pinch                | Touch thumb tip and index fingertip together |
| 5   | Three Fingers Up     | Index + middle + ring fingers extended   |
| 6   | Three Fingers Down   | Three fingers curled/pointing downward   |

### Gesture → Cursor Action Mapping

| # | Gesture | How To Do It | Cursor Action | Details |
|---|---------|-------------|---------------|---------|
| 0 | **Open Palm** ✋ | Spread all 5 fingers wide, palm facing camera | **Move cursor** | Cursor follows your index fingertip position on screen |
| 1 | **Index Point** ☝️ | Only index finger extended, rest curled | **Left click** | Triggers after 3 steady frames (debounce prevents accidental clicks) |
| 2 | **Two Fingers Up** ✌️ | Index + middle finger up (peace sign) | **Right click** | Same debounce as left click |
| 3 | **Fist** ✊ | Close all fingers into a fist | **Neutral / Stop** | No action — use this to pause cursor control |
| 4 | **Pinch** 🤏 | Thumb tip touching index fingertip | **Drag** | Holds left mouse button down; move hand to drag items |
| 5 | **Three Fingers Up** 🤟 | Index + middle + ring fingers extended upward | **Scroll up** | Scrolls up 3 units per trigger |
| 6 | **Three Fingers Down** | Three fingers curled/pointing downward | **Scroll down** | Scrolls down 3 units per trigger |

> **Note:** This mapping is defined in `config.py` under `ACTION_MAP` — you can customize it anytime.

### Tips for Better Data

- Record in different lighting conditions (bright, dim, mixed)
- Vary your hand distance from the camera (close, medium, far)
- Slightly rotate your hand angle between captures
- Use both left and right hands if possible
- Keep background varied (don't always sit in the same spot)

### Output

Images are saved to: `data/raw_data/{gesture_class}/`

---

## Step 3: Prepare Dataset (Split into Train/Val/Test)

```bash
python data/prepare_dataset.py
```

This takes your raw images and creates a structured dataset:

```
data/gesture_dataset/
├── train/   (70% of images)
├── val/     (15% of images)
└── test/    (15% of images)
```

Each split contains subfolders for each gesture class.

If prompted about overwriting, type `y` and press Enter.

---

## Step 4: Train CNN Model (Baseline)

```bash
python training/train.py --model cnn --epochs 50
```

### What Happens

- Loads dataset from `data/gesture_dataset/`
- Trains a custom 4-layer CNN (~524K parameters)
- Applies data augmentation (flip, rotation, color jitter)
- Uses early stopping (stops if no improvement for 5 epochs)
- Saves best model to `checkpoints/best_cnn_model.pth`
- Saves training curves to `results/training_curves/`

### Expected Time

- **GPU:** ~5–10 minutes
- **CPU:** ~15–30 minutes

### Expected Output

```
Training Loss: 0.xxxx | Train Acc: xx.xx%
Val Loss:      0.xxxx | Val Acc:   xx.xx%
★ New best model saved!
```

---

## Step 5: Train ViT Model (Proposed System)

```bash
python training/train.py --model vit --epochs 30
```

### What Happens

- Loads pretrained ViT-Base-Patch16-224 from ImageNet
- **Phase 1 (epochs 1–5):** Only trains classification head (backbone frozen)
- **Phase 2 (epochs 6–30):** Unfreezes backbone, fine-tunes everything
- Saves best model to `checkpoints/best_vit_model.pth`

### Expected Time

- **GPU:** ~15–30 minutes
- **CPU:** ~1–2 hours (GPU strongly recommended)

---

## Step 6: Evaluate Both Models

### Evaluate CNN

```bash
python training/evaluate.py --model cnn
```

### Evaluate ViT

```bash
python training/evaluate.py --model vit
```

### What You Get

For each model:

- **Test Accuracy, Precision, Recall, F1-Score** (overall and per-class)
- **Confusion Matrix** heatmap saved as PNG in `results/confusion_matrices/`
- **Inference Latency** and **FPS** benchmarks
- **Model Size** in MB
- Full **classification report** printed to terminal
- All metrics saved as JSON in `results/`

---

## Step 7: Run Real-Time Gesture Control

### Using CNN (faster)

```bash
python realtime/gesture_control.py --model cnn
```

### Using ViT (more accurate)

```bash
python realtime/gesture_control.py --model vit
```

### Debug Mode (no cursor movement, just see predictions)

```bash
python realtime/gesture_control.py --model cnn --no-cursor
```

### What You'll See

- Live webcam feed with MediaPipe hand landmarks drawn
- Bounding box around detected hand
- Current gesture prediction + confidence score
- FPS counter and latency display
- Your cursor will move/click based on gestures!

### Gesture → Cursor Action Mapping

| Gesture              | What Happens         |
|----------------------|----------------------|
| Open Palm            | Cursor moves with hand |
| Index Point          | Left click           |
| Two Fingers Up       | Right click          |
| Fist                 | Nothing (neutral)    |
| Pinch                | Drag (hold left button) |
| Three Fingers Up     | Scroll up            |
| Three Fingers Down   | Scroll down          |

### Controls

- Press **Q** to quit
- **Emergency stop:** Move mouse to any corner of the screen (PyAutoGUI fail-safe)

---

## Step 8: Compare CNN vs ViT

```bash
python analysis/compare_models.py
```

### What You Get

- Side-by-side comparison table printed to terminal
- **Bar charts** comparing accuracy, speed, and model size → saved to `results/`
- **Radar chart** for multi-metric visual comparison → saved to `results/`
- **Per-class F1 comparison** chart → saved to `results/`
- **Markdown report** → saved to `results/comparison_report.md`

---

## Step 9: Generate Research Report

```bash
python analysis/generate_report.py
```

Generates a structured research-paper-style report at `results/research_report.md` with:

- Abstract, Introduction, Methodology, Results, Discussion, Conclusion, References

---

## Step 10 (Optional): Optimize Models

### Quantize for Faster CPU Inference

```bash
python optimization/quantize_model.py --model cnn
python optimization/quantize_model.py --model vit
```

Reduces model size by ~50–75% and improves CPU speed.

### Export to ONNX

```bash
python optimization/export_onnx.py --model cnn
python optimization/export_onnx.py --model vit
```

Exports models to ONNX format for cross-platform deployment.

---

## Step 11: PPT Content

Open `analysis/ppt_content.md` for ready-made content for 20 presentation slides covering:

1. Title → Problem → Objectives → Architecture → Dataset
2. CNN Architecture → ViT Architecture → Training Setup
3. Results → Comparison → Demo → Key Findings
4. Future Work → Conclusion → References → Q&A

---

## Quick Reference — All Commands

```bash
# Install
pip install -r requirements.txt

# Collect Data
python data/collect_data.py

# Prepare Dataset
python data/prepare_dataset.py

# Train
python training/train.py --model cnn --epochs 50
python training/train.py --model vit --epochs 30

# Evaluate
python training/evaluate.py --model cnn
python training/evaluate.py --model vit

# Run Real-Time
python realtime/gesture_control.py --model cnn
python realtime/gesture_control.py --model vit

# Compare
python analysis/compare_models.py
python analysis/generate_report.py

# Optimize (optional)
python optimization/quantize_model.py --model cnn
python optimization/export_onnx.py --model cnn
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |
| Webcam not opening | Check camera ID in `config.py` → `DATA_COLLECTION["camera_id"]` |
| Low accuracy | Collect more data (500+ per class) and vary conditions |
| ViT training too slow | Use GPU, or reduce epochs: `--epochs 15` |
| Cursor going crazy | Use `--no-cursor` flag first to test predictions |
| CUDA out of memory | Reduce batch size: `--batch-size 8` |

---

## Output Files Summary

| File/Folder | Contents |
|-------------|----------|
| `checkpoints/best_cnn_model.pth` | Trained CNN weights |
| `checkpoints/best_vit_model.pth` | Trained ViT weights |
| `results/cnn_metrics.json` | CNN training metrics |
| `results/vit_metrics.json` | ViT training metrics |
| `results/cnn_eval_results.json` | CNN evaluation results |
| `results/vit_eval_results.json` | ViT evaluation results |
| `results/training_curves/` | Loss & accuracy plots |
| `results/confusion_matrices/` | Confusion matrix heatmaps |
| `results/comparison_report.md` | CNN vs ViT report |
| `results/research_report.md` | Full research-style report |
| `results/comparison_bar_charts.png` | Bar chart comparison |
| `results/comparison_radar.png` | Radar chart comparison |
| `analysis/ppt_content.md` | 20-slide PPT content |
