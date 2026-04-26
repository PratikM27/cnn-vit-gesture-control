# PPT Content — Real-Time Cursor Control Using Hand Gestures with Vision Transformer

> Use this as slide-by-slide content for your presentation.
> Each section = one slide.

---

## Slide 1: Title Slide

**Real-Time Cursor Control Using Hand Gestures with Vision Transformer**

*A Comparative Study of CNN and Vision Transformer for Gesture-Based HCI*

- Student Name: [Your Name]
- Guide: [Guide Name]
- Department: [Department]
- Year: 2025-26

---

## Slide 2: Problem Statement

### The Problem
- Traditional cursor control depends on physical devices (mouse, touchpad)
- These are inaccessible for users with motor disabilities
- No natural hand-based interaction support

### The Need
- Touchless interfaces for accessibility & hygiene
- Real-time, low-latency gesture recognition
- Comparing modern AI architectures for this task

---

## Slide 3: Objectives

1. Build a real-time hand gesture recognition system using webcam
2. Implement two models:
   - **CNN** — Convolutional Neural Network (baseline)
   - **ViT** — Vision Transformer (proposed)
3. Compare models on accuracy, speed, and latency
4. Map gestures to cursor actions using PyAutoGUI
5. Evaluate with Accuracy, Precision, Recall, F1-Score

---

## Slide 4: System Architecture

```
Webcam → OpenCV Frame Capture
    → MediaPipe Hand Detection
    → ROI Extraction & Preprocessing
    → Model Inference (CNN or ViT)
    → Gesture Classification
    → Gesture-to-Action Mapping
    → Cursor Control (PyAutoGUI)
```

**Key Components:** OpenCV, MediaPipe, PyTorch, PyAutoGUI

---

## Slide 5: Gesture Classes

| # | Gesture | Cursor Action |
|---|---------|--------------|
| 0 | Open Palm | Move cursor |
| 1 | Index Point | Left click |
| 2 | Two Fingers Up | Right click |
| 3 | Fist | Neutral / Stop |
| 4 | Pinch | Drag |
| 5 | Three Fingers Up | Scroll up |
| 6 | Three Fingers Down | Scroll down |

**Total: 7 gesture classes**

---

## Slide 6: Dataset

- **Collection:** Custom webcam-based data collection using MediaPipe
- **Images:** Hand ROI patches (cropped from webcam frames)
- **Split:** 70% Train / 15% Validation / 15% Test
- **Augmentation:** Random flip, rotation (±15°), color jitter, Gaussian blur

---

## Slide 7: CNN Architecture (Baseline)

```
Input (3×128×128)
  → Conv2D(32) → BN → ReLU → MaxPool
  → Conv2D(64) → BN → ReLU → MaxPool
  → Conv2D(128) → BN → ReLU → MaxPool
  → Conv2D(256) → BN → ReLU → MaxPool
  → AdaptiveAvgPool → FC(512) → Dropout → FC(7)
```

- **Parameters:** ~2-3 Million
- **Optimizer:** Adam (LR=0.001)
- **Scheduler:** StepLR

---

## Slide 8: ViT Architecture (Proposed)

```
Input (3×224×224)
  → Patch Embedding (16×16 → 196 tokens)
  → [CLS] Token + Positional Encoding
  → 12× Transformer Encoder (Multi-Head Self-Attention + MLP)
  → [CLS] → Classification Head → 7 classes
```

- **Model:** vit_base_patch16_224 (pretrained on ImageNet)
- **Parameters:** ~86 Million
- **Training:** Phase 1 (head only) + Phase 2 (full fine-tune)

---

## Slide 9: CNN vs ViT — Key Differences

| Feature | CNN | ViT |
|---------|-----|-----|
| Core Operation | Convolution | Self-Attention |
| Receptive Field | Local | Global |
| Inductive Bias | Yes (locality) | None |
| Data Requirement | Works on small data | Needs pretraining |
| Inference Speed | Fast | Slower |
| Model Size | Small (~3M) | Large (~86M) |

---

## Slide 10: Training Setup

| Setting | CNN | ViT |
|---------|-----|-----|
| Input Size | 128×128 | 224×224 |
| Batch Size | 32 | 16 |
| Learning Rate | 1e-3 | 1e-4 |
| Epochs | 50 | 30 |
| Optimizer | Adam | AdamW |
| Scheduler | StepLR | CosineAnnealingLR |
| Loss | CrossEntropy (label smoothing=0.1) | Same |
| Early Stopping | Patience=5 | Patience=5 |

---

## Slide 11: Results — Accuracy & Metrics

| Metric | CNN | ViT | Winner |
|--------|-----|-----|--------|
| Test Accuracy | _% | _% | |
| Precision (macro) | _% | _% | |
| Recall (macro) | _% | _% | |
| F1-Score (macro) | _% | _% | |

*(Insert actual numbers after training)*

---

## Slide 12: Results — Speed & Latency

| Metric | CNN | ViT | Winner |
|--------|-----|-----|--------|
| FPS | _ | _ | |
| Latency (ms) | _ | _ | |
| Model Size (MB) | _ | _ | |
| Parameters (M) | _ | _ | |

**Threshold for real-time: ≥15 FPS**

---

## Slide 13: Confusion Matrices

*(Insert confusion matrix images for both CNN and ViT)*

- Show which gestures are most confused
- Analyze reasons for misclassification

---

## Slide 14: Training Curves

*(Insert loss and accuracy curve plots for both models)*

- Compare convergence speed
- Check for overfitting (train vs val gap)

---

## Slide 15: Real-Time Demo

- Webcam feed with MediaPipe hand landmarks
- Live gesture classification label
- Real-time cursor movement
- FPS counter displayed on screen

*(Include screenshot or video clip)*

---

## Slide 16: Key Findings

1. **Accuracy:** ViT typically achieves higher accuracy (+2-5%)
2. **Speed:** CNN runs 2-3x faster than ViT
3. **Real-time:** CNN easily meets ≥15 FPS threshold
4. **ViT advantage:** Better at distinguishing similar gestures (global context)
5. **Trade-off:** CNN for deployment, ViT for accuracy-critical applications

---

## Slide 17: Future Work

1. Hybrid CNN+ViT architecture
2. Landmark-based classification (no images needed)
3. Dynamic gesture recognition (temporal sequences)
4. Model optimization (quantization, ONNX, TensorRT)
5. Edge deployment (Raspberry Pi, mobile)
6. Attention visualization for interpretability

---

## Slide 18: Conclusion

- Successfully built a real-time gesture-controlled cursor system
- CNN serves as a fast, reliable baseline
- ViT demonstrates superior classification via global attention
- The system is modular, extensible, and runs on standard hardware
- Contributes to accessible technology and HCI research

---

## Slide 19: References

1. Dosovitskiy et al., "An Image is Worth 16x16 Words," ICLR 2021
2. Zhang et al., "MediaPipe Hands: On-device Real-time Hand Tracking," 2020
3. He et al., "Deep Residual Learning," CVPR 2016
4. Howard et al., "MobileNets," 2017
5. Vaswani et al., "Attention Is All You Need," NeurIPS 2017

---

## Slide 20: Thank You / Q&A

**Thank you!**

*Questions?*
