# future_work.md

# Future Work & Improvements

---

## Overview

This section outlines directions for extending and improving the project beyond its current scope. These are relevant for the viva discussion and final report conclusion.

---

## 1. Hybrid Model (CNN + ViT)

### What
Combine CNN's local feature extraction with ViT's global attention in a single architecture.

### Why
- CNN extracts fine-grained local features (finger textures, edges).
- ViT captures global spatial relationships (relative finger positions).
- A hybrid could outperform either model alone.

### How (Ideas)

**Option A — CNN Feature Extractor + Transformer Encoder:**
```
Input Image
    ↓
CNN Backbone (ResNet / EfficientNet) → Feature Map
    ↓
Flatten feature map → Sequence of tokens
    ↓
Transformer Encoder (self-attention)
    ↓
Classification Head
```

**Option B — Early CNN + Late ViT (parallel branches):**
- CNN processes low-level features; ViT processes patch embeddings.
- Concatenate outputs → joint classifier.

**References:** CvT (Convolutional Vision Transformer), DeiT, ConvNeXt.

---

## 2. Expanded Gesture Set

### What
Add more gesture classes beyond the current 7.

### Why
- More gestures = richer interaction vocabulary.
- Enable application-specific mappings (e.g., zoom in/out, switch tabs).

### Possible New Gestures
| Gesture | Action |
|---|---|
| OK Sign | Confirm / Enter |
| Thumbs Up | Volume up |
| Thumbs Down | Volume down |
| Victory Sign | Open app |
| Swipe Left/Right | Navigate pages |

---

## 3. Dynamic / Temporal Gesture Recognition

### What
Recognize gestures that involve motion over time (e.g., swipe, wave, circle draw), not just static hand poses.

### Why
- Current system only classifies static snapshots — misses intentional motion-based gestures.
- Dynamic gestures are more natural and expressive.

### How
- Use a sequence of frames (e.g., 10–16 frames) as input.
- Apply an LSTM or Temporal Transformer on top of per-frame CNN features.
- Dataset: Use sequences, not single images.

---

## 4. Landmark-Based Classification (No Image)

### What
Replace image-based CNN/ViT with a classifier trained directly on MediaPipe's 21 (x, y, z) landmarks.

### Why
- 63 numbers (21 × 3) instead of a full image → much faster.
- Removes background/lighting sensitivity entirely.
- Viable with a small MLP or SVM.

### How
- Extract normalized landmark coordinates from MediaPipe.
- Train an MLP or Random Forest on landmark vectors.
- Combine with the current image-based model via ensemble for robustness.

---

## 5. Model Optimization for Edge Deployment

### What
Reduce model size and inference time for deployment on low-power devices (Raspberry Pi, mobile).

### Why
- Current ViT model is too large for real-time use on edge hardware.
- Many assistive technology use-cases require portable devices.

### Techniques
| Technique | Description |
|---|---|
| Quantization | Reduce weight precision from FP32 → INT8 |
| Pruning | Remove low-importance weights or channels |
| Knowledge Distillation | Train a small "student" CNN to mimic the ViT "teacher" |
| ONNX Export | Convert model to ONNX for optimized cross-platform inference |
| TensorRT | NVIDIA GPU-optimized inference engine |

---

## 6. User Calibration

### What
Allow the system to calibrate to each user's specific hand size, skin tone, and gesture style.

### Why
- Improves personalization and accuracy.
- Reduces errors from hand size variation across users.

### How
- Brief 30-second calibration session at startup.
- Fine-tune the final classification layer on user-specific samples.

---

## 7. Multi-Hand Support

### What
Support two-hand gestures for richer controls (e.g., pinch-zoom with both hands).

### Why
- Many intuitive interactions (zooming, rotating) naturally involve two hands.

### How
- Enable `max_num_hands=2` in MediaPipe.
- Design new gesture classes for two-hand combinations.
- Requires more complex gesture-to-action mapping logic.

---

## 8. Attention Visualization

### What
Visualize which parts of the input the ViT model attends to when making predictions.

### Why
- Improves interpretability and trust in the model.
- Useful for academic presentation and debugging.

### How
- Extract attention weights from ViT transformer layers.
- Overlay attention heatmaps on the input image.
- Use libraries like `vit-explain` or implement Attention Rollout.

---

## Summary of Priorities

| Improvement | Impact | Effort | Priority |
|---|---|---|---|
| Hybrid CNN+ViT | High | High | Medium |
| Landmark-based classifier | High | Low | High |
| Model quantization | High | Medium | High |
| Dynamic gesture recognition | High | High | Low (future research) |
| User calibration | Medium | Medium | Medium |
| Expanded gesture set | Medium | Low | High |
| Multi-hand support | Medium | Medium | Low |
| Attention visualization | Low | Low | High (for viva) |
