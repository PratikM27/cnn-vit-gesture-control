# models.md

# Models — CNN vs Vision Transformer (ViT)

---

## 1. Convolutional Neural Network (CNN)

### What
A deep learning architecture that uses convolutional filters to extract spatial features from images in a hierarchical manner.

### Why
- Proven baseline for image classification tasks.
- Lightweight and fast — suitable for real-time inference on CPU.
- Efficient at detecting local spatial patterns (edges, textures, shapes).

### How — Architecture

```
Input Image (3 × 64 × 64)
    ↓
Conv2D (32 filters, 3×3) → BatchNorm → ReLU → MaxPool
    ↓
Conv2D (64 filters, 3×3) → BatchNorm → ReLU → MaxPool
    ↓
Conv2D (128 filters, 3×3) → BatchNorm → ReLU → MaxPool
    ↓
Flatten
    ↓
Fully Connected (512) → ReLU → Dropout (0.5)
    ↓
Fully Connected (7) → Softmax
    ↓
Output: Gesture Class (0–6)
```

**Key Properties:**
| Property | Value |
|---|---|
| Input Size | 3 × 64 × 64 |
| Parameters | ~2–5 million (custom) |
| Inductive Bias | Locality, translation equivariance |
| Framework | PyTorch |

**Alternative:** Use a pretrained `MobileNetV2` or `EfficientNet-B0` with fine-tuning for better accuracy with fewer custom parameters.

---

## 2. Vision Transformer (ViT)

### What
A transformer-based architecture that divides an image into fixed-size patches, embeds them as tokens, and processes them using multi-head self-attention — the same mechanism used in NLP transformers.

### Why
- Captures **global context** across the entire image (unlike CNN's local receptive field).
- State-of-the-art performance on large-scale image tasks.
- Allows studying whether global relationships between hand regions improve gesture recognition.

### How — Architecture

```
Input Image (3 × 224 × 224)
    ↓
Patch Embedding: Split into 16×16 patches → 196 tokens
    ↓
Add [CLS] Token + Positional Encoding
    ↓
Transformer Encoder × L layers:
    Multi-Head Self-Attention → LayerNorm → MLP → LayerNorm
    ↓
Extract [CLS] Token Output
    ↓
MLP Classification Head (7 classes) → Softmax
    ↓
Output: Gesture Class (0–6)
```

**Key Properties:**
| Property | Value |
|---|---|
| Input Size | 3 × 224 × 224 |
| Patch Size | 16 × 16 |
| Number of Patches | 196 |
| Transformer Layers (L) | 6–12 (use pretrained `vit_base_patch16_224`) |
| Parameters | ~86 million (base) |
| Inductive Bias | None (learns from data) |
| Framework | PyTorch + `timm` library |

**Decision:** Use a **pretrained ViT** (from `timm`) fine-tuned on the gesture dataset, rather than training from scratch, to compensate for limited dataset size.

---

## Key Differences

| Feature | CNN | Vision Transformer (ViT) |
|---|---|---|
| Core Operation | Convolution | Self-Attention |
| Receptive Field | Local (grows with depth) | Global (from first layer) |
| Inductive Bias | Yes (locality, translation) | None |
| Data Requirement | Works well on small datasets | Needs more data or pretraining |
| Inference Speed | Fast | Slower |
| Model Size | Small | Large |
| Interpretability | Feature maps | Attention maps |
| Best For | Edge/real-time deployment | High-accuracy tasks with data |

---

## Why Compare CNN vs ViT?

### What
A structured empirical comparison of two architectures on the same gesture recognition task.

### Why
- CNNs are the **established standard** for vision tasks; ViTs are the **emerging paradigm**.
- Understanding their trade-offs is academically and practically valuable.
- For a final year project, this comparison provides a clear research contribution.

### Key Research Questions
1. Does ViT achieve higher accuracy than CNN on hand gesture data?
2. Is ViT suitable for real-time inference, or is CNN the only viable option?
3. How does dataset size affect the performance gap between CNN and ViT?

---

## Expected Results

| Metric | CNN (Expected) | ViT (Expected) |
|---|---|---|
| Accuracy | 88–93% | 91–96% |
| Inference Speed | 25–35 FPS | 10–20 FPS |
| Latency per Frame | 30–40 ms | 50–100 ms |
| Training Time | Faster | Slower |
| Real-time Viability | High | Moderate |

**Hypothesis:** ViT will outperform CNN in accuracy but at a significant latency cost, making CNN more suitable for real-time cursor control deployment.
