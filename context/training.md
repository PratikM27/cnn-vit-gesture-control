# training.md

# Training Setup

---

## Overview

Both models (CNN and ViT) are trained independently on the same dataset using the same training strategy to ensure a fair comparison.

---

## Training Environment

| Component | Detail |
|---|---|
| Framework | PyTorch |
| Hardware (preferred) | GPU (NVIDIA CUDA) or Apple MPS |
| Hardware (fallback) | CPU (slower but functional) |
| Python Version | 3.9+ |
| Key Libraries | `torch`, `torchvision`, `timm`, `numpy`, `matplotlib`, `sklearn` |

---

## Hyperparameters

| Hyperparameter | CNN | ViT |
|---|---|---|
| Input Image Size | 64×64 or 128×128 | 224×224 |
| Batch Size | 32 | 16–32 |
| Learning Rate | 1e-3 (custom) | 1e-4 (fine-tuning) |
| Epochs | 30–50 | 20–30 |
| Dropout | 0.4–0.5 | 0.1 (in head) |
| Weight Decay | 1e-4 | 1e-4 |
| Early Stopping Patience | 5 epochs | 5 epochs |

**Why lower LR for ViT?**
Fine-tuning a pretrained model requires a smaller learning rate to avoid destroying pretrained weights.

---

## Loss Function

### What
**Cross-Entropy Loss** (categorical)

### Why
- Standard loss for multi-class classification.
- Works well with softmax output.
- Penalizes confident wrong predictions heavily.

### Formula
```
Loss = -Σ y_true * log(y_pred)
```

**Optional:** Use **Label Smoothing (ε = 0.1)** to prevent overconfidence and improve generalization.

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

---

## Optimizer

| Model | Optimizer | Why |
|---|---|---|
| CNN | Adam | Fast convergence, works well for custom CNNs |
| ViT | AdamW | Adam with decoupled weight decay; preferred for transformers |

```python
# CNN
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# ViT
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
```

---

## Learning Rate Scheduler

| Scheduler | Model | Detail |
|---|---|---|
| StepLR | CNN | Reduce LR by 0.1 every 10 epochs |
| CosineAnnealingLR | ViT | Smooth LR decay; works well for transformers |

```python
# CNN
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# ViT
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
```

---

## Training Strategy

### CNN — Train from Scratch
1. Initialize weights randomly (or use Kaiming init).
2. Train all layers from epoch 1.
3. Apply dropout and batch normalization for regularization.

### ViT — Transfer Learning + Fine-tuning
1. Load pretrained `vit_base_patch16_224` weights from `timm`.
2. **Phase 1 (5 epochs):** Freeze all layers except the classification head. Train head only.
3. **Phase 2 (remaining epochs):** Unfreeze all layers. Fine-tune end-to-end with low LR.

```python
import timm
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=7)
```

---

## Early Stopping

**What:** Stop training if validation loss does not improve for N consecutive epochs.

**Why:** Prevents overfitting; saves compute time.

```python
# Pseudocode
if val_loss < best_val_loss:
    best_val_loss = val_loss
    save_model_checkpoint()
    patience_counter = 0
else:
    patience_counter += 1
    if patience_counter >= patience:
        print("Early stopping triggered.")
        break
```

---

## Metrics Tracked During Training

| Metric | Logged Per Epoch |
|---|---|
| Training Loss | ✅ |
| Validation Loss | ✅ |
| Training Accuracy | ✅ |
| Validation Accuracy | ✅ |
| Learning Rate | ✅ |

- Plot loss and accuracy curves at the end of training.
- Save best model checkpoint based on **minimum validation loss**.

---

## Trade-offs

| Factor | Detail |
|---|---|
| Batch size 16 vs 32 | Smaller batch = noisier gradients but better generalization for ViT |
| More epochs | Risk of overfitting without early stopping |
| Pretrained ViT | Needs less data but fine-tuning must be done carefully |
| CNN from scratch | More transparent; fully customizable; more prone to overfit on small data |
