# experiments.md

# Experiments Log

---

## Experiment Template

Copy and fill this template for each experiment run.

---

### Experiment #[N] — [Short Description]

**Date:** YYYY-MM-DD
**Objective:** What are you trying to find out or improve?

---

#### Configuration

| Parameter | Value |
|---|---|
| Model | CNN / ViT |
| Input Size | |
| Batch Size | |
| Learning Rate | |
| Optimizer | |
| Epochs Run | |
| Early Stopping | Yes / No |
| Pretrained Weights | Yes / No |
| Data Augmentation | Yes / No |
| Dataset Size (train) | |
| Dataset Size (val) | |
| Dataset Size (test) | |

---

#### Results

| Metric | Value |
|---|---|
| Training Loss (final) | |
| Validation Loss (final) | |
| Training Accuracy | |
| Validation Accuracy | |
| Test Accuracy | |
| Precision (macro avg) | |
| Recall (macro avg) | |
| F1-Score (macro avg) | |
| Average FPS (real-time) | |
| Avg Inference Latency (ms) | |

---

#### Per-Class Performance

| Gesture Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Open Palm | | | | |
| Index Point | | | | |
| Two Fingers Up | | | | |
| Fist | | | | |
| Pinch | | | | |
| Three Fingers Up | | | | |
| Three Fingers Down | | | | |

---

#### Observations

- **What worked well:**
  -
  -

- **What did not work:**
  -
  -

- **Unexpected findings:**
  -

- **Confusion matrix notes:** (e.g., which classes were most confused with each other)
  -

---

#### Changes for Next Experiment

- [ ]
- [ ]
- [ ]

---

---

## Experiments Index

| # | Model | Key Change | Test Accuracy | FPS | Notes |
|---|---|---|---|---|---|
| 1 | CNN | Baseline | | | |
| 2 | CNN | + Augmentation | | | |
| 3 | CNN | + LR Scheduler | | | |
| 4 | ViT | Pretrained, head only | | | |
| 5 | ViT | Full fine-tune | | | |
| 6 | ViT | + Augmentation | | | |
| 7 | CNN | Final config | | | |
| 8 | ViT | Final config | | | |

---

## General Observations Section

*(Fill this in progressively as experiments are completed)*

### Key Findings

-

### Model Comparison Summary (Preliminary)

-

### Failure Cases

-

### Surprising Results

-
