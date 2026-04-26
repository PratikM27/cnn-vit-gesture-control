# results.md

# Results & Evaluation

---

## Evaluation Metrics Explained

| Metric | What It Measures | Formula |
|---|---|---|
| **Accuracy** | Overall correct predictions | TP+TN / Total |
| **Precision** | Of predicted positives, how many are correct | TP / (TP + FP) |
| **Recall** | Of actual positives, how many are detected | TP / (TP + FN) |
| **F1-Score** | Harmonic mean of Precision and Recall | 2 × (P × R) / (P + R) |
| **FPS** | Frames processed per second in real-time loop | 1 / avg_frame_time |
| **Latency** | Time from frame capture to cursor action | ms per frame |

*All classification metrics (Accuracy, Precision, Recall, F1) are reported as **macro-averaged** across all 7 gesture classes.*

---

## Main Comparison Table — CNN vs ViT

*(Fill in after experiments are complete)*

| Metric | CNN | ViT | Winner |
|---|---|---|---|
| Test Accuracy (%) | | | |
| Precision (macro) | | | |
| Recall (macro) | | | |
| F1-Score (macro) | | | |
| Average FPS | | | |
| Avg Latency (ms) | | | |
| Model Size (MB) | | | |
| Training Time (min) | | | |
| Parameters (M) | | | |

---

## Per-Class F1-Score Comparison

| Gesture Class | CNN F1 | ViT F1 |
|---|---|---|
| Open Palm | | |
| Index Point | | |
| Two Fingers Up | | |
| Fist | | |
| Pinch | | |
| Three Fingers Up | | |
| Three Fingers Down | | |
| **Macro Average** | | |

---

## Confusion Matrix Notes

### CNN Confusion Matrix
*(Paste or summarize here after evaluation)*

- Most confused classes:
- Reason for confusion:

### ViT Confusion Matrix
*(Paste or summarize here after evaluation)*

- Most confused classes:
- Reason for confusion:

---

## Training Curves

### Loss Curves
*(Attach plots: training loss vs validation loss for both models)*

| Observation | CNN | ViT |
|---|---|---|
| Overfitting signs | | |
| Epoch of best val loss | | |
| Loss convergence quality | | |

### Accuracy Curves
*(Attach plots: training accuracy vs validation accuracy for both models)*

---

## Real-Time Performance Summary

| Scenario | CNN FPS | ViT FPS |
|---|---|---|
| Idle (no hand detected) | | |
| Active cursor movement | | |
| Click gesture | | |
| Scroll gesture | | |
| **Average** | | |

---

## Key Findings

*(Fill in after experiments)*

1. **Accuracy:** [Which model won and by how much?]
2. **Speed:** [Which model is faster and by what factor?]
3. **Real-time viability:** [Is ViT fast enough for live use?]
4. **Per-class insight:** [Which gestures are hardest to classify and why?]
5. **Practical recommendation:** [Which model would you deploy and why?]

---

## Statistical Notes

- Report confidence intervals if multiple runs are performed.
- Ensure test set was **never seen during training or validation**.
- Results are reproducible with `torch.manual_seed(42)` and `numpy.random.seed(42)`.

---

## Template: Final Results Summary for Report / Viva

> "The CNN model achieved **X% accuracy** on the test set with an average inference speed of **Y FPS**, making it suitable for real-time gesture-controlled cursor interaction. The Vision Transformer model achieved **X% accuracy** — a difference of **+Z%** — however at a reduced speed of **Y FPS**. Given the real-time constraint of the system, the CNN model offers a better accuracy-speed trade-off for deployment, while ViT demonstrates superior classification performance when latency is not a limiting factor."
