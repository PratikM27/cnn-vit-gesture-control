"""
generate_report.py — Research Paper Style Report Generator
============================================================
Generates a structured markdown report with academic sections.

Usage:
    python analysis/generate_report.py
"""

import os
import sys
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PATHS, GESTURE_CLASSES, GESTURE_LABELS, CNN_CONFIG, VIT_CONFIG


def load_all_results():
    """Load all available results."""
    results = {}
    for model_type in ['cnn', 'vit']:
        eval_path = os.path.join(PATHS["results"], f"{model_type}_eval_results.json")
        metrics_path = os.path.join(PATHS["results"], f"{model_type}_metrics.json")
        
        model_data = {}
        if os.path.exists(eval_path):
            with open(eval_path, 'r') as f:
                model_data['eval'] = json.load(f)
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                model_data['training'] = json.load(f)
        results[model_type] = model_data
    
    return results


def generate_report():
    """Generate a comprehensive research-style report."""
    results = load_all_results()
    cnn = results.get('cnn', {})
    vit = results.get('vit', {})
    cnn_eval = cnn.get('eval', {})
    vit_eval = vit.get('eval', {})
    cnn_train = cnn.get('training', {})
    vit_train = vit.get('training', {})
    
    report = f"""# Real-Time Cursor Control Using Hand Gestures with Vision Transformer

**A Comparative Study of CNN and Vision Transformer for Gesture-Based Human-Computer Interaction**

*Date: {datetime.now().strftime('%B %d, %Y')}*

---

## Abstract

This project presents a real-time gesture-controlled cursor system that uses webcam
input to detect and classify hand gestures, which are then mapped to cursor actions.
Two deep learning architectures are compared: a Convolutional Neural Network (CNN)
serving as the baseline, and a Vision Transformer (ViT) as the proposed system.
The CNN model achieves {cnn_eval.get('accuracy', 'N/A')}% test accuracy at
{cnn_eval.get('latency', {}).get('fps', 'N/A')} FPS, while the ViT model achieves
{vit_eval.get('accuracy', 'N/A')}% accuracy at {vit_eval.get('latency', {}).get('fps', 'N/A')} FPS.
Results demonstrate the trade-off between accuracy and computational efficiency
in real-time vision applications.

---

## 1. Introduction

### 1.1 Background
Traditional human-computer interaction relies on physical input devices such as
mice, keyboards, and touchpads. These create barriers for users with motor
disabilities and do not leverage the natural expressiveness of human gestures.

### 1.2 Problem Statement
The system must:
- Detect hand gestures in real time via webcam
- Classify gestures accurately under varying conditions
- Translate gestures into cursor actions with low latency (<100ms)

### 1.3 Objectives
1. Build a real-time hand gesture recognition system using webcam input
2. Implement and compare CNN (baseline) and ViT (proposed) classifiers
3. Evaluate performance using accuracy, F1-score, FPS, and latency metrics
4. Deploy a working gesture-controlled cursor system

### 1.4 Contributions
- A modular, end-to-end gesture control pipeline
- Empirical comparison of CNN vs ViT on hand gesture classification
- Analysis of real-time feasibility for transformer-based models

---

## 2. Literature Review

### 2.1 Convolutional Neural Networks
CNNs use convolutional filters to extract hierarchical spatial features (edges,
textures, shapes) through local receptive fields. They exhibit strong inductive
biases (locality, translation equivariance) that make them data-efficient and
fast at inference.

### 2.2 Vision Transformers
Vision Transformers (ViTs) adapt the transformer architecture from NLP to computer
vision by dividing images into patches, embedding them as tokens, and processing
them with multi-head self-attention. ViTs capture global context from the first
layer but require more data and computation.

### 2.3 Hand Gesture Recognition
MediaPipe Hands (Google) provides real-time hand landmark detection using a
lightweight ML pipeline. Combined with deep learning classifiers, it enables
accurate gesture recognition without specialized hardware.

---

## 3. Methodology

### 3.1 System Architecture

```
Webcam → Frame Capture (OpenCV) → Hand Detection (MediaPipe)
    → ROI Extraction → Preprocessing → Model Inference (CNN/ViT)
    → Gesture Classification → Action Mapping → Cursor Control (PyAutoGUI)
```

### 3.2 Dataset

**Gesture Classes:** {len(GESTURE_CLASSES)} classes

| Class | Gesture | Cursor Action |
|-------|---------|--------------|
"""
    
    for class_id, class_name in GESTURE_CLASSES.items():
        label = GESTURE_LABELS.get(class_name, class_name)
        report += f"| {class_id} | {label} | — |\n"
    
    report += f"""
- **Collection:** Custom images captured via webcam using MediaPipe hand ROI extraction
- **Split:** 70% train / 15% validation / 15% test (stratified)
- **Augmentation:** Random flip, rotation (±15°), color jitter, Gaussian blur

### 3.3 CNN Architecture (Baseline)

```
Input (3×{CNN_CONFIG['input_size']}×{CNN_CONFIG['input_size']})
  → Conv2D(3→32) → BN → ReLU → MaxPool
  → Conv2D(32→64) → BN → ReLU → MaxPool
  → Conv2D(64→128) → BN → ReLU → MaxPool
  → Conv2D(128→256) → BN → ReLU → MaxPool
  → AdaptiveAvgPool → FC(256→512) → ReLU → Dropout
  → FC(512→{len(GESTURE_CLASSES)})
```

- **Optimizer:** Adam (LR={CNN_CONFIG['learning_rate']})
- **Scheduler:** StepLR (step={CNN_CONFIG['step_size']}, gamma={CNN_CONFIG['gamma']})

### 3.4 Vision Transformer Architecture (Proposed)

```
Input (3×{VIT_CONFIG['input_size']}×{VIT_CONFIG['input_size']})
  → Patch Embedding (16×16 patches → 196 tokens)
  → [CLS] Token + Positional Encoding
  → Transformer Encoder × 12 layers
  → [CLS] Output → Classification Head → {len(GESTURE_CLASSES)} classes
```

- **Model:** {VIT_CONFIG['model_name']} (pretrained on ImageNet)
- **Training:** Phase 1 (head only, {VIT_CONFIG['freeze_epochs']} epochs) + Phase 2 (full fine-tune)
- **Optimizer:** AdamW (LR={VIT_CONFIG['learning_rate']})
- **Scheduler:** CosineAnnealingLR

---

## 4. Results

### 4.1 Classification Performance

| Metric | CNN | ViT |
|--------|-----|-----|
| Accuracy (%) | {cnn_eval.get('accuracy', 'TBD'):.2f} | {vit_eval.get('accuracy', 'TBD'):.2f} |
| Precision (%) | {cnn_eval.get('precision_macro', 'TBD'):.2f} | {vit_eval.get('precision_macro', 'TBD'):.2f} |
| Recall (%) | {cnn_eval.get('recall_macro', 'TBD'):.2f} | {vit_eval.get('recall_macro', 'TBD'):.2f} |
| F1-Score (%) | {cnn_eval.get('f1_macro', 'TBD'):.2f} | {vit_eval.get('f1_macro', 'TBD'):.2f} |

### 4.2 Speed and Efficiency

| Metric | CNN | ViT |
|--------|-----|-----|
| FPS | {cnn_eval.get('latency', {}).get('fps', 'TBD'):.1f} | {vit_eval.get('latency', {}).get('fps', 'TBD'):.1f} |
| Latency (ms) | {cnn_eval.get('latency', {}).get('mean_ms', 'TBD'):.2f} | {vit_eval.get('latency', {}).get('mean_ms', 'TBD'):.2f} |
| Model Size (MB) | {cnn_eval.get('model_size_mb', 'TBD'):.2f} | {vit_eval.get('model_size_mb', 'TBD'):.2f} |
| Parameters (M) | {cnn_eval.get('total_params', 0)/1e6:.2f} | {vit_eval.get('total_params', 0)/1e6:.2f} |

---

## 5. Discussion

### 5.1 CNN Strengths
- Faster inference due to local operations
- Smaller model size, suitable for edge deployment
- Strong inductive biases reduce data requirements

### 5.2 ViT Strengths
- Global attention captures spatial relationships between fingers
- Higher accuracy potential with sufficient data
- Attention maps provide interpretability

### 5.3 Trade-offs
- **Accuracy vs Speed:** ViT trades speed for accuracy
- **Data Efficiency:** CNN works better with smaller datasets
- **Hardware Requirements:** ViT benefits significantly from GPU

### 5.4 Real-Time Feasibility
For real-time cursor control at ≥15 FPS:
- CNN comfortably meets this threshold
- ViT may require optimization (quantization, ONNX) for real-time use

---

## 6. Conclusion

This project demonstrates a functional real-time gesture-controlled cursor system
comparing CNN and Vision Transformer architectures. The CNN provides a reliable
baseline with fast inference, while the ViT shows the potential of attention-based
models for gesture recognition. The modular design allows easy switching between
models and extension to additional gestures.

---

## 7. Future Work

1. **Hybrid CNN+ViT model** combining local and global features
2. **Landmark-based classification** using MediaPipe coordinates directly
3. **Dynamic gesture recognition** using temporal sequences
4. **Model distillation** to compress ViT into a faster student model
5. **Edge deployment** on Raspberry Pi or mobile devices

---

## References

1. Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," ICLR 2021.
2. Zhang et al., "MediaPipe Hands: On-device Real-time Hand Tracking," 2020.
3. He et al., "Deep Residual Learning for Image Recognition," CVPR 2016.
4. Howard et al., "MobileNets: Efficient Convolutional Neural Networks," 2017.
5. Vaswani et al., "Attention Is All You Need," NeurIPS 2017.
"""
    
    # Save report
    save_path = os.path.join(PATHS["results"], "research_report.md")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"  Research report saved to: {save_path}")
    return report


def main():
    print("=" * 60)
    print("  GENERATING RESEARCH REPORT")
    print("=" * 60)
    generate_report()
    print("\n  Done!")


if __name__ == "__main__":
    main()
