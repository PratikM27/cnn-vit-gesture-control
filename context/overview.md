# overview.md

# Real-Time Cursor Control Using Hand Gestures with Vision Transformer (CNN vs ViT Comparison)

---

## Problem Statement

### What
Traditional cursor control relies on physical input devices (mouse, touchpad). These create barriers for individuals with motor disabilities and limit natural human-computer interaction.

### Why
- Physical peripherals are inaccessible for users with mobility impairments.
- Touchless interfaces are increasingly relevant in hygiene-sensitive and hands-free environments.
- Gesture-based control is a growing area of HCI (Human-Computer Interaction) research.

### How (Problem Definition)
The system must:
- Detect hand gestures in real time via webcam.
- Classify gestures accurately (even under varying lighting and backgrounds).
- Translate gestures into cursor actions (move, click, scroll down , scroll up , drag).

---

## Objectives

1. **Build a real-time hand gesture recognition system** using webcam input.
2. **Implement two classification models:**
   - Convolutional Neural Network (CNN)
   - Vision Transformer (ViT)
3. **Compare CNN vs ViT** on accuracy, speed (FPS), and latency.
4. **Map recognized gestures** to OS-level cursor actions using PyAutoGUI.
5. **Evaluate performance** using standard metrics: Accuracy, Precision, Recall, F1-score.

---

## Motivation

### Why This Project?
- Bridges the gap between accessibility technology and deep learning.
- Provides a practical testbed to compare **classical deep learning (CNN)** vs **modern transformer-based (ViT)** architectures on a real-world task.
- Explores the trade-off between **model complexity** and **real-time performance**.
- Relevant to assistive technology, gaming, AR/VR, and touchless kiosks.

---

## Expected Outcome

| Outcome | Description |
|---|---|
| Functional System | A working real-time gesture-controlled cursor on a standard PC |
| Trained Models | Two trained models (CNN and ViT) on the same gesture dataset |
| Comparison Report | Quantitative comparison of CNN vs ViT on all key metrics |
| Viva-Ready Documentation | Structured analysis suitable for academic evaluation |

- **CNN** is expected to be faster and more efficient on limited hardware.
- **ViT** is expected to achieve higher accuracy given sufficient data, at the cost of latency.
- The final system will run at an acceptable frame rate (≥15 FPS) for real-time use.
