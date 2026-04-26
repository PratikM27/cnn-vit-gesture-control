# architecture.md

# System Architecture

---

## System Pipeline (End-to-End)

```
Webcam Input
    ↓
Frame Capture (OpenCV)
    ↓
Hand Detection & Landmark Extraction (MediaPipe)
    ↓
ROI Cropping & Preprocessing
    ↓
Gesture Classification (CNN or ViT)
    ↓
Gesture-to-Action Mapping
    ↓
Cursor Control (PyAutoGUI)
```

---

## Module Breakdown

---

### 1. Input — Webcam

**What:** Captures real-time video frames from the system webcam.

**Why:** Provides a low-cost, ubiquitous input source without specialized hardware.

**How:**
- Uses `cv2.VideoCapture(0)` from OpenCV.
- Reads frames in BGR format, converts to RGB for MediaPipe.
- Target frame rate: 30 FPS capture.

**Trade-offs:**
| Factor | Detail |
|---|---|
| Lighting sensitivity | Performance degrades in low light |
| Resolution | Higher resolution = more detail but slower processing |
| Decision | Use 640×480 resolution as a balanced default |

---

### 2. Hand Detection — MediaPipe

**What:** Detects the hand in the frame and extracts 21 3D landmarks.

**Why:**
- Pre-trained, highly optimized by Google.
- Runs on CPU efficiently without a GPU.
- Eliminates the need to train a custom hand detector from scratch.

**How:**
- `mediapipe.solutions.hands` detects up to 2 hands per frame.
- Returns 21 (x, y, z) landmark coordinates normalized to [0, 1].
- Bounding box is derived from landmark extents to crop the hand Region of Interest (ROI).

**Decision:** Use only single-hand detection to simplify gesture mapping and reduce processing overhead.

**Trade-offs:**
| Factor | Detail |
|---|---|
| Accuracy | Very high for standard lighting conditions |
| Speed | Real-time capable on CPU |
| Limitation | Fails with heavy occlusion or extreme angles |

---

### 3. Preprocessing

**What:** Prepares the detected hand region for the classification model.

**Why:** Raw frames contain background noise; models perform better on clean, normalized inputs.

**How:**
1. Crop hand ROI from the full frame using MediaPipe bounding box.
2. Resize to model input size:
   - CNN: `64×64` or `128×128`
   - ViT: `224×224`
3. Normalize pixel values to `[0, 1]`.
4. Convert to tensor format.

**Trade-offs:**
| Factor | Detail |
|---|---|
| Smaller input (64×64) | Faster inference, may lose fine detail |
| Larger input (224×224) | Better accuracy for ViT, higher compute cost |

---

### 4. Model — CNN and Vision Transformer (ViT)

**What:** Classifies the preprocessed hand image into one of N gesture classes.

**Why:** Core of the project — comparing two fundamentally different deep learning architectures.

**How:**
- Two separate models are trained on the same dataset.
- At inference, one model is selected (switchable via config).
- Output: softmax probability vector → argmax → gesture class label.

*(See `models.md` for full architecture details.)*

---

### 5. Gesture Mapping

**What:** Translates a predicted gesture class label into a cursor action.

**Why:** Bridges AI output to OS-level interaction.

**How:**

| Gesture | Action |
|---|---|
| Open Palm | Move cursor |
| Index Finger Point | Left click |
| Two Fingers Up | Right click |
| Fist | Stop / Neutral |
| Pinch | Drag |
| Three Fingers Up | Scroll up |
| Three Fingers Down | Scroll down |

- Gesture label from model → lookup in action dictionary → call PyAutoGUI function.

---

### 6. Cursor Control — PyAutoGUI

**What:** Executes OS-level mouse and cursor actions.

**Why:** Cross-platform library for programmatic GUI automation; simple API.

**How:**
- `pyautogui.moveTo(x, y)` — maps hand position to screen coordinates.
- `pyautogui.click()` — triggers left click.
- `pyautogui.rightClick()` — triggers right click.
- `pyautogui.scroll(n)` — scrolls up/down.
- Coordinate mapping: hand x/y (normalized 0–1) → scaled to screen resolution.

**Trade-offs:**
| Factor | Detail |
|---|---|
| Smoothing | Raw hand movement is jittery; apply moving average filter |
| Fail-safe | PyAutoGUI has a corner-exit failsafe enabled by default |
| Latency | Each PyAutoGUI call adds ~1–5ms overhead |

---

## Design Decisions

| Decision | Rationale |
|---|---|
| MediaPipe for hand detection | Pre-trained, fast, no GPU needed |
| Separate CNN and ViT models | Clean comparison with identical training data |
| PyAutoGUI for cursor control | Simple, cross-platform, no OS-level driver needed |
| OpenCV for video capture | Industry-standard, well-documented |
| Single-hand detection only | Reduces complexity; sufficient for cursor control |

---

## Overall Trade-offs

| Trade-off | CNN | ViT |
|---|---|---|
| Inference Speed | Fast (~30+ FPS) | Slower (~10–20 FPS) |
| Accuracy | Good | Higher (with enough data) |
| Hardware Requirement | CPU sufficient | Prefers GPU |
| Model Size | Small | Large |
| Real-time Suitability | High | Moderate |
