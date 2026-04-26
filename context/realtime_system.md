# realtime_system.md

# Real-Time System

---

## Overview

The real-time pipeline continuously reads webcam frames, detects hands, classifies gestures, and executes cursor actions — all within a single loop running at interactive frame rates.

---

## How the Real-Time Pipeline Works

```
while True:
    1. Capture frame from webcam (OpenCV)
    2. Detect hand & extract landmarks (MediaPipe)
    3. If hand detected:
        a. Crop ROI from frame
        b. Preprocess ROI (resize, normalize, tensorize)
        c. Run model inference (CNN or ViT)
        d. Get predicted gesture class
        e. Apply smoothing / debounce filter
        f. Map gesture → cursor action
        g. Execute action via PyAutoGUI
    4. Display annotated frame (optional debug view)
    5. Check for exit key ('q')
```

---

## Integration with OpenCV and MediaPipe

### OpenCV

**What:** Handles video capture, frame processing, and display.

**Key calls:**
```python
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
cv2.imshow("Gesture Control", annotated_frame)
```

### MediaPipe

**What:** Detects hand landmarks in real time.

**Key calls:**
```python
import mediapipe as mp
hands = mp.solutions.hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
results = hands.process(rgb_frame)

if results.multi_hand_landmarks:
    landmarks = results.multi_hand_landmarks[0]
    # Extract bounding box from landmark extents
```

**Landmark indexing (key points):**
| Landmark | Index |
|---|---|
| Wrist | 0 |
| Index fingertip | 8 |
| Middle fingertip | 12 |
| Ring fingertip | 16 |
| Pinky fingertip | 20 |
| Thumb tip | 4 |

---

## Gesture to Action Mapping

### Coordinate Mapping (Hand → Screen)

```python
import pyautogui

screen_w, screen_h = pyautogui.size()
frame_h, frame_w = frame.shape[:2]

# Normalize hand center to screen coordinates
x = int((hand_cx / frame_w) * screen_w)
y = int((hand_cy / frame_h) * screen_h)
pyautogui.moveTo(x, y)
```

### Action Dispatch Table

```python
ACTION_MAP = {
    "open_palm":          lambda x, y: pyautogui.moveTo(x, y),
    "index_point":        lambda x, y: pyautogui.click(x, y),
    "two_fingers_up":     lambda x, y: pyautogui.rightClick(x, y),
    "fist":               lambda x, y: None,  # Neutral, no action
    "pinch":              lambda x, y: pyautogui.mouseDown(),
    "three_fingers_up":   lambda x, y: pyautogui.scroll(3),
    "three_fingers_down": lambda x, y: pyautogui.scroll(-3),
}
```

---

## Smoothing and Stability

### Problem
Raw hand position from MediaPipe is jittery — even small natural hand tremors cause cursor to shake.

### Solution — Moving Average Filter

```python
from collections import deque
import numpy as np

history = deque(maxlen=5)  # Store last 5 (x, y) positions

history.append((raw_x, raw_y))
smooth_x = int(np.mean([p[0] for p in history]))
smooth_y = int(np.mean([p[1] for p in history]))
```

### Debounce for Click Actions
- Require gesture to be held for **N consecutive frames** (e.g., 3) before triggering a click.
- Prevents accidental clicks from transient misclassifications.

```python
gesture_counter = {}
DEBOUNCE_FRAMES = 3

gesture_counter[predicted_class] = gesture_counter.get(predicted_class, 0) + 1
if gesture_counter[predicted_class] >= DEBOUNCE_FRAMES:
    execute_action(predicted_class)
    gesture_counter = {}  # Reset after action
```

---

## Performance Considerations

| Factor | Strategy |
|---|---|
| Frame rate bottleneck | Run model inference on every other frame if FPS drops |
| Model loading time | Load model once at startup; keep in memory |
| GPU vs CPU | Use `torch.device('cuda')` if available; fallback to CPU |
| PyAutoGUI fail-safe | Keep enabled; move mouse to corner to kill process if needed |
| MediaPipe confidence | Tune `min_detection_confidence` to reduce false positives |
| Threading | Optionally separate capture and inference threads for higher FPS |

### Target Performance

| Model | Target FPS | Max Acceptable Latency |
|---|---|---|
| CNN | ≥ 25 FPS | ≤ 40 ms per frame |
| ViT | ≥ 15 FPS | ≤ 70 ms per frame |

---

## Debug View (Optional)

Display an annotated window showing:
- Live webcam feed.
- MediaPipe hand landmarks overlaid.
- Current predicted gesture label.
- Current FPS counter.

```python
mp.solutions.drawing_utils.draw_landmarks(
    frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
)
cv2.putText(frame, f"Gesture: {predicted_label}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
```
