# dataset.md

# Dataset

---

## Gesture Classes

**Total Classes:** 7 (expandable)

| Class ID | Gesture Name | Description | Cursor Action |
|---|---|---|---|
| 0 | Open Palm | All 5 fingers extended, spread | Move cursor |
| 1 | Index Point | Only index finger extended | Left click |
| 2 | Two Fingers Up | Index + middle fingers extended | Right click |
| 3 | Fist | All fingers closed | Neutral / Stop |
| 4 | Pinch | Thumb + index fingertips touching | Drag |
| 5 | Three Fingers Up | Index + middle + ring extended | Scroll up |
| 6 | Three Fingers Down | Three fingers curled downward | Scroll down |

---

## Data Collection Method

### What
Images of hand gestures collected via webcam and optionally supplemented with public datasets.

### Why
- Custom-collected data matches the exact lighting, angle, and hardware conditions of deployment.
- Public datasets provide scale and variety.

### How

**Option A — Custom Collection (Recommended for viva credibility):**
1. Use a Python script with OpenCV to capture frames.
2. Perform each gesture in front of the webcam.
3. Save cropped ROI images (from MediaPipe bounding box) per class.
4. Target: **200–500 images per class** (minimum viable), **1000+ preferred**.

**Option B — Public Datasets (Supplementary):**
- [HaGRID (Hand Gesture Recognition Image Dataset)](https://github.com/hukenovs/hagrid) — 18 gesture classes, 552K images.
- [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) — American Sign Language.
- Subset and relabel classes to match the 7 defined gestures.

**Decision:** Collect custom data + augment with HaGRID subset for volume.

---

## Preprocessing Steps

| Step | Detail |
|---|---|
| Hand Detection | MediaPipe detects hand; extract bounding box |
| ROI Cropping | Crop hand region with 10px padding |
| Resize | CNN: `64×64` or `128×128`; ViT: `224×224` |
| Color Space | BGR → RGB conversion |
| Normalization | Pixel values scaled from [0, 255] → [0.0, 1.0] |
| Tensor Conversion | NumPy array → PyTorch tensor (C, H, W format) |

---

## Data Augmentation

### What
Artificially expand the dataset by applying transformations to existing images.

### Why
- Prevents overfitting, especially with small custom datasets.
- Improves model generalization across lighting, position, and hand size variation.

### How (using `torchvision.transforms`)

| Augmentation | Parameters | Purpose |
|---|---|---|
| Random Horizontal Flip | p=0.5 | Handle left/right hand variation |
| Random Rotation | ±15° | Handle slight tilt variation |
| Color Jitter | brightness=0.3, contrast=0.3 | Handle lighting changes |
| Random Crop | 90% of original | Handle slight framing shifts |
| Gaussian Blur | kernel=3, p=0.2 | Add robustness to noise |
| Normalize | mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5] | Standardize inputs |

**Note:** Augmentation is applied **only to the training set**, not validation or test sets.

---

## Train / Validation / Test Split

| Split | Percentage | Purpose |
|---|---|---|
| Training | 70% | Model learning |
| Validation | 15% | Hyperparameter tuning, early stopping |
| Test | 15% | Final unbiased evaluation |

- Split is **stratified** to ensure balanced class distribution across all splits.
- Use `sklearn.model_selection.train_test_split` with `stratify=y`.

**Trade-offs:**
| Factor | Detail |
|---|---|
| Small dataset (<500/class) | Risk of overfitting; augmentation becomes critical |
| Large dataset (1000+/class) | Better generalization; ViT benefits more |
| Stratified split | Ensures minority gestures are well-represented in all splits |
