"""
gesture_control.py — Main Real-Time Gesture Control System
============================================================
Integrates webcam, hand detection, model inference, and cursor
control into a single real-time loop.

Usage:
    python realtime/gesture_control.py --model cnn
    python realtime/gesture_control.py --model vit
    python realtime/gesture_control.py --model cnn --no-cursor   # Debug mode
"""

import os
import sys
import argparse
import time

import cv2
import numpy as np
import torch
from torchvision import transforms

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CNN_CONFIG, VIT_CONFIG, PATHS, REALTIME, MEDIAPIPE,
    NORMALIZE_MEAN, NORMALIZE_STD, NUM_CLASSES,
    GESTURE_CLASSES, GESTURE_LABELS, ACTION_MAP
)
from models.cnn_model import build_cnn_model
from models.vit_model import build_vit_model
from realtime.hand_detector import HandDetector
from realtime.cursor_controller import CursorController
from realtime.gesture_smoother import CursorSmoother, GestureDebouncer, FPSCounter


class GestureControlSystem:
    """
    Complete real-time gesture control system.
    
    Pipeline:
        Webcam → Hand Detection → ROI → Preprocess → Model → Gesture → Cursor Action
    """
    
    def __init__(self, model_type='cnn', checkpoint_path=None,
                 enable_cursor=True, show_debug=True):
        """
        Args:
            model_type: 'cnn' or 'vit'
            checkpoint_path: Path to model checkpoint
            enable_cursor: Whether to actually move the cursor
            show_debug: Show debug visualization window
        """
        self.model_type = model_type
        self.enable_cursor = enable_cursor
        self.show_debug = show_debug
        
        # Select config
        self.config = CNN_CONFIG if model_type == 'cnn' else VIT_CONFIG
        self.input_size = self.config["input_size"]
        
        # Setup device
        self.device = self._setup_device()
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        
        # Class names (sorted to match ImageFolder ordering)
        self.class_names = sorted(GESTURE_CLASSES.values())
        
        # Setup components
        self.hand_detector = HandDetector(
            max_num_hands=MEDIAPIPE["max_num_hands"],
            min_detection_confidence=MEDIAPIPE["min_detection_confidence"],
            min_tracking_confidence=MEDIAPIPE["min_tracking_confidence"],
            roi_padding=MEDIAPIPE["roi_padding"],
        )
        
        if enable_cursor:
            self.cursor_controller = CursorController(
                scroll_amount=REALTIME["scroll_amount"]
            )
        else:
            self.cursor_controller = None
        
        self.cursor_smoother = CursorSmoother(
            window_size=REALTIME["smoothing_window"]
        )
        
        self.gesture_debouncer = GestureDebouncer(
            debounce_frames=REALTIME["debounce_frames"],
            click_cooldown_ms=REALTIME["click_cooldown_ms"],
            confidence_threshold=REALTIME["confidence_threshold"],
        )
        
        self.fps_counter = FPSCounter()
        
        # Preprocessing transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ])
        
        print(f"\n  System initialized: {model_type.upper()} model on {self.device}")
        if not enable_cursor:
            print("  ⚠ Cursor control DISABLED (debug mode)")
    
    def _setup_device(self):
        """Setup compute device."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"  Device: CUDA ({torch.cuda.get_device_name(0)})")
        else:
            device = torch.device("cpu")
            print("  Device: CPU")
        return device
    
    def _load_model(self, checkpoint_path=None):
        """Load the trained model."""
        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                PATHS["checkpoints"],
                f"best_{self.model_type}_model.pth"
            )
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}\n"
                f"Train the model first: python training/train.py --model {self.model_type}"
            )
        
        print(f"  Loading model from: {checkpoint_path}")
        
        if self.model_type == 'cnn':
            model = build_cnn_model(
                model_name=self.config["model_name"],
                num_classes=NUM_CLASSES,
                dropout=self.config["dropout"],
            )
        else:
            model = build_vit_model(
                model_name=self.config["model_name"],
                num_classes=NUM_CLASSES,
                pretrained=False,
                dropout=self.config["dropout"],
            )
        
        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        print(f"  Model loaded (epoch {checkpoint.get('epoch', '?')}, "
              f"val_acc={checkpoint.get('val_acc', 0):.2f}%)")
        
        return model
    
    def preprocess_roi(self, roi):
        """
        Preprocess the ROI for model inference.
        
        Args:
            roi: BGR image (H, W, 3)
        
        Returns:
            tensor: (1, 3, input_size, input_size)
        """
        # Convert BGR → RGB
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        tensor = self.transform(rgb_roi)
        tensor = tensor.unsqueeze(0)  # Add batch dimension
        tensor = tensor.to(self.device)
        
        return tensor
    
    def predict(self, tensor):
        """
        Run model inference.
        
        Args:
            tensor: Preprocessed input (1, 3, H, W)
        
        Returns:
            class_name: Predicted gesture class name
            confidence: Softmax confidence (0-1)
        """
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = probs.max(1)
        
        class_name = self.class_names[predicted_idx.item()]
        confidence_val = confidence.item()
        
        return class_name, confidence_val
    
    def draw_debug_ui(self, frame, gesture_name, confidence, action_type,
                       fps, latency_ms, hand_detected):
        """Draw debug overlay on the frame."""
        h, w = frame.shape[:2]
        
        # Info panel background
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - 300, 0), (w, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Model info
        x0 = w - 290
        cv2.putText(frame, f"Model: {self.model_type.upper()}", (x0, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # FPS and latency
        fps_color = (0, 255, 0) if fps >= 15 else (0, 165, 255) if fps >= 10 else (0, 0, 255)
        cv2.putText(frame, f"FPS: {fps:.1f}", (x0, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)
        cv2.putText(frame, f"Latency: {latency_ms:.1f}ms", (x0, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Hand status
        if hand_detected:
            cv2.putText(frame, "Hand: DETECTED", (x0, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(frame, "Hand: Not Found", (x0, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Gesture prediction
        if gesture_name:
            label = GESTURE_LABELS.get(gesture_name, gesture_name)
            conf_color = (0, 255, 0) if confidence > 0.8 else (0, 200, 255) if confidence > 0.6 else (0, 0, 255)
            cv2.putText(frame, f"Gesture: {label}", (x0, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, conf_color, 2)
            cv2.putText(frame, f"Conf: {confidence:.2f}", (x0, 165),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, conf_color, 1)
            cv2.putText(frame, f"Action: {action_type}", (x0, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)
        
        # Instructions at bottom
        cv2.putText(frame, "Press 'Q' to quit | Move mouse to top-left corner for fail-safe",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        return frame
    
    def run(self):
        """Main real-time loop."""
        print("\n" + "=" * 60)
        print("  REAL-TIME GESTURE CONTROL")
        print("=" * 60)
        print("  Press 'Q' to quit")
        print("  Move mouse to top-left corner for emergency stop")
        print()
        
        # Open webcam
        cap = cv2.VideoCapture(REALTIME["camera_id"])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, REALTIME["camera_width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REALTIME["camera_height"])
        
        if not cap.isOpened():
            print("ERROR: Cannot open webcam!")
            return
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Mirror the frame
                frame = cv2.flip(frame, 1)
                
                self.fps_counter.tick()
                
                # Detect hand
                detection = self.hand_detector.detect(frame)
                
                gesture_name = None
                confidence = 0.0
                action_type = "none"
                
                if detection is not None:
                    # Draw landmarks
                    if self.show_debug:
                        self.hand_detector.draw_landmarks(frame, detection['landmarks'])
                        self.hand_detector.draw_bbox(frame, detection['bbox'])
                    
                    # Extract and preprocess ROI
                    roi = self.hand_detector.extract_roi(frame, detection['bbox'])
                    
                    if roi is not None and roi.size > 0:
                        # Preprocess
                        tensor = self.preprocess_roi(roi)
                        
                        # Predict
                        gesture_name, confidence = self.predict(tensor)
                        action_type = ACTION_MAP.get(gesture_name, "neutral")
                        
                        # Debounce
                        should_trigger, stable_gesture = self.gesture_debouncer.process(
                            gesture_name, confidence, action_type
                        )
                        
                        # Execute action
                        if should_trigger and self.cursor_controller:
                            # Smooth cursor position
                            raw_x, raw_y = detection['index_tip']
                            smooth_x, smooth_y = self.cursor_smoother.smooth(raw_x, raw_y)
                            
                            self.cursor_controller.execute_action(
                                action_type, smooth_x, smooth_y
                            )
                else:
                    # No hand detected — reset smoothers
                    self.cursor_smoother.reset()
                
                # Draw debug UI
                if self.show_debug:
                    frame = self.draw_debug_ui(
                        frame,
                        gesture_name, confidence, action_type,
                        self.fps_counter.fps,
                        self.fps_counter.latency_ms,
                        detection is not None,
                    )
                    
                    cv2.imshow("Gesture Control", frame)
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    break
        
        except KeyboardInterrupt:
            print("\n  Interrupted by user.")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            self.hand_detector.close()
            if self.cursor_controller:
                self.cursor_controller.cleanup()
            print("\n  System shutdown complete.")


def main():
    parser = argparse.ArgumentParser(description="Real-time gesture control")
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'vit'],
                        help='Model type: cnn or vit')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--no-cursor', action='store_true',
                        help='Disable cursor control (debug mode)')
    parser.add_argument('--no-debug', action='store_true',
                        help='Hide debug visualization window')
    
    args = parser.parse_args()
    
    system = GestureControlSystem(
        model_type=args.model,
        checkpoint_path=args.checkpoint,
        enable_cursor=not args.no_cursor,
        show_debug=not args.no_debug,
    )
    
    system.run()


if __name__ == "__main__":
    main()
