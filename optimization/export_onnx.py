"""
export_onnx.py — ONNX Model Export
=====================================
Exports PyTorch models to ONNX format and benchmarks
ONNX Runtime inference speed.

Usage:
    python optimization/export_onnx.py --model cnn
    python optimization/export_onnx.py --model vit
"""

import os
import sys
import argparse
import time

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CNN_CONFIG, VIT_CONFIG, PATHS, NUM_CLASSES
from models.cnn_model import build_cnn_model
from models.vit_model import build_vit_model
from training.utils import load_checkpoint


def export_to_onnx(model, input_size, save_path, model_name="model"):
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model (eval mode)
        input_size: Input image size
        save_path: Output .onnx file path
        model_name: Name for logging
    """
    model.eval()
    model = model.cpu()
    
    dummy_input = torch.randn(1, 3, input_size, input_size)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        },
    )
    
    file_size = os.path.getsize(save_path) / (1024 * 1024)
    print(f"  ONNX model exported: {save_path} ({file_size:.2f} MB)")
    
    return save_path


def validate_onnx(onnx_path):
    """Validate exported ONNX model."""
    try:
        import onnx
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print("  ONNX model validation: PASSED ✓")
        return True
    except Exception as e:
        print(f"  ONNX model validation: FAILED ✗ ({e})")
        return False


def benchmark_onnx(onnx_path, input_size, num_runs=100):
    """Benchmark ONNX Runtime inference speed."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  onnxruntime not installed. Skipping ONNX benchmark.")
        return None
    
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    dummy = np.random.randn(1, 3, input_size, input_size).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        session.run(None, {input_name: dummy})
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        session.run(None, {input_name: dummy})
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'fps': 1000.0 / np.mean(times),
    }


def benchmark_pytorch(model, input_size, num_runs=100):
    """Benchmark PyTorch inference speed."""
    model.eval()
    model = model.cpu()
    dummy = torch.randn(1, 3, input_size, input_size)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy)
    
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy)
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'fps': 1000.0 / np.mean(times),
    }


def main():
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument('--model', type=str, required=True, choices=['cnn', 'vit'])
    args = parser.parse_args()
    
    model_type = args.model
    config = CNN_CONFIG if model_type == 'cnn' else VIT_CONFIG
    input_size = config["input_size"]
    
    print("=" * 60)
    print(f"  ONNX EXPORT: {model_type.upper()}")
    print("=" * 60)
    
    # Load model
    checkpoint_path = os.path.join(PATHS["checkpoints"], f"best_{model_type}_model.pth")
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return
    
    if model_type == 'cnn':
        model = build_cnn_model(model_name=config["model_name"],
                                num_classes=NUM_CLASSES, dropout=config["dropout"])
    else:
        model = build_vit_model(model_name=config["model_name"],
                                num_classes=NUM_CLASSES, pretrained=False,
                                dropout=config["dropout"])
    
    model, _ = load_checkpoint(model, checkpoint_path, 'cpu')
    model.eval()
    
    # Export
    onnx_path = os.path.join(PATHS["checkpoints"], f"{model_type}_model.onnx")
    print(f"\n  Exporting to ONNX...")
    export_to_onnx(model, input_size, onnx_path, model_type)
    
    # Validate
    validate_onnx(onnx_path)
    
    # Benchmark comparison
    print(f"\n  Benchmarking PyTorch (CPU)...")
    pytorch_bench = benchmark_pytorch(model, input_size)
    print(f"  PyTorch: {pytorch_bench['mean_ms']:.2f} ms ({pytorch_bench['fps']:.1f} FPS)")
    
    print(f"\n  Benchmarking ONNX Runtime (CPU)...")
    onnx_bench = benchmark_onnx(onnx_path, input_size)
    
    if onnx_bench:
        print(f"  ONNX RT: {onnx_bench['mean_ms']:.2f} ms ({onnx_bench['fps']:.1f} FPS)")
        speedup = pytorch_bench['mean_ms'] / onnx_bench['mean_ms']
        print(f"\n  ONNX speedup vs PyTorch: {speedup:.2f}x")
        
        # Summary table
        print("\n" + "-" * 60)
        print(f"  {'Metric':<20s} {'PyTorch':>15s} {'ONNX Runtime':>15s}")
        print("-" * 60)
        print(f"  {'Latency (ms)':<20s} {pytorch_bench['mean_ms']:>15.2f} {onnx_bench['mean_ms']:>15.2f}")
        print(f"  {'FPS':<20s} {pytorch_bench['fps']:>15.1f} {onnx_bench['fps']:>15.1f}")
        print("-" * 60)
    
    # Verify output correctness
    print("\n  Verifying output correctness...")
    dummy_np = np.random.randn(1, 3, input_size, input_size).astype(np.float32)
    dummy_torch = torch.from_numpy(dummy_np)
    
    with torch.no_grad():
        pytorch_out = model(dummy_torch).numpy()
    
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        onnx_out = session.run(None, {input_name: dummy_np})[0]
        
        max_diff = np.abs(pytorch_out - onnx_out).max()
        print(f"  Max output difference: {max_diff:.8f}")
        if max_diff < 1e-4:
            print("  Output verification: PASSED ✓")
        else:
            print("  Output verification: WARNING — significant difference")
    except ImportError:
        print("  Skipping output verification (onnxruntime not installed)")


if __name__ == "__main__":
    main()
