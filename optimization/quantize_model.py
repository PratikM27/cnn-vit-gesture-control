"""
quantize_model.py — Post-Training Quantization
=================================================
Applies dynamic INT8 quantization to reduce model size
and improve CPU inference speed.

Usage:
    python optimization/quantize_model.py --model cnn
    python optimization/quantize_model.py --model vit
"""

import os
import sys
import argparse
import time

import numpy as np
import torch
import torch.quantization

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CNN_CONFIG, VIT_CONFIG, PATHS, NUM_CLASSES
from models.cnn_model import build_cnn_model
from models.vit_model import build_vit_model
from training.utils import load_checkpoint, measure_model_size


def quantize_dynamic(model, model_type):
    """
    Apply dynamic quantization to model.
    
    Quantizes Linear and Conv2d layers to INT8.
    
    Args:
        model: PyTorch model
        model_type: 'cnn' or 'vit'
    
    Returns:
        quantized_model: Quantized model
    """
    model.eval()
    model_cpu = model.cpu()
    
    # Dynamic quantization (works on CPU)
    quantized = torch.quantization.quantize_dynamic(
        model_cpu,
        {torch.nn.Linear, torch.nn.Conv2d},
        dtype=torch.qint8,
    )
    
    return quantized


def benchmark_model(model, input_size, num_runs=100, device='cpu'):
    """Benchmark inference speed."""
    model.eval()
    model = model.to(device)
    dummy = torch.randn(1, 3, input_size, input_size).to(device)
    
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
    parser = argparse.ArgumentParser(description="Quantize model")
    parser.add_argument('--model', type=str, required=True, choices=['cnn', 'vit'])
    args = parser.parse_args()
    
    model_type = args.model
    config = CNN_CONFIG if model_type == 'cnn' else VIT_CONFIG
    input_size = config["input_size"]
    
    print("=" * 60)
    print(f"  MODEL QUANTIZATION: {model_type.upper()}")
    print("=" * 60)
    
    # Load original model
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
    
    # Measure original
    orig_size = measure_model_size(model)
    print(f"\n  Original model size: {orig_size:.2f} MB")
    
    print("  Benchmarking original model...")
    orig_bench = benchmark_model(model, input_size, device='cpu')
    print(f"  Original latency: {orig_bench['mean_ms']:.2f} ms ({orig_bench['fps']:.1f} FPS)")
    
    # Quantize
    print("\n  Applying INT8 dynamic quantization...")
    quantized_model = quantize_dynamic(model, model_type)
    
    # Measure quantized
    quant_size = measure_model_size(quantized_model)
    print(f"  Quantized model size: {quant_size:.2f} MB")
    print(f"  Size reduction: {(1 - quant_size/orig_size)*100:.1f}%")
    
    print("\n  Benchmarking quantized model...")
    quant_bench = benchmark_model(quantized_model, input_size, device='cpu')
    print(f"  Quantized latency: {quant_bench['mean_ms']:.2f} ms ({quant_bench['fps']:.1f} FPS)")
    
    speedup = orig_bench['mean_ms'] / quant_bench['mean_ms']
    print(f"  Speedup: {speedup:.2f}x")
    
    # Verify output correctness
    dummy = torch.randn(1, 3, input_size, input_size)
    with torch.no_grad():
        orig_out = model(dummy)
        quant_out = quantized_model(dummy)
    
    diff = torch.abs(orig_out - quant_out).max().item()
    print(f"\n  Max output difference: {diff:.6f}")
    
    # Save quantized model
    save_path = os.path.join(PATHS["checkpoints"], f"quantized_{model_type}_model.pth")
    torch.save(quantized_model.state_dict(), save_path)
    print(f"\n  Quantized model saved to: {save_path}")
    
    # Summary
    print("\n" + "-" * 60)
    print(f"  {'Metric':<25s} {'Original':>12s} {'Quantized':>12s}")
    print("-" * 60)
    print(f"  {'Size (MB)':<25s} {orig_size:>12.2f} {quant_size:>12.2f}")
    print(f"  {'Latency (ms)':<25s} {orig_bench['mean_ms']:>12.2f} {quant_bench['mean_ms']:>12.2f}")
    print(f"  {'FPS':<25s} {orig_bench['fps']:>12.1f} {quant_bench['fps']:>12.1f}")
    print("-" * 60)


if __name__ == "__main__":
    main()
