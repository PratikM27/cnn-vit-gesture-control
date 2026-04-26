"""
compare_models.py — CNN vs ViT Comparison
============================================
Generates side-by-side comparison tables, bar charts,
and radar plots from evaluation results.

Usage:
    python analysis/compare_models.py
"""

import os
import sys
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PATHS, GESTURE_LABELS


def load_results(model_type):
    """Load evaluation results for a model."""
    eval_path = os.path.join(PATHS["results"], f"{model_type}_eval_results.json")
    metrics_path = os.path.join(PATHS["results"], f"{model_type}_metrics.json")
    
    results = {}
    
    if os.path.exists(eval_path):
        with open(eval_path, 'r') as f:
            results['eval'] = json.load(f)
    else:
        print(f"  WARNING: Eval results not found: {eval_path}")
        results['eval'] = {}
    
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            results['training'] = json.load(f)
    else:
        print(f"  WARNING: Training metrics not found: {metrics_path}")
        results['training'] = {}
    
    return results


def create_comparison_table(cnn_results, vit_results):
    """Create and print comparison table."""
    cnn_eval = cnn_results.get('eval', {})
    vit_eval = vit_results.get('eval', {})
    cnn_train = cnn_results.get('training', {})
    vit_train = vit_results.get('training', {})
    
    print("\n" + "=" * 75)
    print("  CNN vs ViT — COMPARISON TABLE")
    print("=" * 75)
    
    metrics = [
        ("Test Accuracy (%)", 
         cnn_eval.get('accuracy', 0), vit_eval.get('accuracy', 0)),
        ("Precision (macro %)", 
         cnn_eval.get('precision_macro', 0), vit_eval.get('precision_macro', 0)),
        ("Recall (macro %)", 
         cnn_eval.get('recall_macro', 0), vit_eval.get('recall_macro', 0)),
        ("F1-Score (macro %)", 
         cnn_eval.get('f1_macro', 0), vit_eval.get('f1_macro', 0)),
        ("Avg Latency (ms)", 
         cnn_eval.get('latency', {}).get('mean_ms', 0),
         vit_eval.get('latency', {}).get('mean_ms', 0)),
        ("FPS", 
         cnn_eval.get('latency', {}).get('fps', 0),
         vit_eval.get('latency', {}).get('fps', 0)),
        ("Model Size (MB)", 
         cnn_eval.get('model_size_mb', cnn_train.get('model_size_mb', 0)),
         vit_eval.get('model_size_mb', vit_train.get('model_size_mb', 0))),
        ("Parameters (M)", 
         cnn_eval.get('total_params', cnn_train.get('total_params', 0)) / 1e6,
         vit_eval.get('total_params', vit_train.get('total_params', 0)) / 1e6),
        ("Training Time (s)", 
         cnn_train.get('training_time_seconds', 0),
         vit_train.get('training_time_seconds', 0)),
    ]
    
    print(f"\n  {'Metric':<25s} {'CNN':>12s} {'ViT':>12s} {'Winner':>10s}")
    print("  " + "-" * 62)
    
    # For latency and size, lower is better
    lower_is_better = {"Avg Latency (ms)", "Model Size (MB)", "Parameters (M)", "Training Time (s)"}
    
    comparison_data = []
    for name, cnn_val, vit_val in metrics:
        if name in lower_is_better:
            winner = "CNN" if cnn_val <= vit_val else "ViT"
        else:
            winner = "CNN" if cnn_val >= vit_val else "ViT"
        
        if isinstance(cnn_val, float):
            print(f"  {name:<25s} {cnn_val:>12.2f} {vit_val:>12.2f} {winner:>10s}")
        else:
            print(f"  {name:<25s} {cnn_val:>12} {vit_val:>12} {winner:>10s}")
        
        comparison_data.append((name, cnn_val, vit_val, winner))
    
    return comparison_data


def plot_bar_comparison(cnn_results, vit_results, save_dir):
    """Create grouped bar charts for key metrics."""
    cnn_eval = cnn_results.get('eval', {})
    vit_eval = vit_results.get('eval', {})
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Accuracy metrics bar chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Accuracy, Precision, Recall, F1
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    cnn_vals = [
        cnn_eval.get('accuracy', 0),
        cnn_eval.get('precision_macro', 0),
        cnn_eval.get('recall_macro', 0),
        cnn_eval.get('f1_macro', 0),
    ]
    vit_vals = [
        vit_eval.get('accuracy', 0),
        vit_eval.get('precision_macro', 0),
        vit_eval.get('recall_macro', 0),
        vit_eval.get('f1_macro', 0),
    ]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    axes[0].bar(x - width/2, cnn_vals, width, label='CNN', color='#2196F3', alpha=0.85)
    axes[0].bar(x + width/2, vit_vals, width, label='ViT', color='#FF5722', alpha=0.85)
    axes[0].set_ylabel('Percentage (%)')
    axes[0].set_title('Classification Metrics', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics_names)
    axes[0].legend()
    axes[0].set_ylim(0, 105)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (c, v) in enumerate(zip(cnn_vals, vit_vals)):
        axes[0].text(i - width/2, c + 1, f'{c:.1f}', ha='center', va='bottom', fontsize=8)
        axes[0].text(i + width/2, v + 1, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
    
    # Speed comparison
    speed_metrics = ['FPS', 'Latency (ms)']
    cnn_speed = [
        cnn_eval.get('latency', {}).get('fps', 0),
        cnn_eval.get('latency', {}).get('mean_ms', 0),
    ]
    vit_speed = [
        vit_eval.get('latency', {}).get('fps', 0),
        vit_eval.get('latency', {}).get('mean_ms', 0),
    ]
    
    x2 = np.arange(len(speed_metrics))
    axes[1].bar(x2 - width/2, cnn_speed, width, label='CNN', color='#2196F3', alpha=0.85)
    axes[1].bar(x2 + width/2, vit_speed, width, label='ViT', color='#FF5722', alpha=0.85)
    axes[1].set_title('Speed Comparison', fontweight='bold')
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(speed_metrics)
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    for i, (c, v) in enumerate(zip(cnn_speed, vit_speed)):
        axes[1].text(i - width/2, c + 0.5, f'{c:.1f}', ha='center', va='bottom', fontsize=8)
        axes[1].text(i + width/2, v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
    
    # Model size comparison
    cnn_train = cnn_results.get('training', {})
    vit_train = vit_results.get('training', {})
    
    size_metrics = ['Size (MB)', 'Params (M)']
    cnn_size = [
        cnn_eval.get('model_size_mb', cnn_train.get('model_size_mb', 0)),
        cnn_eval.get('total_params', cnn_train.get('total_params', 0)) / 1e6,
    ]
    vit_size = [
        vit_eval.get('model_size_mb', vit_train.get('model_size_mb', 0)),
        vit_eval.get('total_params', vit_train.get('total_params', 0)) / 1e6,
    ]
    
    x3 = np.arange(len(size_metrics))
    axes[2].bar(x3 - width/2, cnn_size, width, label='CNN', color='#2196F3', alpha=0.85)
    axes[2].bar(x3 + width/2, vit_size, width, label='ViT', color='#FF5722', alpha=0.85)
    axes[2].set_title('Model Complexity', fontweight='bold')
    axes[2].set_xticks(x3)
    axes[2].set_xticklabels(size_metrics)
    axes[2].legend()
    axes[2].grid(axis='y', alpha=0.3)
    
    for i, (c, v) in enumerate(zip(cnn_size, vit_size)):
        axes[2].text(i - width/2, c + 0.2, f'{c:.1f}', ha='center', va='bottom', fontsize=8)
        axes[2].text(i + width/2, v + 0.2, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('CNN vs Vision Transformer — Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'comparison_bar_charts.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Bar charts saved to: {save_path}")


def plot_radar_chart(cnn_results, vit_results, save_dir):
    """Create radar/spider chart for multi-metric comparison."""
    cnn_eval = cnn_results.get('eval', {})
    vit_eval = vit_results.get('eval', {})
    cnn_train = cnn_results.get('training', {})
    vit_train = vit_results.get('training', {})
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Normalize metrics to 0-100 scale
    categories = ['Accuracy', 'F1-Score', 'FPS\n(normalized)', 'Size\n(inverted)', 'Latency\n(inverted)']
    
    max_fps = max(
        cnn_eval.get('latency', {}).get('fps', 1),
        vit_eval.get('latency', {}).get('fps', 1), 1
    )
    max_size = max(
        cnn_eval.get('model_size_mb', cnn_train.get('model_size_mb', 1)),
        vit_eval.get('model_size_mb', vit_train.get('model_size_mb', 1)), 1
    )
    max_latency = max(
        cnn_eval.get('latency', {}).get('mean_ms', 1),
        vit_eval.get('latency', {}).get('mean_ms', 1), 1
    )
    
    cnn_values = [
        cnn_eval.get('accuracy', 0),
        cnn_eval.get('f1_macro', 0),
        (cnn_eval.get('latency', {}).get('fps', 0) / max_fps) * 100,
        (1 - cnn_eval.get('model_size_mb', cnn_train.get('model_size_mb', 0)) / max_size) * 100,
        (1 - cnn_eval.get('latency', {}).get('mean_ms', 0) / max_latency) * 100,
    ]
    vit_values = [
        vit_eval.get('accuracy', 0),
        vit_eval.get('f1_macro', 0),
        (vit_eval.get('latency', {}).get('fps', 0) / max_fps) * 100,
        (1 - vit_eval.get('model_size_mb', vit_train.get('model_size_mb', 0)) / max_size) * 100,
        (1 - vit_eval.get('latency', {}).get('mean_ms', 0) / max_latency) * 100,
    ]
    
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the polygon
    cnn_values += cnn_values[:1]
    vit_values += vit_values[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    ax.plot(angles, cnn_values, 'o-', linewidth=2, label='CNN', color='#2196F3')
    ax.fill(angles, cnn_values, alpha=0.15, color='#2196F3')
    
    ax.plot(angles, vit_values, 'o-', linewidth=2, label='ViT', color='#FF5722')
    ax.fill(angles, vit_values, alpha=0.15, color='#FF5722')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=12)
    ax.set_title('CNN vs ViT — Multi-Metric Radar', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True)
    
    save_path = os.path.join(save_dir, 'comparison_radar.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Radar chart saved to: {save_path}")


def plot_per_class_comparison(cnn_results, vit_results, save_dir):
    """Plot per-class F1-score comparison."""
    cnn_pc = cnn_results.get('eval', {}).get('per_class', {})
    vit_pc = vit_results.get('eval', {}).get('per_class', {})
    
    if not cnn_pc or not vit_pc:
        print("  Skipping per-class plot (no per-class data)")
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    classes = sorted(cnn_pc.keys())
    cnn_f1 = [cnn_pc[c]['f1'] for c in classes]
    vit_f1 = [vit_pc.get(c, {}).get('f1', 0) for c in classes]
    
    display_names = [GESTURE_LABELS.get(c, c) for c in classes]
    
    x = np.arange(len(classes))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, cnn_f1, width, label='CNN', color='#2196F3', alpha=0.85)
    bars2 = ax.bar(x + width/2, vit_f1, width, label='ViT', color='#FF5722', alpha=0.85)
    
    ax.set_ylabel('F1-Score (%)', fontsize=12)
    ax.set_title('Per-Class F1-Score — CNN vs ViT', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=30, ha='right')
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    
    # Value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'per_class_f1_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Per-class F1 chart saved to: {save_path}")


def generate_comparison_report(cnn_results, vit_results, save_dir):
    """Generate markdown comparison report."""
    os.makedirs(save_dir, exist_ok=True)
    
    cnn_eval = cnn_results.get('eval', {})
    vit_eval = vit_results.get('eval', {})
    cnn_train = cnn_results.get('training', {})
    vit_train = vit_results.get('training', {})
    
    report = """# CNN vs Vision Transformer — Comparison Report

## Overview

This report presents a quantitative comparison of two deep learning models
for hand gesture recognition: a custom CNN (baseline) and a Vision Transformer
(ViT, proposed system).

## Main Results

| Metric | CNN | ViT |
|--------|-----|-----|
"""
    
    metrics = [
        ("Test Accuracy (%)", cnn_eval.get('accuracy', 0), vit_eval.get('accuracy', 0)),
        ("Precision (macro %)", cnn_eval.get('precision_macro', 0), vit_eval.get('precision_macro', 0)),
        ("Recall (macro %)", cnn_eval.get('recall_macro', 0), vit_eval.get('recall_macro', 0)),
        ("F1-Score (macro %)", cnn_eval.get('f1_macro', 0), vit_eval.get('f1_macro', 0)),
        ("FPS", cnn_eval.get('latency', {}).get('fps', 0), vit_eval.get('latency', {}).get('fps', 0)),
        ("Latency (ms)", cnn_eval.get('latency', {}).get('mean_ms', 0), vit_eval.get('latency', {}).get('mean_ms', 0)),
        ("Model Size (MB)", cnn_eval.get('model_size_mb', 0), vit_eval.get('model_size_mb', 0)),
    ]
    
    for name, cnn_v, vit_v in metrics:
        report += f"| {name} | {cnn_v:.2f} | {vit_v:.2f} |\n"
    
    report += """
## Analysis

### Accuracy
"""
    acc_diff = vit_eval.get('accuracy', 0) - cnn_eval.get('accuracy', 0)
    if acc_diff > 0:
        report += f"ViT outperforms CNN by {acc_diff:.2f}% in test accuracy.\n"
    else:
        report += f"CNN outperforms ViT by {-acc_diff:.2f}% in test accuracy.\n"
    
    report += """
### Speed
"""
    cnn_fps = cnn_eval.get('latency', {}).get('fps', 0)
    vit_fps = vit_eval.get('latency', {}).get('fps', 0)
    if cnn_fps > 0 and vit_fps > 0:
        report += f"CNN runs at {cnn_fps:.1f} FPS vs ViT at {vit_fps:.1f} FPS.\n"
        report += f"CNN is {cnn_fps/vit_fps:.1f}x faster than ViT.\n"
    
    report += """
### Real-Time Feasibility

For real-time cursor control, the system requires ≥15 FPS.
"""
    if cnn_fps >= 15:
        report += "- CNN: ✓ Suitable for real-time use\n"
    else:
        report += "- CNN: ✗ Below real-time threshold\n"
    if vit_fps >= 15:
        report += "- ViT: ✓ Suitable for real-time use\n"
    else:
        report += "- ViT: ✗ Below real-time threshold (may need optimization)\n"
    
    report += """
### Conclusion

"""
    report += (
        f"The CNN model achieves {cnn_eval.get('accuracy', 0):.2f}% accuracy "
        f"with {cnn_fps:.1f} FPS, making it highly suitable for real-time deployment. "
        f"The ViT model achieves {vit_eval.get('accuracy', 0):.2f}% accuracy "
        f"but at {vit_fps:.1f} FPS. "
    )
    
    if acc_diff > 0 and vit_fps >= 15:
        report += "ViT offers the best balance of accuracy and speed for this task."
    elif acc_diff > 0:
        report += "While ViT is more accurate, CNN is recommended for deployment due to superior real-time performance."
    else:
        report += "CNN is the clear winner for this task, offering both higher accuracy and faster inference."
    
    save_path = os.path.join(save_dir, 'comparison_report.md')
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n  Comparison report saved to: {save_path}")


def main():
    print("=" * 60)
    print("  CNN vs ViT — MODEL COMPARISON")
    print("=" * 60)
    
    # Load results
    print("\n  Loading results...")
    cnn_results = load_results('cnn')
    vit_results = load_results('vit')
    
    # Print comparison table
    create_comparison_table(cnn_results, vit_results)
    
    # Generate plots
    print("\n  Generating comparison plots...")
    plot_bar_comparison(cnn_results, vit_results, PATHS["results"])
    plot_radar_chart(cnn_results, vit_results, PATHS["results"])
    plot_per_class_comparison(cnn_results, vit_results, PATHS["results"])
    
    # Generate report
    generate_comparison_report(cnn_results, vit_results, PATHS["results"])
    
    print("\n  Done! All comparison outputs saved to:", PATHS["results"])


if __name__ == "__main__":
    main()
