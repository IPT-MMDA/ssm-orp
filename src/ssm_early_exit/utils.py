import os
import torch
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Union

def calculate_entropy(logits: torch.Tensor) -> torch.Tensor:
    # розраховує ентропію Шеннона для батчу логітів: H(p) = - sum(p_i * log(p_i))
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
    return entropy


def calculate_confidence_interval(data: Union[List[float], np.ndarray], confidence: float = 0.95) -> Tuple[float, float, float]:
    # розраховує середнє значення та довірчий інтервал
    a = 1.0 * np.array(data)
    n = len(a)
    m = np.mean(a)
    
    if n < 2 or np.std(a) == 0:
        return m, m, m
        
    se = stats.sem(a) 
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


def plot_pareto_curve(results: Dict[float, Dict[str, Any]], save_dir: str = "results"):
    # будує криву Парето
    os.makedirs(save_dir, exist_ok=True)
    
    thresholds = sorted(results.keys())
    latencies = [results[t]["latency_ms"] for t in thresholds]
    accuracies = [results[t]["accuracy"] for t in thresholds]
    
    plt.figure(figsize=(10, 6))
    plt.plot(latencies, accuracies, marker='o', linestyle='-', color='#1f77b4', markersize=8, linewidth=2)
    
    grouped_labels = {}
    for i, thr in enumerate(thresholds):
        coord_key = (round(latencies[i], 1), round(accuracies[i], 3))
        
        if coord_key not in grouped_labels:
            grouped_labels[coord_key] = {"lat": latencies[i], "acc": accuracies[i], "thrs": []}
        grouped_labels[coord_key]["thrs"].append(thr)
        
    for coord_key, data in grouped_labels.items():
        thrs = data["thrs"]
        if len(thrs) > 1:
            label = f"Thr: {min(thrs):.1f} - {max(thrs):.1f}"
        else:
            label = f"Thr: {thrs[0]:.2f}"

        plt.annotate(
            label, 
            (data["lat"], data["acc"]),
            textcoords="offset points", 
            xytext=(0, 12),
            ha='center',
            fontsize=9,
            fontweight='bold',
            color='#333333',
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", alpha=0.9)
        )
        
    plt.title('Pareto Curve: Average Inference Latency vs. Accuracy', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Average Batch Latency (ms) ↓ (Lower is Better)', fontsize=12)
    plt.ylabel('Overall Accuracy ↑ (Higher is Better)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.axvspan(min(latencies), min(latencies) + (max(latencies)-min(latencies))*0.3, 
                ymin=0.7, ymax=1, color='#2ca02c', alpha=0.1, label='Optimal Trade-off Zone')
    
    plt.legend(loc='lower right', frameon=True, shadow=True)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'pareto_curve.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"The Pareto curve saved in {save_path}")


def plot_anomaly_comparison(results: Dict[float, Dict[str, Any]], save_dir: str = "results"):
    """
    будує графік порівняння точності виявлення раптових
    та поступових аномалій в залежності від порогу
    """
    os.makedirs(save_dir, exist_ok=True)
    
    thresholds = sorted(results.keys())
    
    acc_sudden = [results[t]["acc_sudden"] for t in thresholds]
    acc_subtle = [results[t]["acc_subtle"] for t in thresholds]
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(thresholds, acc_sudden, marker='s', linestyle='-', color='#d62728', label='Sudden Anomalies (Spikes)', linewidth=2)
    plt.plot(thresholds, acc_subtle, marker='^', linestyle='-', color='#2ca02c', label='Subtle Anomalies (Drifts)', linewidth=2)
    
    plt.title('Detection Accuracy by Anomaly Type vs. Confidence Threshold', fontsize=14, fontweight='bold', pad=15)
    
    plt.xlabel('Entropy Threshold\n← (Deep Exits / High Confidence Requirement)   ---   (Early Exits / Low Confidence) →', fontsize=11)
    plt.ylabel('Accuracy', fontsize=12)
    
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower left', fontsize=11, frameon=True, shadow=True)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'anomaly_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"The anomaly comparison chart saved to {save_path}")