import torch
import time
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any
from torch.utils.data import DataLoader
import torch.nn as nn

from ssm_early_exit.utils import calculate_confidence_interval

@torch.no_grad()
def evaluate_early_exits(
    model: nn.Module, 
    test_loader: DataLoader, 
    thresholds: List[float], 
    device: str = "cpu"
) -> Dict[float, Dict[str, Any]]:
    
    #проганяє інференс по тестовому датасету для масиву різних порогів ентропії (thresholds)
    #також вимірює точність (загальну та покласову) та latency з довірчими інтервалами
    print("\n" + "="*50)
    print("Launch the evaluation Early Exiting (Latency vs Accuracy with CI)")
    print("="*50)

    model.to(device)
    model.eval()
    
    is_cuda = "cuda" in device
    results = {}

    for thr in thresholds:
        total_samples = 0
        correct_total = 0
        
        class_correct = {0: 0, 1: 0, 2: 0}
        class_totals = {0: 0, 1: 0, 2: 0}
        exit_counts = {}
        
        # списки для збереження метрик кожного батчу
        batch_latencies = []
        batch_accuracies = []
        
        if is_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
        dummy_x, _ = next(iter(test_loader))
        model.forward_inference(dummy_x.to(device), threshold=thr)
        if is_cuda:
            torch.cuda.synchronize()

        loop = tqdm(test_loader, desc=f"Threshold: {thr:.2f}")
        for X_batch, y_batch in loop:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            batch_size = y_batch.size(0)
            
            # вимірювання часу 
            if is_cuda:
                torch.cuda.synchronize()
                start_event.record()
                
                logits, exit_head = model.forward_inference(X_batch, threshold=thr)
                
                end_event.record()
                torch.cuda.synchronize()
                batch_latency = start_event.elapsed_time(end_event)
            else:
                start_time = time.perf_counter()
                logits, exit_head = model.forward_inference(X_batch, threshold=thr)
                batch_latency = (time.perf_counter() - start_time) * 1000.0 

            # зберігаю затримку батчу
            batch_latencies.append(batch_latency)
            
            exit_counts[exit_head] = exit_counts.get(exit_head, 0) + batch_size
            
            # точность для батчу
            preds = torch.argmax(logits, dim=1)
            batch_correct = (preds == y_batch).sum().item()
            
            # зберігаю точність конкретного батчу ( від 0 до 1)
            batch_accuracies.append(batch_correct / batch_size)
            
            correct_total += batch_correct
            total_samples += batch_size
            
            for i in range(batch_size):
                true_label = y_batch[i].item()
                pred_label = preds[i].item()
                class_totals[true_label] += 1
                if true_label == pred_label:
                    class_correct[true_label] += 1

        # розрахунок довірчих інтервалів (95% CI)
        mean_lat, lower_lat, upper_lat = calculate_confidence_interval(batch_latencies)
        mean_acc, lower_acc, upper_acc = calculate_confidence_interval(batch_accuracies)
        
        # похибка (± margin)
        lat_margin = upper_lat - mean_lat
        acc_margin = upper_acc - mean_acc

        acc_sudden = class_correct[1] / class_totals[1] if class_totals[1] > 0 else 0
        acc_subtle = class_correct[2] / class_totals[2] if class_totals[2] > 0 else 0
        
        results[thr] = {
            "accuracy": mean_acc,
            "accuracy_ci_margin": acc_margin,
            "acc_sudden": acc_sudden,
            "acc_subtle": acc_subtle,
            "latency_ms": mean_lat,
            "latency_ci_margin": lat_margin,
            "exit_distribution": {k: v / total_samples for k, v in exit_counts.items()}
        }
        
        most_common_exit = max(exit_counts, key=exit_counts.get)
        print(f"Thr: {thr:.2f} | Acc: {mean_acc:.3f}±{acc_margin:.3f} | Latency: {mean_lat:.2f}±{lat_margin:.2f} ms | Main: {most_common_exit}")

    return results