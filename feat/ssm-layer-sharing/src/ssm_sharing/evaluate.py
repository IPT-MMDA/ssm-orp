# test: verify training convergence on small batch (перевірив, що вчиться)
import torch
import torch.nn as nn

import scipy.stats as stats
import numpy as np


class Perturbator:
    @staticmethod
    def apply_nothing(x: torch.Tensor) -> torch.Tensor:
        """Повертає те, що дали."""
        return x

    @staticmethod
    def apply_masking(x: torch.Tensor, mask_ratio: float = 0.2) -> torch.Tensor:
        """Занулює випадкові 20% сигналу."""
        mask = torch.rand_like(x) > mask_ratio
        return x * mask

    @staticmethod
    def apply_gaussian_noise(x: torch.Tensor, std: float = 0.1) -> torch.Tensor:
        """Додає гауссівський шум N(0, std^2)."""
        noise = torch.randn_like(x) * std
        return x + noise


class Evaluator:
    @staticmethod
    def get_confidence_interval(data: list, confidence_level: float = 0.95):
        """Розраховує 95% CI для середнього значення."""
        n = len(data)
        m = np.mean(data)
        se = stats.sem(data) # sem = s / sqrt(n)
        h = se * stats.t.ppf((1 + confidence_level) / 2, n-1)
        return m, [m-h, m+h]
    
    @staticmethod
    def run_stress_test(model, dataloader, device, perturbation_fn, n_runs=10):
        model.eval()
        results = []
        
        for i in range(n_runs):
            torch.manual_seed(42 + i)
            
            correct = 0
            total = 0
            with torch.no_grad():
                for X, y in dataloader:
                    X, y = X.to(device), y.to(device)
                    
                    X_noisy = perturbation_fn(X)
                    
                    logits = model(X_noisy)
                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == y).sum().item()
                    total += y.size(0)
            
            acc = correct / total
            results.append(acc)
        
        return Evaluator.get_confidence_interval(results)
