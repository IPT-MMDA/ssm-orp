import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import BaselineSSM, RegularizedSSM

def generate_adding_task(batch_size, seq_len):
    """Standard benchmark: adds two numbers flagged by a mask."""
    x_val = torch.rand(batch_size, seq_len, 1)
    mask = torch.zeros(batch_size, seq_len, 1)
    for i in range(batch_size):
        indices = np.random.choice(seq_len, 2, replace=False)
        mask[i, indices, 0] = 1.0
    x = torch.cat([x_val, mask], dim=-1) # Input dim = 2
    y = (x_val * mask).sum(dim=1) 
    return x, y

def run_robustness_trial(model, seq_len=50, noise_std=0.5):
    """Performs a single noise injection test."""
    model.eval()
    with torch.no_grad():
        clean_x, _ = generate_adding_task(batch_size=1, seq_len=seq_len)
        
        # 1. Clean Pass
        _, clean_h = model(clean_x)
        
        # 2. Noisy Pass (High-frequency white noise)
        noise = torch.randn_like(clean_x) * noise_std
        noisy_x = clean_x + noise
        _, noisy_h = model(noisy_x)
        
        # 3. Calculate Delta H (Stability Metric)
        delta_h = torch.norm(noisy_h - clean_h, p=2, dim=-1).mean().item()
    return delta_h

def evaluate_stability(model_path, model_type="baseline", trials=10):
    """Runs multiple trials to get Mean and Std Dev."""
    if model_type == "baseline":
        model = BaselineSSM(input_dim=2, hidden_dim=64, output_dim=1)
    else:
        model = RegularizedSSM(input_dim=2, hidden_dim=64, output_dim=1)
    
    # Load weights (handle cases where file doesn't exist yet)
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Run train.py first!")
        return None

    model.load_state_dict(torch.load(model_path))
    
    results = []
    for _ in range(trials):
        results.append(run_robustness_trial(model))
    
    return np.mean(results), np.std(results)

def plot_noise_sensitivity(base_model, reg_model):
    noise_levels = np.linspace(0, 1.0, 10)
    base_variances = []
    reg_variances = []

    print("Generating stability plot...")
    for sigma in noise_levels:
        # Run multiple trials for each noise level to get an average
        b_v = np.mean([run_robustness_trial(base_model, noise_std=sigma) for _ in range(5)])
        r_v = np.mean([run_robustness_trial(reg_model, noise_std=sigma) for _ in range(5)])
        base_variances.append(b_v)
        reg_variances.append(r_v)

    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, base_variances, label='Baseline (Unconstrained)', marker='o', color='red')
    plt.plot(noise_levels, reg_variances, label='Regularized (Lipschitz-Bounded)', marker='s', color='blue')
    
    plt.title('SSM Hidden State Stability: Baseline vs. Regularized')
    plt.xlabel('Input Noise Intensity (Standard Deviation)')
    plt.ylabel('Hidden State Deviation (Δh)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig('stability_plot.png')
    print("Plot saved as stability_plot.png")

if __name__ == "__main__":
    print("--- Starting Scientific Evaluation ---")
    
    # Initialize models and load the saved weights
    base_model = BaselineSSM(input_dim=2, hidden_dim=64, output_dim=1)
    reg_model = RegularizedSSM(input_dim=2, hidden_dim=64, output_dim=1)
    
    base_model.load_state_dict(torch.load("baseline.pth"))
    reg_model.load_state_dict(torch.load("regularized.pth"))

    # Run the Table evaluation (Mean ± Std Dev)
    print("\nCalculating Confidence Intervals...")
    base_res = evaluate_stability("baseline.pth", "baseline")
    reg_res = evaluate_stability("regularized.pth", "regularized")
    
    if base_res and reg_res:
        print("\n" + "="*40)
        print(f"{'Model':<15} | {'Hidden Δh (Mean ± Std)':<25}")
        print("-" * 40)
        print(f"{'Baseline':<15} | {base_res[0]:.4f} ± {base_res[1]:.4f}")
        print(f"{'Regularized':<15} | {reg_res[0]:.4f} ± {reg_res[1]:.4f}")
        print("="*40)

    # Plot noise sensitivity
    plot_noise_sensitivity(base_model, reg_model)