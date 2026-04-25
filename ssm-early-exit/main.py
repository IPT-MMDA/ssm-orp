import os
import torch

from ssm_early_exit.data import get_dataloaders
from ssm_early_exit.model import DeepSSM
from ssm_early_exit.train import train_pipeline
from ssm_early_exit.evaluate import evaluate_early_exits
from ssm_early_exit.utils import plot_pareto_curve, plot_anomaly_comparison

def main():
    print("Dynamic-Compute SSM with Early Exiting")
    print("="*40)
    
    # 1. налаштування пристрою
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"The device is used: {device.upper()}")

    # 2. генерація синтетичних даних
    print("\nStep 1: Generating synthetic time series...")
    dataloaders = get_dataloaders(
        num_train=3000, 
        num_val=900, 
        num_test=900, 
        seq_len=128, 
        batch_size=32
    )
    
    # 3. ініціалізація моделі
    print("\nStep 2: Initializing the DeepSSM architecture...")
    # L=8 шарів. хеди будуть на шарах: 1 (L/4), 3 (L/2), 5 (3L/4), та 7 (Final)
    model = DeepSSM(input_dim=1, d_model=32, n_layers=8, num_classes=3)
    
    # 4. двохетапне навчання
    # етап 1 (Backbone) та етап 2 (Linear Heads)
    model = train_pipeline(model, dataloaders, device=device)
    
    # 5. евалюація (Early Exiting)
    # задаємо масив порогів ентропії (від суворих вимог до слабких)
    # низький поріг = висока впевненість = модель йде глибше
    # високий поріг = низька впевненість = ранній вихід
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
    
    results = evaluate_early_exits(model, dataloaders['test'], thresholds, device=device)
    
    # 6. побудова та збереження графіків
    print("\nStep 3: Plotting graphs (Pareto Curve and Anomaly Comparison)...")
    save_directory = "results"
    
    plot_pareto_curve(results, save_dir=save_directory)
    plot_anomaly_comparison(results, save_dir=save_directory)
    
    print("="*60)
    print(f"✅ Pipeline successfully completed! All graphs saved to folder './{save_directory}'.")

if __name__ == "__main__":
    main()