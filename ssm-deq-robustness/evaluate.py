import torch
import os
import numpy as np
from models.deq_modules import MDEQSmall
from models.resnet_modules import BalancedResNet
from utils.data_loader import get_data_loaders
from utils.engine import test_robustness
from attacks.adversarial import pgd_attack
from attacks.corruption import corrupt_data


def load_best_model(model, name):

    subfolder = "mdeq" if "MDEQ" in name else "sresnet"
    path = f"checkpoints/{subfolder}/{name}_best.pth"

    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        model.eval()
        print(f"[УСПІХ] Завантажено {name} з {path}")
    else:
        print(f"[ПОМИЛКА] Файл не знайдено: {path}")
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, test_loader = get_data_loaders(batch_size=100)
    resnet = load_best_model(BalancedResNet().to(device), "ResNet")
    mdeq = load_best_model(MDEQSmall().to(device), "MDEQ")


    scenarios = [
        {"name": "clean", "attack": None, "eps": 0.0, "corr": None, "sev": 1},
        {"name": "gaussian_noise", "attack": None, "eps": 0.0, "corr": "gaussian_noise", "sev": 3},
        {"name": "defocus_blur", "attack": None, "eps": 0.0, "corr": "defocus_blur", "sev": 3},
        {"name": "pgd", "attack": pgd_attack, "eps": 0.01, "corr": None, "sev": 1},
        {"name": "pgd", "attack": pgd_attack, "eps": 0.03, "corr": None, "sev": 1},
    ]

    print("\n" + "=" * 80)
    print(f"{'Scenario':<20} | {'Model':<8} | {'Acc':<8} | {'Iters':<8} | {'Residual'}")
    print("=" * 80)

    for sc in scenarios:
        for model, m_name in [(resnet, "ResNet"), (mdeq, "MDEQ")]:

            acc, iters, res = test_robustness(
                model, test_loader, device, m_name,
                attack_func=sc['attack'],
                eps=sc['eps'],
                corruption_func=corrupt_data if sc['corr'] else None,
                # 'defocus_blur'
                corruption_type=sc['corr'] if sc['corr'] else 'gaussian_noise',
                severity=sc['sev']
            )

            iter_str = f"{iters:.2f}" if not np.isnan(iters) else "nan"
            res_str = f"{res:.6f}" if not np.isnan(res) else "nan"

            print(f"{sc['name']:<20} | {m_name:<8} | {acc:.4f} | {iter_str:<8} | {res_str}")


if __name__ == "__main__":
    main()