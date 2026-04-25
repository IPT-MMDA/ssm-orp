import torch
import timm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import SSMStudent
from utils import distillation_loss, evaluate_robustness
import matplotlib.pyplot as plt

def print_model_stats(model, name):
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model {name}: {params:.2f}M parameters")

def plot_robustness_results(results_dict):
    """Створення Bar Chart з довірчими інтервалами (Error Bars)."""
    modes = list(results_dict.keys())
    accs = [res[0] for res in results_dict.values()]
    cis = [res[1] for res in results_dict.values()]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(modes, accs, yerr=cis, capsize=10, color='skyblue', edgecolor='black', alpha=0.8)
    
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('SSM Student Robustness Evaluation (95% CI)', fontsize=14)
    plt.ylim(0, max(accs) + 0.05) 
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.002,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('robustness_results.png', dpi=300)
    print("\n[INFO] Графік збережено у файл 'robustness_results.png'")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}\n")

    # 1. Завантаження даних
    transform = transforms.Compose([transforms.ToTensor()])
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    # 2. Ініціалізація моделей
    teacher = timm.create_model('deit_tiny_patch16_224', pretrained=True, num_classes=10).to(device)
    student = SSMStudent().to(device)

    print_model_stats(teacher, "Teacher (ViT)")
    print_model_stats(student, "Student (SSM)")

    # 3. Єдиний цикл оцінки стійкості
    print("\n--- Початок оцінки стійкості (Robustness Evaluation) ---")
    plot_data = {}
    
    # Список тестів: (назва, режим, рівень шуму)
    experiments = [
        ("Clean", "noise", 0.0),
        ("Noise (std=0.2)", "noise", 0.2),
        ("Gaussian Blur", "blur", 0.0),
        ("Occlusion", "occlusion", 0.0)
    ]

    for label, mode, lvl in experiments:
        acc, ci = evaluate_robustness(student, test_loader, device, mode=mode, noise_lvl=lvl)
        print(f"{label:<15}: Accuracy = {acc:.4f} ± {ci:.4f}")
        plot_data[label] = (acc, ci)

    # 4. Генерація фінального графіка
    plot_robustness_results(plot_data)
    print("\n--- Всі завдання виконано успішно ---")

if __name__ == "__main__":
    main()