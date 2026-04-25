import torch
import time

from src.models.cifar_model import CIFARModel
from src.verification.solver import solve_sdp
from src.utils.extract_weights import extract_weights
from src.data.cifar import get_cifar_loaders
from src.attacks.pgd import pgd_attack


def run():
    eps_list = [1e-4, 5e-4, 1e-3, 1e-2, 5e-2]
    #eps_list = [1e-3, 1e-2]
    total = 10 #3

    mean_cifar = torch.tensor([0.4914, 0.4822, 0.4465])
    std_cifar = torch.tensor([0.2470, 0.2435, 0.2616])

    device = torch.device("cpu")

    _, testloader = get_cifar_loaders(batch_size=1)

    model = CIFARModel()
    model.load_state_dict(torch.load("best_cifar_model.pth", map_location=device))
    model.eval()

    W, b = extract_weights(model)
    for eps in eps_list:

        cert_count = 0
        pgd_success = 0
        clean_correct = 0

        start_time = time.time()

        for i, (x, y) in enumerate(testloader):
            if i >= total:
                break

            x = x.to(device)
            y = y.to(device)

            yi = y.item()

            with torch.no_grad():
                clean_pred = model(x).argmax(1).item()

            clean_correct += int(clean_pred == yi)

            x_flat = x[0].cpu().view(-1).numpy()

            mean_flat = mean_cifar.repeat(16 * 16)
            std_flat = std_cifar.repeat(16 * 16)

            xi = (x_flat - mean_flat.numpy()) / std_flat.numpy()
            eps_iqc = eps / std_cifar.mean().item()

            cert = solve_sdp(W, b, xi, yi, eps_iqc)
            cert_count += int(cert)

            x_adv = pgd_attack(
                model,
                x[0:1],
                y[0:1],
                eps,
                alpha=eps / 5,
                steps=25
            )

            with torch.no_grad():
                adv_pred = model(x_adv).argmax(1).item()

            pgd_success += int(adv_pred != yi)

        clean_acc = clean_correct / total
        cert_rate = cert_count / total
        robust_acc = 1.0 - pgd_success / total

        elapsed = time.time() - start_time

        print(
            f"\nEPS={eps:.5f} | "
            f"clean_acc={clean_acc:.2f} | "
            f"cert_rate={cert_rate:.2f} | "
            f"robust_acc(PGD)={robust_acc:.2f} | "
            f"time={elapsed:.1f}s"
        )