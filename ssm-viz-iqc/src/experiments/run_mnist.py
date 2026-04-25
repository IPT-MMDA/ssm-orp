import torch
import time

from src.models.mnist_model import MNISTModel
from src.verification.solver import solve_sdp
from src.utils.extract_weights import extract_weights
from src.data.mnist import get_mnist_loaders
from src.attacks.pgd import pgd_attack


def run():
    # eps_list = [1e-4, 5e-4, 1e-3, 1e-2, 5e-2]
    # eps_list = [1e-4, 5e-4]
    # eps_list = [1e-4, 1e-2, 1e-1]
    # eps_list = [1e-2, 1e-1]
    # eps_list = [1e-4, 1e-1]
    eps_list = [1e-4, 5e-4, 1e-3, 1e-2, 5e-2]
    total = 12 #20

    mean_mnist = 0.1307
    std_mnist = 0.3081

    device = torch.device("cpu")
    _, testloader = get_mnist_loaders(batch_size=1)

    model = MNISTModel()
    model.load_state_dict(torch.load("best_mnist_model.pth", map_location=device))
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

            with torch.no_grad():
                clean_pred = model(x).argmax(1).item()

            yi = y.item()
            clean_correct += int(clean_pred == yi)
            xi = (x[0].cpu().view(-1).numpy() - mean_mnist) / std_mnist
            eps_iqc = eps / std_mnist

            cert = solve_sdp(W, b, xi, yi, eps_iqc)
            cert_count += int(cert)
            x_adv = pgd_attack(
                model,
                x[0:1],
                y[0:1],
                eps,
                alpha=eps / 10, #5
                steps=50 #20 # 100
            )

            with torch.no_grad():
                adv_pred = model(x_adv).argmax(1).item()

            pgd_success += int(adv_pred != yi)

        cert_rate = cert_count / total
        robust_acc = 1.0 - pgd_success / total
        clean_acc = clean_correct / total

        elapsed = time.time() - start_time

        print(
            f"\nEPS={eps:.5f} | "
            f"clean_acc={clean_acc:.2f} | "
            f"cert_rate={cert_rate:.2f} | "
            f"robust_acc(PGD)={robust_acc:.2f} | "
            f"time={elapsed:.1f}s"
        )