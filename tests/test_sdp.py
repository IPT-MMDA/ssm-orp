import sys
import os
import torch
sys.path.append(os.path.abspath(".."))
from src.verification.solver import solve_sdp
from src.models.mnist_model import MNISTModel
from src.attacks.pgd import pgd_attack
from src.utils.extract_weights import extract_weights
import numpy as np



def test_sdp_runs():
    W = [np.random.randn(10, 5), np.random.randn(5, 10)]
    b = [np.random.randn(10), np.random.randn(5)]
    x = np.random.randn(5)

    out = solve_sdp(W, b, x, 0, 0.1)

    assert out in [0, 1]

    print("test_sdp_runs OK")

def test_sdp_deterministic():
    W = [np.random.randn(10, 5), np.random.randn(5, 10)]
    b = [np.random.randn(10), np.random.randn(5)]
    x = np.random.randn(5)

    out1 = solve_sdp(W, b, x, 0, 0.1)
    out2 = solve_sdp(W, b, x, 0, 0.1)

    assert out1 == out2

def test_sdp_soundness_property():
    model = MNISTModel()
    model.eval()

    eps = 0.1

    x = torch.randn(1, 1, 28, 28)
    y = torch.tensor([1])

    W, b = extract_weights(model)
    x0 = x.view(-1).numpy()

    x_adv = pgd_attack(model, x, y, eps)

    is_pgd_fail = (model(x_adv).argmax(1) != y).item()

    sdp_cert = solve_sdp(W, b, x0, int(y.item()), eps)

    if is_pgd_fail:
        assert sdp_cert == 0
