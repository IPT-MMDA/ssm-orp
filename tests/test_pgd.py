import torch
import sys
import os
sys.path.append(os.path.abspath(".."))
from src.models.mnist_model import MNISTModel
from src.attacks.pgd import pgd_attack


def test_pgd_runs():
    model = MNISTModel()

    x = torch.randn(1, 1, 28, 28)
    y = torch.tensor([1])

    x_adv = pgd_attack(model, x, y, eps=0.1)

    assert x_adv.shape == x.shape

    #print("test_pgd_runs OK")

def test_pgd_within_epsilon():
    model = MNISTModel()
    x = torch.randn(1, 1, 28, 28)
    y = torch.tensor([1])

    eps = 0.1
    x_adv = pgd_attack(model, x, y, eps=eps)

    diff = (x_adv - x).abs().max().item()

    assert diff <= eps + 1e-5