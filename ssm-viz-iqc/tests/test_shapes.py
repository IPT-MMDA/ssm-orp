import torch
import sys
import os
sys.path.append(os.path.abspath(".."))
from src.models.mnist_model import MNISTModel
from src.utils.extract_weights import extract_weights

import numpy as np
from src.verification.solver import solve_sdp


def test_model_forward():
    model = MNISTModel()

    x = torch.randn(2, 1, 28, 28)
    out = model(x)

    assert out.shape == (2, 10)

    #print("test_shapes_runs OK")

def test_sdp_input_shapes():
    W = [np.random.randn(10, 5), np.random.randn(5, 10)]
    b = [np.random.randn(10), np.random.randn(5)]
    x = np.random.randn(5)

    try:
        solve_sdp(W, b, x, 0, 0.1)
    except Exception:
        assert False, "SDP crashed on valid input"