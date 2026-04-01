"""
Tests for MambaForecaster. Require CUDA + mamba-ssm, skipped otherwise.
"""
import pytest
import torch

try:
    from magnitude_pruning.model import MambaForecaster
    HAS_MODEL = True
except ImportError:
    HAS_MODEL = False

model_required = pytest.mark.skipif(
    not HAS_MODEL or not torch.cuda.is_available(),
    reason="mamba-ssm not installed or no CUDA"
)


@model_required
def test_forward_shape():
    device = torch.device("cuda")
    model = MambaForecaster(d_model=32, d_state=8, d_conv=4, expand=2,
                            n_layers=2, horizon=1).to(device)
    x = torch.randn(4, 64, 1, device=device)
    assert model(x).shape == (4, 1)


@model_required
def test_forward_multi_horizon():
    device = torch.device("cuda")
    model = MambaForecaster(d_model=32, d_state=8, d_conv=4, expand=2,
                            n_layers=2, horizon=5).to(device)
    assert model(torch.randn(2, 64, 1, device=device)).shape == (2, 5)


@model_required
def test_named_prunable_params_non_empty():
    device = torch.device("cuda")
    model = MambaForecaster(d_model=32, n_layers=2).to(device)
    assert len(model.named_prunable_params()) > 0


@model_required
def test_named_prunable_params_excludes_A_log():
    device = torch.device("cuda")
    model = MambaForecaster(d_model=32, n_layers=2).to(device)
    names = [name for name, _, _ in model.named_prunable_params()]
    assert not any("A_log" in n for n in names), "A_log must not be prunable"


@model_required
def test_count_params_consistent():
    device = torch.device("cuda")
    model = MambaForecaster(d_model=32, n_layers=2).to(device)
    counts = model.count_params()
    assert counts["total"] == counts["nonzero"] + counts["zero"]
    assert counts["zero"] == 0


@model_required
def test_no_nan_in_forward():
    device = torch.device("cuda")
    torch.manual_seed(42)
    model = MambaForecaster(d_model=32, n_layers=2).to(device)
    y = model(torch.randn(8, 128, 1, device=device))
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()
