"""
Tests for pruning.py - uses a small stub model so no GPU / mamba-ssm needed.
"""
import pytest
import torch
import torch.nn as nn

from magnitude_pruning.pruning import (
    _mask_buf_name,
    apply_masks,
    compute_threshold,
    get_sparsity,
    prune_to_sparsity,
    register_gradient_masks,
    sparsity_schedule,
)


# ---------------------------------------------------------------------------
# Minimal stub that satisfies the interface expected by pruning helpers
# ---------------------------------------------------------------------------

class TinyModel(nn.Module):
    """Two linear layers - no mamba-ssm dependency."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.fc1 = nn.Linear(16, 16, bias=False)
        self.fc2 = nn.Linear(16, 1,  bias=False)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

    # mimic MambaForecaster.named_prunable_params
    def named_prunable_params(self):
        return [
            ("fc1.weight", self.fc1, "weight"),
            ("fc2.weight", self.fc2, "weight"),
        ]

    def count_params(self):
        total   = sum(p.numel() for p in self.parameters())
        nonzero = sum((p != 0).sum().item() for p in self.parameters())
        return {"total": total, "nonzero": nonzero, "zero": total - nonzero}


# ---------------------------------------------------------------------------
# compute_threshold
# ---------------------------------------------------------------------------

def test_threshold_zero_sparsity():
    model = TinyModel()
    t = compute_threshold(model, 0.0)
    # At 0% sparsity the threshold is the minimum magnitude (no weights pruned)
    all_mags = torch.cat([p.data.abs().flatten() for p in model.parameters()])
    assert t == pytest.approx(all_mags.min().item(), rel=1e-4)


def test_threshold_increases_with_sparsity():
    model = TinyModel()
    t30 = compute_threshold(model, 0.30)
    t70 = compute_threshold(model, 0.70)
    assert t70 > t30


def test_threshold_invalid_raises():
    model = TinyModel()
    with pytest.raises(ValueError):
        compute_threshold(model, 1.0)
    with pytest.raises(ValueError):
        compute_threshold(model, -0.1)


# ---------------------------------------------------------------------------
# prune_to_sparsity + get_sparsity
# ---------------------------------------------------------------------------

def test_prune_creates_masks():
    model = TinyModel()
    prune_to_sparsity(model, 0.5)
    for _, module, param_name in model.named_prunable_params():
        assert hasattr(module, _mask_buf_name(param_name)), \
            "Mask buffer should be registered after pruning"


def test_actual_sparsity_approximate():
    """After pruning to 50%, measured sparsity should be ~50%."""
    model = TinyModel()
    prune_to_sparsity(model, 0.5)
    s = get_sparsity(model)
    # Allow +-5% tolerance because quantile at discrete weights
    assert 0.45 <= s <= 0.55, f"Expected ~50% sparsity, got {s:.2%}"


def test_progressive_pruning_monotone():
    """Each pruning step should not decrease sparsity."""
    model = TinyModel()
    orig = None
    prev_s = 0.0
    for target in [0.2, 0.4, 0.6, 0.8]:
        orig = prune_to_sparsity(model, target, orig)
        s = get_sparsity(model)
        assert s >= prev_s - 1e-6, \
            f"Sparsity decreased: {prev_s:.3f} → {s:.3f} at target {target}"
        prev_s = s


def test_fully_pruned_weights_are_zero():
    model = TinyModel()
    prune_to_sparsity(model, 0.9)
    zero_count = sum((p == 0).sum().item() for p in model.parameters())
    total      = sum(p.numel() for p in model.parameters())
    assert zero_count / total >= 0.85, \
        "At least 85% of weights should be zero at 90% target sparsity"


# ---------------------------------------------------------------------------
# apply_masks
# ---------------------------------------------------------------------------

def test_apply_masks_keeps_pruned_zero():
    model = TinyModel()
    prune_to_sparsity(model, 0.5)
    # Manually set all weights to 1 (simulates an optimiser step)
    for p in model.parameters():
        p.data.fill_(1.0)
    apply_masks(model)
    # After re-applying masks, previously pruned weights should be 0
    s = get_sparsity(model)
    assert s >= 0.45, "apply_masks should restore sparsity after weight update"


# ---------------------------------------------------------------------------
# register_gradient_masks
# ---------------------------------------------------------------------------

def test_gradient_masks_zero_pruned_grads():
    model = TinyModel()
    prune_to_sparsity(model, 0.5)
    handles = register_gradient_masks(model)

    x = torch.randn(4, 16)
    loss = model(x).sum()
    loss.backward()

    for _, module, param_name in model.named_prunable_params():
        mask = getattr(module, _mask_buf_name(param_name))
        param = getattr(module, param_name)
        if param.grad is not None:
            zero_in_mask  = (mask == 0)
            grad_at_zeros = param.grad[zero_in_mask]
            assert grad_at_zeros.abs().max().item() < 1e-6, \
                "Gradients at pruned positions should be zeroed by hook"

    for h in handles:
        h.remove()


# ---------------------------------------------------------------------------
# sparsity_schedule
# ---------------------------------------------------------------------------

def test_schedule_length():
    s = sparsity_schedule(0.0, 0.99, n_steps=10)
    assert len(s) == 10


def test_schedule_bounds():
    s = sparsity_schedule(0.0, 0.99, n_steps=5)
    assert s[0]  == pytest.approx(0.0,  abs=1e-4)
    assert s[-1] == pytest.approx(0.99, abs=1e-2)


def test_schedule_monotone():
    s = sparsity_schedule(0.0, 0.95, n_steps=8)
    for a, b in zip(s, s[1:]):
        assert b >= a - 1e-6, "Schedule should be non-decreasing"


def test_log_schedule_has_more_high_end():
    lin = sparsity_schedule(0.0, 0.99, n_steps=10, log_scale=False)
    log = sparsity_schedule(0.0, 0.99, n_steps=10, log_scale=True)
    # Log schedule should have more points in the upper half [0.5, 0.99]
    lin_high = sum(1 for x in lin if x > 0.5)
    log_high = sum(1 for x in log if x > 0.5)
    assert log_high >= lin_high
