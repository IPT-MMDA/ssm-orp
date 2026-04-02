"""Tests for prefix_encoder.py.

All tests run on CPU with randomly-initialised tiny configs — no model
downloads or GPU required.
"""

import math

import pytest
import torch

from ssm_prefix_tuning.config import PeriodicInjectionConfig, PrefixConfig
from ssm_prefix_tuning.prefix_encoder import PeriodicPrefixInjector, PrefixEncoder


# PrefixEncoder — direct mode

def test_prefix_encoder_direct_output_shape():
    """Direct PrefixEncoder returns [K, D]."""
    cfg = PrefixConfig(prefix_length=5, hidden_size=64, projection=False)
    enc = PrefixEncoder(cfg)
    out = enc()
    assert out.shape == (5, 64), f"Expected (5, 64), got {out.shape}"


def test_prefix_encoder_direct_gradients():
    """Gradients flow back to the embedding parameters."""
    cfg = PrefixConfig(prefix_length=5, hidden_size=64, projection=False)
    enc = PrefixEncoder(cfg)
    loss = enc().sum()
    loss.backward()
    for name, p in enc.named_parameters():
        assert p.grad is not None, f"No gradient on {name}"


def test_prefix_encoder_requires_grad_default():
    """Prefix parameters have requires_grad=True by default."""
    cfg = PrefixConfig(prefix_length=3, hidden_size=32)
    enc = PrefixEncoder(cfg)
    for p in enc.parameters():
        assert p.requires_grad


# PrefixEncoder — projected mode

def test_prefix_encoder_projected_output_shape():
    """Projected PrefixEncoder also returns [K, D] despite the MLP bottleneck."""
    cfg = PrefixConfig(
        prefix_length=5,
        hidden_size=64,
        projection=True,
        projection_hidden_size=16,
    )
    enc = PrefixEncoder(cfg)
    out = enc()
    assert out.shape == (5, 64)


def test_prefix_encoder_projected_gradients():
    """Gradients flow through the MLP transform in projected mode."""
    cfg = PrefixConfig(prefix_length=4, hidden_size=32, projection=True, projection_hidden_size=8)
    enc = PrefixEncoder(cfg)
    enc().sum().backward()
    for name, p in enc.named_parameters():
        assert p.grad is not None, f"No gradient on {name}"


def test_prefix_encoder_projected_has_transform():
    """Projected encoder exposes a non-None transform module."""
    cfg = PrefixConfig(prefix_length=3, hidden_size=16, projection=True, projection_hidden_size=8)
    enc = PrefixEncoder(cfg)
    assert enc.transform is not None


def test_prefix_encoder_direct_has_no_transform():
    """Direct encoder has transform=None."""
    cfg = PrefixConfig(prefix_length=3, hidden_size=16, projection=False)
    enc = PrefixEncoder(cfg)
    assert enc.transform is None


# PeriodicPrefixInjector — output lengths

def _make_injector(K: int, D: int, period: int) -> tuple[PeriodicPrefixInjector, PrefixEncoder]:
    prefix_cfg = PrefixConfig(prefix_length=K, hidden_size=D)
    enc = PrefixEncoder(prefix_cfg)
    inj_cfg = PeriodicInjectionConfig(enabled=True, period=period)
    injector = PeriodicPrefixInjector(inj_cfg, enc)
    return injector, enc


def test_injector_output_length_exact():
    """With T=10, K=3, period=4: inject at 0, 4, 8 → 3 injections, total T'=19."""
    # Positions: 0, 4, 8 → n_injections = ceil(10/4) = 3
    # T' = 10 + 3*3 = 19
    injector, _ = _make_injector(K=3, D=8, period=4)
    embeds = torch.randn(2, 10, 8)
    mask = torch.ones(2, 10, dtype=torch.long)
    out_embeds, out_mask = injector(embeds, mask)
    n_injections = math.ceil(10 / 4)  # = 3
    expected_len = 10 + 3 * n_injections
    assert out_embeds.shape == (2, expected_len, 8), f"Got shape {out_embeds.shape}"
    assert out_mask.shape == (2, expected_len)


def test_injector_output_length_period_larger_than_T():
    """When period ≥ T only one injection happens (at position 0)."""
    injector, _ = _make_injector(K=2, D=8, period=100)
    embeds = torch.randn(1, 10, 8)
    mask = torch.ones(1, 10, dtype=torch.long)
    out_embeds, out_mask = injector(embeds, mask)
    expected_len = 10 + 2 * 1  # 1 injection
    assert out_embeds.shape == (1, expected_len, 8)


def test_injector_output_length_period_equals_one():
    """With period=1, a prefix is injected before every single token."""
    K, T = 2, 5
    injector, _ = _make_injector(K=K, D=8, period=1)
    embeds = torch.randn(1, T, 8)
    mask = torch.ones(1, T, dtype=torch.long)
    out_embeds, out_mask = injector(embeds, mask)
    expected_len = T + K * T  # T injections
    assert out_embeds.shape == (1, expected_len, 8)


# PeriodicPrefixInjector — mask integrity

def test_injector_prefix_positions_are_attended():
    """All injected prefix positions have mask value 1."""
    K, period = 2, 5
    injector, _ = _make_injector(K=K, D=8, period=period)
    B, T = 1, 5
    embeds = torch.randn(B, T, 8)
    mask = torch.ones(B, T, dtype=torch.long)
    _, out_mask = injector(embeds, mask)
    # First K positions are a prefix block → should all be 1.
    assert out_mask[0, :K].all(), "Injected prefix positions must be masked-in (=1)"


def test_injector_preserves_original_padding():
    """Original pad tokens (mask=0) keep their 0 value after injection."""
    K, period = 2, 5
    injector, _ = _make_injector(K=K, D=8, period=period)
    B, T = 1, 8
    embeds = torch.randn(B, T, 8)
    # Last 3 positions are padding.
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.long)
    _, out_mask = injector(embeds, mask)
    # The last 3 entries in the output should still be 0 (original padding).
    assert out_mask[0, -3:].sum().item() == 0, "Original padding must be preserved"


# PeriodicPrefixInjector — gradient flow

def test_injector_gradient_flows_to_prefix_encoder():
    """Loss through injected embeddings must reach PrefixEncoder parameters."""
    K, D = 3, 16
    prefix_cfg = PrefixConfig(prefix_length=K, hidden_size=D)
    enc = PrefixEncoder(prefix_cfg)
    inj_cfg = PeriodicInjectionConfig(enabled=True, period=4)
    injector = PeriodicPrefixInjector(inj_cfg, enc)

    embeds = torch.randn(2, 8, D)
    mask = torch.ones(2, 8, dtype=torch.long)
    out_embeds, _ = injector(embeds, mask)
    out_embeds.sum().backward()

    for name, p in enc.named_parameters():
        assert p.grad is not None, f"No gradient on PrefixEncoder.{name}"
