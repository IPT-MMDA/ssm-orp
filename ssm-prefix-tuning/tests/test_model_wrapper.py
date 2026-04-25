"""Tests for model_wrapper.py.

All tests use a tiny randomly-initialised MambaConfig (2 layers, hidden_size=64)
so no model downloads are required and tests run quickly on CPU.
"""

import pytest
import torch
from transformers import MambaConfig

from ssm_prefix_tuning.config import PeriodicInjectionConfig, PrefixConfig
from ssm_prefix_tuning.model_wrapper import (
    MambaPrefixModel,
    build_prefix_model_from_config,
)


# Shared helpers

def _tiny_mamba_config(num_labels: int = 2) -> MambaConfig:
    """Return a minimal MambaConfig that runs quickly on CPU."""
    return MambaConfig(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        state_size=4,
        expand=2,
        num_labels=num_labels,
    )


def _build_model(
    prefix_length: int = 3,
    periodic: bool = False,
    period: int = 4,
    num_labels: int = 2,
) -> MambaPrefixModel:
    cfg = _tiny_mamba_config(num_labels)
    p_cfg = PrefixConfig(prefix_length=prefix_length, hidden_size=cfg.hidden_size)
    i_cfg = PeriodicInjectionConfig(enabled=periodic, period=period)
    return build_prefix_model_from_config(cfg, p_cfg, i_cfg, num_labels)


# Output shape

def test_forward_logits_shape_binary():
    """MambaPrefixModel.forward returns logits of shape [B, 2] for SST-2."""
    model = _build_model()
    input_ids = torch.randint(0, 100, (2, 10))
    mask = torch.ones(2, 10, dtype=torch.long)
    out = model(input_ids, mask)
    assert out.logits.shape == (2, 2), f"Got {out.logits.shape}"


def test_forward_logits_shape_multiclass():
    """logits shape is [B, num_labels] for a multiclass setup."""
    model = _build_model(num_labels=5)
    input_ids = torch.randint(0, 100, (3, 8))
    out = model(input_ids)
    assert out.logits.shape == (3, 5)


def test_forward_returns_loss_when_labels_given():
    """A scalar loss is returned when labels are provided."""
    model = _build_model()
    input_ids = torch.randint(0, 100, (2, 10))
    labels = torch.tensor([0, 1])
    out = model(input_ids, labels=labels)
    assert out.loss is not None
    assert out.loss.ndim == 0, "Loss must be a scalar tensor"


def test_forward_no_loss_without_labels():
    """Loss is None when no labels are provided."""
    model = _build_model()
    input_ids = torch.randint(0, 100, (2, 10))
    out = model(input_ids)
    assert out.loss is None


# Freezing / trainability

def test_backbone_is_frozen():
    """All backbone (MambaModel) parameters must be frozen after construction."""
    model = _build_model()
    # The backbone lives inside hf_model.backbone.
    backbone = model._backbone
    for name, p in backbone.named_parameters():
        assert not p.requires_grad, f"Backbone param {name} should be frozen"


def test_prefix_encoder_is_trainable():
    """PrefixEncoder parameters must have requires_grad=True."""
    model = _build_model()
    for name, p in model.prefix_encoder.named_parameters():
        assert p.requires_grad, f"Prefix param {name} should be trainable"


def test_classifier_is_trainable():
    """The classification head must remain trainable."""
    model = _build_model()
    for name, p in model.classifier_model.classifier.named_parameters():
        assert p.requires_grad, f"Classifier param {name} should be trainable"


def test_trainable_parameters_list():
    """trainable_parameters() returns exactly the non-frozen parameters."""
    model = _build_model()
    trainable_ids = {id(p) for p in model.trainable_parameters()}
    for p in model.parameters():
        if p.requires_grad:
            assert id(p) in trainable_ids
        else:
            assert id(p) not in trainable_ids


# Gradient flow

def test_loss_backward_updates_prefix():
    """Cross-entropy loss must flow gradients back to the prefix encoder."""
    model = _build_model()
    input_ids = torch.randint(0, 100, (2, 10))
    labels = torch.tensor([0, 1])
    out = model(input_ids, labels=labels)
    out.loss.backward()
    for name, p in model.prefix_encoder.named_parameters():
        assert p.grad is not None, f"No gradient on prefix_encoder.{name}"


def test_loss_backward_does_not_update_frozen_params():
    """Frozen backbone parameters must have grad=None after backward."""
    model = _build_model()
    input_ids = torch.randint(0, 100, (2, 10))
    labels = torch.tensor([0, 1])
    out = model(input_ids, labels=labels)
    out.loss.backward()
    for name, p in model._backbone.named_parameters():
        assert p.grad is None, f"Frozen param {name} received a gradient"


# Periodic injection integration

def test_periodic_injection_forward_runs():
    """MambaPrefixModel with periodic injection completes a forward pass."""
    model = _build_model(prefix_length=2, periodic=True, period=3)
    input_ids = torch.randint(0, 100, (2, 9))
    out = model(input_ids)
    assert out.logits.shape == (2, 2)


def test_periodic_injection_gradient_flows():
    """Gradients reach the prefix encoder when periodic injection is active."""
    model = _build_model(prefix_length=2, periodic=True, period=3)
    input_ids = torch.randint(0, 100, (2, 9))
    labels = torch.tensor([0, 1])
    out = model(input_ids, labels=labels)
    out.loss.backward()
    for name, p in model.prefix_encoder.named_parameters():
        assert p.grad is not None, f"No gradient on prefix_encoder.{name}"
