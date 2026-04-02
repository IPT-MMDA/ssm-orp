"""Tests for trainer.py.

Uses a tiny randomly-initialised MambaPrefixModel and a synthetic DataLoader
so no downloads or GPU are required.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import MambaConfig

from ssm_prefix_tuning.config import EpochResult, PeriodicInjectionConfig, PrefixConfig, TrainingConfig
from ssm_prefix_tuning.model_wrapper import build_prefix_model_from_config
from ssm_prefix_tuning.trainer import build_optimizer, evaluate, train_one_epoch


# Helpers

def _tiny_config() -> MambaConfig:
    return MambaConfig(vocab_size=100, hidden_size=64, num_hidden_layers=2,
                       state_size=4, expand=2, num_labels=2)


def _tiny_model():
    p_cfg = PrefixConfig(prefix_length=3, hidden_size=64)
    i_cfg = PeriodicInjectionConfig(enabled=False)
    return build_prefix_model_from_config(_tiny_config(), p_cfg, i_cfg)


def _tiny_loader(n: int = 8, seq_len: int = 10, batch_size: int = 4) -> DataLoader:
    """Synthetic DataLoader with random token ids and alternating 0/1 labels."""
    ids = torch.randint(0, 100, (n, seq_len))
    mask = torch.ones(n, seq_len, dtype=torch.long)
    labels = torch.tensor([i % 2 for i in range(n)])
    ds = TensorDataset(ids, mask, labels)

    def collate(batch):
        ids_, mask_, lbl = zip(*batch)
        return {
            "input_ids": torch.stack(ids_),
            "attention_mask": torch.stack(mask_),
            "labels": torch.stack(lbl),
        }

    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)


# build_optimizer

def test_build_optimizer_only_trainable_params():
    """Optimiser parameter groups must not include frozen parameters."""
    model = _tiny_model()
    opt = build_optimizer(model, lr=1e-3, weight_decay=0.01)
    opt_param_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    for p in model.parameters():
        if p.requires_grad:
            assert id(p) in opt_param_ids, "Trainable param missing from optimiser"
        else:
            assert id(p) not in opt_param_ids, "Frozen param should not be in optimiser"


def test_build_optimizer_returns_adamw():
    """build_optimizer returns a torch.optim.AdamW instance."""
    model = _tiny_model()
    opt = build_optimizer(model, lr=1e-3, weight_decay=0.0)
    assert isinstance(opt, torch.optim.AdamW)


# train_one_epoch

def test_train_one_epoch_returns_float():
    """train_one_epoch returns a scalar float loss."""
    model = _tiny_model()
    loader = _tiny_loader()
    opt = build_optimizer(model, lr=1e-3, weight_decay=0.0)
    device = torch.device("cpu")
    loss = train_one_epoch(model, loader, opt, device)
    assert isinstance(loss, float)
    assert loss > 0


def test_train_one_epoch_updates_prefix_only():
    """After one step only prefix_encoder parameters should have changed."""
    model = _tiny_model()
    # Record initial frozen param values.
    frozen_before = {
        name: p.data.clone()
        for name, p in model._backbone.named_parameters()
    }
    loader = _tiny_loader()
    opt = build_optimizer(model, lr=1e-2, weight_decay=0.0)
    train_one_epoch(model, loader, opt, torch.device("cpu"))

    for name, p in model._backbone.named_parameters():
        assert torch.equal(frozen_before[name], p.data), (
            f"Frozen backbone param {name} was modified during training"
        )


def test_two_epochs_loss_can_decrease():
    """Running two epochs on a fixed seed should give reasonable losses (not NaN)."""
    torch.manual_seed(0)
    model = _tiny_model()
    loader = _tiny_loader(n=16, batch_size=4)
    opt = build_optimizer(model, lr=1e-2, weight_decay=0.0)
    device = torch.device("cpu")
    loss1 = train_one_epoch(model, loader, opt, device)
    loss2 = train_one_epoch(model, loader, opt, device)
    assert not (loss1 != loss1), "Loss is NaN in epoch 1"
    assert not (loss2 != loss2), "Loss is NaN in epoch 2"


# evaluate

def test_evaluate_returns_required_keys():
    """evaluate() returns a dict with 'accuracy' and 'f1' keys."""
    model = _tiny_model()
    loader = _tiny_loader()
    out = evaluate(model, loader, torch.device("cpu"))
    assert "accuracy" in out
    assert "f1" in out


def test_evaluate_accuracy_in_range():
    """Accuracy must be in [0, 1]."""
    model = _tiny_model()
    loader = _tiny_loader()
    out = evaluate(model, loader, torch.device("cpu"))
    assert 0.0 <= out["accuracy"] <= 1.0


def test_evaluate_f1_in_range():
    """Macro F1 must be in [0, 1]."""
    model = _tiny_model()
    loader = _tiny_loader()
    out = evaluate(model, loader, torch.device("cpu"))
    assert 0.0 <= out["f1"] <= 1.0


def test_evaluate_predictions_list():
    """evaluate() includes 'predictions' and 'references' lists."""
    model = _tiny_model()
    loader = _tiny_loader(n=8)
    out = evaluate(model, loader, torch.device("cpu"))
    assert "predictions" in out and "references" in out
    assert len(out["predictions"]) == 8
    assert len(out["references"]) == 8


# EpochResult

def test_epoch_result_is_dataclass():
    """EpochResult can be constructed with positional arguments."""
    r = EpochResult(epoch=1, train_loss=0.5, val_accuracy=0.8,
                    val_f1=0.79, ci_lower=0.77, ci_upper=0.83)
    assert r.epoch == 1
    assert r.val_accuracy == 0.8
