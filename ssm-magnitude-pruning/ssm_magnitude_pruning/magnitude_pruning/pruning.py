"""
Iterative Magnitude Pruning (IMP) for MambaForecaster.

Strategy
--------
Global unstructured magnitude pruning:
    For a given target sparsity s Є [0, 1), compute a single threshold t such
    that exactly s% of all prunable weights have |w| < t, then set them to zero
    and freeze the corresponding mask.

Why global rather than per-layer?
    Layer-wise pruning with a uniform rate destroys the narrower Mamba matrices
    (which have fewer parameters) disproportionately.  A global threshold
    distributes the budget according to actual weight magnitudes.

Mask storage
------------
Masks are stored as non-parameter buffers so they survive state_dict round-
trips and are excluded from gradient updates.  The prune/restore helpers
support the "lottery ticket" style: save the *original* dense weights and
apply successive masks on top of them (rather than re-computing magnitudes on
already-zeroed tensors, which would bias later iterations).
"""

from __future__ import annotations

import copy
from typing import Iterator

import torch
import torch.nn as nn
import numpy as np

# ---------------------------------------------------------------------------
# Core pruning logic
# ---------------------------------------------------------------------------

def _all_prunable_weights(model: "MambaForecaster") -> torch.Tensor:
    """Concatenate all prunable weight tensors into one flat vector."""
    parts = []
    for _, module, param_name in model.named_prunable_params():
        w = getattr(module, param_name)
        if w is not None:
            parts.append(w.data.abs().flatten())
    return torch.cat(parts)


def compute_threshold(model: "MambaForecaster", target_sparsity: float) -> float:
    """
    Return the magnitude threshold t such that 'target_sparsity' fraction of
    all prunable weights satisfy |w| <= t.
    """
    if not (0.0 <= target_sparsity < 1.0):
        raise ValueError(f"target_sparsity must be in [0, 1), got {target_sparsity}")
    magnitudes = _all_prunable_weights(model)
    return torch.quantile(magnitudes, target_sparsity).item()


def apply_masks(model: "MambaForecaster") -> None:
    """
    Multiply each prunable weight by its stored mask buffer.
    Call this after any optimiser step to re-enforce sparsity.
    """
    for full_name, module, param_name in model.named_prunable_params():
        mask_name = _mask_buf_name(param_name)
        if hasattr(module, mask_name):
            mask = getattr(module, mask_name)
            w    = getattr(module, param_name)
            w.data.mul_(mask)


def _mask_buf_name(param_name: str) -> str:
    return f"_prune_mask_{param_name}"


def prune_to_sparsity(
    model: "MambaForecaster",
    target_sparsity: float,
    original_weights: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """
    Compute a new global mask achieving 'target_sparsity' and register it as
    a buffer on each sub-module.

    If 'original_weights' is provided (lottery-ticket style), the threshold is
    computed on the *original* (dense) weights, so we never mask weights that
    were already zero from a previous round.

    Returns the original_weights dict (useful on the first call where None is
    passed - the function saves a copy before masking).
    """
    # Save original weights on first call
    if original_weights is None:
        original_weights = {
            n: p.data.clone()
            for n, p in model.named_parameters()
        }

    # Temporarily restore originals to compute a fair threshold
    _restore_weights(model, original_weights)

    threshold = compute_threshold(model, target_sparsity)

    for full_name, module, param_name in model.named_prunable_params():
        w = getattr(module, param_name)
        mask = (w.data.abs() > threshold).float()

        mask_buf = _mask_buf_name(param_name)
        # Register / update the buffer
        if hasattr(module, mask_buf):
            getattr(module, mask_buf).copy_(mask)
        else:
            module.register_buffer(mask_buf, mask)

        # Apply immediately
        w.data.mul_(mask)

    return original_weights


def _restore_weights(
    model: "MambaForecaster",
    original_weights: dict[str, torch.Tensor],
) -> None:
    """Re-set model weights to their pre-pruning values."""
    for name, param in model.named_parameters():
        if name in original_weights:
            param.data.copy_(original_weights[name])


def get_sparsity(model: "MambaForecaster") -> float:
    """Actual fraction of prunable weights that are exactly zero."""
    counts = model.count_params()
    return counts["zero"] / max(counts["total"], 1)


# ---------------------------------------------------------------------------
# Gradient mask hook (keeps pruned weights at zero during fine-tuning)
# ---------------------------------------------------------------------------

def register_gradient_masks(model: "MambaForecaster") -> list:
    """
    Register backward hooks that zero out gradients for pruned weights.
    This prevents the optimiser from accidentally reviving dead weights.

    Returns a list of hook handles (call '.remove()' on each to detach).
    """
    handles = []
    for _, module, param_name in model.named_prunable_params():
        mask_buf = _mask_buf_name(param_name)
        if not hasattr(module, mask_buf):
            continue
        param = getattr(module, param_name)
        mask  = getattr(module, mask_buf)

        handle = param.register_hook(lambda grad, m=mask: grad * m)
        handles.append(handle)
    return handles


# ---------------------------------------------------------------------------
# Sparsity schedule helper
# ---------------------------------------------------------------------------

def sparsity_schedule(
    start: float = 0.0,
    end: float   = 0.99,
    n_steps: int = 13,
    log_scale: bool = False,
) -> list[float]:
    """
    Return a list of target sparsity values for the IMP outer loop.

    log_scale=True gives finer resolution at high sparsities, which is where
    the "breaking point" typically occurs.
    """
    if log_scale:
        # Power ramp: t^3 maps [0,1] -> [0,1] with most density near t=1
        # so the schedule has finer resolution at high sparsities (where the
        # "breaking point" typically occurs).
        # 1-(1-t)^3 maps [0,1]->[0,1] with high density near t=1
        linspace = np.linspace(0, 1, n_steps)
        schedule = [start + (end - start) * float(1 - (1 - t) ** 3) for t in linspace]
        return [round(s, 4) for s in schedule]
    else:
        return [round(float(s), 4) for s in np.linspace(start, end, n_steps)]
