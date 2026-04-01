"""
IMP experiment runner.

Encapsulates the full Iterative Magnitude Pruning loop:

  1. Train a dense baseline.
  2. Save original (dense) weights for lottery-ticket-style masking.
  3. For each target sparsity in the schedule:
       a. Prune to target sparsity (global magnitude threshold on originals).
       b. Fine-tune for 'finetune_epochs' epochs.
       c. Evaluate on validation and test sets.
       d. Optionally evaluate on anomaly-corrupted test set.
  4. Return a results dict that the notebook can plot.

Multiple seeds are supported for confidence-interval estimation.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from magnitude_pruning.dataset import build_loaders
from magnitude_pruning.model   import MambaForecaster
from magnitude_pruning.pruning import (
    get_sparsity,
    prune_to_sparsity,
    register_gradient_masks,
    sparsity_schedule,
)
from magnitude_pruning.train import evaluate, finetune, train_model


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    # Model
    d_model:  int = 64
    d_state:  int = 16
    d_conv:   int = 4
    expand:   int = 2
    n_layers: int = 4
    horizon:  int = 1

    # Data
    seq_len:      int   = 128
    n_total:      int   = 12_000
    batch_size:   int   = 64
    tau:          int   = 17      # 17=mildly chaotic, 30=strongly chaotic

    # Training (dense baseline)
    n_epochs:  int   = 30
    lr:        float = 1e-3
    patience:  int   = 7

    # Fine-tuning per pruning step
    finetune_epochs: int   = 5
    finetune_lr:     float = 3e-4

    # Pruning schedule
    sparsity_start: float = 0.0
    sparsity_end:   float = 0.99
    n_sparsity_steps: int = 14
    log_scale_schedule: bool = True  # more resolution near 99%

    # Experiment
    seeds: list[int] = field(default_factory=lambda: [42, 7, 123])
    device: str = "cuda"           # "cuda" on Colab, "cpu" for local debug

    # "Breaking point" criterion:
    #   val_mse > breaking_factor x baseline_mse  AND  val_mse > breaking_floor
    # The floor prevents early triggering when baseline is near zero (perfect fit),
    # where even numerical noise would satisfy the relative criterion alone.
    breaking_factor: float = 5.0
    breaking_floor:  float = 1e-3


# ---------------------------------------------------------------------------
# Single-seed run
# ---------------------------------------------------------------------------

def run_single_seed(
    cfg: ExperimentConfig,
    seed: int,
    verbose: bool = True,
) -> dict:
    """
    Full IMP experiment for one random seed.

    Returns
    -------
    dict with keys:
      'sparsities'       : list[float]  - actual measured sparsity after pruning
      'val_mse'          : list[float]
      'val_mae'          : list[float]
      'test_mse'         : list[float]
      'test_mae'         : list[float]
      'test_mse_anomaly' : list[float]  - test MSE on anomaly-corrupted data
      'baseline_val_mse' : float
      'breaking_point'   : float | None - first sparsity where MSE > threshold
      'train_history'    : dict          - dense training curves
    """
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Data
    train_loader, val_loader, test_loader = build_loaders(
        seq_len=cfg.seq_len, horizon=cfg.horizon,
        n_total=cfg.n_total, batch_size=cfg.batch_size,
        tau=cfg.tau, seed=seed, add_anomalies=False,
    )
    # Anomaly test set (same series, corrupted copy)
    _, _, anom_loader = build_loaders(
        seq_len=cfg.seq_len, horizon=cfg.horizon,
        n_total=cfg.n_total, batch_size=cfg.batch_size,
        tau=cfg.tau, seed=seed, add_anomalies=True,
    )

    # Dense model
    model = MambaForecaster(
        d_model=cfg.d_model, d_state=cfg.d_state,
        d_conv=cfg.d_conv,   expand=cfg.expand,
        n_layers=cfg.n_layers, horizon=cfg.horizon,
    ).to(device)

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[seed={seed}] Model parameters: {n_params:,}")

    # Train dense baseline
    if verbose:
        print(f"[seed={seed}] Training dense baseline …")
    train_history = train_model(
        model, train_loader, val_loader, device,
        n_epochs=cfg.n_epochs, lr=cfg.lr, patience=cfg.patience,
        verbose=verbose,
    )
    baseline_val_mse = evaluate(model, val_loader, device)["mse"]

    # Store pristine dense weights for lottery-ticket masking
    original_weights = {n: p.data.clone() for n, p in model.named_parameters()}

    # IMP loop
    schedule = sparsity_schedule(
        cfg.sparsity_start, cfg.sparsity_end,
        cfg.n_sparsity_steps, log_scale=cfg.log_scale_schedule,
    )

    results: dict[str, list] = {
        "sparsities":       [],
        "val_mse":          [],
        "val_mae":          [],
        "test_mse":         [],
        "test_mae":         [],
        "test_mse_anomaly": [],
    }
    breaking_point = None

    for target_s in schedule:
        if verbose:
            print(f"  [seed={seed}] Pruning to sparsity {target_s:.2%} …", end=" ")

        # Prune (overwrites model weights with masked versions)
        original_weights = prune_to_sparsity(model, target_s, original_weights)
        actual_s = get_sparsity(model)

        # Register gradient hooks so the optimiser can't revive pruned weights
        handles = register_gradient_masks(model)

        # Fine-tune on the sparse subnetwork
        finetune(model, train_loader, val_loader, device,
                 n_epochs=cfg.finetune_epochs, lr=cfg.finetune_lr,
                 verbose=False)

        # Remove gradient hooks after fine-tuning
        for h in handles:
            h.remove()

        # Evaluate
        val_m   = evaluate(model, val_loader,   device)
        test_m  = evaluate(model, test_loader,  device)
        anom_m  = evaluate(model, anom_loader,  device)

        results["sparsities"].append(actual_s)
        results["val_mse"].append(val_m["mse"])
        results["val_mae"].append(val_m["mae"])
        results["test_mse"].append(test_m["mse"])
        results["test_mae"].append(test_m["mae"])
        results["test_mse_anomaly"].append(anom_m["mse"])

        if verbose:
            print(f"actual={actual_s:.2%}  val_mse={val_m['mse']:.5f}")

        # Check breaking point: relative degradation AND above absolute floor.
        # The floor guards against false triggers when baseline_val_mse ≈ 0.
        bp_threshold = max(cfg.breaking_factor * baseline_val_mse, cfg.breaking_floor)
        if breaking_point is None and val_m["mse"] > bp_threshold:
            breaking_point = actual_s

    results["baseline_val_mse"] = baseline_val_mse
    results["breaking_point"]   = breaking_point
    results["train_history"]    = train_history
    return results


# ---------------------------------------------------------------------------
# Multi-seed experiment (provides confidence intervals)
# ---------------------------------------------------------------------------

def run_experiment(
    cfg: ExperimentConfig,
    verbose: bool = True,
) -> dict:
    """
    Run IMP across all seeds in cfg.seeds and aggregate results.

    Returns
    -------
    dict with keys:
      'seeds'            : list[int]
      'per_seed'         : list[dict]   - raw output of run_single_seed per seed
      'sparsities'       : np.ndarray   - common sparsity axis (mean across seeds)
      'val_mse_mean/std' : np.ndarray
      'test_mse_mean/std': np.ndarray
      'test_mse_anomaly_mean/std' : np.ndarray
      'breaking_points'  : list[float | None]
      'baseline_val_mse' : list[float]
    """
    per_seed = []
    for seed in cfg.seeds:
        result = run_single_seed(cfg, seed, verbose=verbose)
        per_seed.append(result)

    # Aggregate: use the sparsity axis from the first seed (schedules are
    # deterministic so they are identical across seeds)
    sparsities = np.array(per_seed[0]["sparsities"])

    def _agg(key):
        matrix = np.array([r[key] for r in per_seed])  # (n_seeds, n_steps)
        return matrix.mean(0), matrix.std(0)

    val_mse_mean,   val_mse_std   = _agg("val_mse")
    test_mse_mean,  test_mse_std  = _agg("test_mse")
    anom_mse_mean,  anom_mse_std  = _agg("test_mse_anomaly")

    return {
        "seeds":       cfg.seeds,
        "per_seed":    per_seed,
        "sparsities":  sparsities,
        "val_mse_mean":              val_mse_mean,
        "val_mse_std":               val_mse_std,
        "test_mse_mean":             test_mse_mean,
        "test_mse_std":              test_mse_std,
        "test_mse_anomaly_mean":     anom_mse_mean,
        "test_mse_anomaly_std":      anom_mse_std,
        "breaking_points":  [r["breaking_point"]   for r in per_seed],
        "baseline_val_mse": [r["baseline_val_mse"] for r in per_seed],
    }
