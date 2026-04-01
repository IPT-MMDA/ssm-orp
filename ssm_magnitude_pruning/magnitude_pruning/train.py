"""
Training and fine-tuning routines for MambaForecaster.

Two public entry points:
  train_epoch     – run one full epoch, return mean MSE loss
  evaluate        – compute MSE and MAE on a DataLoader (no grad)
  train_model     – full training loop with early stopping
  finetune        – short fine-tuning pass after a pruning step
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from magnitude_pruning.pruning import apply_masks


# ---------------------------------------------------------------------------
# Single epoch
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
    mask_grads: bool = False,
    grad_handles: list | None = None,
) -> float:
    """Run one training epoch and return the mean MSE loss."""
    model.train()
    criterion = nn.MSELoss()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimiser.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        # Gradient clipping guards against exploding dynamics in sparse models
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        # Re-enforce the binary mask after the parameter update so the
        # optimiser cannot revive pruned weights through momentum accumulation.
        apply_masks(model)
        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Return {'mse': ..., 'mae': ...} on the given loader."""
    model.eval()
    mse_total = 0.0
    mae_total = 0.0
    n = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        mse_total += nn.functional.mse_loss(pred, y, reduction="sum").item()
        mae_total += nn.functional.l1_loss( pred, y, reduction="sum").item()
        n += y.numel()

    return {"mse": mse_total / n, "mae": mae_total / n}


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    device: torch.device,
    n_epochs: int   = 30,
    lr: float       = 1e-3,
    patience: int   = 7,
    verbose: bool   = True,
) -> dict:
    """
    Train 'model' from its current state with Adam + cosine LR annealing.

    Returns a history dict:
      {'train_mse': [...], 'val_mse': [...], 'val_mae': [...],
       'best_val_mse': float, 'best_epoch': int}
    """
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=n_epochs, eta_min=lr * 0.01
    )

    history = {"train_mse": [], "val_mse": [], "val_mae": []}
    best_val_mse = float("inf")
    best_state   = None
    no_improve   = 0

    pbar = tqdm(range(1, n_epochs + 1), desc="Training", disable=not verbose)
    for epoch in pbar:
        train_loss = train_epoch(model, train_loader, optimiser, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        history["train_mse"].append(train_loss)
        history["val_mse"].append(val_metrics["mse"])
        history["val_mae"].append(val_metrics["mae"])

        if val_metrics["mse"] < best_val_mse:
            best_val_mse = val_metrics["mse"]
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1

        pbar.set_postfix(
            train_mse=f"{train_loss:.5f}",
            val_mse=f"{val_metrics['mse']:.5f}",
        )

        if no_improve >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            break

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)

    history["best_val_mse"] = best_val_mse
    history["best_epoch"]   = len(history["train_mse"]) - no_improve
    return history


# ---------------------------------------------------------------------------
# Fine-tuning (post-pruning)
# ---------------------------------------------------------------------------

def finetune(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    device: torch.device,
    n_epochs: int = 5,
    lr: float     = 3e-4,
    verbose: bool = False,
) -> dict:
    """
    Short fine-tuning pass after a pruning step.  Same as train_model but
    with fewer epochs and a lower LR, keeping masks active throughout.
    """
    return train_model(
        model,
        train_loader,
        val_loader,
        device,
        n_epochs=n_epochs,
        lr=lr,
        patience=n_epochs,   # no early stopping for short fine-tune
        verbose=verbose,
    )
