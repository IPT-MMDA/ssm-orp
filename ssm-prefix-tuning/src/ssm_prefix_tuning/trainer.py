"""Unified training and evaluation loop.

The same loop works for all three methods (prefix, LoRA, full fine-tuning)
because each produces a model with the following interface:

    model(input_ids, attention_mask, labels) → SequenceClassifierOutput

The trainer never inspects which method it is using.  The only method-aware
decision is made in build_optimizer: it automatically filters to parameters
with requires_grad=True, so:
  - Prefix tuning → only prefix encoder + classifier weights are updated.
  - LoRA          → only LoRA adapter matrices + classifier.
  - Full FT       → all weights.

Gradient clipping (max_norm=1.0) is applied in every method.  This is
particularly important for prefix tuning because the prefix parameters start
from a random initialisation and can produce large gradient magnitudes in
the first few steps.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import evaluate as hf_evaluate

from ssm_prefix_tuning.config import EpochResult, TrainingConfig

# Load metric objects once at module level to avoid re-downloading / re-initialising
# on every call to evaluate().  hf_evaluate metrics are stateless between compute()
# calls so sharing a single instance is safe.
_acc_metric = hf_evaluate.load("accuracy")
_f1_metric = hf_evaluate.load("f1")
from ssm_prefix_tuning.evaluator import bootstrap_ci


def build_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    """Build an AdamW optimiser over only the trainable parameters.

    Filtering to requires_grad=True ensures no unnecessary gradient
    state is allocated for frozen weights, saving memory and compute.
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> float:
    """Run one full pass over the training set.

    Args:
        model:     Any model with a forward(input_ids, attention_mask, labels)
                   interface that returns a SequenceClassifierOutput.
        loader:    Training DataLoader.
        optimizer: Pre-built optimiser (see build_optimizer).
        device:    Compute device.
        scaler:    Optional GradScaler for fp16/bf16 mixed-precision training.
                   Pass None to use fp32.

    Returns:
        Mean training loss over all batches.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc="  train", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.autocast(device_type=device.type):
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            scaler.scale(out.loss).backward()
            # Unscale before clipping so the clip threshold is in the original
            # gradient units, not the scaled ones.
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()

        total_loss += out.loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate the model on a DataLoader.

    Returns:
        dict with keys "accuracy" and "f1" (macro-averaged).
    """
    model.eval()

    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  eval ", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = out.logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    accuracy = _acc_metric.compute(predictions=all_preds, references=all_labels)["accuracy"]
    f1 = _f1_metric.compute(
        predictions=all_preds,
        references=all_labels,
        average="macro",
    )["f1"]
    return {"accuracy": accuracy, "f1": f1, "predictions": all_preds, "references": all_labels}


def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
) -> list[EpochResult]:
    """Train for config.num_epochs and return per-epoch metrics.

    The best checkpoint (by val_accuracy) is saved to config.output_dir.

    Args:
        model:        Any model compatible with the training interface.
        train_loader: DataLoader for training data.
        val_loader:   DataLoader for validation data.
        config:       TrainingConfig with lr, epochs, output_dir, etc.

    Returns:
        List of EpochResult, one per epoch, in order.
    """
    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)
    model = model.to(device)

    optimizer = build_optimizer(model, config.learning_rate, config.weight_decay)

    # Use mixed-precision on CUDA to accelerate training.
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    os.makedirs(config.output_dir, exist_ok=True)

    results: list[EpochResult] = []
    best_acc = 0.0
    best_path = os.path.join(config.output_dir, "best_checkpoint.pt")

    for epoch in range(1, config.num_epochs + 1):
        print(f"\nEpoch {epoch}/{config.num_epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler)
        eval_out = evaluate(model, val_loader, device)

        preds = eval_out["predictions"]
        refs = eval_out["references"]

        import numpy as np
        ci_lo, ci_hi = bootstrap_ci(
            labels=np.array(refs),
            preds=np.array(preds),
            metric_fn=lambda y, p: (y == p).mean(),
            n_bootstrap=config.bootstrap_n,
            ci=config.bootstrap_ci,
        )

        result = EpochResult(
            epoch=epoch,
            train_loss=round(train_loss, 4),
            val_accuracy=round(eval_out["accuracy"], 4),
            val_f1=round(eval_out["f1"], 4),
            ci_lower=round(ci_lo, 4),
            ci_upper=round(ci_hi, 4),
        )
        results.append(result)

        print(
            f"  loss={result.train_loss:.4f}  "
            f"acc={result.val_accuracy:.4f} "
            f"[{result.ci_lower:.4f}, {result.ci_upper:.4f}]  "
            f"f1={result.val_f1:.4f}"
        )

        # Save only the trainable parameters — for prefix tuning this is a few KB
        # instead of the full ~500 MB backbone state dict.
        if result.val_accuracy > best_acc:
            best_acc = result.val_accuracy
            trainable_state = {
                n: p.data for n, p in model.named_parameters() if p.requires_grad
            }
            torch.save(trainable_state, best_path)
            print(f"  ✓ New best checkpoint saved ({best_acc:.4f})")

    return results
