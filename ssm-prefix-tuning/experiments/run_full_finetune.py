"""Experiment: full fine-tuning on Mamba SSM for SST-2.

Every weight in the backbone and the classification head is updated.
This serves as the upper-bound baseline for prefix tuning and LoRA comparisons.

Usage:

    python experiments/run_full_finetune.py
python experiments/run_full_finetune.py --epochs 3 --lr 1e-5
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ssm_prefix_tuning.config import TrainingConfig
from ssm_prefix_tuning.data import load_sst2
from ssm_prefix_tuning.lora_model import count_trainable_parameters
from ssm_prefix_tuning.model_wrapper import build_full_finetune_model
from ssm_prefix_tuning.trainer import run_training
from ssm_prefix_tuning.utils import save_results, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full fine-tuning on Mamba for SST-2")
    p.add_argument("--model-name", default="state-spaces/mamba-130m-hf")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    # Full fine-tuning uses a lower LR to avoid catastrophic forgetting.
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="results/full_finetune")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    cfg = TrainingConfig(
        method="full",
        model_name=args.model_name,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    print(f"\n=== Full Fine-Tuning ===")
    print(f"Model : {args.model_name}")
    print(f"Output: {args.output_dir}\n")

    model = build_full_finetune_model(args.model_name)
    param_info = count_trainable_parameters(model)
    print(
        f"Parameters — total: {param_info['total']:,}  "
        f"trainable: {param_info['trainable']:,}  "
        f"({param_info['trainable_ratio']*100:.3f} %)"
    )

    train_loader, val_loader = load_sst2(
        model_name=args.model_name,
        max_length=args.max_seq_len,
        batch_size=args.batch_size,
    )

    results = run_training(model, train_loader, val_loader, cfg)
    save_results(results, os.path.join(args.output_dir, "epoch_results.json"))

    best = max(results, key=lambda r: r.val_accuracy)
    print(
        f"\nBest result: epoch {best.epoch}  "
        f"acc={best.val_accuracy:.4f} [{best.ci_lower:.4f}, {best.ci_upper:.4f}]  "
        f"f1={best.val_f1:.4f}"
    )


if __name__ == "__main__":
    main()
