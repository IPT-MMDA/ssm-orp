"""Experiment: LoRA fine-tuning on Mamba SSM for SST-2.

Injects LoRA adapters into the in_proj, out_proj, and x_proj linear layers
of each MambaMixer block.  All other weights (including the SSM recurrence
parameters A, B, C, Δ) remain frozen.

Usage examples:

    # Default LoRA r=8
python experiments/run_lora.py

# Higher rank
python experiments/run_lora.py --lora-r 16 --lora-alpha 32
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ssm_prefix_tuning.config import LoraHyperparams, TrainingConfig
from ssm_prefix_tuning.data import load_sst2
from ssm_prefix_tuning.lora_model import build_lora_model, count_trainable_parameters
from ssm_prefix_tuning.trainer import run_training
from ssm_prefix_tuning.utils import save_results, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA fine-tuning on Mamba for SST-2")
    p.add_argument("--model-name", default="state-spaces/mamba-130m-hf")
    p.add_argument("--lora-r", type=int, default=8, help="LoRA rank r")
    p.add_argument("--lora-alpha", type=int, default=16, help="LoRA scaling alpha")
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="results/lora")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = os.path.join(args.output_dir, f"r{args.lora_r}")
    cfg = TrainingConfig(
        method="lora",
        model_name=args.model_name,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
        output_dir=output_dir,
    )
    lora_cfg = LoraHyperparams(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    print(f"\n=== LoRA Fine-Tuning (r={args.lora_r}, α={args.lora_alpha}) ===")
    print(f"Model : {args.model_name}")
    print(f"Output: {output_dir}\n")

    model = build_lora_model(args.model_name, lora_cfg)
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
    save_results(results, os.path.join(output_dir, "epoch_results.json"))

    best = max(results, key=lambda r: r.val_accuracy)
    print(
        f"\nBest result: epoch {best.epoch}  "
        f"acc={best.val_accuracy:.4f} [{best.ci_lower:.4f}, {best.ci_upper:.4f}]  "
        f"f1={best.val_f1:.4f}"
    )


if __name__ == "__main__":
    main()
