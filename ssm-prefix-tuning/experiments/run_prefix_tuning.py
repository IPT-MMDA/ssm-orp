"""Experiment: prefix tuning on Mamba SSM for SST-2.

Trains a frozen Mamba backbone with K trainable prefix vectors.
Two variants are supported via --periodic-injection:
  - Standard: single prefix prepended at position 0.
  - Periodic:  prefix re-injected every --period tokens (mitigates fading memory).

Usage examples:

    # Standard prefix, K=10, 5 epochs
python experiments/run_prefix_tuning.py --prefix-length 10 --epochs 5

# With periodic injection every 32 tokens
python experiments/run_prefix_tuning.py --prefix-length 10 --periodic-injection --period 32

# Use projected parameterisation (Li & Liang 2021)
python experiments/run_prefix_tuning.py --prefix-length 10 --projection
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Make the src package importable when running from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ssm_prefix_tuning.config import (
    PeriodicInjectionConfig,
    PrefixConfig,
    TrainingConfig,
)
from ssm_prefix_tuning.data import load_sst2
from ssm_prefix_tuning.evaluator import compare_methods
from ssm_prefix_tuning.lora_model import count_trainable_parameters
from ssm_prefix_tuning.model_wrapper import build_prefix_model
from ssm_prefix_tuning.trainer import run_training
from ssm_prefix_tuning.utils import save_results, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prefix tuning on Mamba for SST-2")
    p.add_argument("--model-name", default="state-spaces/mamba-130m-hf")
    p.add_argument("--prefix-length", type=int, default=10, help="Number of prefix vectors K")
    p.add_argument("--projection", action="store_true",
                   help="Use MLP reparameterisation for prefix (Li & Liang 2021)")
    p.add_argument("--periodic-injection", action="store_true",
                   help="Re-inject prefix every --period tokens to mitigate fading memory")
    p.add_argument("--period", type=int, default=64,
                   help="Token interval between prefix re-injections (ignored without --periodic-injection)")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="results/prefix_tuning")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    method_tag = "prefix_periodic" if args.periodic_injection else "prefix"
    output_dir = os.path.join(args.output_dir, f"K{args.prefix_length}_{method_tag}")

    cfg = TrainingConfig(
        method=method_tag,
        model_name=args.model_name,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
        output_dir=output_dir,
    )
    prefix_cfg = PrefixConfig(
        prefix_length=args.prefix_length,
        projection=args.projection,
    )
    injection_cfg = PeriodicInjectionConfig(
        enabled=args.periodic_injection,
        period=args.period,
    )

    print(f"\n=== Prefix Tuning (K={args.prefix_length}, periodic={args.periodic_injection}) ===")
    print(f"Model : {args.model_name}")
    print(f"Output: {output_dir}\n")

    model = build_prefix_model(args.model_name, prefix_cfg, injection_cfg)
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
