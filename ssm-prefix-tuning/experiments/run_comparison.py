"""Trains all four fine-tuning methods on SST-2 with the same seed and prints
a Markdown comparison table: prefix, prefix+periodic, LoRA, full fine-tuning.
Parameter counts are shown alongside accuracy so the efficiency trade-off is clear.

Usage:
    python experiments/run_comparison.py --epochs 3
    python experiments/run_comparison.py --epochs 5 --output-dir results/comparison
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ssm_prefix_tuning.config import (
    LoraHyperparams,
    PeriodicInjectionConfig,
    PrefixConfig,
    TrainingConfig,
)
from ssm_prefix_tuning.data import load_sst2
from ssm_prefix_tuning.evaluator import compare_methods
from ssm_prefix_tuning.lora_model import build_lora_model, count_trainable_parameters
from ssm_prefix_tuning.model_wrapper import build_full_finetune_model, build_prefix_model
from ssm_prefix_tuning.trainer import EpochResult, run_training
from ssm_prefix_tuning.utils import save_results, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare prefix, LoRA, and full fine-tuning")
    p.add_argument("--model-name", default="state-spaces/mamba-130m-hf")
    p.add_argument("--prefix-length", type=int, default=10)
    p.add_argument("--period", type=int, default=64,
                   help="Periodic injection period for prefix_periodic variant")
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--prefix-lr", type=float, default=3e-4)
    p.add_argument("--lora-lr", type=float, default=3e-4)
    p.add_argument("--full-lr", type=float, default=2e-5,
                   help="Lower LR for full fine-tuning to avoid catastrophic forgetting")
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="results/comparison")
    p.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Subsample training set to at most N examples (useful on small GPUs).",
    )
    p.add_argument(
        "--device",
        default="auto",
        help='Compute device: "auto" (default), "cpu", "cuda", or "cuda:0" etc.',
    )
    return p.parse_args()


def _make_training_cfg(args, method: str, lr: float) -> TrainingConfig:
    return TrainingConfig(
        method=method,
        model_name=args.model_name,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=lr,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
        output_dir=os.path.join(args.output_dir, method),
        device=args.device,
    )


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    import torch
    if args.device == "auto":
        device_str = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    else:
        device_str = args.device
    # Write resolved device back so _make_training_cfg picks it up.
    args.device = device_str

    print(f"\nMethod comparison")
    print(f"  model  : {args.model_name}")
    print(f"  epochs : {args.epochs}  |  K={args.prefix_length}  |  LoRA r={args.lora_r}  |  device={device_str}")
    if args.max_train_samples:
        print(f"  train cap: {args.max_train_samples}")
    print(f"  output : {args.output_dir}\n")

    # Load data once — all four methods use the same tokenised dataset.
    train_loader, val_loader = load_sst2(
        model_name=args.model_name,
        max_length=args.max_seq_len,
        batch_size=args.batch_size,
        max_train_samples=args.max_train_samples,
    )

    all_results: dict[str, list[EpochResult]] = {}
    all_param_counts: dict[str, dict] = {}

    print("\nStandard prefix tuning")
    set_seed(args.seed)
    model = build_prefix_model(
        args.model_name,
        PrefixConfig(prefix_length=args.prefix_length),
        PeriodicInjectionConfig(enabled=False),
    )
    all_param_counts["prefix"] = count_trainable_parameters(model)
    results = run_training(model, train_loader, val_loader,
                           _make_training_cfg(args, "prefix", args.prefix_lr))
    all_results["prefix"] = results
    save_results(results, os.path.join(args.output_dir, "prefix", "epoch_results.json"))

    print("\nPrefix tuning with periodic injection")
    set_seed(args.seed)
    model = build_prefix_model(
        args.model_name,
        PrefixConfig(prefix_length=args.prefix_length),
        PeriodicInjectionConfig(enabled=True, period=args.period),
    )
    all_param_counts["prefix_periodic"] = count_trainable_parameters(model)
    results = run_training(model, train_loader, val_loader,
                           _make_training_cfg(args, "prefix_periodic", args.prefix_lr))
    all_results["prefix_periodic"] = results
    save_results(results, os.path.join(args.output_dir, "prefix_periodic", "epoch_results.json"))

    print("\nLoRA fine-tuning")
    set_seed(args.seed)
    model = build_lora_model(args.model_name, LoraHyperparams(r=args.lora_r))
    all_param_counts["lora"] = count_trainable_parameters(model)
    results = run_training(model, train_loader, val_loader,
                           _make_training_cfg(args, "lora", args.lora_lr))
    all_results["lora"] = results
    save_results(results, os.path.join(args.output_dir, "lora", "epoch_results.json"))

    print("\n[4/4] Full fine-tuning")
    set_seed(args.seed)
    model = build_full_finetune_model(args.model_name)
    all_param_counts["full"] = count_trainable_parameters(model)
    results = run_training(model, train_loader, val_loader,
                           _make_training_cfg(args, "full", args.full_lr))
    all_results["full"] = results
    save_results(results, os.path.join(args.output_dir, "full", "epoch_results.json"))

    print("\n[4/4] Full fine-tuning")
    df = compare_methods(all_results, all_param_counts)
    csv_path = os.path.join(args.output_dir, "comparison.csv")
    df.to_csv(csv_path, index=False)

    print("\nResults:")
    print(df.to_markdown(index=False))
    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
