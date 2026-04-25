"""Experiment: sweep prefix length K ∈ [1, 5, 10, 20, 50].

For each value of K, trains a fresh prefix-tuning model from the same random
seed and reports the best validation accuracy with 95 % bootstrap CIs.
Results are saved to a CSV and a matplotlib plot.

The sweep reveals how prefix length interacts with SSM fading memory:
  - Too small K → insufficient capacity to encode task information.
  - Larger K → more capacity, but farther from the first real token,
    increasing fading-memory attenuation.
  - Very large K → diminishing returns or even degradation on short sequences.

Usage:

    # Pipeline smoke test (CPU, ~30 s, no GPU / no pretrained weights needed):
python experiments/run_prefix_length_sweep.py --tiny --epochs 1 --max-train-samples 200 --ks 1 5

# Real run on GPU (auto-detect):
python experiments/run_prefix_length_sweep.py --epochs 3 --ks 1 5 10 20 50

# Force a specific device:
python experiments/run_prefix_length_sweep.py --device cuda --epochs 1 --max-train-samples 500
python experiments/run_prefix_length_sweep.py --device cpu  --tiny --epochs 1

# Periodic injection:
python experiments/run_prefix_length_sweep.py --periodic-injection --period 32
"""

from __future__ import annotations

import argparse
import gc
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from ssm_prefix_tuning.config import (
    PeriodicInjectionConfig,
    PrefixConfig,
    TrainingConfig,
)
from ssm_prefix_tuning.data import load_sst2
from ssm_prefix_tuning.evaluator import (
    plot_prefix_length_vs_accuracy,
    prefix_length_sweep_table,
)
from ssm_prefix_tuning.trainer import EpochResult, run_training
from ssm_prefix_tuning.utils import set_seed


DEFAULT_KS = [1, 5, 10, 20, 50]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prefix length sweep on Mamba SST-2")
    p.add_argument("--model-name", default="state-spaces/mamba-130m-hf")
    p.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KS,
                   help="List of prefix lengths to evaluate")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--periodic-injection", action="store_true")
    p.add_argument("--period", type=int, default=64)
    p.add_argument("--output-dir", default="results/prefix_sweep")
    p.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Subsample the training set to at most N examples (useful for quick CPU runs).",
    )
    p.add_argument(
        "--tiny",
        action="store_true",
        help=(
            "Use a randomly-initialised 2-layer Mamba (hidden=64) instead of "
            "the pretrained backbone. Runs entirely on CPU in seconds — useful "
            "for verifying the pipeline end-to-end without a GPU."
        ),
    )
    p.add_argument(
        "--device",
        default="auto",
        help='Compute device: "auto" (default), "cpu", "cuda", or "cuda:0" etc.',
    )
    return p.parse_args()


def main() -> None:
    from ssm_prefix_tuning.model_wrapper import (
        MambaClassifier,
        MambaPrefixModel,
        build_prefix_model_from_config,
    )
    from ssm_prefix_tuning.prefix_encoder import PeriodicPrefixInjector, PrefixEncoder

    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve the compute device once so it is printed before any slow I/O.
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"\n=== Prefix Length Sweep ===")
    print(f"K values : {args.ks}")
    print(f"Periodic : {args.periodic_injection} (period={args.period})")
    if args.max_train_samples:
        print(f"Train cap: {args.max_train_samples} examples")
    print(f"Device   : {device}")
    print(f"Tiny mode: {args.tiny}")
    print(f"Output   : {args.output_dir}\n")

    # Load data once; all runs share the same tokenised dataset.
    train_loader, val_loader = load_sst2(
        model_name=args.model_name,
        max_length=args.max_seq_len,
        batch_size=args.batch_size,
        max_train_samples=args.max_train_samples,
    )

    num_labels = 2

    if args.tiny:
        # Build a tiny random-weight model from config — no download needed.
        # vocab_size=50280 matches the real Mamba tokenizer so embedding indices
        # from the real tokenizer are always in range.
        from transformers import MambaConfig
        tiny_cfg = MambaConfig(hidden_size=64, num_hidden_layers=2, vocab_size=50280)
        print("Tiny mode: using randomly-initialised 2-layer Mamba (hidden=64)")
        shared_backbone = None  # not used in tiny mode
    else:
        # Load the backbone ONCE and reuse it across all K iterations.
        # Each iteration wraps the SAME underlying MambaModel weights in a fresh
        # MambaClassifier, so we only pay the ~500 MB download/load cost once.
        from transformers import MambaModel
        print("Loading backbone weights (one-time) …")
        shared_backbone = MambaModel.from_pretrained(args.model_name)
        tiny_cfg = None  # not used in normal mode

    sweep_results: dict[int, EpochResult] = {}

    for K in sorted(args.ks):
        print(f"\n--- K = {K} ---")
        # Reset seed for every K so results are comparable.
        set_seed(args.seed)

        prefix_cfg = PrefixConfig(prefix_length=K)
        injection_cfg = PeriodicInjectionConfig(
            enabled=args.periodic_injection,
            period=args.period,
        )
        cfg = TrainingConfig(
            method="prefix",
            model_name=args.model_name,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            max_seq_len=args.max_seq_len,
            seed=args.seed,
            output_dir=os.path.join(args.output_dir, f"K{K}"),
            device=str(device),
        )

        if args.tiny:
            # build_prefix_model_from_config sets prefix_cfg.hidden_size automatically.
            model = build_prefix_model_from_config(
                tiny_cfg, prefix_cfg, injection_cfg, num_labels
            )
        else:
            # Build classifier wrapping the shared backbone, then a fresh prefix encoder.
            classifier_model = MambaClassifier(shared_backbone, num_labels)
            # Set hidden_size to match the backbone so PrefixEncoder allocates correctly.
            prefix_cfg.hidden_size = shared_backbone.config.hidden_size
            enc = PrefixEncoder(prefix_cfg)
            injector = (
                PeriodicPrefixInjector(injection_cfg, enc)
                if injection_cfg.enabled
                else None
            )
            model = MambaPrefixModel(classifier_model, enc, injector, num_labels)

        epoch_results = run_training(model, train_loader, val_loader, cfg)
        best = max(epoch_results, key=lambda r: r.val_accuracy)
        sweep_results[K] = best
        print(
            f"  K={K}  best acc={best.val_accuracy:.4f} "
            f"[{best.ci_lower:.4f}, {best.ci_upper:.4f}]"
        )

        # Free the per-K model (prefix encoder + classifier head) to reclaim memory.
        # In normal mode the shared backbone tensors are kept alive via shared_backbone.
        del model
        if not args.tiny:
            del classifier_model, enc, injector
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save results table.
    df = prefix_length_sweep_table(sweep_results)
    csv_path = os.path.join(args.output_dir, "sweep_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSweep table saved to {csv_path}")
    print(df.to_markdown(index=False))

    # Save plot.
    plot_path = os.path.join(args.output_dir, "accuracy_vs_K.png")
    plot_prefix_length_vs_accuracy(sweep_results, plot_path)


if __name__ == "__main__":
    main()
