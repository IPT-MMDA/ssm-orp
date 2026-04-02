"""Reads saved checkpoints and logged metrics, then produces three figures.

No training happens here — just loading weights, running inference, and plotting.

Outputs:
  results/figures/comparison_bar_chart.png  — best accuracy for all 4 methods
  results/figures/training_curves.png       — accuracy + loss over epochs
  results/figures/k_sweep.png               — accuracy vs prefix length K

Usage (from ssm-prefix-tuning/):
  python experiments/analyze.py                 # evaluate missing K checkpoints + plot
  python experiments/analyze.py --skip-eval     # plot only, using existing sweep_results.csv
  python experiments/analyze.py --device cpu    # force CPU (slower but works without GPU)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ssm_prefix_tuning.config import PeriodicInjectionConfig, PrefixConfig
from ssm_prefix_tuning.data import load_sst2
from ssm_prefix_tuning.evaluator import bootstrap_ci
from ssm_prefix_tuning.model_wrapper import MambaClassifier, MambaPrefixModel
from ssm_prefix_tuning.prefix_encoder import PrefixEncoder
from ssm_prefix_tuning.trainer import evaluate

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESULTS = os.path.join(_HERE, "..", "results")
COMPARISON_DIR = os.path.join(_RESULTS, "comparison")
SWEEP_DIR = os.path.join(_RESULTS, "prefix_sweep")
FIGURES_DIR = os.path.join(_RESULTS, "figures")

MODEL_NAME = "state-spaces/mamba-130m-hf"
KS = [1, 5, 10, 20, 50]
METHODS = ["lora", "full", "prefix_periodic", "prefix"]

# Same colours and display names used in every figure so the legend is consistent.
_COLOURS = {
    "lora": "#4C72B0",
    "full": "#55A868",
    "prefix_periodic": "#C44E52",
    "prefix": "#DD8452",
}
_LABELS = {
    "lora": "LoRA (r=8)",
    "full": "Full fine-tuning",
    "prefix_periodic": "Prefix + periodic",
    "prefix": "Prefix (K=10)",
}


def _fmt_params(n: int) -> str:
    """Turn a raw parameter count into something readable like '129M' or '9.2K'."""
    if n >= 1_000_000:
        v = n / 1_000_000
        return f"{v:.0f}M" if v >= 100 else f"{v:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def evaluate_sweep_checkpoints(device: torch.device) -> None:
    """Load each saved K checkpoint and measure its accuracy on the SST-2 validation set.

    Results are written to sweep_results.csv. K values that are already in the
    CSV are skipped so you can resume a partial run without redoing work.
    """
    csv_path = os.path.join(SWEEP_DIR, "sweep_results.csv")
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        done_ks: set[int] = set(int(k) for k in existing["prefix_length"].tolist())
    else:
        existing = pd.DataFrame(columns=["prefix_length", "val_accuracy", "val_f1", "ci_lower", "ci_upper"])
        done_ks = set()

    missing_ks = [k for k in KS if k not in done_ks]
    if not missing_ks:
        print("All K-sweep checkpoints already evaluated — nothing to do.")
        return

    print(f"Evaluating checkpoints for K = {missing_ks} ...")

    # The backbone is heavy (~130M params), so we load it once and reuse it for every K.
    print("  Loading Mamba-130M backbone (may download on first run) ...")
    classifier_model = MambaClassifier.from_pretrained(MODEL_NAME)
    classifier_model = classifier_model.to(device)

    # We only need the validation split here — no training.
    # max_train_samples=1 prevents loading the full 67K training set into memory.
    print("  Loading SST-2 validation set ...")
    _, val_loader = load_sst2(
        MODEL_NAME,
        max_length=64,
        batch_size=16,
        max_train_samples=1,
    )

    new_rows: list[dict] = []
    for k in missing_ks:
        ckpt_path = os.path.join(SWEEP_DIR, f"K{k}", "best_checkpoint.pt")
        if not os.path.exists(ckpt_path):
            print(f"  WARNING: checkpoint missing for K={k} ({ckpt_path}), skipping.")
            continue

        print(f"  K={k} ...")

        # Build a fresh prefix encoder for this K, then wrap the shared backbone.
        prefix_cfg = PrefixConfig(
            prefix_length=k,
            hidden_size=classifier_model.backbone.config.hidden_size,
        )
        enc = PrefixEncoder(prefix_cfg)
        model = MambaPrefixModel(classifier_model, enc, injector=None, num_labels=2)
        model = model.to(device)

        # The checkpoint only stores trainable params (prefix + classifier head),
        # not the frozen backbone, so strict=False lets the rest be filled from
        # the already-loaded backbone.
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=False)

        eval_out = evaluate(model, val_loader, device)

        preds = np.array(eval_out["predictions"])
        refs = np.array(eval_out["references"])
        ci_lo, ci_hi = bootstrap_ci(
            labels=refs,
            preds=preds,
            metric_fn=lambda y, p: (y == p).mean(),
        )

        row = {
            "prefix_length": k,
            "val_accuracy": round(float(eval_out["accuracy"]), 4),
            "val_f1": round(float(eval_out["f1"]), 4),
            "ci_lower": round(ci_lo, 4),
            "ci_upper": round(ci_hi, 4),
        }
        new_rows.append(row)
        print(f"    acc={row['val_accuracy']:.4f}  f1={row['val_f1']:.4f}  "
              f"CI=[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]")

    merged = (
        pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
        .sort_values("prefix_length")
        .reset_index(drop=True)
    )
    merged.to_csv(csv_path, index=False)
    print(f"  Saved updated {csv_path}")


def load_comparison_data() -> tuple[pd.DataFrame, dict[str, list[dict]]]:
    """Read the comparison summary CSV and per-method epoch logs."""
    comp_df = pd.read_csv(os.path.join(COMPARISON_DIR, "comparison.csv"))

    histories: dict[str, list[dict]] = {}
    for method in METHODS:
        path = os.path.join(COMPARISON_DIR, method, "epoch_results.json")
        if os.path.exists(path):
            with open(path) as f:
                histories[method] = json.load(f)
    return comp_df, histories


def plot_comparison_bar(comp_df: pd.DataFrame) -> None:
    """Horizontal bar chart showing best validation accuracy for all four methods.

    Bars are sorted so the best result appears at the top. Error bars come from
    the 95% bootstrap CI. Each bar is annotated with how many parameters were
    actually trained, so the accuracy-vs-efficiency trade-off is immediately visible.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)

    df = comp_df.sort_values("best_val_acc", ascending=True).reset_index(drop=True)
    methods = df["method"].tolist()
    accs = df["best_val_acc"].tolist()
    err_lo = [a - lo for a, lo in zip(accs, df["ci_lower"].tolist())]
    err_hi = [hi - a for a, hi in zip(accs, df["ci_upper"].tolist())]
    params = df["trainable_params"].tolist()

    bar_colours = [_COLOURS.get(m, "#888888") for m in methods]
    tick_labels = [_LABELS.get(m, m) for m in methods]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(
        tick_labels,
        accs,
        xerr=[err_lo, err_hi],
        color=bar_colours,
        capsize=4,
        height=0.5,
        error_kw={"elinewidth": 1.4, "ecolor": "black", "capthick": 1.4},
    )

    x_offset = max(err_hi) * 1.5 + 0.002
    for bar, p in zip(bars, params):
        ax.text(
            bar.get_width() + x_offset,
            bar.get_y() + bar.get_height() / 2,
            _fmt_params(int(p)),
            va="center", ha="left", fontsize=9, color="#333333",
        )

    ax.axvline(0.5, linestyle="--", linewidth=1.0, color="grey", alpha=0.8, label="Random chance (0.5)")
    ax.set_xlabel("Validation accuracy (SST-2)", fontsize=11)
    ax.set_title("Best validation accuracy by fine-tuning method\n(Mamba-130M, SST-2, 3 epochs)", fontsize=11)
    ax.set_xlim(0.43, 0.78)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()

    out = os.path.join(FIGURES_DIR, "comparison_bar_chart.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_training_curves(histories: dict[str, list[dict]]) -> None:
    """Two-panel plot: validation accuracy on the left, training loss on the right.

    The dashed line at 0.5 in the accuracy panel makes it easy to see that
    prefix tuning never learns anything useful — it just flatlines at chance level.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(11, 4))

    for method in METHODS:
        epochs = histories.get(method)
        if not epochs:
            continue
        xs = [e["epoch"] for e in epochs]
        accs = [e["val_accuracy"] for e in epochs]
        losses = [e["train_loss"] for e in epochs]
        colour = _COLOURS.get(method, "#888888")
        label = _LABELS.get(method, method)

        ax_acc.plot(xs, accs, "o-", color=colour, label=label, linewidth=1.8, markersize=5)
        ax_loss.plot(xs, losses, "o-", color=colour, label=label, linewidth=1.8, markersize=5)

    ax_acc.axhline(0.5, linestyle="--", linewidth=1.0, color="grey", alpha=0.8, label="Random chance")
    ax_acc.set_xlabel("Epoch", fontsize=11)
    ax_acc.set_ylabel("Validation accuracy", fontsize=11)
    ax_acc.set_title("Validation accuracy per epoch", fontsize=11)
    ax_acc.set_xticks([1, 2, 3])
    ax_acc.legend(fontsize=8)
    ax_acc.grid(True, linestyle="--", alpha=0.4)

    ax_loss.set_xlabel("Epoch", fontsize=11)
    ax_loss.set_ylabel("Training loss", fontsize=11)
    ax_loss.set_title("Training loss per epoch", fontsize=11)
    ax_loss.set_xticks([1, 2, 3])
    ax_loss.legend(fontsize=8)
    ax_loss.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "training_curves.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_k_sweep(sweep_csv: str) -> None:
    """Line plot of validation accuracy vs prefix length K, with 95% CI error bars."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    df = pd.read_csv(sweep_csv).sort_values("prefix_length")
    ks = df["prefix_length"].tolist()
    accs = df["val_accuracy"].tolist()
    err_lo = [a - lo for a, lo in zip(accs, df["ci_lower"].tolist())]
    err_hi = [hi - a for a, hi in zip(accs, df["ci_upper"].tolist())]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(
        ks, accs,
        yerr=[err_lo, err_hi],
        fmt="o-",
        capsize=4,
        linewidth=1.8,
        markersize=6,
        color="steelblue",
        label="Prefix tuning",
    )
    ax.axhline(0.5, linestyle="--", linewidth=1.0, color="grey", alpha=0.8, label="Random chance (0.5)")
    ax.set_xlabel("Prefix length K", fontsize=11)
    ax.set_ylabel("Validation accuracy", fontsize=11)
    ax.set_title("Effect of prefix length K on SST-2 accuracy\n(Mamba-130M, 3 epochs)", fontsize=11)
    ax.set_xticks(ks)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    out = os.path.join(FIGURES_DIR, "k_sweep.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def print_comparison_table(comp_df: pd.DataFrame) -> None:
    """Print a Markdown comparison table to stdout."""
    df = comp_df.sort_values("best_val_acc", ascending=False).reset_index(drop=True)
    print("\n## Method comparison (Mamba-130M, SST-2, 3 epochs)\n")
    print("| Method | Best val acc | Best F1 | 95% CI | Trainable params | Epochs to best |")
    print("|--------|-------------|---------|--------|-----------------|----------------|")
    for _, row in df.iterrows():
        ci = f"[{row.ci_lower:.4f}, {row.ci_upper:.4f}]"
        print(
            f"| {_LABELS.get(row.method, row.method)} "
            f"| {row.best_val_acc:.4f} "
            f"| {row.best_val_f1:.4f} "
            f"| {ci} "
            f"| {_fmt_params(int(row.trainable_params))} "
            f"| {int(row.epochs_to_best)} |"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse SSM prefix-tuning results.")
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"],
        help="Compute device for checkpoint evaluation (default: auto).",
    )
    parser.add_argument(
        "--skip-eval", action="store_true",
        help="Skip checkpoint evaluation; use existing sweep_results.csv as-is.",
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}\n")

    if not args.skip_eval:
        evaluate_sweep_checkpoints(device)

    comp_df, histories = load_comparison_data()
    sweep_csv = os.path.join(SWEEP_DIR, "sweep_results.csv")

    plot_comparison_bar(comp_df)
    plot_training_curves(histories)
    plot_k_sweep(sweep_csv)

    print_comparison_table(comp_df)
    print(f"All figures saved to {os.path.abspath(FIGURES_DIR)}/")


if __name__ == "__main__":
    main()
