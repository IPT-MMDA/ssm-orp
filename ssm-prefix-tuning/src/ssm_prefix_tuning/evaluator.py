"""Post-training evaluation utilities.

Provides:
  - bootstrap_ci      — non-parametric confidence intervals for any metric.
  - compare_methods   — builds a summary DataFrame from training histories.
  - prefix_length_sweep_table — formats K-sweep results as a DataFrame.
  - plot_prefix_length_vs_accuracy — saves an accuracy-vs-K line plot.

All functions are pure (no side effects beyond file I/O in the plot function)
and operate on plain Python / NumPy / pandas data structures so they can be
easily tested and reused.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def bootstrap_ci(
    labels: np.ndarray,
    preds: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute a bootstrap confidence interval for a metric.

    Draws n_bootstrap samples with replacement from (labels, preds), computes
    metric_fn for each sample, and returns the (lower, upper) percentile bounds
    at the requested confidence level.

    Args:
        labels:      Ground-truth integer labels, shape [N].
        preds:       Model predictions, shape [N].
        metric_fn:   Function (labels, preds) → float.  Must accept two NumPy
                     arrays and return a scalar.
        n_bootstrap: Number of bootstrap resamples.
        ci:          Confidence level, e.g. 0.95 for a 95 % CI.
        seed:        Random seed for reproducibility.

    Returns:
        (lower_bound, upper_bound) as floats.
    """
    rng = np.random.default_rng(seed)
    N = len(labels)
    scores = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        idx = rng.integers(0, N, size=N)
        scores[i] = metric_fn(labels[idx], preds[idx])

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(scores, 100.0 * alpha))
    upper = float(np.percentile(scores, 100.0 * (1.0 - alpha)))
    return lower, upper


def compare_methods(
    results: dict[str, list],
    param_counts: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """Build a side-by-side comparison DataFrame from per-method training logs.

    Args:
        results:      Mapping from method name to list of EpochResult objects
                      (e.g. {"prefix": [...], "lora": [...], "full": [...]}).
        param_counts: Optional mapping from method name to the dict returned
                      by count_trainable_parameters.  If provided, the
                      "trainable_params" column is populated.

    Returns:
        DataFrame with one row per method and columns:
            method, best_val_acc, best_val_f1, ci_lower, ci_upper,
            trainable_params, epochs_to_best
    """
    rows = []
    for method, epoch_results in results.items():
        if not epoch_results:
            continue
        best = max(epoch_results, key=lambda r: r.val_accuracy)
        row = {
            "method": method,
            "best_val_acc": best.val_accuracy,
            "best_val_f1": best.val_f1,
            "ci_lower": best.ci_lower,
            "ci_upper": best.ci_upper,
            "epochs_to_best": best.epoch,
            "trainable_params": (
                param_counts[method]["trainable"] if param_counts and method in param_counts else None
            ),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("best_val_acc", ascending=False).reset_index(drop=True)
    return df


def prefix_length_sweep_table(
    sweep_results: dict[int, object],
) -> pd.DataFrame:
    """Format {K: best_EpochResult} into a clean DataFrame sorted by K.

    Args:
        sweep_results: Mapping from prefix length K (int) to the best
                       EpochResult achieved for that K.

    Returns:
        DataFrame with columns: prefix_length, val_accuracy, val_f1,
        ci_lower, ci_upper.  Rows are sorted by prefix_length ascending.
    """
    rows = []
    for K, result in sorted(sweep_results.items()):
        rows.append(
            {
                "prefix_length": K,
                "val_accuracy": result.val_accuracy,
                "val_f1": result.val_f1,
                "ci_lower": result.ci_lower,
                "ci_upper": result.ci_upper,
            }
        )
    return pd.DataFrame(rows)


def plot_prefix_length_vs_accuracy(
    sweep_results: dict[int, object],
    output_path: str,
) -> None:
    """Save a line plot of validation accuracy vs prefix length K.

    Error bars represent the 95 % bootstrap confidence interval bounds.
    The plot is saved to output_path (any format supported by matplotlib,
    e.g. .png, .pdf).

    Args:
        sweep_results: {K: best_EpochResult} as returned by the sweep loop.
        output_path:   File path for the saved figure.
    """
    import os
    import matplotlib.pyplot as plt

    df = prefix_length_sweep_table(sweep_results)

    K_values = df["prefix_length"].tolist()
    accs = df["val_accuracy"].tolist()
    err_lower = [a - lo for a, lo in zip(accs, df["ci_lower"].tolist())]
    err_upper = [hi - a for a, hi in zip(accs, df["ci_upper"].tolist())]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(
        K_values,
        accs,
        yerr=[err_lower, err_upper],
        fmt="o-",
        capsize=4,
        linewidth=1.5,
        markersize=5,
        color="steelblue",
    )
    ax.set_xlabel("Prefix length K", fontsize=12)
    ax.set_ylabel("Validation accuracy", fontsize=12)
    ax.set_title("Effect of prefix length on SST-2 accuracy (Mamba SSM)", fontsize=12)
    ax.set_xticks(K_values)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {output_path}")
