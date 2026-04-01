"""
Plotting utilities for IMP results.

All functions accept the dict returned by experiment.run_experiment()
and produce matplotlib figures.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


# ---------------------------------------------------------------------------
# Sparsity vs MSE curve (main result)
# ---------------------------------------------------------------------------

def plot_sparsity_vs_mse(
    results: dict,
    ax: plt.Axes | None = None,
    show_anomaly: bool  = True,
) -> plt.Figure:
    """
    Plots val_mse and (optionally) anomaly test_mse against sparsity with
    +-1σ confidence bands across seeds.

    A vertical dashed line marks the median breaking point.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.figure

    s = results["sparsities"] * 100   # → percentage

    # Dense baseline reference line
    baseline_mean = float(np.mean(results["baseline_val_mse"]))
    ax.axhline(baseline_mean, color="green", linestyle=":", linewidth=1.5,
               label=f"Dense baseline MSE ({baseline_mean:.4f})", zorder=3)

    # Val MSE curve
    _plot_band(ax, s, results["val_mse_mean"], results["val_mse_std"],
               color="steelblue", label="Val MSE (+-1σ)")

    # Anomaly test MSE curve
    if show_anomaly:
        _plot_band(ax, s, results["test_mse_anomaly_mean"],
                   results["test_mse_anomaly_std"],
                   color="tomato", label="Test MSE w/ anomalies (+-1σ)", alpha=0.2)

    # Breaking-point line (median across seeds)
    bps = [b for b in results["breaking_points"] if b is not None]
    if bps:
        median_bp = float(np.median(bps)) * 100
        ax.axvline(median_bp, color="red", linestyle="--", linewidth=2,
                   label=f"Breaking point ≈ {median_bp:.1f}%")

    ax.set_xlabel("Sparsity (%)")
    ax.set_ylabel("MSE")
    ax.set_title("Sparsity vs. MSE  ·  Iterative Magnitude Pruning on Mamba SSM")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def _plot_band(ax, x, mean, std, color, label, alpha=0.15):
    ax.plot(x, mean, color=color, linewidth=2, label=label, zorder=4)
    ax.fill_between(x, mean - std, mean + std,
                    color=color, alpha=alpha, zorder=2)


# ---------------------------------------------------------------------------
# Robustness degradation bar chart
# ---------------------------------------------------------------------------

def plot_robustness_degradation(
    results: dict,
    sparsity_checkpoints: list[float] = (0.0, 0.5, 0.8, 0.9, 0.95),
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    For a set of representative sparsity levels, shows:
      - Clean test MSE
      - Anomaly test MSE
    as grouped bars to visualise how robustness degrades with sparsity.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    sparsities = results["sparsities"]
    # Find the closest index for each checkpoint
    idxs = [int(np.argmin(np.abs(sparsities - cp))) for cp in sparsity_checkpoints]

    labels  = [f"{sparsities[i]*100:.0f}%" for i in idxs]
    clean   = [results["test_mse_mean"][i]         for i in idxs]
    anomaly = [results["test_mse_anomaly_mean"][i]  for i in idxs]

    x   = np.arange(len(labels))
    w   = 0.35
    ax.bar(x - w/2, clean,   width=w, label="Clean test MSE",   color="steelblue")
    ax.bar(x + w/2, anomaly, width=w, label="Anomaly test MSE", color="tomato")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Sparsity")
    ax.set_ylabel("MSE")
    ax.set_title("Robustness Degradation: Clean vs. Anomaly MSE at Key Sparsity Levels")
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Training curves (dense baseline)
# ---------------------------------------------------------------------------

def plot_training_curves(
    results: dict,
    seed_idx: int = 0,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot the dense-baseline training/validation loss curves for one seed."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    hist = results["per_seed"][seed_idx]["train_history"]
    epochs = range(1, len(hist["train_mse"]) + 1)
    ax.plot(epochs, hist["train_mse"], label="Train MSE", color="steelblue")
    ax.plot(epochs, hist["val_mse"],   label="Val MSE",   color="darkorange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Dense Baseline Training Curves")
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Summary dashboard (all plots on one figure)
# ---------------------------------------------------------------------------

def plot_dashboard(results: dict) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(21, 5))
    plot_training_curves(results, ax=axes[0])
    plot_sparsity_vs_mse(results, ax=axes[1])
    plot_robustness_degradation(results, ax=axes[2])
    fig.suptitle(
        "IMP on Mamba SSM — Mackey-Glass Forecasting",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    return fig
