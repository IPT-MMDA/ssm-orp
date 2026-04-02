"""Tests for evaluator.py.

All tests run without model downloads or GPU.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ssm_prefix_tuning.evaluator import (
    bootstrap_ci,
    compare_methods,
    plot_prefix_length_vs_accuracy,
    prefix_length_sweep_table,
)
from ssm_prefix_tuning.config import EpochResult


# bootstrap_ci

def _accuracy(labels: np.ndarray, preds: np.ndarray) -> float:
    return float((labels == preds).mean())


def test_bootstrap_ci_returns_tuple_of_two():
    labels = np.array([0, 1, 0, 1, 0])
    preds  = np.array([0, 1, 0, 0, 1])
    result = bootstrap_ci(labels, preds, _accuracy, n_bootstrap=100, ci=0.95)
    assert len(result) == 2


def test_bootstrap_ci_lower_leq_upper():
    labels = np.array([0, 1, 0, 1, 0, 1])
    preds  = np.array([0, 1, 0, 1, 0, 1])
    lo, hi = bootstrap_ci(labels, preds, _accuracy, n_bootstrap=200)
    assert lo <= hi


def test_bootstrap_ci_bounds_in_unit_interval():
    labels = np.array([0, 1, 0, 1])
    preds  = np.array([0, 0, 0, 1])
    lo, hi = bootstrap_ci(labels, preds, _accuracy, n_bootstrap=200)
    assert 0.0 <= lo <= 1.0
    assert 0.0 <= hi <= 1.0


def test_bootstrap_ci_width_decreases_with_more_samples():
    """With more data, the CI should generally be narrower (not strictly, but on average)."""
    rng = np.random.default_rng(42)
    # Small dataset: 20 samples
    small_labels = rng.integers(0, 2, 20)
    small_preds  = rng.integers(0, 2, 20)
    # Large dataset: 500 samples
    large_labels = rng.integers(0, 2, 500)
    large_preds  = rng.integers(0, 2, 500)

    lo_s, hi_s = bootstrap_ci(small_labels, small_preds, _accuracy, n_bootstrap=500, seed=0)
    lo_l, hi_l = bootstrap_ci(large_labels, large_preds, _accuracy, n_bootstrap=500, seed=0)
    assert (hi_s - lo_s) > (hi_l - lo_l), (
        "CI for larger dataset should be narrower than for smaller dataset"
    )


def test_bootstrap_ci_perfect_predictions_tight_interval():
    """Perfect predictions should give a CI close to [1.0, 1.0]."""
    n = 200
    labels = np.array([i % 2 for i in range(n)])
    preds  = labels.copy()
    lo, hi = bootstrap_ci(labels, preds, _accuracy, n_bootstrap=500, seed=1)
    assert lo >= 0.99, f"Expected CI lower ≥ 0.99, got {lo}"
    assert hi <= 1.0 + 1e-9


def test_bootstrap_ci_reproducible_with_same_seed():
    labels = np.array([0, 1, 0, 1, 0])
    preds  = np.array([0, 0, 0, 1, 1])
    r1 = bootstrap_ci(labels, preds, _accuracy, n_bootstrap=300, seed=42)
    r2 = bootstrap_ci(labels, preds, _accuracy, n_bootstrap=300, seed=42)
    assert r1 == r2


# compare_methods

def _make_epoch_result(epoch, acc, f1=0.8, ci_lo=0.75, ci_hi=0.85):
    return EpochResult(
        epoch=epoch,
        train_loss=0.4,
        val_accuracy=acc,
        val_f1=f1,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
    )


def test_compare_methods_returns_dataframe():
    results = {
        "prefix": [_make_epoch_result(1, 0.82), _make_epoch_result(2, 0.85)],
        "lora":   [_make_epoch_result(1, 0.88), _make_epoch_result(2, 0.90)],
        "full":   [_make_epoch_result(1, 0.92), _make_epoch_result(2, 0.93)],
    }
    df = compare_methods(results)
    assert isinstance(df, pd.DataFrame)


def test_compare_methods_required_columns():
    results = {
        "prefix": [_make_epoch_result(1, 0.82)],
        "full":   [_make_epoch_result(1, 0.90)],
    }
    df = compare_methods(results)
    for col in ("method", "best_val_acc", "best_val_f1", "ci_lower", "ci_upper", "epochs_to_best"):
        assert col in df.columns, f"Missing column: {col}"


def test_compare_methods_one_row_per_method():
    results = {"prefix": [_make_epoch_result(1, 0.8)], "lora": [_make_epoch_result(1, 0.9)]}
    df = compare_methods(results)
    assert len(df) == 2


def test_compare_methods_sorted_by_accuracy():
    results = {
        "low":  [_make_epoch_result(1, 0.70)],
        "high": [_make_epoch_result(1, 0.95)],
        "mid":  [_make_epoch_result(1, 0.83)],
    }
    df = compare_methods(results)
    assert df.iloc[0]["best_val_acc"] >= df.iloc[1]["best_val_acc"]


def test_compare_methods_picks_best_epoch():
    """The best epoch's accuracy should be selected, not the last one."""
    results = {
        "prefix": [
            _make_epoch_result(1, 0.80),
            _make_epoch_result(2, 0.92),   # best
            _make_epoch_result(3, 0.88),
        ]
    }
    df = compare_methods(results)
    assert df.iloc[0]["best_val_acc"] == pytest.approx(0.92)
    assert df.iloc[0]["epochs_to_best"] == 2


def test_compare_methods_with_param_counts():
    results = {"prefix": [_make_epoch_result(1, 0.85)]}
    param_counts = {"prefix": {"trainable": 7680}}
    df = compare_methods(results, param_counts)
    assert df.iloc[0]["trainable_params"] == 7680


# prefix_length_sweep_table

def test_sweep_table_returns_dataframe():
    sweep = {5: _make_epoch_result(3, 0.82), 10: _make_epoch_result(3, 0.85)}
    df = prefix_length_sweep_table(sweep)
    assert isinstance(df, pd.DataFrame)


def test_sweep_table_sorted_by_K():
    sweep = {50: _make_epoch_result(1, 0.88), 1: _make_epoch_result(1, 0.78),
             10: _make_epoch_result(1, 0.83)}
    df = prefix_length_sweep_table(sweep)
    assert list(df["prefix_length"]) == [1, 10, 50]


def test_sweep_table_required_columns():
    sweep = {10: _make_epoch_result(2, 0.84)}
    df = prefix_length_sweep_table(sweep)
    for col in ("prefix_length", "val_accuracy", "val_f1", "ci_lower", "ci_upper"):
        assert col in df.columns


# plot_prefix_length_vs_accuracy  (smoke test — no visual validation)

def test_plot_saves_file(tmp_path):
    sweep = {
        1:  _make_epoch_result(1, 0.75, ci_lo=0.72, ci_hi=0.78),
        5:  _make_epoch_result(2, 0.80, ci_lo=0.77, ci_hi=0.83),
        10: _make_epoch_result(3, 0.83, ci_lo=0.80, ci_hi=0.86),
    }
    out = str(tmp_path / "sweep.png")
    plot_prefix_length_vs_accuracy(sweep, out)
    import os
    assert os.path.exists(out)
