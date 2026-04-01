"""
Tests for dataset.py - run without GPU, no mamba-ssm required.
"""
import numpy as np
import pytest
import torch

from magnitude_pruning.dataset import (
    TimeSeriesDataset,
    build_loaders,
    inject_anomalies,
    mackey_glass,
)


# ---------------------------------------------------------------------------
# mackey_glass
# ---------------------------------------------------------------------------

def test_mackey_glass_shape():
    x = mackey_glass(n_steps=500)
    assert x.shape == (500,), "Output must be 1-D with requested length"


def test_mackey_glass_normalised():
    x = mackey_glass(n_steps=1000)
    assert x.min() >= -1e-6,  "Min should be ~0 after normalisation"
    assert x.max() <=  1 + 1e-6, "Max should be ~1 after normalisation"


def test_mackey_glass_reproducible():
    a = mackey_glass(n_steps=300, seed=0)
    b = mackey_glass(n_steps=300, seed=0)
    np.testing.assert_array_equal(a, b)


def test_mackey_glass_different_seeds():
    a = mackey_glass(n_steps=300, seed=0)
    b = mackey_glass(n_steps=300, seed=99)
    assert not np.allclose(a, b), "Different seeds should give different series"


# ---------------------------------------------------------------------------
# TimeSeriesDataset
# ---------------------------------------------------------------------------

def test_dataset_len():
    series = np.random.rand(500)
    ds = TimeSeriesDataset(series, seq_len=50, horizon=5)
    expected = 500 - 50 - 5 + 1
    assert len(ds) == expected


def test_dataset_item_shapes():
    series = np.random.rand(300)
    ds = TimeSeriesDataset(series, seq_len=64, horizon=1)
    x, y = ds[0]
    assert x.shape == (64, 1), f"x should be (seq_len, 1), got {x.shape}"
    assert y.shape == (1,),    f"y should be (horizon,), got {y.shape}"


def test_dataset_no_overlap_between_x_and_y():
    series = torch.arange(200, dtype=torch.float32).numpy()
    ds = TimeSeriesDataset(series, seq_len=10, horizon=3)
    x, y = ds[5]
    # x covers indices 5..14, y covers 15..17
    assert float(x[-1, 0]) == 14.0
    assert float(y[0]) == 15.0


# ---------------------------------------------------------------------------
# inject_anomalies
# ---------------------------------------------------------------------------

def test_inject_anomalies_shape():
    series = np.random.rand(1000)
    corrupted = inject_anomalies(series)
    assert corrupted.shape == series.shape


def test_inject_anomalies_differs():
    series = np.random.rand(1000)
    corrupted = inject_anomalies(series, missing_frac=0.1, spike_frac=0.05)
    assert not np.allclose(series, corrupted)


def test_inject_anomalies_within_bounds():
    series = np.random.rand(1000)
    corrupted = inject_anomalies(series)
    assert corrupted.min() >= 0.0
    assert corrupted.max() <= 1.0 + 1e-6, "Clipping to [0,1] should hold"


def test_inject_anomalies_reproducible():
    series = np.random.rand(500)
    a = inject_anomalies(series, seed=1)
    b = inject_anomalies(series, seed=1)
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# build_loaders (CPU-only, small scale)
# ---------------------------------------------------------------------------

def test_build_loaders_returns_three():
    loaders = build_loaders(n_total=1000, batch_size=16, seq_len=32, horizon=1)
    assert len(loaders) == 3


def test_build_loaders_batch_shape():
    train_loader, _, _ = build_loaders(
        n_total=800, batch_size=8, seq_len=32, horizon=1
    )
    x, y = next(iter(train_loader))
    assert x.shape == (8, 32, 1), f"Unexpected x shape: {x.shape}"
    assert y.shape == (8, 1),     f"Unexpected y shape: {y.shape}"


def test_build_loaders_anomaly_differs():
    """Anomaly test loader data should differ from clean test loader."""
    _, _, clean_loader = build_loaders(
        n_total=800, batch_size=8, seq_len=32, horizon=1,
        add_anomalies=False, seed=0
    )
    _, _, anom_loader = build_loaders(
        n_total=800, batch_size=8, seq_len=32, horizon=1,
        add_anomalies=True, seed=0
    )
    x_clean, _ = next(iter(clean_loader))
    x_anom,  _ = next(iter(anom_loader))
    # At least some batches must differ
    assert not torch.allclose(x_clean, x_anom)
