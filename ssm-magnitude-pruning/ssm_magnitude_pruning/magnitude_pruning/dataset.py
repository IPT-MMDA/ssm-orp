"""
Synthetic time-series dataset generation.

Uses the Mackey-Glass delay-differential equation - a canonical benchmark for
chaotic dynamics that stresses the long-range memory of sequence models.

  dx/dt = beta * theta^n * x(t-tau) / (theta^n + x(t-tau)^n) - gamma * x(t)
  theta = 1, beta=0.2, gamma=0.1, n=10, tau=17  -> mildly chaotic regime
  tau=30                                        -> strongly chaotic regime
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Mackey-Glass ODE (solved with a simple Euler + history buffer)
# ---------------------------------------------------------------------------

def mackey_glass(
    n_steps: int = 12_000,
    tau: int = 17,
    beta: float = 0.2,
    gamma: float = 0.1,
    n: float = 10.0,
    dt: float = 1.0,
    x0: float = 1.2,
    seed: int = 42,
) -> np.ndarray:
    """
    Numerically integrate the Mackey-Glass DDE via simple Euler with a
    history buffer.

    Returns
    -------
    x : np.ndarray, shape (n_steps,)
        Normalised time series in [0, 1].
    """
    rng = np.random.default_rng(seed)
    # Initialise history with small random perturbations around x0
    history = np.full(tau + 1, x0) + rng.uniform(-0.05, 0.05, tau + 1)

    x = np.empty(n_steps)
    x[:tau + 1] = history

    for t in range(tau, n_steps - 1):
        x_tau = x[t - tau]
        dx = beta * x_tau / (1.0 + x_tau ** n) - gamma * x[t]
        x[t + 1] = x[t] + dt * dx

    # Min-max normalise to [0, 1] so that losses are scale-independent
    x_min, x_max = x.min(), x.max()
    return (x - x_min) / (x_max - x_min + 1e-8)


# ---------------------------------------------------------------------------
# Sliding-window Dataset
# ---------------------------------------------------------------------------

class TimeSeriesDataset(Dataset):
    """
    Wraps a 1-D time series into (input_window, target_window) pairs.

    Parameters
    ----------
    series  : np.ndarray, shape (T,)
    seq_len : int
              Number of past time steps given as input.
    horizon : int
              Number of future time steps to predict.
    """

    def __init__(self, series: np.ndarray, seq_len: int = 128, horizon: int = 1):
        self.series = torch.tensor(series, dtype=torch.float32)
        self.seq_len = seq_len
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.series) - self.seq_len - self.horizon + 1

    def __getitem__(self, idx: int):
        x = self.series[idx : idx + self.seq_len].unsqueeze(-1)   # (seq_len, 1)
        y = self.series[idx + self.seq_len : idx + self.seq_len + self.horizon]  # (horizon,)
        return x, y


# ---------------------------------------------------------------------------
# Anomaly injection (for robustness evaluation)
# ---------------------------------------------------------------------------

def inject_anomalies(
    series: np.ndarray,
    missing_frac: float = 0.05,
    spike_frac: float = 0.03,
    spike_magnitude: float = 3.0,
    seed: int = 0,
) -> np.ndarray:
    """
    Corrupt a copy of 'series' by:
      - randomly replacing 'missing_frac' of values with linear interpolation
        (simulates sensor drop-outs)
      - randomly adding signed spikes of +-'spike_magnitude' * std to
        'spike_frac' of values (simulates measurement outliers)

    Returns
    -------
    corrupted : np.ndarray (same shape as series)
    """
    rng = np.random.default_rng(seed)
    corrupted = series.copy()
    T = len(series)

    # Missing values: replace with linear interpolation
    n_missing = int(missing_frac * T)
    missing_idx = rng.choice(T, n_missing, replace=False)
    for i in sorted(missing_idx):
        left  = i - 1 if i > 0 else i
        right = i + 1 if i < T - 1 else i
        corrupted[i] = 0.5 * (corrupted[left] + corrupted[right])

    # Additive spikes
    n_spikes = int(spike_frac * T)
    spike_idx = rng.choice(T, n_spikes, replace=False)
    std = corrupted.std()
    signs = rng.choice([-1, 1], n_spikes)
    corrupted[spike_idx] += signs * spike_magnitude * std

    # Reclip to [0, 1] since series is already normalised
    return np.clip(corrupted, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def build_loaders(
    seq_len: int = 128,
    horizon: int = 1,
    n_total: int = 12_000,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    batch_size: int = 64,
    tau: int = 17,
    seed: int = 42,
    add_anomalies: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Generate a Mackey-Glass series, split it chronologically into train /
    val / test sets, and return DataLoaders.

    The test loader optionally uses anomaly-corrupted data.

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    series = mackey_glass(n_steps=n_total, tau=tau, seed=seed)

    n_train = int(train_frac * n_total)
    n_val   = int(val_frac   * n_total)

    train_series = series[:n_train]
    val_series   = series[n_train : n_train + n_val]
    test_series  = series[n_train + n_val :]

    if add_anomalies:
        test_series = inject_anomalies(test_series, seed=seed)

    train_ds = TimeSeriesDataset(train_series, seq_len, horizon)
    val_ds   = TimeSeriesDataset(val_series,   seq_len, horizon)
    test_ds  = TimeSeriesDataset(test_series,  seq_len, horizon)

    kwargs = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader
