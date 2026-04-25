import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import DataConfig
from utils import set_seed


def make_stable_matrix(hidden_dim: int, rng: np.random.Generator):
    a = rng.normal(0.0, 0.25, size=(hidden_dim, hidden_dim)).astype(np.float32)
    eigvals = np.linalg.eigvals(a)
    spectral_radius = np.max(np.abs(eigvals))
    if spectral_radius > 0:
        a = a / (1.25 * spectral_radius)
    return a.astype(np.float32)


def generate_ssm_data(num_samples: int, cfg: DataConfig):
    rng = np.random.default_rng(cfg.seed)

    A_true = make_stable_matrix(cfg.hidden_dim, rng)
    B_true = rng.normal(0.0, 0.5, size=(cfg.hidden_dim, cfg.input_dim)).astype(np.float32)
    C_true = rng.normal(0.0, 0.5, size=(cfg.ssm_output_dim, cfg.hidden_dim)).astype(np.float32)

    X = np.zeros((num_samples, cfg.seq_len, cfg.input_dim), dtype=np.float32)

    if cfg.task_type == "classification":
        y = np.zeros((num_samples,), dtype=np.int64)
    else:
        y = np.zeros((num_samples, 1), dtype=np.float32)

    for i in range(num_samples):
        x = rng.normal(0.0, 1.0, size=(cfg.seq_len, cfg.input_dim)).astype(np.float32)
        h = np.zeros((cfg.hidden_dim,), dtype=np.float32)
        y_seq = []

        for t in range(cfg.seq_len):
            h = A_true @ h + B_true @ x[t]
            out = C_true @ h
            y_seq.append(out)

        y_seq = np.stack(y_seq, axis=0)  # [T, ssm_output_dim]
        summary = y_seq.mean(axis=0)

        if cfg.task_type == "classification":
            score = summary[0] + 0.5 * summary[1] - 0.25 * summary[2]
            if score < -0.5:
                label = 0
            elif score < 0.5:
                label = 1
            else:
                label = 2
            y[i] = label
        else:
            target = float(summary[0] + 0.2 * np.mean(y_seq[:, 1]))
            y[i, 0] = target

        X[i] = x

    return X, y


class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        if y.dtype.kind in ("i", "u"):
            self.y = torch.tensor(y, dtype=torch.long)
        else:
            self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {"inputs": self.X[idx], "targets": self.y[idx]}


def build_dataloaders(cfg: DataConfig):
    set_seed(cfg.seed)

    X_train, y_train = generate_ssm_data(cfg.num_train, cfg)
    X_val, y_val = generate_ssm_data(cfg.num_val, cfg)
    X_test, y_test = generate_ssm_data(cfg.num_test, cfg)

    train_ds = SequenceDataset(X_train, y_train)
    val_ds = SequenceDataset(X_val, y_val)
    test_ds = SequenceDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
