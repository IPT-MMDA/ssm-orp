"""
MambaForecaster: univariate time-series forecasting with a selective SSM.

Architecture:
  Linear(1 -> d_model)
  N x Mamba block with residual
  RMSNorm
  Linear(d_model -> horizon)

Requires mamba-ssm and a CUDA GPU.
For IMP: A_log is excluded from pruning - it controls the SSM eigenvalues,
zeroing it collapses the recurrent dynamics.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mamba_ssm import Mamba


class RMSNorm(nn.Module):
    """RMSNorm - like LayerNorm but without the mean shift, only rescales."""

    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * x / rms


class MambaForecaster(nn.Module):
    """
    Stacks Mamba blocks to forecast univariate time series.

    Params:
        d_model   -- hidden size
        d_state   -- SSM state size
        d_conv    -- conv kernel width inside each block
        expand    -- inner dim = expand * d_model
        n_layers  -- how many blocks to stack
        horizon   -- how many steps ahead to predict
    """

    def __init__(
        self,
        d_model: int  = 64,
        d_state: int  = 16,
        d_conv:  int  = 4,
        expand:  int  = 2,
        n_layers: int = 4,
        horizon: int  = 1,
    ):
        super().__init__()
        self.input_proj  = nn.Linear(1, d_model)
        self.layers      = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])
        self.norm        = RMSNorm(d_model)
        self.output_proj = nn.Linear(d_model, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, 1) -> pred: (B, horizon)
        h = self.input_proj(x)
        for layer in self.layers:
            h = h + layer(h)          # residual around each block
        h = self.norm(h)
        return self.output_proj(h[:, -1, :])   # predict from the last timestep

    def named_prunable_params(self) -> list[tuple[str, nn.Module, str]]:
        """All weight tensors that IMP can prune.

        Excluded:
        - A_log: controls SSM eigenvalues, zeroing it breaks the dynamics
        - D: scalar skip weight, too small to matter
        - biases: not pruned by convention
        """
        excluded = {"A_log", "D"}
        result = []
        for full_name, module in self.named_modules():
            for param_name, _ in list(module.named_parameters(recurse=False)):
                if param_name in excluded or "bias" in param_name:
                    continue
                result.append((f"{full_name}.{param_name}", module, param_name))
        return result

    def count_params(self) -> dict[str, int]:
        """Returns total / nonzero / zero param counts."""
        total, nonzero = 0, 0
        for _, p in self.named_parameters():
            total   += p.numel()
            nonzero += (p != 0).sum().item()
        return {"total": total, "nonzero": nonzero, "zero": total - nonzero}
