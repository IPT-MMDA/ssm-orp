import torch
import torch.nn as nn
from mamba_ssm import Mamba

class StandardMamba(nn.Module):
    """
    Стандартна імплементація: N незалежних шарів Mamba.
    Слугує бейзлайном для порівняння кількості параметрів.
    """
    def __init__(self, d_model: int, n_layers: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=4,
                expand=expand
            ) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x) + x  # Residual connection
        return self.norm(x)