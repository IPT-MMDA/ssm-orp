import torch
import torch.nn as nn
from mamba_ssm import Mamba

class StandardMamba(nn.Module):
    """
    Standard implementation: N independent Mamba layers.
    Serves as a baseline for comparing the number of parameters.
    """
    def __init__(self, d_model: int, n_layers: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "mamba": Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=expand),
                "norm": nn.LayerNorm(d_model)
            }) for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        """
        for layer_dict in self.layers:
            x_norm = layer_dict["norm"](x)
            x = layer_dict["mamba"](x_norm) + x
        return x