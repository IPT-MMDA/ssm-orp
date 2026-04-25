import torch
import torch.nn as nn
from mamba_ssm import Mamba

class SharedMamba(nn.Module):
    """
    ALBERT-style Weight Sharing for Mamba.
    Instead of creating N independent layers, we create 1 block
    and recursively pass the tensor through it N times.
    """
    def __init__(self, d_model: int, n_layers: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        
        self.shared_block = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=4,
            expand=expand
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            (batch_size, seq_len, d_model)
        """
        for _ in range(self.n_layers):
            x = self.shared_block(self.norm(x)) + x
            
        return x