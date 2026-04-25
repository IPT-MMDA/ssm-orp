import torch.nn as nn
from mamba_ssm import Mamba

class MambaBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.input_proj = nn.Linear(config.input_dim, config.d_model)

        self.mamba = Mamba(
            d_model=config.d_model,
            d_state=16,
            d_conv=4,
            expand=2
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.mamba(x)
        return x.mean(dim=1)