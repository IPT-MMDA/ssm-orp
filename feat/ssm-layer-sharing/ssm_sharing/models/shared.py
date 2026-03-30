import torch
import torch.nn as nn
from .blocks import SimpleSSMBlock


class SharedSSM(nn.Module):
    """
    Розподілена модель.
    """
    def __init__(self, d_model: int, num_layers: int, num_classes: int, state_dim: int = 16):
        super().__init__()
        self.num_layers = num_layers
        self.layer = SimpleSSMBlock(d_model, state_dim)
        
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x має форму: (batch_size, sequence_length, d_model)
        """
        for _ in range(self.num_layers):
            x = self.layer(x)
        
        x_pooled = x.mean(dim=1)  # по часу
        
        return self.classifier(x_pooled)