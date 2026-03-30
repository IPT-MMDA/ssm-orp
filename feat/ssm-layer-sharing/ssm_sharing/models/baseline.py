import torch
import torch.nn as nn
from .blocks import SimpleSSMBlock


class StandardSSM(nn.Module):
    """
    Базова модель: N незалежних шарів (стандартний глибокий підхід).
    """
    def __init__(self, d_model: int, num_layers: int, num_classes: int, state_dim: int = 16):
        super().__init__()
        self.layers = nn.ModuleList([SimpleSSMBlock(d_model, state_dim) for _ in range(num_layers)])
        
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x має форму: (batch_size, sequence_length, d_model)
        """
        for layer in self.layers:
            x = layer(x)
            
        x_pooled = x.mean(dim=1)  # по часу
        
        return self.classifier(x_pooled)