import torch
import torch.nn as nn
from mamba_ssm import Mamba

class SharedMamba(nn.Module):
    """
    ALBERT-style Weight Sharing for Mamba.
    Замість створення N незалежних шарів, ми створюємо 1 блок
    і рекурсивно пропускаємо через нього тензор N разів.
    """
    def __init__(self, d_model: int, n_layers: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        
        # Ініціалізуємо ТІЛЬКИ ОДИН блок.
        # Це кардинально зменшить споживання пам'яті для зберігання ваг.
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
        # Рекурсивний прохід по глибині. 
        for _ in range(self.n_layers):
            x = self.shared_block(x) + x  # Residual connection є критичним для збіжності!
            
        return self.norm(x)