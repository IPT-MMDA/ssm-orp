import torch
import numpy as np

class LoRALinear(torch.nn.Module):
    def __init__(self, linear_layer, rank=4, alpha=8):
        super().__init__()
        self.linear = linear_layer
        self.rank = rank
        self.alpha = alpha
        self.lora_A = torch.nn.Parameter(torch.zeros(linear_layer.in_features, rank))
        self.lora_B = torch.nn.Parameter(torch.zeros(rank, linear_layer.out_features))
        self.scaling = self.alpha / self.rank
        torch.nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        torch.nn.init.zeros_(self.lora_B)
        
    def forward(self, x):
        return self.linear(x) + (x @ self.lora_A @ self.lora_B) * self.scaling

def test_lora_output_shape():
    base_linear = torch.nn.Linear(32, 64)
    lora_layer = LoRALinear(base_linear, rank=4, alpha=8)
    dummy_input = torch.randn(2, 10, 32)
    
    output = lora_layer(dummy_input)
    
    assert output.shape == (2, 10, 64), f"Expected shape (2, 10, 64), got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN values!"