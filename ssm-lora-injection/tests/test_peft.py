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
        
def test_lora_requires_grad():
    base_linear = torch.nn.Linear(32, 64)
    base_linear.weight.requires_grad = False
    if base_linear.bias is not None:
        base_linear.bias.requires_grad = False
        
    lora_layer = LoRALinear(base_linear, rank=4, alpha=8)

    assert not lora_layer.linear.weight.requires_grad, "Base weights should be frozen!"

    assert lora_layer.lora_A.requires_grad, "LoRA matrix A must require gradients!"
    assert lora_layer.lora_B.requires_grad, "LoRA matrix B must require gradients!"