import torch
import torch.nn as nn


class SimpleSSMBlock(nn.Module):
    def __init__(self, d_model: int, state_dim: int = 16):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        
        self.A = nn.Parameter(torch.randn(d_model, state_dim, state_dim) * 0.1) # d n m
        self.B = nn.Parameter(torch.randn(d_model, state_dim) * 0.1) # d n
        self.C = nn.Parameter(torch.randn(d_model, state_dim) * 0.1) # d n
        self.D = nn.Parameter(torch.randn(d_model) * 0.1) # d
        
        self.norm = nn.LayerNorm(d_model)
        self.activation = nn.GELU()
        
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x має форму: (batch_size, sequence_length, d_model)
        """
        batch_size, sequence_length, _ = x.shape
        device = x.device

        y = torch.zeros_like(x)  # b s d
        h = torch.zeros(batch_size, self.d_model, self.state_dim, device=device)  # b d m
        
        for t in range(sequence_length):
            x_t = x[:, t, :]  # b d
            
            # h_t = A * h_{t-1} + B * x_t
            Ah = torch.einsum('d n m, b d m -> b d n', self.A, h)
            Bx = self.B[None, ...] * x_t[..., None]  # 1 d n * b d 1 -> b d n
            h = Ah + Bx
            
            # y_t = C * h_t + D * x_t
            Ch = torch.einsum('d n, b d n -> b d', self.C, h)
            Dx = self.D[None, ...] * x_t  # 1 d * b d -> b d
            y_t = Ch + Dx
            
            y[:, t, :] = y_t  # b d = b d
            
        y = self.norm(y)
        y = self.activation(y)
        y = self.out_proj(y)
        
        return x + y