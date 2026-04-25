import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

class BaselineSSM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Matrix B: Input to Hidden
        self.B = nn.Linear(input_dim, hidden_dim, bias=False)
        # Matrix A: Hidden to Hidden (Transition)
        self.A = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # Matrix C: Hidden to Output
        self.C = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x):
        """
        x shape: (batch_size, sequence_length, input_dim)
        Returns output trajectory and hidden state trajectory.
        """
        batch_size, seq_len, _ = x.shape
        
        # Initialize the hidden state at t=0
        h_t = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        
        outputs = []
        hidden_states = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            # SSM Equations
            # h_t = A*h_{t-1} + B*x_t
            h_t = self.A(h_t) + self.B(x_t)
            
            # y_t = C*h_t
            y_t = self.C(h_t)
            
            outputs.append(y_t)
            hidden_states.append(h_t)
            
        # Stack lists into tensors: (batch_size, seq_len, dim)
        return torch.stack(outputs, dim=1), torch.stack(hidden_states, dim=1)
    

class RegularizedSSM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Apply Spectral Normalization to A, B, and C matrices
        self.B = spectral_norm(nn.Linear(input_dim, hidden_dim, bias=False))
        self.A = spectral_norm(nn.Linear(hidden_dim, hidden_dim, bias=False))
        self.C = spectral_norm(nn.Linear(hidden_dim, output_dim, bias=False))

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        h_t = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        
        outputs = []
        hidden_states = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            # The equations remain exactly the same, but the matrices are now bounded
            h_t = self.A(h_t) + self.B(x_t)
            y_t = self.C(h_t)
            
            outputs.append(y_t)
            hidden_states.append(h_t)
            
        return torch.stack(outputs, dim=1), torch.stack(hidden_states, dim=1)