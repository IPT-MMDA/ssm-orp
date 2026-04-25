import torch
import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from models import BaselineSSM, RegularizedSSM

def test_baseline_norm_can_exceed_one():
    """Checks that a standard unconstrained matrix can have a spectral norm > 1."""
    model = BaselineSSM(input_dim=10, hidden_dim=32, output_dim=1)
    
    # Randomly initialize weights heavily to force a large norm
    torch.nn.init.normal_(model.A.weight, mean=0, std=2.0)
    
    # Calculate the largest singular value (Spectral Norm, L2 norm of the matrix)
    spectral_norm = torch.linalg.matrix_norm(model.A.weight, ord=2).item()
    
    # We expect the unregularized model to easily exceed 1
    assert spectral_norm > 1.0, "Baseline norm should be unconstrained."

def test_regularized_norm_is_bounded():
    model = RegularizedSSM(input_dim=10, hidden_dim=32, output_dim=1)
    
    # Run 5 dummy passes to let Power Iteration converge
    for _ in range(5):
        dummy_input = torch.randn(1, 5, 10)
        model(dummy_input)
    
    # Access the weight (which is now normalized)
    effective_weight = model.A.weight
    spectral_norm = torch.linalg.matrix_norm(effective_weight, ord=2).item()
    
    assert spectral_norm <= 1.05, f"Expected bounded norm <= 1.05, got {spectral_norm}"

def test_forward_pass_shapes():
    """Ensures the models output the correct tensor shapes."""
    batch_size, seq_len, input_dim = 4, 15, 8
    hidden_dim, output_dim = 16, 2
    
    model = RegularizedSSM(input_dim, hidden_dim, output_dim)
    x = torch.randn(batch_size, seq_len, input_dim)
    
    outputs, hidden_states = model(x)
    
    assert outputs.shape == (batch_size, seq_len, output_dim)
    assert hidden_states.shape == (batch_size, seq_len, hidden_dim)