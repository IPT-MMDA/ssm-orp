import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import BaselineSSM, RegularizedSSM

def generate_adding_task(batch_size, seq_len):
    """
    The Adding Task: Input has 2 channels. 
    Channel 0: Random numbers [0, 1].
    Channel 1: A binary mask with exactly two 1s.
    Target: Sum of the two numbers in Channel 0 where Channel 1 is 1.
    """
    # Initialize tensors
    x_val = torch.rand(batch_size, seq_len, 1)
    mask = torch.zeros(batch_size, seq_len, 1)
    
    for i in range(batch_size):
        # Select two unique random indices to place the '1' markers
        indices = np.random.choice(seq_len, 2, replace=False)
        mask[i, indices, 0] = 1.0
        
    x = torch.cat([x_val, mask], dim=-1) # (batch, seq_len, 2)
    y = (x_val * mask).sum(dim=1)         # (batch, 1)
    return x, y

def train_ssm(model, model_name, epochs=100, lr=0.001):
    print(f"\n--- Training {model_name} ---")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # We use a relatively long sequence to test stability
    seq_len = 50 
    batch_size = 32
    
    model.train()
    for epoch in range(epochs):
        # Generate fresh data every epoch
        x_train, y_train = generate_adding_task(batch_size, seq_len)
        
        optimizer.zero_grad()
        
        # SSM returns (outputs, hidden_states)
        # We only need the final output for the loss
        outputs, _ = model(x_train)
        
        # We take the last output of the sequence for the adding task result
        final_output = outputs[:, -1, :] 
        
        loss = criterion(final_output, y_train)
        loss.backward()
        
        # Step the optimizer
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.6f}")

    # Save the model weights to the root directory
    save_path = f"{model_name.lower()}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved as {save_path}")
    return model

if __name__ == "__main__":
    # Parameters match the Adding Task (2 inputs, 1 output)
    input_dim = 2
    hidden_dim = 64
    output_dim = 1
    
    # Train Baseline
    base_model = BaselineSSM(input_dim, hidden_dim, output_dim)
    train_ssm(base_model, "Baseline")
    
    # Train Regularized
    reg_model = RegularizedSSM(input_dim, hidden_dim, output_dim)
    train_ssm(reg_model, "Regularized")