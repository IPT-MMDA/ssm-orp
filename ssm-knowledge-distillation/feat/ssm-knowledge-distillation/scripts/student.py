import torch
import torch.nn as nn
from mamba_ssm import Mamba


#student is ~4x smaller than teacher: d_model=64 vs 128, 4 layers vs 6
class StudentSSM(nn.Module):
    def __init__(self, n_classes=35, d_model=64, n_layers=4, d_state=16, d_conv=4, expand=2, stride=16):
        super().__init__()
        #strided conv to compress 16k -> 1k steps
        self.projection = nn.Conv1d(1, d_model, kernel_size=stride, stride=stride)

        self.layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])

        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        #x: (batch, 16000)
        x = x.unsqueeze(1)                #(batch, 1, 16000)
        x = self.projection(x)            #(batch, d_model, 1000)
        x = x.transpose(1, 2)             #(batch, 1000, d_model)

        for norm, layer in zip(self.norms, self.layers):
            x = x + layer(norm(x))  #pre-norm residual

        x = x.mean(dim=1)          #global average pool -> (batch, d_model)
        return self.head(x)         #(batch, n_classes)
