import torch
import torch.nn as nn


class SequenceClassifier(nn.Module):
    def __init__(self, ssm_model: nn.Module, d_model: int, num_classes: int, vocab_size: int = None, input_dim: int = None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model) if vocab_size else None
        self.proj = nn.Linear(input_dim, d_model) if input_dim else nn.Identity()
        self.ssm_model = ssm_model
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, sequence_length) | (batch_size, sequence_length, d_model)
        """
        if self.embedding is not None and x.dtype == torch.long:
            x = self.embedding(x)  # b s -> b s d
        else:
            x = self.proj(x)  # b, 784, 1 -> b 784 d | b s d -> b s d
        out = self.ssm_model(x)  # b s d -> b s d
        pooled = out.mean(dim=1)  # Mean Pooling: b s d -> b d
        logits = self.head(pooled)  # b d -> b n
        return logits