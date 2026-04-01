import torch
import torch.nn as nn


# 1. Задача (Classification Task)
# Наші моделі зараз видають тензор форми (B, L, d_model). Тобі треба:
# Написати клас-обгортку SequenceClassifier(ssm_model, num_classes), яка приймає нашу Mamba (Shared або Standard).
# У forward зробити усереднення по часу (Mean Pooling): x.mean(dim=1).
# Пропустити результат через nn.Linear(d_model, num_classes).

# feat: add SequenceClassifier wrapper model for mamba models
class SequenceClassifier(nn.Module):
    def __init__(self, ssm_model: nn.Module, d_model: int, num_classes: int):
        super().__init__()
        self.ssm_model = ssm_model
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.ssm_model(x)  # b s d -> b s d
        pooled = out.mean(dim=1)  # Mean Pooling: b s d -> b d
        logits = self.head(pooled)  # b d -> b n
        return logits