import torch
import torch.nn as nn


class SimpleSSM(nn.Module):
    """
    Discrete-time SSM with explicit trainable matrices:
        h_t = A h_{t-1} + B x_t
        y_t = C h_t

    Then:
    - for classification: mean-pool y_t and classify
    - for regression: mean-pool y_t and regress
    """
    def __init__(self, input_dim, hidden_dim, ssm_output_dim, task_type="classification", num_classes=3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.ssm_output_dim = ssm_output_dim
        self.task_type = task_type
        self.num_classes = num_classes

        # Explicit A, B, C
        self.A = nn.Parameter(torch.eye(hidden_dim) * 0.8)
        self.B = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.1)
        self.C = nn.Parameter(torch.randn(ssm_output_dim, hidden_dim) * 0.1)

        if task_type == "classification":
            self.head = nn.Linear(ssm_output_dim, num_classes)
        else:
            self.head = nn.Linear(ssm_output_dim, 1)

    def forward_features(self, x):
        # x: [B, T, input_dim]
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]                         # [B, input_dim]
            h = h @ self.A.T + x_t @ self.B.T       # [B, hidden_dim]
            y_t = h @ self.C.T                      # [B, ssm_output_dim]
            outputs.append(y_t)

        y_seq = torch.stack(outputs, dim=1)         # [B, T, ssm_output_dim]
        pooled = y_seq.mean(dim=1)                  # [B, ssm_output_dim]
        return pooled, y_seq

    def forward(self, x):
        pooled, _ = self.forward_features(x)
        out = self.head(pooled)
        return out


def build_model(cfg):
    return SimpleSSM(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        ssm_output_dim=cfg.ssm_output_dim,
        task_type=cfg.task_type,
        num_classes=cfg.num_classes,
    )
