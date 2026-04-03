from dataclasses import dataclass
import torch

@dataclass
class Config:
    batch_size: int = 64
    num_epochs: int = 20
    lr: float = 1e-3

    image_size: int = 32
    channels: int = 3
    num_classes: int = 10

    seq_len: int = 32 * 32 * 3
    input_dim: int = 1
    d_model: int = 128

    device: str = "cuda" if torch.cuda.is_available() else "cpu"