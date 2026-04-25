import torch


def add_gaussian_noise(x: torch.Tensor, sigma: float = 0.05):
    return x + torch.randn_like(x) * sigma


def shift_sequence(x: torch.Tensor, shift: int = 1):
    if shift == 0:
        return x

    b, t, d = x.shape
    out = torch.zeros_like(x)

    if shift > 0:
        out[:, shift:, :] = x[:, :t - shift, :]
    else:
        shift = abs(shift)
        out[:, :t - shift, :] = x[:, shift:, :]

    return out
