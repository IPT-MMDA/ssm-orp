import torch
import torch.nn as nn
from torchdeq import get_deq


class DEQCell(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim)
        )
    def forward(self, z, x):
        return torch.relu(self.conv(z) + x)

class MDEQSmall(nn.Module):
    def __init__(self, dim=64, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.cell = DEQCell(dim)
        self.deq = get_deq()
        self.deq_kwargs = {'solver': 'broyden', 'f_max_iter': 40, 'f_tol': 1e-3}
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(dim, num_classes))
        self.last_info = {}

    def forward(self, x):
        x_in = self.stem(x)
        z0 = torch.zeros_like(x_in)
        z_out, info = self.deq(lambda z: self.cell(z, x_in), z0, **self.deq_kwargs)
        # Фіксуємо стан сольвера
        self.last_info['nstep'] = info['nstep'].float().mean().item()
        self.last_info['res'] = info.get('abs_lowest', info.get('lowest_res', torch.tensor(0.0))).mean().item()
        return self.head(z_out[0] if isinstance(z_out, list) else z_out)
