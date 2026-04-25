import torch
from configs.config import Config
from data.dataset import get_dataloaders
from models.frozen_model import FrozenModel

config = Config()

train_loader, _ = get_dataloaders(config)

model = FrozenModel(config).to(config.device)

x, y = next(iter(train_loader))
x = x.to(config.device)

with torch.no_grad():
    out = model(x)

print("Input:", x.shape)
print("Output:", out.shape)