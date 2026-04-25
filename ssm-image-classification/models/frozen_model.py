import torch.nn as nn
from models.mamba_backbone import MambaBackbone

class FrozenModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.backbone = MambaBackbone(config)

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.classifier = nn.Linear(config.d_model, config.num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)