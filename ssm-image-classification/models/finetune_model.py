import torch.nn as nn
from models.mamba_backbone import MambaBackbone

class FinetuneModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.backbone = MambaBackbone(config)
        self.classifier = nn.Linear(config.d_model, config.num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)