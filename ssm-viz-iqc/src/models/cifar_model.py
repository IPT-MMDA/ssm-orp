import torch.nn as nn
import torch.nn.functional as F

class CIFARModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 16) #12
        self.fc2 = nn.Linear(16, 10)
        self.layers = nn.ModuleList([self.fc1, self.fc2])

    def forward(self, x):
        x = x.view(-1, 768)
        x = F.relu(self.fc1(x))
        return self.fc2(x)