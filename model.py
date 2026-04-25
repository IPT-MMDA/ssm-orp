import torch
import torch.nn as nn

class SSMStudent(nn.Module):
    """
    Мініатюрна модель-учень, що імітує SSM (State Space Model) 
    через рекурентну структуру для обробки патчів зображення.
    """
    def __init__(self, patch_size=4, embed_dim=128, num_classes=10):
        super().__init__()
        self.patch_size = patch_size
        # Перетворюємо патч (C*P*P) у вектор embed_dim
        self.patch_to_embed = nn.Linear(3 * patch_size * patch_size, embed_dim)
        
        # Використовуємо GRU як ефективну апроксимацію SSM для послідовностей
        self.ssm_layer = nn.GRU(embed_dim, embed_dim, batch_first=True, num_layers=2)
        
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: [B, 3, 32, 32] -> CIFAR-10
        B, C, H, W = x.shape
        p = self.patch_size
        
        # Розбиття на патчі та розгортання в послідовність (1D sequence)
        # Shape: [B, Num_Patches, Patch_Dim]
        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(B, -1, C * p * p)
        
        x = self.patch_to_embed(x)
        output, h_n = self.ssm_layer(x)
        
        # Беремо останній стан послідовності для класифікації
        return self.classifier(h_n[-1])