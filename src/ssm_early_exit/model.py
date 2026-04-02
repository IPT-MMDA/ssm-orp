import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Union

class SimplifiedSSMBlock(nn.Module):
    """
    дещо спрощений згортковий блок SSM.
    замість повільного рекурентного циклу використав Causal Conv1d
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # параметри SSM: h_t = A*h_{t-1} + B*x_t; y_t = C*h_t + D*x_t
        # для стабільності A ініціалізується так щоб після сигмоїди бути в межах (0, 1)
        self.A_param = nn.Parameter(torch.randn(d_model) * 0.1) 
        self.B = nn.Parameter(torch.randn(d_model))
        self.C = nn.Parameter(torch.randn(d_model))
        self.D = nn.Parameter(torch.randn(d_model))
        
        self.act = nn.GELU()
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        batch_size, seq_len, d_model = x.shape
        
        # 1. формуємо згорткове ядро K для SSM
        # обмежуємо A від 0 до 1 для стабільності
        A = torch.sigmoid(self.A_param) 
        
        # вісь часу: t = [0, 1, 2, ..., seq_len-1]
        t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        
        # обчислюємо A^t. розмірність: (d_model, seq_len)
        A_t = A.unsqueeze(1) ** t.unsqueeze(0) 
        
        # ядро згортки K_t = C * A^t * B. розмірність: (d_model, 1, seq_len)
        K = (self.C.unsqueeze(1) * A_t * self.B.unsqueeze(1)).unsqueeze(1)
        
        # 2. застосовуємо causal convolution
        # міняємо розмірності для conv1d: (batch, d_model, seq_len)
        x_t = x.transpose(1, 2)
        
        # робимо depthwise згортку (groups=d_model), padding=seq_len-1 для причинності
        out = F.conv1d(x_t, weight=K, padding=seq_len - 1, groups=d_model)
        
        # відрізає зайвий padding справа щоб вихід відповідав seq_len
        out = out[..., :seq_len] 
        
        # додаємо skip-connection (D * x)
        out = out + (self.D.unsqueeze(0).unsqueeze(-1) * x_t)
        
        # повертаємо розмірності: (batch, seq_len, d_model)
        out = out.transpose(1, 2)
        
        # 3. нелінійність проекція та residual connection
        out = self.proj(self.act(out))
        return self.norm(x + out)


class DeepSSM(nn.Module):
    """
    глибока SSM з Early Exiting
    має проміжні класифікатори на шарах L/4, L/2, 3L/4 та фінальний класифікатор
    """
    def __init__(self, input_dim: int = 1, d_model: int = 64, n_layers: int = 8, num_classes: int = 3):
        super().__init__()
        self.n_layers = n_layers
        
        # вхідна проекція (з 1 фічі до d_model)
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # масив SSM блоків
        self.layers = nn.ModuleList([SimplifiedSSMBlock(d_model) for _ in range(n_layers)])
        
        # визначаємо точки виходу (Exit points)
        # наприклад для 8 шарів це будуть індекси: 1 (L/4), 3 (L/2), 5 (3L/4), 7 (Final)
        self.exit_points = {
            n_layers // 4 - 1: "head_1/4",
            n_layers // 2 - 1: "head_1/2",
            3 * n_layers // 4 - 1: "head_3/4",
            n_layers - 1: "head_final"
        }
        
        # лінійні хеди для кожної точки виходу
        self.heads = nn.ModuleDict({
            name: nn.Linear(d_model, num_classes) for name in self.exit_points.values()
        })

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        метод для навчання 
        проганяє дані через усі шари і повертає прогнози з усіх голів
        це дозволить нам рахувати loss для кожної голови і навчати їх
        """
        x = self.input_proj(x)
        outputs = {}
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            if i in self.exit_points:
                head_name = self.exit_points[i]
                # global average pooling (усереднення по часу)
                pooled = x.mean(dim=1) 
                logits = self.heads[head_name](pooled)
                outputs[head_name] = logits
                
        return outputs

    def forward_inference(self, x: torch.Tensor, threshold: float) -> Tuple[torch.Tensor, str]:
        """
        метод для інференсу (Early Exiting)
        працює шар за шаром. якщо ентропія на проміжному хеду
        висока (ентропія < threshold), зупиняє обчислення і повертає результат.
        """
        x = self.input_proj(x)
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            if i in self.exit_points:
                head_name = self.exit_points[i]
                
                # global average pooling та класифікація
                pooled = x.mean(dim=1)
                logits = self.heads[head_name](pooled)
                
                # рахуємо ймовірності та ентропію
                probs = F.softmax(logits, dim=-1)
                # додаємо 1e-9 для уникнення log(0)
                entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1) 
                
                # якщо це останній шар то виходимо в будь-якому випадку
                if i == self.n_layers - 1:
                    return logits, head_name
                
                # тут перевіряю умову раннього виходу (всі елементи батчу мають бути впевнені)
                # для батчевого інференсу ми перевіряємо середню ентропію батчу, 
                # або можна реалізувати динамічний розмір батчу. для простоти беремо середнє
                if entropy.mean().item() < threshold:
                    return logits, head_name
                    
        return logits, "head_final"