import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict

def train_stage_1(
    model: nn.Module, 
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    epochs: int = 15, 
    device: str = "cpu"
) -> nn.Module:
    """
    етап 1: навчання backbone (SSM шарів) + фінального хеду.
    метою є дозволити моделі вивчити максимально складні часові залежності
    без втручання проміжних класифікаторів
    """
    print("\n" + "="*50)
    print("Step 1: Training the Backbone and the Final Classifier")
    print("="*50)
    
    model.to(device)
    
    # використовую CrossEntropyLoss, оскільки у нас задача багатокласової класифікації
    # вона автоматично застосовує LogSoftmax до наших сирих логітів
    criterion = nn.CrossEntropyLoss()
    
    # AdamW - стандарт 
    # він краще регуляризує ваги (weight decay) порівняно зі звичайним Adam.
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct_final = 0
        total_samples = 0
        
        # tqdm додає гарний прогрес бар
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for X_batch, y_batch in loop:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            
            # робимо forward pass. на етапі навчання модель повертає словник усіх хедів.
            outputs = model(X_batch)
            
            # для етапу 1 цікавить лише фінальний вихід (backbone -> final_head)
            logits_final = outputs["head_final"]
            
            # рахуємо loss тільки для фінального хеду
            loss = criterion(logits_final, y_batch)
            loss.backward() # градієнти
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # статистика
            total_loss += loss.item()
            preds = torch.argmax(logits_final, dim=1)
            correct_final += (preds == y_batch).sum().item()
            total_samples += y_batch.size(0)
            
            loop.set_postfix(loss=loss.item(), acc=correct_final/total_samples)
            
        # валідація після кожної епохи
        val_accs = evaluate(model, val_loader, device)
        print(f"Val Accuracies -> Final: {val_accs.get('head_final', 0):.4f}")

    return model


def train_stage_2(
    model: nn.Module, 
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    epochs: int = 10, 
    device: str = "cpu"
) -> nn.Module:
    """
    етап 2: заморозка Backbone та навчання проміжних хедів (Early Exits).
    метою є навчити проміжні шари робити точні передбачення на основі
    вже сформованих, якісних hidden states від SSM.
    """
    print("\n" + "="*50)
    print("Step 2: Freezing the Backbone + Training the Intermediate Heads")
    print("="*50)

    # 1. заморозка backbone
    for param in model.input_proj.parameters():
        param.requires_grad = False
    for param in model.layers.parameters():
        param.requires_grad = False
    # 2. переконаємся що всі хеди заморожені    
    for param in model.heads.parameters():
        param.requires_grad = True

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # передаємо в оптимізатор лише параметри хедів
    # це економить память і гарантує що backbone не зміниться
    optimizer = optim.AdamW(model.heads.parameters(), lr=2e-3)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train Heads]")
        for X_batch, y_batch in loop:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            
            # збірка loss з усіх проміжних хедів (ігнорується фінальна бо вона вже навчена)
            loss = 0.0
            for head_name, logits in outputs.items():
                if head_name != "head_final":
                    loss += criterion(logits, y_batch)                    
            # усереднюємо loss між кількістю проміжних хедів. для стабільності
            loss = loss / (len(outputs) - 1)
            
            loss.backward()
            optimizer.step() 
                      
            total_loss += loss.item()
            loop.set_postfix(avg_heads_loss=loss.item())
            
        val_accs = evaluate(model, val_loader, device)
        acc_str = " | ".join([f"{k}: {v:.3f}" for k, v in val_accs.items()])
        print(f"Val Accuracies -> {acc_str}")

    return model


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, device: str) -> Dict[str, float]:
    """
    допоміжна функція для оцінки точності всіх хедів моделі на валідаційному датасеті
    
    """
    model.eval()
    correct_counts = {}
    total_samples = 0
    
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        
        total_samples += y_batch.size(0)
        
        # рахуємо правильні відповіді для кожної голови окремо
        for head_name, logits in outputs.items():
            if head_name not in correct_counts:
                correct_counts[head_name] = 0
                
            preds = torch.argmax(logits, dim=1)
            correct_counts[head_name] += (preds == y_batch).sum().item()
            
    # кількість -> відсотки (точність)
    accuracies = {head: count / total_samples for head, count in correct_counts.items()}
    return accuracies


def train_pipeline(model: nn.Module, dataloaders: dict, device: str = "cpu") -> nn.Module:
    """
    запускає етап 1, потім етап 2, і повертає повністю готову модель!
    
    """
    # етап 1 (навчаємо тільки backbone та кінець, ~15 епох)
    model = train_stage_1(
        model, 
        dataloaders['train'], 
        dataloaders['val'], 
        epochs=15, 
        device=device
    )
    
    # етап 2 (навчаємо ранні виходи, backbone фіксований, ~10 епох)
    model = train_stage_2(
        model, 
        dataloaders['train'], 
        dataloaders['val'], 
        epochs=10, 
        device=device
    )
    
    print("\nTraining successfully completed!")
    return model