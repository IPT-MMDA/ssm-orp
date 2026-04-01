import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from models import SequenceClassifier, StandardMamba, SharedMamba
from dataset import get_synthetic_dataloader

available_models = {"standard": StandardMamba,"shared": SharedMamba}


def parse_args():
    parser = argparse.ArgumentParser(description="Train Mamba model for Sequence Classification")
    
    parser.add_argument("--epochs", type=int, default=10, help="Amount of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for DataLoader")
    parser.add_argument("--d-model", type=int, default=128, help="Dimension of secret state (d_model)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for optimizator")
    choices = list(available_models.keys())
    parser.add_argument("--model-type", type=str, default=choices[0], choices=choices, help="Mamba model type")
    
    return parser.parse_args()

if __name__ == "__main__":
    # 3. Тренувальний цикл (The Loop)
    # Напиши стандартний PyTorch цикл на 5-10 епох.
    # Оптимізатор AdamW, лосс — CrossEntropyLoss.
    # Навчи обидві моделі (окремо Standard, окремо Shared) до збіжності на цих синтетичних даних.
    # Вони мають запам'ятати датасет (точність має наблизитися до 100%).

    # feat: add training loop with AdamW (написав цикл)
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train Started]\n\n[Model] {args.model_type}\n[Epochs] {args.epochs}\n[Learning Rate] {args.lr}\n[Device] {device}\n")
    dataloader = get_synthetic_dataloader(d_model=args.d_model, batch_size=args.batch_size, num_classes=2)
    mamba = available_models.get(args.model_type)(d_model=args.d_model, n_layers=6)
    model = SequenceClassifier(ssm_model=mamba, d_model=args.d_model, num_classes=2)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    # 1. Оптимізатор: AdamW
    # Ми не використовуємо звичайний Adam.
    # Ми беремо torch.optim.AdamW.
    # Ваги в нашій SharedMamba відчувають шалене навантаження,
    # бо вони одночасно відповідають за витягування ознак і на першому шарі (де дані сирі), і на останньому (де дані вже абстрактні).
    # AdamW коректно застосовує Weight Decay (регуляризацію), не дозволяючи вагам розростатися до гігантських значень.
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()

            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()

            # 2. Захист: Gradient Clipping (КРИТИЧНО)
            # Це наш запобіжник.
            # Перед кожним кроком оптимізатора (optimizer.step()) ми будемо примусово обрізати довжину вектора градієнтів, якщо він стає занадто великим.
            # Ти будеш писати такий рядок: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0). Без цього наша Shared-модель не виживе.
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # TODO: 3. Scheduler: Linear Warmup + Cosine Decay
            # На старті навчання ваги ініціалізовані випадково.
            # Якщо ми відразу дамо великий крок навчання (Learning Rate), градієнти зійдуть з розуму через N проходів.
            # Тому ми почнемо з мікроскопічних кроків і будемо плавно їх збільшувати перші 10% епох (Warmup).
            # Коли модель "намацає" правильний напрямок,
            # ми почнемо плавно зменшувати крок по косинусоїді (Cosine Decay), щоб акуратно спуститися в локальний мінімум лосс-функції.
            optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        pad = len(str(args.epochs))
        print(f"[Epochs] [{(epoch+1):>{pad}}/{args.epochs}] | [Loss] {avg_loss:.8f}")
    print("\n[Train Finished]")
