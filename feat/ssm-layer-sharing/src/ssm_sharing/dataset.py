# Ми не будемо тренувати модель на безглуздому шумі. Mamba славиться здатністю вловлювати довгострокові залежності (long-range dependencies).
#     Твоя задача: Написати кастомний torch.utils.data.Dataset.
#       Це буде генератор Copying Task (класичний тест для RNN/SSM) або класифікатор частот, де сигнал змішаний із шумом.
#       Модель повинна навчитися ігнорувати шум і "згадувати", що було на початку послідовності.


# 2. Дані (Synthetic Data)
# Не витрачай час на завантаження реальних датасетів.
# Згенеруй X форми (1000, 64, 128) (1000 семплів, довжина 64, d_model 128) за допомогою torch.randn.
# Згенеруй y (мітки класів від 0 до 1) через torch.randint. Створи з цього стандартний DataLoader.

# feat: implement synthetic dataset generator (написав дані)
import torch
from torch.utils.data import TensorDataset, DataLoader

def get_synthetic_dataloader(
        num_samples: int = 1000, 
        seq_len: int = 64, 
        d_model: int = 128, 
        num_classes: int = 2, 
        batch_size: int = 32
    ) -> DataLoader:
    """
    Генерує синтетичний датасет для перевірки збіжності моделі.
    """
    X = torch.randn(num_samples, seq_len, d_model)
    y = torch.randint(0, num_classes, (num_samples,))
    
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)