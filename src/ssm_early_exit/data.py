import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict

class SyntheticAnomalyDataset(Dataset):
    """
    генератор часових рядів для Early Exiting в SSM
    генерує 3 класи:
    - нормальний сигнал
    - раптова аномалія (sudden/obvious)
    - поступова аномалія (subtle/long-term)
    """
    def __init__(self, num_samples: int = 3000, seq_len: int = 512, seed: int = 42):
        super().__init__()
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.seed = seed
        self.data, self.labels = self._generate_dataset()

    def _generate_dataset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        np.random.seed(self.seed)
        
        # розподіляю семпли між 3ма класами
        samples_per_class = self.num_samples // 3
        
        X_all = []
        y_all = []

        # базовий часовий вектор t
        t = np.linspace(0, 4 * np.pi, self.seq_len)

        for class_idx in range(3):
            for _ in range(samples_per_class):
                # базовий сигнал: дві частоти + шум
                f1, f2 = np.random.uniform(0.5, 1.5), np.random.uniform(2.0, 3.0)
                signal = np.sin(f1 * t) + 0.5 * np.cos(f2 * t)
                noise = np.random.normal(0, 0.2, self.seq_len)
                x = signal + noise

                if class_idx == 1:
                    # sudden anomaly: різкий стрибок на короткому проміжку
                    spike_start = np.random.randint(int(0.1 * self.seq_len), int(0.8 * self.seq_len))
                    spike_duration = np.random.randint(5, 15) # дуже короткочасно
                    spike_amplitude = np.random.uniform(4.0, 6.0) # очевидна зміна
                    
                    # стрибок випадкового знаку
                    sign = np.random.choice([-1, 1])
                    x[spike_start:spike_start+spike_duration] += sign * spike_amplitude

                elif class_idx == 2:
                    # subtle anomaly: повільний дрейф , починається з середини
                    # також вимагає довгострокової пам'яті для виявлення (smm state integration)
                    drift_start = np.random.randint(int(0.3 * self.seq_len), int(0.5 * self.seq_len))
                    drift_slope = np.random.uniform(0.01, 0.02)
                    
                    # масив дрейфу: 0 до drift_start, потім лінійне зростання
                    drift = np.zeros(self.seq_len)
                    drift[drift_start:] = np.arange(self.seq_len - drift_start) * drift_slope
                    
                    sign = np.random.choice([-1, 1])
                    x += sign * drift

                X_all.append(x)
                y_all.append(class_idx)

        X_tensor = torch.tensor(np.array(X_all), dtype=torch.float32).unsqueeze(-1)
        y_tensor = torch.tensor(np.array(y_all), dtype=torch.long)

        # перемішую датасет
        indices = torch.randperm(len(X_tensor))
        return X_tensor[indices], y_tensor[indices]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]


def get_dataloaders(
    num_train: int = 6000, 
    num_val: int = 1500, 
    num_test: int = 1500, 
    seq_len: int = 512, 
    batch_size: int = 64
) -> Dict[str, DataLoader]:
    
    #створення DataLoader-ів.
    #повертає словник з train, val та 'test' завантажувачами.
    
    # різні сіди для тренування та тестування щоб уникнути витоку даних
    train_dataset = SyntheticAnomalyDataset(num_samples=num_train, seq_len=seq_len, seed=42)
    val_dataset = SyntheticAnomalyDataset(num_samples=num_val, seq_len=seq_len, seed=43)
    test_dataset = SyntheticAnomalyDataset(num_samples=num_test, seq_len=seq_len, seed=44)

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    }
    return dataloaders