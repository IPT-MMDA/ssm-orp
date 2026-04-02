import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

import torchvision
from torchvision import transforms

import random

def generate_bracket_sequence(seq_len=128, max_depth=10):
    """
    Генерує послідовність дужок та чисел.
    Клас 1: Правильно збалансована.
    Клас 0: Небаланс (зайва дужка або неправильний порядок).
    """
    seq = torch.zeros(seq_len, dtype=torch.long)
    depth = 0
    balanced = random.choice([True, False])
    
    actual_len = random.randint(seq_len // 2, seq_len)
    
    # 0: padding, 1: '(', 2: ')', 3-10: випадкові числа
    for i in range(actual_len):
        if random.random() < 0.3:
            seq[i] = random.randint(3, 10)
            continue
        
        if balanced:
            if depth == 0 or (depth < max_depth and random.random() > 0.5):
                seq[i] = 1 # (
                depth += 1
            else:
                seq[i] = 2 # )
                depth -= 1
        else:
            seq[i] = random.randint(1, 2)

    actual_depth = 0
    is_valid = True
    for val in seq:
        if val == 1: actual_depth += 1
        elif val == 2: actual_depth -= 1
        if actual_depth < 0: is_valid = False
    
    if actual_depth != 0: is_valid = False
    
    return seq, int(is_valid)

def split(dataset, batch_size, num_samples, train_split):
    train_size = int(train_split * num_samples)
    test_size = num_samples - train_size
    
    train_dataset, test_dataset = random_split(
        dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


# feat: implement synthetic dataset generator
def get_synthetic_dataloaders(num_samples: int = 1000, seq_len: int = 64, d_model: int = 128, num_classes: int = 2, batch_size: int = 32, train_split: float = 0.8, **kwargs) -> tuple[DataLoader, DataLoader]:
    """
    Генерує випадкові дані та ділить їх на Train та Test лоадери.
    """
    X = torch.randn(num_samples, seq_len, d_model)
    y = torch.randint(0, num_classes, (num_samples,))
    
    full_dataset = TensorDataset(X, y)
    return split(full_dataset, batch_size, num_samples, train_split)

# feat: implement listops dataset generator
def get_listops_dataloaders(num_samples=2000, seq_len=128, batch_size=32, train_split=0.8, **kwargs):
    X_list, y_list = [], []
    for _ in range(num_samples):
        x, y = generate_bracket_sequence(seq_len)
        X_list.append(x)
        y_list.append(y)
    
    X = torch.stack(X_list) 
    y = torch.tensor(y_list)
    
    full_dataset = TensorDataset(X, y)
    return split(full_dataset, batch_size, num_samples, train_split)


# feat: implement mnist dataset generator
def get_mnist_dataloaders(batch_size=32, **kwargs):
    """Завантажує MNIST і розгортає його в послідовності довжиною 784."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1, 1)) # (1, 28, 28) -> (784, 1)
    ])
    
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader