import torch
import pytest
from ssm_early_exit.data import SyntheticAnomalyDataset, get_dataloaders

def test_dataset_length_and_shapes():
    """
    перевіряє чи датасет повертає правильну кількість елементів
    та чи кожен елемент має правильну розмірність і тип даних
    """
    num_samples = 30
    seq_len = 128
    dataset = SyntheticAnomalyDataset(num_samples=num_samples, seq_len=seq_len)
    
    assert len(dataset) == num_samples, "The length of the dataset does not match num_samples"
    
    x, y = dataset[0]
    
    assert x.shape == (seq_len, 1), f"Expected dimension ({seq_len}, 1), received {x.shape}"
    assert y.dim() == 0, "A label must be a scalar"
    
    assert x.dtype == torch.float32, "The data must be of type float32"
    assert y.dtype == torch.long, "Labels must be of type long (int64)"

def test_dataset_class_distribution():

    #перевіряє, чи генератор рівномірно розподіляє семпли між трьома класами

    num_samples = 60
    dataset = SyntheticAnomalyDataset(num_samples=num_samples, seq_len=32)
    
    unique_classes, counts = torch.unique(dataset.labels, return_counts=True)
    
    assert len(unique_classes) == 3, "There must be exactly 3 classes (0, 1, 2)"
    
    expected_count = num_samples // 3
    for count in counts:
        assert count.item() == expected_count, "The classes are unevenly distributed"

def test_dataloaders_batches():
    """
    перевіряє роботу функції get_dataloaders
    наявність потрібних ключів та правильні розмірності батчів
    """
    batch_size = 16
    seq_len = 64
    dataloaders = get_dataloaders(
        num_train=48, 
        num_val=32, 
        num_test=32, 
        seq_len=seq_len, 
        batch_size=batch_size
    )
    
    assert set(dataloaders.keys()) == {'train', 'val', 'test'}, "The required data splits are missing"
    
    train_loader = dataloaders['train']
    x_batch, y_batch = next(iter(train_loader))
    
    assert x_batch.shape == (batch_size, seq_len, 1), "Incorrect data batch size"
    assert y_batch.shape == (batch_size,), "Incorrect batch size for tags"
    
    # перевірка меж класів
    assert torch.all((y_batch >= 0) & (y_batch <= 2)), "The labels fall outside the valid classes [0, 1, 2]"