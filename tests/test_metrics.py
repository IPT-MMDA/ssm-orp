import torch
import math
import pytest
from ssm_early_exit.utils import calculate_entropy, calculate_confidence_interval

def test_entropy_certainty():
    """
    якщо модель на 100% впевнена в одному класі ентропія має бути близькою до 0
    """
    # логіти: дуже високе значення для класу 0, дуже низькі для інших
    # після softmax це дасть ймовірності [1.0, 0.0, 0.0]
    logits = torch.tensor([[100.0, -100.0, -100.0]])
    entropy = calculate_entropy(logits)
    
    assert entropy.shape == (1,)
    # використовую torch.isclose через можливі похибки
    assert torch.isclose(entropy[0], torch.tensor(0.0), atol=1e-5), \
        f"The entropy of a completely certain prediction should be ~0; the result is {entropy[0].item()}"

def test_entropy_uncertainty():
    """
    якщо модель абсолютно не впевнена (всі логіти рівні), 
    ентропія має досягати свого максимуму що дорівнює ln(num_classes).
    """
    num_classes = 3
    # рівні логіти означають рівномірний розподіл ймовірностей: [1/3, 1/3, 1/3]
    logits = torch.tensor([[0.0, 0.0, 0.0]])
    entropy = calculate_entropy(logits)
    
    expected_entropy = math.log(num_classes)
    assert torch.isclose(entropy[0], torch.tensor(expected_entropy), atol=1e-5), \
        f"The maximum entropy for the 3 classes should be ~{expected_entropy}; the result is {entropy[0].item()}"

def test_entropy_batch_processing():
    
    # перевірка, чи функція правильно обробляє багатовимірні тензори
    
    batch_size = 16
    num_classes = 5
    logits = torch.randn(batch_size, num_classes)
    entropy = calculate_entropy(logits)
    
    assert entropy.shape == (batch_size,), "The output size must match the batch size"
    assert torch.all(entropy >= 0), "Entropy cannot be a negative quantity"

def test_confidence_interval_zero_variance():
    """
    якщо всі значення у вибірці однакові (дисперсія = 0) то довірчий інтервал 
    має бути нульовим (нижня і верхня межі дорівнюють середньому).
    """
    data = [15.0, 15.0, 15.0, 15.0, 15.0]
    mean, lower, upper = calculate_confidence_interval(data)
    
    assert mean == 15.0
    assert lower == 15.0
    assert upper == 15.0

def test_confidence_interval_normal_data():
    """
    перевірка розрахунку на звичайних даних. межі мають бути коректними 
    і симетричними відносно середнього значення
    """
    data = [10.0, 12.0, 11.0, 13.0, 9.0] # середнє = 11.0
    mean, lower, upper = calculate_confidence_interval(data, confidence=0.95)
    
    assert mean == 11.0
    assert lower < mean, "The lower limit must be strictly less than the mean"
    assert upper > mean, "The upper limit must be strictly greater than the mean"
    
    # перевірка симетрії довірчого інтервалу
    margin_lower = mean - lower
    margin_upper = upper - mean
    assert math.isclose(margin_lower, margin_upper, rel_tol=1e-5), "Інтервал Стьюдента має бути симетричним"

def test_confidence_interval_small_sample():
    """
    edge case: якщо масив містить лише 1 елемент або він порожній, функція 
    не повинна падати з помилкою ділення на нуль, а має безпечно повернути наявне значення
    """
    data = [42.0]
    mean, lower, upper = calculate_confidence_interval(data)
    
    assert mean == 42.0
    assert lower == 42.0
    assert upper == 42.0