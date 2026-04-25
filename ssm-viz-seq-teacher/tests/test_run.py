import torch
from model import SSMStudent

def test_student_forward():
    """Перевірка, що модель приймає вхід потрібного розміру і видає логіти."""
    model = SSMStudent(patch_size=4, embed_dim=32)
    dummy_input = torch.randn(4, 3, 32, 32) # Batch 4, RGB, 32x32
    output = model(dummy_input)
    assert output.shape == (4, 10)

def test_model_params():
    """Перевірка, що учень справді 'мініатюрний'."""
    model = SSMStudent()
    params = sum(p.numel() for p in model.parameters())
    assert params < 1_000_000 # Має бути менше 1М параметрів