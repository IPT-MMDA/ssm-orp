import pytest
import torch
import torch.nn as nn
from ssm_sharing.models import StandardSSM, SharedSSM

@pytest.fixture
def model_config():
    """Спільні налаштування для всіх моделей у тестах"""
    return {
        "d_model": 64,
        "num_layers": 6,
        "num_classes": 10
    }

@pytest.fixture
def baseline_model(model_config):
    """Створює стандартну модель"""
    return StandardSSM(**model_config)

@pytest.fixture
def shared_model(model_config):
    """Створює модель зі спільними вагами"""
    return SharedSSM(**model_config)


def get_parameter_count(model: nn.Module) -> int:
    """Рахує кількість параметрів, які навчаються."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

def test_parameter_reduction(baseline_model, shared_model):
    """Перевіряємо, що Shared модель легша за Baseline"""
    params_base = get_parameter_count(baseline_model)
    params_shared = get_parameter_count(shared_model)

    print()
    print(f"[Base params] {params_base}")
    print(f"[Shared params]: {params_shared}")
    print()

    assert params_shared < params_base

def test_output_shape(baseline_model, shared_model):
    """Перевіряємо, чи обидві моделі видають правильний розмір тензора на виході"""
    # Створюємо фейковий аудіо-сигнал (batch=2, length=100, d_model=64)
    dummy_input = torch.randn(2, 100, 64)
    
    out_base = baseline_model(dummy_input)
    out_shared = shared_model(dummy_input)
    
    expected_shape = (2, 10)
    assert out_base.shape == expected_shape
    assert out_shared.shape == expected_shape
