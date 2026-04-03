import pytest
import torch
from ssm_sharing.models import StandardMamba
from ssm_sharing.models import SharedMamba
from ssm_sharing.utils import count_parameters


def test_baseline_complexity():
    d_model = 128
    n_layers = 4
    
    # Ініціалізуємо стандартну модель
    model = StandardMamba(d_model=d_model, n_layers=n_layers)
    total_params = count_parameters(model)
    
    print(f"\n[Baseline] Total parameters for {n_layers} layers: {total_params}")
    
    # Перевіряємо, що параметри дійсно множаться на кількість шарів
    single_layer_model = StandardMamba(d_model=d_model, n_layers=1)
    single_params = count_parameters(single_layer_model)
    
    assert total_params > single_params * (n_layers - 1)

def test_shared_vs_baseline_params():
    d_model = 128
    n_layers = 6  # Візьмемо глибшу мережу для наочності
    
    baseline = StandardMamba(d_model=d_model, n_layers=n_layers)
    shared = SharedMamba(d_model=d_model, n_layers=n_layers)
    
    params_base = count_parameters(baseline)
    params_shared = count_parameters(shared)
    
    print(f"\n[Baseline 6 layers] Parameters: {params_base}")
    print(f"[Shared 6 layers]   Parameters: {params_shared}")
    
    # Shared модель повинна бути суттєво меншою
    assert params_shared < params_base, "Shared модель не зменшила кількість параметрів!"
    
    # Перевіримо, чи працює forward pass і чи не ламаються розмірності
    # Mamba вимагає CUDA для своїх оптимізованих ядер
    if torch.cuda.is_available():
        dummy_input = torch.randn(2, 64, d_model).cuda()
        baseline = baseline.cuda()
        shared = shared.cuda()
        
        out_base = baseline(dummy_input)
        out_shared = shared(dummy_input)
        
        assert out_base.shape == (2, 64, d_model)
        assert out_shared.shape == (2, 64, d_model)
    else:
        pytest.skip("CUDA не знайдена, пропускаємо forward pass тест для Mamba-ядер.")
