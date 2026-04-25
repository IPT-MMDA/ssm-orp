import torch
import pytest
import math
from ssm_early_exit.model import SimplifiedSSMBlock, DeepSSM

def test_ssm_block_output_shape():

    # перевіряє, чи базовий блок SSM зберігає правильні розмірності тензора

    batch_size = 4
    seq_len = 32
    d_model = 16
    
    block = SimplifiedSSMBlock(d_model=d_model)
    x = torch.randn(batch_size, seq_len, d_model)
    
    out = block(x)
    
    # вихідна розмірність має точно збігатися з вхідною
    assert out.shape == (batch_size, seq_len, d_model), \
        f"Expected dimensions {(batch_size, seq_len, d_model)}, received {out.shape}"

def test_deep_ssm_forward_training():
    """
    перевіряє метод forward() для навчання.
    модель має повернути словник із прогнозами від усіх лінійних хедів.
    """
    batch_size = 4
    seq_len = 64
    input_dim = 1
    d_model = 16
    n_layers = 8
    num_classes = 3
    
    model = DeepSSM(input_dim=input_dim, d_model=d_model, n_layers=n_layers, num_classes=num_classes)
    x = torch.randn(batch_size, seq_len, input_dim)
    
    outputs = model(x)
    
    assert isinstance(outputs, dict), "During training, the model must return the vocabulary"
    
    # перевіряємо чи є виходи для всіх заявлених точок (L/4, L/2, 3L/4, Final)
    expected_heads = ["head_1/4", "head_1/2", "head_3/4", "head_final"]
    for head_name in expected_heads:
        assert head_name in outputs, f"No output for {head_name}"
        assert outputs[head_name].shape == (batch_size, num_classes), \
            f"Incorrect dimension of logits for {head_name}"

def test_deep_ssm_forward_inference_early_exit():
    """
    перевіряє механізм Early Exiting
    максимальна ентропія для 3 класів = ln(3) ≈ 1.098.
    якщо поставити поріг > 1.1, модель повинна вийти на першому доступному хеді.
    """
    model = DeepSSM(input_dim=1, d_model=16, n_layers=8, num_classes=3)
    x = torch.randn(2, 32, 1) # batch = 2
    
    high_threshold = 2.0 
    
    logits, exit_head = model.forward_inference(x, threshold=high_threshold)
    
    assert logits.shape == (2, 3)
    assert exit_head == "head_1/4", f"The model was supposed to appear on head_1/4, but it appeared on {exit_head}"

def test_deep_ssm_forward_inference_late_exit():
    """
    перевіряє механізм Early Exiting при жорстких вимогах
    якщо поставити поріг < 0 (що неможливо для ентропії), модель пройде всі шари
    """
    model = DeepSSM(input_dim=1, d_model=16, n_layers=8, num_classes=3)
    x = torch.randn(2, 32, 1)
    
    impossible_threshold = -1.0 
    
    logits, exit_head = model.forward_inference(x, threshold=impossible_threshold)
    
    assert logits.shape == (2, 3)
    # модель має дійти до самого кінця
    assert exit_head == "head_final", f"The model supposed to run to end, but it exited at {exit_head}"

def test_backbone_freezing_logic():
    """
    перевіряє можливість правильного заморожування ваг для етапу 2 (навчання проміжних хедів)
    """
    model = DeepSSM(n_layers=4)
    
    for param in model.input_proj.parameters():
        param.requires_grad = False
    for param in model.layers.parameters():
        param.requires_grad = False
        
    for param in model.layers.parameters():
        assert param.requires_grad is False, "The SSM parameters for layers not frozen"
        
    for param in model.heads.parameters():
        assert param.requires_grad is True, "The linear heads accidentally froze"