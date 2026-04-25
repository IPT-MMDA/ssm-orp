from dataclasses import dataclass
import copy
import torch
import torch.nn as nn


@dataclass
class QuantParams:
    scale: torch.Tensor
    zero_point: torch.Tensor
    qmin: int
    qmax: int
    symmetric: bool


def calc_symmetric_qparams(x: torch.Tensor, num_bits: int = 8) -> QuantParams:
    qmin = -(2 ** (num_bits - 1))
    qmax = (2 ** (num_bits - 1)) - 1

    x_abs_max = x.abs().max()
    scale = x_abs_max / qmax
    if scale.item() == 0:
        scale = torch.tensor(1.0, device=x.device, dtype=x.dtype)

    zero_point = torch.tensor(0, device=x.device, dtype=torch.int32)
    return QuantParams(scale=scale, zero_point=zero_point, qmin=qmin, qmax=qmax, symmetric=True)


def calc_asymmetric_qparams(x: torch.Tensor, num_bits: int = 8) -> QuantParams:
    qmin = -(2 ** (num_bits - 1))
    qmax = (2 ** (num_bits - 1)) - 1

    x_min = x.min()
    x_max = x.max()
    scale = (x_max - x_min) / float(qmax - qmin)
    if scale.item() == 0:
        scale = torch.tensor(1.0, device=x.device, dtype=x.dtype)

    zero_point = torch.round(qmin - x_min / scale).to(torch.int32)
    zero_point = torch.clamp(zero_point, qmin, qmax)

    return QuantParams(scale=scale, zero_point=zero_point, qmin=qmin, qmax=qmax, symmetric=False)


def quantize_tensor(x: torch.Tensor, qparams: QuantParams) -> torch.Tensor:
    q_x = torch.round(x / qparams.scale + qparams.zero_point)
    q_x = torch.clamp(q_x, qparams.qmin, qparams.qmax)
    return q_x.to(torch.int8)


def dequantize_tensor(q_x: torch.Tensor, qparams: QuantParams) -> torch.Tensor:
    return (q_x.float() - qparams.zero_point.float()) * qparams.scale


def quantize_weight_tensor(x: torch.Tensor, symmetric: bool = True, num_bits: int = 8):
    if symmetric:
        qparams = calc_symmetric_qparams(x, num_bits=num_bits)
    else:
        qparams = calc_asymmetric_qparams(x, num_bits=num_bits)
    q_x = quantize_tensor(x, qparams)
    return q_x, qparams


def get_abc_named_params(model: nn.Module):
    return {
        "A": ("A", model.A),
        "B": ("B", model.B),
        "C": ("C", model.C),
    }


def set_parameter_by_name(model: nn.Module, param_name: str, new_value: torch.Tensor):
    setattr(model, param_name, nn.Parameter(new_value))


def run_calibration(model, calibration_batches, device):
    """
    Minimal calibration pass:
    - run several representative batches through the baseline model
    - confirms model behaves normally before PTQ
    """
    model.eval()
    with torch.no_grad():
        for batch in calibration_batches:
            x = batch["inputs"].to(device)
            _ = model(x)


def get_calibration_batches(loader, num_batches=10):
    batches = []
    for idx, batch in enumerate(loader):
        if idx >= num_batches:
            break
        batches.append(batch)
    return batches


def build_quantized_model(model: nn.Module, symmetric: bool = True, num_bits: int = 8):
    q_model = copy.deepcopy(model)
    q_model.eval()

    abc_named = get_abc_named_params(q_model)
    quant_artifacts = {}

    for key in ("A", "B", "C"):
        param_name, param = abc_named[key]
        q_w, qparams = quantize_weight_tensor(param.data, symmetric=symmetric, num_bits=num_bits)
        dq_w = dequantize_tensor(q_w, qparams).to(param.dtype)
        set_parameter_by_name(q_model, param_name, dq_w)

        quant_artifacts[key] = {
            "param_name": param_name,
            "int8_weight": q_w.cpu(),
            "scale": float(qparams.scale.item()),
            "zero_point": int(qparams.zero_point.item()),
            "symmetric": qparams.symmetric,
            "shape": list(param.shape),
        }

    return q_model, quant_artifacts


def relative_weight_error(original: torch.Tensor, qdq: torch.Tensor):
    num = torch.norm(original - qdq)
    den = torch.norm(original) + 1e-12
    return float((num / den).item())
