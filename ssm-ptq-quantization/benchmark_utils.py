import time
import torch


def tensor_num_bytes(tensor: torch.Tensor):
    return tensor.numel() * tensor.element_size()


def abc_fp32_memory_bytes(model):
    total = 0
    total += tensor_num_bytes(model.A.data)
    total += tensor_num_bytes(model.B.data)
    total += tensor_num_bytes(model.C.data)
    return int(total)


def abc_int8_memory_bytes(quant_artifacts):
    total = 0
    for key in ("A", "B", "C"):
        q_tensor = quant_artifacts[key]["int8_weight"]
        total += tensor_num_bytes(q_tensor)
    return int(total)


def memory_savings_percent(fp_bytes: int, q_bytes: int):
    return 100.0 * (fp_bytes - q_bytes) / fp_bytes


@torch.no_grad()
def benchmark_latency(model, sample_batch, device, n_warmup=20, n_runs=100):
    model.eval()
    x = sample_batch["inputs"].to(device)

    for _ in range(n_warmup):
        _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    for _ in range(n_runs):
        _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    return elapsed / n_runs


def speedup_ratio(baseline_latency: float, quant_latency: float):
    return baseline_latency / quant_latency
