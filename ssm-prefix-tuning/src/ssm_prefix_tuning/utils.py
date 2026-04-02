"""Utility helpers shared across the whole package."""

import json
import os
import random
from dataclasses import asdict
from typing import Any


def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch (CPU + all GPUs)."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preferred: str = "cuda"):
    """Return the requested device, falling back to CPU with a warning."""
    import torch

    if preferred == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[utils] CUDA not available — using CPU.")
    return torch.device("cpu")


def count_parameters(model, ) -> dict[str, Any]:
    """Return total, trainable, and frozen parameter counts for a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_ratio": round(trainable / total, 6) if total > 0 else 0.0,
    }


def save_results(results: list[Any], path: str) -> None:
    """Serialise a list of EpochResult (or any dataclass) to JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    serialisable = []
    for r in results:
        if hasattr(r, "__dataclass_fields__"):
            serialisable.append(asdict(r))
        elif hasattr(r, "__dict__"):
            serialisable.append(r.__dict__)
        else:
            serialisable.append(r)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(serialisable, fh, indent=2)


def load_results(path: str) -> list[dict]:
    """Load previously saved results from a JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
