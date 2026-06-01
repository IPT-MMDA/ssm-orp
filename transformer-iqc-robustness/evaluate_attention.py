import torch
import csv
import numpy as np
from pathlib import Path
from model import VerifiableAttention
from sdp_verifier import verify_attention_softmax
import warnings
import pandas as pd


def run_robustness_evaluation():
    torch.manual_seed(1)
    np.random.seed(1)

    SEQ_LEN = 2
    EMBED_DIM = 64
    HEAD_DIM = 64

    model = VerifiableAttention(embed_dim=EMBED_DIM, head_dim=HEAD_DIM)

    x_clean = torch.randn(SEQ_LEN, EMBED_DIM)

    attack_scenarios = [
        {"name": "Clean (No Attack)", "epsilon": 0.0001},
        {"name": "Single Char Typo", "epsilon": 0.05},
        {"name": "Synonym (Close)", "epsilon": 0.2},
        {"name": "Synonym (Distant)", "epsilon": 0.5},
        {"name": "Adversarial Swap", "epsilon": 1.2},
        {"name": "Complete Context Shift", "epsilon": 2.5},
    ]

    results = []

    for attack in attack_scenarios:
        eps = attack["epsilon"]

        max_attn, status = verify_attention_softmax(model, x_clean, epsilon=eps)

        if max_attn is not None:
            results.append({
                "scenario": attack["name"],
                "epsilon": eps,
                "certified_max_attention": round(max_attn, 4),
                "solver_status": status
            })

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "transformer_robustness_metrics.csv"

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scenario", "epsilon", "certified_max_attention", "solver_status"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Metrics saved to {out_path}")

    metrics = pd.read_csv(out_path)
    print(f"\n{metrics}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    run_robustness_evaluation()
