import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error


@torch.no_grad()
def evaluate_model(model, loader, device, task_type="classification", perturb_fn=None, perturb_kwargs=None):
    model.eval()
    perturb_kwargs = perturb_kwargs or {}

    preds_all = []
    targets_all = []

    for batch in loader:
        x = batch["inputs"].to(device)
        y = batch["targets"].to(device)

        if perturb_fn is not None:
            x = perturb_fn(x, **perturb_kwargs)

        outputs = model(x)

        if task_type == "classification":
            preds = torch.argmax(outputs, dim=-1)
            preds_all.extend(preds.cpu().numpy().tolist())
            targets_all.extend(y.cpu().numpy().tolist())
        else:
            preds_all.append(outputs.detach().cpu())
            targets_all.append(y.detach().cpu())

    if task_type == "classification":
        acc = accuracy_score(targets_all, preds_all)
        f1 = f1_score(targets_all, preds_all, average="macro")
        return {"accuracy": float(acc), "f1": float(f1)}
    else:
        preds_all = torch.cat(preds_all, dim=0).numpy().reshape(-1)
        targets_all = torch.cat(targets_all, dim=0).numpy().reshape(-1)
        mse = mean_squared_error(targets_all, preds_all)
        return {"mse": float(mse)}


def robustness_degradation(clean_metrics, perturbed_metrics, task_type="classification"):
    if task_type == "classification":
        return {
            "accuracy_drop": float(clean_metrics["accuracy"] - perturbed_metrics["accuracy"]),
            "f1_drop": float(clean_metrics["f1"] - perturbed_metrics["f1"]),
        }
    else:
        return {
            "mse_increase": float(perturbed_metrics["mse"] - clean_metrics["mse"]),
        }
