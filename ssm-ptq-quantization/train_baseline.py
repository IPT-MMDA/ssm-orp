import os
import json
import torch
import torch.nn as nn

from config import DataConfig, TrainConfig
from data import build_dataloaders
from model import build_model
from utils import set_seed, get_device, ensure_dir
from eval_utils import evaluate_model


def train_one_epoch(model, loader, optimizer, criterion, device, task_type):
    model.train()
    total_loss = 0.0

    for batch in loader:
        x = batch["inputs"].to(device)
        y = batch["targets"].to(device)

        optimizer.zero_grad()
        outputs = model(x)

        if task_type == "classification":
            loss = criterion(outputs, y)
        else:
            loss = criterion(outputs, y.float())

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


def main():
    data_cfg = DataConfig()
    train_cfg = TrainConfig()

    set_seed(data_cfg.seed)
    device = get_device(train_cfg.device)
    ensure_dir(train_cfg.save_dir)

    train_loader, val_loader, test_loader = build_dataloaders(data_cfg)
    model = build_model(data_cfg).to(device)

    if data_cfg.task_type == "classification":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay
    )

    best_metric = None
    history = []

    for epoch in range(1, train_cfg.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, data_cfg.task_type
        )
        val_metrics = evaluate_model(model, val_loader, device, data_cfg.task_type)

        if data_cfg.task_type == "classification":
            current_metric = val_metrics["f1"]
        else:
            current_metric = -val_metrics["mse"]

        improved = best_metric is None or current_metric > best_metric
        if improved:
            best_metric = current_metric
            checkpoint_path = os.path.join(train_cfg.save_dir, train_cfg.checkpoint_name)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "data_cfg": data_cfg.__dict__,
                    "task_type": data_cfg.task_type
                },
                checkpoint_path
            )

        row = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        history.append(row)
        print(row)

    # final test with best checkpoint
    checkpoint_path = os.path.join(train_cfg.save_dir, train_cfg.checkpoint_name)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate_model(model, test_loader, device, data_cfg.task_type)

    out = {
        "best_checkpoint": checkpoint_path,
        "test_metrics": test_metrics,
        "history": history
    }

    with open(os.path.join(train_cfg.save_dir, "baseline_training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\nSaved baseline checkpoint:", checkpoint_path)
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()
