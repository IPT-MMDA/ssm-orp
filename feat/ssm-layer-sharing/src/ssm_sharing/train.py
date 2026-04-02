import time
# start = time.time()

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim

from ssm_sharing.models import SequenceClassifier, StandardMamba, SharedMamba
from ssm_sharing.dataset import get_synthetic_dataloader, DataLoader

available_models = {"standard": StandardMamba,"shared": SharedMamba}


def parse_args():
    parser = argparse.ArgumentParser(description="Train Mamba model for Sequence Classification")
    
    parser.add_argument("--epochs", type=int, default=10, help="Amount of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for DataLoader")
    parser.add_argument("--d-model", type=int, default=128, help="Dimension of secret state (d_model)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for optimizator")
    choices = list(available_models.keys())
    parser.add_argument("--model", type=str, default="", choices=choices, help="Mamba model type")  #choices[0]

    parser.add_argument("--no-save", action="store_true", help="Don't save the final model")
    parser.add_argument("--save-iters", action="store_true", help="Save at the end of each epoch")
    
    return parser.parse_args()

def train(dataloader: DataLoader, mamba: nn.Module, args: argparse.Namespace, num_classes: int, device: torch.device):
    dir_name = "models_saved"
    if args.save_iters or not args.no_save: 
        os.makedirs(dir_name, exist_ok=True)
    start = time.time()

    model = SequenceClassifier(ssm_model=mamba, d_model=args.d_model, num_classes=num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"[Train Started] On {time.time() - start:.2f}s\n\n[Model] {mamba._get_name()}\n[Epochs] {args.epochs}\n[Learning Rate] {args.lr}\n[Device] {device}\n")
    for epoch in range(args.epochs):
        start_ = time.time()
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_idx, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()

            logits = model(X)
            preds = torch.argmax(logits, dim=1)
            correct = (preds == y).sum().item()

            total_correct += correct
            total_samples += y.size(0)

            loss = criterion(logits, y)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        avg_loss = total_loss / len(dataloader)

        pad = len(str(args.epochs))
        epochs = str(epoch+1).zfill(pad)
        print(f"[Time] {time.time() - start_:.2f}s | [Epochs] [{epochs}/{args.epochs}] | [Current LR] {scheduler.get_last_lr()[0]:.6f} | [Loss] {avg_loss:.8f} | [Accuracy] {total_correct/total_samples}")

        if args.save_iters:
            torch.save(model.state_dict(), f"{dir_name}/{mamba._get_name()}_e{epochs}_l{avg_loss:.8f}.pt")

    print(f"\n[Train Finished] {time.time() - start:.2f}s\n")

    if not args.no_save:
        path = f"{dir_name}/{mamba._get_name()}.pt"
        print(f"[Saved] {path}")
        torch.save(model.state_dict(), path)

    del model
    torch.cuda.empty_cache()

def train_launch(mamba: nn.Module, args: argparse.Namespace, n_layers: int, num_classes: int, device: torch.device):
    dataloader = get_synthetic_dataloader(d_model=args.d_model, batch_size=args.batch_size, num_classes=num_classes)
    train(dataloader, mamba(d_model=args.d_model, n_layers=n_layers), args, num_classes, device)

def train_command():
    # feat: add training loop with AdamW, CrossEntropyLoss and CosineAnnealingLR
    args = parse_args()
    num_classes = 2
    n_layers = 6

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mamba = available_models.get(args.model)
    if mamba is not None:
        train_launch(mamba, args, n_layers, num_classes, device)
    else:
        for mamba in available_models.values():
            train_launch(mamba, args, n_layers, num_classes, device)


if __name__ == "__main__":
    train_command()
