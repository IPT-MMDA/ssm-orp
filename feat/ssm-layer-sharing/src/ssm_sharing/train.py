import time
# start = time.time()

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim

from ssm_sharing.models import SequenceClassifier, StandardMamba, SharedMamba
from ssm_sharing.dataset import get_synthetic_dataloaders, get_listops_dataloaders, DataLoader
from ssm_sharing.evaluate import Perturbator, Evaluator

available_models = {"standard": StandardMamba,"shared": SharedMamba}


def parse_args():
    parser = argparse.ArgumentParser(description="Train Mamba model for Sequence Classification")
    
    choices = list(available_models.keys())
    parser.add_argument("--model", type=str, default="", choices=choices, help="Mamba model type")  #choices[0]

    parser.add_argument("--epochs", type=int, default=10, help="Amount of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for DataLoader")
    parser.add_argument("--d-model", type=int, default=128, help="Dimension of secret state (d_model)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for optimizator")

    parser.add_argument("--samples", type=int, default=1000, help="Amount of samples generated")
    parser.add_argument("--sequence-length", type=int, default=64, help="Length of sequence")
    parser.add_argument("--layers", type=int, default=6, help="Amount of layers in mamba")
    parser.add_argument("--classes", type=int, default=2, help="Amount of classes to classify")
    parser.add_argument("--split", type=float, default=0.8, help="Split percentage of train/test")

    parser.add_argument("--no-save", action="store_true", help="Don't save the final model")
    parser.add_argument("--save-iters", action="store_true", help="Save at the end of each epoch")
    
    return parser.parse_args()

def train(train_dataloader: DataLoader, test_dataloader: DataLoader, mamba: nn.Module, args: argparse.Namespace, device: torch.device):
    dir_name = "models_saved"
    if args.save_iters or not args.no_save: 
        os.makedirs(dir_name, exist_ok=True)
    start = time.time()

    model = SequenceClassifier(ssm_model=mamba, d_model=args.d_model, num_classes=args.classes)
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

        for batch_idx, (X, y) in enumerate(train_dataloader):
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

        acc_mean, interval = Evaluator.run_stress_test(model, test_dataloader, device, Perturbator.apply_nothing, n_runs=10)

        avg_loss = total_loss / len(train_dataloader)

        time_took = time.time() - start_
        pad = len(str(args.epochs))
        epochs = str(epoch+1).zfill(pad)
        print(f"[Time] {time_took:.2f}s\n[Epochs] [{epochs}/{args.epochs}] | [Current LR] {scheduler.get_last_lr()[0]:.6f}\n[Loss Train] {avg_loss:.8f} | [Accuracy Train] {total_correct/total_samples:.2f} | [Accuracy Test] {acc_mean:.2f}\n")

        scheduler.step()

        if args.save_iters:
            torch.save(model.state_dict(), f"{dir_name}/{mamba._get_name()}_e{epochs}_l{avg_loss:.8f}.pt")

    print(f"\n[Train Finished] {time.time() - start:.2f}s\n")

    if not args.no_save:
        path = f"{dir_name}/{mamba._get_name()}.pt"
        print(f"[Saved] {path}")
        torch.save(model.state_dict(), path)

    del model
    torch.cuda.empty_cache()
    return time_took

def train_launch(mamba: nn.Module, args: argparse.Namespace, device: torch.device):
    train_loader, test_loader = get_synthetic_dataloaders(
        num_samples=args.samples, 
        seq_len=args.sequence_length,
        d_model=args.d_model,
        num_classes=args.classes,
        batch_size=args.batch_size,
        train_split=args.split
    )

    # train_loader, test_loader = get_listops_dataloaders(
    #     num_samples=args.samples, 
    #     seq_len=args.sequence_length,
    #     batch_size=args.batch_size,
    #     train_split=args.split
    # )
    train(
        train_loader, test_loader,
        mamba(d_model=args.d_model, n_layers=args.layers), args, device
    )

def train_command():
    # feat: add training loop with AdamW, CrossEntropyLoss and CosineAnnealingLR
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mamba = available_models.get(args.model)
    if mamba is not None:
        train_launch(mamba, args, device)
    else:
        for mamba in available_models.values():
            train_launch(mamba, args, device)


if __name__ == "__main__":
    train_command()
