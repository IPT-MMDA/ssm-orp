import time

import os

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from ssm_sharing.models import SequenceClassifier
from ssm_sharing.dataset import DataLoader
from ssm_sharing.evaluate import Perturbator, Evaluator

from ssm_sharing.utils import argparse, parse_args, AVAILABLE_MODELS, AVAILABLE_PERTURBATIONS, AVAILABLE_GENERATORS


def train(train_dataloader: DataLoader, test_dataloader: DataLoader, mamba: nn.Module, args: argparse.Namespace, device: torch.device):
    """
    Main function for training

    Creates folders for models (if needed).
    Creates or loads model and evals or starts training it using `CrossEntropyLoss`, optimizer `AdamW` and lr_scheduler `CosineAnnealingLR`.
    Saves model in the end and/or on each epoch.
    """
    pad = len(str(args.epochs))
    dir_name = "models_saved"
    if args.save_iters or not args.no_save: 
        os.makedirs(dir_name, exist_ok=True)
    start = time.time()

    model = SequenceClassifier(ssm_model=mamba, d_model=args.d_model, num_classes=args.classes, vocab_size=args.vocab_size, input_dim=args.input_dim)
    model.to(device)
    
    if args.checkpoint:
        print(f"[Loading Checkpoint] {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    if args.eval_only:
        acc_mean, interval = Evaluator.run_stress_test(model, test_dataloader, device, AVAILABLE_PERTURBATIONS.get(args.perturbation, Perturbator.apply_nothing), args.mask, n_runs=args.runs)
        print(f"[Evaluation] Accuracy: {acc_mean:.5} | Interval: {interval}")
        return 0


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"[Train Started] On {time.time() - start:.2f}s\n\n[Model] {mamba._get_name()}\n[Epochs] {args.epochs}\n[Learning Rate] {args.lr}\n[Device] {device}\n[Dataset] {args.dataset}\n[Perturbation] {args.perturbation}\n[Layers] {args.layers}\n")
    for epoch in range(args.epochs):
        start_ = time.time()
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        pbar = tqdm(train_dataloader, desc=f"Epoch {str(epoch+1).zfill(pad)}/{args.epochs}", leave=False)

        for batch_idx, (X, y) in enumerate(pbar):
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
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        if epoch == args.epochs - 1:
            acc_mean, interval = Evaluator.run_stress_test(model, test_dataloader, device, AVAILABLE_PERTURBATIONS.get(args.perturbation, Perturbator.apply_nothing), args.mask, n_runs=args.runs)
        else:
            acc_mean, interval = Evaluator.run_stress_test(model, test_dataloader, device, AVAILABLE_PERTURBATIONS.get(args.perturbation, Perturbator.apply_nothing), args.mask, n_runs=2)
        avg_loss = total_loss / len(train_dataloader)

        time_took = time.time() - start_
        epochs = str(epoch+1).zfill(pad)
        print(f"[Time] {time_took:.2f}s\n[Epochs] [{epochs}/{args.epochs}] | [Current LR] {scheduler.get_last_lr()[0]:.6f}\n[Loss Train] {avg_loss:.8f} | [Accuracy Train] {total_correct/total_samples:.5f} | [Accuracy Test] {acc_mean:.5f} | [Accuracy Test Interval] {interval}\n")

        scheduler.step()

        if args.save_iters:
            torch.save(model.state_dict(), f"{dir_name}/{mamba._get_name()}_dataset_{args.dataset}_lyrs_{args.layers}_e{epochs}_l{avg_loss:.8f}_testacc_{acc_mean:.5f}.pt")

    print(f"\n[Train Finished] {time.time() - start:.2f}s\n")

    if not args.no_save:
        path = f"{dir_name}/{mamba._get_name()}_dataset_{args.dataset}_lyrs_{args.layers}_l{avg_loss:.8f}_testacc_{acc_mean:.5f}.pt"
        print(f"[Saved] {path}")
        torch.save(model.state_dict(), path)

    del model
    torch.cuda.empty_cache()
    return time_took

def train_launch(mamba: nn.Module, args: argparse.Namespace, device: torch.device):
    """
    Called from `train_command`

    Creates train and test dataset loaders from args.
    Starts train of `SequenceClassifier`.
    """
    train_loader, test_loader = AVAILABLE_GENERATORS[args.dataset](
        num_samples=args.samples, 
        seq_len=args.sequence_length,
        d_model=args.d_model,
        num_classes=args.classes,
        batch_size=args.batch_size,
        train_split=args.split
    )

    return train(
        train_loader, test_loader,
        mamba(d_model=args.d_model, n_layers=args.layers), args, device
    )

def train_command():
    """
    Used when tou type in terminal `train`

    Parses argumets given with train.
    Automaticaly decides which device to use.
    Starts train for needed model.
    """
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mamba = AVAILABLE_MODELS.get(args.model)
    if mamba is not None:
        train_launch(mamba, args, device)
    else:
        for mamba in AVAILABLE_MODELS.values():
            train_launch(mamba, args, device)
