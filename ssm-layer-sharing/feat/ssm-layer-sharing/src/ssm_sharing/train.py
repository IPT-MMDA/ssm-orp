import time

import os

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from ssm_sharing.models import SequenceClassifier
from ssm_sharing.dataset import DataLoader
from ssm_sharing.evaluate import Perturbator, Evaluator

from ssm_sharing.utils import argparse, count_parameters, parse_args, AVAILABLE_MODELS, AVAILABLE_PERTURBATIONS, AVAILABLE_GENERATORS


# def train(train_dataloader: DataLoader, test_dataloader: DataLoader, mamba: nn.Module, args: argparse.Namespace, device: torch.device):
def train(
    train_dataloader: DataLoader, 
    test_dataloader: DataLoader, 
    mamba: nn.Module,
    device: torch.device,
    # learning parameters
    epochs: int = 10,
    lr: float = 1e-3,
    # model parameters
    d_model: int = 128,
    classes: int = 2,
    vocab_size: int = None,
    input_dim: int = None,
    layers: int = 6,
    # training and evaluation
    perturbation: str = list(AVAILABLE_PERTURBATIONS.keys())[0],
    mask: float = 0.2,
    runs: int = 10,
    dataset: str = list(AVAILABLE_GENERATORS.keys())[0],
    # utils
    checkpoint: str = None,
    eval_only: bool = False,
    save_iters: bool = False,
    no_save: bool = False
):
    """
    Main function for training

    Creates folders for models (if needed).
    Creates or loads model and evals or starts training it using `CrossEntropyLoss`, optimizer `AdamW` and lr_scheduler `CosineAnnealingLR`.
    Saves model in the end and/or on each epoch.
    """
    pad = len(str(epochs))
    dir_name = "models_saved"
    if save_iters or not no_save: 
        os.makedirs(dir_name, exist_ok=True)
    start = time.time()

    model = SequenceClassifier(ssm_model=mamba, d_model=d_model, num_classes=classes, vocab_size=vocab_size, input_dim=input_dim)
    model.to(device)
    
    if checkpoint:
        print(f"[Loading Checkpoint] {checkpoint}")
        model.load_state_dict(torch.load(checkpoint, map_location=device))

    if eval_only:
        acc_mean, interval = Evaluator.run_stress_test(model, test_dataloader, device, AVAILABLE_PERTURBATIONS.get(perturbation, Perturbator.apply_nothing), mask, n_runs=runs)
        print(f"[Evaluation] Accuracy: {acc_mean:.5} | Interval: {interval}")
        return model, time.time() - start


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"[Train Started] On {time.time() - start:.2f}s\n\n[Model] {mamba._get_name()}\n[Epochs] {epochs}\n[Learning Rate] {lr}\n[Device] {device}\n[Dataset] {dataset}\n[Perturbation] {perturbation}\n[Layers] {layers}\n")
    for epoch in range(epochs):
        start_ = time.time()
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        pbar = tqdm(train_dataloader, desc=f"Epoch {str(epoch+1).zfill(pad)}/{epochs}", leave=False)

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

        if epoch == epochs - 1:
            acc_mean, interval = Evaluator.run_stress_test(model, test_dataloader, device, AVAILABLE_PERTURBATIONS.get(perturbation, Perturbator.apply_nothing), mask, n_runs=runs)
        else:
            acc_mean, interval = Evaluator.run_stress_test(model, test_dataloader, device, AVAILABLE_PERTURBATIONS.get(perturbation, Perturbator.apply_nothing), mask, n_runs=2)
        avg_loss = total_loss / len(train_dataloader)

        time_took = time.time() - start_
        epochs_ = str(epoch+1).zfill(pad)
        print(f"[Time] {time_took:.2f}s\n[Epochs] [{epochs_}/{epochs}] | [Current LR] {scheduler.get_last_lr()[0]:.6f} | [Trainable Parameters] {count_parameters(model)}\n[Loss Train] {avg_loss:.8f} | [Accuracy Train] {total_correct/total_samples:.5f} | [Accuracy Test] {acc_mean:.5f} | [Accuracy Test Interval] {interval}\n")

        scheduler.step()

        if save_iters:
            torch.save(model.state_dict(), f"{dir_name}/{mamba._get_name()}_dataset_{dataset}_lyrs_{layers}_e{epochs}_l{avg_loss:.8f}_testacc_{acc_mean:.5f}.pt")

    print(f"\n[Train Finished] {time.time() - start:.2f}s\n")

    if not no_save:
        path = f"{dir_name}/{mamba._get_name()}_dataset_{dataset}_lyrs_{layers}_l{avg_loss:.8f}_testacc_{acc_mean:.5f}.pt"
        print(f"[Saved] {path}")
        torch.save(model.state_dict(), path)

    torch.cuda.empty_cache()
    return model, time.time() - start

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

    mamba_instance = mamba(d_model=args.d_model, n_layers=args.layers)

    return train(
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        mamba=mamba_instance,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        d_model=args.d_model,
        classes=args.classes,
        vocab_size=args.vocab_size,
        input_dim=args.input_dim,
        layers=args.layers,
        perturbation=args.perturbation,
        mask=args.mask,
        runs=args.runs,
        dataset=args.dataset,
        checkpoint=args.checkpoint,
        eval_only=args.eval_only,
        save_iters=args.save_iters,
        no_save=args.no_save
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
