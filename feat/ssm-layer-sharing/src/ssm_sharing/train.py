import time
# start = time.time()

import argparse
import os

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from ssm_sharing.models import SequenceClassifier, StandardMamba, SharedMamba
from ssm_sharing.dataset import get_synthetic_dataloaders, get_listops_dataloaders, get_mnist_dataloaders, DataLoader
from ssm_sharing.evaluate import Perturbator, Evaluator

available_models = {"standard": StandardMamba,"shared": SharedMamba}
available_perturbations = {"nothing": Perturbator.apply_nothing, "masking": Perturbator.apply_masking, "noise": Perturbator.apply_gaussian_noise}
available_generators = {"synthetic": get_synthetic_dataloaders, "listops": get_listops_dataloaders, "mnist": get_mnist_dataloaders}


def parse_args():
    parser = argparse.ArgumentParser(description="Train Mamba model for Sequence Classification")
    
    models = list(available_models.keys())
    perturbations = list(available_perturbations.keys())
    generators = list(available_generators.keys())
    parser.add_argument("--model", type=str, default=None, choices=models, help="Mamba model type")  #models[0]
    parser.add_argument("--perturbation", type=str, default=perturbations[0], choices=perturbations, help="Evaluate with perturbation function")
    parser.add_argument("--dataset", type=str, default=generators[0], choices=generators, help="Dataset to train on")

    parser.add_argument("--epochs", type=int, default=10, help="Amount of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for DataLoader")
    parser.add_argument("--d-model", type=int, default=128, help="Dimension of secret state (d_model)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for optimizator")

    parser.add_argument("--samples", type=int, default=1000, help="Amount of samples generated")
    parser.add_argument("--sequence-length", type=int, default=64, help="Length of sequence")
    parser.add_argument("--layers", type=int, default=6, help="Amount of layers in mamba")
    parser.add_argument("--classes", type=int, default=None, help="Amount of classes to classify")
    parser.add_argument("--split", type=float, default=0.8, help="Split percentage of train/test")
    parser.add_argument("--mask", type=float, default=0.2, help="Parameter for perturbartions")
    parser.add_argument("--runs", type=int, default=10, help="Amount of runs in final evaluation")

    parser.add_argument("--vocab-size", type=int, default=None, help="Size of alphabet")
    parser.add_argument("--input-dim", type=int, default=None, help="Dimenions of dataset (mnist)")

    parser.add_argument("--no-save", action="store_true", help="Don't save the final model")
    parser.add_argument("--save-iters", action="store_true", help="Save at the end of each epoch")

    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation")

    args = parser.parse_args()

    if args.dataset == "mnist":
        if args.input_dim is None or args.classes is None:
            parser.error(
                "\n[Error] For '--dataset mnist', you MUST specify:\n"
                "  --input-dim (e.g., 1)\n"
                "  --classes (e.g., 10)"
            )
            
    elif args.dataset == "listops":
        if args.vocab_size is None:
            parser.error(
                "\n[Error] For '--dataset listops', you MUST specify:\n"
                "  --vocab-size (e.g., 11)"
            )

    if args.classes is None:
        args.classes = 2

    return args

def train(train_dataloader: DataLoader, test_dataloader: DataLoader, mamba: nn.Module, args: argparse.Namespace, device: torch.device):
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
        acc_mean, interval = Evaluator.run_stress_test(model, test_dataloader, device, available_perturbations.get(args.perturbation, Perturbator.apply_nothing), args.mask, n_runs=args.runs)
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
            acc_mean, interval = Evaluator.run_stress_test(model, test_dataloader, device, available_perturbations.get(args.perturbation, Perturbator.apply_nothing), args.mask, n_runs=args.runs)
        else:
            acc_mean, interval = Evaluator.run_stress_test(model, test_dataloader, device, available_perturbations.get(args.perturbation, Perturbator.apply_nothing), args.mask, n_runs=2)
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
    train_loader, test_loader = available_generators[args.dataset](
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
