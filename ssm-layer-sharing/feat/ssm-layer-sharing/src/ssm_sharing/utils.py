import argparse

import torch
import torch.nn as nn

from ssm_sharing.models import StandardMamba, SharedMamba
from ssm_sharing.dataset import get_synthetic_dataloaders, get_listops_dataloaders, get_mnist_dataloaders, DataLoader
from ssm_sharing.evaluate import Perturbator


AVAILABLE_MODELS = {"standard": StandardMamba,"shared": SharedMamba}
AVAILABLE_PERTURBATIONS = {"nothing": Perturbator.apply_nothing, "masking": Perturbator.apply_masking, "noise": Perturbator.apply_gaussian_noise}
AVAILABLE_GENERATORS = {"synthetic": get_synthetic_dataloaders, "listops": get_listops_dataloaders, "mnist": get_mnist_dataloaders}


def parse_args():
    parser = argparse.ArgumentParser(description="Train Mamba model for Sequence Classification")
    
    models = list(AVAILABLE_MODELS.keys())
    perturbations = list(AVAILABLE_PERTURBATIONS.keys())
    generators = list(AVAILABLE_GENERATORS.keys())
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


def count_parameters(model: torch.nn.Module) -> int:
    """Returns sum of amount of trainable parameters"""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
