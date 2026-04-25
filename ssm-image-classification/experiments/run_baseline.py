import torch
import torch.nn as nn
import torch.optim as optim

from configs.config import Config
from data.dataset import get_dataloaders
from models.baseline_cnn import BaselineCNN
from training.train import train_one_epoch
from training.evaluate import evaluate


def main():
    config = Config()

    train_loader, test_loader = get_dataloaders(config)

    model = BaselineCNN(config).to(config.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    for epoch in range(config.num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, config.device
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, config.device
        )

        print(
            f"Epoch {epoch+1}/{config.num_epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
        )


if __name__ == "__main__":
    main()