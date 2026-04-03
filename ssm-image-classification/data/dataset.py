import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class ToSequence:
    def __call__(self, x):
        return x.view(-1, 1)

def get_dataloaders(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        ToSequence()
    ])

    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    return train_loader, test_loader