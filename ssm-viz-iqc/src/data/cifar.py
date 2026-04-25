import torch
import torchvision
import torchvision.transforms as T


def get_cifar_loaders(batch_size=64):

    cifar_transform = T.Compose([
        T.Resize((16, 16)),
        T.ToTensor(),
        T.Normalize(
            (0.4914, 0.4822, 0.4465),   
            (0.2023, 0.1994, 0.2010)   
        )
    ])

    trainset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=cifar_transform
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=cifar_transform
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False
    )

    return trainloader, testloader