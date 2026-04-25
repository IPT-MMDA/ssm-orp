import torch
import torchvision
import torchvision.transforms as T

def get_mnist_loaders(batch_size=64):
    mnist_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.1307,), (0.3081,))
    ])

    trainset = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=mnist_transform
    )

    testset = torchvision.datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=mnist_transform
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