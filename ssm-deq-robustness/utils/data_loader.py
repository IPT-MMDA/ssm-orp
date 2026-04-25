from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_data_loaders(batch_size=100):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Виклик через модуль datasets
    train_set = datasets.CIFAR10('./data', train=True, download=True, transform=tf)
    test_set = datasets.CIFAR10('./data', train=False, download=True, transform=tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader