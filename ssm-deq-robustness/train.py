import torch
from models.deq_modules import MDEQSmall
from models.resnet_modules import BalancedResNet
from utils.data_loader import get_data_loaders
from utils.engine import train_and_save

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_data_loaders(batch_size=64)

    # Навчання ResNet
    resnet = BalancedResNet()
    train_and_save(resnet, "ResNet", train_loader, test_loader, device, epochs=50)

    # Навчання MDEQ
    mdeq = MDEQSmall()
    train_and_save(mdeq, "MDEQ", train_loader, test_loader, device, epochs=50)

if __name__ == "__main__":
    main()