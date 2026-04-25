import torch
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np

# 1. Підключення Google Drive
# drive.mount('/content/drive')
# SAVE_PATH_MDEQ = "/content/drive/MyDrive/mdeq"
# SAVE_PATH_RESNET = "/content/drive/MyDrive/sresnet"

SAVE_PATH_RESNET = "checkpoints/sresnet"
SAVE_PATH_MDEQ = "checkpoints/mdeq"

os.makedirs(SAVE_PATH_MDEQ, exist_ok=True)
os.makedirs(SAVE_PATH_RESNET, exist_ok=True)

def train_and_save(model, name, train_loader, test_loader, device, epochs=50):
    print(f"\nTraining {name} | Params: {sum(p.numel() for p in model.parameters()):,}")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                pred = model(data).argmax(dim=1)
                correct += pred.eq(target).sum().item()

        acc = 100. * correct / len(test_loader.dataset)
        print(f"Epoch {epoch:02d} | Loss: {train_loss/len(train_loader):.4f} | Acc: {acc:.2f}%")

        # найкраща модель
        if name == SAVE_PATH_MDEQ:
          save_path = SAVE_PATH_MDEQ

        elif name == SAVE_PATH_RESNET:
          save_path = SAVE_PATH_RESNET

        else:
          print("Змінена назва моделі або шлях збереження")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f"{save_path}/{name}_best.pth")
            print(f" Best model saved!")

        # Збереження кожні 5 епох
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"{save_path}/{name}_epoch_{epoch}.pth")
            print(f" Checkpoint saved (Epoch {epoch})")


def test_robustness(model, loader, device, name, attack_func=None, eps=0.0, corruption_func=None, corruption_type='gaussian_noise', severity=1):
    model.eval()
    correct, total = 0, 0
    all_iters = []
    all_residuals = []

    for data, target in loader:
        data, target = data.to(device), target.to(device)

        if corruption_func:
            data = corruption_func(data, corruption_type=corruption_type, severity=severity)

        if attack_func:
            data = attack_func(model, data, target, eps)

        with torch.no_grad():
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

            #  статистика DEQ
            if hasattr(model, 'last_info') and model.last_info:
                if 'nstep' in model.last_info:
                    all_iters.append(model.last_info['nstep'])

                res_keys = ['lowest_res', 'abs_lowest', 'rel_lowest', 'res']
                # found_res = False
                for key in res_keys:
                    if key in model.last_info:
                        val = model.last_info[key]
                        # тензор у число
                        res_val = val.mean().item() if torch.is_tensor(val) else val
                        all_residuals.append(res_val)
                        # found_res = True
                        break

    acc = correct / total
    avg_iters = np.mean(all_iters) if all_iters else float('nan')
    avg_res = np.mean(all_residuals) if all_residuals else float('nan')

    return acc, avg_iters, avg_res