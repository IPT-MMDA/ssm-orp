import torch
import torch.nn.functional as F


def train_model(
    model,
    trainloader,
    testloader,
    optimizer,
    device,
    num_epochs=50,
    patience=10,
    save_path="best_model.pth"
):
    model = model.to(device)

    best_acc = 0.0
    trigger_times = 0

    print(f"Starting training...")

    for epoch in range(num_epochs):

        model.train()
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for x_val, y_val in testloader:
                x_val, y_val = x_val.to(device), y_val.to(device)

                pred = model(x_val).argmax(dim=1)

                correct += (pred == y_val).sum().item()
                total += y_val.size(0)

        acc = correct / total
        print(f"Epoch {epoch + 1}/{num_epochs}, Test Accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            trigger_times = 0

            torch.save(model.state_dict(), save_path)
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    print(f"Best accuracy: {best_acc:.4f}")
    print(f"Model saved to: {save_path}")