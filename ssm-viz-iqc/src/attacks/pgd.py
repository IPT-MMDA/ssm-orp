import torch

def pgd_attack(model, x, y, eps=0.3, alpha=0.03, steps=20):
    model.eval()

    x_adv = x.clone().detach()
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)

    for _ in range(steps):
        x_adv.requires_grad_(True)

        loss = torch.nn.functional.cross_entropy(model(x_adv), y)

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)

        x_adv = x_adv.detach()

    return x_adv