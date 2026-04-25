import torch
import torch.nn.functional as F



def fgsm_attack(model, images, labels, eps):
    # 1. Створюємо копію і вмикаємо градієнти
    images = images.clone().detach().requires_grad_(True)

    # 2. Прямий прохід
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)

    # 3. Обчислюємо градієнти
    model.zero_grad()
    loss.backward()

    # ПЕРЕВІРКА: якщо градієнт раптом None, повертаємо оригінал, щоб не "валити" скрипт
    if images.grad is None:
        return images.detach()

    # 4. Формуємо атаку
    adv_images = images + eps * images.grad.sign()
    return torch.clamp(adv_images, -1, 1).detach()

def pgd_attack(model, images, labels, eps, alpha=0.01, steps=10):
    adv_images = images.clone().detach()

    for _ in range(steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels)

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            if adv_images.grad is not None:
                adv_images = adv_images + alpha * adv_images.grad.sign()
                # Проекція на L-infinity куб
                delta = torch.clamp(adv_images - images, -eps, eps)
                adv_images = torch.clamp(images + delta, -1, 1).detach()
            else:
                break # Якщо градієнт пропав, зупиняємо ітерації

    return adv_images