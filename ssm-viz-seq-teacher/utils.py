import torch
import torch.nn.functional as F
import numpy as np
from scipy import stats
import torchvision.transforms.functional as TF

def distillation_loss(student_logits, teacher_logits, labels, T=3.0, alpha=0.5):
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T ** 2)
    hard_loss = F.cross_entropy(student_logits, labels)
    return alpha * soft_loss + (1 - alpha) * hard_loss

def get_confidence_interval(data, confidence=0.95):
    n = len(data)
    if n < 2: return 0.0
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return h

def evaluate_robustness(model, loader, device, mode='clean', noise_lvl=0.0):
    model.eval()
    batch_accs = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            # Вибір типу викривлення зображення
            if mode == 'noise':
                images = images + torch.randn_like(images) * noise_lvl
            elif mode == 'blur':
                images = TF.gaussian_blur(images, kernel_size=(5, 5), sigma=(1.5, 1.5))
            elif mode == 'occlusion':
                images[:, :, 12:20, 12:20] = 0 # Закриваємо центр (оклюзія)
            
            images = torch.clamp(images, 0, 1)
            
            outputs = model(images)
            acc = (outputs.argmax(1) == labels).float().mean().item()
            batch_accs.append(acc)
            
    mean_acc = np.mean(batch_accs)
    ci = get_confidence_interval(batch_accs)
    return mean_acc, ci