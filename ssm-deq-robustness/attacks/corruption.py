import torch
from torchvision import transforms

def corrupt_data(images, corruption_type='gaussian_noise', severity=1):
    """Симуляція пошкоджень набору даних CIFAR-C."""
    if corruption_type == 'gaussian_noise':
        # Severity керує рівнем стандартного відхилення
        noise_level = [0.04, 0.06, 0.08, 0.12, 0.18][severity-1]
        noise = torch.randn_like(images) * noise_level
        return torch.clamp(images + noise, -1, 1)

    elif corruption_type == 'defocus_blur':
        # Використання Gaussian Blur як проксі для ImageNet-C defocus blur
        kernel_size = 2 * severity + 1
        blurrer = transforms.GaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=(0.5 * severity))
        return blurrer(images)

    return images