# preprocess.py
import torch

CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465])
CIFAR_STD  = torch.tensor([0.2023, 0.1994, 0.2010])

def denormalize(x: torch.Tensor, device: torch.device):
    """Convert normalized tensor (model input space) -> pixel space [0,1]."""
    mean = CIFAR_MEAN.to(device).view(1, 3, 1, 1)
    std  = CIFAR_STD.to(device).view(1, 3, 1, 1)
    return x * std + mean

def renormalize(x: torch.Tensor, device: torch.device):
    """Convert pixel-space tensor [0,1] -> normalized model input space."""
    mean = CIFAR_MEAN.to(device).view(1, 3, 1, 1)
    std  = CIFAR_STD.to(device).view(1, 3, 1, 1)
    return (x - mean) / std
