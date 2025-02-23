import torch


def modulate(x: torch.Tensor, scale: float) -> torch.Tensor:
    return x * (1 + scale)
