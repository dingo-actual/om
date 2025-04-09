from typing import Callable

import numpy as np
import torch

def set_om_dtypes(model: torch.nn.Module, dtype: torch.dtype) -> torch.nn.Module:
    """Convert the model to the specified dtype, while keeping LayerNorm layers in fp32.

    Args:
        model (torch.nn.Module): model to convert
        dtype (torch.dtype): desired dtype

    Returns:
        torch.nn.Module: model with converted dtypes
    """
    model = model.to(dtype=dtype)
    for name, param in model.named_parameters():
        if "norm" in name or "thetas" in name:
            param.data = param.data.to(dtype=torch.float32)
        if "ixs_sin" in name:
            param.data = param.data.to(dtype=torch.long)
            
    return model

def cosine_with_warmup_mult(warmup_steps: int, total_steps: int, min_lr_mult: float) -> Callable[[int], float]:
    """Creates a cosine learning rate schedule with warmup.

    Args:
        warmup_steps (int): Number of warmup steps
        total_steps (int): Total number of steps
        min_lr_mult (float): Minimum learning rate multiplier

    Returns:
        Callable[[int], float]: Learning rate multiplier function
    """
    def lr_mult(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        else:
            if step > total_steps:
                step = total_steps
            return min_lr_mult + (1 - min_lr_mult) * (1 + np.cos((step - warmup_steps) / (total_steps - warmup_steps) * np.pi)) / 2.0
        
    return lr_mult

def grad_norm(model: torch.nn.Module, predicate: Callable[[str, torch.nn.Parameter], bool]) -> torch.Tensor:
    """Get the gradient norm of the model.

    Args:
        model (torch.nn.Module): model to get gradient norm of
        predicate (Callable[[str, torch.nn.Parameter], bool]): predicate to filter parameters
    
    Returns:
        torch.Tensor: gradient norm
    """
    return torch.sqrt(
        torch.sum(
            torch.tensor([torch.sum(torch.norm(p.grad)**2) for name, p in model.named_parameters() if p.requires_grad and p.grad is not None and predicate(name, p)])
        )
    )
