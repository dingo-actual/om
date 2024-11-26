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
        if "norm" in name:
            param.data = param.data.to(dtype=torch.float32)
            
    return model
