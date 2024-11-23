import torch

from .om_llm import OmLLM


def set_om_dtypes(model: OmLLM, dtype: torch.dtype) -> OmLLM:
    """Convert the model to the specified dtype, while keeping LayerNorm layers in fp32.

    Args:
        model (OmLLM): model to convert
        dtype (torch.dtype): desired dtype

    Returns:
        OmLLM: model with converted dtypes
    """
    model = model.to(dtype=dtype)
    for name, param in model.named_parameters():
        if "LayerNorm" in name:
            param = param.to(dtype=torch.float32)
            
    return model
