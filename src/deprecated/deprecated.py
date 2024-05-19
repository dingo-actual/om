from typing import Optional

import torch

class StackedLinear(torch.nn.Module):
    """Implements a stacked linear layer"""
    
    def __init__(self, dim_in: int, dim_out: int, layers_in: int, layers_out: int, bias: bool = True, device: Optional[str] = None):
        super(StackedLinear, self).__init__()
        
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.layers_in = layers_in
        self.layers_out = layers_out
        self.device = device
        
        repeats, rem = divmod(layers_out, layers_in)
        if rem != 0:
            raise ValueError("layers_out must be a multiple of layers_in")
        
        self.layer_repeats = repeats
        
        self.weights = torch.nn.Parameter(
            (2. * torch.rand(1, layers_out, 1, dim_in, dim_out, device=device, dtype=torch.bfloat16) - 1.) / dim_out ** 0.5,
        )
        if bias:
            self.biases = torch.nn.Parameter(
                torch.zeros(1, layers_out, 1, dim_out, device=device, dtype=torch.bfloat16)
            )
        else:
            self.biases = None
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies a stack of linear layers to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, layers_in, seq_len, dim_in) 

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, layers_out, seq_len, dim_out).
        """
        batch_size, _, seq_len, _ = x.shape
        
        x = x.unsqueeze(-2)
        
        if self.layer_repeats != 1:
            x = x.repeat(1, self.layer_repeats, 1, 1, 1)
        
        if self.biases is None:
            return (x @ self.weights.repeat(batch_size, 1, seq_len, 1, 1)).squeeze(-2)
        else:
            return (x @ self.weights.repeat(batch_size, 1, seq_len, 1, 1)).squeeze(-2) + self.biases