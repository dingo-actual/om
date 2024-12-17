import os
import platform
from typing import Tuple

import torch


class StackedLinear(torch.nn.Module):
    """A stack of linear layers."""
    def __init__(self, dim_in: int, dim_out: int, num_layers: int, bias: bool = False):
        """Initialize the stack of linear layers.

        Args:
            dim_in (int): Input dimension.
            dim_out (int): Output dimension.
            num_layers (int): Number of layers.
            bias (bool): Whether to use bias in the linear layers. Defaults to False.
        """
        super(StackedLinear, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_layers = num_layers
        
        self.weights = torch.nn.Parameter(torch.empty(1, num_layers, dim_in, dim_out))
        torch.nn.init.normal_(self.weights, mean=0.0, std=(1.0 / dim_out) ** 0.5)
        
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(1, num_layers, 1, dim_out))
        else:
            self.bias = None
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the stack of linear layers.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_heads, seq_len, dim_in).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_heads, seq_len, dim_out).
        """
        out = torch.matmul(x, self.weights)
        
        if self.bias is not None:
            out += self.bias
            
        return out
    
    def __repr__(self):
        return f"StackedLinear(dim_in={self.dim_in}, dim_out={self.dim_out}, num_layers={self.num_layers}, bias={self.bias is not None})"


def count_optimized_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of optimized parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model to count the optimized parameters for.
    Returns:
        int: The number of optimized parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def extract_state(
    x: torch.Tensor, 
    state_len: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extracts the state from the input tensor x.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len + 2 * state_len, dim) or (batch_size, num_heads, seq_len + 2 * state_len, dim).
        state_len (int): Length of the state.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - state_start: Tensor of shape (batch_size, state_len, dim) or (batch_size, num_heads, state_len, dim)
            - x: Tensor of shape (batch_size, seq_len, dim) or (batch_size, num_heads, seq_len, dim)
            - state_end: Tensor of shape (batch_size, state_len, dim) or (batch_size, num_heads, state_len, dim)
    """
    return x[...,:state_len,:], x[...,state_len:-state_len,:], x[...,-state_len:,:]

def split_last_dim(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Splits the last dimension of a tensor into two tensors.
    
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim) or (batch_size, num_heads, seq_len, dim).
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - x_start: Tensor of shape (batch_size, seq_len, dim // 2) or (batch_size, num_heads, seq_len, dim // 2)
            - x_end: Tensor of shape (batch_size, seq_len, dim // 2) or (batch_size, num_heads, seq_len, dim // 2)
    """
    return x[..., :x.size(-1) // 2], x[..., x.size(-1) // 2:]

def reverse_state_end(x: torch.Tensor, state_len: int) -> torch.Tensor:
    """Reverses the end state portion of a tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len + 2 * state_len, dim) or (batch_size, num_heads, seq_len + 2 * state_len, dim).
        state_len (int): Length of the state.

    Returns:
        torch.Tensor: Tensor with reversed end state portion. Has shape (batch_size, seq_len + 2 * state_len, dim) or (batch_size, num_heads, seq_len + 2 * state_len, dim).
    """
    x[..., -state_len:, :] = x[..., -state_len:, :].flip(1)
    return x

def check_if_linux() -> bool:
    """Check if the current operating system is Linux."""
    if platform.system() == "Linux":
        if os.path.exists('/proc/version'):
            with open('/proc/version', 'r') as f:
                if "microsoft" in f.read().lower():
                    return False
                else:
                    return True
        else:
            raise FileNotFoundError("Could not find /proc/version")
    else:
        return False


if __name__ == "__main__":
    # Run unit tests

    def test_count_optimized_parameters():
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param1 = torch.nn.Parameter(torch.randn(10, 20))
                self.param2 = torch.nn.Parameter(torch.randn(5, 5))
                self.param3 = torch.nn.Parameter(torch.randn(3, 3))
                self.param3.requires_grad = False

        model = TestModel()
        assert count_optimized_parameters(model) == 10 * 20 + 5 * 5

    def test_count_optimized_parameters_no_params():
        class EmptyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

        model = EmptyModel()
        assert count_optimized_parameters(model) == 0

    def test_count_optimized_parameters_all_frozen():
        class FrozenModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param1 = torch.nn.Parameter(torch.randn(10, 20))
                self.param2 = torch.nn.Parameter(torch.randn(5, 5))
                self.param1.requires_grad = False
                self.param2.requires_grad = False

        model = FrozenModel()
        assert count_optimized_parameters(model) == 0
