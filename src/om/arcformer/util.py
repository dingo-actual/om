import os
import platform
from typing import Tuple

import torch


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
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len + 2 * state_len, dim).
        state_len (int): Length of the state.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - state_start: Tensor of shape (batch_size, state_len, dim)
            - x: Tensor of shape (batch_size, seq_len, dim)
            - state_end: Tensor of shape (batch_size, state_len, dim)
    """
    return x[:,:state_len,:], x[:,state_len:-state_len,:], x[:,-state_len:,:]

def split_last_dim(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Splits the last dimension of a tensor into two tensors.
    
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - x_start: Tensor of shape (batch_size, seq_len, dim // 2)
            - x_end: Tensor of shape (batch_size, seq_len, dim // 2)
    """
    return x[..., :x.size(-1) // 2], x[..., x.size(-1) // 2:]

def reverse_state_end(x: torch.Tensor, state_len: int) -> torch.Tensor:
    """Reverses the end state portion of a tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len + 2 * state_len, dim).
        state_len (int): Length of the state.

    Returns:
        torch.Tensor: Tensor with reversed end state portion. Has shape (batch_size, seq_len + 2 * state_len, dim).
    """
    x[:, -state_len:, :] = x[:, -state_len:, :].flip(1)
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
