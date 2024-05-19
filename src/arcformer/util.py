from typing import List, Optional, Tuple, Union

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
    x: Union[torch.Tensor, List[torch.Tensor]], 
    state_len: int
) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]]:
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
    if isinstance(x, torch.Tensor):
        out = x[:,:state_len,:], x[:,state_len:-state_len,:], x[:,-state_len:,:]
    else:
        splits = [(x_[:,:state_len,:], x_[:,state_len:-state_len,:], x_[:,-state_len:,:]) for x_ in x]
        state_start = [split[0] for split in splits]
        x_out = [split[1] for split in splits]
        state_end = [split[2] for split in splits]
        out = state_start, x_out, state_end
    return out


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
