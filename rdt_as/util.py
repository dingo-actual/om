import torch


class StackedLinear(torch.nn.Module):
    """Implements a stacked linear layer"""
    
    def __init__(self, dim_in: int, dim_out: int, layers_in: int, layers_out: int, bias: bool = True):
        super(StackedLinear, self).__init__()
        
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.layers_in = layers_in
        self.layers_out = layers_out
        
        repeats, rem = divmod(layers_out, layers_in)
        if rem != 0:
            raise ValueError("layers_out must be a multiple of layers_in")
        
        self.layer_repeats = repeats
        
        self.weights = torch.nn.Parameter(
            (2. * torch.rand(1, layers_out, 1, dim_in, dim_out) - 1.) / dim_out ** 0.5,
        )
        if bias:
            self.biases = torch.nn.Parameter(
                torch.zeros(1, layers_out, 1, dim_out)
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


def count_optimized_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of optimized parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model to count the optimized parameters for.
    Returns:
        int: The number of optimized parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
