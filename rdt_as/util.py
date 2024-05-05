import torch


class StackedLinear(torch.nn.Module):
    """Implements a stacked linear layer"""
    
    def __init__(self, dim_in, dim_out, num_layers, bias=True):
        super(StackedLinear, self).__init__()
        
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_layers = num_layers
        
        self.weights = torch.nn.Parameter(
            (2. * torch.rand(1, num_layers, 1, dim_in, dim_out) - 1.) / dim_out ** 0.5,
        )
        if bias:
            self.biases = torch.nn.Parameter(
                torch.zeros(1, num_layers, 1, dim_out)
            )
        else:
            self.biases = None
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies a stack of linear layers to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim_in).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_layers, seq_len, dim_out).
        """
        batch_size, seq_len, _ = x.shape
        if self.biases is None:
            return (x.unsqueeze(1).unsqueeze(-2).repeat(1, self.num_layers, 1, 1, 1) @ self.weights.repeat(batch_size, 1, seq_len, 1, 1)).squeeze(-2)
        else:
            return (x.unsqueeze(1).unsqueeze(-2).repeat(1, self.num_layers, 1, 1, 1) @ self.weights.repeat(batch_size, 1, seq_len, 1, 1)).squeeze(-2) + self.biases
