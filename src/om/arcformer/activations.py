from typing import Optional

import torch


class Swish(torch.nn.Module):
    """Swish activation module"""
    def __init__(self, beta: Optional[float] = None):
        """Initialize the module.

        Args:
            beta (Optional[float], optional): Shape parameter. If None, it's a learnable parameter. Defaults to None.
        """
        super(Swish, self).__init__()
        # If beta is None, make it a learnable parameter
        if beta is None:
            self.beta = torch.nn.Parameter(torch.ones(1))
        # Otherwise, set it to a fixed constant
        else:
            self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Activation tensor.
        """
        return x * torch.nn.functional.sigmoid(self.beta * x)

class SwiGLU(torch.nn.Module):
    """SwiGLU activation module."""
    def __init__(self, dim_in: int, dim_out: int, num_layers: Optional[int] = None):
        """Initialize the module.

        Args:
            dim_in (int): Dimension of the input tensor.
            dim_out (int): Dimension of the output tensor.
            num_layers (Optional[int], optional): Number of layers for depth-aware initialization. If None, standard initialization is used. Defaults to None.
        """
        super(SwiGLU, self).__init__()
        self.swish = Swish()
        if num_layers is None:
            self.W = torch.nn.Parameter((2. * torch.rand(dim_in, dim_out) - 1.) / dim_out ** 0.5)
            self.V = torch.nn.Parameter((2. * torch.rand(dim_in, dim_out) - 1.) / dim_out ** 0.5)
        else:
            self.W = torch.nn.Parameter(torch.empty((dim_in, dim_out)))
            self.V = torch.nn.Parameter(torch.empty((dim_in, dim_out)))
            torch.nn.init.normal_(self.W, mean=0.0, std=(1. / (2 * self.num_layers)) ** 0.5)
            torch.nn.init.normal_(self.V, mean=0.0, std=(1. / (2 * self.num_layers)) ** 0.5)
        self.b = torch.nn.Parameter(torch.zeros(dim_out))
        self.c = torch.nn.Parameter(torch.zeros(dim_out))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Activation tensor.
        """
        return self.swish(x @ self.W + self.b) * (x @ self.V + self.c)
        
        
class GEGLU(torch.nn.Module):
    """GEGLU activation module."""
    def __init__(self, dim_in: int, dim_out: int, num_layers: Optional[int] = None):
        """Initialize the module.

        Args:
            dim_in (int): Dimension of the input tensor.
            dim_out (int): Dimension of the output tensor.
            num_layers (Optional[int], optional): Number of layers for depth-aware initialization. If None, standard initialization is used. Defaults to None.
        """
        super(GEGLU, self).__init__()
        if num_layers is None:
            self.W = torch.nn.Parameter((2. * torch.rand(dim_in, dim_out) - 1.) / dim_out ** 0.5)
            self.V = torch.nn.Parameter((2. * torch.rand(dim_in, dim_out) - 1.) / dim_out ** 0.5)
        else:
            self.W = torch.nn.Parameter(torch.empty((dim_in, dim_out)))
            self.V = torch.nn.Parameter(torch.empty((dim_in, dim_out)))
            torch.nn.init.normal_(self.W, mean=0.0, std=(1. / (2 * self.num_layers)) ** 0.5)
            torch.nn.init.normal_(self.V, mean=0.0, std=(1. / (2 * self.num_layers)) ** 0.5)
        self.b = torch.nn.Parameter(torch.zeros(dim_out))
        self.c = torch.nn.Parameter(torch.zeros(dim_out))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Activation tensor.
        """
        return torch.nn.functional.gelu(x @ self.W + self.b) * (x @ self.V + self.c)
    
class FFNGLU(torch.nn.Module):
    """FFN GLU activation module."""
    def __init__(self, dim_in: int, dim_hidden: int, num_layers: Optional[int] = None):
        """Initialize the module.

        Args:
            dim_in (int): Dimension of the input tensor.
            dim_hidden (int): Dimension of the hidden tensor.
            num_layers (Optional[int], optional): Number of layers for depth-aware initialization. If None, standard initialization is used. Defaults to None.
        """
        super(FFNGLU, self).__init__()
        self.W1 = torch.nn.Parameter((2. * torch.rand(dim_in, dim_hidden) - 1.) /  dim_hidden ** 0.5)
        self.V = torch.nn.Parameter((2. * torch.rand(dim_in, dim_hidden) - 1.) / dim_hidden ** 0.5)
        
        if num_layers is None:
            self.W2 = torch.nn.Parameter((2. * torch.rand(dim_hidden, dim_in) - 1.) / dim_in ** 0.5)
        else:
            self.W2 = torch.nn.Parameter(torch.empty((dim_hidden, dim_in)))
            torch.nn.init.normal_(self.W2, mean=0.0, std=(1. / (2 * self.num_layers)) ** 0.5)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Activation tensor.
        """
        return (torch.nn.functional.sigmoid(x @ self.W1) * (x @ self.V)) @ self.W2
    
class FFNGEGLU(torch.nn.Module):
    """FFN GELU activation module."""
    def __init__(self, dim_in: int, dim_hidden: int, num_layers: Optional[int] = None):
        """Initialize the module.

        Args:
            dim_in (int): Dimension of the input tensor.
            dim_hidden (int): Dimension of the hidden tensor.
            num_layers (Optional[int], optional): Number of layers for depth-aware initialization. If None, standard initialization is used. Defaults to None.
        """
        super(FFNGEGLU, self).__init__()
        self.W1 = torch.nn.Parameter((2. * torch.rand(dim_in, dim_hidden) - 1.) /  dim_hidden ** 0.5)
        self.V = torch.nn.Parameter((2. * torch.rand(dim_in, dim_hidden) - 1.) / dim_hidden ** 0.5)
        
        if num_layers is None:
            self.W2 = torch.nn.Parameter((2. * torch.rand(dim_hidden, dim_in) - 1.) / dim_in ** 0.5)
        else:
            self.W2 = torch.nn.Parameter(torch.empty((dim_hidden, dim_in)))
            torch.nn.init.normal_(self.W2, mean=0.0, std=(1. / (2 * self.num_layers)) ** 0.5)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Activation tensor.
        """
        return (torch.nn.functional.gelu(x @ self.W1) * (x @ self.V)) @ self.W2
    
class FFNSwiGLU(torch.nn.Module):
    """FFN SwiGLU activation module."""
    def __init__(self, dim_in: int, dim_hidden: int, num_layers: Optional[int] = None):
        """Initialize the module.

        Args:
            dim (int): Dimension of the input tensor.
            dim_hidden (int): Dimension of the hidden tensor.
            num_layers (Optional[int], optional): Number of layers for depth-aware initialization. If None, standard initialization is used. Defaults to None.
        """
        super(FFNSwiGLU, self).__init__()
        self.swish = Swish(beta=1)
        self.W1 = torch.nn.Parameter((2. * torch.rand(dim_in, dim_hidden) - 1.) /  dim_hidden ** 0.5)
        self.V = torch.nn.Parameter((2. * torch.rand(dim_in, dim_hidden) - 1.) / dim_hidden ** 0.5)
        
        if num_layers is None:
            self.W2 = torch.nn.Parameter((2. * torch.rand(dim_hidden, dim_in) - 1.) / dim_in ** 0.5)
        else:
            self.W2 = torch.nn.Parameter(torch.empty((dim_hidden, dim_in)))
            torch.nn.init.normal_(self.W2, mean=0.0, std=(1. / (2 * self.num_layers)) ** 0.5)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Activation tensor.
        """
        return (self.swish(x @ self.W1) * (x @ self.V)) @ self.W2
    
class Abs(torch.nn.Module):
    """Absolute value activation module."""
    def __init__(self):
        """Initialize the module."""
        super(Abs, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Activation tensor.
        """
        return torch.abs(x)

# Importable container for available activations
ACTIVATIONS = {
    "relu": torch.nn.ReLU,
    "gelu": torch.nn.GELU,
    "swish": Swish,
    "swiglu": SwiGLU,
    "geglu": GEGLU,
    "ffnglu": FFNGLU,
    "ffngeglu": FFNGEGLU,
    "ffnswiglu": FFNSwiGLU,
    "abs": Abs
}
