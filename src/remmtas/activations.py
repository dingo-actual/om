from typing import Optional

import torch

class Swish(torch.nn.Module):
    """Swish activation module"""
    def __init__(self, beta: Optional[float] = None, device: Optional[str] = None):
        """Initialize the module.

        Args:
            beta (Optional[float], optional): Shape parameter. If None, it's a learnable parameter. Defaults to None.
            device (Optional[str], optional): Device to store the parameter on. Defaults to None.
        """
        super(Swish, self).__init__()
        # If beta is None, make it a learnable parameter
        if beta is None:
            self.beta = torch.nn.Parameter(torch.ones(1, device=device, dtype=torch.bfloat16))
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
    def __init__(self, dim: int, device: Optional[str] = None):
        """Initialize the module.

        Args:
            dim (int): Dimension of the input tensor.
            device (Optional[str], optional): Device to store the parameters on. Defaults to None.
        """
        super(SwiGLU, self).__init__()
        self.swish = Swish()
        self.W = torch.nn.Parameter((2. * torch.rand(dim, dim, device=device, dtype=torch.bfloat16) - 1.) / dim ** 0.5)
        self.V = torch.nn.Parameter((2. * torch.rand(dim, dim, device=device, dtype=torch.bfloat16) - 1.) / dim ** 0.5)
        self.b = torch.nn.Parameter(torch.zeros(dim, device=device, dtype=torch.bfloat16))
        self.c = torch.nn.Parameter(torch.zeros(dim, device=device, dtype=torch.bfloat16))
    
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
    def __init__(self, dim: int, device: Optional[str] = None):
        """Initialize the module.

        Args:
            dim (int): Dimension of the input tensor.
            device (Optional[str], optional): Device to store the parameters on. Defaults to None.
        """
        super(GEGLU, self).__init__()
        self.W = torch.nn.Parameter((2. * torch.rand(dim, dim, device=device, dtype=torch.bfloat16) - 1.) / dim ** 0.5)
        self.V = torch.nn.Parameter((2. * torch.rand(dim, dim, device=device, dtype=torch.bfloat16) - 1.) / dim ** 0.5)
        self.b = torch.nn.Parameter(torch.zeros(dim, device=device, dtype=torch.bfloat16))
        self.c = torch.nn.Parameter(torch.zeros(dim, device=device, dtype=torch.bfloat16))
        
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
    def __init__(self, dim_in: int, dim_hidden: int, device: Optional[str] = None):
        """Initialize the module.

        Args:
            dim (int): Dimension of the input tensor.
            device (Optional[str], optional): Device to store the parameters on. Defaults to None.
        """
        super(FFNGLU, self).__init__()
        self.W1 = torch.nn.Parameter((2. * torch.rand(dim_in, dim_hidden, device=device, dtype=torch.bfloat16) - 1.) /  dim_hidden ** 0.5)
        self.W2 = torch.nn.Parameter((2. * torch.rand(dim_hidden, dim_in, device=device, dtype=torch.bfloat16) - 1.) / dim_in ** 0.5)
        self.V = torch.nn.Parameter((2. * torch.rand(dim_in, dim_hidden, device=device, dtype=torch.bfloat16) - 1.) / dim_hidden ** 0.5)
        
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
    def __init__(self, dim_in: int, dim_hidden: int, device: Optional[str] = None):
        """Initialize the module.

        Args:
            dim (int): Dimension of the input tensor.
            device (Optional[str], optional): Device to store the parameters on. Defaults to None.
        """
        super(FFNGEGLU, self).__init__()
        self.W1 = torch.nn.Parameter((2. * torch.rand(dim_in, dim_hidden, device=device, dtype=torch.bfloat16) - 1.) /  dim_hidden ** 0.5)
        self.W2 = torch.nn.Parameter((2. * torch.rand(dim_hidden, dim_in, device=device, dtype=torch.bfloat16) - 1.) / dim_in ** 0.5)
        self.V = torch.nn.Parameter((2. * torch.rand(dim_in, dim_hidden, device=device, dtype=torch.bfloat16) - 1.) / dim_hidden ** 0.5)
        
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
    def __init__(self, dim_in: int, dim_hidden: int, device: Optional[str] = None):
        """Initialize the module.

        Args:
            dim (int): Dimension of the input tensor.
            device (Optional[str], optional): Device to store the parameters on. Defaults to None.
        """
        super(FFNSwiGLU, self).__init__()
        self.swish = Swish(beta=1)
        self.W1 = torch.nn.Parameter((2. * torch.rand(dim_in, dim_hidden, device=device, dtype=torch.bfloat16) - 1.) /  dim_hidden ** 0.5)
        self.W2 = torch.nn.Parameter((2. * torch.rand(dim_hidden, dim_in, device=device, dtype=torch.bfloat16) - 1.) / dim_in ** 0.5)
        self.V = torch.nn.Parameter((2. * torch.rand(dim_in, dim_hidden, device=device, dtype=torch.bfloat16) - 1.) / dim_hidden ** 0.5)
        
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
