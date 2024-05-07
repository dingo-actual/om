from typing import List, Optional

import torch
from torch import nn

from .activations import ACTIVATIONS
from .remmtas_memory import ReMMTAS
from .positional_embeddings import RoPEEmbeddings


class ReMMTASformer(nn.Module):
    """Transformer layer with ReMMTAS memory."""

    def __init__(
        self,
        dim_input: int,
        dim_hidden: int,
        dims_key: List[int],
        dims_value: List[int],
        num_heads: int,
        activation: str,
        segment_len: int,
        state_len: int,
        position_embedders: List[Optional[RoPEEmbeddings]],
        dropout: float = 0.0
    ):
        """Initializes the module.

        Args:
            dim_input (int): Input dimension.
            dim_hidden (int): Hidden dimension for the MLP.
            dims_key (int): Key dimensions for the memory modules.
            dims_value (int): Value dimensions for the memory modules.
            num_heads (int): Number of attention heads for the memory modules.
            activation (str): Activation function to use for the MLP. Must be a key in the ACTIVATIONS dictionary.
            segment_len (int): Segment length for the memory modules.
            state_len (int): Length of the state (i.e., number of tokens) for the memory modules.
            position_embedders (List[Optional[RoPEEmbeddings]]): Position embedding modules for the memory modules.
            dropout (float, optional): Dropout rate for the MLP. Defaults to 0.0.
        """
        super(ReMMTASformer, self).__init__()

        # Multi-head attention
        self.attn = ReMMTAS(
            dim_input=dim_input, 
            dims_key=dims_key, 
            dims_value=dims_value, 
            num_heads=num_heads, 
            segment_len=segment_len, 
            state_len=state_len, 
            position_embedders=position_embedders, 
        )
        
        # Set learnable initial state
        self.init_state = nn.Parameter(torch.randn(1, state_len, dim_input) / dim_input ** 0.5)
        
        # MLP
        if activation not in ACTIVATIONS:
            raise ValueError(f"Invalid activation function: {activation}")
        if activation in ["swiglu", "geglu", "ffnglu", "ffngeglu", "ffnswiglu"]:
            act = ACTIVATIONS[activation](dim_hidden)
        else:
            act = ACTIVATIONS[activation]()
        self.mlp = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.Dropout(dropout),
            act,
            nn.Linear(dim_hidden, dim_input),
            nn.Dropout(dropout)
        )
        self.layer_norm = nn.LayerNorm(dim_input)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim_input).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim_input).
        """
        # Apply multi-head attention, followed by MLP and layer normalization with residual connection.
        x_, _ = self.attn(x, self.init_state)
        x_ = self.layer_norm(x_ + x)
        x_ = self.mlp(x_)

        return self.layer_norm(x_ + x)
