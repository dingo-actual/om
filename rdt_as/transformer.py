from typing import Optional

import torch
from torch import nn

from .activations import ACTIVATIONS
from .redotras_memory import ReDoTrAS
from .positional_embeddings import RoPEEmbeddings


class ReDoTransformer(nn.Module):
    """Transformer layer with ReDoTrAS memory."""

    def __init__(
        self,
        dim_input: int,
        dim_hidden: int,
        dim_key: int,
        dim_value: int,
        num_heads: int,
        activation: str,
        segment_len: int,
        state_len: int,
        position_embedder_1: Optional[RoPEEmbeddings] = None,
        position_embedder_2: Optional[RoPEEmbeddings] = None,
        dropout: float = 0.0
    ):
        """Initializes the module.

        Args:
            dim_input (int): Input dimension.
            dim_hidden (int): Hidden dimension for the MLP.
            dim_key (int): Key dimension for the memory module.
            dim_value (int): Value dimension for the memory module.
            num_heads (int): Number of attention heads for the memory module.
            activation (str): Activation function to use for the MLP. Must be a key in the ACTIVATIONS dictionary.
            segment_len (int): Segment length for the memory module.
            state_len (int): Length of the state (i.e., number of tokens) for the memory module.
            position_embedder_1 (Optional[RoPEEmbeddings], optional): First position embedding module for the memory module. Defaults to None.
            position_embedder_2 (Optional[RoPEEmbeddings], optional): Second position embedding module for the memory module. Defaults to None.
            dropout (float, optional): Dropout rate for the MLP. Defaults to 0.0.
        """
        super(ReDoTransformer, self).__init__()

        # Multi-head attention
        self.attn = ReDoTrAS(
            dim_input=dim_input, 
            dim_key=dim_key, 
            dim_value=dim_value, 
            num_heads=num_heads, 
            segment_len=segment_len, 
            position_embedder_1=position_embedder_1, 
            position_embedder_2=position_embedder_2
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
