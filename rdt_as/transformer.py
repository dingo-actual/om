from typing import Optional

import torch
from torch import nn

from .activations import ACTIVATIONS
from .redotras_memory import ReDoTrAS
from .positional_embeddings import RoPEEmbeddings


# TODO: make any necessary changes for compatibility for ReDoTRAS
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
        update: str = "linear",
        causal: bool = False,
        position_embedder: Optional[RoPEEmbeddings] = None,
        init_state_learnable: bool = False,
        dropout: float = 0.0,
        **kwargs
    ):
        """Initializes the module.

        Args:
            dim_input (int): Input dimension.
            dim_hidden (int): Hidden dimension for the MLP.
            dim_key (int): Key dimension for the CompressiveMemory.
            dim_value (int): Value dimension for the CompressiveMemory.
            num_heads (int): Number of attention heads for the CompressiveMemory.
            activation (str): Activation function to use for the MLP. Must be a key in the ACTIVATIONS dictionary.
            segment_len (int): Segment length for the CompressiveMemory.
            update (str, optional): Type of memory update rule to use for the CompressiveMemory ("linear" or "delta"). Defaults to "linear".
            causal (bool, optional): Whether to use causal attention masking for the CompressiveMemory. Defaults to False.
            position_embedder (Optional[PositionEmbeddings], optional): Position embedding module for the CompressiveMemory. Defaults to None.
            init_state_learnable (bool, optional): Whether the initial state of the CompressiveMemory should be learnable. Defaults to False.
            dropout (float, optional): Dropout rate for the MLP. Defaults to 0.0.
        """
        super(ReDoTransformer, self).__init__()
        
        # If sampling_factor passed to kwargs, use it, otherwise set to None
        sampling_factor = kwargs.get("sampling_factor", None)
        
        # Multi-head attention
        self.attn = CompressiveMemory(
            dim_input=dim_input, 
            dim_key=dim_key, 
            dim_value=dim_value, 
            num_heads=num_heads, 
            segment_len=segment_len, 
            sampling_factor=sampling_factor,
            update=update, 
            causal=causal, 
            position_embedder=position_embedder, 
            init_state_learnable=init_state_learnable)
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
        x_ = self.attn(x)
        x_ = self.mlp(x_)

        return self.layer_norm(x_ + x)
