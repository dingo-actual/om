from typing import List, Optional, Tuple

import torch
from torch import nn

from .activations import ACTIVATIONS
from .arc_memory import ARC
from .positional_embeddings import RoPEEmbeddings


class ARCformer(nn.Module):
    """Transformer layer with ARC memory."""

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
        attn_normalize: bool,
        num_layers: int,
        cope: bool,
        position_embedders: List[Optional[RoPEEmbeddings]],
        dropout: float = 0.0,
        mlp_multiplier: int = 1,
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
            attn_normalize (bool): Whether to normalize attention inputs for the memory modules.
            num_layers (int): Number of ARC transformer layers in the parent model.
            cope (bool): Whether to use CoPE for the memory modules.
            position_embedders (List[Optional[RoPEEmbeddings]]): Position embedding modules for the memory modules.
            dropout (float, optional): Dropout rate for the MLP. Defaults to 0.0.
            mlp_multiplier (int, optional): Multiplier for the hidden state dimensions of the MLP. Defaults to 1.
        """
        super(ARCformer, self).__init__()

        # Multi-head attention
        self.attn = ARC(
            dim_input=dim_input, 
            dims_key=dims_key, 
            dims_value=dims_value, 
            num_heads=num_heads, 
            segment_len=segment_len, 
            state_len=state_len, 
            attn_normalize=attn_normalize,
            num_layers=num_layers,
            cope=cope,
            position_embedders=position_embedders
        )
        self.mlp_multiplier = mlp_multiplier
        self.num_layers = num_layers
        self.attn_norm = nn.LayerNorm(dim_input, eps=1e-5)
        self.attn_dropout = nn.Dropout(dropout)
        
        # MLP
        if activation not in ACTIVATIONS:
            raise ValueError(f"Invalid activation function: {activation}")
        elif activation in ["swiglu", "geglu"]:
            self.mlp = ACTIVATIONS[activation](dim_input)
        elif activation in ["ffnglu", "ffngeglu", "ffnswiglu"]:
            self.mlp = ACTIVATIONS[activation](dim_input, dim_hidden * mlp_multiplier)
        else:
            act = ACTIVATIONS[activation]()
            self.mlp = nn.Sequential(
                nn.Linear(dim_input, dim_hidden * mlp_multiplier),
                act,
                nn.Linear(dim_hidden * mlp_multiplier, dim_input * mlp_multiplier),
            )
            torch.nn.init.normal_(self.mlp[3].weight, mean=0.0, std=(1. / (2 * self.num_layers)) ** 0.5)
        self.mlp_norm = nn.LayerNorm(dim_input * mlp_multiplier, eps=1e-5)
        self.mlp_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, state: torch.Tensor, offset: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, segment_len, dim_input).
            state (torch.Tensor): State tensor of shape (batch_size, state_len, dim_input).
            offset (int): Offset for position embeddings.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, segment_len, dim_input).
            torch.Tensor: State tensor of shape (batch_size, state_len, dim_input * mlp_multiplier).
        """
        # Apply multi-head attention, followed by layer normalization with residual connection then MLP.
        x_, state = self.attn(x, state, offset)
        x_ = self.attn_norm(self.attn_dropout(x_) + x)
        x_ = self.mlp_dropout(self.mlp(x_))
        
        # If no MLP multiplier, then add residual connection.
        if self.mlp_multiplier == 1:
            x_ = self.mlp_norm(x_ + x)
        else:
            x_ = self.mlp_norm(x_)

        return x_, state
