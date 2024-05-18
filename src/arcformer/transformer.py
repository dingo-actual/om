from typing import List, Optional

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
        mem_iters: List[int],
        num_heads: int,
        activation: str,
        segment_len: int,
        state_len: int,
        normalize_qkv: bool,
        position_embedders: List[Optional[RoPEEmbeddings]],
        dropout: float = 0.0,
        init_conv: bool = False,
        device: Optional[str] = None
    ):
        """Initializes the module.

        Args:
            dim_input (int): Input dimension.
            dim_hidden (int): Hidden dimension for the MLP.
            dims_key (int): Key dimensions for the memory modules.
            dims_value (int): Value dimensions for the memory modules.
            mem_iters (int): Number of iterations for the memory modules.
            num_heads (int): Number of attention heads for the memory modules.
            activation (str): Activation function to use for the MLP. Must be a key in the ACTIVATIONS dictionary.
            segment_len (int): Segment length for the memory modules.
            state_len (int): Length of the state (i.e., number of tokens) for the memory modules.
            normalize (bool): Whether to normalize attention inputs for the memory modules.
            position_embedders (List[Optional[RoPEEmbeddings]]): Position embedding modules for the memory modules.
            dropout (float, optional): Dropout rate for the MLP. Defaults to 0.0.
            init_conv (bool, optional): Whether to use an initial convolution layer. Defaults to False.
            device (Optional[str], optional): Device to use. Defaults to None.
        """
        super(ARCformer, self).__init__()
        
        if init_conv:
            self.conv = nn.Conv1d(dim_input, dim_input, kernel_size=3, device=device, dtype=torch.bfloat16)
        else:
            self.conv = None

        # Multi-head attention
        self.attn = ARC(
            dim_input=dim_input, 
            dims_key=dims_key, 
            dims_value=dims_value, 
            iters=mem_iters,
            num_heads=num_heads, 
            segment_len=segment_len, 
            state_len=state_len, 
            normalize=normalize_qkv,
            position_embedders=position_embedders,
            device=device
        )
        self.attn_norm = nn.LayerNorm(dim_input, device=device, dtype=torch.bfloat16)
        
        # MLP
        if activation not in ACTIVATIONS:
            raise ValueError(f"Invalid activation function: {activation}")
        elif activation in ["swiglu", "geglu"]:
            self.mlp = ACTIVATIONS[activation](dim_input, device=device)
        elif activation in ["ffnglu", "ffngeglu", "ffnswiglu"]:
            self.mlp = ACTIVATIONS[activation](dim_input, dim_hidden, device=device)
        else:
            if activation != "abs":
                act = ACTIVATIONS[activation]()
            else:
                act = ACTIVATIONS[activation](device=device)
            self.mlp = nn.Sequential(
                nn.Linear(dim_input, dim_hidden, device=device, dtype=torch.bfloat16),
                nn.Dropout(dropout),
                act,
                nn.Linear(dim_hidden, dim_input, device=device, dtype=torch.bfloat16),
                nn.Dropout(dropout)
            )
        self.mlp_norm = nn.LayerNorm(dim_input, device=device, dtype=torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim_input).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim_input).
        """
        # If initial convolution is defined, use it
        if self.conv is not None:
            x_ = self.conv(x.transpose(1, 2)).transpose(1, 2) + x[:, 2:, :]
        else:
            x_ = x
        # Apply multi-head attention, followed by MLP and layer normalization with residual connection.
        x_, _ = self.attn(x_)
        x_ = self.attn_norm(x_ + x)
        x_ = self.mlp(x_)

        return self.mlp_norm(x_ + x)
