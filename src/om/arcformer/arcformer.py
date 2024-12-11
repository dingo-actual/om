from typing import List, Optional, Tuple

import torch
from torch import nn
from xformers.components.positional_embedding import RotaryEmbedding

from .activations import ACTIVATIONS
from .arc_memory import ARC


class ARCformer(nn.Module):
    """Transformer layer with ARC memory."""

    def __init__(
        self,
        dim_input: int,
        dim_hidden: int,
        dims_key: List[int],
        dims_value: List[int],
        num_iters: List[int],
        num_heads: int,
        activation: str,
        segment_len: int,
        state_len: int,
        num_layers: int,
        layer_num: int,
        cope: bool,
        position_embedders: List[Optional[RotaryEmbedding]],
        betas: List[Optional[float]],
        dropout: float = 0.0,
        diff_attn: bool = False,
        attn_dropout: float = 0.0,
        attn_logit_dropout: float = 0.0,
        attn_proj_rank: int = -1,
        mlp_multiplier: int = 1,
        mlp_1221: bool = False,
    ):
        """Initializes the module.

        Args:
            dim_input (int): Input dimension.
            dim_hidden (int): Hidden dimension for the MLP.
            dims_key (List[int]): Key dimensions for the memory modules.
            dims_value (List[int]): Value dimensions for the memory modules.
            num_iters (List[int]): Number of iterations for the memory modules.
            num_heads (int): Number of attention heads for the memory modules.
            activation (str): Activation function to use for the MLP. Must be a key in the ACTIVATIONS dictionary.
            segment_len (int): Segment length for the memory modules.
            state_len (int): Length of the state (i.e., number of tokens) for the memory modules.
            num_layers (int): Number of ARC transformer layers in the parent model.
            layer_num (int): The position of the layer.
            cope (bool): Whether to use CoPE for the memory modules.
            position_embedders (List[Optional[RotaryEmbedding]]): Position embedding modules for the memory modules.
            betas (List[Optional[float]]): Betas for Hopfield memory / scaling factors for SDP attention.
            dropout (float, optional): Dropout rate for the MLP. Defaults to 0.0.
            diff_attn (bool, optional): Whether to use diff attention. Defaults to False.
            attn_dropout (float, optional): Dropout rate for attention outputs. Defaults to 0.0.
            attn_logit_dropout (float, optional): Dropout rate for attention logits. Defaults to 0.0.
            attn_proj_rank (int, optional): Rank of the attention projection back to the input dimension. If -1 will use dim_input // num_heads. Defaults to -1.
            mlp_multiplier (int, optional): Multiplier for the hidden state dimensions of the MLP. Defaults to 1.
            mlp_1221 (bool, optional): Whether to use the 1-2-2-1 MLP architecture. Defaults to False.
        """
        super(ARCformer, self).__init__()

        # Multi-head attention
        self.attn = ARC(
            dim_input=dim_input, 
            dims_key=dims_key, 
            dims_value=dims_value, 
            num_iters=num_iters,
            num_heads=num_heads, 
            segment_len=segment_len, 
            state_len=state_len, 
            dropout=attn_logit_dropout,
            betas=betas,
            attn_proj_rank=attn_proj_rank if attn_proj_rank > 0 else dim_input // num_heads,
            num_layers=num_layers,
            layer_num=layer_num,
            cope=cope,
            diff_attn=diff_attn,
            position_embedders=position_embedders
        )
        self.mlp_multiplier = mlp_multiplier
        self.mlp_1221 = mlp_1221
        self.num_layers = num_layers
        self.layer_num = layer_num
        self.diff_attn = diff_attn
        self.attn_norm = nn.LayerNorm(dim_input, eps=1e-5)
        self.dropout1 = nn.Dropout(attn_dropout)
        
        # MLP
        if activation not in ACTIVATIONS:
            raise ValueError(f"Invalid activation function: {activation}")
        elif activation in ["ffnglu", "ffngeglu", "ffnswiglu"]:
            self.mlp = ACTIVATIONS[activation](dim_input, dim_hidden * mlp_multiplier)
        else:
            act = ACTIVATIONS[activation]()
            if self.mlp_1221:
                self.mlp = nn.Sequential(
                    nn.Linear(dim_input, (dim_hidden * mlp_multiplier) // 2),
                    act,
                    nn.Linear((dim_hidden * mlp_multiplier) // 2, (dim_hidden * mlp_multiplier) // 2),
                    act,
                    nn.Linear((dim_hidden * mlp_multiplier) // 2, dim_input * mlp_multiplier),
                )
                torch.nn.init.normal_(self.mlp[4].weight, mean=0.0, std=(1. / (2 * self.num_layers)) ** 0.5)
            else:
                self.mlp = nn.Sequential(
                    nn.Linear(dim_input, dim_hidden * mlp_multiplier),
                    act,
                    nn.Linear(dim_hidden * mlp_multiplier, dim_input * mlp_multiplier),
                )
                torch.nn.init.normal_(self.mlp[2].weight, mean=0.0, std=(1. / (2 * self.num_layers)) ** 0.5)
        self.mlp_norm = nn.LayerNorm(dim_input * mlp_multiplier, eps=1e-5)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, state: torch.Tensor, skip_update_state: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, segment_len, dim_input).
            state (torch.Tensor): State tensor of shape (batch_size, state_len, dim_input).
            skip_update_state (bool, optional): Whether to skip updating the state. Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
             - Output tensor of shape (batch_size, segment_len, dim_input).
             - State tensor of shape (batch_size, state_len, dim_input * mlp_multiplier).
        """
        dtype = x.dtype
        # Apply multi-head attention, followed by layer normalization with residual connection then MLP.
        attn, state = self.attn(x, state, skip_update_state=skip_update_state)
        x = self.attn_norm((self.dropout1(attn) + x).to(torch.float32)).to(dtype)
        mlp_out = self.dropout2(self.mlp(x))
        
        # If no MLP multiplier, then add residual connection.
        if self.mlp_multiplier == 1:
            x = self.mlp_norm((mlp_out + x).to(torch.float32)).to(dtype)
        else:
            x = self.mlp_norm(mlp_out.to(torch.float32)).to(dtype)

        return x, state
