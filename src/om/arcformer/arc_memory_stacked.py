from typing import List, Optional, Tuple, Union

from einops import rearrange
import numpy as np
import torch
from torch import nn
from xformers.ops import memory_efficient_attention

from .positional_embeddings import RoPEEmbeddings
from .util import check_if_linux, extract_state, split_last_dim, StackedLinear


class ARCStacked(nn.Module):
    """Implements ARC (Attentive Recurrent Cell) Transformer memory module."""

    def __init__(
        self, 
        dim_input: int, 
        dims_key: List[int], 
        dims_value: List[int], 
        num_heads: int, 
        segment_len: int, 
        state_len: int,
        dropout: float,
        scaling_factors: List[Optional[float]],
        attn_proj_rank: int,
        num_layers: int,
        layer_num: int,
        cope: bool,
        diff_attn: bool,
        position_embedders: List[Optional[RoPEEmbeddings]],
        xformers_override: bool = False
    ):
        """Initialize module.

        Args:
            dim_input (int): Input dimension.
            dims_key (List[int]): Key dimensions.
            dims_value (List[int]): Value dimensions.
            num_heads (int): Number of attention heads.
            segment_len (int): Segment length (must be a factor of the input sequence length).
            state_len (int): Length of the state (i.e., number of tokens).
            dropout (float): The dropout rate.
            scaling_factors (List[Optional[float]]): Betas for Hopfield memory / scaling factor for SDP attention.
            attn_proj_rank (int): The rank of the attention projection.
            num_layers (int): Number of ARC transformer layers in the parent model.
            layer_num (int): The position of the layer.
            cope (bool): Whether to use CoPE.
            diff_attn (bool): Whether to use diff attention.
            position_embedders (List[Optional[RoPEEmbeddings]]): Position embedding modules.
            xformers_override (bool, optional): Whether to override the use of XFormers attention implementation. Defaults to False.
        """
        super(ARCStacked, self).__init__()

        # Record input parameters
        self.num_heads = num_heads
        self.segment_len = segment_len
        self.state_len = state_len
        self.dropout = dropout
        self.attn_proj_rank = attn_proj_rank
        self.diff_attn = diff_attn
        self.num_layers = num_layers
        self.layer_num = layer_num

        self.dim_input = dim_input
        self.dims_key = dims_key
        self.dims_value = dims_value
        self.scaling_factors = scaling_factors
        self.cope = cope
        self.xformers_override = xformers_override
        
        first_layer = layer_num == 0
        
        # Build attention modules
        self.attn = StatefulCausalMultiAttention(
            dim_input=dim_input,
            dims_key=dims_key,
            dims_value=dims_value,
            num_heads=num_heads,
            seq_len=segment_len,
            state_len=state_len,
            dropout=dropout,
            scaling_factors=scaling_factors,
            attn_proj_rank=attn_proj_rank,
            cope=cope,
            diff_attn=diff_attn,
            layer_num=layer_num,
            position_embedders=position_embedders,
            xformers_override=xformers_override,
        )
        
        # Projection for next state
        diff_attn_mult = 2 if diff_attn else 1
        self.proj_out_state = nn.Linear(num_heads * attn_proj_rank * diff_attn_mult, dim_input, bias=False)
        torch.nn.init.normal_(self.proj_out_state.weight, mean=0.0, std=(1. / (2 * self.num_layers) ** 0.5))
        
        # Projection for output
        self.proj_out = nn.Linear(num_heads * attn_proj_rank * diff_attn_mult, dim_input, bias=False)
        torch.nn.init.normal_(self.proj_out.weight, mean=0.0, std=(1. / (2 * self.num_layers) ** 0.5))
        
        # Set learnable initial state
        if first_layer:
            self.init_state = torch.nn.Parameter(torch.randn(1, state_len, dim_input) / (2. / 5.) ** 0.5)
        else:
            self.init_state = torch.nn.Parameter(torch.randn(1, state_len, dim_input))


    def forward(self, x: torch.Tensor, state: torch.Tensor, skip_update_state: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies recurrent stateful attention to the input tensor x.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, segment_len, dim_input).
            state (torch.Tensor): State tensor of shape (batch_size, state_len, dim_input).
            skip_update_state (bool, optional): Whether to skip updating the state. Defaults to False.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
              - Output tensor of shape (batch_size, segment_len, dim_input)
              - State tensor of shape (batch_size, state_len, dim_input)
        """
        if state.size(0) != x.size(0):
            state = state.repeat(x.size(0), 1, 1)
        
        if skip_update_state:
            # Prepend state to x_seg
            x = torch.cat([state, x], dim=1)
        else:
            # Prepend and append state to x_seg
            x = torch.cat([state, x, state], dim=1)
        
        # Apply attention
        x = self.attn(x, skip_update_state=skip_update_state)
        
        if skip_update_state:
            # Extract state from result
            state = torch.zeros_like(state)
            att = x[:, :, self.state_len:, :]
        else:
            # Extract state from result
            _, att, state_end = extract_state(x, self.state_len)
            
            # Get next state
            state = self.proj_out_state(rearrange(state_end, "b h s d -> b s (h d)"))
        
        # Prepare output to be passed to MLP
        x = self.proj_out(rearrange(att, "b h s d -> b s (h d)"))

        return x, state

class StatefulCausalMultiAttention(nn.Module):
    """Implements a Stateful Causal Multi-Attention module."""
    def __init__(
        self,  
        dim_input: int, 
        dims_key: List[int], 
        dims_value: List[int], 
        num_heads: int,
        seq_len: int,
        state_len: int,
        dropout: float,
        scaling_factors: List[Optional[float]],
        attn_proj_rank: int,
        cope: bool,
        diff_attn: bool,
        layer_num: int,
        position_embedders: List[Optional[RoPEEmbeddings]],
        xformers_override: bool = False,
    ):
        """Initializes the module

        Args:
            dim_input int: The input dimension.
            dims_key (List[int]): The key dimension.
            dims_value (List[int]): The value dimension.
            num_heads (int): Number of attention heads.
            seq_len (int): The maximum length of the sequence.
            state_len (int): The length of the state tensor.
            dropout (float): Dropout rate.
            scaling_factors (List[Optional[float]]): The betas for the Hopfield attention / scaling factor for SDP attention.
            attn_proj_rank (int): The rank of the attention projection.
            cope (bool): Whether to use CoPE.
            diff_attn (bool): Whether to use diff attention.
            layer_num (int): The position of the layer. Only used if diff_attn is True.
            position_embedder (List[Optional[RoPEEmbeddings]]): The position embedder to use.
            xformers_override (bool, optional): Whether to override the use of XFormers attention implementation. Defaults to False.
        """
        super(StatefulCausalMultiAttention, self).__init__()
        
        self.dim_input = dim_input
        self.dims_key = dims_key
        self.dims_value = dims_value
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.state_len = state_len
        self.dropout = dropout
        self.scaling_factors = scaling_factors
        self.attn_proj_rank = attn_proj_rank
        self.diff_attn = diff_attn
        self.layer_num = layer_num
        self.position_embedders = position_embedders
        self.xformers_override = xformers_override
        
        if dims_value[-1] != attn_proj_rank or diff_attn:
            diff_attn_mult = 2 if diff_attn else 1
            self.proj_out = StackedLinear(diff_attn_mult * dims_value[-1], attn_proj_rank, num_heads, bias=False)
            self.proj_out_state = StackedLinear(diff_attn_mult * dims_value[-1], attn_proj_rank, num_heads, bias=False)
            self.use_out_proj = True
        else:
            self.use_out_proj = False
        
        if diff_attn:
            proj_modules = []
            proj_modules_state_start = []
            proj_modules_state_end = []
            attn_modules = [
                StatefulCausalDiffAttention(
                    dim_input=dim_input,
                    dim_key=dims_key[0],
                    dim_value=dims_value[0],
                    num_heads=num_heads,
                    seq_len=seq_len,
                    state_len=state_len,
                    layer_num=layer_num,
                    dropout=dropout,
                    scaling_factor=scaling_factors[0],
                    cope=cope,
                    position_embedder=position_embedders[0],
                )
            ]
        else:
            attn_modules = [
                StatefulCausalAttention(
                    dim_input=dim_input,
                    dim_key=dims_key[0],
                    dim_value=dims_value[0],
                    num_heads=num_heads,
                    seq_len=seq_len,
                    state_len=state_len,
                    dropout=dropout,
                    scaling_factor=scaling_factors[0],
                    cope=cope,
                    position_embedder=position_embedders[0]
                )
            ]
        
        for ix in range(1, len(dims_value)):
            if diff_attn:
                attn_modules.append(
                    StatefulCausalDiffAttention(
                        dim_input=dims_value[ix-1],
                        dim_key=dims_key[ix],
                        dim_value=dims_value[ix],
                        num_heads=num_heads,
                        seq_len=seq_len,
                        state_len=state_len,
                        layer_num=layer_num,
                        dropout=dropout,
                        scaling_factor=scaling_factors[ix],
                        cope=cope,
                        position_embedder=position_embedders[ix],
                        xformers_override=xformers_override
                    )
                )
                proj_modules.append(
                    StackedLinear(2 * dims_value[ix-1], dims_value[ix-1], num_heads, bias=False)
                )
                proj_modules_state_start.append(
                    StackedLinear(2 * dims_value[ix-1], dims_value[ix-1], num_heads, bias=False)
                )
                proj_modules_state_end.append(
                    StackedLinear(2 * dims_value[ix-1], dims_value[ix-1], num_heads, bias=False)
                )
            else:
                attn_modules.append(
                    StatefulCausalAttention(
                        dim_input=dims_value[ix-1],
                        dim_key=dims_key[ix],
                        dim_value=dims_value[ix],
                        num_heads=num_heads,
                        seq_len=seq_len,
                        state_len=state_len,
                        dropout=dropout,
                        scaling_factor=scaling_factors[ix],
                        cope=cope,
                        position_embedder=position_embedders[ix],
                        xformers_override=xformers_override
                    )
                )
                    
        if diff_attn:
            self.proj_modules = nn.ModuleList(proj_modules)
            self.proj_modules_state_start = nn.ModuleList(proj_modules_state_start)
            self.proj_modules_state_end = nn.ModuleList(proj_modules_state_end)
                
        self.attn_modules = nn.ModuleList(attn_modules)
        
    def forward(self, x: torch.Tensor, skip_update_state: bool = False) -> torch.Tensor:
        """
        Applies the StatefulCausalMultiAttention layer to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len + 2 * state_len, dim_in) or (batch_size, seq_len + state_len, dim_in).
            skip_update_state (bool, optional): Whether to skip updating the state. Defaults to False.

        Returns:
            Output tensor. Shape is (batch_size, num_heads, seq_len + 2 * state_len, attn_proj_rank) if skip_update_state is False
            and (batch_size, num_heads, seq_len + state_len, attn_proj_rank) if skip_update_state is True.
                
        """
        x = x.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        if self.diff_attn:
            for ix in range(len(self.attn_modules)):
                x = self.attn_modules[ix](x)
                if ix < len(self.attn_modules) - 1:
                    if skip_update_state:
                        x_state_start = x[:, :self.state_len, :]
                        x = x[:, self.state_len:, :]
                    else:
                        x_state_start, x, x_state_end = extract_state(x, self.state_len)
                    
                    x = self.proj_modules[ix](x)
                    x_state_start = self.proj_modules_state_start[ix](x_state_start)
                    if not skip_update_state:
                        x_state_end = self.proj_modules_state_end[ix](x_state_end)
                    
                    if skip_update_state:
                        x = torch.concat([x_state_start, x], dim=2)
                    else:
                        x = torch.concat([x_state_start, x, x_state_end], dim=2)
        else:
            for attn_module in self.attn_modules:
                x = attn_module(x)
        
        if self.use_out_proj:
            if skip_update_state:
                x_state_start = x[:, :self.state_len, :]
                x = x[:, self.state_len:, :]
            else:
                x_state_start, x, x_state_end = extract_state(x, self.state_len)
            
            x = self.proj_out(x)
            x_state_start = self.proj_out_state(x_state_start)
            if not skip_update_state:
                x_state_end = self.proj_out_state(x_state_end)
            
            if skip_update_state:
                x = torch.concat([x_state_start, x], dim=2)
            else:
                x = torch.concat([x_state_start, x, x_state_end], dim=2)
            
        return x
    
class StatefulCausalAttention(nn.Module):
    """Implements a Stateful Causal Attention module."""
    def __init__(
        self,  
        dim_input: int, 
        dim_key: int, 
        dim_value: int, 
        num_heads: int,
        seq_len: int,
        state_len: int,
        dropout: float = 0.0,
        scaling_factor: Optional[float] = None,
        cope: bool = False,
        position_embedder: Optional[RoPEEmbeddings] = None,
        xformers_override: bool = False,
    ):
        """Initializes the module

        Args:
            dim_input (int): The input dimension.
            dim_key (int): The key dimension.
            dim_value (int): The value dimension.
            num_heads (int): The number of attention heads.
            seq_len (int): The maximum length of the sequence.
            state_len (int): The length of the state tensor.
            dropout (float, optional): The dropout rate. Defaults to 0.0.
            scaling_factor (Optional[float], optional): The scaling factor for attention calculations. Defaults to None (1 / sqrt(dim_key)).
            cope (bool, optional): Whether to use CoPE. Defaults to False.
            position_embedder (Optional[RoPEEmbeddings], optional): The position embedder to use. Defaults to None.
            xformers_override (bool, optional): Whether to override the use of XFormers attention implementation. Defaults to False.
        """
        super(StatefulCausalAttention, self).__init__()
        
        self.dim_input = dim_input
        self.dim_key = dim_key
        self.dim_value = dim_value
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.state_len = state_len
        self.dropout = dropout
        self.cope = cope
        self.position_embedder = position_embedder
        self.xformers_override = xformers_override
        
        self.init_scaling_factor = 1.0 / np.sqrt(dim_key) if scaling_factor is None else scaling_factor
        self.scaling_factor = nn.Parameter(
            torch.Tensor([self.init_scaling_factor]).repeat(num_heads)
        )
        
        # Projections from the attention layer to the next attention layer
        self.proj_q = StackedLinear(dim_input, dim_key, num_heads, bias=False)
        self.proj_k = StackedLinear(dim_input, dim_key, num_heads, bias=False)
        self.proj_v = StackedLinear(dim_input, dim_value, num_heads, bias=False)
        
        # State projections from attention layer to the next attention layer
        self.proj_q_state = StackedLinear(dim_input, dim_key, num_heads, bias=False)
        self.proj_k_state = StackedLinear(dim_input, dim_key, num_heads, bias=False)
        self.proj_v_state = StackedLinear(dim_input, dim_value, num_heads, bias=False)
        
        if cope:
            self.cope_emb = nn.Parameter(
                torch.randn(1, num_heads, self.dim_key, self.seq_len + 2 * self.state_len)
            )
            
    def scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: Optional[float] = None,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Computes scaled dot product attention.

        Args:
            q (torch.Tensor): Query tensor of shape (batch_size, num_heads, seq_len + 2 * state_len, dim_key) or (batch_size, num_heads, seq_len + state_len, dim_key).
            k (torch.Tensor): Key tensor of shape (batch_size, num_heads, seq_len + 2 * state_len, dim_key) or (batch_size, num_heads, seq_len + state_len, dim_key).
            v (torch.Tensor): Value tensor of shape (batch_size, num_heads, seq_len + 2 * state_len, dim_value) or (batch_size, num_heads, seq_len + state_len, dim_value).
            attn_mask (Optional[torch.Tensor], optional): Attention mask tensor of shape (batch_size, num_heads, seq_len, seq_len). Defaults to None.
            dropout_p (Optional[float], optional): Dropout probability. Defaults to None.
            scale (Optional[Union[float, torch.Tensor]], optional): Scaling factor. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_heads, seq_len + 2 * state_len, dim_value) or (batch_size, num_heads, seq_len + state_len, dim_value).
        """
        attn_weights = (q @ k.transpose(-2, -1))
        if scale is None:
            scale = 1.0 / np.sqrt(self.dim_key)
        attn_weights *= scale
        if attn_mask is not None:
            attn_weights += attn_mask
        if dropout_p is not None and dropout_p > 0.0:
            attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout_p, training=self.training)
        
        return attn_weights.softmax(dim=-1) @ v
    
    def apply_attention(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        bias: Optional[torch.Tensor] = None,
        skip_update_state: bool = False,
    ) -> torch.Tensor:
        """
        Applies attention to the input tensors.

        Args:
            q (torch.Tensor): Query tensor of shape (batch_size, num_heads, seq_len + 2 * state_len, dim_key) or (batch_size, num_heads, seq_len + state_len, dim_key).
            k (torch.Tensor): Key tensor of shape (batch_size, num_heads, seq_len + 2 * state_len, dim_key) or (batch_size, num_heads, seq_len + state_len, dim_key).
            v (torch.Tensor): Value tensor of shape (batch_size, num_heads, seq_len + 2 * state_len, dim_value) or (batch_size, num_heads, seq_len + state_len, dim_key).
            bias (Optional[torch.Tensor]): Attention bias vector of shape (batch_size, num_heads, seq_len, seq_len).
            skip_update_state (bool): Whether to skip the state update step.
            
        Returns:
            torch.Tensor: Output tensor. Shape is (batch_size, num_heads, seq_len + 2 * state_len, dim_value) if skip_update_state is False
            and (batch_size, num_heads, seq_len + state_len, dim_value) if skip_update_state is True.
        """
        # Pad the input tensors to have sequence length divisible by 8
        if q.size(-2) % 8 > 0 and not self.training:
            pad_tokens = 8 - (q.size(-2) % 8)
            q = torch.concat([q, torch.full((q.size(0), q.size(1), pad_tokens, q.size(3)), float("-inf"), dtype=q.dtype, device=q.device)], dim=-2)
            k = torch.concat([k, torch.full((k.size(0), k.size(1), pad_tokens, k.size(3)), float("-inf"), dtype=k.dtype, device=k.device)], dim=-2)
            v = torch.concat([v, torch.full((v.size(0), v.size(1), pad_tokens, v.size(3)), float("-inf"), dtype=v.dtype, device=v.device)], dim=-2)
        else:
            pad_tokens = 0
        
        # Get the state length multiplier
        state_len_mult = 2 if skip_update_state else 1
        
        # Get the device and check if xformers attention is available
        device = q.device
        use_xformers_attn = check_if_linux() and "cuda" in str(device) and q.size(-2) % 8 == 0 and not self.xformers_override
        
        # Get the sequence length
        seq_len = q.size(-2) - pad_tokens - state_len_mult * self.state_len
        
        # If position embedder is specified, add positional embeddings to q and k
        if self.position_embedder is not None:
            q, k = self.position_embedder(q), self.position_embedder(k)
        
        # Create attention bias tensor
        attn_bias = torch.tril(
            torch.ones((q.size(0), q.size(1), q.size(2), q.size(2)), device=device, dtype=q.dtype), 
            diagonal=0,
        )
        
        # If not skipping state update, set attention bias for end state tokens reading from start state tokens to 0
        if not skip_update_state:
            attn_bias[..., seq_len + self.state_len:seq_len + 2 * self.state_len, :self.state_len] = 0.0
        attn_bias = attn_bias.log()
        
        # If bias is specified, apply it to the attention for non-state tokens
        if bias is not None:
            if pad_tokens > 0:
                bias = torch.concat([bias, torch.full((bias.size(0), bias.size(1), pad_tokens, bias.size(3)), float("-inf"), dtype=bias.dtype, device=bias.device)], dim=-2)
                bias = torch.concat([bias, torch.full((bias.size(0), bias.size(1), bias.size(2), pad_tokens), float("-inf"), dtype=bias.dtype, device=bias.device)], dim=-1)

            attn_bias += bias.to(dtype=q.dtype)
        
        # If xformers attention is available, use it
        if use_xformers_attn:
            att = torch.concat(
                [
                    memory_efficient_attention(
                        self.scaling_factor[ix] * q[:, ix, :, :],
                        k[:, ix, :, :],
                        v[:, ix, :, :],
                        attn_bias=attn_bias[:, ix, :, :], 
                        p=self.dropout, 
                        scale=1.0
                    ).unsqueeze(1)
                    for ix in range(q.size(1))
                ],
                dim=1
            )
        # Otherwise, use scaled_dot_product_attention method
        else:
            att = torch.concat(
                [
                    self.scaled_dot_product_attention(
                        q[:, ix, :, :],
                        k[:, ix, :, :],
                        v[:, ix, :, :],
                        attn_mask=attn_bias[:, ix, :, :], 
                        dropout_p=self.dropout, 
                        scale=self.scaling_factor[ix]
                    ).unsqueeze(1)
                    for ix in range(q.size(1))
                ],
                dim=1
            )

        # If padding tokens are present, remove them from the output
        if pad_tokens > 0:
            att = att[:, :, :-pad_tokens, :]

        return att

    def forward(self, x: torch.Tensor, skip_update_state: bool = False) -> torch.Tensor:
        """
        Forward pass. Applies StatefulCausalAttention to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_heads, seq_len + 2 * state_len, dim_in) or (batch_size, num_heads, seq_len + state_len, dim_in).
            skip_update_state (bool): Whether to skip updating the state. Defaults to False.

        Returns:
            Output tensor. Shape is (batch_size, num_heads, seq_len + 2 * state_len, dim_value) if skip_update_state is False
            and (batch_size, num_heads, seq_len + state_len, dim_value) if skip_update_state is True.
                
        """
        dtype = x.dtype
        
        if skip_update_state:
            x_state_start = x[:, :self.state_len, :]
            x = x[:, self.state_len:, :]
        else:
            x_state_start, x, x_state_end = extract_state(x, self.state_len)
        
        q = self.proj_q(x).to(torch.float32)
        k = self.proj_k(x).to(torch.float32)
        v = self.proj_v(x).to(torch.float32)
        
        q_state_start = self.proj_q_state(x_state_start).to(torch.float32)
        k_state_start = self.proj_k_state(x_state_start).to(torch.float32)
        v_state_start = self.proj_v_state(x_state_start).to(torch.float32)
        
        # Apply QKV normalization
        q = nn.functional.normalize(q, p=2.0, dim=-1)
        k = nn.functional.normalize(k, p=2.0, dim=-1)
        v = nn.functional.normalize(v, p=2.0, dim=-1)
        
        q_state_start = nn.functional.normalize(q_state_start, p=2.0, dim=-1)
        k_state_start = nn.functional.normalize(k_state_start, p=2.0, dim=-1)
        v_state_start = nn.functional.normalize(v_state_start, p=2.0, dim=-1)
        
        if not skip_update_state:
            k_state_end = self.proj_k_state(x_state_end).to(torch.float32)
            q_state_end = self.proj_q_state(x_state_end).to(torch.float32)
            v_state_end = self.proj_v_state(x_state_end).to(torch.float32)
            
            # Apply QKV normalization
            k = nn.functional.normalize(k, p=2.0, dim=-1)
            q = nn.functional.normalize(q, p=2.0, dim=-1)
            v = nn.functional.normalize(v, p=2.0, dim=-1)
        
        if skip_update_state:
            k = torch.cat([k_state_start, k], dim=2)
            q = torch.cat([q_state_start, q], dim=2)
            v = torch.cat([v_state_start, v], dim=2)
        else:
            k = torch.cat([k_state_start, k, k_state_end], dim=2)
            q = torch.cat([q_state_start, q, q_state_end], dim=2)
            v = torch.cat([v_state_start, v, v_state_end], dim=2)
        
        if self.cope:
            logits = torch.matmul(q, k.transpose(-2, -1))
            gates = torch.sigmoid(logits)
            pos = gates.flip(-1).cumsum(dim=-1).flip(-1)
            state_len_mult = 1 if skip_update_state else 2
            pos = pos.clamp(max=self.seq_len + state_len_mult * self.state_len - 1)
            
            pos_ceil = pos.ceil().long()
            pos_floor = pos.floor().long()
            
            logits_int = torch.matmul(q, self.cope_emb.to(torch.float32))
            logits_ceil = logits_int.gather(-1, pos_ceil)
            logits_floor = logits_int.gather(-1, pos_floor)
            
            w = pos - pos_floor
            
            bias = logits_ceil * w + logits_floor * (1 - w)
        else:
            bias = None
        
        att = self.apply_attention(q, k, v, bias=bias, skip_update_state=skip_update_state).to(dtype)
        
        return att
    
class StatefulCausalDiffAttention(nn.Module):
    """Implements a Stateful Causal Attention module with diff attention."""
    def __init__(
        self,  
        dim_input: int, 
        dim_key: int, 
        dim_value: int, 
        num_heads: int,
        seq_len: int,
        state_len: int,
        layer_num: int,
        dropout: float = 0.0,
        scaling_factor: Optional[float] = None,
        cope: bool = False,
        position_embedder: Optional[RoPEEmbeddings] = None,
        xformers_override: bool = False
    ):
        """Initializes the module

        Args:
            dim_input (int): The input dimension.
            dim_key (int): The key dimension.
            dim_value (int): The value dimension.
            num_heads (int): The number of attention heads.
            seq_len (int): The maximum length of the sequence.
            state_len (int): The length of the state tensor.
            layer_num (int): The position of the layer.
            dropout (float, optional): The dropout rate. Defaults to 0.0.
            scaling_factor (Optional[float], optional): The scaling factor for attention calculations. Defaults to None (1 / sqrt(dim_key)).
            position_embedder (Optional[RoPEEmbeddings], optional): The position embedder to use. Defaults to None.
            xformers_override (bool, optional): Whether to override the use of XFormers attention implementation. Defaults to False.
        """
        super(StatefulCausalDiffAttention, self).__init__()
        
        self.dim_input = dim_input
        self.dim_key = dim_key
        self.dim_value = dim_value
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.state_len = state_len
        self.layer_num = layer_num
        self.dropout = dropout
        self.cope = cope
        self.position_embedder = position_embedder
        self.xformers_override = xformers_override
        
        self.scaling_factor_init = 1.0 / np.sqrt(dim_key) if scaling_factor is None else scaling_factor
        self.scaling_factor = nn.Parameter(
            torch.Tensor([[self.scaling_factor_init, self.scaling_factor_init]]).repeat(num_heads, 1)
        )
        
        # Calculate initial lambda
        self.lambda_init = 0.8 - 0.6 * np.exp(-0.3 * layer_num)
        
        # Initialize lambda params
        self.lambda_q1 = nn.Parameter(torch.randn((num_heads, 1, dim_key)))
        self.lambda_q2 = nn.Parameter(torch.randn((num_heads, 1, dim_key)))
        self.lambda_k1 = nn.Parameter(torch.randn((num_heads, dim_key, 1)))
        self.lambda_k2 = nn.Parameter(torch.randn((num_heads, dim_key, 1)))
        
        # Projections from the attention layer to the next attention layer
        self.proj_q = StackedLinear(dim_input, 2  * dim_key, num_heads, bias=False)
        self.proj_k = StackedLinear(dim_input, 2 * dim_key, num_heads, bias=False)
        self.proj_v = StackedLinear(dim_input, 2 * dim_value, num_heads, bias=False)
        
        # State projections from attention layer to the next attention layer
        self.proj_q_state = StackedLinear(dim_input, 2 * dim_key, num_heads, bias=False)
        self.proj_k_state = StackedLinear(dim_input, 2 * dim_key, num_heads, bias=False)
        self.proj_v_state = StackedLinear(dim_input, 2 * dim_value, num_heads, bias=False)
        
        # If cope is enabled, initialize cope embeddings
        if cope:
            self.cope_emb_1 = nn.Parameter(
                torch.randn(1, self.num_heads, self.dim_key, self.seq_len + 2 * self.state_len)
            )
            self.cope_emb_2 = nn.Parameter(
                torch.randn(1, self.num_heads, self.dim_key, self.seq_len + 2 * self.state_len)
            )
        
        # Initialize output normalization layers
        self.out_norms = nn.ModuleList([nn.RMSNorm(2 * dim_value, eps=1e-5) for _ in range(num_heads)])
    
    def scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: Optional[float] = None,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Computes scaled dot product attention.

        Args:
            q (torch.Tensor): Query tensor of shape (batch_size, num_heads, seq_len + 2 * state_len, dim_key) or (batch_size, num_heads, seq_len + state_len, dim_key).
            k (torch.Tensor): Key tensor of shape (batch_size, num_heads, seq_len + 2 * state_len, dim_key) or (batch_size, num_heads, seq_len + state_len, dim_key).
            v (torch.Tensor): Value tensor of shape (batch_size, num_heads, seq_len + 2 * state_len, dim_value) or (batch_size, num_heads, seq_len + state_len, dim_value).
            attn_mask (Optional[torch.Tensor], optional): Attention mask tensor of shape (batch_size, num_heads, seq_len, seq_len). Defaults to None.
            dropout_p (Optional[float], optional): Dropout probability. Defaults to None.
            scale (Optional[Union[float, torch.Tensor]], optional): Scaling factor. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_heads, seq_len + 2 * state_len, dim_value) or (batch_size, num_heads, seq_len + state_len, dim_value).
        """
        attn_weights = (q @ k.transpose(-2, -1))
        if scale is None:
            scale = 1.0 / np.sqrt(self.dim_key)
        attn_weights *= scale
        if attn_mask is not None:
            attn_weights += attn_mask
        if dropout_p is not None and dropout_p > 0.0:
            attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout_p, training=self.training)
        
        return attn_weights.softmax(dim=-1) @ v
    
    def apply_attention(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        bias: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]] = (None, None),
        skip_update_state: bool = False,
    ) -> torch.Tensor:
        """
        Applies attention to the input tensors.

        Args:
            q (torch.Tensor): Query tensor of shape (batch_size, num_heads, seq_len + 2 * state_len, 2 * dim_key).
            k (torch.Tensor): Key tensor of shape (batch_size, num_heads, seq_len + 2 * state_len, 2 * dim_key).
            v (torch.Tensor): Value tensor of shape (batch_size, num_heads, seq_len + 2 * state_len, 2 * dim_value).
            bias (Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]): Attention bias tensors of shape (batch_size, num_heads, seq_len, seq_len).
            skip_update_state (bool): Whether to skip the state update step.
            
        Returns:
            torch.Tensor: Output tensor. Shape is (batch_size, num_heads, seq_len + 2 * state_len, 2 * dim_value) if skip_update_state is False
            and (batch_size, num_heads, seq_len + 2 * state_len, 2 * dim_value) if skip_update_state is True.
        """
        # Pad the input tensors to have sequence length divisible by 8
        if q.size(-2) % 8 > 0 and not self.training:
            pad_tokens = 8 - (q.size(-2) % 8)
            q = torch.concat([q, torch.full((q.size(0), q.size(1), pad_tokens, q.size(3)), float("-inf"), dtype=q.dtype, device=q.device)], dim=-2)
            k = torch.concat([k, torch.full((k.size(0), k.size(1), pad_tokens, k.size(3)), float("-inf"), dtype=k.dtype, device=k.device)], dim=-2)
            v = torch.concat([v, torch.full((v.size(0), v.size(1), pad_tokens, v.size(3)), float("-inf"), dtype=v.dtype, device=v.device)], dim=-2)
        else:
            pad_tokens = 0
        
        # Check if xformers attention is available
        use_xformers_attn = check_if_linux() and "cuda" in str(q.device) and q.size(-2) % 8 == 0 and not self.xformers_override
        
        # Get the state length multiplier
        state_len_mult = 2 if skip_update_state else 1
        
        # Split q and k into q1, q2, k1, k2
        q1, q2 = split_last_dim(q)
        k1, k2 = split_last_dim(k)
        
        # Split biases
        bias1, bias2 = bias
        
        # Get the sequence length
        seq_len = q.size(-2) - pad_tokens - state_len_mult * self.state_len
        
        # If position embedder is specified, add positional embeddings to q and k
        if self.position_embedder is not None:
            q1, k1 = self.position_embedder(q1), self.position_embedder(k1)
            q2, k2 = self.position_embedder(q2), self.position_embedder(k2)
            
        # Create attention bias tensors
        attn_bias_1 = torch.tril(
            torch.ones((q.size(0), q.size(1), q.size(2), q.size(2)), device=q.device, dtype=q.dtype), 
            diagonal=0,
        )
        
        # If not skipping state update, set attention bias for end state tokens reading from start state tokens to 0
        if not skip_update_state:
            attn_bias_1[..., seq_len + self.state_len:seq_len + 2 * self.state_len, :self.state_len] = 0.0
        attn_bias_1 = attn_bias_1.log()
        
        attn_bias_2 = torch.tril(
            torch.ones((q.size(0), q.size(1), q.size(2), q.size(2)), device=q.device, dtype=q.dtype), 
            diagonal=0,
        )
        
        # If not skipping state update, set attention bias for end state tokens reading from start state tokens to 0
        if not skip_update_state:
            attn_bias_2[..., seq_len + self.state_len:seq_len + 2 * self.state_len, :self.state_len] = 0.0
        attn_bias_2 = attn_bias_2.log()
        
        # If bias is specified, apply it to the attention for non-state tokens
        if bias1 is not None:
            if pad_tokens > 0:
                bias1 = torch.concat([bias1, torch.full((bias1.size(0), bias1.size(1), pad_tokens, bias1.size(3)), float("-inf"), dtype=bias1.dtype, device=bias1.device)], dim=-2)
                bias1 = torch.concat([bias1, torch.full((bias1.size(0), bias1.size(1), bias1.size(2), pad_tokens), float("-inf"), dtype=bias1.dtype, device=bias1.device)], dim=-1)
            
            attn_bias_1 += bias1.to(dtype=q.dtype)
            
        if bias2 is not None:
            if pad_tokens > 0:
                bias2 = torch.concat([bias2, torch.full((bias2.size(0), bias2.size(1), pad_tokens, bias2.size(3)), float("-inf"), dtype=bias2.dtype, device=bias2.device)], dim=-2)
                bias2 = torch.concat([bias2, torch.full((bias2.size(0), bias2.size(1), bias2.size(2), pad_tokens), float("-inf"), dtype=bias2.dtype, device=bias2.device)], dim=-1)
            
            attn_bias_2 += bias2.to(dtype=q.dtype)
            
        # Compute lambda
        lambda_ = (torch.exp(self.lambda_q1 @ self.lambda_k1) - torch.exp(self.lambda_q2 @ self.lambda_k2)).squeeze(0) + self.lambda_init
        
        # If xformers attention is available, use it
        if use_xformers_attn:
            att1 = torch.concat(
                [
                    memory_efficient_attention(
                        self.scaling_factor[ix, 0] * q[:, ix, :, :],
                        k[:, ix, :, :],
                        v[:, ix, :, :],
                        attn_bias=attn_bias_1[:, ix, :, :], 
                        p=self.dropout, 
                        scale=1.0
                    ).unsqueeze(1)
                    for ix in range(q.size(1))
                ],
                dim=1
            )
            att2 = torch.concat(
                [
                    memory_efficient_attention(
                        self.scaling_factor[ix, 1] * q[:, ix, :, :],
                        k[:, ix, :, :],
                        v[:, ix, :, :],
                        attn_bias=attn_bias_2[:, ix, :, :], 
                        p=self.dropout, 
                        scale=1.0
                    ).unsqueeze(1)
                    for ix in range(q.size(1))
                ],
                dim=1
            )
        # Otherwise, use scaled_dot_product_attention method
        else:
            att1 = torch.concat(
                [
                    self.scaled_dot_product_attention(
                        q[:, ix, :, :],
                        k[:, ix, :, :],
                        v[:, ix, :, :],
                        attn_mask=attn_bias_1[:, ix, :, :], 
                        dropout_p=self.dropout, 
                        scale=self.scaling_factor[ix, 0]
                    ).unsqueeze(1)
                    for ix in range(q.size(1))
                ],
                dim=1
            )
            att2 = torch.concat(
                [
                    self.scaled_dot_product_attention(
                        q[:, ix, :, :],
                        k[:, ix, :, :],
                        v[:, ix, :, :],
                        attn_mask=attn_bias_2[:, ix, :, :], 
                        dropout_p=self.dropout, 
                        scale=self.scaling_factor[ix, 1]
                    ).unsqueeze(1)
                    for ix in range(q.size(1))
                ],
                dim=1
            )
        
        # Compute output attention
        att = att1 - lambda_ * att2
        
        # If padding tokens are present, remove them from the output
        if pad_tokens > 0:
            att = att[:, :, :-pad_tokens, :]

        return att

    def forward(self, x: torch.Tensor, skip_update_state: bool = False) -> torch.Tensor:
        """
        Forward pass. Applies StatefulCausalDiffAttention to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len + 2 * state_len, dim_in) or (batch_size, seq_len + state_len, dim_in).
            skip_update_state (bool, optional): Whether to skip updating the state. Defaults to False.

        Returns:
            Output tensor. Shape is (batch_size, seq_len + 2 * state_len, 2 * dim_value) if skip_update_state is False
            and (batch_size, seq_len + state_len, 2 * dim_value) if skip_update_state is True.
                
        """
        dtype = x.dtype
        
        if skip_update_state:
            x_state_start = x[..., :self.state_len, :]
            x = x[..., self.state_len:, :]
        else:
            x_state_start, x, x_state_end = extract_state(x, self.state_len)
        
        q = self.proj_q(x).to(torch.float32)
        k = self.proj_k(x).to(torch.float32)
        v = self.proj_v(x).to(torch.float32)
        
        k_state_start = self.proj_k_state(x_state_start).to(torch.float32)
        q_state_start = self.proj_q_state(x_state_start).to(torch.float32)
        v_state_start = self.proj_v_state(x_state_start).to(torch.float32)
        
        # Apply QKV normalization
        q = nn.functional.normalize(q, p=2.0, dim=-1)
        k = nn.functional.normalize(k, p=2.0, dim=-1)
        v = nn.functional.normalize(v, p=2.0, dim=-1)
        
        q_state_start = nn.functional.normalize(q_state_start, p=2.0, dim=-1)
        k_state_start = nn.functional.normalize(k_state_start, p=2.0, dim=-1)
        v_state_start = nn.functional.normalize(v_state_start, p=2.0, dim=-1)
        
        if not skip_update_state:
            q_state_end = self.proj_q_state(x_state_end).to(torch.float32)
            k_state_end = self.proj_k_state(x_state_end).to(torch.float32)
            v_state_end = self.proj_v_state(x_state_end).to(torch.float32)
            
            # Apply QKV normalization
            q_state_end = nn.functional.normalize(q_state_end, p=2.0, dim=-1)
            k_state_end = nn.functional.normalize(k_state_end, p=2.0, dim=-1)
            v_state_end = nn.functional.normalize(v_state_end, p=2.0, dim=-1)
        
        if skip_update_state:
            q = torch.cat([q_state_start, q], dim=2)
            k = torch.cat([k_state_start, k], dim=2)
            v = torch.cat([v_state_start, v], dim=2)
        else:
            q = torch.cat([q_state_start, q, q_state_end], dim=2)
            k = torch.cat([k_state_start, k, k_state_end], dim=2)
            v = torch.cat([v_state_start, v, v_state_end], dim=2)
        
        if self.cope:
            q1, q2 = split_last_dim(q)
            k1, k2 = split_last_dim(k)
            
            logits1 = q1 @ k1.transpose(-2, -1)
            logits2 = q2 @ k2.transpose(-2, -1)
            
            gates1 = torch.sigmoid(logits1)
            gates2 = torch.sigmoid(logits2)
            
            state_len_mult = 1 if skip_update_state else 2
            
            pos1 = gates1.flip(-1).cumsum(dim=-1).flip(-1)
            pos1 = pos1.clamp(max=self.seq_len + state_len_mult * self.state_len - 1)
            pos2 = gates2.flip(-1).cumsum(dim=-1).flip(-1)
            pos2 = pos2.clamp(max=self.seq_len + state_len_mult * self.state_len - 1)
            
            pos_ceil_1 = pos1.ceil().long()
            pos_floor_1 = pos1.floor().long()
            pos_ceil_2 = pos2.ceil().long()
            pos_floor_2 = pos2.floor().long()
            
            logits_int_1 = q1 @ self.cope_emb_1.to(torch.float32)
            logits_ceil_1 = logits_int_1.gather(-1, pos_ceil_1)
            logits_floor_1 = logits_int_1.gather(-1, pos_floor_1)
            logits_int_2 = q2 @ self.cope_emb_2.to(torch.float32)
            logits_ceil_2 = logits_int_2.gather(-1, pos_ceil_2)
            logits_floor_2 = logits_int_2.gather(-1, pos_floor_2)
            
            w1 = pos1 - pos_floor_1
            w2 = pos2 - pos_floor_2
            
            bias1 = logits_ceil_1 * w1 + logits_floor_1 * (1 - w1)
            bias2 = logits_ceil_2 * w2 + logits_floor_2 * (1 - w2)
        else:
            bias1 = None
            bias2 = None
        
        att = self.apply_attention(q, k, v, bias=(bias1, bias2), skip_update_state=skip_update_state)
        
        att = torch.concatenate(
            [norm(att[:, ix, :, :].to(torch.float32).to(dtype)).unsqueeze(1) for ix, norm in enumerate(self.out_norms)],
            dim=1
        )
        
        att = (1. - self.lambda_init) * att
        
        return att
