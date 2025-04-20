from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from xformers.ops import memory_efficient_attention

from .positional_embeddings import RoPEEmbeddings
from .util import check_if_linux, extract_state, split_last_dim


class ARCUnstacked(nn.Module):
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
        position_embedders: List[Optional[RoPEEmbeddings]]
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
        """
        super(ARCUnstacked, self).__init__()

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
        
        first_layer = layer_num == 0
        
        # Build attention modules
        self.attn = StatefulCausalMHMA(
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
            x = torch.cat([state.clone(), x], dim=1)
        else:
            # Prepend and append state to x_seg
            x = torch.cat([state.clone(), x, state.clone()], dim=1)
        
        # Apply attention
        x = self.attn(x, skip_update_state=skip_update_state)
        
        if skip_update_state:
            # Extract state from result
            state = torch.zeros_like(state)
            att = x[:, self.state_len:, :]
        else:
            # Extract state from result
            _, att, state_end = extract_state(x, self.state_len)
            
            # Get next state
            state = self.proj_out_state(state_end)
        
        # Prepare output to be passed to MLP
        x = self.proj_out(att)

        return x, state

class StatefulCausalMHMA(nn.Module):
    """Implements a Stateful Causal Multi-Head Multi-Attention (MHMA) module."""
    def __init__(
        self,  
        dim_input: List[int], 
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
    ):
        """Initializes the module

        Args:
            dim_input (int): The input dimension.
            dims_key (List[int]): The key dimension.
            dims_value (List[int]): The value dimension.
            num_heads (int): Number of attention heads.
            seq_len (int): The maximum length of the input sequence.
            state_len (int): The length of the state tensor.
            dropout (float): The dropout rate.
            scaling_factors (List[Optional[float]]): The betas for Hopfield attention / scaling factors for SDP attention.
            attn_proj_rank (int): The rank of the attention projection.
            cope (bool): Whether to use CoPE.
            diff_attn (bool): Whether to use diff attention.
            layer_num (int): The position of the layer.
            position_embedders (List[Optional[RoPEEmbeddings]]): The position embedder to use.
        """
        super(StatefulCausalMHMA, self).__init__()
        
        self.dim_input = dim_input
        self.dims_key = dims_key
        self.dims_value = dims_value
        self.num_heads = num_heads
        self.layer_num = layer_num
        self.seq_len = seq_len
        self.state_len = state_len
        self.diff_attn = diff_attn
        self.dropout = dropout
        self.scaling_factors = scaling_factors
        self.attn_proj_rank = attn_proj_rank
        self.position_embedders = position_embedders
        
        self.attn_heads = nn.ModuleList(
            [
                StatefulCausalMultiAttention(
                    dim_input=dim_input,
                    dims_key=dims_key,
                    dims_value=dims_value,
                    seq_len=seq_len,
                    state_len=state_len,
                    dropout=dropout,
                    scaling_factors=scaling_factors,
                    attn_proj_rank=attn_proj_rank,
                    cope=cope,
                    diff_attn=diff_attn,
                    layer_num=layer_num,
                    position_embedders=position_embedders,
                ) for _ in range(num_heads)
            ]
        )
        
    def forward(self, x: torch.Tensor, skip_update_state: bool = False) -> torch.Tensor:
        """Applies stateful causal multi-layer multi-head attention to the input tensor x.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len + 2 * state_len, dim_input) or (batch_size, seq_len + state_len, dim_input).
            skip_update_state (bool, optional): Whether to skip updating the state. Defaults to False.

        Returns:
            torch.Tensor: Output tensor. Shape is (batch_size, seq_len + 2 * state_len, attn_proj_rank * num_heads) if skip_update_state is False
            and (batch_size, seq_len + state_len, attn_proj_rank * num_heads) if skip_update_state is True.
        """
        out_mult = 1. if not self.diff_attn else (1. - self.attn_heads[0].attn_modules[0].lambda_init)
        
        return out_mult * torch.concat(
            [attn_head(x, skip_update_state=skip_update_state) for attn_head in self.attn_heads], 
            dim=-1
        )
    

class StatefulCausalMultiAttention(nn.Module):
    """Implements a Stateful Causal Multi-Attention module."""
    def __init__(
        self,  
        dim_input: int, 
        dims_key: List[int], 
        dims_value: List[int], 
        seq_len: int,
        state_len: int,
        dropout: float,
        scaling_factors: List[Optional[float]],
        attn_proj_rank: int,
        cope: bool,
        diff_attn: bool,
        layer_num: int,
        position_embedders: List[Optional[RoPEEmbeddings]],
    ):
        """Initializes the module

        Args:
            dim_input int: The input dimension.
            dims_key (List[int]): The key dimension.
            dims_value (List[int]): The value dimension.
            seq_len (int): The maximum length of the sequence.
            state_len (int): The length of the state tensor.
            dropout (float): Dropout rate.
            scaling_factors (List[Optional[float]]): The betas for the Hopfield attention / scaling factor for SDP attention.
            attn_proj_rank (int): The rank of the attention projection.
            cope (bool): Whether to use CoPE.
            diff_attn (bool): Whether to use diff attention.
            layer_num (int): The position of the layer. Only used if diff_attn is True.
            position_embedder (List[Optional[RoPEEmbeddings]]): The position embedder to use.
        """
        super(StatefulCausalMultiAttention, self).__init__()
        
        self.dim_input = dim_input
        self.dims_key = dims_key
        self.dims_value = dims_value
        self.seq_len = seq_len
        self.state_len = state_len
        self.dropout = dropout
        self.scaling_factors = scaling_factors
        self.attn_proj_rank = attn_proj_rank
        self.diff_attn = diff_attn
        self.layer_num = layer_num
        self.position_embedders = position_embedders
        
        if dims_value[-1] != attn_proj_rank:
            diff_attn_mult = 2 if diff_attn else 1
            self.proj_out = nn.Linear(diff_attn_mult * dims_value[-1], attn_proj_rank, bias=False)
            self.proj_out_state = nn.Linear(diff_attn_mult * dims_value[-1], attn_proj_rank, bias=False)
            self.use_out_proj = True
        else:
            self.use_out_proj = False
        
        if diff_attn:
            proj_modules = []
            proj_modules_state = []
            attn_modules = [
                StatefulCausalDiffAttentionHead(
                    dim_input=dim_input,
                    dim_key=dims_key[0],
                    dim_value=dims_value[0],
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
                StatefulCausalAttentionHead(
                    dim_input=dim_input,
                    dim_key=dims_key[0],
                    dim_value=dims_value[0],
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
                    StatefulCausalDiffAttentionHead(
                        dim_input=dims_value[ix-1],
                        dim_key=dims_key[ix],
                        dim_value=dims_value[ix],
                        seq_len=seq_len,
                        state_len=state_len,
                        layer_num=layer_num,
                        dropout=dropout,
                        scaling_factor=scaling_factors[ix],
                        cope=cope,
                        position_embedder=position_embedders[ix],
                    )
                )
                proj_modules.append(
                    nn.Linear(2 * dims_value[ix-1], dims_value[ix-1], bias=False)
                )
                proj_modules_state.append(
                    nn.Linear(2 * dims_value[ix-1], dims_value[ix-1], bias=False)
                )
            else:
                attn_modules.append(
                    StatefulCausalAttentionHead(
                        dim_input=dims_value[ix-1],
                        dim_key=dims_key[ix],
                        dim_value=dims_value[ix],
                        seq_len=seq_len,
                        state_len=state_len,
                        dropout=dropout,
                        scaling_factor=scaling_factors[ix],
                        cope=cope,
                        position_embedder=position_embedders[ix]
                    )
                )
                    
        if diff_attn:
            self.proj_modules = nn.ModuleList(proj_modules)
            self.proj_modules_state = nn.ModuleList(proj_modules_state)
                
        self.attn_modules = nn.ModuleList(attn_modules)
        
    def forward(self, x: torch.Tensor, skip_update_state: bool = False) -> torch.Tensor:
        """
        Applies the StatefulCausalMultiAttention layer to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len + 2 * state_len, dim_in) or (batch_size, seq_len + state_len, dim_in).
            skip_update_state (bool, optional): Whether to skip updating the state. Defaults to False.

        Returns:
            Output tensor. Shape is (batch_size, seq_len + 2 * state_len, attn_proj_rank) if skip_update_state is False
            and (batch_size, seq_len + state_len, attn_proj_rank) if skip_update_state is True.
                
        """
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
                    x_state_start = self.proj_modules_state[ix](x_state_start)
                    if not skip_update_state:
                        x_state_end = self.proj_modules_state[ix](x_state_end)
                    
                    if skip_update_state:
                        x = torch.concat([x_state_start, x], dim=1)
                    else:
                        x = torch.concat([x_state_start, x, x_state_end], dim=1)
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
                x = torch.concat([x_state_start, x], dim=1)
            else:
                x = torch.concat([x_state_start, x, x_state_end], dim=1)
            
        return x
    
class StatefulCausalAttentionHead(nn.Module):
    """Implements a Stateful Causal Attention Head module."""
    def __init__(
        self,  
        dim_input: int, 
        dim_key: int, 
        dim_value: int, 
        seq_len: int,
        state_len: int,
        dropout: float = 0.0,
        scaling_factor: Optional[float] = None,
        cope: bool = False,
        position_embedder: Optional[RoPEEmbeddings] = None,
    ):
        """Initializes the module

        Args:
            dim_input (int): The input dimension.
            dim_key (int): The key dimension.
            dim_value (int): The value dimension.
            seq_len (int): The maximum length of the sequence.
            state_len (int): The length of the state tensor.
            dropout (float, optional): The dropout rate. Defaults to 0.0.
            scaling_factor (Optional[float], optional): The scaling factor for attention calculations. Defaults to None (1 / sqrt(dim_key)).
            cope (bool, optional): Whether to use CoPE. Defaults to False.
            position_embedder (Optional[RoPEEmbeddings], optional): The position embedder to use. Defaults to None.
        """
        super(StatefulCausalAttentionHead, self).__init__()
        
        self.dim_input = dim_input
        self.dim_key = dim_key
        self.dim_value = dim_value
        self.seq_len = seq_len
        self.state_len = state_len
        self.dropout = dropout
        self.cope = cope
        self.position_embedder = position_embedder
        
        self.init_scaling_factor = scaling_factor
        self.scaling_factor = nn.Parameter(
            torch.Tensor([self.init_scaling_factor])
        )
        
        # Projections from the attention layer to the next attention layer
        self.proj_q = nn.Linear(dim_input, dim_key, bias=False)
        self.proj_k = nn.Linear(dim_input, dim_key, bias=False)
        self.proj_v = nn.Linear(dim_input, dim_value, bias=False)
        
        # State projections from attention layer to the next attention layer
        self.proj_q_state = nn.Linear(dim_input, dim_key, bias=False)
        self.proj_k_state = nn.Linear(dim_input, dim_key, bias=False)
        self.proj_v_state = nn.Linear(dim_input, dim_value, bias=False)
        
        if cope:
            self.cope_emb = nn.Parameter(
                torch.randn(1, self.dim_key, self.seq_len + 2 * self.state_len)
            )
    
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
            q (torch.Tensor): Query tensor of shape (batch_size, seq_len + 2 * state_len, dim_key) or (batch_size, seq_len + state_len, dim_key).
            k (torch.Tensor): Key tensor of shape (batch_size, seq_len + 2 * state_len, dim_key) or (batch_size, seq_len + state_len, dim_key).
            v (torch.Tensor): Value tensor of shape (batch_size, seq_len + 2 * state_len, dim_value) or (batch_size, seq_len + state_len, dim_key).
            bias (Optional[torch.Tensor]): Attention bias vector of shape (batch_size, seq_len, seq_len).
            skip_update_state (bool): Whether to skip the state update step.
            
        Returns:
            torch.Tensor: Output tensor. Shape is (batch_size, seq_len + 2 * state_len, dim_value) if skip_update_state is False
            and (batch_size, seq_len + state_len, dim_value) if skip_update_state is True.
        """
        device = q.device
        
        # Pad the input tensors to have sequence length divisible by 8
        if q.size(-2) % 8 > 0 and not self.training:
            pad_tokens = 8 - (q.size(-2) % 8)
            q = torch.concat([q, torch.full((q.size(0), pad_tokens, q.size(2)), float("-inf"), dtype=q.dtype, device=device)], dim=-2)
            k = torch.concat([k, torch.full((k.size(0), pad_tokens, k.size(2)), float("-inf"), dtype=k.dtype, device=device)], dim=-2)
            v = torch.concat([v, torch.full((v.size(0), pad_tokens, v.size(2)), float("-inf"), dtype=v.dtype, device=device)], dim=-2)
        else:
            pad_tokens = 0
        
        # Get the state length multiplier
        state_len_mult = 2 if skip_update_state else 1
        
        # Get the sequence length
        seq_len = q.size(-2) - pad_tokens - state_len_mult * self.state_len
        
        # Check if xformers attention is available
        use_xformers_attn = check_if_linux() and "cuda" in str(device) and q.size(-2) % 8 == 0
        
        # If position embedder is specified, add positional embeddings to q and k
        if self.position_embedder is not None:
            q, k = self.position_embedder(q), self.position_embedder(k)
        
        # Get the attention bias
        attn_bias = torch.tril(
            torch.ones((q.size(0), q.size(1), q.size(1)), device=device, dtype=q.dtype), 
            diagonal=0,
        )
        if not skip_update_state:
            attn_bias[..., seq_len + self.state_len:seq_len + 2 * self.state_len, :self.state_len] = 0.0
        attn_bias = attn_bias.log()
        
        # If bias is specified, apply it to the attention for non-state tokens
        if bias is not None:
            if pad_tokens > 0:
                bias = torch.concat([bias, torch.full((bias.size(0), pad_tokens, bias.size(2)), float("-inf"), dtype=bias.dtype, device=bias.device)], dim=-2)
                bias = torch.concat([bias, torch.full((bias.size(0), bias.size(1), pad_tokens), float("-inf"), dtype=bias.dtype, device=bias.device)], dim=-1)
            
            attn_bias += bias.to(dtype=q.dtype)
        
        # If xformers attention is available, use it
        if use_xformers_attn:
            att = memory_efficient_attention(q, k, v, attn_bias=attn_bias, p=self.dropout, scale=self.scaling_factor[0])
        # Otherwise, use the standard attention
        else:
            att = torch.softmax((q @ k.transpose(-2, -1)) * self.scaling_factor[0] + attn_bias, dim=-1)
            if self.training and self.dropout > 0:
                att = torch.dropout(att, p=self.dropout, train=True)
            att = att @ v

        # If padding tokens are present, remove them from the output
        if pad_tokens > 0:
            att = att[:, :-pad_tokens, :]

        return att

    def forward(self, x: torch.Tensor, skip_update_state: bool = False) -> torch.Tensor:
        """
        Forward pass. Applies StatefulCausalAttention to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len + 2 * state_len, dim_in) or (batch_size, seq_len + state_len, dim_in).
            skip_update_state (bool): Whether to skip updating the state. Defaults to False.

        Returns:
            Output tensor. Shape is (batch_size, seq_len + 2 * state_len, dim_value) if skip_update_state is False
            and (batch_size, seq_len + state_len, dim_value) if skip_update_state is True.
                
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
            q_state_end = self.proj_q_state(x_state_end).to(torch.float32)
            k_state_end = self.proj_k_state(x_state_end).to(torch.float32)
            v_state_end = self.proj_v_state(x_state_end).to(torch.float32)
            
            # Apply QKV normalization
            q_state_end = nn.functional.normalize(q_state_end, p=2.0, dim=-1)
            k_state_end = nn.functional.normalize(k_state_end, p=2.0, dim=-1)
            v_state_end = nn.functional.normalize(v_state_end, p=2.0, dim=-1)
        
        if skip_update_state:
            q = torch.cat([q_state_start, q], dim=1)
            k = torch.cat([k_state_start, k], dim=1)
            v = torch.cat([v_state_start, v], dim=1)
        else:
            q = torch.cat([q_state_start, q, q_state_end], dim=1)
            k = torch.cat([k_state_start, k, k_state_end], dim=1)
            v = torch.cat([v_state_start, v, v_state_end], dim=1)
        
        if self.cope:
            logits = q @ k.transpose(-2, -1)
            gates = torch.sigmoid(logits)
            pos = gates.flip(-1).cumsum(dim=-1).flip(-1)
            state_len_mult = 1 if skip_update_state else 2
            pos = pos.clamp(max=self.seq_len + state_len_mult * self.state_len - 1)
            
            pos_ceil = pos.ceil().long()
            pos_floor = pos.floor().long()
            
            logits_int = q @ self.cope_emb.to(torch.float32)
            logits_ceil = logits_int.gather(-1, pos_ceil)
            logits_floor = logits_int.gather(-1, pos_floor)
            
            w = pos - pos_floor
            
            bias = logits_ceil * w + logits_floor * (1 - w)
        else:
            bias = None
        
        att = self.apply_attention(q, k, v, bias=bias, skip_update_state=skip_update_state).to(dtype)
        
        return att
    
class StatefulCausalDiffAttentionHead(nn.Module):
    """Implements a Stateful Causal Attention Head module with diff attention."""
    def __init__(
        self,  
        dim_input: int, 
        dim_key: int, 
        dim_value: int, 
        seq_len: int,
        state_len: int,
        layer_num: int,
        dropout: float = 0.0,
        scaling_factor: Optional[float] = None,
        cope: bool = False,
        position_embedder: Optional[RoPEEmbeddings] = None,
    ):
        """Initializes the module

        Args:
            dim_input (int): The input dimension.
            dim_key (int): The key dimension.
            dim_value (int): The value dimension.
            seq_len (int): The maximum length of the sequence.
            state_len (int): The length of the state tensor.
            layer_num (int): The position of the layer.
            dropout (float, optional): The dropout rate. Defaults to 0.0.
            scaling_factor (Optional[float], optional): The scaling factor for attention calculations. Defaults to None (1 / sqrt(dim_key)).
            position_embedder (Optional[RoPEEmbeddings], optional): The position embedder to use. Defaults to None.
        """
        super(StatefulCausalDiffAttentionHead, self).__init__()
        
        self.dim_input = dim_input
        self.dim_key = dim_key
        self.dim_value = dim_value
        self.seq_len = seq_len
        self.state_len = state_len
        self.layer_num = layer_num
        self.dropout = dropout
        self.cope = cope
        self.position_embedder = position_embedder
        
        self.init_scaling_factor = 1.0 / np.sqrt(dim_key) if scaling_factor is None else scaling_factor
        self.scaling_factor = nn.Parameter(
            torch.Tensor([self.init_scaling_factor, self.init_scaling_factor])
        )
        
        # Calculate initial lambda
        self.lambda_init = 0.8 - 0.6 * np.exp(-0.3 * layer_num)
        
        # Initialize lambda params
        self.lambda_q1 = nn.Parameter(torch.randn((1, dim_key)))
        self.lambda_q2 = nn.Parameter(torch.randn((1, dim_key)))
        self.lambda_k1 = nn.Parameter(torch.randn((dim_key, 1)))
        self.lambda_k2 = nn.Parameter(torch.randn((dim_key, 1)))
        
        # Projections from the attention layer to the next attention layer
        self.proj_q = nn.Linear(dim_input, 2  * dim_key, bias=False)
        self.proj_k = nn.Linear(dim_input, 2 * dim_key, bias=False)
        self.proj_v = nn.Linear(dim_input, 2 * dim_value, bias=False)
        
        # State projections from attention layer to the next attention layer
        self.proj_q_state = nn.Linear(dim_input, 2 * dim_key, bias=False)
        self.proj_k_state = nn.Linear(dim_input, 2 * dim_key, bias=False)
        self.proj_v_state = nn.Linear(dim_input, 2 * dim_value, bias=False)
        
        if cope:
            self.cope_emb_1 = nn.Parameter(
                torch.randn(1, self.dim_key, self.seq_len + 2 * self.state_len)
            )
            self.cope_emb_2 = nn.Parameter(
                torch.randn(1, self.dim_key, self.seq_len + 2 * self.state_len)
            )
            
        self.out_norm = nn.RMSNorm(2 * dim_value, eps=1e-5)
    
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
            q (torch.Tensor): Query tensor of shape (batch_size, seq_len + 2 * state_len, 2 * dim_key).
            k (torch.Tensor): Key tensor of shape (batch_size, seq_len + 2 * state_len, 2 * dim_key).
            v (torch.Tensor): Value tensor of shape (batch_size, seq_len + 2 * state_len, 2 * dim_value).
            bias (Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]): Attention bias tensors of shape (batch_size, seq_len, seq_len).
            skip_update_state (bool): Whether to skip the state update step.
            
        Returns:
            torch.Tensor: Output tensor. Shape is (batch_size, seq_len + 2 * state_len, 2 * dim_value) if skip_update_state is False
            and (batch_size, seq_len + 2 * state_len, 2 * dim_value) if skip_update_state is True.
        """
        device = q.device
        
        # Pad the input tensors to have sequence length divisible by 8
        if q.size(-2) % 8 > 0 and not self.training:
            pad_tokens = 8 - (q.size(-2) % 8)
            q = torch.concat([q, torch.full((q.size(0), pad_tokens, q.size(2)), float("-inf"), dtype=q.dtype, device=device)], dim=-2)
            k = torch.concat([k, torch.full((k.size(0), pad_tokens, k.size(2)), float("-inf"), dtype=k.dtype, device=device)], dim=-2)
            v = torch.concat([v, torch.full((v.size(0), pad_tokens, v.size(2)), float("-inf"), dtype=v.dtype, device=device)], dim=-2)
        else:
            pad_tokens = 0
        
        # Check if xformers attention is available
        use_xformers_attn = check_if_linux() and "cuda" in str(device) and q.size(-2) % 8 == 0
        
        # Split q and k into q1, q2, k1, k2
        q1, q2 = split_last_dim(q)
        k1, k2 = split_last_dim(k)
        
        # Get the state length multiplier
        state_len_mult = 2 if not skip_update_state else 1
        
        # Get the sequence length
        seq_len = q.size(-2) - pad_tokens - state_len_mult * self.state_len
        
        # Split biases
        bias1, bias2 = bias
        
        # If position embedder is specified, add positional embeddings to q and k
        if self.position_embedder is not None:
            q1, k1 = self.position_embedder(q1), self.position_embedder(k1)
            q2, k2 = self.position_embedder(q2), self.position_embedder(k2)
        
        # Create attention bias tensors
        attn_bias_1 = torch.tril(
            torch.ones((q.size(0), q.size(1), q.size(1)), device=device, dtype=q.dtype), 
            diagonal=0,
        )
        if not skip_update_state:
            attn_bias_1[..., seq_len + self.state_len:seq_len + 2 * self.state_len, :self.state_len] = 0.0
        attn_bias_1 = attn_bias_1.log()
        
        attn_bias_2 = torch.tril(
            torch.ones((q.size(0), q.size(1), q.size(1)), device=device, dtype=q.dtype), 
            diagonal=0,
        )
        if not skip_update_state:
            attn_bias_2[..., seq_len + self.state_len:seq_len + 2 * self.state_len, :self.state_len] = 0.0
        attn_bias_2 = attn_bias_2.log()
        
        # If bias is specified, apply it to the attention
        if bias1 is not None:
            if pad_tokens > 0:
                bias1 = torch.concat([bias1, torch.full((bias1.size(0), pad_tokens, bias1.size(2)), float("-inf"), dtype=bias1.dtype, device=bias1.device)], dim=-2)
                bias1 = torch.concat([bias1, torch.full((bias1.size(0), bias1.size(1), pad_tokens), float("-inf"), dtype=bias1.dtype, device=bias1.device)], dim=-1)
            attn_bias_1 += bias1.to(dtype=q.dtype)
            
        if bias2 is not None:
            if pad_tokens > 0:
                bias2 = torch.concat([bias2, torch.full((bias2.size(0), pad_tokens, bias2.size(2)), float("-inf"), dtype=bias2.dtype, device=bias2.device)], dim=-2)
                bias2 = torch.concat([bias2, torch.full((bias2.size(0), bias2.size(1), pad_tokens), float("-inf"), dtype=bias2.dtype, device=bias2.device)], dim=-1)
            attn_bias_2 += bias2.to(dtype=q.dtype)
        
        # Compute lambda
        lambda_ = (torch.exp(self.lambda_q1 @ self.lambda_k1) - torch.exp(self.lambda_q2 @ self.lambda_k2)).squeeze(0) + self.lambda_init
        
        # If xformers is available, use it for attention computation
        if use_xformers_attn:
            att1 = memory_efficient_attention(q1, k1, v, attn_bias=attn_bias_1, p=self.dropout, scale=self.scaling_factor[0])
            att2 = memory_efficient_attention(q2, k2, v, attn_bias=attn_bias_2, p=self.dropout, scale=self.scaling_factor[1])
        # Otherwise, use PyTorch's attention implementation
        else:
            att1 = torch.nn.functional.scaled_dot_product_attention(
                q1, k1, v, attn_mask=attn_bias_1, dropout_p=self.dropout, scale=self.scaling_factor[0]
            )
            att2 = torch.nn.functional.scaled_dot_product_attention(
                q2, k2, v, attn_mask=attn_bias_2, dropout_p=self.dropout, scale=self.scaling_factor[1]
            )
        
        # Combine the attended values
        att = att1 - lambda_ * att2
        
        # If padding tokens are present, remove them from the output
        if pad_tokens > 0:
            att = att[:, :-pad_tokens, :]

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
            q_state_end = self.proj_q_state(x_state_end).to(torch.float32)
            k_state_end = self.proj_k_state(x_state_end).to(torch.float32)
            v_state_end = self.proj_v_state(x_state_end).to(torch.float32)
            
            # Apply QKV normalization
            q_state_end = nn.functional.normalize(q_state_end, p=2.0, dim=-1)
            k_state_end = nn.functional.normalize(k_state_end, p=2.0, dim=-1)
            v_state_end = nn.functional.normalize(v_state_end, p=2.0, dim=-1)
        
        if skip_update_state:
            q = torch.cat([q_state_start, q], dim=1)
            k = torch.cat([k_state_start, k], dim=1)
            v = torch.cat([v_state_start, v], dim=1)
        else:
            q = torch.cat([q_state_start, q, q_state_end], dim=1)
            k = torch.cat([k_state_start, k, k_state_end], dim=1)
            v = torch.cat([v_state_start, v, v_state_end], dim=1)
        
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
        
        return self.out_norm(att).to(dtype)
