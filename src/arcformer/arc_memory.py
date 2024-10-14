from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from xformers.ops import memory_efficient_attention, LowerTriangularMask

from .positional_embeddings import RoPEEmbeddings
from .util import extract_state, split_last_dim


class ARC(nn.Module):
    """Implements ARC (Attentive Recurrent Cell) Transformer memory module."""

    def __init__(
        self, 
        dim_input: int, 
        dims_key: List[int], 
        dims_value: List[int], 
        num_heads: int, 
        segment_len: int, 
        state_len: int,
        attn_normalize: bool,
        dropout: float,
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
            attn_normalize (bool): Whether to normalize the attention inputs.
            dropout (float): The dropout rate.
            attn_proj_rank (int): The rank of the attention projection.
            num_layers (int): Number of ARC transformer layers in the parent model.
            layer_num (int): The position of the layer.
            cope (bool): Whether to use CoPE.
            diff_attn (bool): Whether to use diff attention.
            position_embedders (List[Optional[RoPEEmbeddings]]): Position embedding modules.
        """
        super(ARC, self).__init__()

        # Record input parameters
        self.num_heads = num_heads
        self.segment_len = segment_len
        self.state_len = state_len
        self.attn_normalize = attn_normalize
        self.dropout = dropout
        self.attn_proj_rank = attn_proj_rank
        self.diff_attn = diff_attn
        self.num_layers = num_layers
        self.layer_num = layer_num

        self.dim_input = dim_input
        self.dims_key = dims_key
        self.dims_value = dims_value
        
        first_layer = layer_num == 0
        
        # Build attention modules
        self.attn = StatefulCausalMHMA(
            dim_input=dim_input,
            dims_key=dims_key,
            dims_value=dims_value,
            num_heads=num_heads,
            seq_len=segment_len,
            state_len=state_len,
            attn_normalize=attn_normalize,
            dropout=dropout,
            attn_proj_rank=attn_proj_rank,
            cope=cope,
            diff_attn=diff_attn,
            layer_num=layer_num,
            position_embedders=position_embedders,
        )
        
        # Projection for next state
        self.proj_out_state = nn.Linear(num_heads * attn_proj_rank, dim_input, bias=False)
        torch.nn.init.normal_(self.proj_out_state.weight, mean=0.0, std=(1. / (2 * self.num_layers) ** 0.5))
        
        # Projection for output
        self.proj_out = nn.Linear(num_heads * attn_proj_rank, dim_input, bias=False)
        torch.nn.init.normal_(self.proj_out.weight, mean=0.0, std=(1. / (2 * self.num_layers) ** 0.5))
        
        # Set learnable initial state
        if first_layer:
            self.init_state = torch.nn.Parameter(torch.randn(1, state_len, dim_input) / (2. / 5.) ** 0.5)
        else:
            self.init_state = torch.nn.Parameter(torch.randn(1, state_len, dim_input))


    def forward(self, x: torch.Tensor, state: Optional[torch.Tuple], offset: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies recurrent stateful attention to the input tensor x.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, segment_len, dim_input).
            state (Optional[torch.Tensor]): State tensor of shape (batch_size, state_len, dim_input).
            offset (int): Offset for the position embeddings.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
              - Output tensor of shape (batch_size, segment_len, dim_input)
              - State tensor of shape (batch_size, state_len, dim_input)
        """
        if state is None:
            state = self.init_state
        
        if state.size(0) != x.size(0):
            state = state.repeat(x.size(0), 1, 1)
        
        # Prepend and append state to x_seg
        x = torch.cat([state, x, state], dim=1)
        
        # Apply attention
        x = self.attn(x, offset=offset - self.state_len)
        
        # Extract state from result
        _, att, state_end = extract_state(x, self.state_len)
        
        # Get next state
        state = self.proj_out_state(state_end)
        
        # Append output to buffer
        x = self.proj_out(att)

        return x, state

class StatefulCausalMHMA(nn.Module):
    """Implements a Stateful Causal Multi-Head Multi-Attention (MHMA) module."""
    def __init__(
        self,  
        dim_input: int, 
        dims_key: int, 
        dims_value: int, 
        num_heads: int,
        seq_len: int,
        state_len: int,
        attn_normalize: bool,
        dropout: float,
        attn_proj_rank: int,
        cope: bool,
        diff_attn: bool,
        layer_num: int,
        position_embedders: List[Optional[RoPEEmbeddings]],
    ):
        """Initializes the module

        Args:
            dim_input (int): The input dimension.
            dims_key (int): The key dimension.
            dims_value (int): The value dimension.
            num_heads (int): Number of attention heads.
            seq_len (int): The maximum length of the input sequence.
            state_len (int): The length of the state tensor.
            attn_normalize (bool): Whether to normalize the input to the attention projections.
            dropout (float): The dropout rate.
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
        self.attn_normalize = attn_normalize
        self.dropout = dropout
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
                    attn_normalize=attn_normalize,
                    dropout=dropout,
                    attn_proj_rank=attn_proj_rank,
                    cope=cope,
                    diff_attn=diff_attn,
                    layer_num=layer_num,
                    position_embedders=position_embedders,
                ) for _ in range(num_heads)
            ]
        )
        
    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Applies stateful causal multi-layer multi-head attention to the input tensor x.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len + 2 * state_len, dim_input).
            offset (int, optional): Offset for the position embeddings. Defaults to 0.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len + 2 * state_len, dims_value[0] * num_heads).
        """
        out_mult = 1. if not self.diff_attn else (1. - self.attn_heads[0].attn_modules[0].lambda_init)
        
        return out_mult * torch.concat(
            [
                attn_head(x, offset=offset) for attn_head in self.attn_heads
            ], 
            dim=-1
        )
    

class StatefulCausalMultiAttention(nn.Module):
    """Implements a Stateful Causal Multi-Attention module."""
    def __init__(
        self,  
        dim_input: int, 
        dims_key: int, 
        dims_value: int, 
        seq_len: int,
        state_len: int,
        attn_normalize: bool,
        dropout: float,
        attn_proj_rank: int,
        cope: bool,
        diff_attn: bool,
        layer_num: int,
        position_embedders: List[Optional[RoPEEmbeddings]],
    ):
        """Initializes the module

        Args:
            dim_input (int): The input dimension.
            dims_key (int): The key dimension.
            dims_value (int): The value dimension.
            seq_len (int): The maximum length of the sequence.
            state_len (int): The length of the state tensor.
            attn_normalize (bool): Whether to normalize the input to the attention projections.
            dropout (float): Dropout rate.
            attn_proj_rank (int): The rank of the attention projection.
            cope (bool): Whether to use CoPE.
            diff_attn (bool): Whether to use diff attention.
            layer_num (int): The position of the layer. Only used if diff_attn is True.
            position_embedder (Optional[RoPEEmbeddings]): The position embedder to use.
        """
        super(StatefulCausalMultiAttention, self).__init__()
        
        self.dim_input = dim_input
        self.dims_key = dims_key
        self.dims_value = dims_value
        self.seq_len = seq_len
        self.state_len = state_len
        self.attn_normalize = attn_normalize
        self.dropout = dropout
        self.attn_proj_rank = attn_proj_rank
        self.diff_attn = diff_attn
        self.layer_num = layer_num
        self.position_embedders = position_embedders
        
        if dims_value[-1] != attn_proj_rank:
            diff_attn_mult = 2 if diff_attn else 1
            self.proj_out = nn.Linear(diff_attn_mult * dims_value[-1], attn_proj_rank, bias=False)
            self.proj_out_state_start = nn.Linear(diff_attn_mult * dims_value[-1], attn_proj_rank, bias=False)
            self.proj_out_state_end = nn.Linear(diff_attn_mult * dims_value[-1], attn_proj_rank, bias=False)
            self.use_out_proj = True
        else:
            self.use_out_proj = False
        
        if diff_attn:
            attn_modules = [
                StatefulCausalDiffAttentionHead(
                    dim_input=dim_input,
                    dim_key=dims_key[0],
                    dim_value=dims_value[0],
                    seq_len=seq_len,
                    state_len=state_len,
                    layer_num=layer_num,
                    attn_normalize=attn_normalize,
                    dropout=dropout,
                    cope=cope,
                    position_embedder=position_embedders[0],
                )
            ]
            proj_modules = []
            proj_modules_state_start = []
            proj_modules_state_end = []
            for ix in range(1, len(dims_key)):
                attn_modules.append(
                    StatefulCausalDiffAttentionHead(
                        dim_input=dims_value[ix-1],
                        dim_key=dims_key[ix],
                        dim_value=dims_value[ix],
                        seq_len=seq_len,
                        state_len=state_len,
                        layer_num=layer_num,
                        attn_normalize=attn_normalize,
                        dropout=dropout,
                        cope=cope,
                        position_embedder=position_embedders[ix],
                    )
                )
                proj_modules.append(
                    nn.Linear(2 * dims_value[ix-1], dims_value[ix-1], bias=False)
                )
                proj_modules_state_start.append(
                    nn.Linear(2 * dims_value[ix-1], dims_value[ix-1], bias=False)
                )
                proj_modules_state_end.append(
                    nn.Linear(2 * dims_value[ix-1], dims_value[ix-1], bias=False)
                )
                
                self.proj_modules = nn.ModuleList(proj_modules)
                self.proj_modules_state_start = nn.ModuleList(proj_modules_state_start)
                self.proj_modules_state_end = nn.ModuleList(proj_modules_state_end)
        else:
            attn_modules = [
                StatefulCausalAttentionHead(
                    dim_input=dim_input,
                    dim_key=dims_key[0],
                    dim_value=dims_value[0],
                    seq_len=seq_len,
                    state_len=state_len,
                    attn_normalize=attn_normalize,
                    dropout=dropout,
                    cope=cope,
                    position_embedder=position_embedders[0],
                )
            ]
            for ix in range(1, len(dims_key)):
                attn_modules.append(
                    StatefulCausalAttentionHead(
                        dim_input=dims_value[ix-1],
                        dim_key=dims_key[ix],
                        dim_value=dims_value[ix],
                        seq_len=seq_len,
                        state_len=state_len,
                        attn_normalize=attn_normalize,
                        dropout=dropout,
                        cope=cope,
                        position_embedder=position_embedders[ix],
                    )
                )
                
        self.attn_modules = nn.ModuleList(attn_modules)
        
    def forward(self, x: torch.Tensor, offset: int) -> torch.Tensor:
        """
        Applies the StatefulCausalMultiAttention layer to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len + 2 * state_len, dim_in).
            offset (int): Offset for the position embeddings.

        Returns:
            Output tensor of shape (batch_size, seq_len + 2 * state_len, attn_proj_rank).
                
        """
        if self.diff_attn:
            for ix in range(len(self.attn_modules)):
                x = self.attn_modules[ix](x, offset=offset)
                if ix < len(self.attn_modules) - 1:
                    x_state_start, x, x_state_end = extract_state(x, self.state_len)
                    
                    x = self.proj_modules[ix](x)
                    x_state_start = self.proj_modules_state_start[ix](x_state_start)
                    x_state_end = self.proj_modules_state_end[ix](x_state_end)
                    
                    x = torch.concat([x_state_start, x, x_state_end], dim=1)
        else:
            for attn_module in self.attn_modules:
                x = attn_module(x, offset=offset)
        
        if self.use_out_proj:
            x_state_start, x, x_state_end = extract_state(x, self.state_len)
            
            x = self.proj_out(x)
            x_state_start = self.proj_out_state_start(x_state_start)
            x_state_end = self.proj_out_state_end(x_state_end)
            
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
        attn_normalize: bool = True,
        dropout: float = 0.0,
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
            attn_normalize (bool, optional): Whether to normalize input to the attention projections. Defaults to True.
            dropout (float, optional): The dropout rate. Defaults to 0.0.
            cope (bool, optional): Whether to use CoPE. Defaults to False.
            position_embedder (Optional[RoPEEmbeddings], optional): The position embedder to use. Defaults to None.
        """
        super(StatefulCausalAttentionHead, self).__init__()
        
        self.dim_input = dim_input
        self.dim_key = dim_key
        self.dim_value = dim_value
        self.seq_len = seq_len
        self.state_len = state_len
        self.attn_normalize = attn_normalize
        self.dropout = dropout
        self.cope = cope
        self.position_embedder = position_embedder
        
        # Projections from the attention layer to the next attention layer
        self.proj_k = nn.Linear(dim_input, dim_key, bias=False)
        self.proj_q = nn.Linear(dim_input, dim_key, bias=False)
        self.proj_v = nn.Linear(dim_input, dim_value, bias=False)
        
        # If normalize is True, define qkv normalizations
        if self.attn_normalize:
            self.norm_in = nn.LayerNorm(self.dim_input, eps=1e-5)
            self.norm_in_state_start = nn.LayerNorm(self.dim_input, eps=1e-5)
            self.norm_in_state_end = nn.LayerNorm(self.dim_input, eps=1e-5)
        
        # State projections from attention layer to the next attention layer
        self.proj_k_state_start = nn.Linear(dim_input, dim_key, bias=False)
        self.proj_q_state_start = nn.Linear(dim_input, dim_key, bias=False)
        self.proj_v_state_start = nn.Linear(dim_input, dim_value, bias=False)
        self.proj_k_state_end = nn.Linear(dim_input, dim_key, bias=False)
        self.proj_q_state_end = nn.Linear(dim_input, dim_key, bias=False)
        self.proj_v_state_end = nn.Linear(dim_input, dim_value, bias=False)
        
        if cope:
            self.cope_emb = nn.Parameter(
                torch.randn(1, self.dim_key, self.state_len)
            )
    
    def apply_attention(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        offset: Optional[int] = None,
        bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Applies attention to the input tensors.

        Args:
            q (torch.Tensor): Query tensor of shape (batch_size, seq_len + 2 * state_len, dim_key).
            k (torch.Tensor): Key tensor of shape (batch_size, seq_len + 2 * state_len, dim_key).
            v (torch.Tensor): Value tensor of shape (batch_size, seq_len + 2 * state_len, dim_value).
            offset (Optional[int]): Optional offset to apply to the position embeddings.
            bias (Optional[torch.Tensor]): Attention bias vector of shape (batch_size, seq_len, seq_len).
            
        Returns:
            torch.Tensor: Output tensors of shape (batch_size, seq_len + 2 * state_len, dim_value).
        """
        # If position embedder is specified, add positional embeddings to q and k
        if self.position_embedder is not None:
            k = self.position_embedder(k, offset=offset)
            q = self.position_embedder(q, offset=offset)
        
        # If bias is specified, apply it to the attention for non-state tokens
        if bias is None:
            attn_bias = LowerTriangularMask()
        else:
            device = k.device
            attn_bias = torch.tril(
                torch.ones((k.size(0), k.size(1), k.size(1)), device=device, dtype=k.dtype), 
                diagonal=0,
            )
            attn_bias = attn_bias.log()
            attn_bias[:, self.state_len:-self.state_len, self.state_len:-self.state_len] += bias.to(dtype=k.dtype)
            
        att = memory_efficient_attention(q, k, v, attn_bias=attn_bias, p=self.dropout)

        return att

    def forward(self, x: torch.Tensor, offset: int) -> torch.Tensor:
        """
        Forward pass. Applies StatefulCausalAttention to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len + 2 * state_len, dim_in).
            offset (int): Offset for the position embeddings.

        Returns:
            Output tensor of shape (batch_size, seq_len + 2 * state_len, dim_value).
                
        """
        x_state_start, x, x_state_end = extract_state(x, self.state_len)
        
        if self.attn_normalize:
            x = self.norm_in(x)
            x_state_start = self.norm_in_state_start(x_state_start)
            x_state_end = self.norm_in_state_end(x_state_end)
        
        k = self.proj_k(x)
        q = self.proj_q(x)
        v = self.proj_v(x)
        
        k_state_start = self.proj_k_state_start(x_state_start)
        q_state_start = self.proj_q_state_start(x_state_start)
        v_state_start = self.proj_v_state_start(x_state_start)
        
        k_state_end = self.proj_k_state_end(x_state_end)
        q_state_end = self.proj_q_state_end(x_state_end)
        v_state_end = self.proj_v_state_end(x_state_end)
        
        self.k = k
        self.k_state_start = k_state_start
        self.k_state_end = k_state_end
        
        self.v = v
        self.v_state_start = v_state_start
        self.v_state_end = v_state_end
        
        if self.cope:
            logits = q @ k.transpose(-2, -1)
            gates = torch.sigmoid(logits)
            pos = gates.flip(-1).cumsum(dim=-1).flip(-1)
            pos = pos.clamp(max=self.state_len - 1)
            
            pos_ceil = pos.ceil().long()
            pos_floor = pos.floor().long()
            
            logits_int = q @ self.cope_emb
            logits_ceil = logits_int.gather(-1, pos_ceil)
            logits_floor = logits_int.gather(-1, pos_floor)
            
            w = pos - pos_floor
            
            bias = logits_ceil * w + logits_floor * (1 - w)
        else:
            bias = None
        
        k = torch.cat([k_state_start, k, k_state_end], dim=1)
        q = torch.cat([q_state_start, q, q_state_end], dim=1)
        v = torch.cat([v_state_start, v, v_state_end], dim=1)
        
        att = self.apply_attention(q, k, v, offset=offset, bias=bias)
        
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
        attn_normalize: bool = True,
        dropout: float = 0.0,
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
            attn_normalize (bool, optional): Whether to normalize input to the attention projections. Defaults to True.
            dropout (float, optional): The dropout rate. Defaults to 0.0.
            position_embedder (Optional[RoPEEmbeddings], optional): The position embedder to use. Defaults to None.
        """
        super(StatefulCausalDiffAttentionHead, self).__init__()
        
        self.dim_input = dim_input
        self.dim_key = dim_key
        self.dim_value = dim_value
        self.seq_len = seq_len
        self.state_len = state_len
        self.layer_num = layer_num
        self.attn_normalize = attn_normalize
        self.dropout = dropout
        self.cope = cope
        self.position_embedder = position_embedder
        
        # Calculate initial lambda
        self.lambda_init = 0.8 - 0.6 * np.exp(-0.3 * layer_num)
        
        # Initialize lambda params
        self.lambda_q1 = nn.Parameter(torch.randn((dim_key, 1)))
        self.lambda_q2 = nn.Parameter(torch.randn((dim_key, 1)))
        self.lambda_k1 = nn.Parameter(torch.randn((dim_key, 1)))
        self.lambda_k2 = nn.Parameter(torch.randn((dim_key, 1)))
        
        # Projections from the attention layer to the next attention layer
        self.proj_k = nn.Linear(dim_input, 2 * dim_key, bias=False)
        self.proj_q = nn.Linear(dim_input, 2  * dim_key, bias=False)
        self.proj_v = nn.Linear(dim_input, 2 * dim_value, bias=False)
        
        # If normalize is True, define qkv normalizations
        if self.attn_normalize:
            self.norm_in = nn.LayerNorm(self.dim_input, eps=1e-5)
            self.norm_in_state_start = nn.LayerNorm(self.dim_input, eps=1e-5)
            self.norm_in_state_end = nn.LayerNorm(self.dim_input, eps=1e-5)
        
        # State projections from attention layer to the next attention layer
        self.proj_k_state_start = nn.Linear(dim_input, 2 * dim_key, bias=False)
        self.proj_q_state_start = nn.Linear(dim_input, 2 * dim_key, bias=False)
        self.proj_v_state_start = nn.Linear(dim_input, 2 * dim_value, bias=False)
        self.proj_k_state_end = nn.Linear(dim_input, 2 * dim_key, bias=False)
        self.proj_q_state_end = nn.Linear(dim_input, 2 * dim_key, bias=False)
        self.proj_v_state_end = nn.Linear(dim_input, 2 * dim_value, bias=False)
        
        if cope:
            self.cope_emb_1 = nn.Parameter(
                torch.randn(1, self.dim_key, self.state_len)
            )
            self.cope_emb_2 = nn.Parameter(
                torch.randn(1, self.dim_key, self.state_len)
            )
            
        self.out_norm = nn.LayerNorm(2 * dim_value, eps=1e-5)
    
    def apply_attention(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        offset: Optional[int] = None,
        bias: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]] = (None, None)
    ) -> torch.Tensor:
        """
        Applies attention to the input tensors.

        Args:
            q (torch.Tensor): Query tensor of shape (batch_size, seq_len + 2 * state_len, 2 * dim_key).
            k (torch.Tensor): Key tensor of shape (batch_size, seq_len + 2 * state_len, 2 * dim_key).
            v (torch.Tensor): Value tensor of shape (batch_size, seq_len + 2 * state_len, 2 * dim_value).
            offset (Optional[int]): Optional offset to apply to the position embeddings.
            bias (Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]): Attention bias tensors of shape (batch_size, seq_len, seq_len).
            
        Returns:
            torch.Tensor: Output tensors of shape (batch_size, seq_len + 2 * state_len, 2 * dim_value).
        """
        # Split q and k into q1, q2, k1, k2
        q1, q2 = split_last_dim(q)
        k1, k2 = split_last_dim(k)
        
        # Split biases
        bias1, bias2 = bias
        
        # If position embedder is specified, add positional embeddings to q and k
        if self.position_embedder is not None:
            q1 = self.position_embedder(q1, offset=offset)
            q2 = self.position_embedder(q2, offset=offset)
            k1 = self.position_embedder(k1, offset=offset)
            k2 = self.position_embedder(k2, offset=offset)
        
        # If bias is specified, apply it to the attention for non-state tokens
        if bias1 is None:
            attn_bias_1 = torch.triu(torch.full((self.seq_len, self.seq_len), fill_value=float("-inf")), diagonal=1)
            attn_bias_2 = torch.triu(torch.full((self.seq_len, self.seq_len), fill_value=float("-inf")), diagonal=1)
        else:
            device = k.device
            attn_bias_1 = torch.tril(
                torch.ones((k.size(0), k.size(1), k.size(1)), device=device, dtype=k.dtype), 
                diagonal=0,
            )
            attn_bias_1 = attn_bias_1.log()
            attn_bias_1[:, self.state_len:-self.state_len, self.state_len:-self.state_len] += bias1.to(dtype=k.dtype)
            
            attn_bias_2 = torch.tril(
                torch.ones((k.size(0), k.size(1), k.size(1)), device=device, dtype=k.dtype), 
                diagonal=0,
            )
            attn_bias_2 = attn_bias_2.log()
            attn_bias_2[:, self.state_len:-self.state_len, self.state_len:-self.state_len] += bias2.to(dtype=k.dtype)
        
        lambda_ = torch.exp(self.lambda_q1.transpose(0, 1) @ self.lambda_k1) - torch.exp(self.lambda_q2.transpose(0, 1) @ self.lambda_k2) + self.lambda_init
        
        att1 = torch.nn.functional.softmax(q1 @ k1.transpose(-2, -1) / np.sqrt(self.dim_key) + attn_bias_1, dim=-1)
        att2 = lambda_ * torch.nn.functional.softmax(q2 @ k2.transpose(-2, -1) / np.sqrt(self.dim_key) + attn_bias_2, dim=-1)
        
        att = torch.nn.functional.dropout(att1 - att2, p=self.dropout) @ v

        return att

    def forward(self, x: torch.Tensor, offset: int) -> torch.Tensor:
        """
        Forward pass. Applies StatefulCausalAttention to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len + 2 * state_len, dim_in).
            offset (int): Offset for the position embeddings.

        Returns:
            Output tensor of shape (batch_size, seq_len + 2 * state_len, 2 * dim_value).
                
        """
        x_state_start, x, x_state_end = extract_state(x, self.state_len)
        
        if self.attn_normalize:
            x = self.norm_in(x)
            x_state_start = self.norm_in_state_start(x_state_start)
            x_state_end = self.norm_in_state_end(x_state_end)
        
        k = self.proj_k(x)
        q = self.proj_q(x)
        v = self.proj_v(x)
        
        k_state_start = self.proj_k_state_start(x_state_start)
        q_state_start = self.proj_q_state_start(x_state_start)
        v_state_start = self.proj_v_state_start(x_state_start)
        
        k_state_end = self.proj_k_state_end(x_state_end)
        q_state_end = self.proj_q_state_end(x_state_end)
        v_state_end = self.proj_v_state_end(x_state_end)
        
        self.k = k
        self.k_state_start = k_state_start
        self.k_state_end = k_state_end
        
        self.v = v
        self.v_state_start = v_state_start
        self.v_state_end = v_state_end
        
        if self.cope:
            q1, q2 = split_last_dim(q)
            k1, k2 = split_last_dim(k)
            
            logits1 = q1 @ k1.transpose(-2, -1)
            logits2 = q2 @ k2.transpose(-2, -1)
            
            gates1 = torch.sigmoid(logits1)
            gates2 = torch.sigmoid(logits2)
            
            pos1 = gates1.flip(-1).cumsum(dim=-1).flip(-1)
            pos1 = pos1.clamp(max=self.state_len - 1)
            pos2 = gates2.flip(-1).cumsum(dim=-1).flip(-1)
            pos2 = pos2.clamp(max=self.state_len - 1)
            
            pos_ceil_1 = pos1.ceil().long()
            pos_floor_1 = pos1.floor().long()
            pos_ceil_2 = pos2.ceil().long()
            pos_floor_2 = pos2.floor().long()
            
            logits_int_1 = q1 @ self.cope_emb_1
            logits_ceil_1 = logits_int_1.gather(-1, pos_ceil_1)
            logits_floor_1 = logits_int_1.gather(-1, pos_floor_1)
            logits_int_2 = q2 @ self.cope_emb_2
            logits_ceil_2 = logits_int_2.gather(-1, pos_ceil_2)
            logits_floor_2 = logits_int_2.gather(-1, pos_floor_2)
            
            w1 = pos1 - pos_floor_1
            w2 = pos2 - pos_floor_2
            
            bias1 = logits_ceil_1 * w1 + logits_floor_1 * (1 - w1)
            bias2 = logits_ceil_2 * w2 + logits_floor_2 * (1 - w2)
        else:
            bias1 = None
            bias2 = None
        
        k = torch.cat([k_state_start, k, k_state_end], dim=1)
        q = torch.cat([q_state_start, q, q_state_end], dim=1)
        v = torch.cat([v_state_start, v, v_state_end], dim=1)
        
        att = self.apply_attention(q, k, v, offset=offset, bias=(bias1, bias2))
        
        return self.out_norm(att)