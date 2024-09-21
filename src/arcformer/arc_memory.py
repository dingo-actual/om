from typing import List, Optional, Tuple

import torch
from torch import nn
from xformers.ops import memory_efficient_attention, LowerTriangularMask

from .positional_embeddings import RoPEEmbeddings
from .util import extract_state


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
        num_layers: int,
        cope: bool,
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
            num_layers (int): Number of ARC transformer layers in the parent model.
            cope (bool): Whether to use CoPE.
            position_embedders (List[Optional[RoPEEmbeddings]]): Position embedding modules.
        """
        super(ARC, self).__init__()

        # Record input parameters
        self.num_heads = num_heads
        self.segment_len = segment_len
        self.state_len = state_len
        self.attn_normalize = attn_normalize
        self.num_layers = num_layers

        self.dim_input = dim_input
        self.dims_key = dims_key
        self.dims_value = dims_value
        
        # Set learnable initial state
        self.init_state = nn.Parameter(torch.randn(1, state_len, dim_input) / (2. / 5.) ** 0.5)
        
        # Build attention modules
        self.attn = StatefulCausalMHMA(
            dim_input=dim_input,
            dims_key=dims_key,
            dims_value=dims_value,
            num_heads=num_heads,
            seq_len=segment_len,
            state_len=state_len,
            attn_normalize=attn_normalize,
            cope=cope,
            position_embedders=position_embedders,
        )
        
        # Projection for next state
        self.proj_out_state = nn.Linear(num_heads * dims_value[0], dim_input, bias=False)
        torch.nn.init.normal_(self.proj_out_state.weight, mean=0.0, std=(1. / (2 * self.num_layers) ** 0.5))
        
        # Projection for output
        self.proj_out = nn.Linear(num_heads * dims_value[0], dim_input, bias=False)
        torch.nn.init.normal_(self.proj_out.weight, mean=0.0, std=(1. / (2 * self.num_layers) ** 0.5))


    def forward(self, x: torch.Tensor, state: torch.Tuple, offset: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies recurrent stateful attention to the input tensor x.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, segment_len, dim_input).
            state (torch.Tensor): State tensor of shape (batch_size, state_len, dim_input).
            offset (int): Offset for the position embeddings.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
              - Output tensor of shape (batch_size, segment_len, dim_input)
              - State tensor of shape (batch_size, state_len, dim_input)
        """
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
        cope: bool,
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
            cope (bool): Whether to use CoPE.
            position_embedders (List[Optional[RoPEEmbeddings]]): The position embedder to use.
        """
        super(StatefulCausalMHMA, self).__init__()
        
        self.dim_input = dim_input
        self.dims_key = dims_key
        self.dims_value = dims_value
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.state_len = state_len
        self.attn_normalize = attn_normalize
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
                    cope=cope,
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
        return torch.concat(
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
        cope: bool,
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
            cope (bool): Whether to use CoPE.
            position_embedder (Optional[RoPEEmbeddings]): The position embedder to use.
        """
        super(StatefulCausalMultiAttention, self).__init__()
        
        self.dim_input = dim_input
        self.dims_key = dims_key
        self.dims_value = dims_value
        self.seq_len = seq_len
        self.state_len = state_len
        self.attn_normalize = attn_normalize
        self.position_embedders = position_embedders
        
        if len(dims_value) > 1 and dims_value[0] != dims_value[-1]:
            self.proj_out = nn.Linear(dims_value[-1], dims_value[0], bias=False)
            self.proj_out_state_start = nn.Linear(dims_value[-1], dims_value[0], bias=False)
            self.proj_out_state_end = nn.Linear(dims_value[-1], dims_value[0], bias=False)
            self.use_out_proj = True
        else:
            self.use_out_proj = False
        
        attn_modules = [
            StatefulCausalAttentionHead(
                dim_input=dim_input,
                dim_key=dims_key[0],
                dim_value=dims_value[0],
                seq_len=seq_len,
                state_len=state_len,
                attn_normalize=attn_normalize,
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
            Output tensor of shape (batch_size, seq_len + 2 * state_len, dim_value[0]).
                
        """
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
        cope: bool = False,
        position_embedder: Optional[RoPEEmbeddings] = None,
    ):
        """Initializes the module

        Args:
            dim_input (int): The input dimension.
            dim_key (int): The key dimension.
            dim_value (int): The value dimension.
            state_len (int): The length of the state tensor.
            normalize (bool, optional): Whether to normalize input to the attention projections. Defaults to True.
            position_embedder (Optional[RoPEEmbeddings], optional): The position embedder to use. Defaults to None.
        """
        super(StatefulCausalAttentionHead, self).__init__()
        
        self.dim_input = dim_input
        self.dim_key = dim_key
        self.dim_value = dim_value
        self.seq_len = seq_len
        self.state_len = state_len
        self.attn_normalize = attn_normalize
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
            
        att = memory_efficient_attention(q, k, v, attn_bias=attn_bias)

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