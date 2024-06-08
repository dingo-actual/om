from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from xformers.ops import memory_efficient_attention, LowerTriangularMask

from .positional_embeddings import RoPEEmbeddings
from .util import extract_state

#TODO: add depth-aware weight initialization

class ARC(nn.Module):
    """Implements ARC Transformer memory module."""

    def __init__(
        self, 
        dim_input: int, 
        dims_key: List[int], 
        dims_value: List[int], 
        iters: List[int],
        num_heads: int, 
        segment_len: int, 
        state_len: int,
        normalize: bool,
        position_embedders: List[Optional[RoPEEmbeddings]]
    ):
        """Initialize module.

        Args:
            dim_input (int): Input dimension.
            dims_key (List[int]): Key dimensions.
            dims_value (List[int]): Value dimensions.
            iters (List[int]): Number of iterations for each memory module.
            num_heads (int): Number of attention heads.
            segment_len (int): Segment length (must be a factor of the input sequence length).
            state_len (int): Length of the state (i.e., number of tokens).
            normalize (bool): Whether to normalize the attention inputs.
            position_embedders (List[Optional[RoPEEmbeddings]]): Position embedding modules.
        """
        super(ARC, self).__init__()

        # Record input parameters
        self.num_heads = num_heads
        self.segment_len = segment_len
        self.state_len = state_len
        self.normalize = normalize

        self.dim_input = dim_input
        self.dims_key = dims_key
        self.dims_value = dims_value
        
        self.iters = iters
        
        # Set learnable initial state
        self.init_state = nn.Parameter(torch.randn(1, state_len, dim_input) / (2.0 / (5 * dim_input)) ** 0.5)
        
        # Build attention modules
        self.attn = StatefulCausalMMHA(
            dim_input=dim_input,
            dims_key=dims_key,
            dims_value=dims_value,
            num_heads=num_heads,
            state_len=state_len,
            normalize=normalize,
            position_embedders=position_embedders,
            iters=iters,
        )
        
        # Projection for next state
        self.proj_out_state = nn.Linear(num_heads * dims_value[-1], dim_input, bias=False)
        
        # Projection for output
        self.proj_out = nn.Linear(num_heads * dims_value[-1], dim_input, bias=False)


    def forward(self, x: torch.Tensor, state: torch.Tuple, offset: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies recurrent dual-attention to the input tensor x.

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
        _, att_end, state_end = extract_state(x, self.state_len)
        
        # Get next state
        state = self.proj_out_state(state_end)
        
        # Append output to buffer
        x = self.proj_out(att_end)

        return x, state

class StatefulCausalMMHA(nn.Module):
    def __init__(
        self,  
        dim_input: int, 
        dims_key: int, 
        dims_value: int, 
        num_heads: int,
        state_len: int,
        normalize: bool,
        position_embedders: List[Optional[RoPEEmbeddings]],
        iters: List[int]
    ):
        """Initializes the module

        Args:
            dim_input (int): The input dimension.
            dims_key (int): The key dimension.
            dims_value (int): The value dimension.
            num_heads (int): Number of attention heads.
            state_len (int): The length of the state tensor.
            normalize (bool): Whether to normalize the input to the attention projections.
            position_embedders (List[Optional[RoPEEmbeddings]]): The position embedder to use.
            iters (List[int]): The number of attention iterations to perform.
        """
        super(StatefulCausalMMHA, self).__init__()
        
        self.dim_input = dim_input
        self.dims_key = dims_key
        self.dims_value = dims_value
        self.num_heads = num_heads
        self.state_len = state_len
        self.normalize = normalize
        self.position_embedders = position_embedders
        self.iters = iters
        
        self.attn_heads = nn.ModuleList(
            [
                StatefulCausalMultiAttention(
                    dim_input=dim_input,
                    dims_key=dims_key,
                    dims_value=dims_value,
                    state_len=state_len,
                    normalize=normalize,
                    position_embedders=position_embedders,
                    iters=iters
                ) for _ in range(num_heads)
            ]
        )
        
    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Applies stateful causal multi-layer multi-head attention to the input tensor x.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len + 2 * state_len, dim_input).
            offset (int, optional): Offset for the position embeddings. Defaults to 0.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len + 2 * state_len, dim_input * num_heads).
        """
        return torch.concat(
            [
                attn_head(x, offset=offset) for attn_head in self.attn_heads
            ], 
            dim=-1
        )
    

class StatefulCausalMultiAttention(nn.Module):
    def __init__(
        self,  
        dim_input: int, 
        dims_key: int, 
        dims_value: int, 
        state_len: int,
        normalize: bool,
        position_embedders: List[Optional[RoPEEmbeddings]],
        iters: List[int]
    ):
        """Initializes the module

        Args:
            dim_input (int): The input dimension.
            dims_key (int): The key dimension.
            dims_value (int): The value dimension.
            state_len (int): The length of the state tensor.
            normalize (bool): Whether to normalize the input to the attention projections.
            position_embedder (Optional[RoPEEmbeddings]): The position embedder to use.
            iters (List[int]): The number of attention iterations to perform.
        """
        super(StatefulCausalMultiAttention, self).__init__()
        
        self.dim_input = dim_input
        self.dims_key = dims_key
        self.dims_value = dims_value
        self.state_len = state_len
        self.normalize = normalize
        self.position_embedders = position_embedders
        self.iters = iters
        
        attn_modules = [
            StatefulCausalAttentionHead(
                dim_input=dim_input,
                dim_key=dims_key[0],
                dim_value=dims_value[0],
                state_len=state_len,
                normalize=normalize,
                position_embedder=position_embedders[0],
                iters=iters[0]
            )
        ]
        for ix in range(1, len(dims_key)):
            attn_modules.append(
                StatefulCausalAttentionHead(
                    dim_input=dims_value[ix-1],
                    dim_key=dims_key[ix],
                    dim_value=dims_value[ix],
                    state_len=state_len,
                    normalize=normalize,
                    position_embedder=position_embedders[ix],
                    iters=iters[ix]
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
            Output tensor of shape (batch_size, seq_len + 2 * state_len, dim_value[-1]).
                
        """
        for attn_module in self.attn_modules:
            x = attn_module(x, offset=offset)
            
        return x
    
class StatefulCausalAttentionHead(nn.Module):
    def __init__(
        self,  
        dim_input: int, 
        dim_key: int, 
        dim_value: int, 
        state_len: int,
        normalize: bool = True,
        position_embedder: Optional[RoPEEmbeddings] = None,
        iters: int = 1
    ):
        """Initializes the module

        Args:
            dim_input (int): The input dimension.
            dim_key (int): The key dimension.
            dim_value (int): The value dimension.
            state_len (int): The length of the state tensor.
            normalize (bool, optional): Whether to normalize input to the attention projections. Defaults to True.
            position_embedder (Optional[RoPEEmbeddings], optional): The position embedder to use. Defaults to None.
            iters (int, optional): The number of attention iterations to perform. Defaults to 1.
        """
        super(StatefulCausalAttentionHead, self).__init__()
        
        self.dim_input = dim_input
        self.dim_key = dim_key
        self.dim_value = dim_value
        self.state_len = state_len
        self.normalize = normalize
        self.position_embedder = position_embedder
        self.iters = iters
        
        # Projections from the attention layer to the next attention layer
        self.proj_k = nn.Linear(dim_input, dim_key, bias=False)
        self.proj_q = nn.Linear(dim_input, dim_key, bias=False)
        self.proj_v = nn.Linear(dim_input, dim_value, bias=False)
        
        # If normalize is True, define qkv normalizations
        if self.normalize:
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
        
        if iters > 1:
            self.proj_inv = nn.Linear(dim_value, dim_input, bias=False)
            self.proj_inv_state_begin = nn.Linear(dim_value, dim_input, bias=False)
            self.proj_inv_state_end = nn.Linear(dim_value, dim_input, bias=False)
    
    def apply_attention(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        offset: Optional[int] = None
    ) -> torch.Tensor:
        """
        Applies attention to the input tensors.

        Args:
            q (torch.Tensor): Query tensor of shape (batch_size, seq_len + 2 * state_len, dim_key).
            k (torch.Tensor): Key tensor of shape (batch_size, seq_len + 2 * state_len, dim_key).
            v (torch.Tensor): Value tensor of shape (batch_size, seq_len + 2 * state_len, dim_value).
            offset (Optional[int]): Optional offset to apply to the position embeddings.
            
        Returns:
            torch.Tensor: Output tensors of shape (batch_size, seq_len + 2 * state_len, dim_value).
        """
        # If position embedder is specified, add positional embeddings to q and k
        if self.position_embedder is not None:
            k = self.position_embedder(k, offset=offset)
            q = self.position_embedder(q, offset=offset)
        
        #device = k.device
        
        #mask = torch.tril(torch.ones((k.size(1), k.size(1)), device=device), diagonal=0)
        #att = torch.nn.functional.scaled_dot_product_attention(q, k, v, mask)
        
        att = memory_efficient_attention(q, k, v, attn_bias=LowerTriangularMask())

        return att
    
    def forward(self, x: torch.Tensor, offset: int) -> torch.Tensor:
        """
        Applies the StatefulCausalAttention layer to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len + 2 * state_len, dim_in).
            offset (int): Offset for the position embeddings.

        Returns:
            Output tensor of shape (batch_size, seq_len + 2 * state_len, dim_value).
                
        """
        if self.iters > 1:
            for _ in range(self.iters - 1):
                x = self.forward_(x, offset)
                x_state_start, x, x_state_end = extract_state(x, self.state_len)
                x = self.proj_inv(x)
                x_state_start = self.proj_inv_state_begin(x_state_start)
                x_state_end = self.proj_inv_state_begin(x_state_end)
                x = torch.cat([x_state_start, x, x_state_end], dim=1) 
            x = self.forward_(x, offset)
        else:
            x = self.forward_(x, offset)
            
        return x
    
    def forward_(self, x: torch.Tensor, offset: int) -> torch.Tensor:
        """
        Applies StatefulCausalAttention to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len + 2 * state_len, dim_in).
            offset (int): Offset for the position embeddings.

        Returns:
            Output tensor of shape (batch_size, seq_len + 2 * state_len, dim_value).
                
        """
        x_state_start, x, x_state_end = extract_state(x, self.state_len)
        
        if self.normalize:
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
        
        k = torch.cat([k_state_start, k, k_state_end], dim=1)
        q = torch.cat([q_state_start, q, q_state_end], dim=1)
        v = torch.cat([v_state_start, v, v_state_end], dim=1)
        
        att = self.apply_attention(q, k, v, offset=offset)
        
        return att