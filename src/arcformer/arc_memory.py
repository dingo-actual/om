from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from .positional_embeddings import RoPEEmbeddings
from .util import extract_state

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
        attn_modules = []
        dim_value_last = dim_input
        for dim_key, dim_value, n_iter, position_embedder in zip(self.dims_key, self.dims_value, self.iters, position_embedders):
            attn_modules.append(
                StatefulCausalMHA(
                    dim_in=dim_value_last,
                    dim_key=dim_key,
                    dim_value=dim_value,
                    iters=n_iter,
                    num_heads=num_heads,
                    state_len=state_len,
                    normalize=normalize,
                    position_embedder=position_embedder
                )
            )
            dim_value_last = dim_value
            
        self.attn_modules = nn.ModuleList(attn_modules)
        
        # Projection for next state
        self.proj_out_state = nn.Linear(num_heads * dims_value[-1], dim_input, bias=False)
        
        # Projection for output
        self.proj_out = nn.Linear(num_heads * dim_value, dim_input, bias=False)


    def forward(self, x: torch.Tensor, state: Optional[torch.Tuple] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies recurrent dual-attention to the input tensor x.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim_input).
            state (Optional[torch.Tensor], optional): Initial state tensor of shape (batch_size, state_len, dim_input). Default is None.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
              - Output tensor of shape (batch_size, seq_len, dim_input)
              - Terminal state tensor of shape (batch_size, state_len, dim_input)
        """
        batch_size, seq_len, _ = x.shape

        num_segments, rem = divmod(seq_len, self.segment_len)
        num_segments += 1 if rem > 0 else 0

        out = []
        
        if state is None:
            state = self.init_state.repeat(batch_size, 1, 1)
        
        for ix in range(num_segments):
            ix_lo = ix * self.segment_len
            ix_hi = min(ix_lo + self.segment_len, x.size(1))
            seg_len = ix_hi - ix_lo

            # Extract segment from x
            x_seg = x[:, ix_lo:ix_hi, :]
            
            # Prepend and append state to x_seg
            x_seg = torch.cat([state, x_seg, state], dim=1)
            
            # Apply attention passes
            for attn_module in self.attn_modules:
                x_seg = attn_module(x_seg, offset=ix_lo - self.state_len)
            
            # Extract state from result
            _, att_end, state_end = extract_state(x_seg, self.state_len)
            
            # Consolidated state and attention
            att_end = torch.concat(att_end, dim=-1)
            state_end = torch.concat(state_end, dim=-1)
            
            # Get next state
            state = self.proj_out_state(state_end)
            
            # Append output to buffer
            out.append(self.proj_out(att_end))

        # Return concatenated full sequence from buffer
        out = torch.concat(out, dim=1)

        return out, state

class StatefulCausalMHA(nn.Module):
    def __init__(
        self,  
        dim_in: int, 
        dim_key: int, 
        dim_value: int, 
        num_heads: int, 
        state_len: int,
        normalize: bool = False,
        position_embedder: Optional[RoPEEmbeddings] = None,
        iters: int = 1
    ):
        """Initializes the module

        Args:
            dim_in (int): The input dimension.
            dim_key (int): The key dimension.
            dim_value (int): The value dimension.
            num_heads (int): The number of attention heads.
            state_len (int): The length of the state tensor.
            normalize (bool, optional): Whether to normalize the attention weights. Defaults to False.
            position_embedder (Optional[RoPEEmbeddings], optional): The position embedder to use. Defaults to None.
            iters (int, optional): The number of attention iterations to perform. Defaults to 1.
        """
        super(StatefulCausalMHA, self).__init__()
        
        self.dim_in = dim_in
        self.dim_key = dim_key
        self.dim_value = dim_value
        self.num_heads = num_heads
        self.state_len = state_len
        self.normalize = normalize
        self.position_embedder = position_embedder
        self.iters = iters
        
        # Projections from the attention layer to the next attention layer
        self.proj_k = torch.nn.ModuleList([nn.Linear(dim_in, dim_key, bias=False) for _ in range(num_heads)])
        self.proj_q = torch.nn.ModuleList([nn.Linear(dim_in, dim_key, bias=False) for _ in range(num_heads)])
        self.proj_v = torch.nn.ModuleList([nn.Linear(dim_in, dim_value, bias=False) for _ in range(num_heads)])
        
        # If normalize is True, define qkv normalizations
        if self.normalize:
            self.norm_q = torch.nn.ModuleList([nn.LayerNorm(self.dim_key) for _ in range(num_heads)])
            self.norm_k = torch.nn.ModuleList([nn.LayerNorm(self.dim_key) for _ in range(num_heads)])
            self.norm_v = torch.nn.ModuleList([nn.LayerNorm(self.dim_value) for _ in range(num_heads)])
        
        # State projections from attention layer to the next attention layer
        self.proj_k_state_start = torch.nn.ModuleList([nn.Linear(dim_in, dim_key, bias=False) for _ in range(num_heads)])
        self.proj_q_state_start = torch.nn.ModuleList([nn.Linear(dim_in, dim_key, bias=False) for _ in range(num_heads)])
        self.proj_v_state_start = torch.nn.ModuleList([nn.Linear(dim_in, dim_value, bias=False) for _ in range(num_heads)])
        self.proj_k_state_end = torch.nn.ModuleList([nn.Linear(dim_in, dim_key, bias=False) for _ in range(num_heads)])
        self.proj_q_state_end = torch.nn.ModuleList([nn.Linear(dim_in, dim_key, bias=False) for _ in range(num_heads)])
        self.proj_v_state_end = torch.nn.ModuleList([nn.Linear(dim_in, dim_value, bias=False) for _ in range(num_heads)])
        
        # If normalize is True, define qkv normalization for state
        if self.normalize:
            self.norm_k_state_start = torch.nn.ModuleList([nn.LayerNorm(self.dim_key) for _ in range(num_heads)])
            self.norm_q_state_start = torch.nn.ModuleList([nn.LayerNorm(self.dim_key) for _ in range(num_heads)])
            self.norm_v_state_start = torch.nn.ModuleList([nn.LayerNorm(self.dim_value) for _ in range(num_heads)])
            self.norm_k_state_end = torch.nn.ModuleList([nn.LayerNorm(self.dim_key) for _ in range(num_heads)])
            self.norm_q_state_end = torch.nn.ModuleList([nn.LayerNorm(self.dim_key) for _ in range(num_heads)])
            self.norm_v_state_end = torch.nn.ModuleList([nn.LayerNorm(self.dim_value) for _ in range(num_heads)])
            
        if iters > 1:
            self.proj_inv = torch.nn.ModuleList([nn.Linear(dim_value, dim_in, bias=False) for _ in range(num_heads)])
            self.proj_inv_state_begin = torch.nn.ModuleList([nn.Linear(dim_value, dim_in, bias=False) for _ in range(num_heads)])
            self.proj_inv_state_end = torch.nn.ModuleList([nn.Linear(dim_value, dim_in, bias=False) for _ in range(num_heads)])
    
    def apply_attention(
        self, 
        qs: List[torch.Tensor], 
        ks: List[torch.Tensor], 
        vs: List[torch.Tensor], 
        offset: Optional[int] = None
    ) -> List[torch.Tensor]:
        """
        Applies attention to the input tensors.

        Args:
            qs (List[torch.Tensor]): Query tensors, each of shape (batch_size, seq_len, dim_key).
            ks (List[torch.Tensor]): Key tensors, each of shape (batch_size, seq_len, dim_key).
            vs (List[torch.Tensor]): Value tensors, each of shape (batch_size, seq_len, dim_value).
            offset (Optional[int]): Optional offset to apply to the position embeddings.
            
        Returns:
            List[torch.Tensor]: Output tensors, each of shape (batch_size, seq_len, dim_value).
        """
        # If position embedder is specified, add positional embeddings to q and k
        if self.position_embedder is not None:
            ks = [self.position_embedder(k, offset=offset) for k in ks]
            qs = [self.position_embedder(q, offset=offset) for q in ks]
        
        device = qs[0].device
        
        mask = torch.tril(torch.ones((ks[0].size(1), ks[0].size(1)), device=device), diagonal=0)
        att = [torch.nn.functional.scaled_dot_product_attention(q, k, v, mask) for q, k, v in zip(qs, ks, vs)]

        return att
    
    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]], offset: int) -> List[torch.Tensor]:
        """
        Applies the StatefulCausalMHA layer to the input tensor.

        Args:
            x (torch.Tensor or List[torch.Tensor]): Input tensor or tensors of shape (batch_size, seq_len, dim_in).
            offset (int): Offset for the position embeddings.

        Returns:
            Output tensors, each of shape (batch_size, seq_len, dim_value).
                
        """
        if self.iters > 1:
            for _ in range(self.iters - 1):
                x = self.forward_(x, offset)
                x_state_start, x, x_state_end = extract_state(x, self.state_len)
                x = [proj_inv(x_) for proj_inv, x_ in zip(self.proj_inv, x)]
                x_state_start = [
                    proj_inv_state_begin(x_state_start_) 
                    for proj_inv_state_begin, x_state_start_ in zip(self.proj_inv_state_begin, x_state_start)
                ]
                x_state_end = [
                    proj_inv_state_begin(x_state_end_) 
                    for proj_inv_state_begin, x_state_end_ in zip(self.proj_inv_state_end, x_state_end)
                ]
                x = [
                    torch.cat([x_state_start_, x_, x_state_end_], dim=1) 
                    for x_state_start_, x_, x_state_end_ in zip(x_state_start, x, x_state_end)
                ]
            x = self.forward_(x, offset)
        else:
            x = self.forward_(x, offset)
            
        return x
    
    def forward_(self, x: Union[torch.Tensor, List[torch.Tensor]], offset: int) -> List[torch.Tensor]:
        """
        Applies StatefulCausalMHA to the input tensor.

        Args:
            x (torch.Tensor or List[torch.Tensor]): Input tensor or tensors of shape (batch_size, seq_len, dim_in).
            offset (int): Offset for the position embeddings.

        Returns:
            Output tensors, each of shape (batch_size, seq_len, dim_value).
                
        """
        x_state_start, x, x_state_end = extract_state(x, self.state_len)
        
        if isinstance(x, torch.Tensor):
            k = [proj_k(x) for proj_k in self.proj_k]
            q = [proj_q(x) for proj_q in self.proj_q]
            v = [proj_v(x) for proj_v in self.proj_v]
        else:
            k = [proj_k(x_) for proj_k, x_ in zip(self.proj_k, x)]
            q = [proj_q(x_) for proj_q, x_ in zip(self.proj_q, x)]
            v = [proj_v(x_) for proj_v, x_ in zip(self.proj_v, x)]
        
        if self.normalize:
            k = [norm_k(k_) for k_, norm_k in zip(k, self.norm_k)]
            q = [norm_q(q_) for q_, norm_q in zip(q, self.norm_q)]
            v = [norm_v(v_) for v_, norm_v in zip(v, self.norm_v)]
        
        if isinstance(x_state_start, torch.Tensor):
            k_state_start = [proj_k_state_start(x_state_start) for proj_k_state_start in self.proj_k_state_start]
            q_state_start = [proj_q_state_start(x_state_start) for proj_q_state_start in self.proj_q_state_start]
            v_state_start = [proj_v_state_start(x_state_start) for proj_v_state_start in self.proj_v_state_start]
        else:
            k_state_start = [proj_k_state_start(x_state_start_) for x_state_start_, proj_k_state_start in zip(x_state_start, self.proj_k_state_start)]
            q_state_start = [proj_q_state_start(x_state_start_) for x_state_start_, proj_q_state_start in zip(x_state_start, self.proj_q_state_start)]
            v_state_start = [proj_v_state_start(x_state_start_) for x_state_start_, proj_v_state_start in zip(x_state_start, self.proj_v_state_start)]
        
        if self.normalize:
            k_state_start = [norm_k_state_start(k_state_start_) for k_state_start_, norm_k_state_start in zip(k_state_start, self.norm_k_state_start)]
            q_state_start = [norm_q_state_start(q_state_start_) for q_state_start_, norm_q_state_start in zip(q_state_start, self.norm_q_state_start)]
            v_state_start = [norm_v_state_start(v_state_start_) for v_state_start_, norm_v_state_start in zip(v_state_start, self.norm_v_state_start)]
        
        if isinstance(x_state_end, torch.Tensor):
            k_state_end = [proj_k_state_end(x_state_end) for proj_k_state_end in self.proj_k_state_end]
            q_state_end = [proj_q_state_end(x_state_end) for proj_q_state_end in self.proj_q_state_end]
            v_state_end = [proj_v_state_end(x_state_end) for proj_v_state_end in self.proj_v_state_end]
        else:
            k_state_end = [proj_k_state_end(x_state_end_) for x_state_end_, proj_k_state_end in zip(x_state_end, self.proj_k_state_end)]
            q_state_end = [proj_q_state_end(x_state_end_) for x_state_end_, proj_q_state_end in zip(x_state_end, self.proj_q_state_end)]
            v_state_end = [proj_v_state_end(x_state_end_) for x_state_end_, proj_v_state_end in zip(x_state_end, self.proj_v_state_end)]
        
        if self.normalize:
            k_state_end = [norm_k_state_end(k_state_end_) for k_state_end_, norm_k_state_end in zip(k_state_end, self.norm_k_state_end)]
            q_state_end = [norm_q_state_end(q_state_end_) for q_state_end_, norm_q_state_end in zip(q_state_end, self.norm_q_state_end)]
            v_state_end = [norm_v_state_end(v_state_end_) for v_state_end_, norm_v_state_end in zip(v_state_end, self.norm_v_state_end)]
        
        k = [
            torch.cat([k_state_start_, k_, k_state_end_], dim=1)
            for k_state_start_, k_, k_state_end_ in zip(k_state_start, k, k_state_end)
        ]
        q = [
            torch.cat([q_state_start_, q_, q_state_end_], dim=1)
            for q_state_start_, q_, q_state_end_ in zip(q_state_start, q, q_state_end)
        ]
        v = [
            torch.cat([v_state_start_, v_, v_state_end_], dim=1)
            for v_state_start_, v_, v_state_end_ in zip(v_state_start, v, v_state_end)
        ]
        
        att = self.apply_attention(q, k, v, offset=offset)
        
        return att
