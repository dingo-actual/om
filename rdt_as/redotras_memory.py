from typing import List, Optional, Tuple

import torch
from torch import nn

from .positional_embeddings import RoPEEmbeddings
from .util import extract_state, StackedLinear

class ReDoTrAS(nn.Module):
    """Implements Recurrent Double Transformer with Attentive State (ReDoTrAS) memory module."""

    def __init__(
        self, 
        dim_input: int, 
        dims_key: List[int], 
        dims_value: List[int], 
        num_heads: int, 
        segment_len: int, 
        state_len: int,
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
            position_embedders (List[Optional[RoPEEmbeddings]]): Position embedding modules.
        """
        super(ReDoTrAS, self).__init__()

        # Record input parameters
        self.num_heads = num_heads
        self.segment_len = segment_len
        self.state_len = state_len

        self.dim_input = dim_input
        self.dims_key = dims_key
        self.dims_value = dims_value
        
        # Build attention modules
        attn_modules = []
        dim_value_last = dim_input
        for ix, (dim_key, dim_value, position_embedder) in enumerate(zip(self.dims_key[:-1], self.dims_value[:-1], position_embedders[:-1])):
            attn_modules.append(
                CausalMHAWithState(
                    dim_in=dim_value_last,
                    dim_key=dim_key,
                    dim_value=dim_value,
                    num_heads_in=1 if ix == 0 else num_heads,
                    state_len=state_len,
                    position_embedder=position_embedder
                )
            )
            dim_value_last = dim_value
            
        self.attn_modules = nn.ModuleList(attn_modules)
        
        # Projection for next state
        self.proj_out_state = nn.Linear(num_heads * dim_value, dim_input, bias=False)
        
        # Projection for output
        self.proj_out = nn.Linear(num_heads * dim_value, dim_input, bias=False)


    def forward(self, x: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies recurrent dual-attention to the input tensor x.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim_input).
            state (torch.Tensor): Initial state tensor of shape (1, state_len, dim_input).
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensor of shape (batch_size, seq_len, dim_input) and terminal state tensor of shape (batch_size, state_len, dim_input).
        """
        batch_size, seq_len, _ = x.shape

        num_segments, rem = divmod(seq_len, self.segment_len)
        num_segments += 1 if rem > 0 else 0

        out = []
        
        x = x.unsqueeze(1)
        state = state.unsqueeze(1).repeat(batch_size, 1, 1, 1)
        
        for ix in range(num_segments):
            ix_lo = ix * self.segment_len
            ix_hi = min(ix_lo + self.segment_len, x.size(2))
            seg_len = ix_hi - ix_lo

            # Extract segment from x
            x_seg = x[:, :, ix_lo:ix_hi, :]
            
            # Apply attention passes
            for attn_module in self.attn_modules:
                x_seg = attn_module(x_seg, offset=ix_lo - self.state_len)
            
            # Extract state from result
            _, att_end, state_end = extract_state(x_seg, self.state_len)
            
            # Reshape before final projections
            att_end = out.reshape((batch_size, seg_len, -1))
            state_end = state_end.reshape((batch_size, self.state_len, -1))
            
            # Get next state
            state = self.proj_out_state(state_end).unsqueeze(1)
            
            # Append output to buffer
            out.append(self.proj_out(att_end))

        # Return concatenated full sequence from buffer
        out = torch.concat(out, dim=1)

        return out, state.squeeze(1)

class CausalMHAWithState(nn.Module):
    def __init__(
        self,  
        dim_in: int, 
        dim_key: int, 
        dim_value: int, 
        num_heads_in: int, 
        num_heads_out: int, 
        state_len: int,
        position_embedder: Optional[RoPEEmbeddings] = None
    ):
        self.dim_in = dim_in
        self.dim_key = dim_key
        self.dim_value = dim_value
        
        self.num_heads_in = num_heads_in
        self.num_heads_out = num_heads_out
        
        self.state_len = state_len
        
        self.position_embedder = position_embedder
        
        # Projections from the attention layer to the next attention layer
        self.proj_k = StackedLinear(dim_in, dim_key, num_heads_in, num_heads_out, bias=False)
        self.proj_q = StackedLinear(dim_in, dim_key, num_heads_in, num_heads_out, bias=False)
        self.proj_v = StackedLinear(dim_in, dim_value, num_heads_in, num_heads_out, bias=False)
        
        # State projections from attention layer to the next attention layer
        self.proj_k_state_start = StackedLinear(dim_in, dim_key, num_heads_in, num_heads_out, bias=False)
        self.proj_q_state_start = StackedLinear(dim_in, dim_key, num_heads_in, num_heads_out, bias=False)
        self.proj_v_state_start = StackedLinear(dim_in, dim_value, num_heads_in, num_heads_out, bias=False)
        self.proj_k_state_end = StackedLinear(dim_in, dim_key, num_heads_in, num_heads_out, bias=False)
        self.proj_q_state_end = StackedLinear(dim_in, dim_key, num_heads_in, num_heads_out, bias=False)
        self.proj_v_state_end = StackedLinear(dim_in, dim_value, num_heads_in, num_heads_out, bias=False)
    
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
            q (torch.Tensor): Query tensor of shape (batch_size, num_heads, seq_len, dim_key).
            k (torch.Tensor): Key tensor of shape (batch_size, num_heads, seq_len, dim_key).
            v (torch.Tensor): Value tensor of shape (batch_size, num_heads, seq_len, dim_value).
            offset (Optional[int]): Optional offset to apply to the position embeddings.
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_heads, seq_len, dim_value).
        """
        # If position embedder is specified, add positional embeddings to q and k
        if self.position_embedder is not None:
            k = self.position_embedder(k, offset=offset)
            q = self.position_embedder(q, offset=offset)
        
        # Calculate attention scores using the projected key, and query tensors
        scores = q @ k.transpose(-2, -1) / self.dim_key ** 0.5

        # Calculate and apply causal attention mask
        mask = torch.tril(torch.ones((k.size(2), k.size(2)), dtype=torch.bool), diagonal=0)
        mask = mask.unsqueeze(0).unsqueeze(0).repeat((k.size(0), self.num_heads, 1, 1))
        scores.masked_fill_(torch.logical_not(mask), float('-inf'))

        # Calculate SDP attention
        att = nn.functional.softmax(scores, dim=-1) @ v

        return att
    
    def forward(self, x: torch.Tensor, offset: int) -> torch.Tensor:
        """
        Applies the CausalMHAWithState layer to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_heads_in, seq_len, dim_in).
            offset (int): Offset for the position embeddings.

        Returns:
            Output tensor of shape (batch_size, num_heads_out, seq_len, dim_value).
                
        """
        x_state_start, x, x_state_end = extract_state(x, self.state_len)
        
        k = self.proj_k(x)
        q = self.proj_q(x)
        v = self.proj_v(x)
        
        k_state_start = self.proj_k_state_start(x_state_start)
        q_state_start = self.proj_q_state_start(x_state_start)
        v_state_start = self.proj_v_state_start(x_state_start)
        
        k_state_end = self.proj_k_state_end(x_state_end)
        q_state_end = self.proj_q_state_end(x_state_end)
        v_state_end = self.proj_v_state_end(x_state_end)
        
        k = torch.cat([k_state_start, k, k_state_end], dim=2)
        q = torch.cat([q_state_start, q, q_state_end], dim=2)
        v = torch.cat([v_state_start, v, v_state_end], dim=2)
        
        att = self.apply_attention(q, k, v, offset=offset)
        
        return att
