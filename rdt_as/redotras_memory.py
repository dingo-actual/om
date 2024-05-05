from typing import Optional, Tuple

import torch
from torch import nn

from .positional_embeddings import RoPEEmbeddings
from .util import StackedLinear

class ReDoTrAS(nn.Module):
    """Implements Recurrent Double Transformer with Attentive State (ReDoTrAS) memory module."""

    def __init__(
        self, 
        dim_input: int, 
        dim_key: int, 
        dim_value: int, 
        num_heads: int, 
        segment_len: int, 
        position_embedder_1: Optional[RoPEEmbeddings] = None,
        position_embedder_2: Optional[RoPEEmbeddings] = None
    ):
        """Initialize module.

        Args:
            dim_input (int): Input dimension.
            dim_key (int): Key dimension.
            dim_value (int): Value dimension.
            num_heads (int): Number of attention heads.
            segment_len (int): Segment length (must be a factor of the input sequence length).
            position_embedder_1 (Optional[RoPEEmbeddings], optional): Position embedding module for first memory layer. Defaults to None.
            position_embedder_2 (Optional[RoPEEmbeddings], optional): Position embedding module for second memory layer. Defaults to None.
        """
        super(ReDoTrAS, self).__init__()

        # Record input parameters
        self.num_heads = num_heads
        self.segment_len = segment_len

        self.dim_input = dim_input
        self.dim_key = dim_key
        self.dim_value = dim_value
        
        # Save position embedders (if given)
        self.position_embedder_1 = position_embedder_1
        self.position_embedder_2 = position_embedder_2
        
        # Projections from input to first attention layer
        self.proj_k = StackedLinear(dim_input, dim_key, 1, num_heads, bias=False)
        self.proj_v = StackedLinear(dim_input, dim_value, 1, num_heads, bias=False)
        self.proj_q = StackedLinear(dim_input, dim_key, 1, num_heads, bias=False)
        
        # Projections from first attention layer to the second attention layer
        self.proj_down_k = StackedLinear(dim_value, dim_key // 2, num_heads, num_heads, bias=False)
        self.proj_down_q = StackedLinear(dim_value, dim_key // 2, num_heads, num_heads, bias=False)
        self.proj_down_v = StackedLinear(dim_value, dim_value // 2, num_heads, num_heads, bias=False)
        
        # Projections from the second attention layer to the first attention layer
        self.proj_up_k = StackedLinear(dim_value // 2, dim_key, num_heads, num_heads, bias=False)
        self.proj_up_q = StackedLinear(dim_value // 2, dim_key, num_heads, num_heads, bias=False)
        self.proj_up_v = StackedLinear(dim_value // 2, dim_value, num_heads, num_heads, bias=False)

        # Projection for output
        self.proj_out = nn.Linear(num_heads * dim_value, dim_input, bias=False)
        
        # State projections from input to first attention layer
        self.proj_k_state = StackedLinear(dim_input, dim_key, 1, num_heads, bias=False)
        self.proj_v_state = StackedLinear(dim_input, dim_value, 1, num_heads, bias=False)
        self.proj_q_state = StackedLinear(dim_input, dim_key, 1, num_heads, bias=False)
        
        # State projections from first attention layer to the second attention layer
        self.proj_down_k_state_start = StackedLinear(dim_value, dim_key // 2, num_heads, num_heads, bias=False)
        self.proj_down_q_state_start = StackedLinear(dim_value, dim_key // 2, num_heads, num_heads, bias=False)
        self.proj_down_v_state_start = StackedLinear(dim_value, dim_value // 2, num_heads, num_heads, bias=False)
        self.proj_down_k_state_end = StackedLinear(dim_value, dim_key // 2, num_heads, num_heads, bias=False)
        self.proj_down_q_state_end = StackedLinear(dim_value, dim_key // 2, num_heads, num_heads, bias=False)
        self.proj_down_v_state_end = StackedLinear(dim_value, dim_value // 2, num_heads, num_heads, bias=False)
        
        # State projections from the second attention layer to the first attention layer
        self.proj_up_k_state_start = StackedLinear(dim_value // 2, dim_key, num_heads, num_heads, bias=False)
        self.proj_up_q_state_start = StackedLinear(dim_value // 2, dim_key, num_heads, num_heads, bias=False)
        self.proj_up_v_state_start = StackedLinear(dim_value // 2, dim_value, num_heads, num_heads, bias=False)
        self.proj_up_k_state_end = StackedLinear(dim_value // 2, dim_key, num_heads, num_heads, bias=False)
        self.proj_up_q_state_end = StackedLinear(dim_value // 2, dim_key, num_heads, num_heads, bias=False)
        self.proj_up_v_state_end = StackedLinear(dim_value // 2, dim_value, num_heads, num_heads, bias=False)
        
        # State projection for output
        self.proj_out_state = nn.Linear(num_heads * dim_value, dim_input, bias=False)
    
    def apply_attention(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        position_embedder: Optional[RoPEEmbeddings] = None, 
        offset: Optional[int] = None
    ) -> torch.Tensor:
        """
        Applies attention to the input tensors.

        Args:
            q (torch.Tensor): Query tensor of shape (batch_size, num_heads, seq_len, dim_key).
            k (torch.Tensor): Key tensor of shape (batch_size, num_heads, seq_len, dim_key).
            v (torch.Tensor): Value tensor of shape (batch_size, num_heads, seq_len, dim_value).
            position_embedder (Optional[RoPEEmbeddings]): Optional position embedder to apply to the query and key tensors.
            offset (Optional[int]): Optional offset to apply to the position embeddings.
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_heads, seq_len, dim_value).
        """
        # If position embedder is specified, add positional embeddings to q and k
        if position_embedder is not None:
            k = position_embedder(k, total_seq_len=k.size(2), offset=offset)
            q = position_embedder(q, total_seq_len=q.size(2), offset=offset)
        
        # Calculate attention scores using the projected key, and query tensors
        scores = q @ k.transpose(-2, -1) / self.dim_key ** 0.5

        # Calculate and apply causal attention mask
        mask = torch.tril(torch.ones((k.size(2), k.size(2)), dtype=torch.bool), diagonal=0)
        mask = mask.unsqueeze(0).unsqueeze(0).repeat((k.size(1), self.num_heads, 1, 1))
        scores.masked_fill_(torch.logical_not(mask), float('-inf'))

        # Calculate SDP attention
        att = nn.functional.softmax(scores, dim=-1) @ v

        return att
    
    
    @staticmethod
    def extract_state(x: torch.Tensor, state_len: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extracts the state from the input tensor x.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_heads, seq_len + 2 * state_len, dim_embedding).
            state_len (int): Length of the state.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - state_start: Tensor of shape (batch_size, num_heads, state_len, dim_embedding)
                - x: Tensor of shape (batch_size, num_heads, seq_len, dim_embedding)
                - state_end: Tensor of shape (batch_size, num_heads, state_len, dim_embedding)
        """
        return x[...,state_len:,:], x[...,state_len:-state_len,:], x[...,:-state_len,:]


    def forward(self, x: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies recurrent dual-attention to the input tensor x.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim_input).
            state (torch.Tensor): Initial state tensor of shape (1, state_len, dim_input).
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensor of shape (batch_size, seq_len, dim_input) and terminal state tensor of shape (1, state_len, dim_input).
        """
        batch_size, seq_len, _ = x.shape
        state_len = state.size(1)

        num_segments, rem = divmod(seq_len, self.segment_len)
        num_segments += 1 if rem > 0 else 0

        out = []
        
        x = x.unsqueeze(1)
        state = state.unsqueeze(1).repeat(batch_size, 1, 1, 1)
        
        # Project the input tensor to get the key, value, and query tensors
        k_full = self.proj_k(x)
        q_full = self.proj_q(x)
        v_full = self.proj_v(x)
        
        for ix in range(num_segments):
            ix_lo = ix * self.segment_len
            ix_hi = min(ix_lo + self.segment_len, x.size(2))
            seg_len = ix_hi - ix_lo
            
            # FIRST OUTER ATTENTION PASS

            # Extract segment from key, value and query tensors
            k = k_full[:, :, ix_lo:ix_hi, :]
            v = v_full[:, :, ix_lo:ix_hi, :]
            q = q_full[:, :, ix_lo:ix_hi, :]
            
            # Project the state tensor to get the key, value, and query state tensors
            k_state = self.proj_k_state(state)
            v_state = self.proj_v_state(state)
            q_state = self.proj_q_state(state)
            
            # Append and prepend state tensors to the key, value and query tensors
            k = torch.cat([k_state, k, k_state], dim=2)
            v = torch.cat([v_state, v, v_state], dim=2)
            q = torch.cat([q_state, q, q_state], dim=2)
            
            # Calculate SDP attention with the concatenated tensors
            att = self.apply_attention(q, k, v, self.position_embedder_1, ix_lo - state.size(1))
            
            # Extract state from result
            att_state_start, att, att_state_end = self.extract_state(att, state_len)
            
            # INNER ATTENTION PASS
            
            # Project attention result to get new key, value and query tensors
            k_inner = self.proj_down_k(att)
            q_inner = self.proj_down_k(att)
            v_inner = self.proj_down_k(att)
            
            # Project start state tensors for key, value and query tensors
            k_state_start = self.proj_down_k_state_start(att_state_start)
            q_state_start = self.proj_down_q_state_start(att_state_start)
            v_state_start = self.proj_down_v_state_start(att_state_start)
            
            # Project end state tensors for key, value and query tensors
            k_state_end = self.proj_down_k_state_end(att_state_end)
            q_state_end = self.proj_down_q_state_end(att_state_end)
            v_state_end = self.proj_down_v_state_end(att_state_end)
            
            # Concatenate state tensors with the key, value and query tensors
            k_inner = torch.cat([k_state_start, k_inner, k_state_end], dim=2)
            q_inner = torch.cat([q_state_start, q_inner, q_state_end], dim=2)
            v_inner = torch.cat([v_state_start, v_inner, v_state_end], dim=2)
            
            # Calculate SDP attention with the concatenated tensors
            att_inner = self.apply_attention(q_inner, k_inner, v_inner, self.position_embedder_2, ix_lo - state.size(1))
            
            # Extract state from result
            att_inner_state_start, att_inner, att_inner_state_end = self.extract_state(att_inner, state_len)
            
            # SECOND OUTER ATTENTION PASS
            
            # Project outer attention result to get new key, value and query tensors
            k_final = self.proj_up_k(att_inner)
            q_final = self.proj_up_q(att_inner)
            v_final = self.proj_up_v(att_inner)
            
            # Project start state tensors for key, value and query tensors
            k_state_start_final = self.proj_up_k_state_start(att_inner_state_start)
            q_state_start_final = self.proj_up_q_state_start(att_inner_state_start)
            v_state_start_final = self.proj_up_v_state_start(att_inner_state_start)
            
            # Project end state tensors for key, value and query tensors
            k_state_end_final = self.proj_up_k_state_end(att_inner_state_end)
            q_state_end_final = self.proj_up_q_state_end(att_inner_state_end)
            v_state_end_final = self.proj_up_v_state_end(att_inner_state_end)
            
            # Concatenate state tensors with the key, value and query tensors
            k_final = torch.cat([k_state_start_final, k_final, k_state_end_final], dim=2)
            q_final = torch.cat([q_state_start_final, q_final, q_state_end_final], dim=2)
            v_final = torch.cat([v_state_start_final, v_final, v_state_end_final], dim=2)
            
            # Calculate SDP attention with the concatenated tensors
            att_final = self.apply_attention(q_final, k_final, v_final, self.position_embedder_1, ix_lo - state.size(1))
            
            # Extract next state from result
            _, att_final, att_state_end_final = self.extract_state(att_final, state_len)
            
            # Reshape before final projections
            att_final = att_final.view((batch_size, seg_len, -1))
            att_state_end_final = att_state_end_final.view((batch_size, state_len, -1))
            
            # Get next state
            state = self.proj_out_state(att_state_end_final)
            
            # Append output to buffer
            out.append(self.proj_out(att_final))

        # Return concatenated full sequence from buffer
        out = torch.concat(out, dim=1)

        return out
