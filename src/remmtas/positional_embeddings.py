import math
from typing import Optional

import torch


class RoPEEmbeddings(torch.nn.Module):
    """Implements rotary positional embeddings (RoPE) as described in the paper:
    "RoFormer: Enhanced Transformer with Rotary Position Embedding" by Su et al.
    (https://arxiv.org/abs/2104.09864).
    
    Modifications have been made to make it compatible with ReMMTAS."""
    def __init__(self, dim: int, seq_len: int, dim_embedding_pct: float = 0.5, base: int = 10000, device: Optional[str] = None):
        """Instantiate the module.

        Args:
            dim (int): Key/Value dimension of the attention layer.
            seq_len (int): Maximum sequence length.
            dim_embedding_pct (float): Percentage of the total embedding dimension to use for the positional embeddings. Must be within the interval (0, 1]. Defaults to 0.5.
            base (int, optional): Base used for calculating thetas. Defaults to 10000.
            device (Optional[str], optional): Device to use for the positional embeddings. Defaults to None.
        """
        super(RoPEEmbeddings, self).__init__()
        
        # Record input parameters
        self.dim = dim
        self.effective_dim = math.ceil(dim * dim_embedding_pct)
        self.seq_len = seq_len
        self.dim_embedding_pct = dim_embedding_pct
        self.base = base
        self.last_offset = 0
        self.device = device
        
        self._calculate_thetas()
        
        # Initialize sin component indices for input tensor
        # Indices for rearranging the input follow the pattern [1, 0, 3, 2, 5, 4, ...]
        # Indices that need to be negated in calculating the positional embeddings are [0, 2, 4, ...]
        self.ixs_sin = torch.empty(self.effective_dim, dtype=torch.long, device=device)
        self.ixs_sin_neg = 2 * torch.arange(self.effective_dim // 2, device=device)
        self.ixs_sin[self.ixs_sin_neg] = self.ixs_sin_neg + 1
        self.ixs_sin[self.ixs_sin_neg + 1] = self.ixs_sin_neg
        
    def _calculate_thetas(self, offset: int = 0) -> None:
        """Calculate the cosine and sine component matrices for the rotary positional embeddings.
        Uses multidimensional extension of theta as defined in Sec 3.2.2 as well as equation (34)
        from the RoFormer paper

        Args:
            offset (int, optional): Position offset for ReDoTAS compatibility. Defaults to 0.
        """
        # Calculate matrix of angles: thetas[i,j] = base^(-2 * ceil(i/2)) * (j + offset)
        thetas = torch.repeat_interleave(
            (self.base ** (-2. * torch.arange(1, self.effective_dim//2 + 1, device=self.device))).unsqueeze(-1).repeat((1, self.seq_len)), 
            repeats=2, 
            dim=0
        )
        # Multiply by index positions, then transpose to get correct shape
        if offset < 0:
            mults = torch.cat([torch.ones(-offset, device=self.device), torch.arange(1, self.seq_len + 1 + offset, device=self.device)], dim=0)
        else:
            mults = torch.arange(1 + offset, self.seq_len + 1 + offset, device=self.device)
        thetas *= mults.unsqueeze(0)
        self.thetas = thetas.transpose(0, 1).unsqueeze(0).unsqueeze(0)
        
    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Applies rotary positional embeddings to the input tensor. Uses a multidimensional
        extension of equation (34) of the RoFormer paper.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_heads, seq_len, dim).
            offset (int, optional): Position offset for ReDoTAS compatibility. Defaults to 0.

        Returns:
            torch.Tensor: Transformed input tensor with rotary positional embeddings applied.
        """
        if offset != self.last_offset:
            self._calculate_thetas(offset=offset)
            self.last_offset = max(offset, 0)
        
        if self.dim_embedding_pct < 1.0:
            x_pos = x[..., :self.effective_dim]
            x_pass = x[..., self.effective_dim:]
        else:
            x_pos = x
        
        # If the sequence length is less than the maximum sequence length, perform calculations
        # with truncated cos_component and sin_component, along the sequence axis
        if x.size(2) < self.seq_len:
            x_cos = self.thetas.cos()[:, :, :x_pos.size(2), :].repeat(x_pos.size(0), x_pos.size(1), 1, 1) * x_pos
            x_sin = x_pos[..., self.ixs_sin]
            x_sin[..., self.ixs_sin_neg] = -x_sin[...,self.ixs_sin_neg]
            x_sin *= self.thetas.sin()[:, :, :x_pos.size(2), :].repeat(x_pos.size(0), x_pos.size(1), 1, 1)
        # Otherwise, perform calculations with the full cos_component and sin_component
        else:
            x_cos = self.thetas.cos().repeat(x_pos.size(0), x_pos.size(1), 1, 1) * x_pos
            x_sin = x_pos[..., self.ixs_sin]
            x_sin[..., self.ixs_sin_neg] = -x_sin[...,self.ixs_sin_neg]
            x_sin *= self.thetas.sin().repeat(x_pos.size(0), x_pos.size(1), 1, 1)
    
            
        # If the sequence length is less than the maximum sequence length, concatenate positionally embedded
        # entries with original entries, otherwise return the positionally embedded entries
        if self.dim_embedding_pct < 1.0:
            out = torch.cat([x_cos + x_sin, x_pass], dim=-1)
        else:
            out = x_cos + x_sin
        
        return out
