import math

import torch


class RoPEEmbeddings(torch.nn.Module):
    """Implements rotary positional embeddings (RoPE) as described in the paper:
    "RoFormer: Enhanced Transformer with Rotary Position Embedding" by Su et al.
    (https://arxiv.org/abs/2104.09864).
    
    Modifications have been made to make it compatible with ARC."""
    def __init__(self, dim: int, seq_len: int, num_dims: int, dim_embedding_pct: float = 0.5, base: int = 10000):
        """Instantiate the module.

        Args:
            dim (int): Query/Key dimension of the attention layer.
            seq_len (int): Maximum sequence length.
            num_dims (int): Number of dimensions for inputs.
            dim_embedding_pct (float): Percentage of the total embedding dimension to use for the positional embeddings. Must be within the interval (0, 1]. Defaults to 0.5.
            base (int, optional): Base used for calculating thetas. Defaults to 10000.
        """
        super(RoPEEmbeddings, self).__init__()
        
        # Record input parameters
        self.dim = dim
        self.effective_dim = math.ceil(dim * dim_embedding_pct)
        self.seq_len = seq_len
        self.num_dims = num_dims
        self.dim_embedding_pct = dim_embedding_pct
        self.base = base
        
        if num_dims == 3:
            thetas = torch.empty((1, seq_len, self.effective_dim), dtype=torch.float32)
        elif num_dims == 4:
            thetas = torch.empty((1, 1, seq_len, self.effective_dim), dtype=torch.float32)
        else:
            raise ValueError("num_dims must be 3 or 4")
        
        self.register_buffer("thetas", thetas)
        self._calculate_thetas()
        
        # Initialize sin component indices for input tensor
        # Indices for rearranging the input follow the pattern [1, 0, 3, 2, 5, 4, ...]
        # Indices that need to be negated in calculating the positional embeddings are [0, 2, 4, ...]
        ixs_sin = torch.arange(self.effective_dim, dtype=torch.long)
        ixs_sin_neg = 2 * torch.arange(self.effective_dim // 2, dtype=torch.long)
        ixs_sin[ixs_sin_neg] = ixs_sin_neg + 1
        ixs_sin[ixs_sin_neg + 1] = ixs_sin_neg
        self.register_buffer("ixs_sin", ixs_sin)
        self.register_buffer("ixs_sin_neg", ixs_sin_neg)
        
    def _calculate_thetas(self) -> None:
        """Calculate the cosine and sine component matrices for the rotary positional embeddings.
        Uses multidimensional extension of theta as defined in Sec 3.2.2 as well as equation (34)
        from the RoFormer paper"""
        device = self.thetas.device
        dtype = self.thetas.dtype
        # Calculate matrix of angles: thetas[i,j] = base^(-2 * ceil(i/2)) * (j + offset)
        thetas = torch.repeat_interleave(
            (self.base ** (-2. * torch.arange(1, self.effective_dim//2 + 1, device=device, dtype=dtype))).unsqueeze(-1).repeat((1, self.seq_len)), 
            repeats=2, 
            dim=0
        )
        # Multiply by index positions, then transpose to get correct shape
        mults = torch.arange(1, self.seq_len + 1, device=device, dtype=dtype)
        thetas *= mults.unsqueeze(0)
        self.thetas.data = thetas.transpose(0, 1).unsqueeze(0)
        if self.num_dims == 4:
            self.thetas.data.unsqueeze_(0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies rotary positional embeddings to the input tensor. Uses a multidimensional
        extension of equation (34) of the RoFormer paper.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).

        Returns:
            torch.Tensor: Transformed input tensor with rotary positional embeddings applied.
        """
        if self.dim_embedding_pct < 1.0:
            x_pos = x[..., :self.effective_dim]
            x_pass = x[..., self.effective_dim:]
        else:
            x_pos = x
        
        if self.num_dims == 3:
            repeats = [x_pos.size(0), 1, 1]
        else:
            repeats = [x_pos.size(0), x_pos.size(1), 1, 1]
        
        # If the sequence length is less than the maximum sequence length, perform calculations
        # with truncated cos_component and sin_component, along the sequence axis
        if x.size(-2) < self.seq_len:
            x_cos = self.thetas.cos()[..., :x_pos.size(-2), :].repeat(*repeats) * x_pos
            x_sin = x_pos[..., self.ixs_sin]
            x_sin[..., self.ixs_sin_neg] = -x_sin[..., self.ixs_sin_neg]
            x_sin *= self.thetas.sin()[..., :x_pos.size(-2), :].repeat(*repeats)
        # Otherwise, perform calculations with the full cos_component and sin_component
        else:
            x_cos = self.thetas.cos().repeat(*repeats) * x_pos
            x_sin = x_pos[..., self.ixs_sin]
            x_sin[..., self.ixs_sin_neg] = -x_sin[..., self.ixs_sin_neg]
            x_sin *= self.thetas.sin().repeat(*repeats)
    
            
        # If the sequence length is less than the maximum sequence length, concatenate positionally embedded
        # entries with original entries, otherwise return the positionally embedded entries
        if self.dim_embedding_pct < 1.0:
            out = torch.cat([x_cos + x_sin, x_pass], dim=-1)
        else:
            out = x_cos + x_sin
        
        return out
