from typing import Literal, Optional, Union

import torch
from torch import nn

from .activations import ACTIVATIONS
from .positional_embeddings import PositionEmbeddings
from .util import StackedLinear

class RDTAS(nn.Module):
    """Implements Recurrent Double Transformer with Attentive State (RDTAS)."""

    def __init__(
        self, 
        dim_input: int, 
        dim_key: int, 
        dim_value: int, 
        num_heads: int, 
        segment_len: int, 
        activation: Optional[Union[Literal["relu"], Literal["gelu"], Literal["swish"], Literal["swiglu"], Literal["geglu"], Literal["ffnglu"], Literal["ffngeglu"], Literal["ffnswiglu"], Literal["abs"]]] = None,
        position_embedder: Optional[PositionEmbeddings] = None
    ):
        """Initialize module.

        Args:
            dim_input (int): Input dimension.
            dim_key (int): Key dimension.
            dim_value (int): Value dimension.
            num_heads (int): Number of attention heads.
            segment_len (int): Segment length (must be a factor of the input sequence length).
            activation (str, optional): Activation function to use. Options are "relu", "gelu", "swish", "swiglu", "geglu", "ffnglu", "ffngeglu", "ffnswiglu", and "abs". Defaults to None.
            position_embedder (Optional[PositionEmbeddings], optional): Position embedding module. Defaults to None.
        """
        super(RDTAS, self).__init__()

        # Record input parameters
        self.num_heads = num_heads
        self.segment_len = segment_len

        self.dim_input = dim_input
        self.dim_key = dim_key
        self.dim_value = dim_value
        
        self.position_embedder = position_embedder

        self.proj_k1 = StackedLinear(dim_input, dim_key, num_heads, bias=False)
        self.proj_v1 = StackedLinear(dim_input, dim_value, num_heads, bias=False)
        self.proj_q1 = StackedLinear(dim_input, dim_key, num_heads, bias=False)
        
        self.norm_kq1 = torch.nn.LayerNorm(dim_key)
        self.norm_v1 = torch.nn.LayerNorm(dim_value)

        # Projection for output
        self.proj_out1 = nn.Linear(num_heads * dim_value, dim_input, bias=False)
        
        if activation in ["swiglu", "geglu", "ffnglu", "ffngeglu", "ffnswiglu"]:
            self.activation = ACTIVATIONS[activation](dim_value)
        elif activation is not None:
            self.activation = ACTIVATIONS[activation]()
        else:
            self.activation = None
        
        

    def forward(self, x: torch.Tensor, sample_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Applies Compressive Memory Attention to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim_input).
            sample_mask (Optional[torch.Tensor], optional): Mask tensor of shape (batch_size, seq_len) used to sample the input sequence. Defaults to None.
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim_input).
        """
        batch_size, seq_len, _ = x.shape

        num_segments, rem = divmod(seq_len, self.segment_len)
        num_segments += 1 if rem > 0 else 0

        out = []

        # Initialize mem and normalization
        if self.init_mem is not None and self.init_z is not None:
            mem = self.init_mem
            z = self.init_z
        else:
            # !!! Initialization was never specified in the paper, so this is an educated guess
            mem = torch.zeros(1, self.num_heads, self.dim_key, self.dim_value)
            z = torch.ones(batch_size, self.num_heads, self.dim_key, 1) / self.dim_key
        
        # Project the input tensor to get the key, value, and query tensors
        k_full = self.proj_k(x).unsqueeze(1).view(
            (batch_size, self.num_heads, x.size(1), self.dim_key))
        v_full = self.proj_v(x).unsqueeze(1).view(
            (batch_size, self.num_heads, x.size(1), self.dim_value))
        q_full = self.proj_q(x).unsqueeze(1).view(
            (batch_size, self.num_heads, x.size(1), self.dim_key))
        
        for ix in range(num_segments):
            ix_lo = ix * self.segment_len
            ix_hi = min(ix_lo + self.segment_len, x.size(1))
            seg_len = ix_hi - ix_lo

            # Extract segment from key, value and query tensors
            k = k_full[:, :, ix_lo:ix_hi, :]
            v = v_full[:, :, ix_lo:ix_hi, :]
            q = q_full[:, :, ix_lo:ix_hi, :]
            
            # If sample_mask was given, extract segment from it
            if sample_mask is not None:
                if self.sampling_factor is None:
                    raise ValueError("sampling_factor must be specified if sample_mask is provided")
                ix_lo_seg = ix * self.segment_len * self.sampling_factor
                ix_hi_seg = min(ix_lo_seg + self.segment_len * self.sampling_factor, sample_mask.size(1))
                sample_mask_seg = sample_mask[:, ix_lo_seg:ix_hi_seg]
            else:
                sample_mask_seg = None
            
            # If position embedder is specified, add positional embeddings to q and k
            if self.position_embedder is not None:
                if sample_mask is None:
                    k_pos = self.position_embedder(k, total_seq_len=seq_len, offset=ix_lo)
                    q_pos = self.position_embedder(q, total_seq_len=seq_len, offset=ix_lo)
                else:
                    k_pos = self.position_embedder(k, total_seq_len=seq_len, offset=ix_lo_seg, select_mask=sample_mask_seg)
                    q_pos = self.position_embedder(q, total_seq_len=seq_len, offset=ix_lo_seg, select_mask=sample_mask_seg)
            
            # Pre-calculate sigma(q) for updating memory and calculating attention
            # The calculation is described on page 4 of the paper under the subsection
            # "Memory retrieval"
            # shape: (batch_size, num_heads, segment_len, dim_key)
            sigma_q = (nn.functional.elu(q) + 1.0)

            # Apply SDP attention, as part of equation (2) of the paper
            if self.position_embedder is not None:
                scores = q_pos @ k_pos.transpose(-2, -1) / self.dim_key ** 0.5
            else:
                scores = q @ k.transpose(-2, -1) / self.dim_key ** 0.5

            # If causal mask specified, calculate and apply it
            if self.causal:
                mask = torch.tril(torch.ones((seg_len, seg_len), dtype=torch.bool), diagonal=0)
                mask = mask.unsqueeze(0).unsqueeze(0).repeat((batch_size, self.num_heads, 1, 1))
                scores.masked_fill_(torch.logical_not(mask), float('-inf'))

            # Calculate SDP attention, completing equation (2) of the paper
            att_dot = nn.functional.softmax(scores, dim=-1) @ v

            # Calculate normalized linear attention
            # The calculation is described in equation (3) of the paper
            # shape: (batch_size, num_heads, segment_len, dim_value)
            att_mem = (sigma_q @ mem) / (sigma_q @ z)

            # Apply mem update
            # The update rules are described in equations (4) and (5) of the paper
            sigma_k = nn.functional.elu(k) + 1.0
            if self.update == "linear":
                mem = mem + sigma_k.transpose(-2, -1) @ v
            elif self.update == "delta":
                mem = mem + \
                    sigma_k.transpose(-2, -1) @ (v - (sigma_k @ mem) / (sigma_k @ z))
                    
            # Apply normalization term update
            # The calculation is described in equation (4) of the paper
            z = z + (nn.functional.elu(k) + 1.0).sum(dim=-2, keepdim=True).transpose(-2, -1)

            # Calculate weighted average of dot-product and memory-based attention
            # The calculation is described in equation (6) of the paper
            att = nn.functional.sigmoid(
                self.betas) * att_mem + (1 - nn.functional.sigmoid(self.betas)) * att_dot
            att = att.view((batch_size, seg_len,
                        self.num_heads * self.dim_value))

            # Append output to buffer
            # The calculation is described in equation (7) of the paper
            out.append(self.proj_out(att))

        # Return concatenated full sequence from buffer
        out = torch.concat(out, dim=1)

        return out


def test_compressive_memory(
    short_seq_len: bool = False, 
    even_seq_len: bool = True, 
    causal_masking: bool = False, 
    update: str = "linear"
) -> None:
    # Set example module parameters
    dim_input = 512
    dim_key = 64
    dim_value = 64
    num_heads = 8
    segment_len = 32
    causal = causal_masking
    
    # Set dummy input dimensions
    batch_size = 4
    
    # Handle sequence length based on test case
    if short_seq_len:
        seq_len = 16
    else:
        if even_seq_len:
            seq_len = 128
        else:
            seq_len = 144

    # Initialize module
    model = CompressiveMemory(
        dim_input, dim_key, dim_value, num_heads, segment_len, update, causal)

    # Generate random input
    batch = torch.randn(batch_size, seq_len, dim_input)

    # Apply the CompressiveMemory module
    model(batch)


if __name__ == "__main__":
    # Test all cases with short sequence lengths
    print("Testing with short sequence lengths:")
    
    short_seq_len = True
    # In this case even_seq_len doesn't matter -- arbitrarily setting it to True
    even_seq_len = True
    
    for causal_masking in [True, False]:
        for update in ["linear", "delta"]:
            print(f"  Testing with causal_masking={causal_masking} and update={update}")
            test_compressive_memory(
                short_seq_len=short_seq_len,
                even_seq_len=even_seq_len,
                causal_masking=causal_masking,
                update=update
            )
            
    # Test all cases with short sequence lengths
    print("Testing with non-short sequence lengths:")
    
    short_seq_len = False
    
    for even_seq_len in [True, False]:
        for causal_masking in [True, False]:
            for update in ["linear", "delta"]:
                print(f"  Testing with even_seq_len={even_seq_len}, causal_masking={causal_masking} and update={update}")
                test_compressive_memory(
                    short_seq_len=short_seq_len,
                    even_seq_len=even_seq_len,
                    causal_masking=causal_masking,
                    update=update
                )