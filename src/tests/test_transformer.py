# tests/test_transformer.py

import torch

from ..arcformer import ARCformer, RoPEEmbeddings
from ..arcformer.util import count_optimized_parameters


def test_arc_transformer():
    dim_input = 2048
    num_heads = 8
    dim_hidden = int(8 * dim_input / 3)
    dims_key = [dim_input // num_heads, 2 * dim_input // num_heads, dim_input // num_heads]
    dims_value = [dim_input // num_heads, 2 * dim_input // num_heads, dim_input // num_heads]
    mem_iters = [1, 3, 1]
    
    activation = "ffngeglu"
    segment_len = 128
    normalize_qkv = True
    state_len = segment_len // 8
    
    dropout = 0.1
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    position_embedders = [
        RoPEEmbeddings(
            dim=dim_key,
            seq_len=segment_len + 2 * state_len,
            dim_embedding_pct=0.25,
            base=10000,
            device=device
        ) for dim_key in dims_key
    ]

    layer = ARCformer(
        dim_input=dim_input,
        dim_hidden=dim_hidden,
        dims_key=dims_key,
        dims_value=dims_value,
        mem_iters=mem_iters,
        num_heads=num_heads,
        activation=activation,
        segment_len=segment_len,
        state_len=state_len,
        normalize_qkv=normalize_qkv,
        position_embedders=position_embedders,
        dropout=dropout,
        device=device
    )

    batch_size = 2
    seq_len = 256
    x = torch.randn(batch_size, seq_len, dim_input, device=device, dtype=torch.bfloat16)

    layer.eval()  # Set the layer to evaluation mode
    x_att = layer(x)

    assert x_att.shape == (batch_size, seq_len, dim_input)
    
    param_ct = count_optimized_parameters(layer)
    print(f"Total optimized parameters: {param_ct:,d}")

def main():
    test_arc_transformer()