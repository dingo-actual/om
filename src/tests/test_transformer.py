# tests/test_transformer.py

import torch

from ..remmtas import ReMMTASformer, RoPEEmbeddings
from ..remmtas.util import count_optimized_parameters


def test_remmtas_transformer():
    dim_input = 1024
    dim_hidden = 2048
    dims_key = [128, 256, 128]
    dims_value = [128, 256, 128]
    mem_iters = [1, 3, 1]
    num_heads = 8
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

    layer = ReMMTASformer(
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
    x = torch.randn(batch_size, seq_len, dim_input, device=device)

    layer.eval()  # Set the layer to evaluation mode
    x_att = layer(x)

    assert x_att.shape == (batch_size, seq_len, dim_input)
    
    param_ct = count_optimized_parameters(layer)
    print(f"Total optimized parameters: {param_ct:,d}")

def main():
    test_remmtas_transformer()