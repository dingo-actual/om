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
    
    activation = "ffngeglu"
    segment_len = 128
    normalize = True
    state_len = segment_len // 8
    num_layers = 8
    
    dropout = 0.1
    
    position_embedders = [
        RoPEEmbeddings(
            dim=dim_key,
            seq_len=segment_len + 2 * state_len,
            dim_embedding_pct=0.25,
            base=10000,
        ) for dim_key in dims_key
    ]

    layer = ARCformer(
        dim_input=dim_input,
        dim_hidden=dim_hidden,
        dims_key=dims_key,
        dims_value=dims_value,
        num_heads=num_heads,
        activation=activation,
        segment_len=segment_len,
        state_len=state_len,
        attn_normalize=normalize,
        num_layers=num_layers,
        cope=True,
        position_embedders=position_embedders,
        dropout=dropout
    )

    batch_size = 2
    seq_len = segment_len
    x = torch.randn(batch_size, seq_len, dim_input)
    state = torch.randn(batch_size, state_len, dim_input)
    offset = 0

    layer.eval()  # Set the layer to evaluation mode
    x_att, state_next = layer(x, state, offset)

    assert x_att.shape == (batch_size, seq_len, dim_input)
    assert state_next.shape == (batch_size, state_len, dim_input)
    
    param_ct = count_optimized_parameters(layer)
    print(f"Total optimized parameters: {param_ct:,d}")
