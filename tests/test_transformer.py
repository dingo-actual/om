# tests/test_transformer.py

import torch

from remmtas import ReMMTASformer, RoPEEmbeddings


def test_remmtas_transformer():
    dim_input = 32
    dim_hidden = 256
    dims_key = [16, 32, 64]
    dims_value = [16, 32, 64]
    num_heads = 2
    activation = "ffngeglu"
    segment_len = 128
    state_len = 2
    
    dropout = 0.1
    
    position_embedders = [
        RoPEEmbeddings(
            dim=dim_key,
            seq_len=segment_len + 2 * state_len,
            dim_embedding_pct=0.5,
            base=10000
        ) for dim_key in dims_key
    ]

    layer = ReMMTASformer(
        dim_input=dim_input,
        dim_hidden=dim_hidden,
        dims_key=dims_key,
        dims_value=dims_value,
        num_heads=num_heads,
        activation=activation,
        segment_len=segment_len,
        state_len=state_len,
        position_embedders=position_embedders,
        dropout=dropout
    )

    batch_size = 2
    seq_len = 256
    x = torch.randn(batch_size, seq_len, dim_input)

    layer.eval()  # Set the layer to evaluation mode
    x_att = layer(x)

    assert x_att.shape == (batch_size, seq_len, dim_input)

if __name__=="__main__":
    test_remmtas_transformer()