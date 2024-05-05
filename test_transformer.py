# tests/test_transformer.py

import torch
from rdt_as import ReDoTransformer, RoPEEmbeddings

def test_redo_transformer():
    dim_input = 512
    dim_hidden = 2048
    dim_key = 64
    dim_value = 64
    num_heads = 8
    activation = "ffngeglu"
    segment_len = 2048
    state_len = 4
    
    dropout = 0.1
    
    position_embedder_1 = RoPEEmbeddings(
        dim=dim_key,
        seq_len=segment_len + 2 * state_len,
        dim_embedding_pct=0.5,
        base=10000
    )
    position_embedder_2 = RoPEEmbeddings(
        dim=dim_key // 2,
        seq_len=segment_len + 2 * state_len,
        dim_embedding_pct=0.5,
        base=10000
    )

    layer = ReDoTransformer(
        dim_input=dim_input,
        dim_hidden=dim_hidden,
        dim_key=dim_key,
        dim_value=dim_value,
        num_heads=num_heads,
        activation=activation,
        segment_len=segment_len,
        state_len=4,
        position_embedder_1=position_embedder_1,
        position_embedder_2=position_embedder_2,
        dropout=dropout
    )

    batch_size = 2
    seq_len = 4096
    x = torch.randn(batch_size, seq_len, dim_input)

    layer.eval()  # Set the layer to evaluation mode
    x_att = layer(x)

    assert x_att.shape == (batch_size, seq_len, dim_input)

if __name__=="__main__":
    test_redo_transformer()