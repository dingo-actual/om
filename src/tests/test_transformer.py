# tests/test_transformer.py

import torch

from ..arcformer import ARCformer, RoPEEmbeddings
from ..arcformer.util import count_optimized_parameters


def test_arc_transformer():
    dim_input = 768
    num_heads = 12
    dim_hidden = 4 * dim_input
    dims_key = [dim_input // num_heads, 2 * dim_input // num_heads, 4 * dim_input // num_heads]
    dims_value = [dim_input // num_heads, 2 * dim_input // num_heads, 4 * dim_input // num_heads]
    attn_proj_rank = 2 * dim_input // num_heads
    
    activation = "gelu"
    mlp_1221 = True
    segment_len = 1024
    attn_normalize = True
    state_len = segment_len // 8
    num_layers = 8
    mlp_multiplier = 1
    
    dropout = 0.1
    attn_dropout = 0.1
    diff_attn = True
    layer_num = 0
    
    position_embedders = [
        RoPEEmbeddings(
            dim=dim_key,
            seq_len=segment_len + 2 * state_len,
            dim_embedding_pct=0.25,
            base=10000,
        ) for dim_key in dims_key
    ]
    cope = True

    layer = ARCformer(
        dim_input=dim_input,
        dim_hidden=dim_hidden,
        dims_key=dims_key,
        dims_value=dims_value,
        num_heads=num_heads,
        activation=activation,
        segment_len=segment_len,
        state_len=state_len,
        attn_normalize=attn_normalize,
        num_layers=num_layers,
        layer_num=layer_num,
        cope=cope,
        position_embedders=position_embedders,
        dropout=dropout,
        diff_attn=diff_attn,
        attn_dropout=attn_dropout,
        attn_proj_rank=attn_proj_rank,
        mlp_multiplier=mlp_multiplier,
        mlp_1221=mlp_1221
    )

    batch_size = 2
    seq_len = segment_len
    x = torch.randn(batch_size, seq_len, dim_input)
    state = torch.randn(batch_size, state_len, dim_input)
    offset = 0
    
    if torch.cuda.is_available():
        layer = layer.cuda()
        x = x.cuda()
        state = state.cuda()

    layer.eval()  # Set the layer to evaluation mode
    x_att, state_next = layer(x, state, offset)

    assert x_att.shape == (batch_size, seq_len, dim_input)
    assert state_next.shape == (batch_size, state_len, dim_input)
    
    param_ct = count_optimized_parameters(layer)
    print(f"Total optimized parameters: {param_ct:,d}")
    
    del layer
