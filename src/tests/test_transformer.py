# tests/test_transformer.py

import torch
from xformers.components.positional_embedding import RotaryEmbedding

from ..om.arcformer import ARCformer
from ..om.arcformer.util import count_optimized_parameters


def test_arc_transformer():
    dim_input = 1024
    num_heads = 8
    dim_hidden = 4 * dim_input
    dims_key = [2 * dim_input // num_heads, dim_input // num_heads, dim_input // (2 * num_heads)]
    dims_value = [2 * dim_input // num_heads, dim_input // num_heads, dim_input // (2 * num_heads)]
    num_iters = [2, 2, 2]
    betas = [None, None, None]
    attn_proj_rank = -1
    
    activation = "gelu"
    mlp_1221 = True
    segment_len = 1024
    attn_normalize = True
    state_len = segment_len // 8
    num_layers = 8
    mlp_multiplier = 1
    
    dropout = 0.1
    attn_dropout = 0.1
    diff_attn = False
    layer_num = 0
    
    position_embedders = [RotaryEmbedding(dim) for dim in dims_key]
    cope = True

    layer = ARCformer(
        dim_input=dim_input,
        dim_hidden=dim_hidden,
        dims_key=dims_key,
        dims_value=dims_value,
        num_iters=num_iters,
        num_heads=num_heads,
        activation=activation,
        segment_len=segment_len,
        state_len=state_len,
        attn_normalize=attn_normalize,
        num_layers=num_layers,
        layer_num=layer_num,
        cope=cope,
        position_embedders=position_embedders,
        betas=betas,
        dropout=dropout,
        diff_attn=diff_attn,
        attn_dropout=attn_dropout,
        attn_proj_rank=attn_proj_rank,
        mlp_multiplier=mlp_multiplier,
        mlp_1221=mlp_1221
    )

    batch_size = 2
    seq_len = segment_len
    x = torch.randn(batch_size, seq_len, dim_input).to(torch.bfloat16)
    state = torch.randn(batch_size, state_len, dim_input).to(torch.bfloat16)
    offset = 0
    
    if torch.cuda.is_available():
        layer = layer.cuda()
        x = x.cuda()
        state = state.cuda()

    layer = layer.to(torch.bfloat16)
    for name, param in layer.named_parameters():
        if "LayerNorm" in name:
            param.data = param.data.to(torch.float32)
    layer.eval()  # Set the layer to evaluation mode
    
    with torch.no_grad():
        x_att, state_next = layer(x, state, offset)

    assert x_att.shape == (batch_size, seq_len, dim_input)
    assert state_next.shape == (batch_size, state_len, dim_input)
    
    param_ct = count_optimized_parameters(layer)
    print(f"Total optimized parameters: {param_ct:,d}")
