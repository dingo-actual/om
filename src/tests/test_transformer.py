import torch

from ..om.arcformer import ARCformer, RoPEEmbeddings
from ..om.arcformer.util import count_optimized_parameters
from ..om.utils import set_om_dtypes


def test_arc_transformer():
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    
    dim_input = 1024
    num_heads = 8
    dim_hidden = 4 * dim_input
    dims_key = [2 * dim_input // num_heads, dim_input // num_heads, dim_input // (2 * num_heads)]
    dims_value = [2 * dim_input // num_heads, dim_input // num_heads, dim_input // (2 * num_heads)]
    scaling_factors = [None, None, None]
    attn_proj_rank = dim_input // (2 * num_heads)
    
    activation = "gelu"
    mlp_1221 = True
    segment_len = 512
    state_len = segment_len // 8
    num_layers = 8
    
    dropout = 0.1
    attn_dropout = 0.1
    attn_logit_dropout = 0.1
    diff_attn = False
    layer_num = 0
    
    stacked_attn = False
    
    max_init_ngrams = 3
    
    position_embedders = [
        RoPEEmbeddings(
            dim, 
            seq_len=segment_len + 2 * state_len, 
            num_dims=4 if stacked_attn else 3
        ) 
        for dim in dims_key
    ]
    cope = False
    
    layer = ARCformer(
        dim_input=dim_input,
        dim_hidden=dim_hidden,
        dims_key=dims_key,
        dims_value=dims_value,
        num_heads=num_heads,
        activation=activation,
        segment_len=segment_len,
        state_len=state_len,
        num_layers=num_layers,
        layer_num=layer_num,
        cope=cope,
        position_embedders=position_embedders,
        scaling_factors=scaling_factors,
        dropout=dropout,
        diff_attn=diff_attn,
        attn_dropout=attn_dropout,
        attn_logit_dropout=attn_logit_dropout,
        attn_proj_rank=attn_proj_rank,
        mlp_1221=mlp_1221,
        stacked_attn=stacked_attn,
    )

    batch_size = 2
    seq_len = segment_len - max_init_ngrams - 1
    x = torch.randn(batch_size, seq_len, dim_input).to(torch.bfloat16)
    state = torch.randn(batch_size, state_len, dim_input).to(torch.bfloat16)
    
    layer = layer.to(device=device)
    x = x.to(device=device)
    state = state.to(device=device)

    layer = set_om_dtypes(layer, torch.bfloat16)
    layer.eval()  # Set the layer to evaluation mode
    
    with torch.no_grad():
        x_att, state_next = layer(x, state)

    assert x_att.shape == (batch_size, seq_len, dim_input)
    assert state_next.shape == (batch_size, state_len, dim_input)
    
    param_ct = count_optimized_parameters(layer)
    print(f"Total optimized parameters: {param_ct:,d}")
