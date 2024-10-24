import torch

from ..om.arcformer import RoPEEmbeddings
from ..om.arcformer.util import count_optimized_parameters
from ..om.om_llm import OmLLM


def test_model():
    num_layers = 8
    vocab_size = 50284
    
    dim_input = 1024
    num_heads = 8
    dim_hidden = 4 * dim_input
    if dim_hidden % 32 != 0:
        dim_hidden += 32 - dim_hidden % 32
    dims_key = [2 * dim_input // num_heads, dim_input // num_heads, dim_input // (2 * num_heads)]
    dims_value = [2 * dim_input // num_heads, dim_input // num_heads, dim_input // (2 * num_heads)]
    num_iters = [2, 2, 2]
    betas = [None, None, None]
    final_mlp_multiplier = 1
    attn_proj_rank = -1
    
    activation = "gelu"
    mlp_1221 = True
    segment_len = 2048
    attn_normalize = True
    cope = True
    state_len = segment_len // 8
    
    init_convs = [2, 3]
    max_init_convs = 1 if len(init_convs) == 0 else max(init_convs)
    
    dropout = 0.1
    attn_dropout = 0.1
    diff_attn = False
    
    batch_size = 2
    num_segments = 4
    next_token = False
    
    position_embedder_1 = RoPEEmbeddings(
        dim=dims_key[0],
        seq_len=segment_len + 2 * state_len,
        dim_embedding_pct=0.25,
        base=10000
    )
    position_embedder_2 = RoPEEmbeddings(
        dim=dims_key[1],
        seq_len=segment_len + 2 * state_len,
        dim_embedding_pct=0.25,
        base=10000
    )
    position_embedder_3 = RoPEEmbeddings(
        dim=dims_key[2],
        seq_len=segment_len + 2 * state_len,
        dim_embedding_pct=0.25,
        base=10000
    )
    
    position_embedders = [
        position_embedder_1,
        position_embedder_2,
        position_embedder_3
    ]
    
    model = OmLLM(
        num_layers=num_layers,
        vocab_size=vocab_size,
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
        cope=cope,
        position_embedders=position_embedders,
        betas=betas,
        dropout=dropout,
        diff_attn=diff_attn,
        attn_dropout=attn_dropout,
        attn_proj_rank=attn_proj_rank,
        init_convs=init_convs,
        final_mlp_multiplier=final_mlp_multiplier,
        mlp_1221=mlp_1221,
    )
    
    seq_len = segment_len * num_segments
    x = vocab_size * torch.rand(batch_size, seq_len + max_init_convs - 1)
    x = x.to(torch.long)

    if torch.cuda.is_available():
        model = model.to("cuda:0")
        x = x.to("cuda:0")
    
    model = model.to(torch.bfloat16)
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        preds, states, offset = model(x, next_token=next_token)

    
    if next_token:
        assert preds.shape == (batch_size, vocab_size)
    else:
        assert preds.shape == (batch_size, seq_len, vocab_size)
    for state in states:
        assert state.shape == (batch_size, state_len, dim_input)
    assert offset == x.size(1)
    
    param_ct = count_optimized_parameters(model)
    print(f"Total optimized parameters: {param_ct:,d}")
    
    del model
