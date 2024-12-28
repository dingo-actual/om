import torch

from ..om.arcformer.util import count_optimized_parameters
from ..om.om_llm import OmLLM
from ..om.arcformer import RoPEEmbeddings
from ..om.utils import set_om_dtypes


def test_model():
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    
    num_layers = 8
    vocab_size = 50283
    
    dim_input = 1024
    num_heads = 8
    dim_hidden = 4 * dim_input
    if dim_hidden % 32 != 0:
        dim_hidden += 32 - dim_hidden % 32
    dims_key = [2 * dim_input // num_heads, dim_input // num_heads, dim_input // (2 * num_heads)]
    dims_value = [2 * dim_input // num_heads, dim_input // num_heads, dim_input // (2 * num_heads)]
    num_iters = [1, 1, 1]
    betas = [1.0 / ((2 * dims_value[0]) ** 0.5), 1.0 / (dims_value[1] ** 0.5), 1.0 / ((dims_value[2] / 2) ** 0.5)]
    attn_proj_rank = dim_input // (2 * num_heads)
    
    activation = "gelu"
    mlp_1221 = True
    segment_len = 512
    cope = False
    state_len = segment_len // 8
    
    init_ngrams = [2, 3]
    max_init_ngrams = 1 if len(init_ngrams) == 0 else max(init_ngrams)
    
    dropout = 0.1
    attn_dropout = 0.1
    attn_logit_dropout = 0.1
    diff_attn = False
    
    batch_size = 2
    num_segments = 4
    
    stacked_attn = True
    
    position_embedders = [
        RoPEEmbeddings(
            dim, 
            seq_len=segment_len + 2 * state_len, 
            num_dims=4 if stacked_attn else 3,
            device=device
        )
        for dim in dims_key
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
        cope=cope,
        position_embedders=position_embedders,
        betas=betas,
        dropout=dropout,
        diff_attn=diff_attn,
        attn_dropout=attn_dropout,
        attn_logit_dropout=attn_logit_dropout,
        attn_proj_rank=attn_proj_rank,
        init_ngrams=init_ngrams,
        mlp_1221=mlp_1221,
        stacked_attn=stacked_attn,
    )
    
    seq_len = segment_len * num_segments
    x = vocab_size * torch.rand(batch_size, seq_len + max_init_ngrams - 1)
    x = x.to(torch.long)

    model = model.to(device=device)
    x = x.to(device=device)
    
    model = set_om_dtypes(model, torch.bfloat16)
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        preds, states = model(x, next_token=False)
        assert preds.shape == (batch_size, seq_len, vocab_size)
        for state in states:
            assert state.shape == (batch_size, state_len, dim_input)
    
        pred, states = model(x, next_token=True)
        assert pred.shape == (batch_size, 1, vocab_size)
        for state in states:
            assert state.shape == (batch_size, state_len, dim_input)
    
    param_ct = count_optimized_parameters(model)
    print(f"Total optimized parameters: {param_ct:,d}")
