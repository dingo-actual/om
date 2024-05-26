import torch

from ..arcformer import RoPEEmbeddings
from ..arcformer.util import count_optimized_parameters
from ..om_llm import OmLLM


def test_model():
    num_layers = 8
    vocab_size = 50284
    
    dim_input = 1024
    num_heads = 8
    dim_hidden = int(8 * dim_input / 3)
    dims_key = [dim_input // num_heads, 2 * dim_input // num_heads, dim_input // num_heads]
    dims_value = [dim_input // num_heads, 2 * dim_input // num_heads, dim_input // num_heads]
    mem_iters = [1, 3, 1]
    
    activation = "gelu"
    segment_len = 1024
    normalize = True
    state_len = segment_len // 8
    
    dropout = 0.1
    
    batch_size = 48
    num_segments = 1
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
    
    position_embedders = [
        position_embedder_1,
        position_embedder_2,
        position_embedder_1
    ]
    
    model = OmLLM(
        num_layers=num_layers,
        vocab_size=vocab_size,
        dim_input=dim_input,
        dim_hidden=dim_hidden,
        dims_key=dims_key,
        dims_value=dims_value,
        mem_iters=mem_iters,
        num_heads=num_heads,
        activation=activation,
        segment_len=segment_len,
        state_len=state_len,
        normalize=normalize,
        position_embedders=position_embedders,
        dropout=dropout,
        final_mlp_multiplier=8
    )
    
    seq_len = segment_len * num_segments
    x = vocab_size * torch.rand(batch_size, seq_len)
    x = x.to(torch.long)

    if torch.cuda.is_available():
        model = model.to("cuda:0")
        x = x.to("cuda:0")
    
    model.eval()  # Set the model to evaluation mode
    model = model.to(torch.bfloat16)
    with torch.no_grad():
        preds, states = model(x, next_token=next_token)

    
    if next_token:
        assert preds.shape == (batch_size, vocab_size)
    else:
        assert preds.shape == (batch_size, seq_len, vocab_size)
    for state in states:
        assert state.shape == (batch_size, state_len, dim_input)
    
    param_ct = count_optimized_parameters(model)
    print(f"Total optimized parameters: {param_ct:,d}")
