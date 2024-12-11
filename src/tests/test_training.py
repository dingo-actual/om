from heavyball import PrecondScheduleSFPaLMSOAP, utils
import torch
from xformers.components.positional_embedding import RotaryEmbedding

from ..om.om_llm import OmLLM
from ..om.utils import set_om_dtypes


def test_model_training():
    num_layers = 4
    vocab_size = 1024
    
    dim_input = 256
    num_heads = 8
    dim_hidden = 4 * dim_input
    if dim_hidden % 32 != 0:
        dim_hidden += 32 - dim_hidden % 32
    dims_key = [2 * dim_input // num_heads, dim_input // num_heads, dim_input // (2 * num_heads)]
    dims_value = [2 * dim_input // num_heads, dim_input // num_heads, dim_input // (2 * num_heads)]
    num_iters = [1, 1, 1]
    betas = [0.5 / (dims_value[0] ** 0.5), 1.0 / (dims_value[1] ** 0.5), 1.5 / (dims_value[0] ** 0.5)]
    final_mlp_multiplier = 1
    attn_proj_rank = dim_input // (2 * num_heads)
    
    activation = "gelu"
    mlp_1221 = True
    segment_len = 128
    cope = True
    state_len = segment_len // 8
    
    init_ngrams = [2, 3]
    max_init_ngrams = 1 if len(init_ngrams) == 0 else max(init_ngrams)
    
    dropout = 0.1
    attn_dropout = 0.1
    attn_logit_dropout = 0.0
    diff_attn = False
    
    batch_size = 4
    num_segments = 4
    # position_embedders = [RotaryEmbedding(dim) for dim in dims_key]
    position_embedders = [None, None, None]
    
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
        final_mlp_multiplier=final_mlp_multiplier,
        mlp_1221=mlp_1221,
    )
    
    seq_len = segment_len * num_segments

    if torch.cuda.is_available():
        model = model.to("cuda:0")
    
    model = set_om_dtypes(model, torch.bfloat16)
    
    wd_ignore_groups = ["bias", "norm"]
    wd_params = [p for n, p in model.named_parameters() if not any(nd in n for nd in wd_ignore_groups)]
    no_wd_params = [p for n, p in model.named_parameters() if any(nd in n for nd in wd_ignore_groups)]
    
    param_groups = [
        {"params": wd_params, "weight_decay": 0.1},
        {"params": no_wd_params, "weight_decay": 0.0}
    ]
    
    utils.set_torch()
    
    optimizer = PrecondScheduleSFPaLMSOAP(param_groups, lr=1e-3, warmup_steps=10)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    model = model.train()
    optimizer.train()
    
    for ix in range(20):
        optimizer.zero_grad()
        
        x = vocab_size * torch.rand(batch_size, seq_len + max_init_ngrams - 1)
        x = x.to(torch.long)
        
        if torch.cuda.is_available():
            x = x.to("cuda:0")
        
        inputs = x[:, :-1]
        targets = x[:, max_init_ngrams:]
        
        logits, _ = model(inputs, next_token=False)
        
        loss = loss_fn(logits.transpose(-1, -2), targets)
        
        loss.backward()
        optimizer.step()
        print(f"Training step {ix + 1} complete")
        
    print("Training successful")
