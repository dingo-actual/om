from schedulefree import AdamWScheduleFree
import torch

from ..om.om_llm import OmLLM
from ..om.arcformer import RoPEEmbeddings
from ..om.utils import set_om_dtypes
from ..om.arcformer.util import check_if_linux


def test_model_training():
    linux = check_if_linux()
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    
    num_layers = 4
    vocab_size = 512
    
    dim_input = 256
    num_heads = 8
    dim_hidden = 4 * dim_input
    if dim_hidden % 32 != 0:
        dim_hidden += 32 - dim_hidden % 32
    dims_key = [2 * dim_input // num_heads, dim_input // num_heads, dim_input // (2 * num_heads)]
    dims_value = [2 * dim_input // num_heads, dim_input // num_heads, dim_input // (2 * num_heads)]
    num_iters = [1, 1, 1]
    scaling_factors = [0.5 / (dims_value[0] ** 0.5), 1.0 / (dims_value[1] ** 0.5), 1.5 / (dims_value[0] ** 0.5)]
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
    
    stacked_attn = linux
    
    position_embedders = [
        RoPEEmbeddings(
            dim, 
            seq_len=segment_len + 2 * state_len, 
            num_dims=4 if stacked_attn else 3
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
        scaling_factors=scaling_factors,
        dropout=dropout,
        diff_attn=diff_attn,
        attn_dropout=attn_dropout,
        attn_logit_dropout=attn_logit_dropout,
        attn_proj_rank=attn_proj_rank,
        init_ngrams=init_ngrams,
        mlp_1221=mlp_1221,
        stacked_attn=stacked_attn
    )
    
    seq_len = segment_len * num_segments

    model = model.to(device=device)
    model = set_om_dtypes(model, torch.bfloat16)
    
    wd_ignore_groups = ["bias", "norm"]
    wd_params = [p for n, p in model.named_parameters() if not any(nd in n for nd in wd_ignore_groups)]
    no_wd_params = [p for n, p in model.named_parameters() if any(nd in n for nd in wd_ignore_groups)]
    
    param_groups = [
        {"params": wd_params, "weight_decay": 0.1},
        {"params": no_wd_params, "weight_decay": 0.0}
    ]
    
    if linux:
        optimizer = AdamWScheduleFree(param_groups, lr=1e-3, warmup_steps=10, foreach=False)
    else:
        optimizer = torch.optim.AdamW(param_groups, lr=1e-3, betas=(0.9, 0.95))
        
    loss_fn = torch.nn.CrossEntropyLoss()
    
    model = model.train()
    
    if linux:
        optimizer.train()
    
    for ix in range(20):
        optimizer.zero_grad()
        
        x = vocab_size * torch.rand(batch_size, seq_len + max_init_ngrams)
        x = x.to(torch.long)
        x = x.to(device=device)
        
        inputs = x[:, :-1]
        targets = x[:, max_init_ngrams:]
        
        logits, _ = model(inputs, next_token=False)
        
        loss = loss_fn(logits.transpose(-1, -2), targets)
        
        loss.backward()
        
        if any([torch.all(p.grad == 0) for p in model.parameters() if p.requires_grad and p.grad is not None]):
            raise RuntimeError("Gradient is zero")
        
        optimizer.step()
        print(f"Training step {ix + 1} complete")
        
    print("Training successful")
