import torch

n = 10
d = 4
x = torch.randn(1, n, d)

cope_emb = torch.randn(1, d, n)

w_q = torch.randn(d, d)
w_kv = torch.randn(d, d)

mask = torch.tril(torch.ones((10, 10))).log()
beta = 1. / (d ** 0.5)

xs = [x.clone()]

k = x @ w_kv
v = k.clone()

for _ in range(100):
    q = x @ w_q
    
    logits = q @ k.transpose(-2, -1)
    gates = torch.sigmoid(logits)
    pos = gates.flip(-1).cumsum(dim=-1).flip(-1)
    pos = pos.clamp(max=n - 1)
    
    pos_ceil = pos.ceil().long()
    pos_floor = pos.floor().long()
    
    logits_int = q @ cope_emb
    logits_ceil = logits_int.gather(-1, pos_ceil)
    logits_floor = logits_int.gather(-1, pos_floor)
    
    w = pos - pos_floor
    
    bias = logits_ceil * w + logits_floor * (1 - w)
    
    x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask + bias)
    
    xs.append(x.clone())
    
_ = None