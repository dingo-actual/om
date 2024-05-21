from typing import List, Optional, Tuple

import torch

from ..arcformer import ARCformer, RoPEEmbeddings


class OmLLM(torch.nn.Module):
    def __init__(
        self,
        num_layers: int,
        vocab_size: int,
        dim_input: int,
        dim_hidden: int,
        dims_key: List[int],
        dims_value: List[int],
        mem_iters: int,
        num_heads: int,
        activation: str,
        segment_len: int,
        state_len: int,
        normalize: bool,
        position_embedders: List[Optional[RoPEEmbeddings]],
        dropout: float = 0.0,
        init_conv: bool = False
    ):
        super(OmLLM, self).__init__()
        
        self.embedder = torch.nn.Embedding(vocab_size, dim_input)
        for p in self.embedder.parameters():
            torch.nn.init.normal_(p, mean=0, std=(2.0 / (5 * dim_input)) ** 0.5)
        
        layers = [
            ARCformer(
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
                init_conv=init_conv
            )
        ]
        for _ in range(num_layers - 1):
            layers.append(
                ARCformer(
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
                    init_conv=False
                )
            )
        self.layers = torch.nn.ModuleList(layers)
        self.proj_out = torch.nn.Linear(dim_input, vocab_size)
        
    def get_logits(self, x: torch.Tensor, states: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        states_out = []
        if states is None:
            states = [None] * len(self.layers)
        x = self.embedder(x)
        for layer, state in zip(self.layers, states):
            x, state = layer(x, state)
            states_out.append(state)
        x = self.proj_out(x)
        
        return x, states_out
    
    def forward(self, x: torch.Tensor, states: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        logits, states = self.get_logits(x, states)
        return torch.nn.functional.sigmoid(logits), states
