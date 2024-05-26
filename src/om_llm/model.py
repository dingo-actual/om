from copy import deepcopy
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
        mem_iters: List[int],
        num_heads: int,
        activation: str,
        segment_len: int,
        state_len: int,
        normalize: bool,
        position_embedders: List[Optional[RoPEEmbeddings]],
        dropout: float = 0.0,
        final_mlp_multiplier: int = 1,
    ):
        """Initialize the model.

        Args:
            num_layers (int): Number of ARCformer layers.
            vocab_size (int): Vocabulary size.
            dim_input (int): Input dimension.
            dim_hidden (int): Hidden dimension for MLP.
            dims_key (List[int]): Key dimensions for ARCformer.
            dims_value (List[int]): Value dimensions for ARCformer.
            mem_iters (List[int]): Number of memory iterations for each memory layer in ARCformer.
            num_heads (int): Number of attention heads for ARCformer.
            activation (str): Activation function for MLP.
            segment_len (int): Segment length.
            state_len (int): State length (in tokens).
            normalize (bool): Normalize the inputs to ARCformer memory projections.
            position_embedders (List[Optional[RoPEEmbeddings]]): Position embedders for each memory layer in ARCformer.
            dropout (float, optional): MLP dropout. Defaults to 0.0.
            final_mlp_multiplier (int, optional): Multiplier for the hidden state dimension of the final MLP. Defaults to 1.
        """
        super(OmLLM, self).__init__()
        
        self.segment_len = segment_len
        
        self.embedder = torch.nn.Embedding(vocab_size, dim_input)
        for p in self.embedder.parameters():
            torch.nn.init.normal_(p, mean=0, std=(2.0 / (5 * dim_input)) ** 0.5)
        
        layers = []
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
                    dropout=dropout
                )
            )
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
                mlp_multiplier=final_mlp_multiplier
            )
        )
        self.init_states = [
            layer.attn.init_state for layer in layers
        ]
        self.layers = torch.nn.ModuleList(layers)
        
        self.proj_out = torch.nn.Linear(dim_input, vocab_size)
        
    def get_logits(self, 
                   x: torch.Tensor, 
                   states: List[torch.Tensor] = [], 
                   offset: int = 0,
                   next_token: bool = False
                   ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        batch_size, seq_len = x.shape
        
        if len(states) == 0:
            states = [state.repeat(batch_size, 1, 1) for state in self.init_states]
        
        num_segments, rem = divmod(seq_len, self.segment_len)
        if rem > 0:
            num_segments += 1
        
        out = []
        
        for segment_num in range(num_segments):
            ix_lo = segment_num * self.segment_len
            ix_hi = min(ix_lo + self.segment_len, seq_len)
            
            x_seg = x[:, ix_lo:ix_hi]
            x_seg = self.embedder(x_seg)
            
            states_next = []
        
            for layer, state in zip(self.layers, states):
                x_seg, state_next = layer(x_seg, state, offset)
                states_next.append(state_next)
            
            states = states_next
            offset += self.segment_len
            
            if next_token:
                if segment_num == num_segments - 1:
                    out = self.proj_out(x_seg[:, -1, :])
            else:
                out.append(self.proj_out(x_seg))
        
        if not next_token:
            out = torch.cat(out, dim=1)
        
        return out, states
    
    def forward(self, 
                x: torch.Tensor, 
                states: List[torch.Tensor] = [],
                offset: int = 0,
                next_token: bool = False
                ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        logits, states = self.get_logits(x, states, offset, next_token)
        return torch.nn.functional.sigmoid(logits), states
