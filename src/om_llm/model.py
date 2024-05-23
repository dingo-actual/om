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
        init_conv: bool = False,
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
            init_conv (bool, optional): Use initial convolutions on token embeddings. Defaults to False.
            final_mlp_multiplier (int, optional): Multiplier for the hidden state dimension of the final MLP. Defaults to 1.
        """
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
        for _ in range(num_layers - 2):
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
                init_conv=False,
                mlp_multiplier=final_mlp_multiplier
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
