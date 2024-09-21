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
        num_heads: int,
        activation: str,
        segment_len: int,
        state_len: int,
        normalize: bool,
        cope: bool,
        position_embedders: List[Optional[RoPEEmbeddings]],
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        init_convs: List[int] = [],
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
            num_heads (int): Number of attention heads for ARCformer.
            activation (str): Activation function for MLP.
            segment_len (int): Segment length.
            state_len (int): State length (in tokens).
            normalize (bool): Normalize the inputs to ARCformer memory projections.
            cope (bool): Use CoPE for ARCformer memory.
            position_embedders (List[Optional[RoPEEmbeddings]]): Position embedders for each memory layer in ARCformer.
            dropout (float, optional): Pre/post MLP dropout. Defaults to 0.0.
            attn_dropout (float, optional): Attention dropout. Defaults to 0.0.
            init_convs (List[int], optional): Initial convolutional layer hidden sizes. Defaults to [].
            final_mlp_multiplier (int, optional): Multiplier for the hidden state dimension of the final MLP. Defaults to 1.
        """
        super(OmLLM, self).__init__()
        
        vocab_offset = 8 - dim_input % 8
        if vocab_offset == 8:
            vocab_offset = 0
        self.vocab_offset = vocab_offset
        vocab_size += vocab_offset
        self.vocab_size = vocab_size
        
        self.segment_len = segment_len
        
        self.embedder = torch.nn.Embedding(vocab_size, dim_input)
        for p in self.embedder.parameters():
            torch.nn.init.normal_(p, mean=0, std=(2. / 5. ) ** 0.5)
        
        self.init_convs = sorted(init_convs)
        if len(init_convs) > 0:
            self.convs = torch.nn.ModuleList(
                [
                    torch.nn.Conv1d(dim_input, dim_input, kernel_size=k)
                    for k in init_convs
                ]
            )
        
        layers = []
        for _ in range(num_layers - 1):
            layers.append(
                ARCformer(
                    dim_input=dim_input,
                    dim_hidden=dim_hidden,
                    dims_key=dims_key,
                    dims_value=dims_value,
                    num_heads=num_heads,
                    activation=activation,
                    segment_len=segment_len,
                    state_len=state_len,
                    attn_normalize=normalize,
                    num_layers=num_layers,
                    cope=cope,
                    position_embedders=position_embedders,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                )
            )
        layers.append(
            ARCformer(
                dim_input=dim_input,
                dim_hidden=dim_hidden,
                dims_key=dims_key,
                dims_value=dims_value,
                num_heads=num_heads,
                activation=activation,
                segment_len=segment_len,
                state_len=state_len,
                attn_normalize=normalize,
                num_layers=num_layers,
                cope=cope,
                position_embedders=position_embedders,
                dropout=dropout,
                attn_dropout=attn_dropout,
                mlp_multiplier=final_mlp_multiplier
            )
        )
        self.init_states = [
            layer.attn.init_state for layer in layers
        ]
        self.layers = torch.nn.ModuleList(layers)
        
        self.proj_out = torch.nn.Linear(dim_input * final_mlp_multiplier, vocab_size)
        
    def forward(self, 
                x: torch.Tensor, 
                states: List[torch.Tensor] = [], 
                offset: int = 0,
                next_token: bool = False
                ) -> Tuple[torch.Tensor, List[torch.Tensor], int]:
        """Forward model pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len)
            states (List[torch.Tensor], optional): State tensors for each ARCformer block. Defaults to [].
            offset (int, optional): Input location offset. Defaults to 0.
            next_token (bool, optional): Whether to generate predictions for only the final token. Defaults to False.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor], int]: 
              - Logits for the next token at each position in the input.
              - State tensors for each ARCformer block.
              - Input location offset.
        """
        batch_size, seq_len = x.shape
        
        if len(states) == 0:
            states = [state.repeat(batch_size, 1, 1) for state in self.init_states]
        
        num_segments, rem = divmod(seq_len, self.segment_len)
        if rem > 0:
            num_segments += 1
        
        out = []
        
        if len(self.init_convs) > 0:
            drop_num = max(self.init_convs) - 1
        else:
            drop_num = 0
        
        for segment_num in range(num_segments):
            ix_lo = segment_num * self.segment_len
            ix_hi = min(ix_lo + self.segment_len, seq_len)
            seg_len_actual = ix_hi - ix_lo
            
            if len(self.init_convs) > 0 and segment_num > 0:
                conv_offset_lo = max(self.init_convs) - 1
            else:
                conv_offset_lo = 0

            x_seg = x[:, ix_lo - conv_offset_lo:ix_hi]
            x_seg = self.embedder(x_seg)
            
            if len(self.init_convs) > 0:
                x_seg_convs = [
                    conv(x_seg.transpose(1, 2)).transpose(1, 2) for conv in self.convs
                ]
                for k, x_seg_conv in zip(self.init_convs, x_seg_convs):
                    x_seg[:, k-1:, :] += x_seg_conv
                    
                x_seg = x_seg[:, drop_num:, :]
            
            states_next = []
        
            for layer, state in zip(self.layers, states):
                x_seg, state_next = layer(x_seg, state, offset)
                states_next.append(state_next)
            
            states = states_next
            offset += seg_len_actual
            
            if next_token:
                if segment_num == num_segments - 1:
                    out = self.proj_out(x_seg[:, -1, :])
            else:
                out.append(self.proj_out(x_seg))
        
        if not next_token:
            out = torch.cat(out, dim=1)
            
        if self.vocab_offset > 0:
            out[:, :, -self.vocab_offset:] = -float("inf")
        
        return out, states, offset
