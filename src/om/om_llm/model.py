from typing import List, Optional, Tuple

import torch

from ..arcformer import ARCformer, RoPEEmbeddings
from .ngram_embedding import NGramEmbedding

class OmLLM(torch.nn.Module):
    def __init__(
        self,
        num_layers: int,
        vocab_size: int,
        dim_input: int,
        dim_hidden: int,
        dims_key: List[int],
        dims_value: List[int],
        num_iters: List[int],
        num_heads: int,
        activation: str,
        segment_len: int,
        state_len: int,
        pad_id: int,
        cope: bool,
        position_embedders: List[Optional[RoPEEmbeddings]],
        scaling_factors: List[Optional[float]],
        dropout: float = 0.0,
        diff_attn: bool = False,
        attn_dropout: float = 0.0,
        attn_logit_dropout: float = 0.0,
        attn_proj_rank: int = -1,
        init_ngrams: List[int] = [],
        mlp_1221: bool = False,
        stacked_attn: bool = True,
    ):
        """Initialize the model.

        Args:
            num_layers (int): Number of ARCformer layers.
            vocab_size (int): Vocabulary size.
            dim_input (int): Input dimension.
            dim_hidden (int): Hidden dimension for MLP.
            dims_key (List[int]): Key dimensions for ARCformer.
            dims_value (List[int]): Value dimensions for ARCformer.
            num_iters (List[int]): Number of iterations for ARCformer memory.
            num_heads (int): Number of attention heads for ARCformer.
            activation (str): Activation function for MLP.
            segment_len (int): Segment length.
            state_len (int): State length (in tokens).
            pad_id (int): Padding token id.
            cope (bool): Use CoPE for ARCformer memory.
            position_embedders (List[Optional[RoPEEmbeddings]]): Position embedders for each memory layer in ARCformer.
            scaling_factors (List[Optional[float]]): Betas for Hopfield memory / scaling factor for SDP attention.
            dropout (float, optional): Pre/post MLP dropout. Defaults to 0.0.
            attn_dropout (float, optional): Dropout rate for attention outputs. Defaults to 0.0.
            attn_logit_dropout (float, optional): Dropout rate for attention logits. Defaults to 0.0.
            attn_proj_rank (int, optional): Rank of the attention projection. If -1 will use min(dims_value). Defaults to -1.
            init_ngrams (List[int], optional): Initial n-gram projection layer hidden sizes. Defaults to [] (unigrams).
            init_ngrams_ranks (List[int], optional): Initial n-gram projection layer ranks. Defaults to [] (unigrams).
            mlp_1221 (bool, optional): Use 1-2-2-1 MLP architecture. Defaults to False.
            stacked_attn (bool, optional): Use stacked attention. Defaults to True.
        """
        super(OmLLM, self).__init__()
        
        vocab_offset = 32 - dim_input % 32
        if vocab_offset == 32:
            vocab_offset = 0
        self.vocab_offset = vocab_offset
        vocab_size += vocab_offset
        self.vocab_size = vocab_size
        
        self.segment_len = segment_len
        self.pad_id = pad_id
        
        self.embedder = torch.nn.Embedding(vocab_size, dim_input)
        for p in self.embedder.parameters():
            torch.nn.init.normal_(p, mean=0, std=(2. / 5. ) ** 0.5)
        
        self.init_ngram_lengths = sorted(init_ngrams)
        if len(init_ngrams) > 0:
            self.ngram_embs = torch.nn.ModuleList(
                [
                    NGramEmbedding(
                        dim_input=dim_input,
                        ngram_size=k
                        
                    )
                    for k in init_ngrams
                ]
            )
            self.max_ngram = max(init_ngrams)
        else:
            self.max_ngram = 1
        
        layers = []
        for ix in range(num_layers):
            layers.append(
                ARCformer(
                    dim_input=dim_input,
                    dim_hidden=dim_hidden,
                    dims_key=dims_key,
                    dims_value=dims_value,
                    num_iters=num_iters,
                    num_heads=num_heads,
                    activation=activation,
                    segment_len=segment_len,
                    state_len=state_len,
                    num_layers=num_layers,
                    layer_num=ix,
                    cope=cope,
                    position_embedders=position_embedders,
                    scaling_factors=scaling_factors,
                    dropout=dropout,
                    diff_attn=diff_attn,
                    attn_dropout=attn_dropout,
                    attn_logit_dropout=attn_logit_dropout,
                    attn_proj_rank=attn_proj_rank,
                    mlp_1221=mlp_1221,
                    stacked_attn=stacked_attn,
                )
            )
        self.layers = torch.nn.ModuleList(layers)
        
        self.proj_out = torch.nn.Linear(dim_input, vocab_size, bias=False)
        
    def forward(self, 
                x: torch.Tensor, 
                states: List[torch.Tensor] = [], 
                next_token: bool = False
                ) -> Tuple[torch.Tensor, List[torch.Tensor], int]:
        """Forward model pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len)
            states (List[torch.Tensor], optional): State tensors for each ARCformer block. Defaults to [].
            next_token (bool, optional): Whether to generate predictions for only the final token. Defaults to False.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor], int]: 
              - Logits for the next token at each position in the input.
              - State tensors for each ARCformer block.
        """
        _, seq_len = x.shape
        drop_num = self.max_ngram - 1
        seq_len = seq_len - drop_num
        
        if len(states) == 0:
            states = [layer.attn.init_state for layer in self.layers]
        
        num_segments, rem = divmod(seq_len, self.segment_len)
        if rem > 0:
            num_segments += 1
        
        out = []
        
        ix_hi = 0
        
        for segment_num in range(num_segments):
            ix_lo = ix_hi
            ix_hi = min(ix_lo + self.segment_len, seq_len + drop_num)
            
            if len(self.init_ngram_lengths) > 0:
                if segment_num > 0:
                    ngram_offset_lo = max(self.init_ngram_lengths) - 1
                    ngram_offset_hi = 0
                else:
                    ngram_offset_lo = 0
                    ngram_offset_hi = max(self.init_ngram_lengths) - 1
            else:
                ngram_offset_lo = 0
                ngram_offset_hi = 0
                
            ix_lo = ix_lo - ngram_offset_lo
            ix_hi = ix_hi + ngram_offset_hi

            x_seg = x[:, ix_lo:ix_hi]
            
            x_seg_emb = self.embedder(x_seg)
            x_seg_mask = (x_seg != self.pad_id).unsqueeze(-1).to(x_seg.device).to(torch.long).to(x_seg_emb.dtype)
            x_seg_emb = x_seg_emb * x_seg_mask
            
            x_seg_emb_ = x_seg_emb.clone()
            
            if len(self.init_ngram_lengths) > 0:
                x_seg_ngram_embs = [
                    ngram_emb(x_seg_emb) for ngram_emb in self.ngram_embs
                ]
                for k, x_seg_ngram_emb in zip(self.init_ngram_lengths, x_seg_ngram_embs):
                    x_seg_emb_[:, k-1:, :] += x_seg_ngram_emb
                    
                x_seg_next = x_seg_emb_[:, drop_num:, :]
            
            skip_update_states = next_token and (segment_num == num_segments - 1) and (rem > 0)
            
            if not skip_update_states:
                states_next = []
        
            for layer, state in zip(self.layers, states):
                x_seg_next, state_next = layer(x_seg_next, state)
                if not skip_update_states:
                    states_next.append(state_next)
            
            if not skip_update_states:
                states = states_next
            
            if next_token:
                if segment_num == num_segments - 1:
                    out = self.proj_out(x_seg_next[:, -1, :].unsqueeze(1))
            else:
                out.append(self.proj_out(x_seg_next))
        
        if not next_token:
            out = torch.cat(out, dim=1)
            
        if self.vocab_offset > 0:
            out[:, :, -self.vocab_offset:] = -float("inf")
        
        return out, states