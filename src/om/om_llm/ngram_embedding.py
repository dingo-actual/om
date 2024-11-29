import numpy as np
import torch


class NGramEmbedding(torch.nn.Module):
    def __init__(
        self,
        dim_input: int,
        ngram_size: int
    ):
        """Initialize the NGramEmbedding module.

        Args:
            embedding_dim (int): embedding dimension for each token
            ngram_size (int): size of ngram
            rank (int): rank of the ngram embedding projection
        """
        super(NGramEmbedding, self).__init__()

        self.embedding_dim = dim_input
        self.ngram_size = ngram_size
        
        self.ngram_down_proj = torch.nn.Linear(
            in_features=self.embedding_dim * self.ngram_size,
            out_features=self.embedding_dim // self.ngram_size,
            bias=False
        )
        self.ngram_up_proj = torch.nn.Linear(
            in_features=self.embedding_dim // self.ngram_size,
            out_features=self.embedding_dim,
            bias=False
        )
        
        self.ngram_down_proj.weight.data.normal_(mean=0.0, std=1./np.sqrt(self.embedding_dim // self.ngram_size))
        self.ngram_up_proj.weight.data.normal_(mean=0.0, std=1./np.sqrt(self.embedding_dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the NGramEmbedding module.

        Args:
            x (torch.Tensor): input tensor of shape (batch_size, seq_len, embedding_dim)

        Returns:
            torch.Tensor: output tensor of shape (batch_size, seq_len - ngram_size + 1, embedding_dim)
        """
        x_stacked = torch.concat([x[:, ix:x.size(1) - self.ngram_size + 1 + ix, :] for ix in range(self.ngram_size)], axis=-1)
        return self.ngram_up_proj(self.ngram_down_proj(x_stacked))
