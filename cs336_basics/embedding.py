import torch
import torch.nn as nn
from typing import Optional

class Embedding(nn.Module):
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            device: Optional[torch.device],
            dtype: Optional[torch.dtype]
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.W_e = nn.Parameter(
            torch.empty(size=(num_embeddings, embedding_dim), device=device, dtype=dtype)
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.W_e, 0.0, 1.0, -3.0, 3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.W_e[token_ids]