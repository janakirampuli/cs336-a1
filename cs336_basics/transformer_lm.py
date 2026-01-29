import torch
import torch.nn as nn
from typing import Optional
from .embedding import Embedding
from .transformer_block import TransformerBlock
from .rmsnorm import RMSNorm
from .linear import Linear

class TransformerLM(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            context_length: int,
            num_layers:  int,
            d_model: int,
            num_heads: int,
            d_ff: int,
            theta: float=10000.0,
            device: Optional[torch.device]=None,
            dtype: Optional[torch.dtype]=None
    ):
        super().__init__()

        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, theta, context_length, device, dtype)
            for _ in range(num_layers)
        ])
        self.rms_norm = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.linear = Linear(d_model, vocab_size, device, dtype)

    def forward(
            self,
            input: torch.Tensor
    ) -> torch.Tensor:
        seq_len = input.shape[1]
        token_positions = torch.arange(seq_len, device=input.device)

        x = self.token_embeddings(input)

        for layer in self.layers:
            x = layer(x, token_positions)

        x = self.rms_norm(x)
        output = self.linear(x)
        return output