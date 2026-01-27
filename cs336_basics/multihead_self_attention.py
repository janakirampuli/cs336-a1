import torch
import torch.nn as nn
from torch import einsum
from typing import Optional
import einops
from .softmax import softmax
from .linear import Linear
from .rope import RotaryPositionalEmbedding
from .scaled_dot_product_attention import scaled_dot_product_attention

class CausalSelfAttention(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            device: Optional[torch.device]=None,
            dtype: Optional[torch.dtype]=None,
            theta: float = 10000.0,
            max_seq_length: int = 4096,
            use_rope: bool = True,
            causal: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_qkv = Linear(d_model, 3*d_model, device, dtype)
        self.W_o = Linear(d_model, d_model, device, dtype)

        self.use_rope = use_rope
        if self.use_rope:
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_length)

        self.causal = causal

    def forward(
            self,
            x: torch.Tensor,
            token_positions: Optional[torch.tensor]=None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        qkv = self.W_qkv(x)

        Q, K, V = torch.chunk(qkv, 3, -1)

        Q = einops.rearrange(Q, 'b s (h d) -> b h s d', h=self.num_heads)
        K = einops.rearrange(K, 'b s (h d) -> b h s d', h=self.num_heads)
        V = einops.rearrange(V, 'b s (h d) -> b h s d', h=self.num_heads)

        if self.use_rope:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        causal_mask = None
        if self.causal:
            causal_mask = torch.tril(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
            )
        attn_ouput = scaled_dot_product_attention(Q, K, V, causal_mask)
        attn_ouput = einops.rearrange(attn_ouput, 'b h s d -> b s (h d)', h=self.num_heads)
        attn_ouput = self.W_o(attn_ouput)
        return attn_ouput
