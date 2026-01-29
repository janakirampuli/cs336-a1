import torch
import torch.nn as nn
from torch import einsum
from typing import Optional
from .rmsnorm import RMSNorm
from .multihead_self_attention import CausalSelfAttention
from .swiglu import SwiGLU

class TransformerBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            theta: float=10000.0,
            max_seq_length: int=4096,
            device: Optional[torch.device]=None,
            dtype: Optional[torch.dtype]=None
    ):
        super().__init__()
        self.rms_norm1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)

        self.attention = CausalSelfAttention(d_model=d_model, num_heads=num_heads, device=device, dtype=dtype, theta=theta, max_seq_length=max_seq_length, use_rope=True, causal=True)

        self.rms_norm2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)

        self.swiglu = SwiGLU(d_model, d_ff, device, dtype)

    def forward(
            self,
            x: torch.Tensor,
            token_positions: torch.Tensor
    ) -> torch.Tensor:
        x_norm = self.rms_norm1(x)
        attn = self.attention(x_norm, token_positions)
        x = x + attn

        x_norm = self.rms_norm2(x)
        ffn = self.swiglu(x_norm)
        x = x + ffn

        return x