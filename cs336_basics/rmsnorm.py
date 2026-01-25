import torch
import torch.nn as nn
from torch import einsum

class RMSNorm(nn.Module):
    def __init__(
            self,
            d_model: int,
            eps: float = 1e-5,
            device=None,
            dtype=None
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(torch.square(x), dim=-1, keepdim=True) + self.eps)
        norm_x = x / rms
        rmsnorm =  einsum('... d, d -> ... d', norm_x, self.gain)
        return rmsnorm.to(in_dtype)