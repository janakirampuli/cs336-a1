import torch
from typing import Iterable

def gradient_clipping(
        params: Iterable[torch.nn.Parameter],
        M: float
):
    params_with_grad = [p for p in params if p.grad is not None]

    if not params_with_grad:
        return
    
    total_norm = torch.sqrt(sum(p.grad.norm(2)**2 for p in params_with_grad))

    if total_norm > M:
        for p in params_with_grad:
            p.grad.detach().mul_(M / (total_norm + 1e-6))
