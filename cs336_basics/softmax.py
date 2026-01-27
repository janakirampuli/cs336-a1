import torch

def softmax(
        input: torch.Tensor,
        dim: int
) -> torch.Tensor:
    input_max = torch.amax(input=input, dim=dim, keepdim=True)
    input_shifted = input - input_max

    exps = torch.exp(input_shifted)
    sum_exps = torch.sum(exps, dim=dim, keepdim=True)
    return exps / sum_exps

