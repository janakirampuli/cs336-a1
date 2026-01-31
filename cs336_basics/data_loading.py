import torch
from numpy.typing import NDArray
from typing import Tuple
import numpy as np

def get_batch(
        dataset: NDArray,
        batch_size: int,
        context_length: int,
        device: torch.device
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    
    data_len = len(dataset)
    assert data_len > context_length
    start_i = torch.randint(low=0, high=data_len - context_length, size=(batch_size, ))

    x_batch = torch.stack([torch.from_numpy((dataset[i : i + context_length]).astype(np.int64)) for i in start_i])
    y_batch = torch.stack([torch.from_numpy((dataset[i + 1 : i + context_length + 1]).astype(np.int64)) for i in start_i])

    if device == 'cuda':
        x_batch = x_batch.pin_memory().to(device, non_blocking=True)
        y_batch = y_batch.pin_memory().to(device, non_blocking=True)
    else:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

    return (x_batch, y_batch)