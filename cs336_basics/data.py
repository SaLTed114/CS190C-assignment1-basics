from __future__ import annotations

import numpy as np
import torch

def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_size: int,
    device: torch.device | str
) -> tuple[torch.Tensor, torch.Tensor]:
    if dataset.ndim != 1:
        raise ValueError(f"Expected dataset to be a 1D array, but got shape {dataset.shape}")
    if len(dataset) < context_size + 1:
        raise ValueError(f"Dataset is too small to create a batch with context_size {context_size}")
    
    max_start = len(dataset) - context_size - 1
    starts = np.random.randint(0, max_start + 1, size=batch_size)

    x = np.stack([dataset[start:start + context_size] for start in starts])
    y = np.stack([dataset[start + 1:start + 1 + context_size] for start in starts])

    x = torch.tensor(x, dtype=torch.long, device=device)
    y = torch.tensor(y, dtype=torch.long, device=device)
    return x, y