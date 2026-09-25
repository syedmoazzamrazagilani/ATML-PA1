import torch
import numpy as np


def sample_inter_class_pairs(labels: torch.Tensor):
    """
    Returns two index vectors (idx_a, idx_b) such that
    labels[idx_a] != labels[idx_b] for every position.

    Tries a simple random shuffle; re-samples positions where the
    class accidentally matches.
    """
    n       = len(labels)
    device  = labels.device
    idx_a   = torch.arange(n, device=device)
    idx_b   = torch.randperm(n, device=device)

    same    = (labels[idx_a] == labels[idx_b]).nonzero(as_tuple=True)[0]
    max_try = 10
    for _ in range(max_try):
        if len(same) == 0:
            break
        swap = torch.randperm(len(same))
        idx_b[same] = idx_b[same[swap]]
        same = (labels[idx_a] == labels[idx_b]).nonzero(as_tuple=True)[0]

    return idx_a, idx_b


def manifold_mixup(h_a: torch.Tensor, h_b: torch.Tensor,
                   alpha: float = 2.0, beta: float = 2.0) -> torch.Tensor:
    """
    Mix two intermediate features:
        h̃ = λ·h_a + (1−λ)·h_b,   λ ~ Beta(alpha, beta)

    Returns h̃ with the same shape as h_a.
    """
    lam = float(np.random.beta(alpha, beta))
    return lam * h_a + (1.0 - lam) * h_b
