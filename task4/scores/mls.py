import torch


def score(logits: torch.Tensor) -> torch.Tensor:
    """Returns unknownness score (higher = more novel). Shape: (N,)"""
    return -logits.max(dim=1).values
