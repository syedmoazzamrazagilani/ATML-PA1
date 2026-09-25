import torch
import torch.nn.functional as F


def score(logits: torch.Tensor) -> torch.Tensor:
    """Returns unknownness score (higher = more novel). Shape: (N,)"""
    probs = F.softmax(logits, dim=1)
    return 1.0 - probs.max(dim=1).values
