import torch
import torch.nn.functional as F


def score(logits_all: torch.Tensor, num_known: int) -> torch.Tensor:
    logits_known = logits_all[:, :num_known]
    logits_dummy = logits_all[:, num_known:]

    p_known = F.softmax(logits_known, dim=1).max(dim=1).values   # (N,)
    p_dummy = F.softmax(logits_dummy, dim=1).max(dim=1).values   # (N,)

    denom = p_dummy + p_known + 1e-8
    return p_dummy / denom
