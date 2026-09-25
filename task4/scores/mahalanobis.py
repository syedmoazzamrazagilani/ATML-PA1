import torch
import numpy as np


def fit_params(train_features: torch.Tensor,
               train_labels:   torch.Tensor,
               num_classes:    int = 10,
               eps:            float = 1e-6):
    D = train_features.size(1)
    means = torch.zeros(num_classes, D)

    for c in range(num_classes):
        idx = (train_labels == c).nonzero(as_tuple=True)[0]
        if len(idx) > 0:
            means[c] = train_features[idx].mean(0)

    pooled_var = torch.zeros(D)
    total_n    = 0
    for c in range(num_classes):
        idx = (train_labels == c).nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            continue
        feats_c     = train_features[idx]
        centered    = feats_c - means[c].unsqueeze(0)
        pooled_var += centered.pow(2).sum(0)
        total_n    += len(idx)

    diag_cov = pooled_var / max(total_n - num_classes, 1) + eps
    inv_cov  = 1.0 / diag_cov          # (D,)

    return means, inv_cov


def score(features: torch.Tensor,
          means:    torch.Tensor,
          inv_cov:  torch.Tensor) -> torch.Tensor:
    N, D = features.shape
    C    = means.shape[0]

    # (N, C, D)
    diff    = features.unsqueeze(1) - means.unsqueeze(0)
    mah_sq  = (diff.pow(2) * inv_cov.unsqueeze(0).unsqueeze(0)).sum(dim=2)
    return mah_sq.min(dim=1).values
