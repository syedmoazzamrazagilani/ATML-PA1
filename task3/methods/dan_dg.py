import torch
import torch.nn as nn
from itertools import combinations


class MMDLoss(nn.Module):
    """
    Three-kernel RBF MMD. Bandwidths are 0.5×, 1×, 2× the median pairwise
    squared distance of the combined batch (computed without gradient).
    Identical to Task 2 implementation.
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def _pdist(x, y):
        xn = (x ** 2).sum(1).view(-1, 1)
        yn = (y ** 2).sum(1).view(1, -1)
        return torch.clamp(xn + yn - 2.0 * x @ y.t(), min=0.0)

    def forward(self, feat_a, feat_b):
        combined = torch.cat([feat_a, feat_b], dim=0)
        all_dists = self._pdist(combined, combined)
        positive   = all_dists[all_dists > 0].detach()
        if positive.numel() == 0:
            return feat_a.new_tensor(0.0)
        median_bw = torch.median(positive)
        bandwidths = [0.5 * median_bw, 1.0 * median_bw, 2.0 * median_bw]

        d_xx = self._pdist(feat_a, feat_a)
        d_yy = self._pdist(feat_b, feat_b)
        d_xy = self._pdist(feat_a, feat_b)

        mmd = feat_a.new_tensor(0.0)
        for bw in bandwidths:
            mmd = mmd + (torch.exp(-d_xx / bw).mean()
                         + torch.exp(-d_yy / bw).mean()
                         - 2.0 * torch.exp(-d_xy / bw).mean())
        return mmd


class DANDGLoss(nn.Module):
    def __init__(self, lambda_dg: float = 1.0):
        super().__init__()
        self.lambda_dg   = lambda_dg
        self.cls_crit    = nn.CrossEntropyLoss()
        self.mmd         = MMDLoss()

    def forward(self, source_feats_list, source_logits, source_labels):
        """
        source_feats_list : list of tensors, one per source domain
        source_logits     : concatenated logits (all domains)
        source_labels     : concatenated labels (all domains)
        """
        cls_loss = self.cls_crit(source_logits, source_labels)

        domain_pairs = list(combinations(range(len(source_feats_list)), 2))
        if len(domain_pairs) == 0:
            return cls_loss, cls_loss, cls_loss.new_tensor(0.0)

        mmd_sum = source_feats_list[0].new_tensor(0.0)
        for i, j in domain_pairs:
            mmd_sum = mmd_sum + self.mmd(source_feats_list[i], source_feats_list[j])

        mmd_loss = (self.lambda_dg / len(domain_pairs)) * mmd_sum
        total    = cls_loss + mmd_loss
        return total, cls_loss, mmd_loss
