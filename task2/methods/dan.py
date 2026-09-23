import torch
import torch.nn as nn

class MMDLoss(nn.Module):
    def __init__(self):
        super(MMDLoss, self).__init__()

    def compute_pairwise_distances(self, x, y):
        # Computes the squared Euclidean distance matrix
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
        return torch.clamp(dist, min=0.0)

    def gaussian_kernel(self, dist, bandwidth):
        return torch.exp(-dist / bandwidth)

    def forward(self, source_features, target_features):
        batch_size = source_features.size(0)
        
        combined = torch.cat([source_features, target_features], dim=0)
        pairwise_dists = self.compute_pairwise_distances(combined, combined)
        
        median_dist = torch.median(pairwise_dists[pairwise_dists > 0].detach())
        
        bandwidths = [0.5 * median_dist, 1.0 * median_dist, 2.0 * median_dist]
        
        xx_dists = self.compute_pairwise_distances(source_features, source_features)
        yy_dists = self.compute_pairwise_distances(target_features, target_features)
        xy_dists = self.compute_pairwise_distances(source_features, target_features)
        
        mmd_loss = 0.0
        for bw in bandwidths:
            k_xx = self.gaussian_kernel(xx_dists, bw).mean()
            k_yy = self.gaussian_kernel(yy_dists, bw).mean()
            k_xy = self.gaussian_kernel(xy_dists, bw).mean()
            # MMD^2 = E[k(x,x)] + E[k(y,y)] - 2E[k(x,y)]
            mmd_loss += k_xx + k_yy - 2 * k_xy
            
        return mmd_loss
