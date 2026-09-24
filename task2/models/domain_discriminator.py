import torch
import torch.nn as nn
from torch.autograd import Function

class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

def grad_reverse(x, alpha=1.0):
    return GradientReversalLayer.apply(x, alpha)

def get_alpha_schedule(p):
    """ Standard DANN Schedule: p is training progress [0, 1] """
    import math
    return (2. / (1. + math.exp(-10. * p))) - 1.

class DomainDiscriminator(nn.Module):
    def __init__(self, in_features=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )
        
    def forward(self, x, alpha):
        x_rev = grad_reverse(x, alpha)
        return self.net(x_rev)
