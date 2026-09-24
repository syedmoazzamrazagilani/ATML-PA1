import torch
import torch.nn as nn

def multilinear_conditioning(features, logits):
    """
    Computes g(x) = vec(f ⊗ p)
    features: (Batch, 512)
    logits: (Batch, 7)
    """
    p = torch.softmax(logits, dim=1)
    
    outer_product = torch.bmm(features.unsqueeze(2), p.unsqueeze(1))
    
    g_x = outer_product.view(features.size(0), -1)
    
    return g_x
