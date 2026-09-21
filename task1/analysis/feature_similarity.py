import torch
import torch.nn.functional as F

def calculate_cosine_stability(features_clean, features_transformed):
    """
    Computes the cosine stability (I_T) between clean and transformed representations.
    Inputs should be PyTorch tensors of shape (N, D).
    """
    if features_clean.shape != features_transformed.shape:
        raise ValueError("Feature matrices must have the same dimensions.")
    
    # Normalize features (L2 norm)
    f_c_norm = F.normalize(features_clean, p=2, dim=1)
    f_t_norm = F.normalize(features_transformed, p=2, dim=1)
    
    # Calculate dot product per example, then average across N
    similarities = torch.sum(f_c_norm * f_t_norm, dim=1)
    i_t = torch.mean(similarities).item()
    
    return i_t
