import torch
import torch.nn.functional as F

def compute_cosine_stability(backbone, clean_loader, transformed_loader):
    """
    Computes I_T = (1/N) * sum( (f(x)^T f(T(x))) / (||f(x)|| * ||f(T(x))||) )
    """
    backbone.eval()
    clean_features, trans_features = [], []
    
    with torch.no_grad():
        for (clean_img, _), (trans_img, _) in zip(clean_loader, transformed_loader):
            f_clean = backbone(clean_img.cuda())
            f_trans = backbone(trans_img.cuda())
            
            clean_features.append(f_clean.cpu())
            trans_features.append(f_trans.cpu())
            
    clean_features = torch.cat(clean_features)
    trans_features = torch.cat(trans_features)
    
    stability_scores = F.cosine_similarity(clean_features, trans_features, dim=1)
    i_t_mean = stability_scores.mean().item()
    
    return i_t_mean
