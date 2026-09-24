import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import open_clip
from torchvision.models import resnet50, ResNet50_Weights, vit_b_16, ViT_B_16_Weights

# Fix seed 6304 for reproducibility across projections
torch.manual_seed(6304)
np.random.seed(6304)

def get_backbones(device):
    """Loads frozen backbones adapted to output their representation vectors."""
    # ResNet-50: Replace fully connected layer to extract the 2048-d GAP feature
    resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    resnet.fc = nn.Identity()
    resnet.eval().to(device)

    # ViT-B/16: Replace classification head to extract the 768-d class token
    vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    vit.heads = nn.Identity()
    vit.eval().to(device)

    # OpenCLIP ViT-B/32: Native encode_image outputs 512-d normalized embeddings
    clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    clip_model.eval().to(device)

    return {"ResNet-50": resnet, "ViT-B/16": vit, "OpenCLIP": clip_model}

def extract_features(model, model_name, loader, device):
    """Extracts representations for all images in a dataloader."""
    features, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            if model_name == "OpenCLIP":
                # CLIP embeddings should be explicitly normalized
                feat = model.encode_image(x)
                feat = F.normalize(feat, dim=-1)
            else:
                feat = model(x)
            features.append(feat.cpu())
            labels.append(y)
    
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)

def compute_cosine_stability(feat_clean, feat_trans):
    """Computes I_T as specified in Task 1."""
    # Normalize features for cosine similarity
    feat_clean = F.normalize(feat_clean, p=2, dim=1)
    feat_trans = F.normalize(feat_trans, p=2, dim=1)
    # Compute batched similarity and return the mean
    similarities = (feat_clean * feat_trans).sum(dim=1)
    return similarities.mean().item()

def plot_tsne(feat_clean, feat_trans, labels, title, save_path):
    """Fits one 2D projection to combined features and plots them."""
    N = len(feat_clean)
    
    # Combine features so they share the exact same projection space
    combined_feats = torch.cat([feat_clean, feat_trans], dim=0).numpy()
    
    # Fit t-SNE
    tsne = TSNE(n_components=2, random_state=6304, init='pca', learning_rate='auto')
    projected = tsne.fit_transform(combined_feats)
    
    proj_clean = projected[:N]
    proj_trans = projected[N:]
    labels_np = labels.numpy()
    
    plt.figure(figsize=(10, 8))
    unique_classes = np.unique(labels_np)
    cmap = plt.cm.get_cmap('tab10', len(unique_classes))
    
    for cls_idx in unique_classes:
        idx = labels_np == cls_idx
        color = cmap(cls_idx % 10)
        
        # Color identifies ground-truth class, marker distinguishes condition
        plt.scatter(proj_clean[idx, 0], proj_clean[idx, 1], 
                    color=color, marker='o', label=f'Clean (Class {cls_idx})', alpha=0.7)
        plt.scatter(proj_trans[idx, 0], proj_trans[idx, 1], 
                    color=color, marker='x', label=f'Transformed (Class {cls_idx})', alpha=0.7)
    
    plt.title(f't-SNE: {title}')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("task1/results", exist_ok=True)
    
    models = get_backbones(device)
    from task1.data.make_subset import get_loaders
    clean_loader = get_loaders("clean", shuffle=False)
    trans_loaders = {
         "Grayscale": get_loaders("grayscale", shuffle=False),
         "Translation": get_loaders("translation", shuffle=False),
         "Patch Shuffle": get_loaders("patch_shuffle", shuffle=False),
         "Cue Conflict": get_loaders("cue_conflict", shuffle=False)
    }
    
    print("Extracting clean features...")
    clean_features = {}
    for name, model in models.items():
        feat, labels = extract_features(model, name, clean_loader, device)
        clean_features[name] = (feat, labels)
        
    print("Running intervention analysis...")
    for trans_name, trans_loader in trans_loaders.items():
        print(f"--- Intervention: {trans_name} ---")
        for model_name, model in models.items():
            feat_clean, labels = clean_features[model_name]
            feat_trans, _ = extract_features(model, model_name, trans_loader, device)
            
            # Compute and print cosine stability
            stability = compute_cosine_stability(feat_clean, feat_trans)
            print(f"{model_name} Cosine Stability: {stability:.4f}")
            
            # Generate and save projection
            save_file = f"task1/results/tsne_{model_name.replace('/', '')}_{trans_name.replace(' ', '')}.png"
            plot_tsne(feat_clean, feat_trans, labels, f"{model_name} - {trans_name}", save_file)
            print(f"Saved plot to {save_file}")

if __name__ == "__main__":
    main()
