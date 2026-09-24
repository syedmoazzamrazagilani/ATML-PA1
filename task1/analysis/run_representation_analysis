import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import umap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_cosine_stability(f_clean, f_trans):
    """
    Computes I_T = (1/N) * sum( (f(x)^T f(T(x))) / (||f(x)||_2 * ||f(T(x))||_2) )
    """
    sim = F.cosine_similarity(f_clean, f_trans, dim=1)
    return float(sim.mean().item())

def plot_joint_umap(f_clean, f_trans, labels, save_path, title):
    """
    Fits ONE 2D UMAP projection to combined clean and transformed features.
    """
    n = len(f_clean)
    combined = torch.cat([f_clean, f_trans], dim=0).cpu().numpy()
    
    reducer = umap.UMAP(n_components=2, random_state=6304)
    embeddings = reducer.fit_transform(combined)
    
    emb_clean = embeddings[:n]
    emb_trans = embeddings[n:]
    labels_np = labels.cpu().numpy()
    
    plt.figure(figsize=(9, 7))
    plt.scatter(emb_clean[:, 0], emb_clean[:, 1], c=labels_np, cmap='tab10', 
                marker='o', alpha=0.7, edgecolors='none', label='Clean')
    plt.scatter(emb_trans[:, 0], emb_trans[:, 1], c=labels_np, cmap='tab10', 
                marker='x', alpha=0.8, label='Transformed (Patch Shuffle)')
    
    plt.title(title)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def run_task1_representations(models_dict, clean_loader, trans_loaders, test_labels):
    """
    trans_loaders: dict of {'grayscale': loader, 'cue_conflict': loader, 
                            'translation': loader, 'patch_shuffle': loader}
    """
    results_dir = "task1/results"
    os.makedirs(results_dir, exist_ok=True)
    
    cosine_stability = {}

    for model_name, backbone in models_dict.items():
        backbone.eval()
        backbone.to(device)
        cosine_stability[model_name] = {}
        
        # 1. Extract clean features
        clean_feats = []
        with torch.no_grad():
            for x, _ in clean_loader:
                f = backbone(x.to(device))
                clean_feats.append(f.squeeze())
        clean_feats = torch.cat(clean_feats)

        # 2. Compute I_T for each intervention
        for t_name, t_loader in trans_loaders.items():
            t_feats = []
            with torch.no_grad():
                for x, _ in t_loader:
                    f = backbone(x.to(device))
                    t_feats.append(f.squeeze())
            t_feats = torch.cat(t_feats)
            
            i_t = compute_cosine_stability(clean_feats, t_feats)
            cosine_stability[model_name][t_name] = i_t
            
            # 3. Generate required UMAP plot for patch shuffle
            if t_name == "patch_shuffle":
                plot_joint_umap(
                    clean_feats, t_feats, test_labels,
                    save_path=os.path.join(results_dir, f"umap_{model_name}.png"),
                    title=f"{model_name.upper()} - Clean vs. Patch Shuffled UMAP"
                )

    with open(os.path.join(results_dir, "cosine_stability.json"), "w") as f:
        json.dump(cosine_stability, f, indent=2)
    print("Representation analysis complete. Results saved in task1/results/.")

if __name__ == "__main__":
    print("Connect clean_loader, trans_loaders, and models to run representation analysis.")
