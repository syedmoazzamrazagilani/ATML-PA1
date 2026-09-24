import torch
import numpy as np
import matplotlib.pyplot as plt
import umap

def visualize_features(clean_features, transformed_features, labels, title="UMAP Projection"):
    """
    Fits one 2D projection to combined features so both conditions appear in the same space.
    Marker style distinguishes clean from transformed.
    """
    combined_features = np.vstack((clean_features, transformed_features))
    
    print("Fitting UMAP... this may take a moment.")
    reducer = umap.UMAP(n_components=2, random_state=6304)
    projected = reducer.fit_transform(combined_features)
    
    N = len(clean_features)
    proj_clean = projected[:N]
    proj_transformed = projected[N:]
    
    plt.figure(figsize=(10, 8))
    
    scatter_clean = plt.scatter(proj_clean[:, 0], proj_clean[:, 1], 
                                c=labels, cmap='tab10', marker='o', alpha=0.7, label='Clean')
    
    scatter_trans = plt.scatter(proj_transformed[:, 0], proj_transformed[:, 1], 
                                c=labels, cmap='tab10', marker='x', alpha=0.7, label='Transformed')
    
    plt.title(title)
    plt.legend(handles=[scatter_clean.legend_elements()[0][0], 
                        scatter_trans.legend_elements()[0][0]], 
               labels=['Clean (o)', 'Transformed (x)'])
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.show()


def plot_joint_representations(clean_features, trans_features, labels, save_path, title):
    N = len(clean_features)
    all_features = torch.cat([clean_features, trans_features], dim=0).numpy()
    
    reducer = umap.UMAP(n_components=2, random_state=6304)
    embeddings = reducer.fit_transform(all_features)
    
    emb_clean = embeddings[:N]
    emb_trans = embeddings[N:]
    labels_np = labels.numpy()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter_clean = ax.scatter(emb_clean[:, 0], emb_clean[:, 1], c=labels_np, 
                               cmap='tab10', marker='o', alpha=0.6, label='Clean', edgecolors='k')
    scatter_trans = ax.scatter(emb_trans[:, 0], emb_trans[:, 1], c=labels_np, 
                               cmap='tab10', marker='X', alpha=0.8, label='Transformed')
    
    ax.set_title(title)
    ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
