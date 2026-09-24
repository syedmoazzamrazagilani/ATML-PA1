import os
import json
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import umap

from shared.pacs_protocol import get_transforms
from task2.models.backbone import ResNet18Backbone, ClassifierHead

def load_config():
    with open("task2/configs/base.yaml", "r") as f:
        return yaml.safe_load(f)

def extract_features(backbone, classifier, loader, device):
    backbone.eval()
    classifier.eval()
    features_list, logits_list, labels_list = [], [], []
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            feats = backbone(x)
            logits = classifier(feats)
            features_list.append(feats.cpu().numpy())
            logits_list.append(logits.cpu().numpy())
            labels_list.append(y.numpy())
            
    features = np.concatenate(features_list, axis=0)
    logits = np.concatenate(logits_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return features, logits, labels

def compute_proxy_a_distance(source_feats, target_feats, seed=6304):
    """
    Computes Proxy A-distance d_A = 2 * (1 - 2 * epsilon)
    where epsilon is the generalization error of discriminating source vs target features.
    """
    n_source = len(source_feats)
    n_target = len(target_feats)
    n_min = min(n_source, n_target, 1000)
    
    np.random.seed(seed)
    s_idx = np.random.choice(n_source, n_min, replace=False)
    t_idx = np.random.choice(n_target, n_min, replace=False)
    
    X = np.vstack([source_feats[s_idx], target_feats[t_idx]])
    y = np.concatenate([np.zeros(n_min), np.ones(n_min)])
    
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    split = int(0.7 * len(y))
    train_idx, test_idx = indices[:split], indices[split:]
    
    clf = LogisticRegression(max_iter=500, random_state=seed)
    clf.fit(X[train_idx], y[train_idx])
    preds = clf.predict(X[test_idx])
    
    error = 1.0 - accuracy_score(y[test_idx], preds)
    pad = 2.0 * (1.0 - 2.0 * error)
    return max(0.0, float(pad))

def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config["results_dir"], exist_ok=True)
    
    _, eval_tf = get_transforms()
    
    target_ds = datasets.ImageFolder(os.path.join(config["data_dir"], config["target_domain"]), transform=eval_tf)
    target_loader = DataLoader(target_ds, batch_size=32, shuffle=False)
    
    with open(config["splits_path"], "r") as f:
        splits = json.load(f)
    
    source_feats_pool = []
    for domain in config["source_domains"]:
        ds = datasets.ImageFolder(os.path.join(config["data_dir"], domain), transform=eval_tf)
        sub = Subset(ds, splits[domain]["val"])
        loader = DataLoader(sub, batch_size=32, shuffle=False)
        # We will extract features from these after loading the model
        
    methods = ["source_only", "dan", "dann", "cdan"]
    final_metrics = {}
    
    for method in methods:
        ckpt_path = os.path.join(config["checkpoints_dir"], f"{method}_best.pth")
        if not os.path.exists(ckpt_path):
            print(f"Skipping {method}: Checkpoint not found at {ckpt_path}")
            continue
            
        print(f"\n================ Evaluating {method.upper()} ================")
        backbone = ResNet18Backbone().to(device)
        classifier = ClassifierHead(num_classes=config["num_classes"]).to(device)
        
        ckpt = torch.load(ckpt_path, map_location=device)
        backbone.load_state_dict(ckpt["backbone"])
        classifier.load_state_dict(ckpt["classifier"])
        
        t_feats, t_logits, t_labels = extract_features(backbone, classifier, target_loader, device)
        t_preds = np.argmax(t_logits, axis=1)
        
        acc = float(accuracy_score(t_labels, t_preds))
        f1 = float(f1_score(t_labels, t_preds, average='macro'))
        
        s_feats_list = []
        for domain in config["source_domains"]:
            ds = datasets.ImageFolder(os.path.join(config["data_dir"], domain), transform=eval_tf)
            sub = Subset(ds, splits[domain]["val"])
            loader = DataLoader(sub, batch_size=32, shuffle=False)
            sf, _, _ = extract_features(backbone, classifier, loader, device)
            s_feats_list.append(sf)
        s_feats = np.concatenate(s_feats_list, axis=0)
        
        pad = compute_proxy_a_distance(s_feats, t_feats, seed=config["seed"])
        
        print(f"Target (Sketch) Accuracy: {acc:.4f}")
        print(f"Target (Sketch) Macro-F1: {f1:.4f}")
        print(f"Proxy A-Distance (Domain Separability): {pad:.4f}")
        
        final_metrics[method] = {
            "target_accuracy": acc,
            "target_macro_f1": f1,
            "proxy_a_distance": pad
        }
        
        # UMAP Visualization
        print("Generating UMAP plot...")
        reducer = umap.UMAP(n_components=2, random_state=config["seed"])
        combined_feats = np.vstack([s_feats[:500], t_feats[:500]])
        proj = reducer.fit_transform(combined_feats)
        
        plt.figure(figsize=(7, 6))
        plt.scatter(proj[:500, 0], proj[:500, 1], c='blue', alpha=0.5, label='Source')
        plt.scatter(proj[500:, 0], proj[500:, 1], c='red', alpha=0.5, label='Target (Sketch)')
        plt.title(f"{method.upper()} Feature Space (PAD: {pad:.2f})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(config["results_dir"], f"umap_{method}.png"))
        plt.close()
        
    out_json = os.path.join(config["results_dir"], "final_eval.json")
    with open(out_json, "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"\nAll metrics successfully saved to {out_json}!")

if __name__ == "__main__":
    main()
