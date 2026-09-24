import os
import json
import yaml
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from shared.pacs_protocol import get_transforms
from task2.models.backbone import ResNet18Backbone, ClassifierHead


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def load_base_config():
    with open("task2/configs/base.yaml", "r") as f:
        return yaml.safe_load(f)


def freeze_backbone(backbone, classifier, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    backbone.load_state_dict(ckpt["backbone"])
    classifier.load_state_dict(ckpt["classifier"])
    backbone.eval()
    classifier.eval()
    return backbone, classifier


def extract_features(backbone, classifier, loader, device):
    feats_list, logits_list, labels_list = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            f = backbone(x)
            l = classifier(f)
            feats_list.append(f.cpu().numpy())
            logits_list.append(l.cpu().numpy())
            labels_list.append(y.numpy())
    return (
        np.concatenate(feats_list, axis=0),
        np.concatenate(logits_list, axis=0),
        np.concatenate(labels_list, axis=0),
    )


def domain_separability_score(source_feats, target_feats, seed=6304):
    """
    Train a logistic-regression binary classifier (C=1) on 70% of the combined
    source+target features to distinguish source (0) from target (1).
    Return accuracy on the held-out 30%.
    Chance = 50%.
    """
    n_src = len(source_feats)
    n_tgt = len(target_feats)
    # Balance: take min(n_src, n_tgt) from each side
    n_min = min(n_src, n_tgt, 1500)

    rng = np.random.default_rng(seed)
    s_idx = rng.choice(n_src, n_min, replace=False)
    t_idx = rng.choice(n_tgt, n_min, replace=False)

    X = np.vstack([source_feats[s_idx], target_feats[t_idx]])
    y = np.concatenate([np.zeros(n_min), np.ones(n_min)])

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 70/30 split
    indices = np.arange(len(y))
    rng.shuffle(indices)
    split = int(0.7 * len(y))
    train_idx, test_idx = indices[:split], indices[split:]

    clf = LogisticRegression(C=1, max_iter=1000, random_state=seed, solver="lbfgs")
    clf.fit(X[train_idx], y[train_idx])
    preds = clf.predict(X[test_idx])
    return float(accuracy_score(y[test_idx], preds))


def plot_training_curves(results_dir, methods):
    """Plot cls_loss, align_loss, and mean_source_f1 for all methods."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = {"source_only": "C0", "dan": "C1", "dann": "C2", "cdan": "C3"}

    for method in methods:
        hist_path = os.path.join(results_dir, f"{method}_history.json")
        if not os.path.exists(hist_path):
            continue
        with open(hist_path) as f:
            h = json.load(f)
        epochs = h["epoch"]
        axes[0].plot(epochs, h["cls_loss"], label=method.upper(), color=colors.get(method))
        axes[1].plot(epochs, h["align_loss"], label=method.upper(), color=colors.get(method))
        axes[2].plot(epochs, h["mean_source_f1"], label=method.upper(), color=colors.get(method))

    axes[0].set_title("Classification Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[1].set_title("Alignment / Domain Loss")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[2].set_title("Mean Source Val Macro-F1")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Macro-F1")

    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(results_dir, "training_curves.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Training curves saved to {out}")


def plot_umap(feats_src, feats_tgt, labels_src, labels_tgt, class_names, method, results_dir, seed=6304):
    try:
        import umap
    except ImportError:
        print("umap-learn not installed; skipping UMAP plot")
        return

    n = min(400, len(feats_src), len(feats_tgt))
    rng = np.random.default_rng(seed)
    si = rng.choice(len(feats_src), n, replace=False)
    ti = rng.choice(len(feats_tgt), n, replace=False)

    combined = np.vstack([feats_src[si], feats_tgt[ti]])
    reducer = umap.UMAP(n_components=2, random_state=seed)
    proj = reducer.fit_transform(combined)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    n_classes = len(class_names)
    cmap = plt.get_cmap("tab10", n_classes)

    # Left: colored by domain
    axes[0].scatter(proj[:n, 0], proj[:n, 1], c="steelblue", alpha=0.5, s=10, label="Source")
    axes[0].scatter(proj[n:, 0], proj[n:, 1], c="tomato", alpha=0.5, s=10, label="Target (Sketch)")
    axes[0].set_title(f"{method.upper()} — Source vs Target")
    axes[0].legend(fontsize=8)

    # Right: colored by class
    all_labels = np.concatenate([labels_src[si], labels_tgt[ti]])
    for c in range(n_classes):
        mask = all_labels == c
        axes[1].scatter(proj[mask, 0], proj[mask, 1], color=cmap(c), alpha=0.5, s=10, label=class_names[c])
    # Mark source vs target by marker shape
    src_mask = np.arange(len(proj)) < n
    axes[1].scatter(proj[src_mask, 0], proj[src_mask, 1], marker="o", s=12,
                    c=[cmap(l) for l in all_labels[src_mask]], alpha=0.5)
    axes[1].scatter(proj[~src_mask, 0], proj[~src_mask, 1], marker="^", s=18,
                    c=[cmap(l) for l in all_labels[~src_mask]], alpha=0.5)
    axes[1].set_title(f"{method.upper()} — By Class (○=source, △=target)")
    axes[1].legend(fontsize=6, ncol=2, markerscale=1.5)

    plt.suptitle(f"UMAP — {method.upper()}", fontsize=12)
    plt.tight_layout()
    out = os.path.join(results_dir, f"umap_{method}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"UMAP plot saved to {out}")


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    config = load_base_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config["results_dir"], exist_ok=True)

    _, eval_tf = get_transforms()

    # ---- Load splits ----
    with open(config["splits_path"], "r") as f:
        splits = json.load(f)

    # ---- Target loader ----
    target_ds = datasets.ImageFolder(
        os.path.join(config["data_dir"], config["target_domain"]), transform=eval_tf
    )
    target_loader = DataLoader(target_ds, batch_size=32, shuffle=False)
    class_names = target_ds.classes  # 7 PACS classes

    # ---- Source val loaders (shared across methods) ----
    source_val_loaders = {}
    for domain in config["source_domains"]:
        ds = datasets.ImageFolder(
            os.path.join(config["data_dir"], domain), transform=eval_tf
        )
        sub = Subset(ds, splits[domain]["val"])
        source_val_loaders[domain] = DataLoader(sub, batch_size=32, shuffle=False)

    methods = ["source_only", "dan", "dann", "cdan"]
    all_metrics = {}
    source_only_target_acc = None  # reference for delta

    for method in methods:
        ckpt_path = os.path.join(config["checkpoints_dir"], f"{method}_best.pth")
        if not os.path.exists(ckpt_path):
            print(f"[SKIP] {method}: no checkpoint at {ckpt_path}")
            continue

        print(f"\n{'='*60}")
        print(f"  Evaluating {method.upper()}")
        print(f"{'='*60}")

        backbone = ResNet18Backbone().to(device)
        classifier = ClassifierHead(num_classes=config["num_classes"]).to(device)
        backbone, classifier = freeze_backbone(backbone, classifier, ckpt_path, device)

        # ---- Per-source validation ----
        src_metrics = {}
        all_src_feats = []

        for domain in config["source_domains"]:
            loader = source_val_loaders[domain]
            feats, logits, labels = extract_features(backbone, classifier, loader, device)
            preds = np.argmax(logits, axis=1)
            acc = float(accuracy_score(labels, preds))
            f1  = float(f1_score(labels, preds, average="macro", zero_division=0))
            src_metrics[domain] = {"acc": acc, "f1": f1}
            all_src_feats.append(feats)
            print(f"  {domain:15s} val — Acc: {acc:.4f}  Macro-F1: {f1:.4f}")

        mean_src_acc = float(np.mean([v["acc"] for v in src_metrics.values()]))
        mean_src_f1  = float(np.mean([v["f1"] for v in src_metrics.values()]))
        print(f"  {'MEAN SOURCE':15s} val — Acc: {mean_src_acc:.4f}  Macro-F1: {mean_src_f1:.4f}")

        # ---- Target (Sketch) evaluation ----
        tgt_feats, tgt_logits, tgt_labels = extract_features(backbone, classifier, target_loader, device)
        tgt_preds = np.argmax(tgt_logits, axis=1)
        tgt_acc = float(accuracy_score(tgt_labels, tgt_preds))
        tgt_f1  = float(f1_score(tgt_labels, tgt_preds, average="macro", zero_division=0))
        print(f"\n  Target (Sketch) — Acc: {tgt_acc:.4f}  Macro-F1: {tgt_f1:.4f}")

        # Per-class target accuracy
        per_class_acc = {}
        for c, cname in enumerate(class_names):
            mask = tgt_labels == c
            if mask.sum() > 0:
                per_class_acc[cname] = float(accuracy_score(tgt_labels[mask], tgt_preds[mask]))
            else:
                per_class_acc[cname] = None

        # ---- Domain separability ----
        src_feats_all = np.concatenate(all_src_feats, axis=0)
        dom_sep = domain_separability_score(src_feats_all, tgt_feats, seed=config["seed"])
        print(f"  Domain separability (LR acc, 70/30, C=1): {dom_sep:.4f}  (chance=0.50)")

        if method == "source_only":
            source_only_target_acc = tgt_acc

        all_metrics[method] = {
            "source_val": src_metrics,
            "mean_source_acc": mean_src_acc,
            "mean_source_f1":  mean_src_f1,
            "target_acc":  tgt_acc,
            "target_f1":   tgt_f1,
            "per_class_target_acc": per_class_acc,
            "domain_separability": dom_sep,
        }

        # ---- UMAP ----
        src_labels_all = []
        src_feats_for_umap = []
        for domain in config["source_domains"]:
            loader = source_val_loaders[domain]
            feats, _, labels = extract_features(backbone, classifier, loader, device)
            src_feats_for_umap.append(feats)
            src_labels_all.append(labels)
        src_feats_for_umap = np.concatenate(src_feats_for_umap, axis=0)
        src_labels_all = np.concatenate(src_labels_all, axis=0)

        plot_umap(src_feats_for_umap, tgt_feats,
                  src_labels_all, tgt_labels,
                  class_names, method,
                  config["results_dir"], seed=config["seed"])

    # ---- Add delta vs source-only ----
    if source_only_target_acc is not None:
        for method, m in all_metrics.items():
            m["target_acc_delta"] = round(m["target_acc"] - source_only_target_acc, 4)

    # ---- Per-class change table ----
    if "source_only" in all_metrics:
        so_pc = all_metrics["source_only"]["per_class_target_acc"]
        print(f"\n{'Class':15s} | {'src_only':>10}", end="")
        for m in ["dan", "dann", "cdan"]:
            if m in all_metrics:
                print(f" | {m:>10}", end="")
        print()
        print("-" * 70)
        for cname in class_names:
            so_val = so_pc.get(cname)
            row = f"{cname:15s} | {so_val or 0:10.4f}"
            for m in ["dan", "dann", "cdan"]:
                if m in all_metrics:
                    v = all_metrics[m]["per_class_target_acc"].get(cname)
                    delta = (v - so_val) if (v is not None and so_val is not None) else 0.0
                    row += f" | {delta:+10.4f}"
            print(row)

    # ---- Summary table ----
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'Method':12s} | {'Tgt Acc':>8} | {'Tgt F1':>8} | {'Tgt Δ':>8} | {'Dom Sep':>8} | {'Src Acc':>8}")
    print("-" * 70)
    for method in methods:
        if method not in all_metrics:
            continue
        m = all_metrics[method]
        delta_str = f"{m.get('target_acc_delta', 0):+.4f}"
        print(
            f"{method:12s} | {m['target_acc']:8.4f} | {m['target_f1']:8.4f} | "
            f"{delta_str:>8} | {m['domain_separability']:8.4f} | {m['mean_source_acc']:8.4f}"
        )

    # ---- Save all metrics ----
    out_path = os.path.join(config["results_dir"], "final_eval.json")
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nAll metrics saved to {out_path}")

    # ---- Training curves ----
    plot_training_curves(config["results_dir"], methods)


if __name__ == "__main__":
    main()
