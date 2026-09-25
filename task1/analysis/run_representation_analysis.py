import os
import json
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless-safe
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torchvision.datasets import STL10
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision.transforms as T
from PIL import Image

from task1.configs.config import SEED, DATA_DIR, RESULTS_DIR, STL10_CLASSES
from task1.models.backbones import ModelWrapper
from task1.models.train_heads import HEADS_DIR
from task1.data.transforms import to_grayscale, translate_image, shuffle_patches_4x4

torch.manual_seed(SEED)
np.random.seed(SEED)

_IMAGENET_NORM = T.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
_CLIP_NORM     = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                              std=[0.26862954, 0.26130258, 0.27577711])

def _norm_for(m_name):
    return _CLIP_NORM if m_name == "clip_vit_b_32" else _IMAGENET_NORM


def _load_backbone(m_name, device):
    w = ModelWrapper(model_name=m_name, num_classes=len(STL10_CLASSES))
    head_path = os.path.join(HEADS_DIR, f"{m_name}_head.pt")
    if os.path.exists(head_path):
        w.head.load_state_dict(torch.load(head_path, map_location=device))
    w.eval().to(device)
    return w


class InterventionDataset(Dataset):
    """Applies an intervention to the pre-norm tensor on the fly."""
    def __init__(self, base_indices, base_stl10, intervention_fn=None):
        self.indices        = base_indices
        self.ds             = base_stl10
        self.intervention   = intervention_fn
        self.prep           = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.ds[self.indices[idx]]
        t = self.prep(img)
        if self.intervention is not None:
            t = self.intervention(t)
        return t, label   # norm applied after, per-model


class CueConflictDataset(Dataset):
    """Yields pre-norm tensors for accepted cue-conflict PNGs."""
    def __init__(self, cue_dir):
        self.records = []
        self.prep    = T.Compose([T.Resize((224, 224)), T.ToTensor()])
        for fname in sorted(os.listdir(cue_dir)):
            if not fname.lower().endswith(".png"):
                continue
            try:
                parts = fname.replace(".png", "").split("_")
                cc, sc = int(parts[1]), int(parts[3])
            except (IndexError, ValueError):
                continue
            img_t = self.prep(Image.open(os.path.join(cue_dir, fname)).convert("RGB"))
            mean_, std_ = img_t.mean().item(), img_t.std().item()
            if 0.05 <= mean_ <= 0.95 and std_ > 0.05:
                self.records.append((img_t, cc))   # use content class as label

    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        img_t, label = self.records[idx]
        return img_t, label


# ── feature extraction ────────────────────────────────────────────────────────
@torch.no_grad()
def extract(model, m_name, loader, device):
    norm  = _norm_for(m_name)
    feats, labels = [], []
    for x, y in loader:
        x_norm = torch.stack([norm(img) for img in x]).to(device)
        _, f   = model(x_norm)
        feats.append(f.cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def cosine_stability(fc, ft):
    fc = F.normalize(fc, dim=1)
    ft = F.normalize(ft, dim=1)
    return float((fc * ft).sum(dim=1).mean())


def plot_tsne(fc, ft, labels, title, save_path):
    N     = len(fc)
    all_f = torch.cat([fc, ft], dim=0).numpy()

    from sklearn.decomposition import PCA
    if all_f.shape[1] > 50:
        all_f = PCA(n_components=50, random_state=SEED).fit_transform(all_f)

    proj = TSNE(n_components=2, random_state=SEED,
                init='pca', learning_rate='auto',
                perplexity=min(30, max(5, N // 10))).fit_transform(all_f)

    pc  = proj[:N]
    pt  = proj[N:]
    lbl = labels.numpy()
    cmap = plt.cm.get_cmap("tab10", len(STL10_CLASSES))

    fig, ax = plt.subplots(figsize=(10, 8))
    for cls_idx in np.unique(lbl):
        m = lbl == cls_idx
        col = cmap(cls_idx % 10)
        ax.scatter(pc[m, 0], pc[m, 1], c=[col], marker='o', alpha=0.65,
                   s=20, label=f"{STL10_CLASSES[cls_idx]} (clean)")
        ax.scatter(pt[m, 0], pt[m, 1], c=[col], marker='x', alpha=0.80,
                   s=30)

    ax.set_title(title)
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker='o', color='grey', linestyle='', label='Clean'),
        Line2D([0], [0], marker='x', color='grey', linestyle='', label='Transformed'),
    ]
    ax.legend(handles=legend_handles, loc='best')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  Saved t-SNE -> {save_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    split_path = os.path.join(DATA_DIR, "stl10_splits_seed6304.json")
    with open(split_path) as f:
        splits = json.load(f)
    test_indices = splits["test_subset_indices"]

    base_ds = STL10(root=DATA_DIR, split='test', download=False)

    def _loader(ds, bs=64):
        return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=2)

    clean_ds   = InterventionDataset(test_indices, base_ds)
    gray_ds    = InterventionDataset(test_indices, base_ds, to_grayscale)
    trans_ds   = InterventionDataset(test_indices, base_ds,
                    lambda x: translate_image(x, 16, 0))
    patch_ds   = InterventionDataset(test_indices, base_ds,
                    lambda x: shuffle_patches_4x4(x, seed=SEED))

    cue_dir    = os.path.join(DATA_DIR, "cue_conflicts")
    has_cue    = os.path.isdir(cue_dir) and len(os.listdir(cue_dir)) > 0

    loaders_clean = _loader(clean_ds)
    loaders_trans = {
        "Grayscale":      _loader(gray_ds),
        "Translation16":  _loader(trans_ds),
        "PatchShuffle":   _loader(patch_ds),
    }
    if has_cue:
        cue_ds = CueConflictDataset(cue_dir)
        print(f"Found {len(cue_ds)} accepted cue-conflict images.")
    else:
        print("WARNING: cue-conflict directory not found. Skipping cue step.")

    models_to_test = ["resnet50", "vit_b_16", "clip_vit_b_32"]
    stability_table = {}

    for m_name in models_to_test:
        print(f"\n{'='*55}")
        print(f"  Backbone: {m_name}")
        print(f"{'='*55}")
        model = _load_backbone(m_name, device)
        norm  = _norm_for(m_name)

        feat_clean, labels_clean = extract(model, m_name, loaders_clean, device)

        stab_row = {}

        for trans_name, trans_loader in loaders_trans.items():
            feat_trans, _ = extract(model, m_name, trans_loader, device)
            IT = cosine_stability(feat_clean, feat_trans)
            stab_row[trans_name] = round(IT, 6)
            print(f"  Cosine stability [{trans_name}] = {IT:.4f}")

            save_path = os.path.join(
                RESULTS_DIR,
                f"tsne_{m_name.replace('/', '')}_{trans_name}.png")
            plot_tsne(feat_clean, feat_trans, labels_clean,
                      f"t-SNE: {m_name} — {trans_name}", save_path)

        if has_cue:
            cue_loader = DataLoader(cue_ds, batch_size=64, shuffle=False, num_workers=2)
            feat_cue, labels_cue = extract(model, m_name, cue_loader, device)

            class_means = {}
            for cls in range(len(STL10_CLASSES)):
                idx = (labels_clean == cls).nonzero(as_tuple=True)[0]
                if len(idx) > 0:
                    class_means[cls] = F.normalize(
                        feat_clean[idx].mean(0, keepdim=True), dim=1)

            sims = []
            for feat, lbl in zip(feat_cue, labels_cue.tolist()):
                if lbl in class_means:
                    f_n = F.normalize(feat.unsqueeze(0), dim=1)
                    sims.append(float((f_n * class_means[lbl]).sum()))
            IT_cue = float(np.mean(sims)) if sims else float('nan')
            stab_row["CueConflict"] = round(IT_cue, 6)
            print(f"  Cosine stability [CueConflict] = {IT_cue:.4f}  "
                  f"(vs. clean class prototype, N={len(sims)})")

            n_cue  = len(feat_cue)
            n_clean= len(feat_clean)
            if n_clean > n_cue:
                rng = np.random.RandomState(SEED)
                idx = rng.choice(n_clean, size=n_cue, replace=False)
                fc_sub = feat_clean[idx]
                lb_sub = labels_clean[idx]
            else:
                fc_sub, lb_sub = feat_clean, labels_clean

            save_path = os.path.join(
                RESULTS_DIR, f"tsne_{m_name.replace('/', '')}_CueConflict.png")
            plot_tsne(fc_sub, feat_cue, lb_sub,
                      f"t-SNE: {m_name} — Cue Conflict", save_path)

        stability_table[m_name] = stab_row

    out_path = os.path.join(RESULTS_DIR, "representation_stability.json")
    with open(out_path, "w") as f:
        json.dump(stability_table, f, indent=2)
    print(f"\nSaved representation stability to {out_path}")

    print("\n=== Cosine Stability Summary ===")
    transforms_order = ["Grayscale", "Translation16", "PatchShuffle", "CueConflict"]
    header = f"{'Model':<20}" + "".join(f"{t:<18}" for t in transforms_order)
    print(header)
    for m, row in stability_table.items():
        vals = "".join(f"{row.get(t, float('nan')):<18.4f}" for t in transforms_order)
        print(f"{m:<20}" + vals)


if __name__ == "__main__":
    main()
