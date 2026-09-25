import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import STL10
import torchvision.transforms as T
from sklearn.metrics import f1_score
import open_clip

from task1.configs.config import (
    SEED, DATA_DIR, RESULTS_DIR, BATCH_SIZE,
    LR, WEIGHT_DECAY, MAX_EPOCHS, PATIENCE, STL10_CLASSES
)
from task1.models.backbones import ModelWrapper

HEADS_DIR = os.path.join(RESULTS_DIR, "heads")


def get_norm(model_name):
    if model_name == "clip_vit_b_32":
        return T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                           std=[0.26862954, 0.26130258, 0.27577711])
    return T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def get_transform(model_name):
    norm = get_norm(model_name)
    return T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), norm])


def train_heads():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(HEADS_DIR, exist_ok=True)

    split_path = os.path.join(DATA_DIR, "stl10_splits_seed6304.json")
    with open(split_path) as f:
        splits = json.load(f)

    models_to_train = ["resnet50", "vit_b_16", "clip_vit_b_32"]
    results = {}

    for m_name in models_to_train:
        print(f"\n{'='*50}")
        print(f"  Training head for: {m_name}")
        print(f"{'='*50}")

        torch.manual_seed(SEED)
        transform = get_transform(m_name)

        train_ds = STL10(root=DATA_DIR, split='train', transform=transform, download=False)
        test_ds  = STL10(root=DATA_DIR, split='test',  transform=transform, download=False)

        train_loader = DataLoader(Subset(train_ds, splits["train_indices"]),
                                  batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=2, pin_memory=True)
        val_loader   = DataLoader(Subset(train_ds, splits["val_indices"]),
                                  batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=2, pin_memory=True)
        test_loader  = DataLoader(Subset(test_ds,  splits["test_subset_indices"]),
                                  batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=2, pin_memory=True)

        wrapper = ModelWrapper(model_name=m_name, num_classes=len(STL10_CLASSES)).to(device)
        optimizer = torch.optim.AdamW(wrapper.head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss()

        best_val_acc   = 0.0
        patience_ctr   = 0
        best_head_wts  = {k: v.clone() for k, v in wrapper.head.state_dict().items()}

        for epoch in range(1, MAX_EPOCHS + 1):
            # --- train ---
            wrapper.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits, _ = wrapper(x)
                criterion(logits, y).backward()
                optimizer.step()

            # --- validate ---
            wrapper.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    logits, _ = wrapper(x)
                    correct += (logits.argmax(1) == y).sum().item()
                    total   += y.size(0)
            val_acc = correct / total

            if val_acc > best_val_acc:
                best_val_acc  = val_acc
                patience_ctr  = 0
                best_head_wts = {k: v.clone() for k, v in wrapper.head.state_dict().items()}
            else:
                patience_ctr += 1

            print(f"  Epoch {epoch:02d} | val_acc={val_acc:.4f} | best={best_val_acc:.4f}"
                  f" | patience={patience_ctr}/{PATIENCE}")

            if patience_ctr >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}.")
                break

        wrapper.head.load_state_dict(best_head_wts)

        head_path = os.path.join(HEADS_DIR, f"{m_name}_head.pt")
        torch.save(best_head_wts, head_path)
        print(f"  Saved head to {head_path}")

        wrapper.eval()
        all_preds, all_labels, all_confs = [], [], []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                probs = wrapper(x)[0].softmax(-1)
                confs, preds = probs.max(1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(y.cpu().tolist())
                all_confs.extend(confs.cpu().tolist())

        acc      = float(np.mean(np.array(all_preds) == np.array(all_labels)))
        macro_f1 = float(f1_score(all_labels, all_preds, average='macro'))
        mean_conf= float(np.mean(all_confs))
        results[f"{m_name}_linear"] = {"acc": acc, "f1": macro_f1, "conf": mean_conf}
        print(f"  Clean test  ->  acc={acc:.4f}  f1={macro_f1:.4f}  conf={mean_conf:.4f}")

        # --- CLIP zero-shot ---
        if m_name == "clip_vit_b_32":
            print("  Running CLIP zero-shot …")
            text_prompts = [f"a photo of a {c}." for c in STL10_CLASSES]
            text_tokens  = open_clip.tokenize(text_prompts).to(device)

            zs_preds, zs_confs = [], []
            with torch.no_grad():
                text_feats = wrapper.full_clip.encode_text(text_tokens)
                text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
                for x, _ in test_loader:
                    img_feats = wrapper.full_clip.encode_image(x.to(device))
                    img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
                    probs     = (100.0 * img_feats @ text_feats.T).softmax(-1)
                    c, p      = probs.max(1)
                    zs_preds.extend(p.cpu().tolist())
                    zs_confs.extend(c.cpu().tolist())

            zs_acc  = float(np.mean(np.array(zs_preds) == np.array(all_labels)))
            zs_f1   = float(f1_score(all_labels, zs_preds, average='macro'))
            zs_conf = float(np.mean(zs_confs))
            results["clip_zeroshot"] = {"acc": zs_acc, "f1": zs_f1, "conf": zs_conf}
            print(f"  CLIP ZS     ->  acc={zs_acc:.4f}  f1={zs_f1:.4f}  conf={zs_conf:.4f}")

    out_path = os.path.join(RESULTS_DIR, "clean_baseline.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved clean baseline to {out_path}")
    return results


def load_head(m_name, device):
    """Utility: loads a ModelWrapper with the saved best head weights."""
    wrapper = ModelWrapper(model_name=m_name, num_classes=len(STL10_CLASSES)).to(device)
    head_path = os.path.join(HEADS_DIR, f"{m_name}_head.pt")
    if not os.path.exists(head_path):
        raise FileNotFoundError(
            f"Head not found at {head_path}. Run train_heads.py first.")
    wrapper.head.load_state_dict(torch.load(head_path, map_location=device))
    wrapper.eval()
    return wrapper


if __name__ == "__main__":
    train_heads()
