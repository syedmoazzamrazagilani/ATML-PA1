import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import STL10
import torchvision.transforms as T
from sklearn.metrics import f1_score

from task1.configs.config import SEED, DATA_DIR, RESULTS_DIR, BATCH_SIZE, STL10_CLASSES
from task1.models.train_heads import load_head, get_norm
from task1.data.transforms import shuffle_patches_4x4


def evaluate_patch_shuffle():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    split_path = os.path.join(DATA_DIR, "stl10_splits_seed6304.json")
    with open(split_path) as f:
        splits = json.load(f)

    pre_norm = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    test_ds  = STL10(root=DATA_DIR, split='test', transform=pre_norm, download=False)
    loader   = DataLoader(Subset(test_ds, splits["test_subset_indices"]),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    models_to_test = ["resnet50", "vit_b_16", "clip_vit_b_32"]
    results = {}

    for m_name in models_to_test:
        print(f"\n--- Patch Shuffle: {m_name} ---")
        norm    = get_norm(m_name)
        wrapper = load_head(m_name, device)

        clean_preds, shuffled_preds, all_labels = [], [], []
        with torch.no_grad():
            for x, y in loader:
                # Clean
                x_clean_norm = torch.stack([norm(img) for img in x]).to(device)
                cp = wrapper(x_clean_norm)[0].argmax(1).cpu().tolist()
                clean_preds.extend(cp)

                # Shuffled — same seed so identical across all models
                x_shuffled = torch.stack(
                    [shuffle_patches_4x4(img, seed=SEED) for img in x])
                x_shuffled_norm = torch.stack([norm(img) for img in x_shuffled]).to(device)
                sp = wrapper(x_shuffled_norm)[0].argmax(1).cpu().tolist()
                shuffled_preds.extend(sp)

                all_labels.extend(y.tolist())

        clean_arr   = np.array(clean_preds)
        shuffled_arr= np.array(shuffled_preds)
        labels_arr  = np.array(all_labels)

        clean_acc    = float((clean_arr   == labels_arr).mean())
        shuffled_acc = float((shuffled_arr == labels_arr).mean())
        consistency  = float((shuffled_arr == clean_arr).mean())
        delta_acc    = round(shuffled_acc - clean_acc, 6)
        f1           = float(f1_score(labels_arr, shuffled_preds, average='macro'))

        results[m_name] = {
            "clean_acc":    clean_acc,
            "shuffled_acc": shuffled_acc,
            "delta_acc":    delta_acc,
            "consistency":  consistency,
            "shuffled_f1":  f1,
        }
        print(f"  clean_acc={clean_acc:.4f}  shuffled_acc={shuffled_acc:.4f}  "
              f"Δacc={delta_acc:+.4f}  consistency={consistency:.4f}  f1={f1:.4f}")

    out_path = os.path.join(RESULTS_DIR, "patch_shuffle.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved patch-shuffle results to {out_path}")
    return results


if __name__ == "__main__":
    evaluate_patch_shuffle()
