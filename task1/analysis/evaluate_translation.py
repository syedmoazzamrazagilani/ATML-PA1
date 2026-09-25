import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import STL10
import torchvision.transforms as T

from task1.configs.config import SEED, DATA_DIR, RESULTS_DIR, BATCH_SIZE, STL10_CLASSES
from task1.models.train_heads import load_head, get_norm
from task1.data.transforms import translate_image

SHIFTS = [0, 8, 16, 32]
DIRECTIONS = {
    "right":  ( 1,  0),
    "left":   (-1,  0),
    "down":   ( 0,  1),
    "up":     ( 0, -1),
}


def evaluate_translation():
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
    all_results    = {}

    for m_name in models_to_test:
        print(f"\n--- Translation: {m_name} ---")
        norm    = get_norm(m_name)
        wrapper = load_head(m_name, device)

        clean_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in loader:
                x_norm = torch.stack([norm(img) for img in x]).to(device)
                clean_preds.extend(wrapper(x_norm)[0].argmax(1).cpu().tolist())
                all_labels.extend(y.tolist())
        clean_preds = np.array(clean_preds)
        labels_arr  = np.array(all_labels)
        clean_acc   = float((clean_preds == labels_arr).mean())

        shift_results = {}
        for δ in SHIFTS:
            dir_accs, dir_consis = [], []
            for dir_name, (dx, dy) in DIRECTIONS.items():
                shift_x, shift_y = δ * dx, δ * dy
                preds = []
                with torch.no_grad():
                    for x, _ in loader:
                        x_shifted = torch.stack(
                            [translate_image(img, shift_x, shift_y) for img in x])
                        x_norm = torch.stack([norm(img) for img in x_shifted]).to(device)
                        preds.extend(wrapper(x_norm)[0].argmax(1).cpu().tolist())
                preds_arr = np.array(preds)
                dir_accs.append(float((preds_arr == labels_arr).mean()))
                dir_consis.append(float((preds_arr == clean_preds).mean()))

            avg_acc   = float(np.mean(dir_accs))
            avg_cons  = float(np.mean(dir_consis))
            shift_results[δ] = {
                "avg_accuracy":    avg_acc,
                "avg_consistency": avg_cons,
                "delta_acc":       round(avg_acc - clean_acc, 6),
            }
            print(f"  δ={δ:2d}px  acc={avg_acc:.4f}  consistency={avg_cons:.4f}  "
                  f"Δacc={avg_acc - clean_acc:+.4f}")

        all_results[m_name] = {"clean_acc": clean_acc, "shifts": shift_results}

    out_path = os.path.join(RESULTS_DIR, "translation.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved translation results to {out_path}")

    _plot_translation_curves(all_results)
    return all_results


def _plot_translation_curves(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    markers    = {"resnet50": "o", "vit_b_16": "s", "clip_vit_b_32": "^"}
    labels_map = {"resnet50": "ResNet-50", "vit_b_16": "ViT-B/16",
                  "clip_vit_b_32": "CLIP ViT-B-32"}

    for m_name, res in results.items():
        shifts   = sorted(int(k) for k in res["shifts"])
        accs     = [res["shifts"][δ]["avg_accuracy"]    for δ in shifts]
        consises = [res["shifts"][δ]["avg_consistency"] for δ in shifts]

        axes[0].plot(shifts, [a * 100 for a in accs],
                     marker=markers[m_name], label=labels_map[m_name], linewidth=2)
        axes[1].plot(shifts, [c * 100 for c in consises],
                     marker=markers[m_name], label=labels_map[m_name], linewidth=2)

    for ax, ylabel, title in zip(
        axes,
        ["Top-1 Accuracy (%)", "Prediction Consistency (%)"],
        ["Accuracy vs. Translation Displacement",
         "Consistency(δ) vs. Translation Displacement"]
    ):
        ax.set_xticks(SHIFTS)
        ax.set_xlabel("Displacement δ (pixels)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, "translation_curve.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved translation curve to {save_path}")


if __name__ == "__main__":
    evaluate_translation()
