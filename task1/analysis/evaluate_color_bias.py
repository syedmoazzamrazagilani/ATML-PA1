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


def _get_prenorm_loader(splits):
    pre_norm = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    test_ds  = STL10(root=DATA_DIR, split='test', transform=pre_norm, download=False)
    return DataLoader(Subset(test_ds, splits["test_subset_indices"]),
                      batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


def _to_grayscale(img_tensor):
    import torchvision.transforms.functional as TF
    return TF.rgb_to_grayscale(img_tensor, num_output_channels=3)


def _rotate_hue(img_tensor, factor=0.5):
    import torchvision.transforms.functional as TF
    return TF.adjust_hue(img_tensor, factor)


def evaluate_color_bias():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    split_path = os.path.join(DATA_DIR, "stl10_splits_seed6304.json")
    with open(split_path) as f:
        splits = json.load(f)

    loader = _get_prenorm_loader(splits)

    interventions = {
        "clean":        None,
        "grayscale":    _to_grayscale,
        "hue_rotation": lambda x: _rotate_hue(x, 0.5),
    }

    models_to_test = ["resnet50", "vit_b_16", "clip_vit_b_32"]
    results = {}

    for m_name in models_to_test:
        print(f"\n--- Color Bias: {m_name} ---")
        norm    = get_norm(m_name)
        wrapper = load_head(m_name, device)

        # Collect predictions for each intervention in one pass over the loader
        inv_preds  = {k: [] for k in interventions}
        all_labels = []

        with torch.no_grad():
            for x_batch, y_batch in loader:
                all_labels.extend(y_batch.tolist())
                for inv_name, inv_fn in interventions.items():
                    x = x_batch.clone()
                    if inv_fn is not None:
                        x = torch.stack([inv_fn(img) for img in x])
                    x_norm = torch.stack([norm(img) for img in x]).to(device)
                    preds  = wrapper(x_norm)[0].argmax(1).cpu().tolist()
                    inv_preds[inv_name].extend(preds)

        clean_preds = np.array(inv_preds["clean"])
        labels_arr  = np.array(all_labels)
        clean_acc   = float((clean_preds == labels_arr).mean())

        m_res = {}
        for inv_name in interventions:
            preds_arr   = np.array(inv_preds[inv_name])
            acc         = float((preds_arr == labels_arr).mean())
            consistency = float((preds_arr == clean_preds).mean())
            delta_acc   = round(acc - clean_acc, 6)
            f1          = float(f1_score(labels_arr, preds_arr, average='macro'))
            m_res[inv_name] = {
                "acc":              acc,
                "f1":               f1,
                "delta_acc":        delta_acc,
                "consistency":      consistency,
            }
            print(f"  [{inv_name:15s}]  acc={acc:.4f}  Δacc={delta_acc:+.4f}  "
                  f"consistency={consistency:.4f}  f1={f1:.4f}")

        results[m_name] = m_res

    out_path = os.path.join(RESULTS_DIR, "color_bias.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved color-bias results to {out_path}")
    return results


if __name__ == "__main__":
    evaluate_color_bias()
