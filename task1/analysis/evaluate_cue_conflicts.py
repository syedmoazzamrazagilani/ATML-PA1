import os
import json
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from sklearn.metrics import f1_score

from task1.configs.config import SEED, DATA_DIR, RESULTS_DIR, STL10_CLASSES
from task1.models.train_heads import load_head, get_norm


# ── visual rejection rule ───────────────────────────────────────────────────
def _is_acceptable(img_tensor: torch.Tensor) -> bool:
    """Return True if the image passes the visual quality filter."""
    mean_val = img_tensor.mean().item()
    std_val  = img_tensor.std().item()
    return (0.05 <= mean_val <= 0.95) and (std_val > 0.05)


def _load_conflicts(cue_dir: str):
    """
    Returns list of dicts:
      {path, content_class (int), style_class (int), accepted (bool)}
    Rejection rule is applied here, before any model prediction.
    """
    records   = []
    accepted  = 0
    rejected  = 0
    pre       = T.Compose([T.Resize((224, 224)), T.ToTensor()])

    for fname in sorted(os.listdir(cue_dir)):
        if not fname.lower().endswith(".png"):
            continue
        try:
            parts         = fname.replace(".png", "").split("_")
            content_class = int(parts[1])
            style_class   = int(parts[3])
        except (IndexError, ValueError):
            continue

        img_path   = os.path.join(cue_dir, fname)
        img_tensor = pre(Image.open(img_path).convert("RGB"))
        accept     = _is_acceptable(img_tensor)

        if accept:
            accepted += 1
        else:
            rejected += 1

        records.append({
            "path":          img_path,
            "content_class": content_class,
            "style_class":   style_class,
            "accepted":      accept,
        })

    print(f"  Loaded {len(records)} cue-conflict images: "
          f"{accepted} accepted, {rejected} rejected.")
    return records, accepted, rejected


def evaluate_cue_conflicts():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    cue_dir = os.path.join(DATA_DIR, "cue_conflicts")
    if not os.path.isdir(cue_dir):
        print(f"ERROR: Cue-conflict directory not found: {cue_dir}")
        print("  Run:  python -m task1.data.make_cue_conflicts  first.")
        return {}

    records, n_accepted, n_rejected = _load_conflicts(cue_dir)
    accepted_records = [r for r in records if r["accepted"]]

    if len(accepted_records) == 0:
        print("No accepted cue-conflict images. Adjust rejection rule or regenerate.")
        return {}

    pre_norm = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    models_to_test = ["resnet50", "vit_b_16", "clip_vit_b_32"]
    all_results    = {}

    for m_name in models_to_test:
        print(f"\n--- Cue Conflicts: {m_name} ---")
        norm    = get_norm(m_name)
        wrapper = load_head(m_name, device)

        n_shape, n_texture, n_other = 0, 0, 0
        pair_stats = {}   # (content, style) -> {"shape":0,"texture":0,"other":0}
        failures   = []   # informative cases for the report

        with torch.no_grad():
            for rec in accepted_records:
                img_t    = pre_norm(Image.open(rec["path"]).convert("RGB"))
                x_norm   = norm(img_t).unsqueeze(0).to(device)
                pred     = wrapper(x_norm)[0].argmax(1).item()

                c, s = rec["content_class"], rec["style_class"]
                pair = (c, s)
                if pair not in pair_stats:
                    pair_stats[pair] = {"shape": 0, "texture": 0, "other": 0}

                if pred == c:
                    decision = "shape"
                    n_shape   += 1
                    pair_stats[pair]["shape"] += 1
                elif pred == s:
                    decision = "texture"
                    n_texture += 1
                    pair_stats[pair]["texture"] += 1
                else:
                    decision = "other"
                    n_other   += 1
                    pair_stats[pair]["other"] += 1

                if len(failures) < 10:
                    failures.append({
                        "file":           os.path.basename(rec["path"]),
                        "content_label":  STL10_CLASSES[c],
                        "style_label":    STL10_CLASSES[s],
                        "pred_label":     STL10_CLASSES[pred],
                        "decision":       decision,
                    })

        n_total  = len(accepted_records)
        sb       = (n_shape / (n_shape + n_texture) * 100) if (n_shape + n_texture) > 0 else 0.0
        coverage = ((n_shape + n_texture) / n_total * 100) if n_total > 0 else 0.0

        pair_stats_named = {
            f"{STL10_CLASSES[c]}_shape_{STL10_CLASSES[s]}_texture": v
            for (c, s), v in pair_stats.items()
        }

        m_res = {
            "n_total_accepted": n_total,
            "n_rejected":        n_rejected,
            "n_shape":           n_shape,
            "n_texture":         n_texture,
            "n_other":           n_other,
            "shape_bias_pct":    round(sb, 2),
            "coverage_pct":      round(coverage, 2),
            "per_pair":          pair_stats_named,
            "sample_cases":      failures,
        }
        all_results[m_name] = m_res

        print(f"  n_shape={n_shape}  n_texture={n_texture}  n_other={n_other}  "
              f"n_total={n_total}")
        print(f"  Shape Bias = {sb:.2f}%   Coverage = {coverage:.2f}%")

    out_path = os.path.join(RESULTS_DIR, "cue_conflicts.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved cue-conflict results to {out_path}")
    return all_results


if __name__ == "__main__":
    evaluate_cue_conflicts()
