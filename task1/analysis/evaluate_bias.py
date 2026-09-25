import os
import json
import numpy as np
import torch

from task1.configs.config import RESULTS_DIR, STL10_CLASSES


def calculate_shape_bias(n_shape, n_texture, n_total):
    """
    Shape Bias(%) = N_shape / (N_shape + N_texture) × 100
    Coverage(%)   = (N_shape + N_texture) / N_total × 100
    """
    if (n_shape + n_texture) == 0:
        return 0.0, 0.0
    shape_bias = (n_shape / (n_shape + n_texture)) * 100.0
    coverage   = ((n_shape + n_texture) / n_total) * 100.0
    return shape_bias, coverage


def evaluate_consistency_and_bias(model, clean_loader, transformed_loader,
                                   norm_fn, device, is_cue_conflict=False):
    """
    Evaluates prediction consistency and optional shape-bias metrics.

    Parameters
    ----------
    model              : ModelWrapper (eval mode, on device)
    clean_loader       : DataLoader yielding (pre_norm_tensor, label)
    transformed_loader : DataLoader yielding (pre_norm_tensor, label|info)
    norm_fn            : normalisation transform (model-specific)
    device             : torch.device
    is_cue_conflict    : if True, loader yields (img, (content_cls, style_cls))

    Returns
    -------
    dict with keys: consistency, [n_shape, n_texture, n_other,
                                  n_total, shape_bias_pct, coverage_pct]
    """
    model.eval()
    clean_preds, trans_preds = [], []
    shape_labels, texture_labels = [], []

    with torch.no_grad():
        for (cx, _), (tx, tinfo) in zip(clean_loader, transformed_loader):
            cx_n = torch.stack([norm_fn(img) for img in cx]).to(device)
            tx_n = torch.stack([norm_fn(img) for img in tx]).to(device)

            clean_preds.append(model(cx_n)[0].argmax(1).cpu())
            trans_preds.append(model(tx_n)[0].argmax(1).cpu())

            if is_cue_conflict:
                # tinfo expected as (content_cls_tensor, style_cls_tensor)
                shape_labels.append(tinfo[0])
                texture_labels.append(tinfo[1])

    clean_preds = torch.cat(clean_preds)
    trans_preds = torch.cat(trans_preds)
    consistency = (clean_preds == trans_preds).float().mean().item()

    if is_cue_conflict:
        shape_labels   = torch.cat(shape_labels)
        texture_labels = torch.cat(texture_labels)
        n_shape   = (trans_preds == shape_labels).sum().item()
        n_texture = (trans_preds == texture_labels).sum().item()
        n_total   = len(trans_preds)
        n_other   = n_total - n_shape - n_texture
        sb, cov   = calculate_shape_bias(n_shape, n_texture, n_total)
        return {
            "consistency":   consistency,
            "n_shape":       n_shape,
            "n_texture":     n_texture,
            "n_other":       n_other,
            "n_total":       n_total,
            "shape_bias_pct": sb,
            "coverage_pct":   cov,
        }

    return {"consistency": consistency}


def print_summary():
    """Print a combined summary of all Task-1 bias results to stdout."""
    files = {
        "Clean Baseline": "clean_baseline.json",
        "Color Bias":     "color_bias.json",
        "Cue Conflicts":  "cue_conflicts.json",
        "Translation":    "translation.json",
        "Patch Shuffle":  "patch_shuffle.json",
    }
    for title, fname in files.items():
        path = os.path.join(RESULTS_DIR, fname)
        if not os.path.exists(path):
            print(f"\n[{title}] Results not found at {path}")
            continue
        with open(path) as f:
            data = json.load(f)
        print(f"\n{'─'*55}")
        print(f"  {title}")
        print(f"{'─'*55}")

        if title == "Clean Baseline":
            for k, v in data.items():
                print(f"  {k:<25}  acc={v['acc']:.4f}  f1={v['f1']:.4f}  "
                      f"conf={v['conf']:.4f}")

        elif title == "Color Bias":
            for m_name, m_data in data.items():
                print(f"\n  {m_name}")
                for inv, vals in m_data.items():
                    print(f"    {inv:<18}  acc={vals['acc']:.4f}  "
                          f"Δacc={vals['delta_acc']:+.4f}  "
                          f"consistency={vals['consistency']:.4f}")

        elif title == "Cue Conflicts":
            for m_name, m_data in data.items():
                print(f"  {m_name:<20}  shape_bias={m_data['shape_bias_pct']:.2f}%  "
                      f"coverage={m_data['coverage_pct']:.2f}%  "
                      f"(n_shape={m_data['n_shape']}  n_tex={m_data['n_texture']}  "
                      f"n_other={m_data['n_other']})")

        elif title == "Translation":
            for m_name, m_data in data.items():
                print(f"  {m_name}")
                for δ, vals in m_data["shifts"].items():
                    print(f"    δ={int(δ):2d}px  acc={vals['avg_accuracy']:.4f}  "
                          f"consistency={vals['avg_consistency']:.4f}")

        elif title == "Patch Shuffle":
            for m_name, v in data.items():
                print(f"  {m_name:<20}  clean={v['clean_acc']:.4f}  "
                      f"shuffled={v['shuffled_acc']:.4f}  "
                      f"Δacc={v['delta_acc']:+.4f}  "
                      f"consistency={v['consistency']:.4f}")


if __name__ == "__main__":
    print_summary()
