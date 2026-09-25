import numpy as np
import torch

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def find_failures(unknown_scores: np.ndarray,
                  unknown_logits: np.ndarray,
                  unknown_class_names: list,
                  threshold: float,
                  n_samples: int = 10,
                  score_name: str = "MLS"):
    accepted_mask = unknown_scores <= threshold
    accepted_idx  = np.where(accepted_mask)[0]

    if len(accepted_idx) == 0:
        return []

    order = accepted_idx[np.argsort(unknown_scores[accepted_idx])]
    order = order[:n_samples]

    failures = []
    for i in order:
        pred_class = int(np.argmax(unknown_logits[i]))
        failures.append({
            "unknown_class":   unknown_class_names[i]
                               if i < len(unknown_class_names) else "unknown",
            "predicted_class": CIFAR10_CLASSES[pred_class],
            "score":           float(unknown_scores[i]),
            "threshold":       float(threshold),
            "score_name":      score_name,
        })
    return failures


def print_failures(failures: list, title: str = "Failure Cases"):
    print(f"\n{'─'*60}")
    print(f"  {title}  ({len(failures)} cases)")
    print(f"{'─'*60}")
    print(f"  {'Unknown class':<20} {'Pred (CIFAR-10)':<18} "
          f"{'Score':>8}  {'Threshold':>10}")
    for f in failures:
        print(f"  {f['unknown_class']:<20} {f['predicted_class']:<18} "
              f"  {f['score']:8.4f}    {f['threshold']:8.4f}")
