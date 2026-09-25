import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from task1.configs.config import RESULTS_DIR

SHIFTS = [0, 8, 16, 32]
LABELS  = {
    "resnet50":      "ResNet-50",
    "vit_b_16":      "ViT-B/16",
    "clip_vit_b_32": "CLIP ViT-B-32",
}
MARKERS = {
    "resnet50":      "o",
    "vit_b_16":      "s",
    "clip_vit_b_32": "^",
}


def plot_translation_curves(results=None):
    if results is None:
        path = os.path.join(RESULTS_DIR, "translation.json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Results not found at {path}.\n"
                "Run  python -m task1.analysis.evaluate_translation  first.")
        with open(path) as f:
            results = json.load(f)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    def _get_series(results, key):
        out = {}
        for m_name, data in results.items():
            vals = []
            for δ in SHIFTS:
                shift_data = data["shifts"].get(str(δ)) or data["shifts"].get(δ)
                vals.append(float(shift_data[key]) * 100)
            out[m_name] = vals
        return out

    accs   = _get_series(results, "avg_accuracy")
    consis = _get_series(results, "avg_consistency")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for m_name in results:
        kw = dict(marker=MARKERS[m_name], label=LABELS[m_name], linewidth=2)
        axes[0].plot(SHIFTS, accs[m_name],   **kw)
        axes[1].plot(SHIFTS, consis[m_name], **kw)

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
        ax.set_ylim(bottom=max(0, min(
            min(v for series in [accs, consis] for v in series[m]) for m in results
        ) - 5))

    plt.tight_layout()
    combined_path = os.path.join(RESULTS_DIR, "translation_curve.png")
    plt.savefig(combined_path, dpi=300)
    plt.close()
    print(f"Saved combined plot -> {combined_path}")

    fig, ax = plt.subplots(figsize=(7, 5))
    for m_name in results:
        ax.plot(SHIFTS, accs[m_name],
                marker=MARKERS[m_name], label=LABELS[m_name], linewidth=2)
    ax.set_xticks(SHIFTS)
    ax.set_xlabel("Displacement δ (pixels)")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("Translation Invariance – Accuracy")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    acc_path = os.path.join(RESULTS_DIR, "translation_acc.png")
    plt.savefig(acc_path, dpi=300)
    plt.close()
    print(f"Saved accuracy plot  -> {acc_path}")

    fig, ax = plt.subplots(figsize=(7, 5))
    for m_name in results:
        ax.plot(SHIFTS, consis[m_name],
                marker=MARKERS[m_name], label=LABELS[m_name], linewidth=2)
    ax.set_xticks(SHIFTS)
    ax.set_xlabel("Displacement δ (pixels)")
    ax.set_ylabel("Prediction Consistency (%)")
    ax.set_title("Translation Invariance – Consistency(δ)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    cons_path = os.path.join(RESULTS_DIR, "translation_cons.png")
    plt.savefig(cons_path, dpi=300)
    plt.close()
    print(f"Saved consistency plot -> {cons_path}")


if __name__ == "__main__":
    plot_translation_curves()
