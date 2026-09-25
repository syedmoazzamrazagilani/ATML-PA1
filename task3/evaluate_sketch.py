import os
import json
import yaml
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from shared.pacs_protocol import get_transforms
from task3.models.backbone import ResNet18Backbone, ClassifierHead
from task3.selection.source_validation import evaluate_source_val
from task3.evaluation.domain_metrics import evaluate_target
from task3.evaluation.source_domain_separability import compute_source_separability
from task3.evaluation.sharpness import compute_sharpness

def load_base_config():
    with open("task3/configs/base.yaml") as f:
        return yaml.safe_load(f)


def load_model(ckpt_path, num_classes, device):
    backbone   = ResNet18Backbone().to(device)
    classifier = ClassifierHead(num_classes=num_classes).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    backbone.load_state_dict(ckpt["backbone"])
    classifier.load_state_dict(ckpt["classifier"])
    backbone.eval()
    classifier.eval()
    return backbone, classifier


def plot_training_curves(results_dir, methods_with_history):
    """Plot cls_loss (and mmd for DAN-DG) and mean source F1."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = {"erm": "C0", "dan_dg": "C1", "sam": "C2"}

    for method, h in methods_with_history.items():
        c = colors.get(method, "gray")
        epochs = h.get("epoch", [])
        if not epochs:
            continue

        cls_key = "cls_loss" if "cls_loss" in h else "cls_loss_ascent"
        if cls_key in h:
            axes[0].plot(epochs, h[cls_key], label=method.upper(), color=c)

        if "mmd_loss" in h:
            axes[1].plot(epochs, h["mmd_loss"], label=method.upper(), color=c)

        if "mean_source_f1" in h:
            axes[2].plot(epochs, h["mean_source_f1"], label=method.upper(), color=c)

        if "worst_source_f1" in h:
            axes[2].plot(epochs, h["worst_source_f1"], linestyle="--",
                         color=c, alpha=0.5, label=f"{method.upper()} worst")

    axes[0].set_title("Classification Loss");    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[1].set_title("MMD Loss (DAN-DG only)"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[2].set_title("Mean/Worst Source Val Macro-F1"); axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Macro-F1")

    for ax in axes:
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(results_dir, "training_curves_task3.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Training curves saved → {out}")


def plot_per_class_delta(results_dir, class_names, per_class_erm, methods_pc):
    """Bar chart of per-class Sketch accuracy change vs ERM."""
    n_classes = len(class_names)
    x = np.arange(n_classes)
    width = 0.25
    colors = {"dan_dg": "C1", "sam": "C2"}

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (method, pc) in enumerate(methods_pc.items()):
        deltas = [pc.get(c, 0) - per_class_erm.get(c, 0) for c in class_names]
        ax.bar(x + i * width, deltas, width, label=method.upper(),
               color=colors.get(method, "gray"), alpha=0.8)

    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_ylabel("Δ Accuracy vs ERM")
    ax.set_title("Per-Class Sketch Accuracy Change vs ERM")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out = os.path.join(results_dir, "per_class_sketch_delta.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Per-class delta plot saved → {out}")

def main():
    cfg    = load_base_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["results_dir"], exist_ok=True)

    train_tf, eval_tf = get_transforms()
    with open(cfg["splits_path"]) as f:
        splits = json.load(f)

    val_loaders = {}
    for domain in cfg["source_domains"]:
        ds = datasets.ImageFolder(os.path.join(cfg["data_dir"], domain), transform=eval_tf)
        val_loaders[domain] = DataLoader(
            Subset(ds, splits[domain]["val"]), batch_size=32, shuffle=False)

    sketch_ds = datasets.ImageFolder(
        os.path.join(cfg["data_dir"], cfg["target_domain"]), transform=eval_tf)
    sketch_loader = DataLoader(sketch_ds, batch_size=32, shuffle=False)
    class_names   = sketch_ds.classes

    ckpt_map = {
        "erm":    os.path.join(cfg["checkpoints_dir"], "erm_best.pth"),
        "dan_dg": os.path.join(cfg["checkpoints_dir"], "dan_dg_best.pth"),
        "sam":    os.path.join(cfg["checkpoints_dir"], "sam_best.pth"),
    }

    all_metrics = {}
    histories   = {}
    erm_sketch_acc = None

    for method, ckpt_path in ckpt_map.items():
        if not os.path.exists(ckpt_path):
            print(f"[SKIP] {method}: no checkpoint at {ckpt_path}")
            continue

        print(f"\n{'='*60}\n  {method.upper()}\n{'='*60}")
        backbone, classifier = load_model(ckpt_path, cfg["num_classes"], device)

        src_metrics = evaluate_source_val(backbone, classifier, val_loaders, device)
        print(f"  Source validation:")
        for d in cfg["source_domains"]:
            print(f"    {d:15s}: Acc={src_metrics[d]['acc']:.4f}  F1={src_metrics[d]['f1']:.4f}")
        print(f"    {'MEAN':15s}: Acc={src_metrics['mean_acc']:.4f}  F1={src_metrics['mean_f1']:.4f}")
        print(f"    {'WORST':15s}: Acc={src_metrics['worst_acc']:.4f}  F1={src_metrics['worst_f1']:.4f}")

        tgt = evaluate_target(backbone, classifier, sketch_loader, class_names, device)
        print(f"  Sketch: Acc={tgt['accuracy']:.4f}  Macro-F1={tgt['macro_f1']:.4f}")

        dom_sep = compute_source_separability(backbone, classifier, val_loaders, device, seed=cfg["seed"])
        print(f"  Source-domain separability (3-way LR acc): {dom_sep:.4f}  (chance=0.333)")

        delta_sharp, loss_clean, loss_perturbed = compute_sharpness(
            backbone, classifier, val_loaders, device,
            seed=cfg["seed"], rho=0.05, examples_per_domain=32)
        print(f"  Sharpness proxy: L(θ)={loss_clean:.4f}  L(θ+ε)={loss_perturbed:.4f}  "
              f"Δ={delta_sharp:.4f}")

        if method == "erm":
            erm_sketch_acc = tgt["accuracy"]

        all_metrics[method] = {
            "source_val": {d: src_metrics[d] for d in cfg["source_domains"]},
            "mean_source_acc":  src_metrics["mean_acc"],
            "mean_source_f1":   src_metrics["mean_f1"],
            "worst_source_acc": src_metrics["worst_acc"],
            "worst_source_f1":  src_metrics["worst_f1"],
            "sketch_acc":     tgt["accuracy"],
            "sketch_f1":      tgt["macro_f1"],
            "per_class_sketch": tgt["per_class_acc"],
            "confusion_matrix": tgt["confusion_matrix"],
            "source_domain_separability": dom_sep,
            "sharpness_delta": delta_sharp,
            "sharpness_loss_clean": loss_clean,
            "sharpness_loss_perturbed": loss_perturbed,
        }

        hist_path = os.path.join(cfg["results_dir"], f"{method}_history.json")
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                histories[method] = json.load(f)

    if erm_sketch_acc is not None:
        for method, m in all_metrics.items():
            m["sketch_acc_delta"] = round(m["sketch_acc"] - erm_sketch_acc, 4)

    print(f"\n{'='*100}")
    print("TASK 3 SUMMARY TABLE")
    print(f"{'='*100}")
    print(f"{'Method':10s} | {'Photo':>6} | {'ArtPt':>6} | {'Cartoon':>7} | "
          f"{'MeanSrc':>7} | {'WorstSrc':>8} | {'SketchAcc':>9} | {'SketchF1':>8} | "
          f"{'Δ':>7} | {'DomSep':>7} | {'Sharp':>7}")
    print("-" * 100)
    for method in ["erm", "dan_dg", "sam"]:
        if method not in all_metrics:
            continue
        m = all_metrics[method]
        print(
            f"{method:10s} | "
            f"{m['source_val']['photo']['acc']:6.4f} | "
            f"{m['source_val']['art_painting']['acc']:6.4f} | "
            f"{m['source_val']['cartoon']['acc']:7.4f} | "
            f"{m['mean_source_acc']:7.4f} | "
            f"{m['worst_source_acc']:8.4f} | "
            f"{m['sketch_acc']:9.4f} | "
            f"{m['sketch_f1']:8.4f} | "
            f"{m.get('sketch_acc_delta', 0):+7.4f} | "
            f"{m['source_domain_separability']:7.4f} | "
            f"{m['sharpness_delta']:7.4f}"
        )

    print(f"\n{'='*70}")
    print("PER-CLASS SKETCH ACCURACY")
    print(f"{'Class':12s} | {'ERM':>8}", end="")
    for m in ["dan_dg", "sam"]:
        if m in all_metrics:
            print(f" | {m:>8} | {'Δ':>7}", end="")
    print()
    print("-" * 70)
    if "erm" in all_metrics:
        erm_pc = all_metrics["erm"]["per_class_sketch"]
        for c in class_names:
            row = f"{c:12s} | {erm_pc.get(c, 0):8.4f}"
            for m in ["dan_dg", "sam"]:
                if m in all_metrics:
                    v = all_metrics[m]["per_class_sketch"].get(c, 0) or 0
                    d = v - (erm_pc.get(c) or 0)
                    row += f" | {v:8.4f} | {d:+7.4f}"
            print(row)

    print(f"\n{'='*50}")
    print("SHARPNESS PROXY (rho=0.05, 32 examples/domain)")
    print(f"{'Method':10s} | L(θ)     | L(θ+ε)  | Δ_sharp")
    print("-" * 50)
    for method in ["erm", "dan_dg", "sam"]:
        if method not in all_metrics:
            continue
        m = all_metrics[method]
        print(f"{method:10s} | {m['sharpness_loss_clean']:8.4f} | "
              f"{m['sharpness_loss_perturbed']:8.4f} | {m['sharpness_delta']:8.4f}")

    plot_training_curves(cfg["results_dir"], histories)

    if "erm" in all_metrics:
        erm_pc = all_metrics["erm"]["per_class_sketch"]
        others = {m: all_metrics[m]["per_class_sketch"]
                  for m in ["dan_dg", "sam"] if m in all_metrics}
        plot_per_class_delta(cfg["results_dir"], class_names, erm_pc, others)

    out = os.path.join(cfg["results_dir"], "final_eval_task3.json")
    with open(out, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nAll metrics saved → {out}")


if __name__ == "__main__":
    main()
