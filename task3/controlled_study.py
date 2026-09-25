import os
import json
import yaml
import torch
import torch.nn as nn
import numpy as np
import itertools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from shared.pacs_protocol import get_transforms
from task3.models.backbone import ResNet18Backbone, ClassifierHead
from task3.methods.dan_dg import DANDGLoss
from task3.selection.source_validation import evaluate_source_val
from task3.evaluation.domain_metrics import evaluate_target
from task3.evaluation.source_domain_separability import compute_source_separability
from task3.evaluation.sharpness import compute_sharpness


def load_base_config():
    with open("task3/configs/base.yaml") as f:
        return yaml.safe_load(f)


def run_dan_dg(lambda_dg, cfg, device):
    """Train DAN-DG with a specific lambda and return all metrics."""
    print(f"\n--- DAN-DG  lambda_dg={lambda_dg} ---")
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    train_tf, eval_tf = get_transforms()
    with open(cfg["splits_path"]) as f:
        splits = json.load(f)

    train_loaders, val_loaders = [], {}
    for domain in cfg["source_domains"]:
        ds_tr  = datasets.ImageFolder(os.path.join(cfg["data_dir"], domain), transform=train_tf)
        ds_val = datasets.ImageFolder(os.path.join(cfg["data_dir"], domain), transform=eval_tf)
        train_loaders.append(DataLoader(
            Subset(ds_tr, splits[domain]["train"]),
            batch_size=cfg["batch_size_per_source"], shuffle=True, drop_last=True))
        val_loaders[domain] = DataLoader(
            Subset(ds_val, splits[domain]["val"]), batch_size=32, shuffle=False)

    sketch_ds = datasets.ImageFolder(
        os.path.join(cfg["data_dir"], cfg["target_domain"]), transform=eval_tf)
    sketch_loader = DataLoader(sketch_ds, batch_size=32, shuffle=False)
    class_names   = sketch_ds.classes

    backbone   = ResNet18Backbone().to(device)
    classifier = ClassifierHead(num_classes=cfg["num_classes"]).to(device)
    criterion  = DANDGLoss(lambda_dg=lambda_dg)
    optimizer  = torch.optim.AdamW(
        list(backbone.parameters()) + list(classifier.parameters()),
        lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    best_mean_f1 = 0.0
    patience_counter = 0
    total_batches = min(len(l) for l in train_loaders)
    best_state = None

    for epoch in range(cfg["max_epochs"]):
        backbone.train(); classifier.train()
        src_iters = [iter(l) for l in train_loaders]

        for _ in range(total_batches):
            batches = [next(it) for it in src_iters]
            domain_feats, all_logits_list, all_labels_list = [], [], []
            optimizer.zero_grad()
            for x, y in batches:
                x, y   = x.to(device), y.to(device)
                feat   = backbone(x)
                logits = classifier(feat)
                domain_feats.append(feat)
                all_logits_list.append(logits)
                all_labels_list.append(y)
            total_loss, _, _ = criterion(
                domain_feats,
                torch.cat(all_logits_list), torch.cat(all_labels_list))
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(backbone.parameters()) + list(classifier.parameters()), max_norm=5.0)
            optimizer.step()

        metrics  = evaluate_source_val(backbone, classifier, val_loaders, device)
        mean_f1  = metrics["mean_f1"]
        print(f"  Epoch {epoch+1:3d} | mean_src_F1={mean_f1:.4f}  worst={metrics['worst_f1']:.4f}")

        if mean_f1 > best_mean_f1:
            best_mean_f1 = mean_f1
            patience_counter = 0
            best_state = {
                "backbone":   {k: v.cpu().clone() for k, v in backbone.state_dict().items()},
                "classifier": {k: v.cpu().clone() for k, v in classifier.state_dict().items()},
            }
        else:
            patience_counter += 1
        if patience_counter >= cfg["patience"]:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    backbone.load_state_dict(best_state["backbone"])
    classifier.load_state_dict(best_state["classifier"])
    backbone.to(device); classifier.to(device)

    src_metrics = evaluate_source_val(backbone, classifier, val_loaders, device)
    tgt         = evaluate_target(backbone, classifier, sketch_loader, class_names, device)
    dom_sep     = compute_source_separability(backbone, classifier, val_loaders, device, seed=cfg["seed"])
    delta_sharp, _, _ = compute_sharpness(backbone, classifier, val_loaders, device,
                                          seed=cfg["seed"], rho=0.05)

    return {
        "lambda_dg":           lambda_dg,
        "mean_src_acc":        src_metrics["mean_acc"],
        "mean_src_f1":         src_metrics["mean_f1"],
        "worst_src_f1":        src_metrics["worst_f1"],
        "sketch_acc":          tgt["accuracy"],
        "sketch_f1":           tgt["macro_f1"],
        "source_domain_sep":   dom_sep,
        "sharpness_delta":     delta_sharp,
    }


def main():
    cfg    = load_base_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["results_dir"], exist_ok=True)

    print("\n=== Expectations ===")
    print("lambda=0.1 : weak alignment → source perf stable, dom_sep high (~ERM), "
          "sketch similar to ERM or slightly better")
    print("lambda=1.0 : main comparison setting → moderate alignment, "
          "some drop in dom_sep, moderate effect on sketch")
    print("lambda=10  : heavy alignment → dom_sep noticeably lower, "
          "risk of removing class-discriminative info, source perf may drop, "
          "sketch may degrade (cf. Task 2 DAN at lambda=10 which collapsed)\n")

    lambdas = [0.1, 1.0, 10.0]
    results = [run_dan_dg(lam, cfg, device) for lam in lambdas]

    print(f"\n{'lambda':>8} | {'SrcAcc':>8} | {'SrcF1':>7} | {'WrstF1':>7} | "
          f"{'SketchAcc':>9} | {'SketchF1':>8} | {'DomSep':>7} | {'Sharp':>7}")
    print("-" * 80)
    for r in results:
        print(f"{r['lambda_dg']:>8.1f} | {r['mean_src_acc']:8.4f} | {r['mean_src_f1']:7.4f} | "
              f"{r['worst_src_f1']:7.4f} | {r['sketch_acc']:9.4f} | {r['sketch_f1']:8.4f} | "
              f"{r['source_domain_sep']:7.4f} | {r['sharpness_delta']:7.4f}")

    lam_labels = [str(r["lambda_dg"]) for r in results]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].bar(lam_labels, [r["mean_src_acc"]   for r in results], color="steelblue")
    axes[0].set_title("Mean Source Val Acc"); axes[0].set_xlabel("λ_dg"); axes[0].set_ylim(0, 1)

    axes[1].bar(lam_labels, [r["sketch_acc"]      for r in results], color="tomato")
    axes[1].set_title("Sketch Acc");          axes[1].set_xlabel("λ_dg"); axes[1].set_ylim(0, 1)

    axes[2].bar(lam_labels, [r["source_domain_sep"] for r in results], color="seagreen")
    axes[2].axhline(1/3, linestyle="--", color="k", label="Chance (33.3%)")
    axes[2].set_title("Source Dom Sep");      axes[2].set_xlabel("λ_dg"); axes[2].set_ylim(0, 1)
    axes[2].legend(fontsize=7)

    axes[3].bar(lam_labels, [r["sharpness_delta"] for r in results], color="darkorange")
    axes[3].set_title("Sharpness Δ");         axes[3].set_xlabel("λ_dg")

    plt.suptitle("DAN-DG Controlled Study: Effect of λ_dg", fontsize=12)
    plt.tight_layout()
    out_png = os.path.join(cfg["results_dir"], "controlled_study_dan_dg.png")
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"\nPlot saved → {out_png}")

    out_json = os.path.join(cfg["results_dir"], "controlled_study_dan_dg.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {out_json}")


if __name__ == "__main__":
    main()
