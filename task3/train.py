import os
import json
import shutil
import yaml
import argparse
import itertools

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from shared.pacs_protocol import get_transforms
from task3.models.backbone import ResNet18Backbone, ClassifierHead
from task3.methods.erm import ERMLoss
from task3.methods.dan_dg import DANDGLoss
from task3.methods.sam import SAMOptimizer, ERMLoss as SAMERMLoss
from task3.selection.source_validation import evaluate_source_val


# --------------------------------------------------------------------------- #
#  Config helpers
# --------------------------------------------------------------------------- #

def load_config(method: str) -> dict:
    with open("task3/configs/base.yaml") as f:
        cfg = yaml.safe_load(f)
    with open(f"task3/configs/{method}.yaml") as f:
        cfg.update(yaml.safe_load(f))
    return cfg


# --------------------------------------------------------------------------- #
#  Data helpers
# --------------------------------------------------------------------------- #

def build_source_loaders(cfg, train_tf, eval_tf, splits):
    """Returns (list of train DataLoaders, dict of val DataLoaders)."""
    train_loaders, val_loaders = [], {}
    for domain in cfg["source_domains"]:
        ds_tr  = datasets.ImageFolder(os.path.join(cfg["data_dir"], domain), transform=train_tf)
        ds_val = datasets.ImageFolder(os.path.join(cfg["data_dir"], domain), transform=eval_tf)
        train_loaders.append(DataLoader(
            Subset(ds_tr, splits[domain]["train"]),
            batch_size=cfg["batch_size_per_source"], shuffle=True, drop_last=True))
        val_loaders[domain] = DataLoader(
            Subset(ds_val, splits[domain]["val"]),
            batch_size=32, shuffle=False)
    return train_loaders, val_loaders


# --------------------------------------------------------------------------- #
#  ERM
# --------------------------------------------------------------------------- #

def handle_erm(cfg):
    """
    The ERM baseline for Task 3 IS the source_only checkpoint from Task 2.
    Load it, evaluate source-side metrics, save under task3/checkpoints/erm_best.pth.
    No Sketch data is touched.
    """
    src  = cfg["task2_erm_checkpoint"]
    dst  = os.path.join(cfg["checkpoints_dir"], "erm_best.pth")
    os.makedirs(cfg["checkpoints_dir"], exist_ok=True)
    os.makedirs(cfg["results_dir"], exist_ok=True)

    if not os.path.exists(src):
        raise FileNotFoundError(
            f"Task 2 ERM checkpoint not found at '{src}'.\n"
            "Run task2.train --method source_only first.")

    shutil.copy2(src, dst)
    print(f"ERM checkpoint copied: {src} -> {dst}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tf, eval_tf = get_transforms()
    with open(cfg["splits_path"]) as f:
        splits = json.load(f)

    _, val_loaders = build_source_loaders(cfg, train_tf, eval_tf, splits)

    backbone   = ResNet18Backbone().to(device)
    classifier = ClassifierHead(num_classes=cfg["num_classes"]).to(device)
    ckpt = torch.load(dst, map_location=device, weights_only=False)
    backbone.load_state_dict(ckpt["backbone"])
    classifier.load_state_dict(ckpt["classifier"])

    metrics = evaluate_source_val(backbone, classifier, val_loaders, device)
    print("\nERM source-validation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    history = {"method": "erm", "source_val_final": metrics,
               "note": "Checkpoint loaded from Task 2; not retrained."}
    with open(os.path.join(cfg["results_dir"], "erm_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print("ERM history saved.")


# --------------------------------------------------------------------------- #
#  DAN-DG
# --------------------------------------------------------------------------- #

def train_dan_dg(cfg):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DAN-DG] device={device}  lambda_dg={cfg['lambda_dg']}")

    os.makedirs(cfg["checkpoints_dir"], exist_ok=True)
    os.makedirs(cfg["results_dir"], exist_ok=True)

    train_tf, eval_tf = get_transforms()
    with open(cfg["splits_path"]) as f:
        splits = json.load(f)

    train_loaders, val_loaders = build_source_loaders(cfg, train_tf, eval_tf, splits)

    backbone   = ResNet18Backbone().to(device)
    classifier = ClassifierHead(num_classes=cfg["num_classes"]).to(device)
    criterion  = DANDGLoss(lambda_dg=cfg["lambda_dg"])

    optimizer = torch.optim.AdamW(
        list(backbone.parameters()) + list(classifier.parameters()),
        lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    best_mean_f1    = 0.0
    patience_counter = 0
    total_batches   = min(len(l) for l in train_loaders)

    history = {
        "epoch": [], "cls_loss": [], "mmd_loss": [], "total_loss": [],
        "mean_source_f1": [], "mean_source_acc": [],
        "worst_source_f1": [], "worst_source_acc": [],
    }
    for domain in cfg["source_domains"]:
        history[f"{domain}_f1"]  = []
        history[f"{domain}_acc"] = []

    for epoch in range(cfg["max_epochs"]):
        backbone.train()
        classifier.train()

        src_iters = [iter(l) for l in train_loaders]
        ep_cls, ep_mmd, ep_total, nb = 0.0, 0.0, 0.0, 0

        for _ in range(total_batches):
            batches = [next(it) for it in src_iters]
            domain_feats  = []
            all_logits_list, all_labels_list = [], []

            optimizer.zero_grad()

            for x, y in batches:
                x, y = x.to(device), y.to(device)
                feat   = backbone(x)
                logits = classifier(feat)
                domain_feats.append(feat)
                all_logits_list.append(logits)
                all_labels_list.append(y)

            all_logits = torch.cat(all_logits_list, dim=0)
            all_labels = torch.cat(all_labels_list, dim=0)

            total_loss, cls_loss, mmd_loss = criterion(
                domain_feats, all_logits, all_labels)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(backbone.parameters()) + list(classifier.parameters()),
                max_norm=5.0)
            optimizer.step()

            ep_cls   += cls_loss.item()
            ep_mmd   += mmd_loss.item()
            ep_total += total_loss.item()
            nb       += 1

        # --- validation ---
        metrics = evaluate_source_val(backbone, classifier, val_loaders, device)
        mean_f1   = metrics["mean_f1"]
        mean_acc  = metrics["mean_acc"]
        worst_f1  = metrics["worst_f1"]
        worst_acc = metrics["worst_acc"]

        history["epoch"].append(epoch + 1)
        history["cls_loss"].append(ep_cls / nb)
        history["mmd_loss"].append(ep_mmd / nb)
        history["total_loss"].append(ep_total / nb)
        history["mean_source_f1"].append(mean_f1)
        history["mean_source_acc"].append(mean_acc)
        history["worst_source_f1"].append(worst_f1)
        history["worst_source_acc"].append(worst_acc)
        for domain in cfg["source_domains"]:
            history[f"{domain}_f1"].append(metrics[domain]["f1"])
            history[f"{domain}_acc"].append(metrics[domain]["acc"])

        print(
            f"Epoch {epoch+1:3d} | cls={ep_cls/nb:.4f}  mmd={ep_mmd/nb:.4f} | "
            f"mean_F1={mean_f1:.4f}  worst_F1={worst_f1:.4f}")
        for d in cfg["source_domains"]:
            print(f"       {d:15s}: F1={metrics[d]['f1']:.4f}  Acc={metrics[d]['acc']:.4f}")

        if mean_f1 > best_mean_f1:
            best_mean_f1     = mean_f1
            patience_counter = 0
            torch.save({
                "backbone":   backbone.state_dict(),
                "classifier": classifier.state_dict(),
                "epoch": epoch + 1,
                "mean_source_f1": mean_f1,
            }, os.path.join(cfg["checkpoints_dir"], "dan_dg_best.pth"))
            print(f"  -> Checkpoint saved (F1={best_mean_f1:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= cfg["patience"]:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    with open(os.path.join(cfg["results_dir"], "dan_dg_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print("DAN-DG history saved.")


# --------------------------------------------------------------------------- #
#  SAM
# --------------------------------------------------------------------------- #

def train_sam(cfg):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SAM] device={device}  rho={cfg['rho']}")

    os.makedirs(cfg["checkpoints_dir"], exist_ok=True)
    os.makedirs(cfg["results_dir"], exist_ok=True)

    train_tf, eval_tf = get_transforms()
    with open(cfg["splits_path"]) as f:
        splits = json.load(f)

    train_loaders, val_loaders = build_source_loaders(cfg, train_tf, eval_tf, splits)

    backbone   = ResNet18Backbone().to(device)
    classifier = ClassifierHead(num_classes=cfg["num_classes"]).to(device)
    criterion  = SAMERMLoss()

    base_opt = torch.optim.AdamW(
        list(backbone.parameters()) + list(classifier.parameters()),
        lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sam = SAMOptimizer(base_opt, rho=cfg["rho"])

    best_mean_f1     = 0.0
    patience_counter = 0
    total_batches    = min(len(l) for l in train_loaders)

    history = {
        "epoch": [], "cls_loss_ascent": [], "cls_loss_descent": [],
        "mean_source_f1": [], "mean_source_acc": [],
        "worst_source_f1": [], "worst_source_acc": [],
    }
    for domain in cfg["source_domains"]:
        history[f"{domain}_f1"]  = []
        history[f"{domain}_acc"] = []

    for epoch in range(cfg["max_epochs"]):
        backbone.train()
        classifier.train()

        src_iters = [iter(l) for l in train_loaders]
        ep_loss1, ep_loss2, nb = 0.0, 0.0, 0

        for _ in range(total_batches):
            batches = [next(it) for it in src_iters]
            x_cat = torch.cat([b[0] for b in batches], dim=0).to(device)
            y_cat = torch.cat([b[1] for b in batches], dim=0).to(device)

            backbone.train()   # BN policy re-applied via override
            classifier.train()
            logits1 = classifier(backbone(x_cat))
            loss1   = criterion(logits1, y_cat)
            loss1.backward()
            sam.first_step(zero_grad=True)
            ep_loss1 += loss1.item()

            backbone.train()
            classifier.train()
            logits2 = classifier(backbone(x_cat))
            loss2   = criterion(logits2, y_cat)
            loss2.backward()

            torch.nn.utils.clip_grad_norm_(
                list(backbone.parameters()) + list(classifier.parameters()),
                max_norm=5.0)
            sam.second_step(zero_grad=True)
            ep_loss2 += loss2.item()
            nb       += 1

        # --- validation ---
        metrics   = evaluate_source_val(backbone, classifier, val_loaders, device)
        mean_f1   = metrics["mean_f1"]
        mean_acc  = metrics["mean_acc"]
        worst_f1  = metrics["worst_f1"]
        worst_acc = metrics["worst_acc"]

        history["epoch"].append(epoch + 1)
        history["cls_loss_ascent"].append(ep_loss1 / nb)
        history["cls_loss_descent"].append(ep_loss2 / nb)
        history["mean_source_f1"].append(mean_f1)
        history["mean_source_acc"].append(mean_acc)
        history["worst_source_f1"].append(worst_f1)
        history["worst_source_acc"].append(worst_acc)
        for domain in cfg["source_domains"]:
            history[f"{domain}_f1"].append(metrics[domain]["f1"])
            history[f"{domain}_acc"].append(metrics[domain]["acc"])

        print(
            f"Epoch {epoch+1:3d} | loss_asc={ep_loss1/nb:.4f}  loss_desc={ep_loss2/nb:.4f} | "
            f"mean_F1={mean_f1:.4f}  worst_F1={worst_f1:.4f}")
        for d in cfg["source_domains"]:
            print(f"       {d:15s}: F1={metrics[d]['f1']:.4f}  Acc={metrics[d]['acc']:.4f}")

        if mean_f1 > best_mean_f1:
            best_mean_f1     = mean_f1
            patience_counter = 0
            torch.save({
                "backbone":   backbone.state_dict(),
                "classifier": classifier.state_dict(),
                "epoch": epoch + 1,
                "mean_source_f1": mean_f1,
            }, os.path.join(cfg["checkpoints_dir"], "sam_best.pth"))
            print(f"  -> Checkpoint saved (F1={best_mean_f1:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= cfg["patience"]:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    with open(os.path.join(cfg["results_dir"], "sam_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print("SAM history saved.")


# --------------------------------------------------------------------------- #
#  Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True,
                        choices=["erm", "dan_dg", "sam"])
    args = parser.parse_args()

    cfg = load_config(args.method)
    torch.manual_seed(cfg["seed"])

    if args.method == "erm":
        handle_erm(cfg)
    elif args.method == "dan_dg":
        train_dan_dg(cfg)
    elif args.method == "sam":
        train_sam(cfg)
