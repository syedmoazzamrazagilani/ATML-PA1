import os
import json
import yaml
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import itertools

from shared.pacs_protocol import get_transforms
from task2.models.backbone import ResNet18Backbone, ClassifierHead
from task2.models.domain_discriminator import DomainDiscriminator, get_alpha_schedule
from task2.methods.source_only import SourceOnlyLoss
from task2.methods.dan import MMDLoss
from task2.methods.cdan import multilinear_conditioning
from task2.methods.dann import DANNLoss


def load_config(method):
    with open("task2/configs/base.yaml", "r") as f:
        config = yaml.safe_load(f)
    with open(f"task2/configs/{method}.yaml", "r") as f:
        method_config = yaml.safe_load(f)
    config.update(method_config)
    return config


def train(method):
    config = load_config(method)
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{method}] Using device: {device}")

    os.makedirs(config["checkpoints_dir"], exist_ok=True)
    os.makedirs(config["results_dir"], exist_ok=True)

    train_tf, eval_tf = get_transforms()

    with open(config["splits_path"], "r") as f:
        splits = json.load(f)

    # ------------------------------------------------------------------ #
    #  Data loaders
    # ------------------------------------------------------------------ #
    source_loaders = []
    val_loaders = {}

    for domain in config["source_domains"]:
        ds_train = datasets.ImageFolder(
            os.path.join(config["data_dir"], domain), transform=train_tf)
        ds_val = datasets.ImageFolder(
            os.path.join(config["data_dir"], domain), transform=eval_tf)

        source_loaders.append(DataLoader(
            Subset(ds_train, splits[domain]["train"]),
            batch_size=config["batch_size_per_source"], shuffle=True, drop_last=True))
        val_loaders[domain] = DataLoader(
            Subset(ds_val, splits[domain]["val"]),
            batch_size=32, shuffle=False)

    ds_target = datasets.ImageFolder(
        os.path.join(config["data_dir"], config["target_domain"]), transform=train_tf)
    target_loader = DataLoader(
        ds_target, batch_size=config["target_batch_size"], shuffle=True, drop_last=True)

    # ------------------------------------------------------------------ #
    #  Models
    # ------------------------------------------------------------------ #
    backbone   = ResNet18Backbone().to(device)
    classifier = ClassifierHead(num_classes=config["num_classes"]).to(device)

    backbone_params   = list(backbone.parameters())
    classifier_params = list(classifier.parameters())

    domain_disc = None
    disc_params  = []

    if method == "dann":
        domain_disc = DomainDiscriminator(in_features=512).to(device)
        disc_params = list(domain_disc.parameters())
    elif method == "cdan":
        domain_disc = DomainDiscriminator(
            in_features=512 * config["num_classes"]).to(device)
        disc_params = list(domain_disc.parameters())

    # ------------------------------------------------------------------ #
    #  Optimizer — discriminator gets a 10× higher LR (standard practice)
    # ------------------------------------------------------------------ #
    param_groups = [
        {"params": backbone_params,   "lr": config["lr"]},
        {"params": classifier_params, "lr": config["lr"]},
    ]
    if disc_params:
        param_groups.append({"params": disc_params, "lr": config["lr"] * 10})

    optimizer = torch.optim.AdamW(
        param_groups, weight_decay=config["weight_decay"])

    class_criterion  = nn.CrossEntropyLoss()
    mmd_criterion    = MMDLoss()   if method == "dan"              else None
    domain_criterion = DANNLoss()  if method in ["dann", "cdan"]   else None

    # ------------------------------------------------------------------ #
    #  Training loop
    # ------------------------------------------------------------------ #
    best_mean_f1    = 0.0
    patience_counter = 0
    total_batches   = min(len(l) for l in source_loaders)

    history = {
        "epoch": [], "cls_loss": [], "align_loss": [], "total_loss": [],
        "mean_source_f1": [], "mean_source_acc": [],
    }
    for domain in config["source_domains"]:
        history[f"{domain}_f1"]  = []
        history[f"{domain}_acc"] = []

    for epoch in range(config["max_epochs"]):
        backbone.train()
        classifier.train()
        if domain_disc:
            domain_disc.train()

        target_iter  = itertools.cycle(target_loader)
        source_iters = [iter(l) for l in source_loaders]

        ep_cls, ep_align, ep_total, nb = 0.0, 0.0, 0.0, 0

        for batch_idx in range(total_batches):
            p     = (epoch * total_batches + batch_idx) / (config["max_epochs"] * total_batches)
            alpha = get_alpha_schedule(p)

            source_x, source_y = [], []
            for s_iter in source_iters:
                x, y = next(s_iter)
                source_x.append(x)
                source_y.append(y)
            source_x = torch.cat(source_x, dim=0).to(device)
            source_y = torch.cat(source_y, dim=0).to(device)

            target_x, _ = next(target_iter)
            target_x = target_x.to(device)

            optimizer.zero_grad()

            source_feat  = backbone(source_x)
            source_logits = classifier(source_feat)
            cls_loss     = class_criterion(source_logits, source_y)

            loss       = cls_loss
            align_val  = 0.0

            if method == "dan":
                target_feat = backbone(target_x)
                align_loss  = config["lambda_mmd"] * mmd_criterion(source_feat, target_feat)
                loss       += align_loss
                align_val   = align_loss.item()

            elif method in ["dann", "cdan"]:
                target_feat   = backbone(target_x)
                target_logits = classifier(target_feat)

                if method == "cdan":
                    s_g = multilinear_conditioning(source_feat, source_logits)
                    t_g = multilinear_conditioning(target_feat, target_logits)
                else:
                    s_g = source_feat
                    t_g = target_feat

                d_labels = torch.cat([
                    torch.ones( source_x.size(0), dtype=torch.long),
                    torch.zeros(target_x.size(0), dtype=torch.long),
                ]).to(device)

                d_logits   = domain_disc(torch.cat([s_g, t_g], dim=0), alpha)
                align_loss = config["loss_weight"] * domain_criterion(d_logits, d_labels)
                loss      += align_loss
                align_val  = align_loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                backbone_params + classifier_params + disc_params,
                max_norm=5.0)

            optimizer.step()

            ep_cls   += cls_loss.item()
            ep_align += align_val
            ep_total += loss.item()
            nb       += 1

        # ---------------------------------------------------------------- #
        #  Validation
        # ---------------------------------------------------------------- #
        backbone.eval()
        classifier.eval()

        domain_f1s, domain_accs = [], []
        with torch.no_grad():
            for domain in config["source_domains"]:
                preds, labels = [], []
                for x, y in val_loaders[domain]:
                    logits = classifier(backbone(x.to(device)))
                    preds.extend(logits.argmax(1).cpu().numpy())
                    labels.extend(y.numpy())
                f1  = f1_score(labels, preds, average="macro", zero_division=0)
                acc = accuracy_score(labels, preds)
                domain_f1s.append(f1)
                domain_accs.append(acc)
                history[f"{domain}_f1"].append(f1)
                history[f"{domain}_acc"].append(acc)

        mean_f1  = float(np.mean(domain_f1s))
        mean_acc = float(np.mean(domain_accs))

        history["epoch"].append(epoch + 1)
        history["cls_loss"].append(ep_cls / nb)
        history["align_loss"].append(ep_align / nb)
        history["total_loss"].append(ep_total / nb)
        history["mean_source_f1"].append(mean_f1)
        history["mean_source_acc"].append(mean_acc)

        print(
            f"Epoch {epoch+1:3d} | "
            f"cls={ep_cls/nb:.4f}  align={ep_align/nb:.4f}  "
            f"src_F1={mean_f1:.4f}  src_Acc={mean_acc:.4f}"
        )
        for d, f, a in zip(config["source_domains"], domain_f1s, domain_accs):
            print(f"       {d:15s}: F1={f:.4f}  Acc={a:.4f}")

        if mean_f1 > best_mean_f1:
            best_mean_f1     = mean_f1
            patience_counter = 0
            torch.save({
                "backbone":        backbone.state_dict(),
                "classifier":      classifier.state_dict(),
                "epoch":           epoch + 1,
                "mean_source_f1":  mean_f1,
            }, os.path.join(config["checkpoints_dir"], f"{method}_best.pth"))
            print(f"  -> Saved best checkpoint (F1={best_mean_f1:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= config["patience"]:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    history_path = os.path.join(config["results_dir"], f"{method}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved → {history_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True,
                        choices=["source_only", "dan", "dann", "cdan"])
    args = parser.parse_args()
    train(args.method)
