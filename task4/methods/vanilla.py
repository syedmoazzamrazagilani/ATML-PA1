import os
import json
import argparse
import yaml

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score

from task4.models.resnet_cifar import CifarResNet18
from task4.data.cifar10 import get_loaders


def load_config(method):
    path = f"task4/configs/{method}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def train(method="vanilla"):
    cfg    = load_config(method)
    seed   = cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{method}] device={device}")

    os.makedirs(cfg["checkpoints_dir"], exist_ok=True)
    os.makedirs(cfg["results_dir"],     exist_ok=True)

    use_ra  = (method == "gcsc")
    train_loader, val_loader, _, train_idx, val_idx = get_loaders(
        data_dir        = cfg["data_dir"],
        batch_size      = cfg["batch_size"],
        seed            = seed,
        val_split       = cfg["cifar10_val_split"],
        use_randaugment = use_ra,
        ra_num_ops      = cfg.get("randaugment_num_ops", 2),
        ra_magnitude    = cfg.get("randaugment_magnitude", 9),
    )

    splits_path = os.path.join(cfg["results_dir"], "cifar10_splits_seed6304.json")
    if not os.path.exists(splits_path):
        with open(splits_path, "w") as f:
            json.dump({"train_indices": train_idx.tolist(),
                       "val_indices":   val_idx.tolist()}, f, indent=2)

    model     = CifarResNet18(num_classes=cfg["num_known_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr           = cfg["lr"],
        momentum     = cfg["momentum"],
        weight_decay = cfg["weight_decay"],
        nesterov     = True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"])

    best_val_acc  = 0.0
    best_epoch    = 0
    history       = {"epoch": [], "train_loss": [], "val_acc": [], "lr": []}
    ckpt_path     = os.path.join(cfg["checkpoints_dir"], f"{method}_best.pth")

    for epoch in range(1, cfg["epochs"] + 1):
        # ── train ──────────────────────────────────────────────────────────
        model.train()
        total_loss, n_batches = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1
        scheduler.step()

        avg_loss = total_loss / n_batches
        cur_lr   = scheduler.get_last_lr()[0]

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                logits = model(x.to(device))
                preds.extend(logits.argmax(1).cpu().tolist())
                labels.extend(y.tolist())
        val_acc = float(accuracy_score(labels, preds))

        history["epoch"].append(epoch)
        history["train_loss"].append(avg_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(cur_lr)

        print(f"Epoch {epoch:3d}/{cfg['epochs']} | "
              f"loss={avg_loss:.4f}  val_acc={val_acc:.4f}  lr={cur_lr:.5f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            torch.save({"model_state": model.state_dict(),
                        "epoch":       epoch,
                        "val_acc":     val_acc}, ckpt_path)
            print(f"  -> Saved best checkpoint (val_acc={val_acc:.4f})")

    print(f"\nBest val_acc={best_val_acc:.4f} at epoch {best_epoch}")
    print(f"Checkpoint saved to {ckpt_path}")

    hist_path = os.path.join(cfg["results_dir"], f"{method}_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved to {hist_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="vanilla",
                        choices=["vanilla", "gcsc"])
    args = parser.parse_args()
    train(args.method)
