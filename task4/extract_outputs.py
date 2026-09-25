import os
import argparse
import json

import torch
import numpy as np
import yaml

from task4.models.resnet_cifar import CifarResNet18, ProserResNet18
from task4.data.cifar10 import get_loaders
from task4.data.cifar100_unknowns import get_unknown_loaders


def load_model(method, device):
    if method in ("vanilla", "gcsc"):
        cfg_path = f"task4/configs/{method}.yaml"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        model     = CifarResNet18(num_classes=cfg["num_known_classes"]).to(device)
        ckpt_path = os.path.join(cfg["checkpoints_dir"], f"{method}_best.pth")
        ckpt      = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model, cfg

    elif method == "proser":
        with open("task4/configs/proser.yaml") as f:
            cfg = yaml.safe_load(f)
        ckpt_path = os.path.join(cfg["checkpoints_dir"], "proser_best.pth")
        ckpt      = torch.load(ckpt_path, map_location=device, weights_only=False)
        model     = ProserResNet18(
            num_known_classes = ckpt.get("num_known", cfg["num_known_classes"]),
            num_dummy_classes = ckpt.get("num_dummy", cfg["num_dummy_classes"]),
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model, cfg

    raise ValueError(f"Unknown method: {method}")


@torch.no_grad()
def extract(model, loader, device, method):
    feats_list, logits_list, labels_list = [], [], []
    for x, y in loader:
        x = x.to(device)
        if method == "proser":
            # known logits only for scoring and CSA
            logits, feat = model.known_logits(x)
        else:
            feat   = model.extract_features(x)
            logits = model.logits_from_features(feat)
        feats_list.append(feat.cpu())
        logits_list.append(logits.cpu())
        labels_list.append(y)
    return {
        "features": torch.cat(feats_list),
        "logits":   torch.cat(logits_list),
        "labels":   torch.cat(labels_list),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True,
                        choices=["vanilla", "gcsc", "proser"])
    args   = parser.parse_args()
    method = args.method

    with open(f"task4/configs/{method}.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = "task4/cache"
    os.makedirs(cache_dir, exist_ok=True)

    model, cfg = load_model(method, device)

    seed = cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, val_loader, test_loader, _, _ = get_loaders(
        data_dir   = cfg["data_dir"],
        batch_size = 256,
        seed       = seed,
        val_split  = cfg["cifar10_val_split"],
    )

    splits = {
        "cifar10_train": train_loader,
        "cifar10_val":   val_loader,
        "cifar10_test":  test_loader,
    }

    for split_name, loader in splits.items():
        out = os.path.join(cache_dir, f"{method}_{split_name}.pt")
        print(f"  Extracting {method} / {split_name} …")
        data = extract(model, loader, device, method)
        torch.save(data, out)
        print(f"    saved {data['features'].shape} → {out}")

    near_loader, far_loader, _ = get_unknown_loaders(
        data_dir=cfg["data_dir"], batch_size=256)

    for split_name, loader in [("cifar100_near", near_loader),
                                ("cifar100_far",  far_loader)]:
        out = os.path.join(cache_dir, f"{method}_{split_name}.pt")
        print(f"  Extracting {method} / {split_name} …")
        data = extract(model, loader, device, method)
        torch.save(data, out)
        print(f"    saved {data['features'].shape} → {out}")

    print(f"\nAll outputs cached for [{method}].")


if __name__ == "__main__":
    main()
