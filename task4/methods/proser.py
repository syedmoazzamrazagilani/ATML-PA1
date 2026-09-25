import os
import json
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score

from task4.models.resnet_cifar import ProserResNet18
from task4.data.cifar10 import get_loaders
from task4.methods.manifold_mixup import sample_inter_class_pairs, manifold_mixup


def load_config():
    with open("task4/configs/proser.yaml") as f:
        return yaml.safe_load(f)

def classifier_placeholder_loss(logits_all: torch.Tensor,
                                 labels:     torch.Tensor,
                                 num_known:  int,
                                 beta:       float) -> torch.Tensor:
    B = logits_all.size(0)
    device = logits_all.device

    logits_masked = logits_all.clone()
    for i in range(B):
        logits_masked[i, labels[i]] = float("-inf")
    targets = logits_masked.detach().argmax(dim=1)

    loss = F.cross_entropy(logits_masked, targets)
    return beta * loss


def data_placeholder_loss(logits_mixed: torch.Tensor,
                           num_known:   int,
                           gamma:       float) -> torch.Tensor:
    """
    Data-placeholder loss: push mixed features toward ANY dummy classifier.

    Uniform label distribution over the dummy positions:
      p_target[d] = 1 / num_dummy  for d in [num_known, num_known+num_dummy)
                  = 0              for d in [0, num_known)
    """
    num_total = logits_mixed.size(1)
    num_dummy = num_total - num_known

    target = torch.zeros_like(logits_mixed)
    target[:, num_known:] = 1.0 / num_dummy

    log_probs = F.log_softmax(logits_mixed, dim=1)
    loss      = -(target * log_probs).sum(dim=1).mean()
    return gamma * loss

def train():
    cfg  = load_config()
    seed = cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PROSER] device={device}")

    os.makedirs(cfg["checkpoints_dir"], exist_ok=True)
    os.makedirs(cfg["results_dir"],     exist_ok=True)

    train_loader, val_loader, _, _, _ = get_loaders(
        data_dir        = cfg["data_dir"],
        batch_size      = cfg["batch_size"],
        seed            = seed,
        val_split       = cfg["cifar10_val_split"],
        use_randaugment = False,
    )

    model = ProserResNet18.from_vanilla_ckpt(
        cfg["vanilla_ckpt"],
        num_known   = cfg["num_known_classes"],
        num_dummy   = cfg["num_dummy_classes"],
        device      = device,
    )

    criterion_cls = nn.CrossEntropyLoss()
    optimizer     = torch.optim.SGD(
        model.parameters(),
        lr           = cfg["lr"],
        momentum     = cfg["momentum"],
        weight_decay = cfg["weight_decay"],
        nesterov     = True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"])

    num_known = cfg["num_known_classes"]
    beta      = cfg["beta"]
    gamma     = cfg["gamma"]
    alpha_mix = cfg["mixup_alpha"]
    beta_mix  = cfg["mixup_beta"]

    best_val_acc = 0.0
    ckpt_path    = os.path.join(cfg["checkpoints_dir"], "proser_best.pth")
    history      = {"epoch": [], "cls_loss": [], "ph_loss": [],
                    "data_ph_loss": [], "total_loss": [], "val_acc": []}

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        ep_cls, ep_ph, ep_dph, ep_tot, nb = 0., 0., 0., 0., 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            B    = x.size(0)

            half    = B // 2
            x_cls   = x[:half];   y_cls  = y[:half]   # classifier-placeholder
            x_mix   = x[half:];   y_mix  = y[half:]   # data-placeholder

            optimizer.zero_grad()

            logits_cls, _ = model(x_cls)
            loss_ce   = criterion_cls(logits_cls[:, :num_known], y_cls)
            loss_cph  = classifier_placeholder_loss(
                logits_cls, y_cls, num_known, beta)

            if x_mix.size(0) >= 2:
                idx_a, idx_b = sample_inter_class_pairs(y_mix)
                h_a = model.forward_pre(x_mix[idx_a])
                h_b = model.forward_pre(x_mix[idx_b])
                h_mixed          = manifold_mixup(h_a, h_b, alpha_mix, beta_mix)
                logits_mixed, _  = model.forward_post(h_mixed)
                loss_dph         = data_placeholder_loss(
                    logits_mixed, num_known, gamma)
            else:
                loss_dph = torch.tensor(0.0, device=device)

            total_loss = loss_ce + loss_cph + loss_dph
            total_loss.backward()
            optimizer.step()

            ep_cls  += loss_ce.item()
            ep_ph   += loss_cph.item()
            ep_dph  += loss_dph.item() if isinstance(loss_dph, torch.Tensor) \
                       else loss_dph
            ep_tot  += total_loss.item()
            nb      += 1

        scheduler.step()

        model.eval()
        preds, labels_list = [], []
        with torch.no_grad():
            for xv, yv in val_loader:
                known_logits, _ = model.known_logits(xv.to(device))
                preds.extend(known_logits.argmax(1).cpu().tolist())
                labels_list.extend(yv.tolist())
        val_acc = float(accuracy_score(labels_list, preds))

        history["epoch"].append(epoch)
        history["cls_loss"].append(ep_cls / nb)
        history["ph_loss"].append(ep_ph / nb)
        history["data_ph_loss"].append(ep_dph / nb)
        history["total_loss"].append(ep_tot / nb)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:3d}/{cfg['epochs']} | "
              f"ce={ep_cls/nb:.4f}  cph={ep_ph/nb:.4f}  "
              f"dph={ep_dph/nb:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state": model.state_dict(),
                        "epoch":       epoch,
                        "val_acc":     val_acc,
                        "num_known":   num_known,
                        "num_dummy":   cfg["num_dummy_classes"]}, ckpt_path)
            print(f"  -> Saved best checkpoint (val_acc={val_acc:.4f})")

    hist_path = os.path.join(cfg["results_dir"], "proser_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nHistory saved to {hist_path}")
    print(f"Best val_acc={best_val_acc:.4f}")


if __name__ == "__main__":
    train()
