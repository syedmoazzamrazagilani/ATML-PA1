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
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from shared.pacs_protocol import get_transforms
from task2.models.backbone import ResNet18Backbone, ClassifierHead
from task2.methods.dan import MMDLoss


def load_base_config():
    with open("task2/configs/base.yaml", "r") as f:
        return yaml.safe_load(f)


def domain_separability_score(source_feats, target_feats, seed=6304):
    n_min = min(len(source_feats), len(target_feats), 1500)
    rng = np.random.default_rng(seed)
    si = rng.choice(len(source_feats), n_min, replace=False)
    ti = rng.choice(len(target_feats), n_min, replace=False)
    X = np.vstack([source_feats[si], target_feats[ti]])
    y = np.concatenate([np.zeros(n_min), np.ones(n_min)])
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    split = int(0.7 * len(y))
    tr, te = idx[:split], idx[split:]
    clf = LogisticRegression(C=1, max_iter=1000, random_state=seed)
    clf.fit(X[tr], y[tr])
    return float(accuracy_score(y[te], clf.predict(X[te])))


def extract_features(backbone, classifier, loader, device):
    feats_list, logits_list, labels_list = [], [], []
    backbone.eval(); classifier.eval()
    with torch.no_grad():
        for x, y in loader:
            f = backbone(x.to(device))
            l = classifier(f)
            feats_list.append(f.cpu().numpy())
            logits_list.append(l.cpu().numpy())
            labels_list.append(y.numpy())
    return (np.concatenate(feats_list),
            np.concatenate(logits_list),
            np.concatenate(labels_list))


def run_dan_lambda(lambda_mmd, config, device):
    """Train a DAN model with a specific lambda_mmd and return val+target metrics."""
    print(f"\n--- DAN lambda_mmd = {lambda_mmd} ---")
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    train_tf, eval_tf = get_transforms()

    with open(config["splits_path"]) as f:
        splits = json.load(f)

    source_loaders, val_loaders = [], {}
    for domain in config["source_domains"]:
        ds_tr = datasets.ImageFolder(os.path.join(config["data_dir"], domain), transform=train_tf)
        ds_val = datasets.ImageFolder(os.path.join(config["data_dir"], domain), transform=eval_tf)
        source_loaders.append(DataLoader(Subset(ds_tr, splits[domain]["train"]),
                                         batch_size=config["batch_size_per_source"],
                                         shuffle=True, drop_last=True))
        val_loaders[domain] = DataLoader(Subset(ds_val, splits[domain]["val"]),
                                         batch_size=32, shuffle=False)

    ds_tgt = datasets.ImageFolder(os.path.join(config["data_dir"], config["target_domain"]),
                                   transform=train_tf)
    target_loader = DataLoader(ds_tgt, batch_size=config["target_batch_size"],
                                shuffle=True, drop_last=True)

    backbone = ResNet18Backbone().to(device)
    classifier = ClassifierHead(num_classes=config["num_classes"]).to(device)
    mmd_crit = MMDLoss()
    cls_crit = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        list(backbone.parameters()) + list(classifier.parameters()),
        lr=config["lr"], weight_decay=config["weight_decay"]
    )

    best_mean_f1 = 0.0
    patience_counter = 0
    best_state = None
    total_batches = min(len(l) for l in source_loaders)

    history = {"epoch": [], "cls_loss": [], "mmd_loss": [], "mean_src_f1": []}

    for epoch in range(config["max_epochs"]):
        backbone.train(); classifier.train()
        target_iter = itertools.cycle(target_loader)
        src_iters = [iter(l) for l in source_loaders]
        ep_cls, ep_mmd, nb = 0.0, 0.0, 0

        for _ in range(total_batches):
            sx, sy = [], []
            for si in src_iters:
                x, y = next(si)
                sx.append(x); sy.append(y)
            sx = torch.cat(sx).to(device)
            sy = torch.cat(sy).to(device)
            tx, _ = next(target_iter)
            tx = tx.to(device)

            optimizer.zero_grad()
            sf = backbone(sx); tl = backbone(tx)
            cls_loss = cls_crit(classifier(sf), sy)
            mmd_loss = lambda_mmd * mmd_crit(sf, tl)
            (cls_loss + mmd_loss).backward()
            optimizer.step()
            ep_cls += cls_loss.item(); ep_mmd += mmd_loss.item(); nb += 1

        backbone.eval(); classifier.eval()
        f1s = []
        with torch.no_grad():
            for domain, vl in val_loaders.items():
                preds, labs = [], []
                for x, y in vl:
                    preds.extend(classifier(backbone(x.to(device))).argmax(1).cpu().numpy())
                    labs.extend(y.numpy())
                f1s.append(f1_score(labs, preds, average="macro", zero_division=0))
        mean_f1 = float(np.mean(f1s))
        history["epoch"].append(epoch + 1)
        history["cls_loss"].append(ep_cls / nb)
        history["mmd_loss"].append(ep_mmd / nb)
        history["mean_src_f1"].append(mean_f1)
        print(f"  Epoch {epoch+1:3d} | cls={ep_cls/nb:.4f} mmd={ep_mmd/nb:.4f} src_f1={mean_f1:.4f}")

        if mean_f1 > best_mean_f1:
            best_mean_f1 = mean_f1
            patience_counter = 0
            best_state = {
                "backbone": {k: v.cpu().clone() for k, v in backbone.state_dict().items()},
                "classifier": {k: v.cpu().clone() for k, v in classifier.state_dict().items()},
            }
        else:
            patience_counter += 1
        if patience_counter >= config["patience"]:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    backbone.load_state_dict(best_state["backbone"])
    classifier.load_state_dict(best_state["classifier"])
    backbone.to(device).eval(); classifier.to(device).eval()

    ds_tgt_eval = datasets.ImageFolder(
        os.path.join(config["data_dir"], config["target_domain"]),
        transform=eval_tf
    )
    tgt_loader_eval = DataLoader(ds_tgt_eval, batch_size=32, shuffle=False)
    tgt_feats, tgt_logits, tgt_labels = extract_features(backbone, classifier, tgt_loader_eval, device)
    tgt_acc = float(accuracy_score(tgt_labels, np.argmax(tgt_logits, 1)))
    tgt_f1  = float(f1_score(tgt_labels, np.argmax(tgt_logits, 1), average="macro", zero_division=0))

    src_feats_all = []
    val_accs = []
    with torch.no_grad():
        for domain in config["source_domains"]:
            feats, logits, labels = extract_features(backbone, classifier, val_loaders[domain], device)
            src_feats_all.append(feats)
            val_accs.append(float(accuracy_score(labels, np.argmax(logits, 1))))
    src_feats_all = np.concatenate(src_feats_all, axis=0)
    mean_src_acc = float(np.mean(val_accs))

    dom_sep = domain_separability_score(src_feats_all, tgt_feats, seed=config["seed"])

    return {
        "lambda_mmd": lambda_mmd,
        "best_mean_src_f1": best_mean_f1,
        "mean_src_acc": mean_src_acc,
        "target_acc": tgt_acc,
        "target_f1": tgt_f1,
        "domain_separability": dom_sep,
        "history": history,
    }


def main():
    config = load_base_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- State expectations BEFORE looking at results ---
    print("\nExpectations:")
    print("  lambda=0.1 (weak): low MMD pressure → src perf stable, dom_sep high, tgt ~ source_only")
    print("  lambda=1.0 (main): moderate alignment → see main comparison results")
    print("  lambda=10  (strong): heavy MMD → may hurt class structure, dom_sep lower, tgt may drop")

    lambdas = [0.1, 1.0, 10.0]
    results = []

    for lam in lambdas:
        res = run_dan_lambda(lam, config, device)
        results.append(res)

    print(f"\n{'lambda_mmd':>12} | {'Src Acc':>8} | {'Tgt Acc':>8} | {'Tgt F1':>8} | {'Dom Sep':>8}")
    print("-" * 60)
    for r in results:
        print(f"{r['lambda_mmd']:>12.1f} | {r['mean_src_acc']:8.4f} | {r['target_acc']:8.4f} | "
              f"{r['target_f1']:8.4f} | {r['domain_separability']:8.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    lam_labels = [str(r["lambda_mmd"]) for r in results]

    axes[0].bar(lam_labels, [r["mean_src_acc"] for r in results], color="steelblue")
    axes[0].set_title("Mean Source Val Acc"); axes[0].set_xlabel("λ_mmd"); axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Accuracy")

    axes[1].bar(lam_labels, [r["target_acc"] for r in results], color="tomato")
    axes[1].set_title("Target (Sketch) Acc"); axes[1].set_xlabel("λ_mmd"); axes[1].set_ylim(0, 1)

    axes[2].bar(lam_labels, [r["domain_separability"] for r in results], color="seagreen")
    axes[2].axhline(0.5, linestyle="--", color="k", label="Chance (50%)")
    axes[2].set_title("Domain Separability"); axes[2].set_xlabel("λ_mmd"); axes[2].set_ylim(0, 1)
    axes[2].legend()

    plt.suptitle("DAN Controlled Study: Effect of λ_mmd", fontsize=12)
    plt.tight_layout()
    out_dir = config["results_dir"]
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "controlled_study_dan.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\nPlot saved to {out}")

    serializable = []
    for r in results:
        entry = dict(r)
        del entry["history"]   # keep results lean; history is per-run noise
        serializable.append(entry)
    out_json = os.path.join(out_dir, "controlled_study_dan.json")
    with open(out_json, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Results saved to {out_json}")


if __name__ == "__main__":
    main()
