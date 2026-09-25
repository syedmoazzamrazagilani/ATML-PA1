import os
import json
import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from task4.scores import msp, mls, energy, mahalanobis, proser_score
from task4.evaluation.metrics import (
    compute_auroc, calibrate_threshold,
    rejection_rates, acceptance_rate, fpr_at_95tpr
)
from task4.evaluation.failure_analysis import (
    find_failures, print_failures
)
from task4.data.cifar100_unknowns import (
    get_unknown_class_labels, NEAR_CLASSES, FAR_CLASSES
)

CACHE_DIR   = "task4/cache"
RESULTS_DIR = "task4/results"
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def _load(method, split):
    path = os.path.join(CACHE_DIR, f"{method}_{split}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cache not found: {path}\n"
            f"Run:  python -m task4.extract_outputs --method {method}")
    return torch.load(path, map_location="cpu", weights_only=False)


def _get_data(method):
    """Returns (val_data, test_data, near_data, far_data)."""
    return (_load(method, "cifar10_val"),
            _load(method, "cifar10_test"),
            _load(method, "cifar100_near"),
            _load(method, "cifar100_far"))

def _fit_mahalanobis(method):
    train_data = _load(method, "cifar10_train")
    means, inv_cov = mahalanobis.fit_params(
        train_data["features"], train_data["labels"], num_classes=10)
    return means, inv_cov

def _compute_scores(logits, features, means=None, inv_cov=None,
                    proser_logits_all=None, num_known=10):
    """Returns dict of {score_name: np.ndarray}."""
    scores = {
        "MSP":    msp.score(logits).numpy(),
        "MLS":    mls.score(logits).numpy(),
        "Energy": energy.score(logits).numpy(),
    }
    if means is not None and inv_cov is not None:
        scores["Mahalanobis"] = mahalanobis.score(
            features, means, inv_cov).numpy()
    if proser_logits_all is not None:
        scores["PROSER"] = proser_score.score(
            proser_logits_all, num_known=num_known).numpy()
    return scores


# ── Table 1: post-hoc scores on vanilla ─────────────────────────────────────

def table1_posthoc(results):
    """MSP / MLS / Energy / Mahalanobis on vanilla model."""
    print("\n" + "="*70)
    print("TABLE 1: Post-hoc scores on Vanilla model")
    print("="*70)

    val_d, test_d, near_d, far_d = _get_data("vanilla")
    means, inv_cov = _fit_mahalanobis("vanilla")

    val_scores  = _compute_scores(val_d["logits"],  val_d["features"],
                                   means, inv_cov)
    test_scores = _compute_scores(test_d["logits"], test_d["features"],
                                   means, inv_cov)
    near_scores = _compute_scores(near_d["logits"], near_d["features"],
                                   means, inv_cov)
    far_scores  = _compute_scores(far_d["logits"],  far_d["features"],
                                   means, inv_cov)
    all_scores  = {k: np.concatenate([near_scores[k], far_scores[k]])
                   for k in near_scores}

    score_names = ["MSP", "MLS", "Energy", "Mahalanobis"]
    table1_rows = {}

    hdr = f"{'Score':<14} | {'AUROC Near':>11} | {'AUROC Far':>9} | " \
          f"{'AUROC All':>9} | {'Acc@95TPR':>10} | {'Rej Near':>9} | " \
          f"{'Rej Far':>8} | {'FPR@95 Near':>12} | {'FPR@95 Far':>10}"
    print(hdr)
    print("─" * len(hdr))

    for sn in score_names:
        ks   = test_scores[sn]        # known test scores
        vs   = val_scores[sn]         # known val scores  (for threshold)
        ns   = near_scores[sn]
        fs   = far_scores[sn]
        aus  = all_scores[sn]

        auroc_near = compute_auroc(ks, ns)
        auroc_far  = compute_auroc(ks, fs)
        auroc_all  = compute_auroc(ks, aus)

        tau        = calibrate_threshold(vs, 95.0)
        acc_rate   = acceptance_rate(ks, tau)
        rej_near   = rejection_rates(ns, tau)
        rej_far    = rejection_rates(fs, tau)
        fpr_near   = fpr_at_95tpr(vs, ns)
        fpr_far    = fpr_at_95tpr(vs, fs)

        row = {
            "auroc_near": auroc_near, "auroc_far": auroc_far,
            "auroc_all":  auroc_all,  "threshold": tau,
            "known_acceptance": acc_rate,
            "rej_near": rej_near,     "rej_far": rej_far,
            "fpr95_near": fpr_near,   "fpr95_far": fpr_far,
        }
        table1_rows[sn] = row

        print(f"{sn:<14} | {auroc_near:11.4f} | {auroc_far:9.4f} | "
              f"{auroc_all:9.4f} | {acc_rate:10.4f} | {rej_near:9.4f} | "
              f"{rej_far:8.4f} | {fpr_near:12.4f} | {fpr_far:10.4f}")

    results["table1_vanilla_posthoc"] = table1_rows

    # Closed-set accuracy on test
    test_preds  = test_d["logits"].argmax(1).numpy()
    test_labels = test_d["labels"].numpy()
    csa = float((test_preds == test_labels).mean())
    results["vanilla_csa"] = csa
    print(f"\nVanilla CSA (CIFAR-10 test): {csa:.4f}")

    return val_scores, test_scores, near_scores, far_scores


# ── Table 2: Vanilla / GCSC / PROSER with MLS ────────────────────────────────

def table2_model_comparison(results):
    print("\n" + "="*70)
    print("TABLE 2: Model comparison  (MLS + PROSER placeholder score)")
    print("="*70)

    rows = {}
    hdr  = (f"{'Model+Score':<22} | {'CSA':>6} | {'AUROC Near':>11} | "
            f"{'AUROC Far':>9} | {'AUROC All':>9} | {'Rej Near':>9} | "
            f"{'Rej Far':>8} | {'FPR@95 Near':>12}")
    print(hdr)
    print("─" * len(hdr))

    entries = [
        ("vanilla",  "MLS",    None),
        ("gcsc",     "MLS",    None),
        ("proser",   "MLS",    None),
        ("proser",   "PROSER", None),
    ]

    for method, sn, _ in entries:
        try:
            val_d, test_d, near_d, far_d = _get_data(method)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue

        test_preds  = test_d["logits"].argmax(1).numpy()
        test_labels = test_d["labels"].numpy()
        csa = float((test_preds == test_labels).mean())

        if sn == "PROSER":
            try:
                full_val_logits, full_near_logits, full_far_logits, \
                    full_test_logits = _load_proser_full_logits()
                val_s  = proser_score.score(full_val_logits,  num_known=10).numpy()
                test_s = proser_score.score(full_test_logits, num_known=10).numpy()
                near_s = proser_score.score(full_near_logits, num_known=10).numpy()
                far_s  = proser_score.score(full_far_logits,  num_known=10).numpy()
            except Exception as ex:
                print(f"  [PROSER-score] Could not compute: {ex}")
                continue
        else:
            score_fn = {"MLS": mls.score, "MSP": msp.score,
                        "Energy": energy.score}[sn]
            val_s  = score_fn(val_d["logits"]).numpy()
            test_s = score_fn(test_d["logits"]).numpy()
            near_s = score_fn(near_d["logits"]).numpy()
            far_s  = score_fn(far_d["logits"]).numpy()

        all_s = np.concatenate([near_s, far_s])
        tau   = calibrate_threshold(val_s, 95.0)

        auroc_near = compute_auroc(test_s, near_s)
        auroc_far  = compute_auroc(test_s, far_s)
        auroc_all  = compute_auroc(test_s, all_s)
        rej_near   = rejection_rates(near_s, tau)
        rej_far    = rejection_rates(far_s,  tau)
        fpr_near   = fpr_at_95tpr(val_s, near_s)

        key = f"{method}_{sn}"
        rows[key] = {
            "csa": csa, "auroc_near": auroc_near,
            "auroc_far": auroc_far,  "auroc_all": auroc_all,
            "rej_near": rej_near,    "rej_far": rej_far,
            "fpr95_near": fpr_near,  "threshold": float(tau),
        }
        print(f"{key:<22} | {csa:6.4f} | {auroc_near:11.4f} | "
              f"{auroc_far:9.4f} | {auroc_all:9.4f} | {rej_near:9.4f} | "
              f"{rej_far:8.4f} | {fpr_near:12.4f}")

    results["table2_model_comparison"] = rows


def _load_proser_full_logits():
    """
    Re-runs inference on PROSER model with ALL (known+dummy) logits.
    Needed for the placeholder detection score only.
    """
    import yaml
    from task4.models.resnet_cifar import ProserResNet18
    from task4.data.cifar10 import get_loaders
    from task4.data.cifar100_unknowns import get_unknown_loaders

    with open("task4/configs/proser.yaml") as f:
        cfg = yaml.safe_load(f)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join(cfg["checkpoints_dir"], "proser_best.pth")
    ckpt      = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = ProserResNet18(
        num_known_classes = ckpt.get("num_known", cfg["num_known_classes"]),
        num_dummy_classes = ckpt.get("num_dummy", cfg["num_dummy_classes"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    _, val_loader, test_loader, _, _ = get_loaders(
        cfg["data_dir"], batch_size=256, seed=cfg["seed"],
        val_split=cfg["cifar10_val_split"])
    near_loader, far_loader, _ = get_unknown_loaders(cfg["data_dir"])

    @torch.no_grad()
    def _infer(loader):
        all_logits = []
        for x, _ in loader:
            logits, _ = model(x.to(device))
            all_logits.append(logits.cpu())
        return torch.cat(all_logits)

    return (_infer(val_loader), _infer(near_loader),
            _infer(far_loader), _infer(test_loader))


# ── Score distribution / ROC figure ──────────────────────────────────────────

def plot_score_figure(results):
    """Multi-panel: score distributions + ROC curves for MSP, MLS, Mahalanobis."""
    print("\nGenerating score-distribution / ROC figure …")

    val_d, test_d, near_d, far_d = _get_data("vanilla")
    means, inv_cov = _fit_mahalanobis("vanilla")

    test_scores = _compute_scores(test_d["logits"], test_d["features"],
                                   means, inv_cov)
    near_scores = _compute_scores(near_d["logits"], near_d["features"],
                                   means, inv_cov)
    far_scores  = _compute_scores(far_d["logits"],  far_d["features"],
                                   means, inv_cov)

    score_names = ["MSP", "MLS", "Mahalanobis"]
    fig, axes   = plt.subplots(2, 3, figsize=(15, 9))

    for col, sn in enumerate(score_names):
        ks = test_scores[sn]
        ns = near_scores[sn]
        fs = far_scores[sn]

        ax = axes[0, col]
        bins = 50
        lo   = min(ks.min(), ns.min(), fs.min())
        hi   = max(ks.max(), ns.max(), fs.max())
        rng  = (float(lo), float(hi))

        ax.hist(ks, bins=bins, range=rng, density=True,
                alpha=0.6, color="steelblue", label="Known (CIFAR-10 test)")
        ax.hist(ns, bins=bins, range=rng, density=True,
                alpha=0.5, color="tomato",    label="Near unknown")
        ax.hist(fs, bins=bins, range=rng, density=True,
                alpha=0.5, color="orange",    label="Far unknown")
        ax.set_title(f"{sn} — Score Distribution")
        ax.set_xlabel("Unknownness score")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)

        ax2 = axes[1, col]
        for unk, unk_name, color in [(ns, "Near", "tomato"),
                                      (fs, "Far", "orange")]:
            labels  = np.concatenate([np.zeros(len(ks)), np.ones(len(unk))])
            scores_ = np.concatenate([ks, unk])
            fpr, tpr, _ = roc_curve(labels, scores_)
            roc_auc     = auc(fpr, tpr)
            ax2.plot(fpr, tpr, color=color,
                     label=f"{unk_name} (AUC={roc_auc:.3f})", linewidth=2)

        ax2.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax2.set_title(f"{sn} — ROC Curve")
        ax2.set_xlabel("FPR")
        ax2.set_ylabel("TPR")
        ax2.legend(fontsize=8)
        ax2.set_xlim([0, 1]); ax2.set_ylim([0, 1])

    plt.suptitle("Task 4: Post-hoc Score Analysis (Vanilla Model)", fontsize=13)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "osr_score_figure.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved score figure → {out}")


# ── Failure analysis ──────────────────────────────────────────────────────────

def failure_analysis(results):
    print("\n" + "="*70)
    print("FAILURE ANALYSIS: MLS on Vanilla  (incorrectly accepted unknowns)")
    print("="*70)

    with open("task4/configs/vanilla.yaml") as f:
        cfg = yaml.safe_load(f)

    val_d, test_d, near_d, far_d = _get_data("vanilla")
    val_mls  = mls.score(val_d["logits"]).numpy()
    near_mls = mls.score(near_d["logits"]).numpy()
    far_mls  = mls.score(far_d["logits"]).numpy()
    tau      = calibrate_threshold(val_mls, 95.0)

    near_class_names, far_class_names = get_unknown_class_labels(cfg["data_dir"])

    near_failures = find_failures(near_mls, near_d["logits"].numpy(),
                                   near_class_names, tau,
                                   n_samples=6, score_name="MLS")
    far_failures  = find_failures(far_mls,  far_d["logits"].numpy(),
                                   far_class_names,  tau,
                                   n_samples=6, score_name="MLS")

    print_failures(near_failures, "Near-Unknown Failures (MLS, Vanilla)")
    print_failures(far_failures,  "Far-Unknown Failures  (MLS, Vanilla)")

    results["failure_analysis"] = {
        "threshold_mls_vanilla":  float(tau),
        "near_failures":          near_failures,
        "far_failures":           far_failures,
    }


# ── Training curve plot ───────────────────────────────────────────────────────

def plot_training_curves():
    methods   = ["vanilla", "gcsc", "proser"]
    colors    = {"vanilla": "C0", "gcsc": "C1", "proser": "C2"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for method in methods:
        hist_path = os.path.join(RESULTS_DIR, f"{method}_history.json")
        if not os.path.exists(hist_path):
            continue
        with open(hist_path) as f:
            h = json.load(f)
        col = colors[method]

        if "train_loss" in h:
            axes[0].plot(h["epoch"], h["train_loss"],
                         label=method, color=col, linewidth=2)
        elif "cls_loss" in h:
            axes[0].plot(h["epoch"], h["cls_loss"],
                         label=f"{method} CE", color=col, linewidth=2)
            if "ph_loss" in h:
                axes[0].plot(h["epoch"], h["ph_loss"],
                             label=f"{method} CLP", color=col,
                             linestyle="--", linewidth=1.5)

        axes[1].plot(h["epoch"], h["val_acc"],
                     label=method, color=col, linewidth=2)

    axes[0].set_title("Training Loss");  axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss");          axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Validation Accuracy"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val Acc");            axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Task 4: Training Curves", fontsize=12)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "training_curves.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Training curves saved → {out}")

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = {}

    val_scores, test_scores, near_scores, far_scores = table1_posthoc(results)
    table2_model_comparison(results)
    plot_score_figure(results)
    failure_analysis(results)
    plot_training_curves()

    out_path = os.path.join(RESULTS_DIR, "osr_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll OSR results saved → {out_path}")


if __name__ == "__main__":
    main()
