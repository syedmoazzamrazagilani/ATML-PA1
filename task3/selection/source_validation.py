from sklearn.metrics import f1_score
import numpy as np


def evaluate_source_val(backbone, classifier, val_loaders, device):
    """
    Returns per-domain accuracy, per-domain macro-F1, mean accuracy,
    mean F1, and worst-domain F1.
    val_loaders : dict  domain_name -> DataLoader
    """
    import torch
    backbone.eval()
    classifier.eval()

    domain_results = {}
    with torch.no_grad():
        for domain, loader in val_loaders.items():
            preds, labels = [], []
            for x, y in loader:
                logits = classifier(backbone(x.to(device)))
                preds.extend(logits.argmax(1).cpu().numpy())
                labels.extend(y.numpy())
            acc = float(np.mean(np.array(preds) == np.array(labels)))
            f1  = float(f1_score(labels, preds, average="macro", zero_division=0))
            domain_results[domain] = {"acc": acc, "f1": f1}

    accs = [v["acc"] for v in domain_results.values()]
    f1s  = [v["f1"]  for v in domain_results.values()]
    domain_results["mean_acc"]   = float(np.mean(accs))
    domain_results["mean_f1"]    = float(np.mean(f1s))
    domain_results["worst_acc"]  = float(np.min(accs))
    domain_results["worst_f1"]   = float(np.min(f1s))
    return domain_results
