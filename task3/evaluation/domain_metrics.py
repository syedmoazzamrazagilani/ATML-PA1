import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def evaluate_target(backbone, classifier, target_loader, class_names, device):
    """
    Returns a dict with overall accuracy, macro-F1, per-class accuracy,
    and confusion matrix.  Uses Sketch labels (only at final eval stage).
    """
    backbone.eval()
    classifier.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in target_loader:
            logits = classifier(backbone(x.to(device)))
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(y.numpy())

    preds  = np.array(all_preds)
    labels = np.array(all_labels)

    acc    = float(accuracy_score(labels, preds))
    f1     = float(f1_score(labels, preds, average="macro", zero_division=0))
    cm     = confusion_matrix(labels, preds).tolist()

    per_class = {}
    for c, name in enumerate(class_names):
        mask = labels == c
        if mask.sum() > 0:
            per_class[name] = float(accuracy_score(labels[mask], preds[mask]))
        else:
            per_class[name] = None

    return {
        "accuracy":       acc,
        "macro_f1":       f1,
        "per_class_acc":  per_class,
        "confusion_matrix": cm,
    }
