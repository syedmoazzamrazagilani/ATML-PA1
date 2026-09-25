import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def compute_source_separability(backbone, classifier, val_loaders,
                                device, seed=6304):
    """
    val_loaders : dict  domain -> DataLoader  (source validation only)
    Returns held-out accuracy (float, 0-1).
    """
    backbone.eval()
    classifier.eval()

    all_feats, all_domain_labels = [], []
    for d_idx, (domain, loader) in enumerate(val_loaders.items()):
        with torch.no_grad():
            for x, _ in loader:
                feats = backbone(x.to(device)).cpu().numpy()
                all_feats.append(feats)
                all_domain_labels.extend([d_idx] * len(feats))

    X = np.concatenate(all_feats, axis=0)
    y = np.array(all_domain_labels)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    split = int(0.7 * len(y))
    tr, te = idx[:split], idx[split:]

    clf = LogisticRegression(C=1, max_iter=1000, random_state=seed,
                             multi_class="multinomial", solver="lbfgs")
    clf.fit(X[tr], y[tr])
    acc = float(accuracy_score(y[te], clf.predict(X[te])))
    return acc
