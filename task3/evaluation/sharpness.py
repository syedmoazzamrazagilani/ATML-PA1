import torch
import torch.nn as nn
import numpy as np


def compute_sharpness(backbone, classifier, val_loaders,
                      device, seed=6304, rho=0.05,
                      examples_per_domain=32):
    """
    Returns delta_sharp = L(theta + eps) - L(theta).

    val_loaders : dict  domain -> DataLoader  (source validation loaders)
    """
    rng = np.random.default_rng(seed)
    criterion = nn.CrossEntropyLoss()

    all_x, all_y = [], []
    for domain, loader in val_loaders.items():
        xs, ys = [], []
        for x, y in loader:
            xs.append(x); ys.append(y)
            if sum(t.size(0) for t in xs) >= examples_per_domain:
                break
        xs = torch.cat(xs, dim=0)[:examples_per_domain]
        ys = torch.cat(ys, dim=0)[:examples_per_domain]
        all_x.append(xs); all_y.append(ys)

    x_batch = torch.cat(all_x, dim=0).to(device)
    y_batch = torch.cat(all_y, dim=0).to(device)

    backbone.eval()
    classifier.eval()

    for p in list(backbone.parameters()) + list(classifier.parameters()):
        p.requires_grad_(True)

    logits_clean = classifier(backbone(x_batch))
    loss_clean   = criterion(logits_clean, y_batch)

    grads = torch.autograd.grad(loss_clean,
                                list(backbone.parameters()) +
                                list(classifier.parameters()),
                                create_graph=False, allow_unused=True)

    flat_grad = torch.cat([g.detach().view(-1) for g in grads if g is not None])
    grad_norm  = flat_grad.norm(2).item()

    if grad_norm < 1e-12:
        return 0.0, loss_clean.item(), loss_clean.item()

    scale = rho / grad_norm

    params = list(backbone.parameters()) + list(classifier.parameters())
    perturbations = []
    for p, g in zip(params, grads):
        if g is not None:
            eps = g.detach() * scale
        else:
            eps = torch.zeros_like(p)
        perturbations.append(eps)
        p.data.add_(eps)

    with torch.no_grad():
        logits_perturbed = classifier(backbone(x_batch))
        loss_perturbed   = criterion(logits_perturbed, y_batch).item()

    loss_clean_val = loss_clean.item()

    for p, eps in zip(params, perturbations):
        p.data.sub_(eps)

    delta_sharp = loss_perturbed - loss_clean_val
    return float(delta_sharp), float(loss_clean_val), float(loss_perturbed)
