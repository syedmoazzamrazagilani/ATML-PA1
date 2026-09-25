import numpy as np
from sklearn.metrics import roc_auc_score


def compute_auroc(known_scores: np.ndarray,
                  unknown_scores: np.ndarray) -> float:
    """
    AUROC for known (label=0) vs unknown (label=1).

    known_scores  : unknownness scores for KNOWN examples (should be low)
    unknown_scores: unknownness scores for UNKNOWN examples (should be high)
    """
    labels = np.concatenate([np.zeros(len(known_scores)),
                              np.ones(len(unknown_scores))])
    scores = np.concatenate([known_scores, unknown_scores])
    return float(roc_auc_score(labels, scores))


def calibrate_threshold(val_scores: np.ndarray,
                         percentile: float = 95.0) -> float:
    """
    τ = p-th percentile of val unknownness scores.
    Accepts ~p% of known-val examples when applied.
    """
    return float(np.percentile(val_scores, percentile))


def rejection_rates(unknown_scores: np.ndarray,
                    threshold: float):
    """
    Fraction of unknown examples with u(x) > τ  (correctly rejected).
    Equivalently, 1 − FPR@95TPR.
    """
    return float((unknown_scores > threshold).mean())


def acceptance_rate(known_scores: np.ndarray, threshold: float) -> float:
    """Fraction of known examples accepted (u(x) ≤ τ). Should be ~0.95."""
    return float((known_scores <= threshold).mean())


def fpr_at_95tpr(known_scores: np.ndarray,
                 unknown_scores: np.ndarray) -> float:
    """
    FPR@95TPR: fraction of unknowns incorrectly accepted when the
    threshold is set so that 95% of knowns are accepted.
    Lower is better.
    """
    tau = calibrate_threshold(known_scores, percentile=95.0)
    return float((unknown_scores <= tau).mean())
