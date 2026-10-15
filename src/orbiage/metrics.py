"""Transparent, testable segment-level alert budget measures."""
import numpy as np

def top_k_true_positives(labels, scores, budget):
    """Positive labels among the highest-scoring segments (stable ties)."""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("labels and scores must be equal-length 1D arrays")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    if budget < 0 or budget > len(labels):
        raise ValueError("budget must be between 0 and the number of segments")
    selected = np.argsort(-scores, kind="stable")[:budget]
    return int(labels[selected].sum())

def normalized_rank(values):
    """Average ranks scaled to (0, 1]."""
    from scipy.stats import rankdata
    values = np.asarray(values)
    return rankdata(values, method="average") / len(values)

def blended_score(probabilities, uncertainty, weight):
    """Heuristic rank blend; NOT a calibrated failure probability."""
    if not 0 <= weight <= 1:
        raise ValueError("weight must be within [0,1]")
    if len(probabilities) != len(uncertainty):
        raise ValueError("input lengths must match")
    return (1-weight)*normalized_rank(probabilities)+weight*normalized_rank(uncertainty)
