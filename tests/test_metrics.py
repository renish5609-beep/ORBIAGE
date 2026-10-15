import numpy as np
import pytest
from orbiage.metrics import top_k_true_positives, normalized_rank, blended_score

def test_review_budget():
    assert top_k_true_positives([1,0,1,0], [.9,.8,.4,.1], 2) == 1
    assert top_k_true_positives([1,0,1], [.2,.1,.9], 3) == 2

def test_zero_budget():
    assert top_k_true_positives([1,0], [.8,.1], 0) == 0

def test_tied_ranks():
    assert np.allclose(normalized_rank([1,1,2]), [0.5,0.5,1.0])

def test_zero_weight_matches_baseline_ranking():
    p=np.array([.1,.9,.4])
    u=np.array([.99,.05,.4])
    assert np.allclose(blended_score(p,u,0), normalized_rank(p))

def test_invalid_inputs():
    with pytest.raises(ValueError):
        top_k_true_positives([1], [.2], 2)
    with pytest.raises(ValueError):
        blended_score([.1], [.2], -1)
