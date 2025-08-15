import pytest
import numpy as np
from medical_qa_system.evaluation import recall_at_k, mrr_from_scores, dcg_at_k, ndcg_at_k, average_precision

def test_metrics_functions():
    sims = np.array([0.1, 0.2, 0.9])
    gold_idx = 2
    assert recall_at_k(sims, gold_idx, 1) == 1
    assert mrr_from_scores(sims, gold_idx) == 1.0
    rels = [1,0,0]
    assert dcg_at_k(rels, 3) > 0
    assert ndcg_at_k(sims, gold_idx, 3) > 0
    assert average_precision(sims, gold_idx) == 1.0

