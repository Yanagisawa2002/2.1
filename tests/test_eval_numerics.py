from __future__ import annotations

import numpy as np
import pytest

from src.baseline_common import evaluate_link_prediction


def test_evaluate_rejects_non_finite_embeddings() -> None:
    node_embeddings = np.full((4, 3), np.nan, dtype=np.float32)
    relation_embedding = np.ones((3,), dtype=np.float32)
    eval_pairs = np.array([[0, 1]], dtype=np.int64)
    disease_candidates = np.array([1, 2, 3], dtype=np.int64)
    filter_tails_by_head = {0: {1}}

    with pytest.raises(ValueError, match="Non-finite embeddings"):
        evaluate_link_prediction(
            node_embeddings=node_embeddings,
            relation_embedding=relation_embedding,
            eval_pairs=eval_pairs,
            disease_candidates=disease_candidates,
            filter_tails_by_head=filter_tails_by_head,
            auprc_negatives_per_positive=1,
            seed=0,
            rank_batch_size=2,
        )
