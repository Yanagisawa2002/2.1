from __future__ import annotations

import numpy as np

from src.models.mechanism_head import MechanismAdjacency, MechanismHead


def test_positive_only_quadruple_loss_uses_observed_targets_without_candidate_negatives() -> None:
    adjacency = MechanismAdjacency(
        drug_to_proteins={},
        disease_to_proteins={},
        protein_to_pathways={},
    )
    head = MechanismHead(
        embedding_dim=4,
        adjacency=adjacency,
        seed=0,
    )
    node_embeddings = np.random.default_rng(0).normal(0.0, 0.1, size=(5, 4)).astype(np.float32)
    quadruples = np.array([[0, 1, 3, 4]], dtype=np.int64)

    out = head.compute_losses_and_grads(
        node_embeddings=node_embeddings,
        quadruples=quadruples,
        protein_only=False,
    )

    assert int(out["protein_count"]) == 1
    assert int(out["pathway_count"]) == 1
    assert float(out["loss_protein"]) > 0.0
    assert float(out["loss_pathway"]) > 0.0

    grad_node_protein = out["grad_node_protein"]
    grad_node_pathway = out["grad_node_pathway"]
    assert isinstance(grad_node_protein, np.ndarray)
    assert isinstance(grad_node_pathway, np.ndarray)
    assert np.linalg.norm(grad_node_protein[3]) > 0.0
    assert np.linalg.norm(grad_node_pathway[4]) > 0.0
