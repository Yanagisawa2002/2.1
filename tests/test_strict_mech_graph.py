from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.baseline_common import QuadrupleSplit, build_mechanism_strict_blocklists
from src.models.mechanism_head import build_mechanism_adjacency


def test_build_mechanism_strict_blocklists_blocks_only_holdout_unique_edges() -> None:
    quad_splits = QuadrupleSplit(
        train=np.array(
            [
                [0, 10, 100, 1000],
                [1, 11, 101, 1001],
            ],
            dtype=np.int64,
        ),
        val=np.array(
            [
                [0, 12, 102, 1002],
                [1, 13, 101, 1003],
            ],
            dtype=np.int64,
        ),
        test=np.array(
            [
                [2, 14, 103, 1004],
                [1, 15, 101, 1001],
            ],
            dtype=np.int64,
        ),
    )

    blocklists = build_mechanism_strict_blocklists(quad_splits)

    assert blocklists.blocked_drug_protein_edges == {(0, 102), (2, 103)}
    assert blocklists.blocked_protein_pathway_edges == {(102, 1002), (101, 1003), (103, 1004)}
    assert blocklists.train_drug_protein_edge_count == 2
    assert blocklists.train_protein_pathway_edge_count == 2
    assert blocklists.holdout_drug_protein_edge_count == 3
    assert blocklists.holdout_protein_pathway_edge_count == 4


def test_build_mechanism_adjacency_respects_blocklists(tmp_path) -> None:
    edges_path = tmp_path / "edges.parquet"
    table = pa.table(
        {
            "src_id": [0, 0, 10, 1000, 1001],
            "dst_id": [100, 101, 100, 100, 101],
            "rel": [
                "drug_protein",
                "drug_protein",
                "disease_protein",
                "pathway_protein",
                "pathway_protein",
            ],
        }
    )
    pq.write_table(table, edges_path)

    node_types = {
        0: "drug",
        10: "disease",
        100: "gene/protein",
        101: "gene/protein",
        1000: "pathway",
        1001: "pathway",
    }
    adjacency = build_mechanism_adjacency(
        edges_path=edges_path,
        node_id_to_type=node_types,
        blocked_drug_protein_edges={(0, 101)},
        blocked_protein_pathway_edges={(101, 1001)},
        cache_dir=tmp_path / "cache",
    )

    assert adjacency.drug_to_proteins[0] == {100}
    assert adjacency.disease_to_proteins[10] == {100}
    assert adjacency.protein_to_pathways[100] == {1000}
    assert 101 not in adjacency.protein_to_pathways

    adjacency_cached = build_mechanism_adjacency(
        edges_path=edges_path,
        node_id_to_type=node_types,
        blocked_drug_protein_edges={(0, 101)},
        blocked_protein_pathway_edges={(101, 1001)},
        cache_dir=tmp_path / "cache",
    )
    assert adjacency_cached.drug_to_proteins == adjacency.drug_to_proteins
    assert adjacency_cached.disease_to_proteins == adjacency.disease_to_proteins
    assert adjacency_cached.protein_to_pathways == adjacency.protein_to_pathways
    assert len(list((tmp_path / "cache").glob("mechanism_adjacency_*.pkl"))) == 1
