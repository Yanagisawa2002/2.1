from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.models.rgcn import build_relation_graph


def test_build_relation_graph_filters_indication_to_train_pairs(tmp_path: Path) -> None:
    edges_path = tmp_path / "edges.parquet"
    table = pa.table(
        {
            "src_id": [0, 0, 10, 11, 12],
            "dst_id": [1, 2, 20, 21, 22],
            "rel": [
                "indication",
                "indication",
                "drug_protein",
                "drug_protein",
                "disease_protein",
            ],
        }
    )
    pq.write_table(table, edges_path)

    graph = build_relation_graph(
        edges_path=edges_path,
        num_nodes=32,
        keep_train_indication_only=True,
        train_indication_pairs={(0, 1)},
    )

    assert "indication" in graph.relations
    ind_edges = graph.edge_index_by_rel["indication"]
    assert ind_edges.shape == (2, 1)
    assert int(ind_edges[0, 0]) == 0
    assert int(ind_edges[1, 0]) == 1
    assert np.allclose(graph.edge_weight_by_rel["indication"], np.array([1.0], dtype=np.float32))


def test_build_relation_graph_can_drop_relations(tmp_path: Path) -> None:
    edges_path = tmp_path / "edges.parquet"
    table = pa.table(
        {
            "src_id": [0, 1, 2],
            "dst_id": [3, 4, 5],
            "rel": ["indication", "drug_protein", "disease_protein"],
        }
    )
    pq.write_table(table, edges_path)

    graph = build_relation_graph(
        edges_path=edges_path,
        num_nodes=16,
        include_relations={"drug_protein"},
        keep_train_indication_only=False,
    )

    assert graph.relations == ("drug_protein",)
    assert graph.edge_index_by_rel["drug_protein"].shape == (2, 1)
