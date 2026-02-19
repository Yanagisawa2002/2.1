import json
from pathlib import Path

import pyarrow.parquet as pq


def test_graph_artifacts_have_required_stats() -> None:
    root = Path(__file__).resolve().parents[1]
    artifacts_dir = root / "artifacts"
    nodes_path = artifacts_dir / "nodes.parquet"
    edges_path = artifacts_dir / "edges.parquet"
    meta_path = artifacts_dir / "graph_meta.json"

    assert nodes_path.exists(), f"Missing artifact: {nodes_path}. Run: python -m src.data.build_graph"
    assert edges_path.exists(), f"Missing artifact: {edges_path}. Run: python -m src.data.build_graph"
    assert meta_path.exists(), f"Missing artifact: {meta_path}. Run: python -m src.data.build_graph"

    nodes_file = pq.ParquetFile(nodes_path)
    edges_file = pq.ParquetFile(edges_path)

    assert set(nodes_file.schema.names) == {"node_id", "type", "original_id"}
    assert set(edges_file.schema.names) == {"src_id", "dst_id", "rel"}
    assert nodes_file.metadata is not None
    assert edges_file.metadata is not None
    assert nodes_file.metadata.num_rows > 0
    assert edges_file.metadata.num_rows > 0

    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    node_type_counts = meta.get("node_type_counts")
    assert isinstance(node_type_counts, dict)
    assert node_type_counts
    for node_type, count in node_type_counts.items():
        assert isinstance(node_type, str)
        assert node_type.strip() != ""
        assert isinstance(count, int)
        assert count > 0

    relation_counts = meta.get("relation_counts")
    assert isinstance(relation_counts, dict)
    assert relation_counts.get("indication", 0) > 0

