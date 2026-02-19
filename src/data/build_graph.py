from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Mapping

from src.data.io import (
    CSVTable,
    DataValidationError,
    load_disease_pathway_direct_mapped,
    load_indication_data_subset,
    load_kg_filtered_subset_ext_drugprotein,
    load_nodes,
    load_pathway_pathway_data_subset,
)

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - exercised by runtime only
    raise ImportError(
        "pyarrow is required to build parquet artifacts. Install with: python -m pip install pyarrow"
    ) from exc


NODES_FILENAME = "nodes.parquet"
EDGES_FILENAME = "edges.parquet"
META_FILENAME = "graph_meta.json"

EDGE_SCHEMA = pa.schema(
    [
        ("src_id", pa.int64()),
        ("dst_id", pa.int64()),
        ("rel", pa.string()),
    ]
)


def _build_nodes(nodes_table: CSVTable, output_path: Path) -> tuple[dict[str, int], Counter[str]]:
    node_lookup: dict[str, int] = {}
    node_ids: list[int] = []
    node_types: list[str] = []
    original_ids: list[str] = []
    type_counts: Counter[str] = Counter()

    for row in nodes_table.iter_rows():
        original_id = row["id"]
        node_type = row["type"]

        if original_id in node_lookup:
            raise DataValidationError(
                f"{nodes_table.path}: duplicate node id detected: {original_id!r}"
            )

        node_id = len(node_lookup)
        node_lookup[original_id] = node_id
        node_ids.append(node_id)
        node_types.append(node_type)
        original_ids.append(original_id)
        type_counts[node_type] += 1

    if not node_lookup:
        raise DataValidationError(f"{nodes_table.path}: nodes file produced zero nodes")

    table = pa.table(
        {
            "node_id": pa.array(node_ids, type=pa.int64()),
            "type": pa.array(node_types, type=pa.string()),
            "original_id": pa.array(original_ids, type=pa.string()),
        }
    )
    pq.write_table(table, output_path)
    return node_lookup, type_counts


def _write_chunk(
    writer: pq.ParquetWriter,
    src_chunk: list[int],
    dst_chunk: list[int],
    rel_chunk: list[str],
) -> None:
    if not src_chunk:
        return

    table = pa.table(
        {
            "src_id": pa.array(src_chunk, type=pa.int64()),
            "dst_id": pa.array(dst_chunk, type=pa.int64()),
            "rel": pa.array(rel_chunk, type=pa.string()),
        },
        schema=EDGE_SCHEMA,
    )
    writer.write_table(table)
    src_chunk.clear()
    dst_chunk.clear()
    rel_chunk.clear()


def _append_edge_rows(
    *,
    table: CSVTable,
    node_lookup: Mapping[str, int],
    writer: pq.ParquetWriter,
    relation_counts: Counter[str],
    src_chunk: list[int],
    dst_chunk: list[int],
    rel_chunk: list[str],
    chunk_size: int,
    src_key: str,
    dst_key: str,
    rel_key: str | None = "relation",
    fixed_rel: str | None = None,
) -> int:
    if rel_key is None and fixed_rel is None:
        raise DataValidationError("Either rel_key or fixed_rel must be provided for edge ingestion")

    rows_written = 0
    for row_index, row in enumerate(table.iter_rows(), start=1):
        src_original = row[src_key]
        dst_original = row[dst_key]
        rel = fixed_rel if fixed_rel is not None else row[rel_key]  # type: ignore[index]

        src_id = node_lookup.get(src_original)
        if src_id is None:
            raise DataValidationError(
                f"{table.path}: row {row_index} source node not found in nodes.csv: {src_original!r}"
            )
        dst_id = node_lookup.get(dst_original)
        if dst_id is None:
            raise DataValidationError(
                f"{table.path}: row {row_index} destination node not found in nodes.csv: {dst_original!r}"
            )

        src_chunk.append(src_id)
        dst_chunk.append(dst_id)
        rel_chunk.append(rel)
        relation_counts[rel] += 1
        rows_written += 1

        if len(src_chunk) >= chunk_size:
            _write_chunk(writer, src_chunk, dst_chunk, rel_chunk)

    return rows_written


def build_graph(
    data_dir: str | Path = "data",
    artifacts_dir: str | Path = "artifacts",
    chunk_size: int = 1_000_000,
) -> dict[str, object]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    output_dir = Path(artifacts_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nodes_out = output_dir / NODES_FILENAME
    edges_out = output_dir / EDGES_FILENAME
    meta_out = output_dir / META_FILENAME

    nodes_table = load_nodes(data_dir)
    kg_table = load_kg_filtered_subset_ext_drugprotein(data_dir)
    indication_table = load_indication_data_subset(data_dir)
    disease_pathway_table = load_disease_pathway_direct_mapped(data_dir)
    pathway_pathway_table = load_pathway_pathway_data_subset(data_dir)

    node_lookup, node_type_counts = _build_nodes(nodes_table, nodes_out)

    relation_counts: Counter[str] = Counter()
    source_row_counts: dict[str, int] = {}
    total_edges = 0

    src_chunk: list[int] = []
    dst_chunk: list[int] = []
    rel_chunk: list[str] = []

    writer = pq.ParquetWriter(edges_out, EDGE_SCHEMA)
    try:
        count = _append_edge_rows(
            table=kg_table,
            node_lookup=node_lookup,
            writer=writer,
            relation_counts=relation_counts,
            src_chunk=src_chunk,
            dst_chunk=dst_chunk,
            rel_chunk=rel_chunk,
            chunk_size=chunk_size,
            src_key="x_id",
            dst_key="y_id",
            rel_key="relation",
        )
        source_row_counts["kg_filtered_subset_ext_drugprotein.csv"] = count
        total_edges += count

        count = _append_edge_rows(
            table=indication_table,
            node_lookup=node_lookup,
            writer=writer,
            relation_counts=relation_counts,
            src_chunk=src_chunk,
            dst_chunk=dst_chunk,
            rel_chunk=rel_chunk,
            chunk_size=chunk_size,
            src_key="x_id",
            dst_key="y_id",
            rel_key="relation",
        )
        source_row_counts["indication_data_subset.csv"] = count
        total_edges += count

        count = _append_edge_rows(
            table=disease_pathway_table,
            node_lookup=node_lookup,
            writer=writer,
            relation_counts=relation_counts,
            src_chunk=src_chunk,
            dst_chunk=dst_chunk,
            rel_chunk=rel_chunk,
            chunk_size=chunk_size,
            src_key="disease_id",
            dst_key="pathway_id",
            rel_key=None,
            fixed_rel="disease_pathway_direct_mapped",
        )
        source_row_counts["disease_pathway_direct_mapped.csv"] = count
        total_edges += count

        count = _append_edge_rows(
            table=pathway_pathway_table,
            node_lookup=node_lookup,
            writer=writer,
            relation_counts=relation_counts,
            src_chunk=src_chunk,
            dst_chunk=dst_chunk,
            rel_chunk=rel_chunk,
            chunk_size=chunk_size,
            src_key="x_id",
            dst_key="y_id",
            rel_key="relation",
        )
        source_row_counts["pathway_pathway_data_subset.csv"] = count
        total_edges += count

        _write_chunk(writer, src_chunk, dst_chunk, rel_chunk)
    finally:
        writer.close()

    indication_count = relation_counts.get("indication", 0)
    if indication_count <= 0:
        raise DataValidationError(
            "Built graph has zero 'indication' edges. Expected indication edges for transductive prediction."
        )

    meta: dict[str, object] = {
        "num_nodes": len(node_lookup),
        "num_edges": total_edges,
        "node_type_counts": dict(sorted(node_type_counts.items())),
        "relation_counts": dict(sorted(relation_counts.items())),
        "indication_edge_count": indication_count,
        "source_row_counts": source_row_counts,
        "files": {
            "nodes": str(nodes_out),
            "edges": str(edges_out),
            "meta": str(meta_out),
        },
    }
    meta_out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Build transductive training graph artifacts.")
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing input datasets (default: data)",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="Directory to write graph artifacts (default: artifacts)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000,
        help="Number of edges buffered before writing a parquet row group (default: 1000000)",
    )
    args = parser.parse_args()
    meta = build_graph(
        data_dir=args.data_dir,
        artifacts_dir=args.artifacts_dir,
        chunk_size=args.chunk_size,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

