from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from src.data.io import DataValidationError, load_high_order, load_indication_data_subset


PAIR_SCHEMA = pa.schema(
    [
        ("drug_id", pa.string()),
        ("disease_id", pa.string()),
    ]
)

QUADRUPLE_SCHEMA = pa.schema(
    [
        ("drug_id", pa.string()),
        ("disease_id", pa.string()),
        ("protein_id", pa.string()),
        ("pathway_id", pa.string()),
    ]
)


def _canonical_pair_from_indication_row(row: dict[str, str], row_index: int, file_path: Path) -> tuple[str, str]:
    relation = row["relation"]
    x_type = row["x_type"]
    y_type = row["y_type"]
    x_id = row["x_id"]
    y_id = row["y_id"]

    if relation != "indication":
        raise DataValidationError(
            f"{file_path}: row {row_index} has relation={relation!r}; expected 'indication'"
        )

    if x_type == "drug" and y_type == "disease":
        return x_id, y_id
    if x_type == "disease" and y_type == "drug":
        return y_id, x_id

    raise DataValidationError(
        f"{file_path}: row {row_index} has invalid indication types: x_type={x_type!r}, y_type={y_type!r}"
    )


def _normalize_pathway_id(pathway_id: str) -> str | None:
    value = pathway_id.strip()
    if not value:
        return None
    if value.startswith("pathway::"):
        return value
    prefix = "http://bioregistry.io/reactome:"
    if value.startswith(prefix):
        return f"pathway::{value[len(prefix):]}"
    return None


def _write_pairs(path: Path, pairs: Iterable[tuple[str, str]]) -> int:
    pairs_list = list(pairs)
    table = pa.table(
        {
            "drug_id": pa.array([p[0] for p in pairs_list], type=pa.string()),
            "disease_id": pa.array([p[1] for p in pairs_list], type=pa.string()),
        },
        schema=PAIR_SCHEMA,
    )
    pq.write_table(table, path)
    return len(pairs_list)


def _write_quadruples(path: Path, quadruples: Iterable[tuple[str, str, str, str]]) -> int:
    quads_list = list(quadruples)
    table = pa.table(
        {
            "drug_id": pa.array([q[0] for q in quads_list], type=pa.string()),
            "disease_id": pa.array([q[1] for q in quads_list], type=pa.string()),
            "protein_id": pa.array([q[2] for q in quads_list], type=pa.string()),
            "pathway_id": pa.array([q[3] for q in quads_list], type=pa.string()),
        },
        schema=QUADRUPLE_SCHEMA,
    )
    pq.write_table(table, path)
    return len(quads_list)


def _dedupe_preserve_order(
    rows: Iterable[tuple[str, str, str, str]]
) -> list[tuple[str, str, str, str]]:
    # Keep the first occurrence for deterministic behavior while removing duplicate supervision rows.
    return list(dict.fromkeys(rows))


def split_indication(
    *,
    data_dir: str | Path = "data",
    output_dir: str | Path = "artifacts/splits",
    seed: int = 0,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> dict[str, object]:
    if any(r < 0 for r in (train_ratio, val_ratio, test_ratio)):
        raise ValueError(
            f"Split ratios must be non-negative. got train={train_ratio}, val={val_ratio}, test={test_ratio}"
        )
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"Split ratios must sum to 1.0. got train={train_ratio}, val={val_ratio}, test={test_ratio}"
        )

    split_dir = Path(output_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    indication_table = load_indication_data_subset(data_dir)
    high_order_table = load_high_order(data_dir)

    unique_pairs: set[tuple[str, str]] = set()
    for row_index, row in enumerate(indication_table.iter_rows(), start=1):
        unique_pairs.add(_canonical_pair_from_indication_row(row, row_index, indication_table.path))

    if not unique_pairs:
        raise DataValidationError(f"{indication_table.path}: no indication pairs found")

    pairs = sorted(unique_pairs)
    rng = random.Random(seed)
    rng.shuffle(pairs)

    total_pairs = len(pairs)
    train_n = int(total_pairs * train_ratio)
    val_n = int(total_pairs * val_ratio)
    test_n = total_pairs - train_n - val_n

    train_pairs = pairs[:train_n]
    val_pairs = pairs[train_n : train_n + val_n]
    test_pairs = pairs[train_n + val_n :]

    if len(test_pairs) != test_n:
        raise RuntimeError("Internal split error: test pair count mismatch")

    pair_to_split: dict[tuple[str, str], str] = {}
    for pair in train_pairs:
        pair_to_split[pair] = "train"
    for pair in val_pairs:
        pair_to_split[pair] = "val"
    for pair in test_pairs:
        pair_to_split[pair] = "test"

    train_quads: list[tuple[str, str, str, str]] = []
    val_quads: list[tuple[str, str, str, str]] = []
    test_quads: list[tuple[str, str, str, str]] = []

    skipped_missing_fields = 0
    skipped_invalid_format = 0
    skipped_pair_not_in_indication = 0

    for row_index, row in enumerate(high_order_table.iter_rows(), start=1):
        drugbank_id = row["drugbank_id"].strip()
        disease_id = row["disease_id"].strip()
        protein_id = row["protein_id"].strip()
        pathway_id = row["pathway_id"].strip()

        if not drugbank_id or not disease_id or not protein_id or not pathway_id:
            skipped_missing_fields += 1
            continue

        drug_id = drugbank_id if drugbank_id.startswith("drug::") else f"drug::{drugbank_id}"
        if not disease_id.startswith("disease::"):
            skipped_invalid_format += 1
            continue
        if not protein_id.startswith("gene/protein::"):
            skipped_invalid_format += 1
            continue
        pathway_norm = _normalize_pathway_id(pathway_id)
        if pathway_norm is None:
            skipped_invalid_format += 1
            continue

        pair = (drug_id, disease_id)
        split = pair_to_split.get(pair)
        if split is None:
            skipped_pair_not_in_indication += 1
            continue

        quad = (drug_id, disease_id, protein_id, pathway_norm)
        if split == "train":
            train_quads.append(quad)
        elif split == "val":
            val_quads.append(quad)
        elif split == "test":
            test_quads.append(quad)
        else:  # pragma: no cover
            raise RuntimeError(f"Unexpected split key: {split!r}")

    train_quads_unique = _dedupe_preserve_order(train_quads)
    val_quads_unique = _dedupe_preserve_order(val_quads)
    test_quads_unique = _dedupe_preserve_order(test_quads)

    pair_counts = {
        "train": _write_pairs(split_dir / "train_pairs.parquet", train_pairs),
        "val": _write_pairs(split_dir / "val_pairs.parquet", val_pairs),
        "test": _write_pairs(split_dir / "test_pairs.parquet", test_pairs),
    }

    quad_counts = {
        "train": _write_quadruples(split_dir / "train_quadruples.parquet", train_quads_unique),
        "val": _write_quadruples(split_dir / "val_quadruples.parquet", val_quads_unique),
        "test": _write_quadruples(split_dir / "test_quadruples.parquet", test_quads_unique),
    }

    meta = {
        "seed": seed,
        "ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
        "pair_counts": pair_counts,
        "quadruple_counts": quad_counts,
        "high_order_filtering": {
            "source_rows": high_order_table.row_count,
            "used_rows": quad_counts["train"] + quad_counts["val"] + quad_counts["test"],
            "raw_split_rows": {
                "train": len(train_quads),
                "val": len(val_quads),
                "test": len(test_quads),
            },
            "skipped_missing_fields": skipped_missing_fields,
            "skipped_invalid_format": skipped_invalid_format,
            "skipped_pair_not_in_indication": skipped_pair_not_in_indication,
        },
        "files": {
            "train_pairs": str(split_dir / "train_pairs.parquet"),
            "val_pairs": str(split_dir / "val_pairs.parquet"),
            "test_pairs": str(split_dir / "test_pairs.parquet"),
            "train_quadruples": str(split_dir / "train_quadruples.parquet"),
            "val_quadruples": str(split_dir / "val_quadruples.parquet"),
            "test_quadruples": str(split_dir / "test_quadruples.parquet"),
        },
    }
    (split_dir / "split_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split indication pairs and mechanism quadruples without transductive leakage."
    )
    parser.add_argument("--data-dir", default="data", help="Input data directory (default: data)")
    parser.add_argument(
        "--output-dir",
        default="artifacts/splits",
        help="Output split artifact directory (default: artifacts/splits)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for pair split (default: 0)")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio (default: 0.8)")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Val split ratio (default: 0.1)")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio (default: 0.1)")
    args = parser.parse_args()

    meta = split_indication(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
