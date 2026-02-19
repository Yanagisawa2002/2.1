from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from src.data.io import DataValidationError


def _ensure_exists(path: Path) -> None:
    if not path.exists():
        raise DataValidationError(f"Missing required split artifact: {path}")
    if not path.is_file():
        raise DataValidationError(f"Expected file but found directory: {path}")


def _load_tuple_set(path: Path, columns: tuple[str, ...]) -> set[tuple[str, ...]]:
    _ensure_exists(path)

    table = pq.read_table(path)
    found_columns = tuple(table.column_names)
    if found_columns != columns:
        raise DataValidationError(
            f"{path}: invalid columns. expected {columns!r}, found {found_columns!r}"
        )

    arrays = [table[column].to_pylist() for column in columns]
    if not arrays:
        return set()

    length = len(arrays[0])
    for idx, arr in enumerate(arrays):
        if len(arr) != length:
            raise DataValidationError(
                f"{path}: column length mismatch at index {idx}; expected {length}, found {len(arr)}"
            )

    out: set[tuple[str, ...]] = set()
    for values in zip(*arrays, strict=True):
        cleaned = tuple("" if v is None else str(v) for v in values)
        out.add(cleaned)
    return out


def run_leakage_check(splits_dir: str | Path = "artifacts/splits") -> dict[str, object]:
    split_dir = Path(splits_dir)
    if not split_dir.exists():
        raise DataValidationError(f"Split directory does not exist: {split_dir}")
    if not split_dir.is_dir():
        raise DataValidationError(f"Expected split directory, got file: {split_dir}")

    pair_columns = ("drug_id", "disease_id")
    quad_columns = ("drug_id", "disease_id", "protein_id", "pathway_id")

    train_pairs = _load_tuple_set(split_dir / "train_pairs.parquet", pair_columns)
    val_pairs = _load_tuple_set(split_dir / "val_pairs.parquet", pair_columns)
    test_pairs = _load_tuple_set(split_dir / "test_pairs.parquet", pair_columns)

    train_quads = _load_tuple_set(split_dir / "train_quadruples.parquet", quad_columns)
    val_quads = _load_tuple_set(split_dir / "val_quadruples.parquet", quad_columns)
    test_quads = _load_tuple_set(split_dir / "test_quadruples.parquet", quad_columns)

    pair_overlap_train_val = train_pairs.intersection(val_pairs)
    pair_overlap_train_test = train_pairs.intersection(test_pairs)
    pair_overlap_val_test = val_pairs.intersection(test_pairs)
    if pair_overlap_train_val:
        raise DataValidationError(
            "Leakage detected: validation pair appears in train. "
            f"examples={sorted(pair_overlap_train_val)[:5]!r}"
        )
    if pair_overlap_train_test:
        raise DataValidationError(
            "Leakage detected: test pair appears in train. "
            f"examples={sorted(pair_overlap_train_test)[:5]!r}"
        )
    if pair_overlap_val_test:
        raise DataValidationError(
            "Leakage detected: test pair appears in validation. "
            f"examples={sorted(pair_overlap_val_test)[:5]!r}"
        )

    train_quad_pairs = {(q[0], q[1]) for q in train_quads}
    val_quad_pairs = {(q[0], q[1]) for q in val_quads}
    test_quad_pairs = {(q[0], q[1]) for q in test_quads}

    quad_pair_overlap_val = train_pairs.intersection(val_quad_pairs)
    quad_pair_overlap_test = train_pairs.intersection(test_quad_pairs)
    if quad_pair_overlap_val:
        raise DataValidationError(
            "Leakage detected: validation quadruple pair (drug,disease) appears in train pairs. "
            f"examples={sorted(quad_pair_overlap_val)[:5]!r}"
        )
    if quad_pair_overlap_test:
        raise DataValidationError(
            "Leakage detected: test quadruple pair (drug,disease) appears in train pairs. "
            f"examples={sorted(quad_pair_overlap_test)[:5]!r}"
        )

    missing_train_pair_membership = train_quad_pairs.difference(train_pairs)
    missing_val_pair_membership = val_quad_pairs.difference(val_pairs)
    missing_test_pair_membership = test_quad_pairs.difference(test_pairs)
    if missing_train_pair_membership:
        raise DataValidationError(
            "Split consistency error: some train quadruple (drug,disease) pairs are not in train pairs. "
            f"examples={sorted(missing_train_pair_membership)[:5]!r}"
        )
    if missing_val_pair_membership:
        raise DataValidationError(
            "Split consistency error: some val quadruple (drug,disease) pairs are not in val pairs. "
            f"examples={sorted(missing_val_pair_membership)[:5]!r}"
        )
    if missing_test_pair_membership:
        raise DataValidationError(
            "Split consistency error: some test quadruple (drug,disease) pairs are not in test pairs. "
            f"examples={sorted(missing_test_pair_membership)[:5]!r}"
        )

    quad_overlap_train_val = train_quads.intersection(val_quads)
    quad_overlap_train_test = train_quads.intersection(test_quads)
    quad_overlap_val_test = val_quads.intersection(test_quads)
    if quad_overlap_train_val:
        raise DataValidationError(
            "Leakage detected: validation full quadruple appears in train. "
            f"examples={sorted(quad_overlap_train_val)[:5]!r}"
        )
    if quad_overlap_train_test:
        raise DataValidationError(
            "Leakage detected: test full quadruple appears in train. "
            f"examples={sorted(quad_overlap_train_test)[:5]!r}"
        )
    if quad_overlap_val_test:
        raise DataValidationError(
            "Leakage detected: test full quadruple appears in validation. "
            f"examples={sorted(quad_overlap_val_test)[:5]!r}"
        )

    report = {
        "pairs": {
            "train": len(train_pairs),
            "val": len(val_pairs),
            "test": len(test_pairs),
            "overlap_train_val": len(pair_overlap_train_val),
            "overlap_train_test": len(pair_overlap_train_test),
            "overlap_val_test": len(pair_overlap_val_test),
        },
        "quadruples": {
            "train": len(train_quads),
            "val": len(val_quads),
            "test": len(test_quads),
            "pair_membership_missing_train": len(missing_train_pair_membership),
            "pair_membership_missing_val": len(missing_val_pair_membership),
            "pair_membership_missing_test": len(missing_test_pair_membership),
            "pair_overlap_train_val": len(quad_pair_overlap_val),
            "pair_overlap_train_test": len(quad_pair_overlap_test),
            "full_overlap_train_val": len(quad_overlap_train_val),
            "full_overlap_train_test": len(quad_overlap_train_test),
            "full_overlap_val_test": len(quad_overlap_val_test),
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Check train/val/test split leakage for pairs and quadruples.")
    parser.add_argument(
        "--splits-dir",
        default="artifacts/splits",
        help="Directory containing split parquet artifacts (default: artifacts/splits)",
    )
    args = parser.parse_args()

    report = run_leakage_check(args.splits_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
