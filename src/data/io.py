from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

csv.field_size_limit(10_000_000)


DEFAULT_DATA_DIR = Path("data")
ENCODING_CANDIDATES: tuple[str, ...] = ("utf-8", "utf-8-sig", "cp1252")

EDGE_COLUMNS: tuple[str, ...] = ("relation", "x_id", "x_type", "y_id", "y_type")
DISEASE_PATHWAY_COLUMNS: tuple[str, ...] = ("disease_id", "pathway_id")
NODES_COLUMNS: tuple[str, ...] = ("id", "type", "name", "source")
EXTERNAL_TRANSDUCTIVE_COLUMNS: tuple[str, ...] = (
    "DrugBank_ID",
    "diseaseId",
    "targetId",
    "phase",
)
HIGH_ORDER_COLUMNS: tuple[str, ...] = (
    "drugbank_id",
    "evidence_level",
    "repurposing_potential",
    "rationale",
    "support_pmids",
    "mechanism_summary",
    "support_pmids_2",
    "protein_id",
    "pathway_id",
    "disease_id",
)
AUGMENTATION_STATS_KEYS: tuple[str, ...] = (
    "quads_total",
    "quads_used",
    "require_indication",
    "indication_pairs",
    "existing_drug_protein",
    "existing_protein_pathway",
    "existing_disease_pathway",
    "added_drug_protein",
    "added_protein_pathway",
    "added_disease_pathway",
)

EXTERNAL_TRANSDUCTIVE_FILES: tuple[str, ...] = (
    "transductive_0.5.csv",
    "transductive_1.0.csv",
    "transductive_2.0.csv",
    "transductive_3.0.csv",
    "transductive_4.0.csv",
)


class DataValidationError(ValueError):
    """Raised when a dataset fails validation."""


@dataclass(frozen=True)
class CSVTable:
    path: Path
    encoding: str
    columns: tuple[str, ...]
    row_count: int
    missing_counts: Mapping[str, int]
    sample_rows: tuple[Mapping[str, str], ...]

    def iter_rows(self):
        with self.path.open("r", encoding=self.encoding, newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if None in row and row[None]:
                    raise DataValidationError(
                        f"{self.path}: encountered unexpected extra fields during iteration: {row[None]!r}"
                    )
                yield {key: value for key, value in row.items() if key is not None}


@dataclass(frozen=True)
class JSONDocument:
    path: Path
    data: Mapping[str, Any]


def _resolve_data_dir(data_dir: str | Path) -> Path:
    path = Path(data_dir)
    if not path.exists():
        raise DataValidationError(f"Data directory does not exist: {path}")
    if not path.is_dir():
        raise DataValidationError(f"Expected data directory path, got file: {path}")
    return path


def _resolve_file(data_dir: str | Path, relative_path: str) -> Path:
    base = _resolve_data_dir(data_dir)
    path = base / relative_path
    if not path.exists():
        raise DataValidationError(f"Expected dataset file not found: {path}")
    if not path.is_file():
        raise DataValidationError(f"Expected file but found directory: {path}")
    return path


def _detect_csv_encoding(path: Path, expected_columns: Sequence[str]) -> str:
    errors: list[str] = []
    expected = tuple(expected_columns)

    for encoding in ENCODING_CANDIDATES:
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                reader = csv.DictReader(handle)
                columns = reader.fieldnames
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: decode failed ({exc})")
            continue

        if columns is None:
            raise DataValidationError(f"{path}: file is empty or missing a CSV header row")

        found = tuple(columns)
        if found == expected:
            return encoding

        errors.append(f"{encoding}: header mismatch, found {found!r}")

    attempted = ", ".join(ENCODING_CANDIDATES)
    detail = " | ".join(errors)
    raise DataValidationError(
        f"{path}: unable to decode with expected header {expected!r}. "
        f"attempted encodings: {attempted}. details: {detail}"
    )


def _validate_csv(
    *,
    path: Path,
    expected_columns: Sequence[str],
    required_non_empty: Sequence[str],
    sample_size: int = 2,
) -> CSVTable:
    expected = tuple(expected_columns)
    required_non_empty_set = set(required_non_empty)
    unknown_required = required_non_empty_set.difference(expected)
    if unknown_required:
        unknown = ", ".join(sorted(unknown_required))
        raise DataValidationError(
            f"{path}: required_non_empty contains unknown columns not in expected header: {unknown}"
        )

    encoding = _detect_csv_encoding(path, expected)
    missing_counts = {column: 0 for column in expected}
    sample_rows: list[dict[str, str]] = []
    row_count = 0

    with path.open("r", encoding=encoding, newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        if columns is None:
            raise DataValidationError(f"{path}: file is empty or missing a CSV header row")
        found = tuple(columns)
        if found != expected:
            raise DataValidationError(
                f"{path}: header changed between encoding detection and parse. "
                f"expected {expected!r}, found {found!r}"
            )

        for line_number, row in enumerate(reader, start=2):
            row_count += 1

            if None in row and row[None]:
                raise DataValidationError(
                    f"{path}: line {line_number} has {len(row[None])} unexpected extra fields: {row[None]!r}"
                )

            clean_row: dict[str, str] = {}
            for column in expected:
                value = row.get(column)
                if value is None:
                    raise DataValidationError(
                        f"{path}: line {line_number} missing value for required column '{column}'"
                    )

                clean_row[column] = value
                if value.strip() == "":
                    missing_counts[column] += 1
                    if column in required_non_empty_set:
                        raise DataValidationError(
                            f"{path}: line {line_number} column '{column}' is empty but required_non_empty"
                        )

            if len(sample_rows) < sample_size:
                sample_rows.append(clean_row)

    return CSVTable(
        path=path,
        encoding=encoding,
        columns=expected,
        row_count=row_count,
        missing_counts=missing_counts,
        sample_rows=tuple(sample_rows),
    )


def _validate_json(path: Path, required_keys: Sequence[str]) -> JSONDocument:
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise DataValidationError(f"{path}: expected utf-8 JSON, decode failed: {exc}") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DataValidationError(f"{path}: invalid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise DataValidationError(f"{path}: expected top-level JSON object, found {type(payload).__name__}")

    expected = set(required_keys)
    found = set(payload.keys())
    missing = sorted(expected - found)
    extra = sorted(found - expected)
    if missing or extra:
        raise DataValidationError(
            f"{path}: JSON keys mismatch. missing={missing!r}, extra={extra!r}, found={sorted(found)!r}"
        )

    return JSONDocument(path=path, data=payload)


def load_augmentation_stats(data_dir: str | Path = DEFAULT_DATA_DIR) -> JSONDocument:
    path = _resolve_file(data_dir, "ExtendedKG/augmentation_stats.json")
    return _validate_json(path, AUGMENTATION_STATS_KEYS)


def load_disease_pathway_direct_mapped(data_dir: str | Path = DEFAULT_DATA_DIR) -> CSVTable:
    path = _resolve_file(data_dir, "ExtendedKG/disease_pathway_direct_mapped.csv")
    return _validate_csv(
        path=path,
        expected_columns=DISEASE_PATHWAY_COLUMNS,
        required_non_empty=DISEASE_PATHWAY_COLUMNS,
    )


def load_disease_protein_data_subset(data_dir: str | Path = DEFAULT_DATA_DIR) -> CSVTable:
    path = _resolve_file(data_dir, "ExtendedKG/disease_protein_data_subset.csv")
    return _validate_csv(path=path, expected_columns=EDGE_COLUMNS, required_non_empty=EDGE_COLUMNS)


def load_indication_data_subset(data_dir: str | Path = DEFAULT_DATA_DIR) -> CSVTable:
    path = _resolve_file(data_dir, "ExtendedKG/indication_data_subset.csv")
    return _validate_csv(path=path, expected_columns=EDGE_COLUMNS, required_non_empty=EDGE_COLUMNS)


def load_kg_filtered_subset_ext_drugprotein(data_dir: str | Path = DEFAULT_DATA_DIR) -> CSVTable:
    path = _resolve_file(data_dir, "ExtendedKG/kg_filtered_subset_ext_drugprotein.csv")
    return _validate_csv(path=path, expected_columns=EDGE_COLUMNS, required_non_empty=EDGE_COLUMNS)


def load_nodes(data_dir: str | Path = DEFAULT_DATA_DIR) -> CSVTable:
    path = _resolve_file(data_dir, "ExtendedKG/nodes.csv")
    return _validate_csv(
        path=path,
        expected_columns=NODES_COLUMNS,
        required_non_empty=("id", "type"),
    )


def load_pathway_pathway_data_subset(data_dir: str | Path = DEFAULT_DATA_DIR) -> CSVTable:
    path = _resolve_file(data_dir, "ExtendedKG/pathway_pathway_data_subset.csv")
    return _validate_csv(path=path, expected_columns=EDGE_COLUMNS, required_non_empty=EDGE_COLUMNS)


def load_external_full_transductive(data_dir: str | Path = DEFAULT_DATA_DIR) -> Mapping[str, CSVTable]:
    loaded: dict[str, CSVTable] = {}
    for filename in EXTERNAL_TRANSDUCTIVE_FILES:
        path = _resolve_file(data_dir, f"External_Full/{filename}")
        loaded[filename] = _validate_csv(
            path=path,
            expected_columns=EXTERNAL_TRANSDUCTIVE_COLUMNS,
            required_non_empty=EXTERNAL_TRANSDUCTIVE_COLUMNS,
        )
    return loaded


def load_high_order(data_dir: str | Path = DEFAULT_DATA_DIR) -> CSVTable:
    path = _resolve_file(data_dir, "HighOrder/HO.csv")
    return _validate_csv(
        path=path,
        expected_columns=HIGH_ORDER_COLUMNS,
        required_non_empty=(
            "drugbank_id",
            "evidence_level",
            "repurposing_potential",
            "rationale",
            "protein_id",
        ),
    )


def load_all_datasets(data_dir: str | Path = DEFAULT_DATA_DIR) -> Mapping[str, CSVTable | JSONDocument]:
    loaded: dict[str, CSVTable | JSONDocument] = {
        "ExtendedKG/augmentation_stats.json": load_augmentation_stats(data_dir),
        "ExtendedKG/disease_pathway_direct_mapped.csv": load_disease_pathway_direct_mapped(data_dir),
        "ExtendedKG/disease_protein_data_subset.csv": load_disease_protein_data_subset(data_dir),
        "ExtendedKG/indication_data_subset.csv": load_indication_data_subset(data_dir),
        "ExtendedKG/kg_filtered_subset_ext_drugprotein.csv": load_kg_filtered_subset_ext_drugprotein(data_dir),
        "ExtendedKG/nodes.csv": load_nodes(data_dir),
        "ExtendedKG/pathway_pathway_data_subset.csv": load_pathway_pathway_data_subset(data_dir),
        "HighOrder/HO.csv": load_high_order(data_dir),
    }
    external = load_external_full_transductive(data_dir)
    for filename, table in external.items():
        loaded[f"External_Full/{filename}"] = table
    return loaded

