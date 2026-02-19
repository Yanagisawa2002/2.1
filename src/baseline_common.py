from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pyarrow.parquet as pq
import yaml


@dataclass(frozen=True)
class PairSplit:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


@dataclass(frozen=True)
class QuadrupleSplit:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


@dataclass(frozen=True)
class MechanismStrictBlocklists:
    blocked_drug_protein_edges: set[tuple[int, int]]
    blocked_protein_pathway_edges: set[tuple[int, int]]
    train_drug_protein_edge_count: int
    train_protein_pathway_edge_count: int
    holdout_drug_protein_edge_count: int
    holdout_protein_pathway_edge_count: int


@dataclass(frozen=True)
class NodeIndex:
    original_to_node_id: dict[str, int]
    original_to_type: dict[str, str]
    node_id_to_original: dict[int, str]
    node_id_to_type: dict[int, str]
    disease_node_ids: np.ndarray
    protein_node_ids: np.ndarray
    pathway_node_ids: np.ndarray
    num_nodes: int


def read_yaml(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be a mapping: {cfg_path}")
    return payload


def write_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_pair_parquet(path: str | Path) -> tuple[list[str], list[str]]:
    pair_path = Path(path)
    if not pair_path.exists():
        raise FileNotFoundError(f"Pair parquet file not found: {pair_path}")
    table = pq.read_table(pair_path, columns=["drug_id", "disease_id"])
    return table["drug_id"].to_pylist(), table["disease_id"].to_pylist()


def _normalize_pathway_id(pathway_id: str) -> str | None:
    value = pathway_id.strip()
    if not value:
        return None
    prefix = "http://bioregistry.io/reactome:"
    if value.startswith(prefix):
        return f"pathway::{value[len(prefix):]}"
    if value.startswith("pathway::"):
        return value
    return None


def load_node_index(nodes_path: str | Path) -> NodeIndex:
    path = Path(nodes_path)
    if not path.exists():
        raise FileNotFoundError(f"Nodes parquet file not found: {path}")
    table = pq.read_table(path, columns=["node_id", "type", "original_id"])

    node_ids = table["node_id"].to_pylist()
    node_types = table["type"].to_pylist()
    original_ids = table["original_id"].to_pylist()

    original_to_node_id: dict[str, int] = {}
    original_to_type: dict[str, str] = {}
    node_id_to_original: dict[int, str] = {}
    node_id_to_type: dict[int, str] = {}
    disease_ids: list[int] = []
    protein_ids: list[int] = []
    pathway_ids: list[int] = []
    max_node_id = -1

    for node_id, node_type, original_id in zip(node_ids, node_types, original_ids, strict=True):
        node_id_int = int(node_id)
        if original_id in original_to_node_id:
            raise ValueError(f"Duplicate original_id in nodes parquet: {original_id!r}")
        if node_id_int in node_id_to_original:
            raise ValueError(f"Duplicate node_id in nodes parquet: {node_id_int}")
        original_to_node_id[original_id] = node_id_int
        node_type_str = str(node_type)
        original_to_type[original_id] = node_type_str
        node_id_to_original[node_id_int] = str(original_id)
        node_id_to_type[node_id_int] = node_type_str
        if node_type_str == "disease":
            disease_ids.append(node_id_int)
        elif node_type_str == "gene/protein":
            protein_ids.append(node_id_int)
        elif node_type_str == "pathway":
            pathway_ids.append(node_id_int)
        if node_id_int > max_node_id:
            max_node_id = node_id_int

    if max_node_id < 0:
        raise ValueError(f"Nodes parquet has no rows: {path}")
    if not disease_ids:
        raise ValueError(f"Nodes parquet has no disease nodes: {path}")

    return NodeIndex(
        original_to_node_id=original_to_node_id,
        original_to_type=original_to_type,
        node_id_to_original=node_id_to_original,
        node_id_to_type=node_id_to_type,
        disease_node_ids=np.array(sorted(disease_ids), dtype=np.int64),
        protein_node_ids=np.array(sorted(protein_ids), dtype=np.int64),
        pathway_node_ids=np.array(sorted(pathway_ids), dtype=np.int64),
        num_nodes=max_node_id + 1,
    )


def _map_pairs_to_node_ids(
    *,
    drug_ids: list[str],
    disease_ids: list[str],
    node_index: NodeIndex,
    source_path: Path,
) -> np.ndarray:
    mapped: list[tuple[int, int]] = []
    for i, (drug_id, disease_id) in enumerate(zip(drug_ids, disease_ids, strict=True), start=1):
        if drug_id not in node_index.original_to_node_id:
            raise ValueError(f"{source_path}: row {i} unknown drug id in nodes.parquet: {drug_id!r}")
        if disease_id not in node_index.original_to_node_id:
            raise ValueError(
                f"{source_path}: row {i} unknown disease id in nodes.parquet: {disease_id!r}"
            )
        drug_type = node_index.original_to_type[drug_id]
        disease_type = node_index.original_to_type[disease_id]
        if drug_type != "drug":
            raise ValueError(
                f"{source_path}: row {i} drug_id {drug_id!r} has type {drug_type!r}, expected 'drug'"
            )
        if disease_type != "disease":
            raise ValueError(
                f"{source_path}: row {i} disease_id {disease_id!r} has type {disease_type!r}, expected 'disease'"
            )
        mapped.append(
            (
                node_index.original_to_node_id[drug_id],
                node_index.original_to_node_id[disease_id],
            )
        )

    if not mapped:
        return np.empty((0, 2), dtype=np.int64)
    return np.array(mapped, dtype=np.int64)


def load_pair_splits(
    *,
    node_index: NodeIndex,
    train_pairs_path: str | Path,
    val_pairs_path: str | Path,
    test_pairs_path: str | Path,
) -> PairSplit:
    train_path = Path(train_pairs_path)
    val_path = Path(val_pairs_path)
    test_path = Path(test_pairs_path)

    train_drugs, train_diseases = _read_pair_parquet(train_path)
    val_drugs, val_diseases = _read_pair_parquet(val_path)
    test_drugs, test_diseases = _read_pair_parquet(test_path)

    train = _map_pairs_to_node_ids(
        drug_ids=train_drugs,
        disease_ids=train_diseases,
        node_index=node_index,
        source_path=train_path,
    )
    val = _map_pairs_to_node_ids(
        drug_ids=val_drugs,
        disease_ids=val_diseases,
        node_index=node_index,
        source_path=val_path,
    )
    test = _map_pairs_to_node_ids(
        drug_ids=test_drugs,
        disease_ids=test_diseases,
        node_index=node_index,
        source_path=test_path,
    )
    return PairSplit(train=train, val=val, test=test)


def _read_quadruple_parquet(path: str | Path) -> tuple[list[str], list[str], list[str], list[str]]:
    quad_path = Path(path)
    if not quad_path.exists():
        raise FileNotFoundError(f"Quadruple parquet file not found: {quad_path}")
    table = pq.read_table(
        quad_path,
        columns=["drug_id", "disease_id", "protein_id", "pathway_id"],
    )
    return (
        table["drug_id"].to_pylist(),
        table["disease_id"].to_pylist(),
        table["protein_id"].to_pylist(),
        table["pathway_id"].to_pylist(),
    )


def _map_quadruples_to_node_ids(
    *,
    drug_ids: list[str],
    disease_ids: list[str],
    protein_ids: list[str],
    pathway_ids: list[str],
    node_index: NodeIndex,
    source_path: Path,
) -> np.ndarray:
    mapped: list[tuple[int, int, int, int]] = []
    for i, (drug_id, disease_id, protein_id, pathway_raw) in enumerate(
        zip(drug_ids, disease_ids, protein_ids, pathway_ids, strict=True), start=1
    ):
        pathway_norm = _normalize_pathway_id(str(pathway_raw))
        if pathway_norm is None:
            raise ValueError(
                f"{source_path}: row {i} invalid pathway_id format {pathway_raw!r}. "
                "Expected 'pathway::<id>' or 'http://bioregistry.io/reactome:<id>'."
            )

        if drug_id not in node_index.original_to_node_id:
            raise ValueError(f"{source_path}: row {i} unknown drug id in nodes.parquet: {drug_id!r}")
        if disease_id not in node_index.original_to_node_id:
            raise ValueError(f"{source_path}: row {i} unknown disease id in nodes.parquet: {disease_id!r}")
        if protein_id not in node_index.original_to_node_id:
            raise ValueError(f"{source_path}: row {i} unknown protein id in nodes.parquet: {protein_id!r}")
        if pathway_norm not in node_index.original_to_node_id:
            raise ValueError(
                f"{source_path}: row {i} unknown normalized pathway id in nodes.parquet: {pathway_norm!r}"
            )

        drug_type = node_index.original_to_type[drug_id]
        disease_type = node_index.original_to_type[disease_id]
        protein_type = node_index.original_to_type[protein_id]
        pathway_type = node_index.original_to_type[pathway_norm]
        if drug_type != "drug":
            raise ValueError(
                f"{source_path}: row {i} drug_id {drug_id!r} has type {drug_type!r}, expected 'drug'"
            )
        if disease_type != "disease":
            raise ValueError(
                f"{source_path}: row {i} disease_id {disease_id!r} has type {disease_type!r}, expected 'disease'"
            )
        if protein_type != "gene/protein":
            raise ValueError(
                f"{source_path}: row {i} protein_id {protein_id!r} has type {protein_type!r}, expected 'gene/protein'"
            )
        if pathway_type != "pathway":
            raise ValueError(
                f"{source_path}: row {i} pathway_id {pathway_norm!r} has type {pathway_type!r}, expected 'pathway'"
            )

        mapped.append(
            (
                node_index.original_to_node_id[drug_id],
                node_index.original_to_node_id[disease_id],
                node_index.original_to_node_id[protein_id],
                node_index.original_to_node_id[pathway_norm],
            )
        )

    if not mapped:
        return np.empty((0, 4), dtype=np.int64)
    return np.array(mapped, dtype=np.int64)


def load_quadruple_splits(
    *,
    node_index: NodeIndex,
    train_quadruples_path: str | Path,
    val_quadruples_path: str | Path,
    test_quadruples_path: str | Path,
) -> QuadrupleSplit:
    train_path = Path(train_quadruples_path)
    val_path = Path(val_quadruples_path)
    test_path = Path(test_quadruples_path)

    train_cols = _read_quadruple_parquet(train_path)
    val_cols = _read_quadruple_parquet(val_path)
    test_cols = _read_quadruple_parquet(test_path)

    train = _map_quadruples_to_node_ids(
        drug_ids=train_cols[0],
        disease_ids=train_cols[1],
        protein_ids=train_cols[2],
        pathway_ids=train_cols[3],
        node_index=node_index,
        source_path=train_path,
    )
    val = _map_quadruples_to_node_ids(
        drug_ids=val_cols[0],
        disease_ids=val_cols[1],
        protein_ids=val_cols[2],
        pathway_ids=val_cols[3],
        node_index=node_index,
        source_path=val_path,
    )
    test = _map_quadruples_to_node_ids(
        drug_ids=test_cols[0],
        disease_ids=test_cols[1],
        protein_ids=test_cols[2],
        pathway_ids=test_cols[3],
        node_index=node_index,
        source_path=test_path,
    )
    return QuadrupleSplit(train=train, val=val, test=test)


def _drug_protein_edges_from_quadruples(quadruples: np.ndarray) -> set[tuple[int, int]]:
    if quadruples.size == 0:
        return set()
    return {(int(row[0]), int(row[2])) for row in quadruples}


def _protein_pathway_edges_from_quadruples(quadruples: np.ndarray) -> set[tuple[int, int]]:
    if quadruples.size == 0:
        return set()
    return {(int(row[2]), int(row[3])) for row in quadruples}


def build_mechanism_strict_blocklists(quad_splits: QuadrupleSplit) -> MechanismStrictBlocklists:
    train_drug_protein = _drug_protein_edges_from_quadruples(quad_splits.train)
    train_protein_pathway = _protein_pathway_edges_from_quadruples(quad_splits.train)

    holdout_drug_protein = (
        _drug_protein_edges_from_quadruples(quad_splits.val)
        | _drug_protein_edges_from_quadruples(quad_splits.test)
    )
    holdout_protein_pathway = (
        _protein_pathway_edges_from_quadruples(quad_splits.val)
        | _protein_pathway_edges_from_quadruples(quad_splits.test)
    )

    blocked_drug_protein = holdout_drug_protein.difference(train_drug_protein)
    blocked_protein_pathway = holdout_protein_pathway.difference(train_protein_pathway)

    return MechanismStrictBlocklists(
        blocked_drug_protein_edges=blocked_drug_protein,
        blocked_protein_pathway_edges=blocked_protein_pathway,
        train_drug_protein_edge_count=len(train_drug_protein),
        train_protein_pathway_edge_count=len(train_protein_pathway),
        holdout_drug_protein_edge_count=len(holdout_drug_protein),
        holdout_protein_pathway_edge_count=len(holdout_protein_pathway),
    )


def load_mechanism_strict_blocklists(
    *,
    node_index: NodeIndex,
    train_quadruples_path: str | Path,
    val_quadruples_path: str | Path,
    test_quadruples_path: str | Path,
) -> MechanismStrictBlocklists:
    quad_splits = load_quadruple_splits(
        node_index=node_index,
        train_quadruples_path=train_quadruples_path,
        val_quadruples_path=val_quadruples_path,
        test_quadruples_path=test_quadruples_path,
    )
    return build_mechanism_strict_blocklists(quad_splits)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def distmult_scores(
    node_embeddings: np.ndarray,
    relation_embedding: np.ndarray,
    heads: np.ndarray,
    tails: np.ndarray,
) -> np.ndarray:
    head_e = node_embeddings[heads]
    tail_e = node_embeddings[tails]
    return np.sum(head_e * relation_embedding * tail_e, axis=1)


def build_positive_tail_index(pairs: np.ndarray) -> dict[int, set[int]]:
    out: dict[int, set[int]] = {}
    for head, tail in pairs:
        h = int(head)
        t = int(tail)
        if h not in out:
            out[h] = set()
        out[h].add(t)
    return out


def sample_negative_tails(
    *,
    heads: np.ndarray,
    disease_candidates: np.ndarray,
    positive_tails_by_head: dict[int, set[int]],
    rng: np.random.Generator,
) -> np.ndarray:
    if disease_candidates.size == 0:
        raise ValueError("disease_candidates is empty")

    negatives = np.empty_like(heads, dtype=np.int64)
    for i, head in enumerate(heads):
        positive_set = positive_tails_by_head.get(int(head), set())
        found = False
        for _ in range(32):
            candidate = int(disease_candidates[rng.integers(0, disease_candidates.size)])
            if candidate not in positive_set:
                negatives[i] = candidate
                found = True
                break
        if found:
            continue

        for candidate in disease_candidates:
            c = int(candidate)
            if c not in positive_set:
                negatives[i] = c
                found = True
                break
        if not found:
            raise ValueError(
                f"Could not sample negative tail for head={int(head)}; all disease candidates are positives."
            )
    return negatives


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    positives = int(np.sum(y_true == 1))
    if positives == 0:
        return 0.0

    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]

    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / positives

    recall_prev = np.concatenate(([0.0], recall[:-1]))
    ap = np.sum((recall - recall_prev) * precision)
    return float(ap)


def evaluate_link_prediction(
    *,
    node_embeddings: np.ndarray,
    relation_embedding: np.ndarray,
    eval_pairs: np.ndarray,
    disease_candidates: np.ndarray,
    filter_tails_by_head: dict[int, set[int]],
    auprc_negatives_per_positive: int,
    seed: int,
    fusion_batch_scorer: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    fusion_all_tails_scorer: Callable[[int, np.ndarray], np.ndarray] | None = None,
    rank_batch_size: int = 256,
) -> dict[str, float]:
    if eval_pairs.size == 0:
        raise ValueError("eval_pairs is empty")
    if auprc_negatives_per_positive <= 0:
        raise ValueError(
            f"auprc_negatives_per_positive must be positive, got {auprc_negatives_per_positive}"
        )
    if rank_batch_size <= 0:
        raise ValueError(f"rank_batch_size must be positive, got {rank_batch_size}")
    if not np.isfinite(node_embeddings).all() or not np.isfinite(relation_embedding).all():
        raise ValueError("Non-finite embeddings passed to evaluation.")

    disease_index = {int(node_id): idx for idx, node_id in enumerate(disease_candidates.tolist())}
    filtered_indices_by_head: dict[int, np.ndarray] = {}
    for head, tails in filter_tails_by_head.items():
        idxs = [disease_index[int(t)] for t in tails if int(t) in disease_index]
        if idxs:
            filtered_indices_by_head[int(head)] = np.array(idxs, dtype=np.int64)

    rng = np.random.default_rng(seed)

    mrr_total = 0.0
    hits1 = 0
    hits3 = 0
    hits10 = 0

    pos_heads = eval_pairs[:, 0].astype(np.int64, copy=False)
    pos_tails = eval_pairs[:, 1].astype(np.int64, copy=False)
    pos_scores = distmult_scores(node_embeddings, relation_embedding, pos_heads, pos_tails)
    if fusion_batch_scorer is not None:
        pos_scores = pos_scores + fusion_batch_scorer(pos_heads, pos_tails)
    if not np.isfinite(pos_scores).all():
        raise ValueError("Non-finite positive scores encountered during evaluation.")

    disease_embeddings = node_embeddings[disease_candidates]
    n_eval = eval_pairs.shape[0]
    for start in range(0, n_eval, rank_batch_size):
        end = min(start + rank_batch_size, n_eval)
        chunk_heads = pos_heads[start:end]
        chunk_tails = pos_tails[start:end]
        chunk_queries = node_embeddings[chunk_heads] * relation_embedding[None, :]
        chunk_scores = disease_embeddings @ chunk_queries.T  # [num_disease, chunk]

        if fusion_all_tails_scorer is not None:
            for i, head in enumerate(chunk_heads):
                chunk_scores[:, i] += fusion_all_tails_scorer(int(head), disease_candidates)

        for i, (head, true_tail) in enumerate(zip(chunk_heads, chunk_tails, strict=True)):
            h = int(head)
            t = int(true_tail)
            true_idx = disease_index.get(t)
            if true_idx is None:
                raise ValueError(f"True tail disease node {t} not in disease candidate set.")

            candidate_scores = chunk_scores[:, i].copy()
            true_score = float(candidate_scores[true_idx])
            idxs = filtered_indices_by_head.get(h)
            if idxs is not None:
                candidate_scores[idxs] = -np.inf
                candidate_scores[true_idx] = true_score

            rank = int(np.sum(candidate_scores > true_score) + 1)

            mrr_total += 1.0 / rank
            if rank <= 1:
                hits1 += 1
            if rank <= 3:
                hits3 += 1
            if rank <= 10:
                hits10 += 1

    repeated_heads = np.repeat(pos_heads, auprc_negatives_per_positive)
    neg_tails = sample_negative_tails(
        heads=repeated_heads,
        disease_candidates=disease_candidates,
        positive_tails_by_head=filter_tails_by_head,
        rng=rng,
    )
    neg_scores = distmult_scores(node_embeddings, relation_embedding, repeated_heads, neg_tails)
    if fusion_batch_scorer is not None:
        neg_scores = neg_scores + fusion_batch_scorer(repeated_heads, neg_tails)
    if not np.isfinite(neg_scores).all():
        raise ValueError("Non-finite negative scores encountered during evaluation.")

    y_true = np.concatenate(
        [
            np.ones(pos_scores.shape[0], dtype=np.int64),
            np.zeros(neg_scores.shape[0], dtype=np.int64),
        ]
    )
    y_score = np.concatenate([pos_scores, neg_scores])
    auprc = average_precision(y_true, y_score)

    n = float(eval_pairs.shape[0])
    return {
        "auprc": float(auprc),
        "mrr": float(mrr_total / n),
        "hits@1": float(hits1 / n),
        "hits@3": float(hits3 / n),
        "hits@10": float(hits10 / n),
    }


def save_checkpoint(
    *,
    checkpoint_path: str | Path,
    node_embeddings: np.ndarray,
    relation_embedding: np.ndarray,
    seed: int,
    extra_arrays: dict[str, np.ndarray] | None = None,
) -> None:
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "node_embeddings": node_embeddings.astype(np.float32),
        "relation_embedding": relation_embedding.astype(np.float32),
        "seed": np.array([seed], dtype=np.int64),
    }
    if extra_arrays:
        for key, value in extra_arrays.items():
            if key in payload:
                raise ValueError(f"Checkpoint extra array key conflicts with reserved field: {key!r}")
            payload[key] = value
    np.savez(path, **payload)


def load_checkpoint(
    checkpoint_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, int, dict[str, np.ndarray]]:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = np.load(path, allow_pickle=False)
    if "node_embeddings" not in ckpt or "relation_embedding" not in ckpt or "seed" not in ckpt:
        raise ValueError(
            f"{path}: checkpoint missing required arrays. expected node_embeddings, relation_embedding, seed"
        )
    node_embeddings = ckpt["node_embeddings"].astype(np.float32, copy=False)
    relation_embedding = ckpt["relation_embedding"].astype(np.float32, copy=False)
    seed = int(ckpt["seed"][0])
    extras: dict[str, np.ndarray] = {}
    for key in ckpt.files:
        if key in {"node_embeddings", "relation_embedding", "seed"}:
            continue
        extras[key] = ckpt[key]
    return node_embeddings, relation_embedding, seed, extras
