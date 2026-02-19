from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
from pathlib import Path
import pickle
from typing import Mapping

import numpy as np
import pyarrow.parquet as pq


@dataclass(frozen=True)
class MechanismAdjacency:
    drug_to_proteins: dict[int, set[int]]
    disease_to_proteins: dict[int, set[int]]
    protein_to_pathways: dict[int, set[int]]


def build_mechanism_adjacency(
    *,
    edges_path: str | Path,
    node_id_to_type: Mapping[int, str],
    batch_size: int = 500_000,
    blocked_drug_protein_edges: set[tuple[int, int]] | None = None,
    blocked_protein_pathway_edges: set[tuple[int, int]] | None = None,
    cache_dir: str | Path | None = "artifacts/cache/mechanism_adjacency",
) -> MechanismAdjacency:
    path = Path(edges_path)
    if not path.exists():
        raise FileNotFoundError(f"Edges parquet not found: {path}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    blocked_drug_protein = blocked_drug_protein_edges or set()
    blocked_protein_pathway = blocked_protein_pathway_edges or set()

    cache_path: Path | None = None
    if cache_dir is not None:
        cache_path = _mechanism_cache_path(
            edges_path=path,
            blocked_drug_protein_edges=blocked_drug_protein,
            blocked_protein_pathway_edges=blocked_protein_pathway,
            cache_dir=cache_dir,
        )
        cached = _try_load_cached_adjacency(cache_path)
        if cached is not None:
            return cached

    max_node_id = max(node_id_to_type.keys(), default=-1)
    type_codes = np.full(max_node_id + 1, -1, dtype=np.int8)
    # 1=drug, 2=disease, 3=gene/protein, 4=pathway
    type_to_code = {
        "drug": 1,
        "disease": 2,
        "gene/protein": 3,
        "pathway": 4,
    }
    for node_id, node_type in node_id_to_type.items():
        if 0 <= int(node_id) <= max_node_id:
            type_codes[int(node_id)] = type_to_code.get(str(node_type), -1)

    drug_to_proteins: dict[int, set[int]] = defaultdict(set)
    disease_to_proteins: dict[int, set[int]] = defaultdict(set)
    protein_to_pathways: dict[int, set[int]] = defaultdict(set)

    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=["src_id", "dst_id", "rel"], batch_size=batch_size):
        src_ids = batch.column(0).to_numpy(zero_copy_only=False)
        dst_ids = batch.column(1).to_numpy(zero_copy_only=False)
        rels = batch.column(2).to_pylist()

        for src_raw, dst_raw, rel_raw in zip(src_ids, dst_ids, rels, strict=True):
            src = int(src_raw)
            dst = int(dst_raw)
            rel = str(rel_raw)
            if src < 0 or dst < 0 or src > max_node_id or dst > max_node_id:
                continue
            src_type_code = int(type_codes[src])
            dst_type_code = int(type_codes[dst])
            if src_type_code < 0 or dst_type_code < 0:
                continue

            if rel == "drug_protein":
                if src_type_code == 1 and dst_type_code == 3:
                    if (src, dst) in blocked_drug_protein:
                        continue
                    drug_to_proteins[src].add(dst)
                elif src_type_code == 3 and dst_type_code == 1:
                    if (dst, src) in blocked_drug_protein:
                        continue
                    drug_to_proteins[dst].add(src)
            elif rel == "disease_protein":
                if src_type_code == 2 and dst_type_code == 3:
                    disease_to_proteins[src].add(dst)
                elif src_type_code == 3 and dst_type_code == 2:
                    disease_to_proteins[dst].add(src)
            elif rel in {"pathway_protein", "protein_pathway"}:
                if src_type_code == 4 and dst_type_code == 3:
                    if (dst, src) in blocked_protein_pathway:
                        continue
                    protein_to_pathways[dst].add(src)
                elif src_type_code == 3 and dst_type_code == 4:
                    if (src, dst) in blocked_protein_pathway:
                        continue
                    protein_to_pathways[src].add(dst)

    out = MechanismAdjacency(
        drug_to_proteins={k: set(v) for k, v in drug_to_proteins.items()},
        disease_to_proteins={k: set(v) for k, v in disease_to_proteins.items()},
        protein_to_pathways={k: set(v) for k, v in protein_to_pathways.items()},
    )
    if cache_path is not None:
        _try_write_cached_adjacency(cache_path, out)
    return out


def _edges_digest(edges: set[tuple[int, int]]) -> str:
    if not edges:
        return "none"
    h = hashlib.sha256()
    for src, dst in sorted(edges):
        h.update(f"{src}:{dst};".encode("ascii"))
    return h.hexdigest()


def _mechanism_cache_path(
    *,
    edges_path: Path,
    blocked_drug_protein_edges: set[tuple[int, int]],
    blocked_protein_pathway_edges: set[tuple[int, int]],
    cache_dir: str | Path,
) -> Path:
    stat = edges_path.stat()
    h = hashlib.sha256()
    h.update(str(edges_path.resolve()).encode("utf-8"))
    h.update(str(stat.st_size).encode("ascii"))
    h.update(str(stat.st_mtime_ns).encode("ascii"))
    h.update(_edges_digest(blocked_drug_protein_edges).encode("ascii"))
    h.update(_edges_digest(blocked_protein_pathway_edges).encode("ascii"))
    key = h.hexdigest()
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    return root / f"mechanism_adjacency_{key}.pkl"


def _try_load_cached_adjacency(path: Path) -> MechanismAdjacency | None:
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            payload = pickle.load(f)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if {"drug_to_proteins", "disease_to_proteins", "protein_to_pathways"} - set(payload.keys()):
        return None
    return MechanismAdjacency(
        drug_to_proteins={int(k): set(v) for k, v in payload["drug_to_proteins"].items()},
        disease_to_proteins={int(k): set(v) for k, v in payload["disease_to_proteins"].items()},
        protein_to_pathways={int(k): set(v) for k, v in payload["protein_to_pathways"].items()},
    )


def _try_write_cached_adjacency(path: Path, adjacency: MechanismAdjacency) -> None:
    payload = {
        "drug_to_proteins": adjacency.drug_to_proteins,
        "disease_to_proteins": adjacency.disease_to_proteins,
        "protein_to_pathways": adjacency.protein_to_pathways,
    }
    tmp = path.with_suffix(".tmp")
    try:
        with tmp.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


class MechanismHead:
    def __init__(
        self,
        *,
        embedding_dim: int,
        adjacency: MechanismAdjacency,
        seed: int = 0,
        fusion_scale: float = 0.1,
    ) -> None:
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        self.adjacency = adjacency
        self.fusion_scale = float(fusion_scale)
        rng = np.random.default_rng(seed)
        self.protein_relation = rng.normal(0.0, 0.05, size=(embedding_dim,)).astype(np.float32)
        self.pathway_relation = rng.normal(0.0, 0.05, size=(embedding_dim,)).astype(np.float32)
        self._protein_candidate_cache: dict[tuple[int, int], np.ndarray] = {}
        self._pathway_candidate_cache: dict[int, np.ndarray] = {}

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        if logits.size == 0:
            return logits
        shifted = logits - np.max(logits)
        exp = np.exp(shifted)
        return exp / np.maximum(np.sum(exp), 1e-12)

    @staticmethod
    def _sigmoid_scalar(x: float) -> float:
        x_clip = float(np.clip(x, -30.0, 30.0))
        return float(1.0 / (1.0 + np.exp(-x_clip)))

    def set_parameters(
        self,
        *,
        protein_relation: np.ndarray | None = None,
        pathway_relation: np.ndarray | None = None,
        fusion_scale: float | None = None,
    ) -> None:
        if protein_relation is not None:
            self.protein_relation = protein_relation.astype(np.float32, copy=True)
        if pathway_relation is not None:
            self.pathway_relation = pathway_relation.astype(np.float32, copy=True)
        if fusion_scale is not None:
            self.fusion_scale = float(fusion_scale)

    def export_parameters(self) -> dict[str, np.ndarray]:
        return {
            "protein_relation_embedding": self.protein_relation.astype(np.float32),
            "pathway_relation_embedding": self.pathway_relation.astype(np.float32),
            "fusion_scale": np.array([self.fusion_scale], dtype=np.float32),
        }

    def protein_candidates(self, drug_id: int, disease_id: int) -> np.ndarray:
        key = (int(drug_id), int(disease_id))
        cached = self._protein_candidate_cache.get(key)
        if cached is not None:
            return cached

        drug_set = self.adjacency.drug_to_proteins.get(key[0], set())
        disease_set = self.adjacency.disease_to_proteins.get(key[1], set())

        out_set: set[int]
        if drug_set and disease_set:
            inter = drug_set.intersection(disease_set)
            out_set = inter if inter else drug_set.union(disease_set)
        else:
            out_set = drug_set.union(disease_set)

        if out_set:
            arr = np.array(sorted(out_set), dtype=np.int64)
        else:
            arr = np.empty((0,), dtype=np.int64)
        self._protein_candidate_cache[key] = arr
        return arr

    def _protein_candidates_for_target(self, drug_id: int, disease_id: int, target_protein: int) -> np.ndarray:
        base = self.protein_candidates(drug_id, disease_id)
        if base.size > 0 and int(target_protein) in set(base.tolist()):
            return base

        drug_set = self.adjacency.drug_to_proteins.get(int(drug_id), set())
        disease_set = self.adjacency.disease_to_proteins.get(int(disease_id), set())
        union = drug_set.union(disease_set)
        if int(target_protein) in union:
            return np.array(sorted(union), dtype=np.int64)
        return base

    def pathway_candidates(self, protein_id: int) -> np.ndarray:
        pid = int(protein_id)
        cached = self._pathway_candidate_cache.get(pid)
        if cached is not None:
            return cached
        out_set = self.adjacency.protein_to_pathways.get(pid, set())
        if out_set:
            arr = np.array(sorted(out_set), dtype=np.int64)
        else:
            arr = np.empty((0,), dtype=np.int64)
        self._pathway_candidate_cache[pid] = arr
        return arr

    def _protein_logits(
        self,
        *,
        node_embeddings: np.ndarray,
        drug_id: int,
        disease_id: int,
        candidate_proteins: np.ndarray,
    ) -> np.ndarray:
        query = (
            node_embeddings[int(drug_id)]
            * node_embeddings[int(disease_id)]
            * self.protein_relation
        )
        return node_embeddings[candidate_proteins] @ query

    def _pathway_logits(
        self,
        *,
        node_embeddings: np.ndarray,
        drug_id: int,
        disease_id: int,
        protein_id: int,
        candidate_pathways: np.ndarray,
    ) -> np.ndarray:
        query = (
            node_embeddings[int(drug_id)]
            * node_embeddings[int(disease_id)]
            * node_embeddings[int(protein_id)]
            * self.pathway_relation
        )
        return node_embeddings[candidate_pathways] @ query

    def compute_losses_and_grads(
        self,
        *,
        node_embeddings: np.ndarray,
        quadruples: np.ndarray,
        protein_only: bool = False,
    ) -> dict[str, np.ndarray | float | int]:
        grad_node_protein = np.zeros_like(node_embeddings, dtype=np.float32)
        grad_node_pathway = np.zeros_like(node_embeddings, dtype=np.float32)
        grad_rel_protein = np.zeros_like(self.protein_relation, dtype=np.float32)
        grad_rel_pathway = np.zeros_like(self.pathway_relation, dtype=np.float32)

        loss_protein = 0.0
        loss_pathway = 0.0
        protein_count = 0
        pathway_count = 0

        for quad in quadruples:
            drug_id = int(quad[0])
            disease_id = int(quad[1])
            protein_id = int(quad[2])
            pathway_id = int(quad[3])

            # Positive-only supervision: missing quadruples are open-world and never treated as negatives.
            query_p = (
                node_embeddings[drug_id]
                * node_embeddings[disease_id]
                * self.protein_relation
            )
            score_p = float(np.dot(node_embeddings[protein_id], query_p))
            loss_protein += float(np.logaddexp(0.0, -score_p))
            protein_count += 1

            grad_score_p = self._sigmoid_scalar(score_p) - 1.0
            grad_node_protein[protein_id] += (grad_score_p * query_p).astype(np.float32)
            grad_query_p = (grad_score_p * node_embeddings[protein_id]).astype(np.float32)
            grad_node_protein[drug_id] += grad_query_p * (
                node_embeddings[disease_id] * self.protein_relation
            )
            grad_node_protein[disease_id] += grad_query_p * (
                node_embeddings[drug_id] * self.protein_relation
            )
            grad_rel_protein += grad_query_p * (
                node_embeddings[drug_id] * node_embeddings[disease_id]
            )

            if protein_only:
                continue

            query_pw = (
                node_embeddings[drug_id]
                * node_embeddings[disease_id]
                * node_embeddings[protein_id]
                * self.pathway_relation
            )
            score_pw = float(np.dot(node_embeddings[pathway_id], query_pw))
            loss_pathway += float(np.logaddexp(0.0, -score_pw))
            pathway_count += 1

            grad_score_pw = self._sigmoid_scalar(score_pw) - 1.0
            grad_node_pathway[pathway_id] += (grad_score_pw * query_pw).astype(np.float32)
            grad_q_pw = (grad_score_pw * node_embeddings[pathway_id]).astype(np.float32)
            grad_node_pathway[drug_id] += grad_q_pw * (
                node_embeddings[disease_id] * node_embeddings[protein_id] * self.pathway_relation
            )
            grad_node_pathway[disease_id] += grad_q_pw * (
                node_embeddings[drug_id] * node_embeddings[protein_id] * self.pathway_relation
            )
            grad_node_pathway[protein_id] += grad_q_pw * (
                node_embeddings[drug_id] * node_embeddings[disease_id] * self.pathway_relation
            )
            grad_rel_pathway += grad_q_pw * (
                node_embeddings[drug_id]
                * node_embeddings[disease_id]
                * node_embeddings[protein_id]
            )

        if protein_count > 0:
            grad_node_protein /= float(protein_count)
            grad_rel_protein /= float(protein_count)
            loss_protein /= float(protein_count)
        if pathway_count > 0:
            grad_node_pathway /= float(pathway_count)
            grad_rel_pathway /= float(pathway_count)
            loss_pathway /= float(pathway_count)

        return {
            "loss_protein": float(loss_protein),
            "loss_pathway": float(loss_pathway),
            "protein_count": int(protein_count),
            "pathway_count": int(pathway_count),
            "grad_node_protein": grad_node_protein,
            "grad_node_pathway": grad_node_pathway,
            "grad_rel_protein": grad_rel_protein,
            "grad_rel_pathway": grad_rel_pathway,
        }

    def fusion_scores_batch(
        self,
        *,
        node_embeddings: np.ndarray,
        heads: np.ndarray,
        tails: np.ndarray,
        protein_only: bool = False,
    ) -> np.ndarray:
        out = np.zeros(heads.shape[0], dtype=np.float32)
        for i, (head, tail) in enumerate(zip(heads, tails, strict=True)):
            drug_id = int(head)
            disease_id = int(tail)
            proteins = self.protein_candidates(drug_id, disease_id)
            if proteins.size == 0:
                continue
            protein_logits = self._protein_logits(
                node_embeddings=node_embeddings,
                drug_id=drug_id,
                disease_id=disease_id,
                candidate_proteins=proteins,
            )
            best_idx = int(np.argmax(protein_logits))
            best_protein_score = float(protein_logits[best_idx])
            if protein_only:
                out[i] = self.fusion_scale * best_protein_score
                continue

            best_protein = int(proteins[best_idx])
            pathways = self.pathway_candidates(best_protein)
            if pathways.size == 0:
                out[i] = self.fusion_scale * best_protein_score
                continue
            pathway_logits = self._pathway_logits(
                node_embeddings=node_embeddings,
                drug_id=drug_id,
                disease_id=disease_id,
                protein_id=best_protein,
                candidate_pathways=pathways,
            )
            best_pathway_score = float(np.max(pathway_logits))
            out[i] = self.fusion_scale * (best_protein_score + 0.5 * best_pathway_score)
        return out

    def fusion_scores_for_all_tails(
        self,
        *,
        node_embeddings: np.ndarray,
        head: int,
        candidate_tails: np.ndarray,
        protein_only: bool = False,
    ) -> np.ndarray:
        heads = np.full(candidate_tails.shape[0], int(head), dtype=np.int64)
        return self.fusion_scores_batch(
            node_embeddings=node_embeddings,
            heads=heads,
            tails=candidate_tails.astype(np.int64, copy=False),
            protein_only=protein_only,
        )

    def explain_pair(
        self,
        *,
        node_embeddings: np.ndarray,
        drug_id: int,
        disease_id: int,
        top_k: int = 10,
        protein_only: bool = False,
    ) -> dict[str, object]:
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        proteins = self.protein_candidates(drug_id, disease_id)
        if proteins.size == 0:
            return {
                "drug_node_id": int(drug_id),
                "disease_node_id": int(disease_id),
                "protein_candidates": [],
                "pathway_candidates": [],
            }

        protein_logits = self._protein_logits(
            node_embeddings=node_embeddings,
            drug_id=drug_id,
            disease_id=disease_id,
            candidate_proteins=proteins,
        )
        protein_probs = self._softmax(protein_logits)
        protein_order = np.argsort(-protein_probs)[:top_k]

        protein_rows: list[dict[str, float | int]] = []
        pathway_weights: dict[int, float] = {}
        for idx in protein_order:
            protein_id = int(proteins[idx])
            protein_weight = float(protein_probs[idx])
            protein_score = float(protein_logits[idx])
            protein_rows.append(
                {
                    "node_id": protein_id,
                    "weight": protein_weight,
                    "score": protein_score,
                }
            )

            if protein_only:
                continue
            pathways = self.pathway_candidates(protein_id)
            if pathways.size == 0:
                continue
            pathway_logits = self._pathway_logits(
                node_embeddings=node_embeddings,
                drug_id=drug_id,
                disease_id=disease_id,
                protein_id=protein_id,
                candidate_pathways=pathways,
            )
            pathway_probs = self._softmax(pathway_logits)
            for pw_id, pw_prob, pw_score in zip(pathways, pathway_probs, pathway_logits, strict=True):
                contribution = protein_weight * float(pw_prob)
                pathway_weights[int(pw_id)] = pathway_weights.get(int(pw_id), 0.0) + contribution

        pathway_rows: list[dict[str, float | int]] = []
        for node_id, weight in sorted(pathway_weights.items(), key=lambda kv: kv[1], reverse=True)[:top_k]:
            pathway_rows.append({"node_id": int(node_id), "weight": float(weight)})

        return {
            "drug_node_id": int(drug_id),
            "disease_node_id": int(disease_id),
            "protein_candidates": protein_rows,
            "pathway_candidates": pathway_rows,
        }
