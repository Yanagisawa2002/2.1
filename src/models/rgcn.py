from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


def require_torch() -> Any:
    try:
        import torch  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        raise ImportError(
            "PyTorch is required for R-GCN training/evaluation. "
            "Install a CUDA build on your cloud machine, for example:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cu121"
        ) from exc
    return torch


@dataclass(frozen=True)
class RelationGraph:
    num_nodes: int
    relations: tuple[str, ...]
    edge_index_by_rel: dict[str, np.ndarray]
    edge_weight_by_rel: dict[str, np.ndarray]

    def to_torch(self, device: Any) -> "TorchRelationGraph":
        torch = require_torch()
        edge_index_by_rel: dict[str, Any] = {}
        edge_weight_by_rel: dict[str, Any] = {}
        for rel in self.relations:
            edge_index_by_rel[rel] = torch.as_tensor(
                self.edge_index_by_rel[rel],
                dtype=torch.long,
                device=device,
            )
            edge_weight_by_rel[rel] = torch.as_tensor(
                self.edge_weight_by_rel[rel],
                dtype=torch.float32,
                device=device,
            )
        return TorchRelationGraph(
            num_nodes=self.num_nodes,
            relations=self.relations,
            edge_index_by_rel=edge_index_by_rel,
            edge_weight_by_rel=edge_weight_by_rel,
        )


@dataclass(frozen=True)
class TorchRelationGraph:
    num_nodes: int
    relations: tuple[str, ...]
    edge_index_by_rel: dict[str, Any]
    edge_weight_by_rel: dict[str, Any]


def _relation_row_norm(src: np.ndarray, num_nodes: int) -> np.ndarray:
    # Row-normalization per relation keeps message scale stable across dense/sparse relations.
    deg = np.bincount(src, minlength=num_nodes).astype(np.float32)
    deg_safe = np.maximum(deg, 1.0)
    return 1.0 / deg_safe[src]


def build_relation_graph(
    *,
    edges_path: str | Path,
    num_nodes: int,
    include_relations: set[str] | None = None,
    keep_train_indication_only: bool = True,
    train_indication_pairs: set[tuple[int, int]] | None = None,
    batch_size: int = 500_000,
) -> RelationGraph:
    path = Path(edges_path)
    if not path.exists():
        raise FileNotFoundError(f"Edges parquet not found: {path}")
    if num_nodes <= 0:
        raise ValueError(f"num_nodes must be positive, got {num_nodes}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if keep_train_indication_only and train_indication_pairs is None:
        raise ValueError(
            "keep_train_indication_only=true requires train_indication_pairs to avoid label leakage."
        )

    src_chunks_by_rel: dict[str, list[np.ndarray]] = defaultdict(list)
    dst_chunks_by_rel: dict[str, list[np.ndarray]] = defaultdict(list)

    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=["src_id", "dst_id", "rel"], batch_size=batch_size):
        src_ids = np.asarray(batch.column(0).to_numpy(zero_copy_only=False), dtype=np.int64)
        dst_ids = np.asarray(batch.column(1).to_numpy(zero_copy_only=False), dtype=np.int64)
        rels = np.asarray(batch.column(2).to_pylist(), dtype=object)

        if src_ids.shape[0] != dst_ids.shape[0] or src_ids.shape[0] != rels.shape[0]:
            raise ValueError(f"{path}: edge batch column length mismatch.")

        for rel in np.unique(rels):
            rel_str = str(rel)
            if include_relations is not None and rel_str not in include_relations:
                continue

            rel_mask = rels == rel
            src_rel = src_ids[rel_mask]
            dst_rel = dst_ids[rel_mask]
            if src_rel.size == 0:
                continue

            if rel_str == "indication" and keep_train_indication_only:
                keep = np.fromiter(
                    (
                        (int(s), int(d)) in train_indication_pairs  # type: ignore[arg-type]
                        for s, d in zip(src_rel, dst_rel, strict=True)
                    ),
                    dtype=bool,
                    count=src_rel.shape[0],
                )
                src_rel = src_rel[keep]
                dst_rel = dst_rel[keep]
                if src_rel.size == 0:
                    continue

            src_chunks_by_rel[rel_str].append(src_rel.copy())
            dst_chunks_by_rel[rel_str].append(dst_rel.copy())

    if not src_chunks_by_rel:
        raise ValueError(f"{path}: no relations were loaded for R-GCN graph.")

    relations = tuple(sorted(src_chunks_by_rel.keys()))
    edge_index_by_rel: dict[str, np.ndarray] = {}
    edge_weight_by_rel: dict[str, np.ndarray] = {}
    for rel in relations:
        src = np.concatenate(src_chunks_by_rel[rel]).astype(np.int64, copy=False)
        dst = np.concatenate(dst_chunks_by_rel[rel]).astype(np.int64, copy=False)
        if src.size == 0:
            edge_index = np.empty((2, 0), dtype=np.int64)
            edge_weight = np.empty((0,), dtype=np.float32)
        else:
            edge_index = np.stack([src, dst], axis=0)
            edge_weight = _relation_row_norm(src, num_nodes)

        edge_index_by_rel[rel] = edge_index
        edge_weight_by_rel[rel] = edge_weight

    return RelationGraph(
        num_nodes=num_nodes,
        relations=relations,
        edge_index_by_rel=edge_index_by_rel,
        edge_weight_by_rel=edge_weight_by_rel,
    )


class RGCNEncoder:  # pragma: no cover - exercised via train/eval scripts
    def __init__(
        self,
        *,
        num_nodes: int,
        hidden_dim: int,
        relations: tuple[str, ...],
        num_layers: int,
        dropout: float,
        seed: int,
    ) -> None:
        torch = require_torch()
        nn = torch.nn

        if num_nodes <= 0:
            raise ValueError(f"num_nodes must be positive, got {num_nodes}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if not relations:
            raise ValueError("relations is empty")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        super().__init__()
        self._nn = nn
        self.num_layers = int(num_layers)
        self.relations = tuple(relations)
        self.node_embedding = nn.Embedding(num_nodes, hidden_dim)
        self.self_weights = nn.ParameterList(
            [nn.Parameter(torch.empty(hidden_dim, hidden_dim)) for _ in range(self.num_layers)]
        )
        self.rel_weights = nn.ParameterList(
            [
                nn.Parameter(torch.empty(len(self.relations), hidden_dim, hidden_dim))
                for _ in range(self.num_layers)
            ]
        )
        self.dropout = nn.Dropout(p=float(dropout))
        self.reset_parameters(seed=seed)

    def reset_parameters(self, *, seed: int) -> None:
        torch = require_torch()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            torch.nn.init.normal_(self.node_embedding.weight, mean=0.0, std=0.05)
            for param in self.self_weights:
                torch.nn.init.xavier_uniform_(param)
            for param in self.rel_weights:
                torch.nn.init.xavier_uniform_(param)

    def forward(self, graph: TorchRelationGraph) -> Any:
        torch = require_torch()
        x = self.node_embedding.weight
        for layer_idx in range(self.num_layers):
            out = x @ self.self_weights[layer_idx]
            rel_w = self.rel_weights[layer_idx]
            for rel_idx, rel in enumerate(self.relations):
                edge_index = graph.edge_index_by_rel[rel]
                if edge_index.numel() == 0:
                    continue
                src = edge_index[0]
                dst = edge_index[1]
                msg = x[src] @ rel_w[rel_idx]
                edge_w = graph.edge_weight_by_rel[rel]
                msg = msg * edge_w.unsqueeze(1)
                out.index_add_(0, dst, msg)

            if layer_idx + 1 < self.num_layers:
                out = torch.relu(out)
                out = self.dropout(out)
            x = out
        return x


class DistMultDecoder:  # pragma: no cover - exercised via train/eval scripts
    def __init__(self, *, hidden_dim: int, seed: int) -> None:
        torch = require_torch()
        nn = torch.nn
        self.relation = nn.Parameter(torch.empty(hidden_dim))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed) + 17)
            torch.nn.init.normal_(self.relation, mean=0.0, std=0.05)

    def score(self, node_embeddings: Any, heads: Any, tails: Any) -> Any:
        return (node_embeddings[heads] * self.relation * node_embeddings[tails]).sum(dim=1)


class RGCNIndicationModel:  # pragma: no cover - exercised via train/eval scripts
    def __init__(
        self,
        *,
        num_nodes: int,
        hidden_dim: int,
        relations: tuple[str, ...],
        num_layers: int,
        dropout: float,
        seed: int,
    ) -> None:
        torch = require_torch()
        nn = torch.nn
        super().__init__()
        self._nn = nn
        self.encoder = RGCNEncoder(
            num_nodes=num_nodes,
            hidden_dim=hidden_dim,
            relations=relations,
            num_layers=num_layers,
            dropout=dropout,
            seed=seed,
        )
        self.decoder = DistMultDecoder(hidden_dim=hidden_dim, seed=seed)

    def parameters(self) -> Any:
        for p in self.encoder.node_embedding.parameters():
            yield p
        for p in self.encoder.self_weights.parameters():
            yield p
        for p in self.encoder.rel_weights.parameters():
            yield p
        yield self.decoder.relation

    def train(self) -> None:
        self.encoder.node_embedding.train()
        self.encoder.dropout.train()

    def eval(self) -> None:
        self.encoder.node_embedding.eval()
        self.encoder.dropout.eval()

    def to(self, device: Any) -> "RGCNIndicationModel":
        self.encoder.node_embedding.to(device)
        self.encoder.dropout.to(device)
        for p in self.encoder.self_weights:
            p.data = p.data.to(device)
        for p in self.encoder.rel_weights:
            p.data = p.data.to(device)
        self.decoder.relation.data = self.decoder.relation.data.to(device)
        return self

    def state_dict(self) -> dict[str, Any]:
        return {
            "node_embedding": self.encoder.node_embedding.state_dict(),
            "dropout": self.encoder.dropout.state_dict(),
            "self_weights": [p.detach().cpu() for p in self.encoder.self_weights],
            "rel_weights": [p.detach().cpu() for p in self.encoder.rel_weights],
            "decoder_relation": self.decoder.relation.detach().cpu(),
            "relations": self.encoder.relations,
            "num_layers": self.encoder.num_layers,
        }

    def load_state_dict(self, payload: dict[str, Any], *, strict: bool = True) -> None:
        torch = require_torch()
        if strict:
            expected_keys = {
                "node_embedding",
                "dropout",
                "self_weights",
                "rel_weights",
                "decoder_relation",
                "relations",
                "num_layers",
            }
            missing = expected_keys.difference(payload.keys())
            if missing:
                raise KeyError(f"RGCN checkpoint missing keys: {sorted(missing)!r}")

        self.encoder.node_embedding.load_state_dict(payload["node_embedding"])
        self.encoder.dropout.load_state_dict(payload["dropout"])
        for dst_param, src_param in zip(
            self.encoder.self_weights,
            payload["self_weights"],
            strict=True,
        ):
            dst_param.data.copy_(torch.as_tensor(src_param, dtype=dst_param.dtype))
        for dst_param, src_param in zip(
            self.encoder.rel_weights,
            payload["rel_weights"],
            strict=True,
        ):
            dst_param.data.copy_(torch.as_tensor(src_param, dtype=dst_param.dtype))
        self.decoder.relation.data.copy_(
            torch.as_tensor(payload["decoder_relation"], dtype=self.decoder.relation.dtype)
        )

    def encode(self, graph: TorchRelationGraph) -> Any:
        return self.encoder.forward(graph)

    def score_pairs(self, node_embeddings: Any, heads: Any, tails: Any) -> Any:
        return self.decoder.score(node_embeddings, heads, tails)
