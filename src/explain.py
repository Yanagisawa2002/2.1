from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.baseline_common import (
    load_checkpoint,
    load_mechanism_strict_blocklists,
    load_node_index,
    read_yaml,
    write_json,
)
from src.models.mechanism_head import MechanismHead, build_mechanism_adjacency


def _normalize_drug_id(value: str) -> str:
    v = value.strip()
    if v.startswith("drug::"):
        return v
    if v.startswith("DB"):
        return f"drug::{v}"
    return v


def _normalize_disease_id(value: str) -> str:
    v = value.strip()
    if v.startswith("disease::"):
        return v
    if v:
        return f"disease::{v}"
    return v


def _resolve_run_and_config(
    *,
    config_path: str | Path,
    run_dir: str | Path | None,
    checkpoint_path: str | Path | None,
) -> tuple[Path | None, dict[str, Any], Path]:
    run_dir_path: Path | None = None
    if run_dir is not None:
        run_dir_path = Path(run_dir)
        if not run_dir_path.exists():
            raise FileNotFoundError(f"run_dir not found: {run_dir_path}")
        if not run_dir_path.is_dir():
            raise NotADirectoryError(f"run_dir is not a directory: {run_dir_path}")
        cfg_in_run = run_dir_path / "config.yaml"
        config = read_yaml(cfg_in_run) if cfg_in_run.exists() else read_yaml(config_path)
    else:
        config = read_yaml(config_path)

    if checkpoint_path is not None:
        ckpt = Path(checkpoint_path)
    elif run_dir_path is not None:
        ckpt = run_dir_path / "checkpoint.npz"
    else:
        raise ValueError("Either --run-dir or --checkpoint must be provided.")
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    return run_dir_path, config, ckpt


def explain_pair(
    *,
    config: dict[str, Any],
    checkpoint_path: str | Path,
    drug_id: str,
    disease_id: str,
    top_k: int,
) -> dict[str, Any]:
    data_cfg = config["data"]
    train_cfg = config.get("train", {})
    mechanism_cache_dir = data_cfg.get("mechanism_cache_dir", "artifacts/cache/mechanism_adjacency")
    protein_only = bool(train_cfg.get("protein_only", False))
    strict_mech_graph = bool(train_cfg.get("strict_mech_graph", False))
    fusion_scale = float(train_cfg.get("fusion_scale", 0.1))

    node_index = load_node_index(data_cfg["nodes_path"])
    node_embeddings, _, ckpt_seed, extras = load_checkpoint(checkpoint_path)

    required = {"protein_relation_embedding"}
    if not protein_only:
        required.add("pathway_relation_embedding")
    missing = sorted(required.difference(extras.keys()))
    if missing:
        raise ValueError(
            "Checkpoint does not contain trained mechanism-head parameters required for explanation. "
            f"Missing arrays: {missing!r}. Train with use_quadruple=true."
        )

    blocked_drug_protein_edges: set[tuple[int, int]] | None = None
    blocked_protein_pathway_edges: set[tuple[int, int]] | None = None
    if strict_mech_graph:
        strict_blocklists = load_mechanism_strict_blocklists(
            node_index=node_index,
            train_quadruples_path=data_cfg["train_quadruples_path"],
            val_quadruples_path=data_cfg["val_quadruples_path"],
            test_quadruples_path=data_cfg["test_quadruples_path"],
        )
        blocked_drug_protein_edges = strict_blocklists.blocked_drug_protein_edges
        blocked_protein_pathway_edges = strict_blocklists.blocked_protein_pathway_edges

    adjacency = build_mechanism_adjacency(
        edges_path=data_cfg["edges_path"],
        node_id_to_type=node_index.node_id_to_type,
        blocked_drug_protein_edges=blocked_drug_protein_edges,
        blocked_protein_pathway_edges=blocked_protein_pathway_edges,
        cache_dir=mechanism_cache_dir,
    )
    mechanism_head = MechanismHead(
        embedding_dim=node_embeddings.shape[1],
        adjacency=adjacency,
        seed=int(ckpt_seed),
        fusion_scale=fusion_scale,
    )
    mechanism_head.set_parameters(
        protein_relation=extras["protein_relation_embedding"],
        pathway_relation=extras.get("pathway_relation_embedding"),
        fusion_scale=(float(extras["fusion_scale"][0]) if "fusion_scale" in extras else None),
    )

    drug_norm = _normalize_drug_id(drug_id)
    disease_norm = _normalize_disease_id(disease_id)
    if drug_norm not in node_index.original_to_node_id:
        raise ValueError(f"Unknown drug_id: {drug_id!r} (normalized={drug_norm!r})")
    if disease_norm not in node_index.original_to_node_id:
        raise ValueError(f"Unknown disease_id: {disease_id!r} (normalized={disease_norm!r})")

    drug_node = node_index.original_to_node_id[drug_norm]
    disease_node = node_index.original_to_node_id[disease_norm]
    raw = mechanism_head.explain_pair(
        node_embeddings=node_embeddings,
        drug_id=drug_node,
        disease_id=disease_node,
        top_k=top_k,
        protein_only=protein_only,
    )

    proteins_out: list[dict[str, Any]] = []
    for row in raw["protein_candidates"]:
        node_id = int(row["node_id"])
        proteins_out.append(
            {
                "protein_id": node_index.node_id_to_original.get(node_id, str(node_id)),
                "node_id": node_id,
                "weight": float(row["weight"]),
                "score": float(row["score"]),
            }
        )

    pathways_out: list[dict[str, Any]] = []
    for row in raw["pathway_candidates"]:
        node_id = int(row["node_id"])
        pathways_out.append(
            {
                "pathway_id": node_index.node_id_to_original.get(node_id, str(node_id)),
                "node_id": node_id,
                "weight": float(row["weight"]),
            }
        )

    return {
        "drug_id": drug_norm,
        "disease_id": disease_norm,
        "top_k": top_k,
        "protein_only": protein_only,
        "proteins": proteins_out,
        "pathways": pathways_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain mechanism candidates for a drug-disease pair.")
    parser.add_argument("--config", default="configs/baseline.yaml", help="Path to YAML config.")
    parser.add_argument("--run-dir", default=None, help="Run directory containing config/checkpoint.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path (overrides run-dir checkpoint).")
    parser.add_argument("--drug-id", required=True, help="Drug ID (e.g. drug::DB00334 or DB00334).")
    parser.add_argument("--disease-id", required=True, help="Disease ID (e.g. disease::2125).")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k proteins/pathways to return.")
    args = parser.parse_args()

    run_dir, config, ckpt_path = _resolve_run_and_config(
        config_path=args.config,
        run_dir=args.run_dir,
        checkpoint_path=args.checkpoint,
    )
    result = explain_pair(
        config=config,
        checkpoint_path=ckpt_path,
        drug_id=args.drug_id,
        disease_id=args.disease_id,
        top_k=args.top_k,
    )
    print(json.dumps(result, indent=2))

    if run_dir is not None:
        out_path = run_dir / f"explain_{args.drug_id.replace(':', '_')}_{args.disease_id.replace(':', '_')}.json"
        write_json(out_path, result)
        print(f"saved_explain={out_path}")


if __name__ == "__main__":
    main()
