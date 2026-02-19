from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.baseline_common import (
    build_positive_tail_index,
    evaluate_link_prediction,
    load_checkpoint,
    load_mechanism_strict_blocklists,
    load_node_index,
    load_pair_splits,
    read_yaml,
    write_json,
)
from src.models.mechanism_head import MechanismHead, build_mechanism_adjacency


def _resolve_run_and_config(
    *,
    config_path: str | Path,
    run_dir: str | Path | None,
    checkpoint_path: str | Path | None,
) -> tuple[Path | None, dict[str, Any], Path]:
    config: dict[str, Any]
    run_dir_path: Path | None = None

    if run_dir is not None:
        run_dir_path = Path(run_dir)
        if not run_dir_path.exists():
            raise FileNotFoundError(f"run_dir not found: {run_dir_path}")
        if not run_dir_path.is_dir():
            raise NotADirectoryError(f"run_dir is not a directory: {run_dir_path}")

        run_cfg = run_dir_path / "config.yaml"
        if run_cfg.exists():
            config = read_yaml(run_cfg)
        else:
            config = read_yaml(config_path)
    else:
        config = read_yaml(config_path)

    if checkpoint_path is not None:
        ckpt_path = Path(checkpoint_path)
    elif run_dir_path is not None:
        ckpt_path = run_dir_path / "checkpoint.npz"
    else:
        raise ValueError("Either --run-dir or --checkpoint must be provided.")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return run_dir_path, config, ckpt_path


def evaluate_from_checkpoint(
    *,
    config: dict[str, Any],
    checkpoint_path: str | Path,
    split: str,
    seed: int,
) -> dict[str, float]:
    data_cfg = config["data"]
    eval_cfg = config["eval"]
    train_cfg = config.get("train", {})
    mechanism_cache_dir = data_cfg.get("mechanism_cache_dir", "artifacts/cache/mechanism_adjacency")

    node_index = load_node_index(data_cfg["nodes_path"])
    splits = load_pair_splits(
        node_index=node_index,
        train_pairs_path=data_cfg["train_pairs_path"],
        val_pairs_path=data_cfg["val_pairs_path"],
        test_pairs_path=data_cfg["test_pairs_path"],
    )

    node_embeddings, relation_embedding, ckpt_seed, extras = load_checkpoint(checkpoint_path)

    if split == "val":
        eval_pairs = splits.val
        filter_pairs = build_positive_tail_index(
            np.concatenate([splits.train, splits.val], axis=0)
        )
    elif split == "test":
        eval_pairs = splits.test
        filter_pairs = build_positive_tail_index(
            np.concatenate([splits.train, splits.val, splits.test], axis=0)
        )
    else:
        raise ValueError(f"Unsupported split: {split!r}. Use 'val' or 'test'.")

    use_mech_fusion = bool(train_cfg.get("use_mech_fusion", False))
    protein_only = bool(train_cfg.get("protein_only", False))
    strict_mech_graph = bool(train_cfg.get("strict_mech_graph", False))
    fusion_batch = None
    fusion_all = None
    if use_mech_fusion:
        required = {"protein_relation_embedding"}
        if not protein_only:
            required.add("pathway_relation_embedding")
        missing = sorted(required.difference(extras.keys()))
        if missing:
            raise ValueError(
                "Checkpoint is missing trained mechanism parameters required for use_mech_fusion=true. "
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
            fusion_scale=float(train_cfg.get("fusion_scale", 0.1)),
        )
        mechanism_head.set_parameters(
            protein_relation=extras["protein_relation_embedding"],
        )
        if "pathway_relation_embedding" in extras:
            mechanism_head.set_parameters(pathway_relation=extras["pathway_relation_embedding"])
        if "fusion_scale" in extras and extras["fusion_scale"].size > 0:
            mechanism_head.set_parameters(
                fusion_scale=float(extras["fusion_scale"][0]),
            )
        fusion_batch = lambda h, t: mechanism_head.fusion_scores_batch(
            node_embeddings=node_embeddings,
            heads=h,
            tails=t,
            protein_only=protein_only,
        )
        fusion_all = lambda h, t: mechanism_head.fusion_scores_for_all_tails(
            node_embeddings=node_embeddings,
            head=h,
            candidate_tails=t,
            protein_only=protein_only,
        )

    metrics = evaluate_link_prediction(
        node_embeddings=node_embeddings,
        relation_embedding=relation_embedding,
        eval_pairs=eval_pairs,
        disease_candidates=node_index.disease_node_ids,
        filter_tails_by_head=filter_pairs,
        auprc_negatives_per_positive=int(eval_cfg.get("auprc_negatives_per_positive", 20)),
        seed=seed,
        fusion_batch_scorer=fusion_batch,
        fusion_all_tails_scorer=fusion_all,
        rank_batch_size=int(eval_cfg.get("rank_batch_size", 256)),
    )
    metrics["checkpoint_seed"] = int(ckpt_seed)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DistMult baseline checkpoint.")
    parser.add_argument("--config", default="configs/baseline.yaml", help="Path to YAML config.")
    parser.add_argument("--run-dir", default=None, help="Run directory containing config/checkpoint.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path (overrides run-dir checkpoint).")
    parser.add_argument("--split", default="val", choices=["val", "test"], help="Evaluation split.")
    parser.add_argument("--seed", type=int, default=0, help="Seed used for AUPRC negative sampling.")
    args = parser.parse_args()

    run_dir, config, ckpt_path = _resolve_run_and_config(
        config_path=args.config,
        run_dir=args.run_dir,
        checkpoint_path=args.checkpoint,
    )
    metrics = evaluate_from_checkpoint(
        config=config,
        checkpoint_path=ckpt_path,
        split=args.split,
        seed=args.seed,
    )
    print(json.dumps(metrics, indent=2))

    if run_dir is not None:
        output_name = f"metrics_{args.split}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_path = run_dir / output_name
        write_json(output_path, metrics)
        print(f"saved_metrics={output_path}")


if __name__ == "__main__":
    main()
