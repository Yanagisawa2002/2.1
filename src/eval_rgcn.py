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
    load_node_index,
    load_pair_splits,
    read_yaml,
    write_json,
)
from src.models.rgcn import RGCNIndicationModel, build_relation_graph, require_torch


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
        ckpt = run_dir_path / "checkpoint.pt"
    else:
        raise ValueError("Either --run-dir or --checkpoint must be provided.")
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    return run_dir_path, config, ckpt


def evaluate_from_checkpoint(
    *,
    config: dict[str, Any],
    checkpoint_path: str | Path,
    split: str,
    seed: int,
    device_name: str | None = None,
) -> dict[str, float]:
    torch = require_torch()
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["train"]
    eval_cfg = config["eval"]

    node_index = load_node_index(data_cfg["nodes_path"])
    splits = load_pair_splits(
        node_index=node_index,
        train_pairs_path=data_cfg["train_pairs_path"],
        val_pairs_path=data_cfg["val_pairs_path"],
        test_pairs_path=data_cfg["test_pairs_path"],
    )

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

    if device_name is None:
        device_name = str(train_cfg.get("resolved_device", train_cfg.get("device", "cpu")))
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"device={device_name!r} requested but CUDA is not available. "
            "Use --device cpu or install CUDA-enabled torch."
        )
    device = torch.device(device_name)

    ckpt = torch.load(Path(checkpoint_path), map_location="cpu")
    if "model_state" not in ckpt:
        raise ValueError(f"{checkpoint_path}: missing 'model_state' in checkpoint.")

    keep_train_indication_only = bool(
        train_cfg.get("keep_train_indication_only", ckpt.get("keep_train_indication_only", True))
    )
    train_pair_set = {(int(h), int(t)) for h, t in splits.train}
    graph = build_relation_graph(
        edges_path=data_cfg["edges_path"],
        num_nodes=node_index.num_nodes,
        keep_train_indication_only=keep_train_indication_only,
        train_indication_pairs=train_pair_set if keep_train_indication_only else None,
    )

    relations = tuple(ckpt.get("relations", graph.relations))
    if set(relations) != set(graph.relations):
        raise ValueError(
            "Checkpoint relation set does not match current graph relation set. "
            f"ckpt={sorted(relations)!r}, graph={sorted(graph.relations)!r}"
        )

    model = RGCNIndicationModel(
        num_nodes=node_index.num_nodes,
        hidden_dim=int(model_cfg.get("hidden_dim", ckpt.get("hidden_dim", 128))),
        relations=relations,
        num_layers=int(model_cfg.get("num_layers", ckpt.get("num_layers", 2))),
        dropout=float(model_cfg.get("dropout", ckpt.get("dropout", 0.1))),
        seed=int(ckpt.get("seed", 0)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    graph_torch = graph.to_torch(device)

    model.eval()
    with torch.no_grad():
        node_embeddings = model.encode(graph_torch).detach().cpu().numpy().astype("float32", copy=False)
        relation_embedding = model.decoder.relation.detach().cpu().numpy().astype("float32", copy=False)

    metrics = evaluate_link_prediction(
        node_embeddings=node_embeddings,
        relation_embedding=relation_embedding,
        eval_pairs=eval_pairs,
        disease_candidates=node_index.disease_node_ids,
        filter_tails_by_head=filter_pairs,
        auprc_negatives_per_positive=int(eval_cfg.get("auprc_negatives_per_positive", 20)),
        seed=seed,
        rank_batch_size=int(eval_cfg.get("rank_batch_size", 256)),
    )
    metrics["checkpoint_seed"] = int(ckpt.get("seed", -1))
    metrics["device"] = str(device)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate R-GCN checkpoint for indication prediction.")
    parser.add_argument("--config", default="configs/rgcn.yaml", help="Path to YAML config.")
    parser.add_argument("--run-dir", default=None, help="Run directory containing config/checkpoint.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path (overrides run-dir checkpoint).")
    parser.add_argument("--split", default="val", choices=["val", "test"], help="Evaluation split.")
    parser.add_argument("--seed", type=int, default=0, help="Seed used for AUPRC negative sampling.")
    parser.add_argument("--device", default=None, help="Override device for evaluation (e.g. cuda, cuda:0, cpu).")
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
        device_name=args.device,
    )
    print(json.dumps(metrics, indent=2))

    if run_dir is not None:
        output_name = f"metrics_{args.split}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_path = run_dir / output_name
        write_json(output_path, metrics)
        print(f"saved_metrics={output_path}")


if __name__ == "__main__":
    main()
