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
    sample_negative_tails,
    write_json,
    write_yaml,
)
from src.models.rgcn import RGCNIndicationModel, build_relation_graph, require_torch

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional at runtime
    tqdm = None  # type: ignore[assignment]


def _get_eval_filter(split_name: str, train: np.ndarray, val: np.ndarray, test: np.ndarray) -> dict[int, set[int]]:
    if split_name == "val":
        data = np.concatenate([train, val], axis=0)
    elif split_name == "test":
        data = np.concatenate([train, val, test], axis=0)
    else:
        raise ValueError(f"Unsupported eval split: {split_name!r}")
    return build_positive_tail_index(data)


def _resolve_device(device_name: str) -> Any:
    torch = require_torch()
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"device={device_name!r} requested but CUDA is not available. "
            "Install CUDA-enabled torch and verify nvidia-smi."
        )
    return torch.device(device_name)


def _maybe_eval(
    *,
    model: RGCNIndicationModel,
    graph_torch: Any,
    eval_pairs: np.ndarray,
    node_index: Any,
    eval_filter: dict[int, set[int]],
    auprc_neg_pp: int,
    seed: int,
    rank_batch_size: int,
) -> dict[str, float]:
    torch = require_torch()
    model.eval()
    with torch.no_grad():
        node_embeddings = model.encode(graph_torch).detach().cpu().numpy().astype(np.float32, copy=False)
        relation_embedding = model.decoder.relation.detach().cpu().numpy().astype(np.float32, copy=False)
    return evaluate_link_prediction(
        node_embeddings=node_embeddings,
        relation_embedding=relation_embedding,
        eval_pairs=eval_pairs,
        disease_candidates=node_index.disease_node_ids,
        filter_tails_by_head=eval_filter,
        auprc_negatives_per_positive=auprc_neg_pp,
        seed=seed,
        rank_batch_size=rank_batch_size,
    )


def train_rgcn(config: dict[str, Any], *, seed: int, smoke: bool) -> tuple[Path, dict[str, Any]]:
    torch = require_torch()
    nnf = torch.nn.functional

    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["train"]
    eval_cfg = config["eval"]
    out_cfg = config["output"]

    node_index = load_node_index(data_cfg["nodes_path"])
    splits = load_pair_splits(
        node_index=node_index,
        train_pairs_path=data_cfg["train_pairs_path"],
        val_pairs_path=data_cfg["val_pairs_path"],
        test_pairs_path=data_cfg["test_pairs_path"],
    )
    train_pairs = splits.train
    if train_pairs.size == 0:
        raise ValueError("Train split is empty.")

    hidden_dim = int(model_cfg.get("hidden_dim", 128))
    num_layers = int(model_cfg.get("num_layers", 2))
    dropout = float(model_cfg.get("dropout", 0.1))

    device_name = str(train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    learning_rate = float(train_cfg.get("learning_rate", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    negatives_per_positive = int(train_cfg.get("negatives_per_positive", 10))
    unlabeled_negative_weight = float(train_cfg.get("unlabeled_negative_weight", 1.0))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 0.0))
    epochs = int(train_cfg.get("epochs", 50))
    eval_every = int(train_cfg.get("eval_every", 1))
    keep_train_indication_only = bool(train_cfg.get("keep_train_indication_only", True))

    early_stopping_patience = int(train_cfg.get("early_stopping_patience", 0))
    early_stopping_metric = str(train_cfg.get("early_stopping_metric", "auprc"))
    early_stopping_mode = str(train_cfg.get("early_stopping_mode", "max"))
    early_stopping_min_delta = float(train_cfg.get("early_stopping_min_delta", 0.0))
    early_stopping_restore_best = bool(train_cfg.get("early_stopping_restore_best", True))

    eval_split_name = str(eval_cfg.get("split", "val"))
    auprc_neg_pp = int(eval_cfg.get("auprc_negatives_per_positive", 20))
    rank_batch_size = int(eval_cfg.get("rank_batch_size", 256))

    if smoke:
        epochs = 1
        auprc_neg_pp = min(auprc_neg_pp, 10)

    if hidden_dim <= 0 or num_layers <= 0:
        raise ValueError(f"Invalid R-GCN model config: hidden_dim={hidden_dim}, num_layers={num_layers}")
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}")
    if negatives_per_positive <= 0:
        raise ValueError(f"negatives_per_positive must be positive, got {negatives_per_positive}")
    if unlabeled_negative_weight < 0.0 or unlabeled_negative_weight > 1.0:
        raise ValueError(
            f"unlabeled_negative_weight must be in [0, 1], got {unlabeled_negative_weight}"
        )
    if eval_every <= 0:
        raise ValueError(f"eval_every must be positive, got {eval_every}")
    if early_stopping_patience < 0:
        raise ValueError(f"early_stopping_patience must be >= 0, got {early_stopping_patience}")
    if early_stopping_mode not in {"max", "min"}:
        raise ValueError(f"early_stopping_mode must be 'max' or 'min', got {early_stopping_mode!r}")
    if early_stopping_min_delta < 0.0:
        raise ValueError(f"early_stopping_min_delta must be >= 0, got {early_stopping_min_delta}")
    if rank_batch_size <= 0:
        raise ValueError(f"rank_batch_size must be positive, got {rank_batch_size}")

    device = _resolve_device(device_name)
    train_pair_set = {(int(h), int(t)) for h, t in train_pairs}
    graph = build_relation_graph(
        edges_path=data_cfg["edges_path"],
        num_nodes=node_index.num_nodes,
        keep_train_indication_only=keep_train_indication_only,
        train_indication_pairs=train_pair_set if keep_train_indication_only else None,
    )
    graph_torch = graph.to_torch(device)

    model = RGCNIndicationModel(
        num_nodes=node_index.num_nodes,
        hidden_dim=hidden_dim,
        relations=graph.relations,
        num_layers=num_layers,
        dropout=dropout,
        seed=seed,
    ).to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    if eval_split_name == "val":
        eval_pairs = splits.val
    elif eval_split_name == "test":
        eval_pairs = splits.test
    else:
        raise ValueError(f"Unsupported eval split in config: {eval_split_name!r}")
    eval_filter = _get_eval_filter(eval_split_name, splits.train, splits.val, splits.test)

    train_heads_np = train_pairs[:, 0].astype(np.int64, copy=False)
    train_tails_np = train_pairs[:, 1].astype(np.int64, copy=False)
    train_pos_by_head = build_positive_tail_index(train_pairs)
    rng = np.random.default_rng(seed)

    run_root = Path(out_cfg["runs_dir"])
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = run_root / f"rgcn_indication_{timestamp}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)

    resolved_cfg = dict(config)
    resolved_cfg["seed"] = seed
    resolved_cfg["smoke"] = smoke
    resolved_cfg["train"] = dict(resolved_cfg["train"])
    resolved_cfg["eval"] = dict(resolved_cfg["eval"])
    resolved_cfg["train"]["epochs"] = epochs
    resolved_cfg["eval"]["auprc_negatives_per_positive"] = auprc_neg_pp
    resolved_cfg["train"]["resolved_device"] = str(device)
    resolved_cfg["graph"] = {
        "num_nodes": graph.num_nodes,
        "relations": list(graph.relations),
        "edge_count_by_relation": {
            rel: int(graph.edge_index_by_rel[rel].shape[1]) for rel in graph.relations
        },
        "keep_train_indication_only": keep_train_indication_only,
    }
    write_yaml(run_dir / "config.yaml", resolved_cfg)

    epoch_logs: list[dict[str, float | int]] = []
    best_metric_value: float | None = None
    best_metric_epoch: int | None = None
    no_improve_rounds = 0
    early_stopped = False
    best_state_dict: dict[str, Any] | None = None

    progress = (
        tqdm(range(1, epochs + 1), desc="train-rgcn", unit="epoch", dynamic_ncols=True)
        if tqdm
        else range(1, epochs + 1)
    )
    for epoch in progress:
        model.train()
        optimizer.zero_grad(set_to_none=True)

        node_embeddings = model.encode(graph_torch)
        heads_t = torch.as_tensor(train_heads_np, dtype=torch.long, device=device)
        tails_t = torch.as_tensor(train_tails_np, dtype=torch.long, device=device)
        pos_scores = model.score_pairs(node_embeddings, heads_t, tails_t)
        pos_loss = nnf.softplus(-pos_scores).mean()

        neg_loss = torch.zeros((), device=device, dtype=pos_loss.dtype)
        neg_scores = torch.empty((0,), device=device, dtype=pos_scores.dtype)
        if unlabeled_negative_weight > 0.0:
            repeat_heads_np = np.repeat(train_heads_np, negatives_per_positive)
            neg_tails_np = sample_negative_tails(
                heads=repeat_heads_np,
                disease_candidates=node_index.disease_node_ids,
                positive_tails_by_head=train_pos_by_head,
                rng=rng,
            )
            repeat_heads_t = torch.as_tensor(repeat_heads_np, dtype=torch.long, device=device)
            neg_tails_t = torch.as_tensor(neg_tails_np, dtype=torch.long, device=device)
            neg_scores = model.score_pairs(node_embeddings, repeat_heads_t, neg_tails_t)
            neg_loss = nnf.softplus(neg_scores).mean()

        total_loss = pos_loss + unlabeled_negative_weight * neg_loss
        if not torch.isfinite(total_loss):
            raise ValueError(
                "Non-finite R-GCN loss detected. "
                "Reduce learning_rate or negatives_per_positive."
            )

        total_loss.backward()
        if max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(list(model.parameters()), max_grad_norm)
        optimizer.step()

        log: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": float(total_loss.detach().cpu().item()),
            "train_loss_indication_pos": float(pos_loss.detach().cpu().item()),
            "train_loss_indication_unlabeled_neg": float(neg_loss.detach().cpu().item()),
            "train_num_pos_pairs": int(train_heads_np.shape[0]),
            "train_num_neg_pairs": int(neg_scores.shape[0]),
        }

        if epoch % eval_every == 0:
            val_metrics = _maybe_eval(
                model=model,
                graph_torch=graph_torch,
                eval_pairs=eval_pairs,
                node_index=node_index,
                eval_filter=eval_filter,
                auprc_neg_pp=auprc_neg_pp,
                seed=seed + epoch,
                rank_batch_size=rank_batch_size,
            )
            for k, v in val_metrics.items():
                log[f"{eval_split_name}_{k}"] = float(v)

            monitor_key = f"{eval_split_name}_{early_stopping_metric}"
            if monitor_key not in log:
                raise ValueError(
                    f"early_stopping_metric={early_stopping_metric!r} not in eval metrics."
                )
            current = float(log[monitor_key])
            improved = False
            if best_metric_value is None:
                improved = True
            elif early_stopping_mode == "max":
                improved = current > (best_metric_value + early_stopping_min_delta)
            else:
                improved = current < (best_metric_value - early_stopping_min_delta)

            if improved:
                best_metric_value = current
                best_metric_epoch = epoch
                no_improve_rounds = 0
                best_state_dict = model.state_dict()
            else:
                no_improve_rounds += 1
                if early_stopping_patience > 0 and no_improve_rounds >= early_stopping_patience:
                    early_stopped = True

        epoch_logs.append(log)
        if tqdm:
            tqdm.write(json.dumps(log))
            postfix = {"loss": f"{log['train_loss']:.4f}"}
            metric_key = f"{eval_split_name}_{early_stopping_metric}"
            if metric_key in log:
                postfix[early_stopping_metric] = f"{float(log[metric_key]):.4f}"
            progress.set_postfix(postfix)
        else:
            print(json.dumps(log), flush=True)

        if early_stopped:
            stop_msg = {
                "event": "early_stopping",
                "epoch": epoch,
                "patience": early_stopping_patience,
                "metric": early_stopping_metric,
                "mode": early_stopping_mode,
                "best_epoch": best_metric_epoch,
                "best_metric": best_metric_value,
            }
            if tqdm:
                tqdm.write(json.dumps(stop_msg))
            else:
                print(json.dumps(stop_msg), flush=True)
            break

    if tqdm:
        progress.close()

    if early_stopping_restore_best and best_state_dict is not None:
        model.load_state_dict(best_state_dict, strict=True)

    checkpoint_path = run_dir / "checkpoint.pt"
    torch.save(
        {
            "seed": int(seed),
            "model_state": model.state_dict(),
            "num_nodes": int(node_index.num_nodes),
            "hidden_dim": int(hidden_dim),
            "num_layers": int(num_layers),
            "dropout": float(dropout),
            "relations": list(graph.relations),
            "keep_train_indication_only": bool(keep_train_indication_only),
        },
        checkpoint_path,
    )

    final_eval = _maybe_eval(
        model=model,
        graph_torch=graph_torch,
        eval_pairs=eval_pairs,
        node_index=node_index,
        eval_filter=eval_filter,
        auprc_neg_pp=auprc_neg_pp,
        seed=seed + 10_000,
        rank_batch_size=rank_batch_size,
    )
    metrics_payload = {
        "seed": seed,
        "smoke": smoke,
        "split": eval_split_name,
        "epochs": epochs,
        "epochs_completed": len(epoch_logs),
        "device": str(device),
        "graph_relations": list(graph.relations),
        "early_stopping": {
            "enabled": early_stopping_patience > 0,
            "patience": early_stopping_patience,
            "metric": early_stopping_metric,
            "mode": early_stopping_mode,
            "min_delta": early_stopping_min_delta,
            "restore_best": early_stopping_restore_best,
            "stopped_early": early_stopped,
            "best_epoch": best_metric_epoch,
            "best_metric": best_metric_value,
        },
        "epoch_logs": epoch_logs,
        "final_eval": final_eval,
    }
    write_json(run_dir / "metrics.json", metrics_payload)
    return run_dir, metrics_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train R-GCN indication predictor (full graph multi-relation message passing)."
    )
    parser.add_argument("--config", default="configs/rgcn.yaml", help="Path to YAML config file.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override.")
    parser.add_argument("--smoke", action="store_true", help="Run 1-epoch smoke training.")
    args = parser.parse_args()

    config = read_yaml(args.config)
    seed = int(config.get("seed", 0) if args.seed is None else args.seed)
    run_dir, metrics = train_rgcn(config, seed=seed, smoke=args.smoke)

    print(f"run_dir={run_dir}")
    print(json.dumps(metrics["final_eval"], indent=2))


if __name__ == "__main__":
    main()
