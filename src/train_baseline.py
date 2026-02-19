from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.baseline_common import (
    build_mechanism_strict_blocklists,
    build_positive_tail_index,
    distmult_scores,
    evaluate_link_prediction,
    load_quadruple_splits,
    load_node_index,
    load_pair_splits,
    read_yaml,
    sample_negative_tails,
    save_checkpoint,
    sigmoid,
    write_json,
    write_yaml,
)
from src.models.mechanism_head import MechanismHead, build_mechanism_adjacency
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency at runtime
    tqdm = None  # type: ignore[assignment]


def _get_eval_filter(split_name: str, train: np.ndarray, val: np.ndarray, test: np.ndarray) -> dict[int, set[int]]:
    if split_name == "val":
        data = np.concatenate([train, val], axis=0)
    elif split_name == "test":
        data = np.concatenate([train, val, test], axis=0)
    else:
        raise ValueError(f"Unsupported eval split: {split_name}")
    return build_positive_tail_index(data)


def _run_train_step(
    *,
    node_embeddings: np.ndarray,
    relation_embedding: np.ndarray,
    batch_pairs: np.ndarray,
    disease_candidates: np.ndarray,
    positive_tails_by_head: dict[int, set[int]],
    negatives_per_positive: int,
    learning_rate: float,
    weight_decay: float,
    rng: np.random.Generator,
    mechanism_head: MechanismHead | None,
    use_quadruple: bool,
    protein_only: bool,
    lambda_protein: float,
    lambda_pathway: float,
    unlabeled_negative_weight: float,
    quad_batch: np.ndarray | None,
) -> dict[str, float]:
    heads = batch_pairs[:, 0].astype(np.int64, copy=False)
    tails = batch_pairs[:, 1].astype(np.int64, copy=False)
    pos_scores = distmult_scores(node_embeddings, relation_embedding, heads, tails)
    if not np.isfinite(pos_scores).all():
        raise ValueError(
            "Non-finite positive indication scores detected. "
            "Training diverged; reduce learning_rate or negatives_per_positive."
        )

    pos_loss = np.logaddexp(0.0, -pos_scores).mean()
    unlabeled_neg_loss = 0.0
    repeat_heads = np.empty((0,), dtype=np.int64)
    neg_tails = np.empty((0,), dtype=np.int64)
    neg_scores = np.empty((0,), dtype=np.float32)
    if unlabeled_negative_weight > 0.0:
        repeat_heads = np.repeat(heads, negatives_per_positive)
        neg_tails = sample_negative_tails(
            heads=repeat_heads,
            disease_candidates=disease_candidates,
            positive_tails_by_head=positive_tails_by_head,
            rng=rng,
        )
        neg_scores = distmult_scores(node_embeddings, relation_embedding, repeat_heads, neg_tails)
        if not np.isfinite(neg_scores).all():
            raise ValueError(
                "Non-finite unlabeled-negative indication scores detected. "
                "Training diverged; reduce learning_rate or negatives_per_positive."
            )
        unlabeled_neg_loss = float(np.logaddexp(0.0, neg_scores).mean())
    indication_loss = float(pos_loss + unlabeled_negative_weight * unlabeled_neg_loss)

    grad_node = np.zeros_like(node_embeddings, dtype=np.float32)
    grad_rel = np.zeros_like(relation_embedding, dtype=np.float32)

    pos_h = node_embeddings[heads]
    pos_t = node_embeddings[tails]
    pos_g = (sigmoid(pos_scores) - 1.0).astype(np.float32)
    np.add.at(grad_node, heads, pos_g[:, None] * (relation_embedding[None, :] * pos_t))
    np.add.at(grad_node, tails, pos_g[:, None] * (relation_embedding[None, :] * pos_h))
    grad_rel += np.sum(pos_g[:, None] * (pos_h * pos_t), axis=0)

    if neg_scores.size > 0:
        neg_h = node_embeddings[repeat_heads]
        neg_t = node_embeddings[neg_tails]
        neg_g = (unlabeled_negative_weight * sigmoid(neg_scores)).astype(np.float32)
        np.add.at(grad_node, repeat_heads, neg_g[:, None] * (relation_embedding[None, :] * neg_t))
        np.add.at(grad_node, neg_tails, neg_g[:, None] * (relation_embedding[None, :] * neg_h))
        grad_rel += np.sum(neg_g[:, None] * (neg_h * neg_t), axis=0)

    denom = float(max(1, heads.shape[0]))
    grad_node /= denom
    grad_rel /= denom

    if weight_decay > 0.0:
        grad_node += weight_decay * node_embeddings
        grad_rel += weight_decay * relation_embedding

    protein_loss = 0.0
    pathway_loss = 0.0
    protein_count = 0
    pathway_count = 0
    if use_quadruple and mechanism_head is not None and quad_batch is not None and quad_batch.size > 0:
        mech = mechanism_head.compute_losses_and_grads(
            node_embeddings=node_embeddings,
            quadruples=quad_batch,
            protein_only=protein_only,
        )
        protein_loss = float(mech["loss_protein"])
        pathway_loss = float(mech["loss_pathway"])
        protein_count = int(mech["protein_count"])
        pathway_count = int(mech["pathway_count"])

        grad_node += lambda_protein * mech["grad_node_protein"]  # type: ignore[operator]
        grad_node += (0.0 if protein_only else lambda_pathway) * mech["grad_node_pathway"]  # type: ignore[operator]

        grad_rel_protein = lambda_protein * mech["grad_rel_protein"]  # type: ignore[operator]
        if weight_decay > 0.0:
            grad_rel_protein += weight_decay * mechanism_head.protein_relation
        mechanism_head.protein_relation -= learning_rate * grad_rel_protein.astype(np.float32)

        if not protein_only:
            grad_rel_pathway = lambda_pathway * mech["grad_rel_pathway"]  # type: ignore[operator]
            if weight_decay > 0.0:
                grad_rel_pathway += weight_decay * mechanism_head.pathway_relation
            mechanism_head.pathway_relation -= learning_rate * grad_rel_pathway.astype(np.float32)

    node_embeddings -= learning_rate * grad_node
    relation_embedding -= learning_rate * grad_rel
    total_loss = indication_loss + lambda_protein * protein_loss + (0.0 if protein_only else lambda_pathway * pathway_loss)
    return {
        "loss_total": float(total_loss),
        "loss_indication": float(indication_loss),
        "loss_indication_pos": float(pos_loss),
        "loss_indication_unlabeled_neg": float(unlabeled_neg_loss),
        "loss_protein": float(protein_loss),
        "loss_pathway": float(pathway_loss),
        "protein_count": float(protein_count),
        "pathway_count": float(pathway_count),
    }


def train_baseline(config: dict[str, Any], *, seed: int, smoke: bool) -> tuple[Path, dict[str, Any]]:
    data_cfg = config["data"]
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

    embedding_dim = int(train_cfg["embedding_dim"])
    learning_rate = float(train_cfg["learning_rate"])
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    batch_size = int(train_cfg["batch_size"])
    negatives_per_positive = int(train_cfg["negatives_per_positive"])
    epochs = int(train_cfg["epochs"])
    eval_every = int(train_cfg.get("eval_every", 1))
    early_stopping_patience = int(train_cfg.get("early_stopping_patience", 0))
    early_stopping_metric = str(train_cfg.get("early_stopping_metric", "auprc"))
    early_stopping_mode = str(train_cfg.get("early_stopping_mode", "max"))
    early_stopping_min_delta = float(train_cfg.get("early_stopping_min_delta", 0.0))
    early_stopping_restore_best = bool(train_cfg.get("early_stopping_restore_best", True))
    eval_split_name = str(eval_cfg.get("split", "val"))
    auprc_neg_pp = int(eval_cfg.get("auprc_negatives_per_positive", 20))
    rank_batch_size = int(eval_cfg.get("rank_batch_size", 256))
    use_quadruple = bool(train_cfg.get("use_quadruple", False))
    protein_only = bool(train_cfg.get("protein_only", False))
    use_mech_fusion = bool(train_cfg.get("use_mech_fusion", False))
    strict_mech_graph = bool(train_cfg.get("strict_mech_graph", False))
    lambda_protein = float(train_cfg.get("lambda_protein", 1.0))
    lambda_pathway = float(train_cfg.get("lambda_pathway", 1.0))
    unlabeled_negative_weight = float(train_cfg.get("unlabeled_negative_weight", 1.0))
    fusion_scale = float(train_cfg.get("fusion_scale", 0.1))
    mechanism_cache_dir = data_cfg.get("mechanism_cache_dir", "artifacts/cache/mechanism_adjacency")

    if smoke:
        epochs = 1
        auprc_neg_pp = min(auprc_neg_pp, 10)

    if embedding_dim <= 0:
        raise ValueError(f"embedding_dim must be > 0, got {embedding_dim}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    if negatives_per_positive <= 0:
        raise ValueError(f"negatives_per_positive must be > 0, got {negatives_per_positive}")
    if epochs <= 0:
        raise ValueError(f"epochs must be > 0, got {epochs}")
    if eval_every <= 0:
        raise ValueError(f"eval_every must be > 0, got {eval_every}")
    if early_stopping_patience < 0:
        raise ValueError(f"early_stopping_patience must be >= 0, got {early_stopping_patience}")
    if early_stopping_mode not in {"max", "min"}:
        raise ValueError(f"early_stopping_mode must be 'max' or 'min', got {early_stopping_mode!r}")
    if early_stopping_min_delta < 0.0:
        raise ValueError(f"early_stopping_min_delta must be >= 0, got {early_stopping_min_delta}")
    if lambda_protein < 0.0 or lambda_pathway < 0.0:
        raise ValueError(
            f"lambda_protein and lambda_pathway must be non-negative. got {lambda_protein}, {lambda_pathway}"
        )
    if unlabeled_negative_weight < 0.0 or unlabeled_negative_weight > 1.0:
        raise ValueError(
            f"unlabeled_negative_weight must be in [0, 1]. got {unlabeled_negative_weight}"
        )
    if use_mech_fusion and not use_quadruple:
        raise ValueError(
            "use_mech_fusion=true requires use_quadruple=true so mechanism parameters are trained."
        )
    if strict_mech_graph and not (use_quadruple or use_mech_fusion):
        raise ValueError(
            "strict_mech_graph=true requires mechanism training/inference (use_quadruple=true)."
        )

    rng = np.random.default_rng(seed)
    node_embeddings = rng.normal(0.0, 0.05, size=(node_index.num_nodes, embedding_dim)).astype(np.float32)
    relation_embedding = rng.normal(0.0, 0.05, size=(embedding_dim,)).astype(np.float32)

    mechanism_head: MechanismHead | None = None
    train_quads = np.empty((0, 4), dtype=np.int64)
    strict_blocklist_meta: dict[str, int | bool] | None = None
    quad_splits = None
    if use_quadruple or strict_mech_graph:
        quad_splits = load_quadruple_splits(
            node_index=node_index,
            train_quadruples_path=data_cfg["train_quadruples_path"],
            val_quadruples_path=data_cfg["val_quadruples_path"],
            test_quadruples_path=data_cfg["test_quadruples_path"],
        )
    if use_quadruple or use_mech_fusion:
        blocked_drug_protein_edges: set[tuple[int, int]] | None = None
        blocked_protein_pathway_edges: set[tuple[int, int]] | None = None
        if strict_mech_graph:
            if quad_splits is None:
                raise RuntimeError("Internal error: strict_mech_graph requested without quadruple splits.")
            strict_blocklists = build_mechanism_strict_blocklists(quad_splits)
            blocked_drug_protein_edges = strict_blocklists.blocked_drug_protein_edges
            blocked_protein_pathway_edges = strict_blocklists.blocked_protein_pathway_edges
            strict_blocklist_meta = {
                "enabled": True,
                "blocked_drug_protein_edges": len(blocked_drug_protein_edges),
                "blocked_protein_pathway_edges": len(blocked_protein_pathway_edges),
                "train_drug_protein_edges": strict_blocklists.train_drug_protein_edge_count,
                "train_protein_pathway_edges": strict_blocklists.train_protein_pathway_edge_count,
                "holdout_drug_protein_edges": strict_blocklists.holdout_drug_protein_edge_count,
                "holdout_protein_pathway_edges": strict_blocklists.holdout_protein_pathway_edge_count,
            }
        else:
            strict_blocklist_meta = {
                "enabled": False,
                "blocked_drug_protein_edges": 0,
                "blocked_protein_pathway_edges": 0,
            }
        adjacency = build_mechanism_adjacency(
            edges_path=data_cfg["edges_path"],
            node_id_to_type=node_index.node_id_to_type,
            blocked_drug_protein_edges=blocked_drug_protein_edges,
            blocked_protein_pathway_edges=blocked_protein_pathway_edges,
            cache_dir=mechanism_cache_dir,
        )
        mechanism_head = MechanismHead(
            embedding_dim=embedding_dim,
            adjacency=adjacency,
            seed=seed,
            fusion_scale=fusion_scale,
        )
    if use_quadruple:
        if quad_splits is None:
            raise RuntimeError("Internal error: use_quadruple requested without loaded quadruple splits.")
        train_quads = quad_splits.train
        if train_quads.size == 0:
            raise ValueError(
                "use_quadruple=true but train quadruples are empty after validation/mapping. "
                "Check split artifacts and pathway/protein ID formats."
            )

    train_pos_by_head = build_positive_tail_index(train_pairs)
    if eval_split_name == "val":
        eval_pairs = splits.val
    elif eval_split_name == "test":
        eval_pairs = splits.test
    else:
        raise ValueError(f"Unsupported eval split in config: {eval_split_name!r}")

    eval_filter = _get_eval_filter(eval_split_name, splits.train, splits.val, splits.test)

    run_root = Path(out_cfg["runs_dir"])
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = run_root / f"baseline_distmult_{timestamp}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)

    resolved_cfg = dict(config)
    resolved_cfg["seed"] = seed
    resolved_cfg["smoke"] = smoke
    resolved_cfg["train"] = dict(resolved_cfg["train"])
    resolved_cfg["eval"] = dict(resolved_cfg["eval"])
    resolved_cfg["train"]["epochs"] = epochs
    resolved_cfg["eval"]["auprc_negatives_per_positive"] = auprc_neg_pp
    write_yaml(run_dir / "config.yaml", resolved_cfg)

    epoch_logs: list[dict[str, float | int]] = []
    best_metric_value: float | None = None
    best_metric_epoch: int | None = None
    no_improve_rounds = 0
    early_stopped = False
    best_node_embeddings: np.ndarray | None = None
    best_relation_embedding: np.ndarray | None = None
    best_mechanism_params: dict[str, np.ndarray] | None = None
    n_train = train_pairs.shape[0]
    indices = np.arange(n_train, dtype=np.int64)

    progress = tqdm(range(1, epochs + 1), desc="train", unit="epoch", dynamic_ncols=True) if tqdm else range(1, epochs + 1)
    for epoch in progress:
        rng.shuffle(indices)
        epoch_loss_total = 0.0
        epoch_loss_indication = 0.0
        epoch_loss_indication_pos = 0.0
        epoch_loss_indication_unlabeled_neg = 0.0
        epoch_loss_protein = 0.0
        epoch_loss_pathway = 0.0
        epoch_protein_count = 0.0
        epoch_pathway_count = 0.0
        epoch_steps = 0
        quad_indices = np.arange(train_quads.shape[0], dtype=np.int64)
        quad_ptr = 0
        if train_quads.shape[0] > 0:
            rng.shuffle(quad_indices)

        for start in range(0, n_train, batch_size):
            batch_idx = indices[start : start + batch_size]
            batch_pairs = train_pairs[batch_idx]
            quad_batch = None
            if use_quadruple and train_quads.shape[0] > 0:
                remaining = train_quads.shape[0] - quad_ptr
                if remaining > 0:
                    qbs = min(batch_pairs.shape[0], remaining)
                    picked = quad_indices[quad_ptr : quad_ptr + qbs]
                    quad_ptr += qbs
                    quad_batch = train_quads[picked]

            step = _run_train_step(
                node_embeddings=node_embeddings,
                relation_embedding=relation_embedding,
                batch_pairs=batch_pairs,
                disease_candidates=node_index.disease_node_ids,
                positive_tails_by_head=train_pos_by_head,
                negatives_per_positive=negatives_per_positive,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                rng=rng,
                mechanism_head=mechanism_head,
                use_quadruple=use_quadruple,
                protein_only=protein_only,
                lambda_protein=lambda_protein,
                lambda_pathway=lambda_pathway,
                unlabeled_negative_weight=unlabeled_negative_weight,
                quad_batch=quad_batch,
            )
            epoch_loss_total += float(step["loss_total"])
            epoch_loss_indication += float(step["loss_indication"])
            epoch_loss_indication_pos += float(step["loss_indication_pos"])
            epoch_loss_indication_unlabeled_neg += float(step["loss_indication_unlabeled_neg"])
            epoch_loss_protein += float(step["loss_protein"])
            epoch_loss_pathway += float(step["loss_pathway"])
            epoch_protein_count += float(step["protein_count"])
            epoch_pathway_count += float(step["pathway_count"])
            epoch_steps += 1

        log: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": float(epoch_loss_total / max(1, epoch_steps)),
            "train_loss_indication": float(epoch_loss_indication / max(1, epoch_steps)),
            "train_loss_indication_pos": float(epoch_loss_indication_pos / max(1, epoch_steps)),
            "train_loss_indication_unlabeled_neg": float(epoch_loss_indication_unlabeled_neg / max(1, epoch_steps)),
            "train_loss_protein": float(epoch_loss_protein / max(1, epoch_steps)),
            "train_loss_pathway": float(epoch_loss_pathway / max(1, epoch_steps)),
            "train_protein_supervised": float(epoch_protein_count),
            "train_pathway_supervised": float(epoch_pathway_count),
            "train_quadruple_rows_used": float(quad_ptr),
        }
        if epoch % eval_every == 0:
            fusion_batch = None
            fusion_all = None
            if use_mech_fusion and mechanism_head is not None:
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
            val_metrics = evaluate_link_prediction(
                node_embeddings=node_embeddings,
                relation_embedding=relation_embedding,
                eval_pairs=eval_pairs,
                disease_candidates=node_index.disease_node_ids,
                filter_tails_by_head=eval_filter,
                auprc_negatives_per_positive=auprc_neg_pp,
                seed=seed + epoch,
                fusion_batch_scorer=fusion_batch,
                fusion_all_tails_scorer=fusion_all,
                rank_batch_size=rank_batch_size,
            )
            for k, v in val_metrics.items():
                log[f"{eval_split_name}_{k}"] = float(v)
            monitor_key = f"{eval_split_name}_{early_stopping_metric}"
            if monitor_key not in log:
                raise ValueError(
                    f"early_stopping_metric={early_stopping_metric!r} not found in evaluation metrics. "
                    f"Available keys: {sorted(val_metrics.keys())!r}"
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
                best_node_embeddings = node_embeddings.copy()
                best_relation_embedding = relation_embedding.copy()
                if mechanism_head is not None:
                    best_mechanism_params = mechanism_head.export_parameters()
            else:
                no_improve_rounds += 1
                if early_stopping_patience > 0 and no_improve_rounds >= early_stopping_patience:
                    early_stopped = True
        epoch_logs.append(log)
        if tqdm:
            tqdm.write(json.dumps(log))
            postfix = {
                "loss": f"{log['train_loss']:.4f}",
                "ind": f"{log['train_loss_indication']:.4f}",
            }
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

    if not np.isfinite(node_embeddings).all() or not np.isfinite(relation_embedding).all():
        raise ValueError(
            "Non-finite model parameters detected after training. "
            "Reduce learning_rate and rerun."
        )

    if early_stopping_restore_best and best_node_embeddings is not None and best_relation_embedding is not None:
        node_embeddings = best_node_embeddings
        relation_embedding = best_relation_embedding
        if mechanism_head is not None and best_mechanism_params is not None:
            mechanism_head.set_parameters(
                protein_relation=best_mechanism_params.get("protein_relation_embedding"),
                pathway_relation=best_mechanism_params.get("pathway_relation_embedding"),
                fusion_scale=(
                    float(best_mechanism_params["fusion_scale"][0])
                    if "fusion_scale" in best_mechanism_params and best_mechanism_params["fusion_scale"].size > 0
                    else None
                ),
            )

    checkpoint_path = run_dir / "checkpoint.npz"
    extra_arrays = mechanism_head.export_parameters() if mechanism_head is not None else None
    save_checkpoint(
        checkpoint_path=checkpoint_path,
        node_embeddings=node_embeddings,
        relation_embedding=relation_embedding,
        seed=seed,
        extra_arrays=extra_arrays,
    )

    fusion_batch = None
    fusion_all = None
    if use_mech_fusion and mechanism_head is not None:
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
    final_eval = evaluate_link_prediction(
        node_embeddings=node_embeddings,
        relation_embedding=relation_embedding,
        eval_pairs=eval_pairs,
        disease_candidates=node_index.disease_node_ids,
        filter_tails_by_head=eval_filter,
        auprc_negatives_per_positive=auprc_neg_pp,
        seed=seed + 10_000,
        fusion_batch_scorer=fusion_batch,
        fusion_all_tails_scorer=fusion_all,
        rank_batch_size=rank_batch_size,
    )
    metrics_payload = {
        "seed": seed,
        "smoke": smoke,
        "split": eval_split_name,
        "epochs": epochs,
        "epochs_completed": len(epoch_logs),
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
        "mechanism_graph": strict_blocklist_meta,
        "epoch_logs": epoch_logs,
        "final_eval": final_eval,
    }
    write_json(run_dir / "metrics.json", metrics_payload)
    return run_dir, metrics_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DistMult baseline for indication link prediction.")
    parser.add_argument("--config", default="configs/baseline.yaml", help="Path to YAML config file.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run 1-epoch smoke training (overrides epochs=1).",
    )
    args = parser.parse_args()

    config = read_yaml(args.config)
    seed = int(config.get("seed", 0) if args.seed is None else args.seed)
    run_dir, metrics = train_baseline(config, seed=seed, smoke=args.smoke)

    print(f"run_dir={run_dir}")
    print(json.dumps(metrics["final_eval"], indent=2))


if __name__ == "__main__":
    main()
