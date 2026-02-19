from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.baseline_common import load_node_index, load_quadruple_splits, save_checkpoint
from src.eval import evaluate_from_checkpoint
from src.explain import explain_pair
from src.train_baseline import _run_train_step, train_baseline


def _write_pairs(path: Path, rows: list[tuple[str, str]]) -> None:
    table = pa.table(
        {
            "drug_id": [r[0] for r in rows],
            "disease_id": [r[1] for r in rows],
        }
    )
    pq.write_table(table, path)


def _write_nodes(path: Path) -> None:
    table = pa.table(
        {
            "node_id": [0, 1, 2, 3, 4],
            "type": ["drug", "disease", "disease", "gene/protein", "pathway"],
            "original_id": [
                "drug::D1",
                "disease::X1",
                "disease::X2",
                "gene/protein::P1",
                "pathway::PW1",
            ],
        }
    )
    pq.write_table(table, path)


def _minimal_config(root: Path) -> dict:
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    splits = artifacts / "splits"
    splits.mkdir(parents=True, exist_ok=True)

    nodes_path = artifacts / "nodes.parquet"
    edges_path = artifacts / "edges.parquet"
    train_pairs = splits / "train_pairs.parquet"
    val_pairs = splits / "val_pairs.parquet"
    test_pairs = splits / "test_pairs.parquet"
    train_quads = splits / "train_quadruples.parquet"
    val_quads = splits / "val_quadruples.parquet"
    test_quads = splits / "test_quadruples.parquet"

    _write_nodes(nodes_path)
    _write_pairs(train_pairs, [("drug::D1", "disease::X1")])
    _write_pairs(val_pairs, [("drug::D1", "disease::X2")])
    _write_pairs(test_pairs, [("drug::D1", "disease::X2")])
    pq.write_table(
        pa.table(
            {
                "drug_id": ["drug::D1"],
                "disease_id": ["disease::X1"],
                "protein_id": ["gene/protein::P1"],
                "pathway_id": ["pathway::PW1"],
            }
        ),
        train_quads,
    )
    pq.write_table(
        pa.table(
            {
                "drug_id": [],
                "disease_id": [],
                "protein_id": [],
                "pathway_id": [],
            }
        ),
        val_quads,
    )
    pq.write_table(
        pa.table(
            {
                "drug_id": [],
                "disease_id": [],
                "protein_id": [],
                "pathway_id": [],
            }
        ),
        test_quads,
    )
    pq.write_table(
        pa.table(
            {
                "src_id": [],
                "dst_id": [],
                "rel": [],
            }
        ),
        edges_path,
    )

    return {
        "seed": 0,
        "data": {
            "nodes_path": str(nodes_path),
            "edges_path": str(edges_path),
            "mechanism_cache_dir": str(artifacts / "cache"),
            "train_pairs_path": str(train_pairs),
            "val_pairs_path": str(val_pairs),
            "test_pairs_path": str(test_pairs),
            "train_quadruples_path": str(train_quads),
            "val_quadruples_path": str(val_quads),
            "test_quadruples_path": str(test_quads),
        },
        "train": {
            "embedding_dim": 4,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "batch_size": 1,
            "negatives_per_positive": 1,
            "epochs": 1,
            "eval_every": 1,
            "use_quadruple": False,
            "protein_only": False,
            "use_mech_fusion": True,
            "strict_mech_graph": False,
            "lambda_protein": 1.0,
            "lambda_pathway": 1.0,
            "unlabeled_negative_weight": 1.0,
            "fusion_scale": 0.1,
        },
        "eval": {
            "split": "val",
            "auprc_negatives_per_positive": 1,
        },
        "output": {
            "runs_dir": str(root / "runs"),
        },
    }


def test_eval_requires_mechanism_parameters_when_fusion_enabled(tmp_path: Path) -> None:
    config = _minimal_config(tmp_path)
    ckpt_path = tmp_path / "checkpoint.npz"
    save_checkpoint(
        checkpoint_path=ckpt_path,
        node_embeddings=np.random.default_rng(0).normal(0, 0.01, size=(3, 4)).astype(np.float32),
        relation_embedding=np.random.default_rng(1).normal(0, 0.01, size=(4,)).astype(np.float32),
        seed=0,
    )

    with pytest.raises(ValueError, match="mechanism parameters"):
        evaluate_from_checkpoint(
            config=config,
            checkpoint_path=ckpt_path,
            split="val",
            seed=0,
        )


def test_explain_requires_mechanism_parameters(tmp_path: Path) -> None:
    config = _minimal_config(tmp_path)
    ckpt_path = tmp_path / "checkpoint.npz"
    save_checkpoint(
        checkpoint_path=ckpt_path,
        node_embeddings=np.random.default_rng(2).normal(0, 0.01, size=(3, 4)).astype(np.float32),
        relation_embedding=np.random.default_rng(3).normal(0, 0.01, size=(4,)).astype(np.float32),
        seed=0,
    )

    with pytest.raises(ValueError, match="mechanism-head parameters"):
        explain_pair(
            config=config,
            checkpoint_path=ckpt_path,
            drug_id="drug::D1",
            disease_id="disease::X1",
            top_k=3,
        )


def test_train_rejects_fusion_without_quadruple_training(tmp_path: Path) -> None:
    config = _minimal_config(tmp_path)
    with pytest.raises(ValueError, match="requires use_quadruple=true"):
        train_baseline(config, seed=0, smoke=True)


def test_train_rejects_empty_train_quadruples_when_enabled(tmp_path: Path) -> None:
    config = _minimal_config(tmp_path)
    pq.write_table(
        pa.table(
            {
                "drug_id": [],
                "disease_id": [],
                "protein_id": [],
                "pathway_id": [],
            }
        ),
        Path(config["data"]["train_quadruples_path"]),
    )
    config["train"]["use_quadruple"] = True
    config["train"]["use_mech_fusion"] = False

    with pytest.raises(ValueError, match="train quadruples are empty"):
        train_baseline(config, seed=0, smoke=True)


def test_load_quadruples_rejects_invalid_pathway_id_format(tmp_path: Path) -> None:
    config = _minimal_config(tmp_path)
    pq.write_table(
        pa.table(
            {
                "drug_id": ["drug::D1"],
                "disease_id": ["disease::X1"],
                "protein_id": ["gene/protein::P1"],
                "pathway_id": ["invalid_pathway_format"],
            }
        ),
        Path(config["data"]["train_quadruples_path"]),
    )

    node_index = load_node_index(config["data"]["nodes_path"])
    with pytest.raises(ValueError, match="invalid pathway_id format"):
        load_quadruple_splits(
            node_index=node_index,
            train_quadruples_path=config["data"]["train_quadruples_path"],
            val_quadruples_path=config["data"]["val_quadruples_path"],
            test_quadruples_path=config["data"]["test_quadruples_path"],
        )


def test_train_uses_each_train_quadruple_at_most_once_per_epoch(tmp_path: Path) -> None:
    config = _minimal_config(tmp_path)
    _write_pairs(
        Path(config["data"]["train_pairs_path"]),
        [
            ("drug::D1", "disease::X1"),
            ("drug::D1", "disease::X1"),
            ("drug::D1", "disease::X1"),
        ],
    )
    config["train"]["batch_size"] = 1
    config["train"]["epochs"] = 1
    config["train"]["use_quadruple"] = True
    config["train"]["use_mech_fusion"] = False
    config["eval"]["split"] = "val"
    _write_pairs(Path(config["data"]["val_pairs_path"]), [("drug::D1", "disease::X1")])

    _, metrics = train_baseline(config, seed=0, smoke=False)
    first_epoch = metrics["epoch_logs"][0]
    assert float(first_epoch["train_quadruple_rows_used"]) == 1.0
    assert float(first_epoch["train_protein_supervised"]) == 1.0
    assert float(first_epoch["train_pathway_supervised"]) == 1.0


def test_run_train_step_supports_positive_only_indication_when_unlabeled_weight_zero() -> None:
    node_embeddings = np.random.default_rng(0).normal(0.0, 0.01, size=(3, 4)).astype(np.float32)
    relation_embedding = np.random.default_rng(1).normal(0.0, 0.01, size=(4,)).astype(np.float32)
    batch_pairs = np.array([[0, 1], [0, 2]], dtype=np.int64)

    out = _run_train_step(
        node_embeddings=node_embeddings,
        relation_embedding=relation_embedding,
        batch_pairs=batch_pairs,
        disease_candidates=np.array([1, 2], dtype=np.int64),
        positive_tails_by_head={0: {1, 2}},
        negatives_per_positive=2,
        learning_rate=0.01,
        weight_decay=0.0,
        rng=np.random.default_rng(2),
        mechanism_head=None,
        use_quadruple=False,
        protein_only=False,
        lambda_protein=1.0,
        lambda_pathway=1.0,
        unlabeled_negative_weight=0.0,
        quad_batch=None,
    )

    assert float(out["loss_indication_unlabeled_neg"]) == 0.0
    assert float(out["loss_indication"]) > 0.0


def test_early_stopping_stops_when_metric_does_not_improve(tmp_path: Path) -> None:
    config = _minimal_config(tmp_path)
    _write_pairs(
        Path(config["data"]["train_pairs_path"]),
        [
            ("drug::D1", "disease::X1"),
            ("drug::D1", "disease::X1"),
            ("drug::D1", "disease::X1"),
        ],
    )
    _write_pairs(Path(config["data"]["val_pairs_path"]), [("drug::D1", "disease::X1")])
    config["train"]["use_quadruple"] = False
    config["train"]["use_mech_fusion"] = False
    config["train"]["learning_rate"] = 0.0
    config["train"]["epochs"] = 6
    config["train"]["eval_every"] = 1
    config["train"]["early_stopping_patience"] = 1
    config["train"]["early_stopping_metric"] = "auprc"
    config["train"]["early_stopping_mode"] = "max"
    config["train"]["early_stopping_min_delta"] = 0.0
    config["train"]["early_stopping_restore_best"] = True

    _, metrics = train_baseline(config, seed=0, smoke=False)
    assert metrics["early_stopping"]["enabled"] is True
    assert metrics["early_stopping"]["stopped_early"] is True
    assert int(metrics["epochs_completed"]) < int(config["train"]["epochs"])


def test_run_train_step_rejects_non_finite_scores() -> None:
    node_embeddings = np.full((3, 4), np.inf, dtype=np.float32)
    relation_embedding = np.ones((4,), dtype=np.float32)
    batch_pairs = np.array([[0, 1]], dtype=np.int64)

    with pytest.raises(ValueError, match="Non-finite positive indication scores"):
        _run_train_step(
            node_embeddings=node_embeddings,
            relation_embedding=relation_embedding,
            batch_pairs=batch_pairs,
            disease_candidates=np.array([1, 2], dtype=np.int64),
            positive_tails_by_head={0: {1}},
            negatives_per_positive=1,
            learning_rate=0.01,
            weight_decay=0.0,
            rng=np.random.default_rng(0),
            mechanism_head=None,
            use_quadruple=False,
            protein_only=False,
            lambda_protein=1.0,
            lambda_pathway=1.0,
            unlabeled_negative_weight=1.0,
            quad_batch=None,
        )
