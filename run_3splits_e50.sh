#!/usr/bin/env bash
set -euo pipefail

# 一键跑：epoch=50，3个随机划分（split seed = 0/1/2）各自训练一次
# 依赖：python 环境已安装并能运行 src.data.build_graph / src.data.split_indication / src.train_baseline

EPOCHS=50
SPLIT_SEEDS=(0 1 2)
TRAIN_RATIO=0.8
VAL_RATIO=0.1
TEST_RATIO=0.1

RUN_ROOT="runs/epoch${EPOCHS}_3splits"
CFG_ROOT="configs/generated"
mkdir -p "${RUN_ROOT}" "${CFG_ROOT}"

echo "[1/3] Build graph artifacts..."
python -m src.data.build_graph

for SPLIT_SEED in "${SPLIT_SEEDS[@]}"; do
  echo "==============================================="
  echo "[split seed=${SPLIT_SEED}] Generating split..."
  SPLIT_DIR="artifacts/splits_seed${SPLIT_SEED}"
  python -m src.data.split_indication \
    --seed "${SPLIT_SEED}" \
    --output-dir "${SPLIT_DIR}" \
    --train-ratio "${TRAIN_RATIO}" \
    --val-ratio "${VAL_RATIO}" \
    --test-ratio "${TEST_RATIO}"

  echo "[split seed=${SPLIT_SEED}] Leakage check..."
  python -m src.data.leakage_check --splits-dir "${SPLIT_DIR}"

  CFG_PATH="${CFG_ROOT}/baseline_quad_seed${SPLIT_SEED}_e${EPOCHS}.yaml"
  cat > "${CFG_PATH}" <<YAML
seed: ${SPLIT_SEED}

data:
  nodes_path: artifacts/nodes.parquet
  edges_path: artifacts/edges.parquet
  mechanism_cache_dir: artifacts/cache/mechanism_adjacency
  train_pairs_path: ${SPLIT_DIR}/train_pairs.parquet
  val_pairs_path: ${SPLIT_DIR}/val_pairs.parquet
  test_pairs_path: ${SPLIT_DIR}/test_pairs.parquet
  train_quadruples_path: ${SPLIT_DIR}/train_quadruples.parquet
  val_quadruples_path: ${SPLIT_DIR}/val_quadruples.parquet
  test_quadruples_path: ${SPLIT_DIR}/test_quadruples.parquet

train:
  embedding_dim: 64
  learning_rate: 5.0
  weight_decay: 1.0e-6
  batch_size: 256
  negatives_per_positive: 20
  epochs: ${EPOCHS}
  eval_every: 1
  early_stopping_patience: 8
  early_stopping_metric: auprc
  early_stopping_mode: max
  early_stopping_min_delta: 0.0
  early_stopping_restore_best: true
  use_quadruple: true
  protein_only: false
  use_mech_fusion: false
  strict_mech_graph: true
  lambda_protein: 1.0
  lambda_pathway: 1.0
  unlabeled_negative_weight: 1.0
  fusion_scale: 0.1

eval:
  split: val
  auprc_negatives_per_positive: 20
  rank_batch_size: 256

output:
  runs_dir: ${RUN_ROOT}/split_seed${SPLIT_SEED}
YAML

  echo "[split seed=${SPLIT_SEED}] Training..."
  python -m src.train_baseline --config "${CFG_PATH}"
done

echo "Done. Runs saved under: ${RUN_ROOT}"
