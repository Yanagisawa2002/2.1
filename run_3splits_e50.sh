#!/usr/bin/env bash
set -euo pipefail

# One-click run: epoch=50 for split seeds 0/1/2.
# Requirement: runnable Python env with project dependencies.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Ensure local project package resolution is stable on cloud/container setups.
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -f "src/data/build_graph.py" || ! -f "src/train_baseline.py" ]]; then
  echo "ERROR: expected project files are missing under ${SCRIPT_DIR}."
  echo "       Make sure you run this script inside the repo root and that src/data exists."
  exit 2
fi

"${PYTHON_BIN}" - <<'PY'
import importlib.util
import os
import sys

mods = ["src", "src.data", "src.data.build_graph", "src.train_baseline"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    print("ERROR: Python module preflight failed.")
    print("Missing:", missing)
    print("cwd:", os.getcwd())
    print("sys.path[0]:", sys.path[0] if sys.path else "")
    raise SystemExit(2)
print("Python module preflight passed.")
PY

EPOCHS=50
SPLIT_SEEDS=(0 1 2)
TRAIN_RATIO=0.8
VAL_RATIO=0.1
TEST_RATIO=0.1

RUN_ROOT="runs/epoch${EPOCHS}_3splits"
CFG_ROOT="configs/generated"
mkdir -p "${RUN_ROOT}" "${CFG_ROOT}"

echo "[1/3] Build graph artifacts..."
"${PYTHON_BIN}" -m src.data.build_graph

for SPLIT_SEED in "${SPLIT_SEEDS[@]}"; do
  echo "==============================================="
  echo "[split seed=${SPLIT_SEED}] Generating split..."
  SPLIT_DIR="artifacts/splits_seed${SPLIT_SEED}"
  "${PYTHON_BIN}" -m src.data.split_indication \
    --seed "${SPLIT_SEED}" \
    --output-dir "${SPLIT_DIR}" \
    --train-ratio "${TRAIN_RATIO}" \
    --val-ratio "${VAL_RATIO}" \
    --test-ratio "${TEST_RATIO}"

  echo "[split seed=${SPLIT_SEED}] Leakage check..."
  "${PYTHON_BIN}" -m src.data.leakage_check --splits-dir "${SPLIT_DIR}"

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
  "${PYTHON_BIN}" -m src.train_baseline --config "${CFG_PATH}"
done

echo "Done. Runs saved under: ${RUN_ROOT}"
