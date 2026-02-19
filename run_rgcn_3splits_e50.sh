#!/usr/bin/env bash
set -euo pipefail

# R-GCN 3-split run (structure-aware model).
# Usage:
#   bash run_rgcn_3splits_e50.sh
# Optional overrides:
#   EPOCHS=50 DEVICE=cuda LR=0.005 bash run_rgcn_3splits_e50.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" - <<'PY'
import importlib.util
mods = ["src", "src.data.build_graph", "src.train_rgcn", "src.eval_rgcn"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    print("ERROR: Python module preflight failed. Missing:", missing)
    raise SystemExit(2)
print("Python module preflight passed.")
PY

EPOCHS="${EPOCHS:-50}"
LR="${LR:-0.005}"
WD="${WD:-1.0e-5}"
DEVICE="${DEVICE:-cuda}"
NEG_PP="${NEG_PP:-20}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
NUM_LAYERS="${NUM_LAYERS:-2}"
DROPOUT="${DROPOUT:-0.1}"
SPLIT_SEEDS=(0 1 2)
TRAIN_RATIO=0.8
VAL_RATIO=0.1
TEST_RATIO=0.1

RUN_ROOT="${RUN_ROOT:-runs/rgcn_epoch${EPOCHS}_3splits}"
CFG_ROOT="${CFG_ROOT:-configs/generated}"
mkdir -p "${RUN_ROOT}" "${CFG_ROOT}"

echo "[1/3] Build graph artifacts..."
"${PYTHON_BIN}" -m src.data.build_graph

for SPLIT_SEED in "${SPLIT_SEEDS[@]}"; do
  echo "===================================================="
  echo "[split seed=${SPLIT_SEED}] Build split + leakage check"
  SPLIT_DIR="artifacts/splits_seed${SPLIT_SEED}"
  "${PYTHON_BIN}" -m src.data.split_indication \
    --seed "${SPLIT_SEED}" \
    --output-dir "${SPLIT_DIR}" \
    --train-ratio "${TRAIN_RATIO}" \
    --val-ratio "${VAL_RATIO}" \
    --test-ratio "${TEST_RATIO}"
  "${PYTHON_BIN}" -m src.data.leakage_check --splits-dir "${SPLIT_DIR}"

  CFG_PATH="${CFG_ROOT}/rgcn_seed${SPLIT_SEED}_e${EPOCHS}.yaml"
  cat > "${CFG_PATH}" <<YAML
seed: ${SPLIT_SEED}

data:
  nodes_path: artifacts/nodes.parquet
  edges_path: artifacts/edges.parquet
  train_pairs_path: ${SPLIT_DIR}/train_pairs.parquet
  val_pairs_path: ${SPLIT_DIR}/val_pairs.parquet
  test_pairs_path: ${SPLIT_DIR}/test_pairs.parquet

model:
  hidden_dim: ${HIDDEN_DIM}
  num_layers: ${NUM_LAYERS}
  dropout: ${DROPOUT}

train:
  device: ${DEVICE}
  learning_rate: ${LR}
  weight_decay: ${WD}
  negatives_per_positive: ${NEG_PP}
  unlabeled_negative_weight: 1.0
  max_grad_norm: 5.0
  keep_train_indication_only: true
  epochs: ${EPOCHS}
  eval_every: 1
  early_stopping_patience: 8
  early_stopping_metric: auprc
  early_stopping_mode: max
  early_stopping_min_delta: 0.0
  early_stopping_restore_best: true

eval:
  split: val
  auprc_negatives_per_positive: 20
  rank_batch_size: 256

output:
  runs_dir: ${RUN_ROOT}/split_seed${SPLIT_SEED}
YAML

  echo "[split seed=${SPLIT_SEED}] Train R-GCN"
  TRAIN_LOG="$(mktemp)"
  "${PYTHON_BIN}" -m src.train_rgcn --config "${CFG_PATH}" | tee "${TRAIN_LOG}"
  RUN_DIR="$(awk -F= '/^run_dir=/{print $2}' "${TRAIN_LOG}" | tail -n1 | tr -d '\r')"
  rm -f "${TRAIN_LOG}"

  if [[ -z "${RUN_DIR}" || ! -d "${RUN_DIR}" ]]; then
    echo "ERROR: failed to resolve run_dir for split ${SPLIT_SEED}"
    exit 3
  fi

  echo "[split seed=${SPLIT_SEED}] Eval val"
  "${PYTHON_BIN}" -m src.eval_rgcn --run-dir "${RUN_DIR}" --split val --seed "${SPLIT_SEED}" >/dev/null
  echo "[split seed=${SPLIT_SEED}] Eval test"
  "${PYTHON_BIN}" -m src.eval_rgcn --run-dir "${RUN_DIR}" --split test --seed "${SPLIT_SEED}" >/dev/null
done

echo "Done. Runs saved under: ${RUN_ROOT}"
