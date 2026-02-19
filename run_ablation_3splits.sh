#!/usr/bin/env bash
set -euo pipefail

# Full evaluation bundle:
# 1) train + eval(test) on each split seed
# 2) aggregate 3-seed mean/std
# 3) ablation: with vs without quadruple supervision
#
# Usage:
#   bash run_ablation_3splits.sh
# Optional env overrides:
#   EPOCHS=50 LR_WITH_QUAD=5.0 LR_NO_QUAD=5.0 RUN_ROOT=runs/ablation_e50_3splits bash run_ablation_3splits.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -f "src/data/build_graph.py" || ! -f "src/train_baseline.py" || ! -f "src/eval.py" ]]; then
  echo "ERROR: expected project files are missing under ${SCRIPT_DIR}."
  exit 2
fi

"${PYTHON_BIN}" - <<'PY'
import importlib.util
mods = ["src", "src.data", "src.data.build_graph", "src.train_baseline", "src.eval"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    print("ERROR: Python module preflight failed. Missing:", missing)
    raise SystemExit(2)
print("Python module preflight passed.")
PY

EPOCHS="${EPOCHS:-50}"
LR_WITH_QUAD="${LR_WITH_QUAD:-5.0}"
LR_NO_QUAD="${LR_NO_QUAD:-5.0}"
NEG_PER_POS="${NEG_PER_POS:-20}"
BATCH_SIZE="${BATCH_SIZE:-256}"
PATIENCE="${PATIENCE:-8}"
SPLIT_SEEDS=(0 1 2)
TRAIN_RATIO=0.8
VAL_RATIO=0.1
TEST_RATIO=0.1

RUN_ROOT="${RUN_ROOT:-runs/ablation_e${EPOCHS}_3splits}"
CFG_ROOT="${CFG_ROOT:-configs/generated}"
mkdir -p "${RUN_ROOT}" "${CFG_ROOT}"

MANIFEST="${RUN_ROOT}/run_manifest.tsv"
printf "mode\tseed\trun_dir\tconfig_path\n" > "${MANIFEST}"

echo "[1/4] Build graph artifacts..."
"${PYTHON_BIN}" -m src.data.build_graph

for SPLIT_SEED in "${SPLIT_SEEDS[@]}"; do
  echo "=========================================================="
  echo "[split seed=${SPLIT_SEED}] Build split + leakage check"
  SPLIT_DIR="artifacts/splits_seed${SPLIT_SEED}"

  "${PYTHON_BIN}" -m src.data.split_indication \
    --seed "${SPLIT_SEED}" \
    --output-dir "${SPLIT_DIR}" \
    --train-ratio "${TRAIN_RATIO}" \
    --val-ratio "${VAL_RATIO}" \
    --test-ratio "${TEST_RATIO}"

  "${PYTHON_BIN}" -m src.data.leakage_check --splits-dir "${SPLIT_DIR}"

  for MODE in with_quad no_quad; do
    if [[ "${MODE}" == "with_quad" ]]; then
      LR="${LR_WITH_QUAD}"
      USE_QUAD="true"
      STRICT_MECH="true"
    else
      LR="${LR_NO_QUAD}"
      USE_QUAD="false"
      STRICT_MECH="false"
    fi

    CFG_PATH="${CFG_ROOT}/ablation_${MODE}_seed${SPLIT_SEED}_e${EPOCHS}.yaml"
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
  learning_rate: ${LR}
  weight_decay: 1.0e-6
  batch_size: ${BATCH_SIZE}
  negatives_per_positive: ${NEG_PER_POS}
  epochs: ${EPOCHS}
  eval_every: 1
  early_stopping_patience: ${PATIENCE}
  early_stopping_metric: auprc
  early_stopping_mode: max
  early_stopping_min_delta: 0.0
  early_stopping_restore_best: true
  use_quadruple: ${USE_QUAD}
  protein_only: false
  use_mech_fusion: false
  strict_mech_graph: ${STRICT_MECH}
  lambda_protein: 1.0
  lambda_pathway: 1.0
  unlabeled_negative_weight: 1.0
  fusion_scale: 0.1

eval:
  split: val
  auprc_negatives_per_positive: 20
  rank_batch_size: 256

output:
  runs_dir: ${RUN_ROOT}/${MODE}/split_seed${SPLIT_SEED}
YAML

    echo "[split seed=${SPLIT_SEED}] [${MODE}] train"
    TRAIN_LOG="$(mktemp)"
    "${PYTHON_BIN}" -m src.train_baseline --config "${CFG_PATH}" | tee "${TRAIN_LOG}"
    RUN_DIR="$(awk -F= '/^run_dir=/{print $2}' "${TRAIN_LOG}" | tail -n1 | tr -d '\r')"
    rm -f "${TRAIN_LOG}"

    if [[ -z "${RUN_DIR}" || ! -d "${RUN_DIR}" ]]; then
      echo "ERROR: failed to resolve run_dir for split=${SPLIT_SEED}, mode=${MODE}"
      exit 3
    fi

    echo "[split seed=${SPLIT_SEED}] [${MODE}] eval val"
    "${PYTHON_BIN}" -m src.eval --run-dir "${RUN_DIR}" --split val --seed "${SPLIT_SEED}" >/dev/null

    echo "[split seed=${SPLIT_SEED}] [${MODE}] eval test"
    "${PYTHON_BIN}" -m src.eval --run-dir "${RUN_DIR}" --split test --seed "${SPLIT_SEED}" >/dev/null

    printf "%s\t%s\t%s\t%s\n" "${MODE}" "${SPLIT_SEED}" "${RUN_DIR}" "${CFG_PATH}" >> "${MANIFEST}"
  done
done

echo "[4/4] Aggregate 3-seed metrics"
RUN_ROOT_ENV="${RUN_ROOT}" MANIFEST_ENV="${MANIFEST}" "${PYTHON_BIN}" - <<'PY'
import json
import math
import os
from pathlib import Path


def mean_std(values):
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    n = len(values)
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / n
    return {"mean": mean, "std": math.sqrt(var), "n": n}


run_root = Path(os.environ["RUN_ROOT_ENV"])
manifest = Path(os.environ["MANIFEST_ENV"])

rows = []
for line in manifest.read_text(encoding="utf-8").strip().splitlines()[1:]:
    mode, seed, run_dir, cfg = line.split("\t")
    rows.append({"mode": mode, "seed": int(seed), "run_dir": run_dir, "config": cfg})

by_mode = {}
for r in rows:
    mode = r["mode"]
    run_dir = Path(r["run_dir"])
    val_payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    val_metrics = val_payload["final_eval"]

    test_files = sorted(run_dir.glob("metrics_test_*.json"), key=lambda p: p.stat().st_mtime_ns)
    if not test_files:
        raise RuntimeError(f"No test metrics found in {run_dir}")
    test_metrics = json.loads(test_files[-1].read_text(encoding="utf-8"))

    if mode not in by_mode:
        by_mode[mode] = {"val": [], "test": []}
    by_mode[mode]["val"].append(val_metrics)
    by_mode[mode]["test"].append(test_metrics)

metric_keys = ["auprc", "mrr", "hits@1", "hits@3", "hits@10"]
summary = {"per_mode": {}, "manifest": str(manifest)}
for mode, payload in by_mode.items():
    summary["per_mode"][mode] = {"val": {}, "test": {}}
    for split in ["val", "test"]:
        for key in metric_keys:
            vals = [float(x[key]) for x in payload[split]]
            summary["per_mode"][mode][split][key] = mean_std(vals)

summary_path = run_root / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("Summary written:", summary_path)
for mode in sorted(summary["per_mode"].keys()):
    print(f"[{mode}]")
    for split in ["val", "test"]:
        m = summary["per_mode"][mode][split]
        print(
            f"  {split}: "
            f"AUPRC {m['auprc']['mean']:.4f}±{m['auprc']['std']:.4f}, "
            f"MRR {m['mrr']['mean']:.4f}±{m['mrr']['std']:.4f}, "
            f"H@10 {m['hits@10']['mean']:.4f}±{m['hits@10']['std']:.4f}"
        )
PY

echo "Done. Manifest: ${MANIFEST}"
