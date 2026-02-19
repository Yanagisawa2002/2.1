# Project: PrimeKG indication link prediction (transductive) + semi-supervised mechanism quadruples

## Goal
We build a transductive drug–disease link prediction model on a PrimeKG-derived graph.
- Only predict: drug–disease = indication (binary link prediction).
- Node features: NONE (graph structure only). Use learnable embeddings + GNN/KG models.

## Data constraints
- We have many drug–disease pairs (~10k).
- We have few mechanism quadruples (~500): (drug, disease, protein, pathway).
- Missing quadruples are NOT negatives (open-world / PU). Never train with "non-observed quadruple = negative".

## Split & leakage rules (NON-NEGOTIABLE)
- Transductive setting: nodes fixed; split only the indication edges.
- Quadruples MUST follow the (drug, disease) split.
- No leakage: no test/val quadruple or pair may appear in train (by (drug,disease) and by full quadruple).

## Deliverables (Definition of Done for any coding task)
- Implement code + minimal tests.
- Provide a runnable command (or Make target) and a short example invocation.
- After changes: run unit tests (pytest) and a tiny smoke run (1 epoch or a small subset).
- Keep changes minimal and cohesive; do not mix unrelated refactors.

## Preferred workflow
- Use Chat mode to propose a plan + files to change + commands to run.
- Switch to Agent mode to implement.
- Use Git checkpoints: keep diffs reviewable; commit frequently.

## Commands (update these once the repo has them)
- Tests: `pytest -q`
- Build graph: `make build-graph`
- Split: `make split`
- Train baseline: `make train-baseline`
- Train with quadruples: `make train-quad`
- Eval: `make eval`
- Explain: `make explain`
