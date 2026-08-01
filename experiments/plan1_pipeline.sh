#!/bin/bash
# Plan-1 validation + campaign pipeline (2026-08-01). Detached driver:
# smoke A/B (warm/soft ON vs OFF), then two deep runs on the new architecture.
# Each step continues even if the previous failed (429/balance-exhaustion must
# not kill the chain); everything logs to runs/plan1-pipeline.log.
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
LOG=runs/plan1-pipeline.log
log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

log "=== pipeline start (pid $$) ==="

log "step 1/4: smoke arm ON"
$PY experiments/plan1_smoke.py --arm on --usd 0.45 >> "$LOG" 2>&1
log "step 1 exit=$?"

log "step 2/4: smoke arm OFF (kill-switched control)"
$PY experiments/plan1_smoke.py --arm off --usd 0.45 >> "$LOG" 2>&1
log "step 2 exit=$?"

export STELLAR_TRAIN_OVERRIDES='{"max_evals":160,"cpu_budget":480.0}'

log "step 3/4: deep run A (resume s204 best, seed 5, \$8)"
$PY experiments/plan1_deep.py --seed 5 \
    --resume "file:runs/stellar_p2-s204-63425638/best.py" --usd 8 >> "$LOG" 2>&1
log "step 3 exit=$?"

log "step 4/4: deep run B (resume s205 best, seed 7, \$8)"
$PY experiments/plan1_deep.py --seed 7 \
    --resume "file:runs/stellar_p2-s205-85293087/best.py" --usd 8 >> "$LOG" 2>&1
log "step 4 exit=$?"

log "=== pipeline done ==="
