#!/usr/bin/env bash
# Supervisor for the FD-gradient polish loop.
#
# The 2026-08-06 run died at step 7 with OOMKilled=true (exit 137) — the box
# swaps under the coolify tenants, not a bug in the loop. qi_polish.py writes
# state.json after every step, so a kill costs at most the current step's ~40
# solves. This restarts it until the target step count is reached, the trust
# region collapses, or the failure budget runs out.
#
#   setsid nohup experiments/polish_supervise.sh 30 >> runs/qi_polish/polish.log 2>&1 &
set -u

STEPS="${1:-30}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/runs/qi_polish"
STATE="$OUT/state.json"
MAX_FAILS=25
fails=0

cd "$ROOT" || exit 1
mkdir -p "$OUT"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

steps_done() {
  python3 - "$STATE" <<'PY' 2>/dev/null || echo 0
import json, sys
try:
    t = json.load(open(sys.argv[1]))["trajectory"]
except Exception:
    print(0); raise SystemExit
print(max(r["step"] for r in t))
PY
}

trust_now() {
  python3 - "$STATE" <<'PY' 2>/dev/null || echo 1
import json, sys
try:
    print(json.load(open(sys.argv[1]))["trust"])
except Exception:
    print(1)
PY
}

log "=== polish supervisor start (pid $$) target=$STEPS ==="

while :; do
  done_steps="$(steps_done)"
  if [ "$done_steps" -ge "$STEPS" ]; then
    log "=== target reached: step $done_steps >= $STEPS — done ==="; break
  fi
  if awk "BEGIN{exit !($(trust_now) < 1e-7)}"; then
    log "=== trust region collapsed — done ==="; break
  fi
  if [ "$fails" -ge "$MAX_FAILS" ]; then
    log "=== $fails consecutive container failures — giving up ==="; break
  fi

  docker rm -f evoh-polish >/dev/null 2>&1
  log "--- launching container from step $((done_steps + 1)) (fails=$fails)"
  docker run --rm --name evoh-polish --cpus 2 --memory 5g --memory-swap 6g \
    --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$ROOT:/work" -w /work evoharness-stellar-eval \
    python -u experiments/qi_polish.py --steps "$STEPS" --k 20
  rc=$?
  after="$(steps_done)"
  log "--- container exit=$rc  steps $done_steps -> $after"

  if [ "$after" -gt "$done_steps" ]; then
    fails=0                      # progress was made; a kill after that is fresh
  else
    fails=$((fails + 1))
    sleep 60                     # no progress — back off, the box may be thrashing
  fi
  [ "$rc" -eq 0 ] && [ "$after" -le "$done_steps" ] && { log "=== clean exit with no progress (loop stopped itself) ==="; break; }
done

log "=== polish supervisor exit ==="
