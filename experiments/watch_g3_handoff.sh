#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$ROOT/runs/composite_polish_campaign/handoff.log"
LEDGER="$ROOT/runs/gradrun/g3/ledger.jsonl"
DEADLINE=$(( $(date +%s) + 900 ))

note() { printf '[%s] %s\n' "$(date -u '+%Y-%m-%d %H:%M:%S')" "$*" >> "$LOG"; }

note "handoff watcher started (15 minute bound)"
while (( $(date +%s) < DEADLINE )); do
    if tail -1 "$LEDGER" 2>/dev/null | grep -q '"type": "run_end"'; then
        note "g3 run_end present; waiting up to 2 minutes for wrappers"
        for _ in $(seq 1 24); do
            pgrep -f 'gradrun.py --exec-run --name g3' >/dev/null || break
            sleep 5
        done
        pid="$(pgrep -f 'gradrun.py --exec-run --name g3' | head -1 || true)"
        if [[ -n "$pid" ]]; then
            note "completed g3 wrapper still alive; sending TERM to pid $pid"
            kill -TERM "$pid" 2>/dev/null || true
        fi
        note "handoff complete; composite supervisor will start on its next poll"
        exit 0
    fi
    sleep 15
done

note "deadline reached without run_end; stopping stale Stellar evaluator containers"
docker ps --filter ancestor=evoharness-stellar-eval --format '{{.Names}}' |
while read -r name; do
    [[ -n "$name" ]] && docker stop -t 10 "$name" >> "$LOG" 2>&1 || true
done
exit 1
