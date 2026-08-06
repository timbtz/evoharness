#!/usr/bin/env bash
# Phase 2 of the unattended chain: once after_campaign.sh has finished (server
# restarted, FD qi probe done), run the campaign driver again so it picks up the
# deep matched pairs p3/p4 and push2 that were added after the first driver had
# already loaded its STEPS list.
#
# This exists so the box keeps doing useful work even if no notification ever
# reaches the operator: everything here is derived from state on disk, restartable,
# and halts on provider death rather than a dollar cap.
#
# Launch detached:
#   setsid nohup experiments/phase2_chain.sh >> runs/phase2_chain.log 2>&1 &

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
LOG() { echo "[$(date '+%F %T')] $*"; }

LOCK="runs/phase2.lock"
if ! mkdir "$LOCK" 2>/dev/null; then LOG "another phase2 holds $LOCK"; exit 0; fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

LOG "=== phase2 start (pid $$) — waiting for the first driver and the probe ==="
while pgrep -f '[g]rad_campaign\.py' > /dev/null \
   || pgrep -f '[a]fter_campaign\.sh' > /dev/null; do
    sleep 120
done
LOG "phase 1 complete"

# Do not start a $65 phase on a dead provider.
if [ -f runs/grad_campaign/state.json ]; then
    HALT=$(.venv/bin/python -c "
import json;print(json.load(open('runs/grad_campaign/state.json')).get('halt') or '')" 2>/dev/null)
    if [ -n "$HALT" ]; then
        LOG "campaign is halted ($HALT) — not starting the deep pairs"
        exit 0
    fi
fi

LOG "--- deep pairs: re-running the driver (skips completed steps)"
.venv/bin/python experiments/grad_campaign.py --global-usd 150 \
    >> runs/grad_campaign/driver.log 2>&1
LOG "driver exit=$?"
.venv/bin/python experiments/grad_campaign.py --status | head -60
LOG "=== phase2 done ==="
