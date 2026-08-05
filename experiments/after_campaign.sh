#!/usr/bin/env bash
# Watcher: wait for the gradient A/B campaign to finish, then (1) restart the
# API server on the new task.py, (2) run the FD d(qi)/d(boundary) probe.
#
# Why a watcher and not a cron entry: both follow-ups must happen exactly once,
# strictly after the campaign, and the probe needs the box's CPUs to itself
# (the eval containers OOM at ~3g caps when campaign evals coincide). A poll
# loop expresses "after that finishes" directly; cron would have to re-derive it
# every minute and guard against double-starts anyway.
#
# Launch detached:
#   setsid nohup experiments/after_campaign.sh >> runs/after_campaign.log 2>&1 &
#
# Skip a stage if you want to run it by hand:
#   SKIP_SERVER=1 experiments/after_campaign.sh
#   SKIP_PROBE=1  experiments/after_campaign.sh

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$PWD"
LOG() { echo "[$(date '+%F %T')] $*"; }

LOCK="runs/after_campaign.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
    LOG "another watcher holds $LOCK — exiting"; exit 0
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

# ---------------------------------------------------------------- 1. wait ----
LOG "=== watcher start (pid $$) — waiting for grad_campaign to finish ==="
while pgrep -f 'experiments/grad_campaign.py$' > /dev/null; do
    sleep 60
done
LOG "campaign driver has exited"
if [ -f runs/grad_campaign/state.json ]; then
    .venv/bin/python - <<'PY'
import json, pathlib
st = json.loads(pathlib.Path("runs/grad_campaign/state.json").read_text())
print("  halt:", st.get("halt"), "| spent: $%s" % st.get("usd"))
for name, r in st.get("done", {}).items():
    print("  %-8s arm=%-3s train=%-8s private=%-8s usd=%-6s stop=%s"
          % (name, r.get("arm"), r.get("train"), r.get("private"),
             r.get("usd"), str(r.get("stop_reason"))[:48]))
PY
fi

# ------------------------------------------------------- 2. server restart ----
if [ "${SKIP_SERVER:-0}" != "1" ]; then
    LOG "--- restarting the API server on the new task.py"
    OLD=$(pgrep -f 'uvicorn api.server:app' | head -1)
    if [ -n "$OLD" ]; then
        LOG "stopping old server (pid $OLD)"
        kill "$OLD" 2>/dev/null
        for _ in $(seq 1 20); do kill -0 "$OLD" 2>/dev/null || break; sleep 1; done
        kill -0 "$OLD" 2>/dev/null && { LOG "server did not stop; leaving it alone"; }
    fi
    if ! pgrep -f 'uvicorn api.server:app' > /dev/null; then
        setsid nohup ./serve.sh >> server.log 2>&1 < /dev/null &
        for _ in $(seq 1 30); do
            curl -sf http://127.0.0.1:8777/ > /dev/null 2>&1 && break
            sleep 2
        done
        if curl -sf http://127.0.0.1:8777/ > /dev/null 2>&1; then
            LOG "server up on :8777 (pid $(pgrep -f 'uvicorn api.server:app' | head -1))"
        else
            LOG "WARNING: server did not answer on :8777 — see server.log"
        fi
    fi
else
    LOG "--- server restart skipped (SKIP_SERVER=1)"
fi

# ------------------------------------------------------------- 3. FD probe ----
if [ "${SKIP_PROBE:-0}" != "1" ]; then
    LOG "--- FD d(qi)/d(boundary) probe (pinned image, ~2 solves per coefficient)"
    mkdir -p runs/fd_probe
    docker run --rm --name evoh-fdprobe \
        --cpus 2 --memory 5g --memory-swap 5g \
        --user "$(id -u):$(id -g)" -e HOME=/tmp \
        -v "$REPO":/work -w /work evoharness-stellar-eval \
        python experiments/fd_qi_probe.py \
            --cases champion,davidkh --k 20 --n-check 5 \
        >> runs/fd_probe/probe.log 2>&1
    LOG "probe exit=$? — results in runs/fd_probe/ (summary.json)"
    tail -20 runs/fd_probe/probe.log
else
    LOG "--- FD probe skipped (SKIP_PROBE=1)"
fi

LOG "=== watcher done ==="
