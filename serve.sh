#!/usr/bin/env bash
# Canonical API-server launcher. ALWAYS start the server with this script:
# a bare `uvicorn api.server:app` misses STELLAR_TRAIN_OVERRIDES and silently
# evaluates stellar_p2 train candidates at 72 evals/240s instead of the
# campaign's 160/480 (cost us runs s107, 2026-07-25).
#   ./serve.sh            foreground
#   setsid nohup ./serve.sh >> server.log 2>&1 &
cd "$(dirname "$0")"
# eval_timeout 60->120: host CPU contention (coolify apps share the 4 vCPUs)
# pushed 12-27s bank-seed evals past 60s, silently -inf-ing them (runs s105 +
# s103-89260631, 2026-07-25); 120s still bounds true VMEC hangs
export STELLAR_TRAIN_OVERRIDES='{"max_evals":160,"cpu_budget":480.0,"eval_timeout":120.0}'
exec .venv/bin/uvicorn api.server:app --host 0.0.0.0 --port 8777
