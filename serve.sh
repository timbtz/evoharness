#!/usr/bin/env bash
# Canonical API-server launcher. ALWAYS start the server with this script:
# a bare `uvicorn api.server:app` misses STELLAR_TRAIN_OVERRIDES and silently
# evaluates stellar_p2 train candidates at 72 evals/240s instead of the
# campaign's 160/480 (cost us runs s107, 2026-07-25).
#   ./serve.sh            foreground
#   setsid nohup ./serve.sh >> server.log 2>&1 &
cd "$(dirname "$0")"
# eval_timeout 60->120->180: host CPU contention (coolify apps share the 4
# vCPUs) pushed 12-27s bank-seed evals past 60s, silently -inf-ing them (runs
# s105 + s103-89260631); 120s+15 still got blown by a single high-mode re-score
# under load (s103-92006336). 180s still bounds true VMEC hangs.
export STELLAR_TRAIN_OVERRIDES='{"max_evals":160,"cpu_budget":480.0,"eval_timeout":180.0}'
# Campaign 2 (2026-07-27): train/val fitness = p2 - 0.92*max(0, feas - 0.002), and
# the novelty ramp is widened past the 1e-3 export bar to 3e-3 — campaign 1's
# champion cleared 1e-3 at bank_dist 2.6e-3 while being cosine 0.999989 to the
# public #1 boundary, so the old ramp switched off exactly where the near-copies
# live. Export guard itself is unchanged (hard 1e-3); this only shapes the search.
export STELLAR_MARGIN='{"target":0.002,"slope":0.92}'
export STELLAR_NOVELTY='{"min":0.003,"pen":0.05}'
exec .venv/bin/uvicorn api.server:app --host 0.0.0.0 --port 8777
