# Persistent C search state / single long C LNS call

Running the C LNS as one continuous call for the full budget (preserving the SA trajectory and working state) instead of multiple ~1.5s slices that restart from `best` was the strongest mechanism-backed idea. It is now refuted: after 10+ attempts, it is entirely score-neutral.

## How it was tried
- Implemented successfully in run cvrp-s5-54146615 (also 3 parse-dead diagnoses c0054/c0037/c0009 in cvrp-s3-45465035): c0002, c0009, c0013, c0016, c0020 all scored -0.0906 (exact baseline tie).
- Variants bundling orchestration changes with the single call failed or regressed:
  - Slightly reserving more time for the overlay (0.35s instead of 0.15s) caused X-n101 time-starvation (c0004 -0.0918).
  - Increasing the overlay reserve heavily to 0.35s dropped X-n101 to the 27825 starvation attractor (c0006 -0.378).
  - Running 2-3 C LNS restarts with different seeds effectively sliced the budget and starved C (c0012 -0.143, c0007 -0.097).
  - Adding the Or-opt polish *before* the C call to give a better starting point (c0020) was an exact tie (-0.0906).
- c0009 (run cvrp-s19-83885116): Added a second short C LNS call from a perturbed best with a different seed after the main call. Train -0.1248. Starved the main SA cooling budget without finding alternate basins.

## Why it failed
- The C kernel's SA cooling schedule adapts seamlessly to the elapsed fraction of the time limit (`tlimit`), meaning late slices already explore cooler regimes efficiently rather than naively "resetting". 
- Preserving the working state (`cand_f`) continuously does not unlock better local optima than repeatedly re-converging from `best` on small train instances (n<=125) where the search saturates anyway.
- Multi-restart variants simply divide the budget, giving the SA schedule less time to cool properly and triggering the 27825 time-starvation signature.

## Verdict
refuted (tried 10+ times, exact ties or regressions). Do not attempt single long C calls, continuous SA trajectories, or multi-start orchestration. The Python orchestration is fully saturated.
