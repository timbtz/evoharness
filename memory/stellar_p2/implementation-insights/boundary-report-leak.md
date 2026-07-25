# Boundary-report leak (fixed 2026-07-25) — pre-fix boundary verdicts are suspect

**The bug.** From the stellar_p2 build (2026-07-23) until 2026-07-25, the train
container's final report rebound its `_bd` variable inside the archive loop, so
the `"boundary"` reported to the host was the **collect_top-th (8th) best logged
eval**, not the boundary the candidate actually returned. The reported train
metrics belonged to the true return; the boundary identity did not.

**What that corrupted (all runs s42/s7/s11/s17/s100/s101/s102/s103):**
- val/public/private clean-room scores were computed on the leaked boundary;
- `bank_dist` / `novelty_penalty` / `submittable` were computed on the leaked boundary;
- `best.py`-adjacent boundary exports and resume-time boundary reasoning mixed identities;
- gate decisions therefore combined train evidence of one boundary with val evidence of another.

**Signatures it produced** (do NOT re-derive strategy conclusions from pre-fix data
showing these): bit-identical val ties across "different" candidates; "escapes
collapse at val"; train metrics identical to the seed while bank_dist is non-zero
(B4 c0003: penalty arbitrage on a boundary the candidate never returned).

**What stays trustworthy:** `memory/stellar_p2/archive.jsonl` rows — each pairs a
boundary with its own in-container eval — and official scores as (boundary, score)
pairs for the specific exported boundary (e.g. c0005f's 0.6335 is real for the
boundary we possess; it just isn't necessarily what c0005f's code returned).

**Also fixed the same day:** `bank_dist` (container + host + export guard) is now
**scale-normalized** — each boundary is divided by its own R0 before the max-coeff
diff, because every official metric is dimensionless, so a uniform rescale is
physics-null and previously manufactured up to ~2e-3 of fake novelty distance
(verified: uniform ×1.002 copy of bank #4 = raw dist 2e-3, normalized dist ~1e-17).
Escape distances measured before the fix (e.g. B2 c0006r1 "1.25e-3") must be
re-measured under the normalized metric before being called out-of-ball.
