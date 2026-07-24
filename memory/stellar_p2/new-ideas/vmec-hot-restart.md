# VMEC++ hot restart
VMEC++ supports initializing an equilibrium solve from a previous converged state; neighboring boundaries then converge in a fraction of the iterations.

## Status
- Untested in this harness. The constellaration wrapper (forward_model) does not expose hot restart; exploiting it would need direct vmecpp use inside the candidate — currently a POLICY violation (all physics through fm).
- If ever exposed harness-side (fm.eval(b, warm=True)), local-search optimizers would get maybe 2-5x more evals per CPU budget; mutation steps are exactly the "nearby boundary" case hot restart is built for.

## Verdict
promising but blocked on a harness decision — do not attempt from candidate code; propose fm API changes instead.
