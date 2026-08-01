# Index-Based Symmetry Zeroing Bug
Zeroing Fourier matrix symmetry entries by hardcoded column indices (`rc_new[0, :ntor] = 0.0`) instead of by the seed's native zero-pattern mask corrupts valid m=0 structures, triggering either `RuntimeError: r_cos for m=0 and n<0 must be 0.0` or unintended geometric collapses that cap the train score.

## How it was tried
- `stellar_p2-s204-63425638` c0011 (ACC, train 0.5835): Replaced the index-based left-zeroing logic in `_contract` with a boolean mask (`r_mask = np.abs(rc) > 1e-16`) derived from the seed's original zero-pattern. Combined with the proven c0034 stage-1/stage-2 parameters. It successfully eliminated the `RuntimeError` crash but capped the run at train 0.5835 / val 0.5968.

## Why it failed
Without the authentic hardcoded B3-lhhhhappy3 escape boundary, the run was forced to rely on dynamic `fm.seed_bank()` seeds. The index-based zeroing assumed a standard `n` ordering (`[n<0..., 0, n>0...]`). When this failed, the mask fix correctly preserved structural zeroes but could not overcome the inherent lack of baseline `objective_L` in the generic bank seeds. 

## Verdict
recurring pitfall — If manipulating matrix symmetries dynamically, ALWAYS use a boolean zero-pattern mask rather than hardcoded column indices. To truly escape the isolated state cap, the exact proven hardcoded matrix from the winner is required.
