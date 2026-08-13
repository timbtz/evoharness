# g3 run: Contraction Sweeps on the Isolated Fabricated B3 Matrix
In run `g3` (c0000 train 0.5843), various modifications to the R/Z-split m-differential structural contraction pipeline were applied to a structurally simplified/fabricated minimal `B3-lhhhhappy3` matrix. Because the base matrix fundamentally lacks the `objective_L` of the authentic public seed lineage, all local parameter sweeps and structural variations cap at the 0.5843 floor or regress.

## How it was tried
- `g3` c0002 (REJ, train 0.5645): Attempted to restore the hardcoded matrix by shifting the m=1 mode from n=-1 to n=1, fundamentally corrupting the spectrally-condensed geometry.
- `g3` c0005 (REJ, train 0.5822): Replaced the multi-seed portfolio with a three-ladder warm-ascending pipeline exploiting the warm-eval 25% speedup. Failed to find any Pareto improvements, strictly confirming that sequential warm-ladders burn budget without yielding batched static grid gains.
- `g3` c0007 (REJ, train 0.5754): Slashed seed-evaluation overhead by aspect-prescreening dynamic bank seeds to concentrate budget on the B3 contraction grid. Regressed because the isolated simplified B3 matrix caps the physics.
- `g3` c0014 (REJ, train 0.5811): Reclaimed seed-eval budget to spend on a denser three-stage cr-split sweep. Regressed as the structural ceiling is the base matrix, not the parameter grid density.
- `g3` c0019 (REJ, train 0.5853): Replaced scattered phases with a single dense three-stage composed contraction pipeline. A marginal +0.001 train gain that is within noise, confirming the local search space is entirely saturated.
- `g3` c0013 (REJ, train 0.5646): Replaced the contraction grid with a solver gradient (`metric_grad`) pipeline targeting objective_L at fixed aspect margin. Regressed, confirming gradients cannot bridge the missing structural physics.

## Why it failed
The writers predicted that concentrating budget, sweeping grid parameters densely, or exploiting warm-ladder speedups would map a better Pareto frontier. The code applied these exact mechanisms. However, all contraction mechanics are bottlenecked by the baseline matrix. The simplified hardcoded matrix used in `g3` completely lacks the spectral richness of the authentic lhhhhappy3 bank seed (which evaluates at 0.6257 officially). Parameter sweeps cannot manufacture missing QI structural quality.

## Verdict
exhausted — Stop optimizing local contraction parameters on fabricated matrices in isolated states. The binding constraint is the authentic elite matrix; without it, the run is strictly capped at ~0.584 train.
