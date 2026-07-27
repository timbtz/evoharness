# Gaussian Spectral Noise Perturbation
Applying additive Gaussian noise to the Fourier modes of a contracted boundary fails to escape local optima and consistently degrades physics/QI balance compared to the deterministic depth-contraction incumbent.
## How it was tried
- `stellar_p2-s105-26196944` c0010 (REJ, train 0.6139, val 0.6277): Generated up to 6 randomized candidates by applying `rng.normal(0.0, noise_scale)` to `r_cos[1:, :]` and `z_sin[1:, ntor+1:]` of bank seeds, scaling noise proportionally to the average coefficient magnitude. Regressed from the deterministic portfolio score, earning no valid structural novelty.
## Why it failed
The writer predicted that undirected Gaussian noise would robustly sample the vast 10-80D Fourier space and find a physically valid new minimum better than static fractional perturbations. The code did exactly this. However, randomly perturbing mid-to-high modes without respecting the delicate spectral condensation required for Quasi-Isodynamic (QI) balance disrupts the magnetic surface geometry, instantly degrading objective_L and QI margin. Undirected noise cannot target the specific aspect-relief tradeoff that deterministic contraction successfully controls.
## Verdict
refuted — Stop applying undirected randomized perturbations or additive noise to Fourier coefficients. The physics are too sensitive; deterministic, coordinated R/Z-split m-differential contraction remains the only viable search axis.
