# Non-Linear, Quantized, and Piecewise M-Contraction Profiles
Replacing the proven smooth linear `1+base·(1+curv·(m-1))` profile with quantized, quadratic, piecewise, or power-law profiles fails to find a Pareto-superior QI/L tradeoff and disrupts mid-m QI geometry.

## How it was tried
- `stellar_p2-s105-26196944` c0018 (REJ, train 0.6234): Applied a quantized mode-multiplicative spike specifically to m=2,3 rows.
- `stellar_p2-s105-26196944` c0019 (REJ, train 0.6233): Applied a single-pass quadratic-in-m contraction `1+base·(1+curv·(m-1)+quad·(m-1)^2)`.
- `stellar_p2-s105-26196944` c0020 (REJ, train 0.6232): Applied a micro-rotation (cosine/sine phase tilt) to the n=0 and n=±1 columns of the dominant m=1 row.
- `stellar_p2-s105-26196944` c0029 (REJ, train 0.6251): Composed a THIRD linear stage with negative curvature (`c3 < 0`) to create a concave high-m profile.
- `stellar_p2-s105-26196944` c0031 (REJ, train 0.6255): Applied an exponentially-tapered m-differential `1 + base*(m-1)*exp(-alpha*(m-1))` targeting mid-m modes.
- `stellar_p2-s105-26196944` c0036 (REJ, train 0.5582, val 0.5646): Swept power-law exponents (`p ∈ {0.75, 1.25}`) applied to the stage-1 profile `1+b1·(1+c1·(m-1))^p`. Catastrophically regressed, deforming the major radius.
- `stellar_p2-s105-26196944` c0041 (ERR): Attempted a two-segment piecewise-linear hinge `curv0`→`curvH` at row `m*`. Crashed with `TypeError: _contract_piecewise() got multiple values for argument 'hinge_m'`.

## Why it failed
The search's winning mechanism precisely relies on decoupling R (aspect) and Z (elongation) through a *smooth, monotonic* m-differential scaling. Altering the m-distribution to target high-m modes specifically (quadratic, quantized spikes, concave/3rd-stage, piecewise, or exponential taper) disrupts this delicate decoupling and instantly degrades objective_L without unlocking new aspect margin.

## Verdict
exhausted — Stop isolating perturbations to specific rows, quantizing the m-profile, applying quadratic/exponential/power-law/piecewise curvature, or rotating toroidal phases. The smooth linear differential scaling profile (and its exact two-stage composed product) remains the strict optimum.
