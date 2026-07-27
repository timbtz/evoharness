# Post-Composition Low-m to High-m Curvature Transfer
Applying a zero-sum `t`-transfer that redistributes contraction depth from low-m (`m<=1`) to high-m (`m>=2`) rows after the proven two-stage contraction fails to improve the QI/aspect tradeoff and silently ties or regresses.

## How it was tried
- `stellar_p2-s105-26196944` c0045 (ACC, train 0.6268, val 0.6379): Swept transfer magnitudes `t ∈ {-2,-1,0,1,2}e-3`. Accepted but tied at the neutral `t=0` point, proving the function evaluated correctly but yielded no gain over the incumbent.

## Why it failed
The writer predicted that aspect load could be offloaded to high-m modes without breaching QI. The code applied `+t` to high-m and `-t*kappa` to low-m rows. The neutral `t=0` point won, confirming the two-stage profile already perfectly balances the aspect relief and QI constraints. Redistributing it explicitly destroys the optimal coordinated scaling.

## Verdict
refuted — Stop artificially redistributing depth across m-rows post-composition. The two-stage profile inherently provides the optimal high-m balance.
