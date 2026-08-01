# m-differential structural contraction (the winning recipe)
Depth contraction pushes the boundary toward the aspect ratio wall, directly raising `objective_L`. The mechanism successfully reached official 0.6400 on the B3-lhhhhappy3 nfp=3 escape boundary.
## How it was tried
- `stellar_p2-s207-18745292` c0001 (ACC, train -0.5619): Generated 15 NAE/ellipse seeds with parameters targeting low aspect ratios (7.0, 8.0, 9.0) and varying nfp (2, 3, 4, 5) to establish a pristine independent baseline. Selected the best seed purely by minimizing maximum constraint violation. Successfully avoided crashes and safely accepted the lowest-violation candidate.
- `stellar_p2-s105-26196944` c0034 (ACC, train 0.6269): R/Z-split m-differential structural contraction applied to the B3 nfp=3 escape boundary. Pushed aspect ratio to 10.096 (feas 0.00963), yielding val 0.6378 and official 0.6400.
## Why it worked / failed
In independent states (where public bank seeds are disabled or lack baseline `objective_L`), the physics strictly demand getting under the 0.01 feasibility tolerance first. The `stellar_p2-s207-18745292` attempt proved that probing dynamic seed space by sorting on maximum violation is the correct survival mechanism, though it purely minimizes violation without raising L. On competitive basins, R/Z-split structural contraction remains the only mechanism proven to safely convert feasibility budget into L.
## Verdict
promising — structural contraction works. For isolated states, an outer loop minimizing constraint violation is required before any gradient ascent on `objective_L` can begin.
