# Mode-Dependent (Per-Row) `cr`/`cz` R/Z Decoupling
Making the R/Z asymmetry split `(cr, cz)` a function of the poloidal mode index `m` (e.g., `cr_m = base * (1 + delta * m^2)`) fails to find a Pareto-superior QI/L tradeoff and regresses to the fallback floor.

## How it was tried
- `stellar_p2-s203-38950787` c0005f (ACC, train 0.6128): Tested a mode-specific R/Z decoupling where the `cr/cz` ratio varied by `m`, attempting to let low-`m` modes preserve QI while high-`m` modes relieved aspect ratio. Regressed to the typo-corrupted fallback floor.

## Why it failed
The writer predicted that decoupling the aspect/elongation tradeoff per-row would uncover a better operating point. The code applied a varying `cr/cz` split across the rows. However, giving each poloidal mode its own `cr/cz` ratio destroys the global, coordinated aspect-relief mechanism that makes the uniform `(cr,cz)=(0.5,0.7)` successful. Localizing the R/Z decoupling disrupts the baseline geometry much more severely than uniform differential scaling.

## Verdict
exhausted — Stop isolating R/Z decoupling perturbations per-row. The uniform `(cr,cz)=(0.5,0.7)` split remains the strict optimum.
