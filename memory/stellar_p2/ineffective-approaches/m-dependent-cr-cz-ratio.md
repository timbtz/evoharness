# Mode-Dependent (Per-Row) `cr`/`cz` R/Z Decoupling and Asymmetric Stage Profiles
Making the R/Z asymmetry split `(cr, cz)` a function of the poloidal mode index `m` (e.g., `cr_m = base * (1 + delta * m^2)`) or shifting it asymmetrically across stages fails to find a Pareto-superior QI/L tradeoff and regresses to the fallback floor.

## How it was tried
- `stellar_p2-s203-38950787` c0005f (ACC, train 0.6128): Tested a mode-specific R/Z decoupling where the `cr/cz` ratio varied by `m`, attempting to let low-`m` modes preserve QI while high-`m` modes relieve aspect ratio. Regressed to the typo-corrupted fallback floor.
- `g2` c0009 (REJ, train 0.5806): Swept independent `cz` variations (0.60-0.80) and fine `cr` micro-shifts off the incumbent. Regressed slightly.
- `g2` c0010 (REJ, train 0.5817): Attempted constant-total-depth `(b1,c1) × s2b` lateral grids and shallow second-stage depth variations off the winner. Regressed slightly.
- `g2` c0013 (ACC, train 0.5841): Swept a nested asymmetric two-stage composition with steep Z-axis-heavy (`cz=1.2`) second-stage profiles and depth variations. Yielded a marginal gain but risks crossing the QI limit if pushed further.

## Why it failed
Writers predicted that decoupling the aspect/elongation tradeoff per-row or across stages would uncover a better operating point. The code applied varying `cr/cz` splits. However, giving each poloidal mode or stage its own `cr/cz` ratio destroys the global, coordinated aspect-relief mechanism that makes the uniform `(cr,cz)=(0.5,0.7)` successful. Localizing the R/Z decoupling disrupts the baseline geometry much more severely than uniform differential scaling.

## Verdict
exhausted — Stop isolating R/Z decoupling perturbations per-row or in asymmetric second stages. The uniform `(cr,cz)=(0.5,0.7)` split remains the strict optimum.
