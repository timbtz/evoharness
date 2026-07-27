"""Seed optimizer for ConStellaration P2: NAE seed portfolio -> scale-aware
Gaussian local search with per-mode exponential step decay (the official ALM
baseline's preconditioning trick, spectrum scaling 1.5), greedy accept on the
shaped score, sigma adaptation, one mode-inflation on hard stall, restart kick.
Working > clever: every trick here is documented in the memory wiki."""
import numpy as np

DECAY = 1.5      # per-mode damping: step_mn ~ DECAY**-(m+|n|)  (ALM baseline)
SIGMA0 = 0.06    # initial relative step vs minor-radius scale
GROW, SHRINK = 1.4, 0.92
INFLATE_AT, KICK_AT = 14, 26   # consecutive rejects


def _mats(b):
    return np.asarray(b["r_cos"], float), np.asarray(b["z_sin"], float)


def _rebuild(b, rc, zs):
    out = dict(b)
    out["r_cos"], out["z_sin"] = rc.tolist(), zs.tolist()
    return out


def _mutate(b, sigma, rng):
    rc, zs = _mats(b)
    ntor = (rc.shape[1] - 1) // 2
    m = np.arange(rc.shape[0])[:, None]
    n = np.abs(np.arange(rc.shape[1])[None, :] - ntor)
    damp = DECAY ** -(m + n)
    ref = max(abs(rc[1:]).max() if rc.shape[0] > 1 else 0.1, abs(zs).max(), 0.05)
    rc = rc + sigma * ref * damp * rng.standard_normal(rc.shape)
    zs = zs + sigma * ref * damp * rng.standard_normal(zs.shape)
    rc[0, ntor] = _mats(b)[0][0, ntor]   # major radius stays fixed
    rc[0, :ntor] = 0.0                   # stellarator-symmetry zero pattern
    zs[0, :ntor + 1] = 0.0
    return _rebuild(b, rc, zs)


def _inflate(b):
    """Zero-pad one poloidal and one toroidal mode: coarse-to-fine continuation."""
    rc, zs = _mats(b)
    rc = np.pad(rc, ((0, 1), (1, 1)))
    zs = np.pad(zs, ((0, 1), (1, 1)))
    return _rebuild(b, rc, zs)


def solve(fm, rng):
    best_b, best_s = None, float("-inf")
    # public-submission seed bank first: triage the top entries by official
    # score at train fidelity (high-mode evals cost 12-27s — be frugal),
    # then spend the rest polishing the best one gently
    try:
        info = sorted(fm.seed_bank_info(), key=lambda e: -e["official_score"])
        info = [e for e in info if e["i"] == 3] or info
    except Exception:
        info = []
    for e in info[:4]:
        b = fm.seed_bank(e["i"])
        # bank boundaries are known-good leaderboard submissions and never truly
        # hang VMEC; a timed-out eval (host CPU contention) returns None/-inf,
        # so retry instead of silently falling through to the NAE portfolio
        s = float("-inf")
        for _retry in range(3):
            s = fm.score(fm.eval(b))
            if s > float("-inf"):
                break
        if s > best_s:
            best_b, best_s = b, s
    forced_bank = False
    if best_b is None and info:
        # bank boundaries are known-good leaderboard submissions: when every
        # triage eval died anyway (timeouts under host CPU contention -> -inf,
        # cost runs s105/s103-* 2026-07-25), adopt the top-ranked entry rather
        # than silently falling through to the NAE portfolio; the polish loop's
        # first successful eval sets the real baseline
        best_b, forced_bank = fm.seed_bank(info[0]["i"]), True
    if best_b is None:                          # no bank: NAE portfolio fallback
        for nfp, mp in ((3, 1), (4, 1), (3, 2)):
            try:
                b = fm.seed_nae(aspect_ratio=9.0, max_elongation=4.0,
                                rotational_transform=0.3 * nfp, mirror_ratio=0.2,
                                n_field_periods=nfp, max_poloidal_mode=mp,
                                max_toroidal_mode=mp)
            except Exception:
                continue
            s = fm.score(fm.eval(b))
            if s > best_s:
                best_b, best_s = b, s
    if best_b is None:
        best_b = fm.seed_ellipse(aspect_ratio=8.0, elongation=3.0,
                                 rotational_transform=0.9, n_field_periods=3)
        best_s = fm.score(fm.eval(best_b))

    # bank seeds are already polished solutions in a RAZOR-thin feasible basin:
    # sigma 0.008 already blows feasibility 0.001 -> 0.2-0.3, so polish with
    # micro-steps and never inflate
    polished_start = best_s > 0 or forced_bank
    sigma, stall, inflated = (0.0012 if polished_start else SIGMA0), 0, polished_start
    while fm.remaining() > 0:
        # batch of 2: fm.eval_many runs them on parallel workers (~half the
        # wall-clock of two sequential evals)
        cands = [_mutate(best_b, sigma, rng)
                 for _ in range(min(2, fm.remaining()))]
        scores = [fm.score(m) for m in fm.eval_many(cands)]
        i = int(np.argmax(scores))
        if scores[i] > best_s:
            best_b, best_s = cands[i], scores[i]
            sigma, stall = min(sigma * GROW, 0.15), 0
        else:
            sigma = max(sigma * SHRINK, 0.0001 if polished_start else 0.004)
            stall += 1
            if stall == INFLATE_AT and not inflated:
                best_b, inflated = _inflate(best_b), True
                sigma, stall = 0.03, 0
            elif stall >= KICK_AT:              # restart kick out of the basin
                sigma, stall = (0.003 if polished_start else SIGMA0), 0
    return best_b

# DAG campaign: pinned to bank seed #3
