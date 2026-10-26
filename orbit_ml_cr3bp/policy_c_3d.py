"""
Part C policy (plan sections 4.1, 4.2, 5.3). Builds on policy_b_3d (Part B is left untouched, so Part B reproduces on this tree).

  fit_policy_c / predict_policy_c   state-feature quantile regressors per integrator; RK4 needs no classifier (feasible on 99.994 % of rows);
                                    leapfrog is allowed only through a gate (classifier probability >= p*, or a censored regressor
                                    whose prediction is above the step floor by a margin), tuned on a SELECTION set, never on
                                    calibration rows
  choose_unscaled                   the integrator choice from the raw predictions (d_hat_k / e_k), which does not depend on any scale
  certify_end_to_end                ONE scale s applied to whichever integrator is chosen; tests exactly the deployed run
                                    (fixed-sequence on a fine grid, or Part B's Bonferroni rule), with the selection-conditional risk
  policy_pred_for_evaluation        pred dict for pb.evaluate_orbits in which only the chosen integrator is feasible
"""
from __future__ import annotations

import math
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import binom

from .audit_3d import clopper_pearson, DT_MIN
from .config import MU, POSITION_TOL_KM, du_km_for_mu
from .fair_baseline_3d import _run_adaptive_ex
from .physics_3d import EVALS_PER_STEP_3D, INTEGRATORS_3D, integrate_endpoint_3d, reference_trajectory_dense_3d
from . import policy_b_3d as pb

from .osculating_3d import OSC_FEATURE_COLS

FEATS_OSC = pb.FEATS_STATE + OSC_FEATURE_COLS      # state + analytic two-body periapsis features (plan 5.2.2)
CANDIDATES = pb.CANDIDATES
ALPHA = pb.ALPHA
DELTA = pb.DELTA
STATE_COLS = pb.STATE_COLS
MAX_STEPS = 4_000_000
LOG_FLOOR_CENSORED = math.log10(0.5 * DT_MIN)     # label given to rows where no step >= DT_MIN works (censored-regression gate)

# 0.1 * 2^(k/8), k = 0..26 (0.1 ... 0.951), then 1.0: 28 values, tested safest-first (plan 2.3a)
FIXED_SEQ_GRID = tuple(float(0.1 * 2.0 ** (k / 8.0)) for k in range(27)) + (1.0,)


# ---------------------------------------------------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------------------------------------------------

def _hgb_regressor(q, seed, max_iter=300, learning_rate=0.05, max_depth=None):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=max_iter, learning_rate=learning_rate,
                                         max_depth=max_depth, random_state=seed)


def _hgb_classifier(seed, max_iter=100, max_depth=None, learning_rate=0.1):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=max_iter, max_depth=max_depth, learning_rate=learning_rate, random_state=seed)


def fit_policy_c(train, features, quantile=pb.PINNED_QUANTILE, seed=42, lf_gate="clf", rk4_classifier=False, sample_weight=None,
                 reg_kw=None, clf_kw=None, structured=False, candidates=CANDIDATES):
    """train: rows with labels (dt_<k>, <k>_success). lf_gate: 'clf' (HGB classifier on leapfrog_success), 'censored' (leapfrog
    regressor trained on ALL rows with the label 0.5 * dt_min where no step works) or 'none' (leapfrog always allowed).
    rk4_classifier=False: RK4 is always feasible (the 1-in-16k miss shows up as a violation and stays in the risk count).
    structured=True: the regressors predict log10 d* - osc_logtau_hat (the analytic timescale, which is a feature); the offset is added
    back at prediction time. It removes the part of the step that is just the local dynamical time and lets small ensembles reach the
    accuracy of much larger ones (D_explore: 10 trees of depth 3 go from 0.47 to 0.29 dex median error).
    candidates=("rk4s",): the Sundman-time policy (one regressor on log10 ds*, labels dt_rk4s / rk4s_success, no gate)."""
    reg_kw, clf_kw = reg_kw or {}, clf_kw or {}
    features = list(features)
    X = train[features].to_numpy(dtype=float)
    off = X[:, features.index("osc_logtau_hat")] if structured else 0.0
    w = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    models = {"features": features, "quantile": quantile, "lf_gate": lf_gate, "rk4_classifier": rk4_classifier, "structured": structured,
              "candidates": tuple(candidates)}
    for k in candidates:
        succ = train[f"{k}_success"].to_numpy() == 1
        y = np.full(len(train), np.nan)
        y[succ] = np.log10(train[f"dt_{k}"].to_numpy(dtype=float)[succ])
        use = succ.copy()
        if k == "leapfrog" and lf_gate == "censored":
            y[~succ] = LOG_FLOOR_CENSORED
            use = np.ones(len(train), dtype=bool)
        reg = _hgb_regressor(quantile, seed, **reg_kw)
        yy = y - off if structured else y
        reg.fit(X[use], yy[use] if structured else y[use], sample_weight=None if w is None else w[use])
        models[f"reg_{k}"] = reg
        need_clf = (k == "leapfrog" and lf_gate == "clf") or (k == "rk4" and rk4_classifier)
        if need_clf:
            clf = _hgb_classifier(seed, **clf_kw)
            clf.fit(X, succ.astype(int), sample_weight=w)
            models[f"clf_{k}"] = clf
    return models


def predict_raw_c(models, df):
    """Unthresholded predictions: log10 dt per candidate, feasibility score per candidate (probability, or the log prediction for
    the censored gate, or +inf where no gate applies)."""
    X = df[models["features"]].to_numpy(dtype=float)
    off = X[:, models["features"].index("osc_logtau_hat")] if models.get("structured") else 0.0
    out = {}
    for k in models["candidates"]:
        logdt = models[f"reg_{k}"].predict(X) + off
        if f"clf_{k}" in models:
            score = models[f"clf_{k}"].predict_proba(X)[:, 1]
        elif k == "leapfrog" and models["lf_gate"] == "censored":
            score = logdt - math.log10(DT_MIN)           # margin above the step floor, in dex
        else:
            score = np.full(len(X), np.inf)
        out[k] = {"logdt": logdt, "score": score}
    return out


def apply_gate(raw, p_star=0.5, g_star=0.0, lf_gate="clf"):
    """pred dict (feasible, dt per candidate) for pb.policy_choice / evaluate_orbits. p_star: classifier threshold; g_star: margin
    (dex above dt_min) for the censored gate."""
    pred = {}
    for k in raw:
        r = raw[k]
        if k == "leapfrog" and lf_gate == "clf":
            feas = r["score"] >= p_star
        elif k == "leapfrog" and lf_gate == "censored":
            feas = r["score"] >= g_star
        elif k == "rk4" and np.isfinite(r["score"]).all():          # an RK4 classifier was fitted
            feas = r["score"] >= 0.5
        else:
            feas = np.ones(len(r["logdt"]), dtype=bool)
        pred[k] = {"feasible": np.asarray(feas, dtype=bool), "dt": np.power(10.0, r["logdt"])}
    return pred


def predict_policy_c(models, df, p_star=0.5, g_star=0.0):
    return apply_gate(predict_raw_c(models, df), p_star=p_star, g_star=g_star, lf_gate=models["lf_gate"])


# ---------------------------------------------------------------------------------------------------------------------
# choice (independent of any scale) and the evaluation view of it
# ---------------------------------------------------------------------------------------------------------------------

def choose_unscaled(pred, candidates=CANDIDATES):
    """Per row: index into `candidates` of the feasible integrator with the largest d_hat / e (cost-normalized choice, ties to the
    first), or -1 when none is feasible; and the raw predicted dt of the choice (NaN where none)."""
    n = len(pred[candidates[0]]["dt"])
    score = np.full((len(candidates), n), -np.inf)
    for j, k in enumerate(candidates):
        d = np.asarray(pred[k]["dt"], dtype=float)
        ok = np.asarray(pred[k]["feasible"], dtype=bool) & np.isfinite(d) & (d > 0)
        score[j] = np.where(ok, d / EVALS_PER_STEP_3D[k], -np.inf)
    idx = np.argmax(score, axis=0)
    none = ~np.isfinite(score.max(axis=0))
    idx = np.where(none, -1, idx)
    raw = np.full(n, np.nan)
    for j, k in enumerate(candidates):
        m = idx == j
        raw[m] = np.asarray(pred[k]["dt"], dtype=float)[m]
    return idx, raw


def policy_pred_for_evaluation(pred, choice_idx, candidates=CANDIDATES):
    """A pred dict in which only the chosen integrator is feasible, so pb.evaluate_orbits (which re-decides from the scaled values)
    cannot pick differently: the choice made on the raw predictions is the deployed one."""
    out = {}
    for j, k in enumerate(candidates):
        out[k] = {"feasible": (choice_idx == j), "dt": np.asarray(pred[k]["dt"], dtype=float)}
    return out


# ---------------------------------------------------------------------------------------------------------------------
# running the chosen fixed-step integrations
# ---------------------------------------------------------------------------------------------------------------------

def _ref_job(a):
    state, T, mu = a
    sol, reached = reference_trajectory_dense_3d(list(state), T, mu)
    return (sol.sol(T)[:3] if reached else np.full(3, np.nan))


def reference_final_positions(df, mu=MU, n_jobs=8):
    states = df[STATE_COLS].to_numpy(dtype=float)
    T = df["sim_time"].to_numpy(dtype=float)
    jobs = [(states[i], float(T[i]), mu) for i in range(len(df))]
    if n_jobs > 1:
        with Pool(n_jobs) as pool:
            res = pool.map(_ref_job, jobs, chunksize=max(1, len(jobs) // (8 * n_jobs)))
    else:
        res = [_ref_job(j) for j in jobs]
    return np.vstack(res)


def _run_job(a):
    """One chosen fixed-step run, optionally guarded (step-over guard g) with an adaptive fallback (rtol, method) from t = 0.
    Returns (ok, err_km, evals, fell_back, reason). evals = steps x evaluations per step (+ the fallback's nfev)."""
    integ, state, dt, T, ref_pos, mu, g, fb = a
    if integ == "rk4s":              # Sundman-time RK4: dt is the fictitious step; the cost is counted, not ceil(T / dt) x 4
        from .sundman_3d import integrate_endpoint_rk4s, rk4s_evals, SUNDMAN_TAU_C
        ok, _prev, curr, _tp, _ta, reason, ns, nf = integrate_endpoint_rk4s(state, dt, T, mu, MAX_STEPS, tau_c=SUNDMAN_TAU_C)
        ev = rk4s_evals(ns, nf)
        if not ok:
            return False, float("nan"), ev, False, int(reason)
        return True, float(np.linalg.norm(np.asarray(curr)[:3] - ref_pos)) * du_km_for_mu(mu), ev, False, 0
    if math.isinf(g):
        ok, _prev, curr, _tp, _ta, reason = integrate_endpoint_3d(INTEGRATORS_3D[integ], list(state), float(dt), float(T), mu, MAX_STEPS)
        steps = int(math.ceil(T / float(dt) - 1e-9)) if ok else 0
    else:
        from .guard_3d import integrate_endpoint_guard, REASON_GUARD
        ok, _prev, curr, _tp, _ta, reason, steps = integrate_endpoint_guard(integ, state, dt, T, mu, MAX_STEPS, g)
        if reason == REASON_GUARD:
            if fb is None:
                return False, float("nan"), steps * EVALS_PER_STEP_3D[integ], False, int(reason)
            res, _capped = _run_adaptive_ex(np.asarray(state, dtype=float), float(T), ref_pos, float(fb[0]), mu, fb[1])
            ev = steps * EVALS_PER_STEP_3D[integ]
            if res is None:
                return False, float("nan"), ev, True, int(reason)
            return True, float(res["err_km"]), ev + int(res["nfev"]), True, int(reason)
    ev = steps * EVALS_PER_STEP_3D[integ]
    if not ok:
        return False, float("nan"), ev, False, int(reason)
    return True, float(np.linalg.norm(np.asarray(curr)[:3] - ref_pos)) * du_km_for_mu(mu), ev, False, 0


def _adaptive_only_job(a):
    state, T, ref_pos, mu, fb = a
    res, _capped = _run_adaptive_ex(np.asarray(state, dtype=float), float(T), ref_pos, float(fb[0]), mu, fb[1])
    if res is None:
        return False, float("nan"), 0, True, -1
    return True, float(res["err_km"]), int(res["nfev"]), True, -1


def run_chosen(pool, df_states, horizons, ref_pos, choice_idx, raw_dt, scale, rows, mu=MU, candidates=CANDIDATES, g=math.inf, fb=None,
               abstain=None):
    """Run the chosen integrator at scale * raw dt on `rows` (positions). Returns (ok, err_km, evals, fell_back) arrays aligned with
    `rows`. g < inf turns the step-over guard on (fb = (rtol, method) is the adaptive fallback); rows where `abstain` is True go straight
    to the adaptive solve at fb."""
    jobs, kinds = [], []
    for i in rows:
        if abstain is not None and abstain[i]:
            jobs.append((df_states[i], horizons[i], ref_pos[i], mu, fb))
            kinds.append(1)
        else:
            jobs.append((candidates[choice_idx[i]], df_states[i], float(raw_dt[i]) * float(scale), horizons[i], ref_pos[i], mu, g, fb))
            kinds.append(0)
    fn = lambda j: _adaptive_only_job(j) if len(j) == 5 else _run_job(j)       # noqa: E731
    if pool is not None:
        res = pool.map(_mixed_job, jobs, chunksize=max(1, len(jobs) // 64))
    else:
        res = [fn(j) for j in jobs]
    return (np.array([r[0] for r in res], dtype=bool), np.array([r[1] for r in res], dtype=float),
            np.array([r[2] for r in res], dtype=float), np.array([r[3] for r in res], dtype=bool))


def _mixed_job(j):
    return _adaptive_only_job(j) if len(j) == 5 else _run_job(j)


# ---------------------------------------------------------------------------------------------------------------------
# certification
# ---------------------------------------------------------------------------------------------------------------------

def certify_sequence(violations_at, grid, alpha=ALPHA, delta=DELTA, rule="fixed_sequence", n_tests=None):
    """Pure statistics, with the integration hidden behind `violations_at(s) -> (k, n)` so it can be tested on its own.
    fixed_sequence: values tested safest-first (ascending) at the FULL delta, stopping at the first non-rejection; the certified value
      is the last rejected one (fixed-sequence Learn-then-Test; family-wise valid because the order is fixed in advance).
    bonferroni: each value tested alone at delta / m (m = n_tests or len(grid)), scanning from the largest value down and stopping at
      the first rejection = the largest certified value of the full chain (Part B's rule)."""
    chain = []
    chosen = None
    if rule == "fixed_sequence":
        for s in sorted(grid):
            k, n = violations_at(s)
            p = float(binom.cdf(k, n, alpha)) if n else 1.0
            chain.append({"scale": float(s), "n": int(n), "violations": int(k), "violation_rate": k / n if n else float("nan"),
                          "p_value": p, "certified": bool(p <= delta)})
            if p > delta:
                break
            chosen = float(s)
        thr = delta
    elif rule == "bonferroni":
        m = n_tests or len(grid)
        thr = delta / m
        for s in sorted(grid, reverse=True):
            k, n = violations_at(s)
            p = float(binom.cdf(k, n, alpha)) if n else 1.0
            chain.append({"scale": float(s), "n": int(n), "violations": int(k), "violation_rate": k / n if n else float("nan"),
                          "p_value": p, "certified": bool(p <= thr)})
            if p <= thr:
                chosen = float(s)
                break
    else:
        raise ValueError(rule)
    return {"rule": rule, "chosen_scale": chosen, "p_threshold": thr, "alpha": alpha, "delta": delta, "chain": chain}


def certify_end_to_end(cal, pred, ref_pos, tol_km=POSITION_TOL_KM, alpha=ALPHA, delta=DELTA, grid=FIXED_SEQ_GRID, rule="fixed_sequence",
                       mu=MU, n_jobs=8, by_choice=False, candidates=CANDIDATES, guard_g=math.inf, fallback=None, abstain=None):
    """Certify ONE scale for the deployed policy. The integrator is chosen once from the unscaled predictions (so the choice cannot
    depend on the scale); every grid value is then tested on the chosen runs only. A violation is a chosen run that collides, escapes,
    goes non-finite, hits the step cap, or ends more than tol_km from the reference (this includes an integrator for which no step
    works). Rows with no feasible integrator abstain (an adaptive solve is deployed there) and are not counted.
    by_choice=True: the stricter variant, one scale per selected-integrator group, each certified at delta / (number of groups).
    guard_g < inf: the step-over guard is part of the deployed policy (aborted runs fall back to the adaptive solve `fallback` =
    (rtol, method) from t = 0); abstain (bool array): rows sent straight to that adaptive solve. Both are inside the certificate: a
    fallback or abstained run that misses the tolerance is a violation, and the denominator is every row with a deployed run.
    Also reports the realized risk of each selected integrator at the chosen scale, with Clopper-Pearson bounds."""
    states = cal[STATE_COLS].to_numpy(dtype=float)
    T = cal["sim_time"].to_numpy(dtype=float)
    idx, raw = choose_unscaled(pred, candidates)
    if abstain is not None:
        abstain = np.asarray(abstain, dtype=bool)
        if by_choice:
            raise ValueError("by_choice and abstain are not combined")
    rows_all = np.where((idx >= 0) | (abstain if abstain is not None else False))[0]
    pool = Pool(n_jobs) if n_jobs > 1 else None
    cache = {}
    try:
        def bad_at(s, rows, tag="all"):
            key = (round(float(s), 12), tag)
            if key not in cache:
                ok, err, _ev, _fb = run_chosen(pool, states, T, ref_pos, idx, raw, s, rows, mu, candidates, guard_g, fallback, abstain)
                cache[key] = ~ok | ~(err <= tol_km)
            return cache[key]

        report = {"n_cal": int(len(cal)), "n_policy_runs": int(len(rows_all)), "n_abstain": int(abstain.sum()) if abstain is not None else int(len(cal) - len(rows_all)),
                  "guard_g": None if math.isinf(guard_g) else guard_g,
                  "choice_counts": {k: int((idx == j).sum()) for j, k in enumerate(candidates)}}
        if not by_choice:
            res = certify_sequence(lambda s: (int(bad_at(s, rows_all).sum()), len(rows_all)), grid, alpha, delta, rule)
            report.update(res)
            report["scale"] = res["chosen_scale"]
            s = res["chosen_scale"]
            report["scales_by_choice"] = None
        else:
            groups = [j for j in range(len(candidates)) if (idx == j).any()]
            G = len(groups)
            per = {}
            for j in groups:
                rows_j = np.where(idx == j)[0]
                per[candidates[j]] = certify_sequence(lambda s, rows_j=rows_j, j=j: (int(bad_at(s, rows_j, j).sum()), len(rows_j)), grid, alpha,
                                                      delta / G, rule, n_tests=(len(grid) * G))
            report["by_choice_certificates"] = per
            report["scales_by_choice"] = {k: v["chosen_scale"] for k, v in per.items()}
            report["scale"] = None
            report["chosen_scale"] = None
            s = None
        # selection-conditional realized risk at the chosen scale(s)
        cond = {}
        for j, k in enumerate(candidates):
            rows_j = np.where(idx == j)[0]
            if not len(rows_j):
                continue
            sk = s if s is not None else report["scales_by_choice"].get(k)
            if sk is None:
                continue
            nb = int(bad_at(sk, rows_j, j).sum())
            lo, hi = clopper_pearson(nb, len(rows_j))
            cond[k] = {"n": int(len(rows_j)), "violations": nb, "rate": nb / len(rows_j), "cp95": [lo, hi], "upper_above_alpha": bool(hi > alpha)}
        report["conditional_risk"] = cond
        report["conditional_breach"] = bool(any(v["upper_above_alpha"] for v in cond.values()))
        if s is not None:
            bad = bad_at(s, rows_all)
            report["violations"] = int(bad.sum())
            report["violation_rate"] = float(bad.mean()) if len(bad) else float("nan")
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return report


# ---------------------------------------------------------------------------------------------------------------------
# gate threshold, tuned on a selection set
# ---------------------------------------------------------------------------------------------------------------------

def infeasible_choice_rate(raw, sel, p_star=0.5, g_star=0.0, lf_gate="clf", candidates=CANDIDATES):
    """Share of selection rows on which the policy picks an integrator for which NO step >= dt_min meets the tolerance (no scale can
    repair such a choice), and the share of rows that pick leapfrog."""
    pred = apply_gate(raw, p_star=p_star, g_star=g_star, lf_gate=lf_gate)
    idx, _ = choose_unscaled(pred, candidates)
    bad = np.zeros(len(sel), dtype=bool)
    for j, k in enumerate(candidates):
        bad |= (idx == j) & (sel[f"{k}_success"].to_numpy() != 1)
    return float(bad.mean()), float((idx == candidates.index("leapfrog")).mean()) if "leapfrog" in candidates else float("nan")


def select_gate_threshold(raw, sel, max_rate=0.005, lf_gate="clf", grid=None, candidates=CANDIDATES):
    """Smallest threshold (p* for the classifier gate, margin g* in dex for the censored gate) whose infeasible-choice rate on the
    selection rows is <= max_rate (a 0.5 % budget leaves room inside alpha = 5 %). Returns (threshold, table)."""
    if grid is None:
        grid = (np.round(np.linspace(0.05, 0.995, 40), 4) if lf_gate == "clf" else np.round(np.linspace(0.0, 1.5, 31), 3))
    rows = []
    chosen = None
    for g in grid:
        kw = {"p_star": g} if lf_gate == "clf" else {"g_star": g}
        rate, lf_share = infeasible_choice_rate(raw, sel, lf_gate=lf_gate, candidates=candidates, **kw)
        rows.append({"threshold": float(g), "infeasible_choice_rate": rate, "leapfrog_share": lf_share})
        if chosen is None and rate <= max_rate:
            chosen = float(g)
    return chosen, pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------------
# abstention and the evaluation pass of the Part C policy (guard + fallback + abstention)
# ---------------------------------------------------------------------------------------------------------------------

def abstain_flags(df, rule="none", logtau_below=None, dispersion=None, frac=0.2):
    """Rows sent straight to the adaptive solver. 'physics' (free): the osculating orbit about the secondary comes back to periapsis within
    the horizon (osc_reach_2 = 1) and the analytic timescale osc_logtau_hat is below `logtau_below`. 'dispersion': the top `frac` of the
    q50 - q10 spread of the RK4 log-step predictions (`dispersion` = that spread, one value per row; a second quantile model, so about
    twice the inference). Chosen on the selection set, never on calibration rows."""
    if rule == "none":
        return np.zeros(len(df), dtype=bool)
    if rule == "physics":
        return ((df["osc_reach_2"].to_numpy() == 1.0) & (df["osc_logtau_hat"].to_numpy() < logtau_below))
    if rule == "dispersion":
        thr = np.quantile(dispersion, 1.0 - frac)
        return np.asarray(dispersion) >= thr
    raise ValueError(rule)


def evaluate_orbits_c(df, choice_idx, raw_dt, scale, rtols=None, do_spec=True, tol_km=POSITION_TOL_KM, mu=MU, n_jobs=8,
                      methods=pb.METHODS, guard_g=math.inf, fallback=None, abstain=None, candidates=CANDIDATES, progress=True,
                      ref_pos=None):
    """The deployed Part C policy (common scale `scale` on the integrator chosen from the raw predictions, optional step-over guard with
    adaptive fallback, optional abstention) and the adaptive comparators, one row per orbit. Same columns as policy_b_3d.evaluate_orbits,
    plus ml_fallback / ml_abstain; ml_integrator is 'adaptive' for abstained rows (they are run, and counted in the risk)."""
    import time
    states = df[STATE_COLS].to_numpy(dtype=float)
    T = df["sim_time"].to_numpy(dtype=float)
    rids = df["row_id"].to_numpy() if "row_id" in df.columns else np.arange(len(df))
    if ref_pos is None:
        ref_pos = reference_final_positions(df, mu, n_jobs)
    abstain = np.zeros(len(df), dtype=bool) if abstain is None else np.asarray(abstain, dtype=bool)
    t0 = time.time()
    # adaptive comparators (pb's own job with the ML part switched off)
    jobs = [(int(rids[i]), states[i], float(T[i]), None, None, tol_km, mu, rtols or {}, do_spec, methods) for i in range(len(df))]
    with Pool(n_jobs) as pool:
        res_ad = pd.DataFrame(list(pool.imap_unordered(pb._orbit_job, jobs, chunksize=4)))
        if progress:
            print(f"  evaluate_orbits_c: adaptive comparators done, {time.time() - t0:.0f}s", flush=True)
        rows = np.arange(len(df))
        sc = np.asarray(scale, dtype=float) if np.ndim(scale) else np.full(len(df), float(scale))
        # per-row scale (by_choice) is handled by running each group at its own scale
        ok = np.zeros(len(df), dtype=bool)
        err = np.full(len(df), np.nan)
        ev = np.zeros(len(df))
        fbk = np.zeros(len(df), dtype=bool)
        for s_val in np.unique(sc[~np.isnan(sc)]):
            r = rows[(sc == s_val)]
            o, e, v, f = run_chosen(pool, states, T, ref_pos, choice_idx, raw_dt, s_val, r, mu, candidates, guard_g, fallback, abstain)
            ok[r], err[r], ev[r], fbk[r] = o, e, v, f
    names = np.array(list(candidates) + ["adaptive"], dtype=object)
    integ = np.where(abstain, "adaptive", np.where(choice_idx >= 0, names[np.maximum(choice_idx, 0)], None))
    out = pd.DataFrame({"row_id": rids.astype(int), "ml_integrator": integ, "ml_raw_dt": np.where(abstain, np.nan, raw_dt),
                        "ml_dt": np.where(abstain, np.nan, raw_dt * sc), "ml_scale": np.where(abstain, np.nan, sc),
                        "ml_ok": ok, "ml_err_km": err, "ml_evals": ev, "ml_fallback": fbk, "ml_abstain": abstain})
    out["ml_steps"] = np.nan
    out.loc[(choice_idx < 0) & ~abstain, ["ml_ok", "ml_err_km", "ml_evals"]] = [False, np.nan, np.nan]
    return out.merge(res_ad, on="row_id", how="left")
