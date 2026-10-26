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


def _hgb_classifier(seed, max_iter=100, max_depth=None):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=max_iter, max_depth=max_depth, random_state=seed)


def fit_policy_c(train, features, quantile=pb.PINNED_QUANTILE, seed=42, lf_gate="clf", rk4_classifier=False, sample_weight=None,
                 reg_kw=None, clf_kw=None):
    """train: rows with labels (dt_<k>, <k>_success). lf_gate: 'clf' (HGB classifier on leapfrog_success), 'censored' (leapfrog
    regressor trained on ALL rows with the label 0.5 * dt_min where no step works) or 'none' (leapfrog always allowed).
    rk4_classifier=False: RK4 is always feasible (the 1-in-16k miss shows up as a violation and stays in the risk count)."""
    reg_kw, clf_kw = reg_kw or {}, clf_kw or {}
    X = train[list(features)].to_numpy(dtype=float)
    w = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    models = {"features": list(features), "quantile": quantile, "lf_gate": lf_gate, "rk4_classifier": rk4_classifier}
    for k in CANDIDATES:
        succ = train[f"{k}_success"].to_numpy() == 1
        y = np.full(len(train), np.nan)
        y[succ] = np.log10(train[f"dt_{k}"].to_numpy(dtype=float)[succ])
        use = succ.copy()
        if k == "leapfrog" and lf_gate == "censored":
            y[~succ] = LOG_FLOOR_CENSORED
            use = np.ones(len(train), dtype=bool)
        reg = _hgb_regressor(quantile, seed, **reg_kw)
        reg.fit(X[use], y[use], sample_weight=None if w is None else w[use])
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
    out = {}
    for k in CANDIDATES:
        logdt = models[f"reg_{k}"].predict(X)
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
    for k in CANDIDATES:
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
    integ, state, dt, T, ref_pos, mu = a
    ok, _prev, curr, _tp, _ta, reason = integrate_endpoint_3d(INTEGRATORS_3D[integ], list(state), float(dt), float(T), mu, MAX_STEPS)
    if not ok:
        return False, float("nan"), int(reason)
    return True, float(np.linalg.norm(np.asarray(curr)[:3] - ref_pos)) * du_km_for_mu(mu), 0


def run_chosen(pool, df_states, horizons, ref_pos, choice_idx, raw_dt, scale, rows, mu=MU, candidates=CANDIDATES):
    """Run the chosen integrator at scale * raw dt on `rows` (positions). Returns (ok, err_km) arrays aligned with `rows`."""
    jobs = [(candidates[choice_idx[i]], df_states[i], float(raw_dt[i]) * float(scale), horizons[i], ref_pos[i], mu) for i in rows]
    res = pool.map(_run_job, jobs, chunksize=max(1, len(jobs) // 64)) if pool is not None else [_run_job(j) for j in jobs]
    return np.array([r[0] for r in res], dtype=bool), np.array([r[1] for r in res], dtype=float)


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
                       mu=MU, n_jobs=8, by_choice=False, candidates=CANDIDATES):
    """Certify ONE scale for the deployed policy. The integrator is chosen once from the unscaled predictions (so the choice cannot
    depend on the scale); every grid value is then tested on the chosen runs only. A violation is a chosen run that collides, escapes,
    goes non-finite, hits the step cap, or ends more than tol_km from the reference (this includes an integrator for which no step
    works). Rows with no feasible integrator abstain (an adaptive solve is deployed there) and are not counted.
    by_choice=True: the stricter variant, one scale per selected-integrator group, each certified at delta / (number of groups).
    Also reports the realized risk of each selected integrator at the chosen scale, with Clopper-Pearson bounds."""
    states = cal[STATE_COLS].to_numpy(dtype=float)
    T = cal["sim_time"].to_numpy(dtype=float)
    idx, raw = choose_unscaled(pred, candidates)
    rows_all = np.where(idx >= 0)[0]
    pool = Pool(n_jobs) if n_jobs > 1 else None
    cache = {}
    try:
        def bad_at(s, rows, tag="all"):
            key = (round(float(s), 12), tag)
            if key not in cache:
                ok, err = run_chosen(pool, states, T, ref_pos, idx, raw, s, rows, mu, candidates)
                cache[key] = ~ok | ~(err <= tol_km)
            return cache[key]

        report = {"n_cal": int(len(cal)), "n_policy_runs": int(len(rows_all)), "n_abstain": int(len(cal) - len(rows_all)),
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
