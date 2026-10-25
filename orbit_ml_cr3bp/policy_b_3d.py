"""
Part B policy evaluation (failure study). Builds the per-orbit ledger that the cost-gap decomposition (failure_study_3d.py) reads.

Pieces
  fit_policy_models      the deployable model: state-only features, pinned learner (HGB quantile q on log10 dt, RF classifiers)
  certify_scale_lazy     LTT-Bonferroni scale for one integrator; scanning from the largest scale down and stopping at the first
                         certified value gives EXACTLY the "largest certified" value of the full chain (each value is tested on
                         its own at delta/m), at a fraction of the integrations. Tested against audit_3d.calibrate_scale_ltt.
  certify_rtol_lazy      the same for an adaptive solver's rtol (scanning from the loosest rtol), plus the equal-risk rtol
  spec_rtol_search       per-orbit oracle rtol ("_spec"): loosest rtol whose final error is <= tolerance, bracket + log bisection
  evaluate_orbits        one parallel pass: ML run, adaptive spec / cal / equal-risk runs, oracle fixed costs
Costs are right-hand-side evaluations (RHS-equivalents for the features and the model), as in Part A.
"""
from __future__ import annotations

import math
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import binom

from .audit_3d import (LTT_SCALE_GRID, ReferenceCache, position_error_audit_3d, calibrate_scale_ltt, clopper_pearson)
from .config import MU, POSITION_TOL_KM, du_km_for_mu
from .fair_baseline_3d import (RTOL_GRID_B, _run_adaptive_ex, MAX_ADAPTIVE_NFEV, measure_prediction_cost,
                               CHEAP_FEATURE_EVALS_EQUIV)
from .modeling_3d import fit_hurdle_models, predict_hurdle, cost_normalized_choice
from .physics_3d import (EVALS_PER_STEP_3D, INTEGRATORS_3D, integrate_endpoint_3d, reference_trajectory_dense_3d)

FEATS_STATE = ["x", "y", "z", "vx", "vy", "vz", "sim_time", "jacobi_3d", "r1_3d", "r2_3d", "r1_r2_ratio_3d",
               "c_minus_c_l1_3d"]
FEATS_FTLE = FEATS_STATE + ["ftle_3d"]
CANDIDATES = ("rk4", "leapfrog")
METHODS = ("RK45", "DOP853")
ALPHA = 0.05
DELTA = 0.1
PINNED_QUANTILE = 0.1
N_ESTIMATORS = 150
STATE_COLS = ["x", "y", "z", "vx", "vy", "vz"]


def usable_rows(df):
    """Rows the study can label: finite FTLE and a reference that reaches the horizon."""
    m = ~df["ftle_nan"].astype(bool)
    if "reference_divergent" in df.columns:
        m &= ~df["reference_divergent"].fillna(False).astype(bool)
    return df[m]


# ---------------------------------------------------------------------------------------------------------------------
# model and certification
# ---------------------------------------------------------------------------------------------------------------------

def fit_policy_models(data, train_idx, features=FEATS_STATE, quantile=PINNED_QUANTILE, seed=42,
                      n_estimators=N_ESTIMATORS):
    return fit_hurdle_models(data, list(features), np.asarray(train_idx), random_state=seed, n_estimators=n_estimators,
                             regressor_kind="hgb_quantile", quantile=quantile)


def certify_scale_lazy(cal, pred, integrator, tol_km=POSITION_TOL_KM, alpha=ALPHA, delta=DELTA, grid=LTT_SCALE_GRID,
                       mu=MU, cache=None):
    """Largest scale on `grid` whose one-sided binomial test of 'violation rate > alpha' rejects at delta/m. Scanning from the
    largest scale down and stopping at the first rejection is the same selection as the full Bonferroni chain."""
    cache = cache or ReferenceCache(mu)
    m = len(grid)
    thr = delta / m
    chain = []
    for s in sorted(grid, reverse=True):
        a = position_error_audit_3d(cal, pred, tol_km, scale=s, mu=mu, cache=cache, integrators=(integrator,))
        n = len(a)
        if n == 0:
            break
        k = int((~a["stable"]).sum())
        p = float(binom.cdf(k, n, alpha))
        chain.append({"scale": float(s), "n": n, "violations": k, "violation_rate": k / n, "p_value": p,
                      "certified": bool(p <= thr)})
        if p <= thr:
            return {"integrator": integrator, "chosen_scale": float(s), "p_threshold": thr, "alpha": alpha,
                    "delta": delta, "chain": chain, "n": n, "violations": k}
    return {"integrator": integrator, "chosen_scale": None, "p_threshold": thr, "alpha": alpha, "delta": delta,
            "chain": chain, "n": chain[-1]["n"] if chain else 0, "violations": chain[-1]["violations"] if chain else 0}


def _cal_job(a):
    state, T, ref_pos, rtol, mu, method, tol_km = a
    res, capped = _run_adaptive_ex(state, T, ref_pos, rtol, mu, method)
    bad = res is None or res["err_km"] > tol_km
    return bool(bad), bool(capped), (None if res is None else int(res["nfev"]))


def certify_rtol_lazy(cal, method, tol_km=POSITION_TOL_KM, alpha=ALPHA, delta=DELTA, grid=RTOL_GRID_B, mu=MU,
                      equal_risk_target=None, n_jobs=1, items=None, refine_steps=4):
    """Loosest certified rtol (scanning from the loosest value, stopping at the first rejection) and, if
    equal_risk_target is given, the loosest rtol whose realized violation rate on the calibration rows is <= that target
    (scanning continues, if needed, until it is found). `items` = precomputed [(state, T, ref_pos)] to share between methods."""
    if items is None:
        items = reference_items(cal, mu)
    m = len(grid)
    thr = delta / m
    chain = []
    chosen = None
    eqr = None
    pool = Pool(processes=n_jobs) if n_jobs > 1 else None
    try:
        for rtol in sorted(grid, reverse=True):
            jobs = [(st, T, rp, float(rtol), mu, method, tol_km) for st, T, rp in items]
            out = pool.map(_cal_job, jobs, chunksize=max(1, len(jobs) // (6 * n_jobs))) if pool else \
                [_cal_job(j) for j in jobs]
            n = len(items)
            k = int(sum(o[0] for o in out))
            p = float(binom.cdf(k, n, alpha))
            nf = [o[2] for o in out if o[2] is not None]
            chain.append({"rtol": float(rtol), "n": n, "violations": k, "violation_rate": k / n, "p_value": p,
                          "certified": bool(p <= thr), "capped": int(sum(o[1] for o in out)),
                          "mean_nfev": float(np.mean(nf)) if nf else float("nan")})
            if chosen is None and p <= thr:
                chosen = float(rtol)
            if eqr is None and equal_risk_target is not None and k / n <= equal_risk_target:
                eqr = float(rtol)
            if chosen is not None and (equal_risk_target is None or eqr is not None):
                break
        # equal-risk refinement: the grid is half a decade wide, so bisect (log10) between the loosest grid value that meets the
        # target and the next looser one that does not, keeping the loosest rtol whose realized rate on the rows is <= target
        if equal_risk_target is not None and eqr is not None and refine_steps > 0:
            tighter = [c for c in chain if c["rtol"] == eqr][0]
            looser = [c for c in chain if c["rtol"] > eqr]
            if looser and looser[-1]["violation_rate"] > equal_risk_target:
                lo, hi = math.log10(eqr), math.log10(min(c["rtol"] for c in looser))
                for _ in range(refine_steps):
                    mid = 0.5 * (lo + hi)
                    jobs = [(st, T, rp, 10.0 ** mid, mu, method, tol_km) for st, T, rp in items]
                    out = pool.map(_cal_job, jobs, chunksize=max(1, len(jobs) // (6 * n_jobs))) if pool else \
                        [_cal_job(j) for j in jobs]
                    k = int(sum(o[0] for o in out))
                    chain.append({"rtol": 10.0 ** mid, "n": len(items), "violations": k, "violation_rate": k / len(items),
                                  "p_value": float(binom.cdf(k, len(items), alpha)), "certified": None, "refine": True})
                    if k / len(items) <= equal_risk_target:
                        lo = mid
                        eqr = 10.0 ** mid
                    else:
                        hi = mid
    finally:
        if pool:
            pool.close()
            pool.join()
    return {"method": method, "chosen_rtol": chosen, "equal_risk_rtol": eqr, "equal_risk_target": equal_risk_target,
            "p_threshold": thr, "n_cal": len(items), "chain": chain}


def reference_items(df, mu=MU):
    items = []
    for _, r in df.iterrows():
        st = [float(r[c]) for c in STATE_COLS]
        T = float(r["sim_time"])
        sol, ok = reference_trajectory_dense_3d(st, T, mu)
        if ok:
            items.append((np.asarray(st), T, sol.sol(T)[:3]))
    return items


# ---------------------------------------------------------------------------------------------------------------------
# per-orbit oracle rtol
# ---------------------------------------------------------------------------------------------------------------------

def spec_rtol_search(state, T, ref_pos, target_km, mu, method, lo=1e-13, hi=1e-1, start=1e-6, width=0.02):
    """Loosest rtol (to `width` decades) at which `method` reaches final error <= target_km. Same meaning as
    fair_baseline_3d.scipy_matched_accuracy (which always solves at 1e-13 first, a run that can exceed the evaluation cap), found
    with a bracket from `start` then log bisection. Returns {"rtol","nfev","err_km"} or None."""
    def run(rt):
        res, _capped = _run_adaptive_ex(state, T, ref_pos, rt, mu, method)
        return res if (res is not None and res["err_km"] <= target_km) else None

    best = run(start)
    if best is not None:
        good, bad = math.log10(start), None
        x = good
        while x + 1.0 <= math.log10(hi) + 1e-9:
            x += 1.0
            r = run(10.0 ** x)
            if r is None:
                bad = x
                break
            best, good = r, x
        if bad is None:
            return best
    else:
        bad = math.log10(start)
        good = None
        x = bad
        while x - 1.0 >= math.log10(lo) - 1e-9:
            x -= 1.0
            r = run(10.0 ** x)
            if r is not None:
                best, good = r, x
                break
            bad = x
        if good is None:
            return None
    while bad - good > width:
        mid = 0.5 * (good + bad)
        r = run(10.0 ** mid)
        if r is not None:
            best, good = r, mid
        else:
            bad = mid
    return best


# ---------------------------------------------------------------------------------------------------------------------
# one parallel pass over orbits
# ---------------------------------------------------------------------------------------------------------------------

def _orbit_job(a):
    (rid, state, T, integ, dt_used, tol_km, mu, rtols, do_spec, methods) = a
    t0 = time.perf_counter()
    out = {"row_id": rid}
    sol, reached = reference_trajectory_dense_3d(list(state), T, mu)
    out["ref_reached"] = bool(reached)
    if not reached:
        return out
    ref_pos = sol.sol(T)[:3]
    if integ is not None:
        ok, prev, curr, t_prev, t_act, reason = integrate_endpoint_3d(INTEGRATORS_3D[integ], list(state), float(dt_used),
                                                                      T, mu, 4_000_000)
        out["ml_ok"] = bool(ok)
        out["ml_reason"] = int(reason)
        if ok:
            err = float(np.linalg.norm(np.asarray(curr)[:3] - ref_pos)) * du_km_for_mu(mu)
            out["ml_err_km"] = err
            out["ml_steps"] = int(math.ceil(T / float(dt_used) - 1e-9))
            out["ml_evals"] = out["ml_steps"] * EVALS_PER_STEP_3D[integ]
    for m in methods:
        for tag in ("cal", "eqr"):
            rt = rtols.get((m, tag))
            if rt is None:
                continue
            res, capped = _run_adaptive_ex(np.asarray(state), T, ref_pos, float(rt), mu, m)
            out[f"{m}_{tag}_rtol"] = float(rt)
            out[f"{m}_{tag}_capped"] = bool(capped)
            if res is None:
                out[f"{m}_{tag}_nfev"] = float("nan")
                out[f"{m}_{tag}_ok"] = False
            else:
                out[f"{m}_{tag}_nfev"] = int(res["nfev"])
                out[f"{m}_{tag}_err_km"] = float(res["err_km"])
                out[f"{m}_{tag}_ok"] = bool(res["err_km"] <= tol_km)
        if do_spec:
            res = spec_rtol_search(np.asarray(state), T, ref_pos, tol_km, mu, m)
            if res is None:
                out[f"{m}_spec_nfev"] = float("nan")
            else:
                out[f"{m}_spec_rtol"] = float(res["rtol"])
                out[f"{m}_spec_nfev"] = int(res["nfev"])
                out[f"{m}_spec_err_km"] = float(res["err_km"])
    out["job_wall_s"] = time.perf_counter() - t0
    return out


def policy_choice(pred, scales, i, candidates=CANDIDATES):
    """Cost-normalized choice among candidate integrators for orbit position i using SCALED predicted dts.
    Returns (integrator | None, raw dt of the choice, scaled dt)."""
    dts = {}
    for k in candidates:
        sc = scales.get(k)
        if sc is None or not pred[k]["feasible"][i]:
            continue
        d = float(pred[k]["dt"][i])
        if np.isfinite(d) and d > 0:
            dts[k] = d * float(sc)
    choice, _ = cost_normalized_choice(dts)
    if choice is None:
        return None, float("nan"), float("nan")
    return choice, dts[choice] / float(scales[choice]), dts[choice]


def evaluate_orbits(df, pred, scales, rtols=None, do_spec=True, tol_km=POSITION_TOL_KM, mu=MU, n_jobs=8,
                    methods=METHODS, candidates=CANDIDATES, chunksize=4, progress=True):
    """Run the policy and the adaptive comparators on every row of `df` (pred aligned with df by position).
    rtols: {(method, "cal"|"eqr"): rtol | None}. Returns a DataFrame indexed like df's row_id column."""
    rtols = rtols or {}
    states = df[STATE_COLS].to_numpy(dtype=float)
    horizons = df["sim_time"].to_numpy(dtype=float)
    rids = df["row_id"].to_numpy() if "row_id" in df.columns else np.arange(len(df))
    jobs, choice_rec = [], []
    for i in range(len(df)):
        integ, raw, used = policy_choice(pred, scales, i, candidates)
        choice_rec.append({"row_id": int(rids[i]), "ml_integrator": integ, "ml_raw_dt": raw, "ml_dt": used,
                           "ml_scale": (used / raw) if integ else float("nan")})
        jobs.append((int(rids[i]), states[i], float(horizons[i]), integ, used, tol_km, mu, rtols, do_spec, methods))
    t0 = time.perf_counter()
    results = []
    with Pool(processes=n_jobs) as pool:
        for k, r in enumerate(pool.imap_unordered(_orbit_job, jobs, chunksize=chunksize)):
            results.append(r)
            if progress and (k + 1) % 250 == 0:
                print(f"  evaluate_orbits: {k + 1}/{len(jobs)} done, {time.perf_counter() - t0:.0f}s", flush=True)
    res = pd.DataFrame(results)
    ch = pd.DataFrame(choice_rec)
    return ch.merge(res, on="row_id", how="left")


def candidate_raw_predictions(df, pred, candidates=CANDIDATES):
    """Raw (unscaled) predicted dt and feasibility per candidate integrator as columns, aligned with df."""
    out = pd.DataFrame({"row_id": df["row_id"].to_numpy() if "row_id" in df.columns else np.arange(len(df))})
    for k in candidates:
        out[f"raw_dt_{k}"] = pred[k]["dt"]
        out[f"feasible_{k}"] = pred[k]["feasible"]
    return out
