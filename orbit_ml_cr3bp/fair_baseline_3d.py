"""
Matched-accuracy comparison: ML-recommended fixed-step integration vs adaptive
scipy solvers (RK45, DOP853) tuned to reach the SAME final position error.

Why this exists: modeling_3d.compare_rkf45_3d runs RK45 at a fixed rtol=1e-6 /
atol=1e-9 that has no relation to the accuracy the ML-recommended fixed step
actually achieves, so its eval-count comparison is not apples-to-apples (a
tolerance looser or tighter than the ML run's own error makes the adaptive
solver look artificially expensive or cheap). Here, for each orbit, the
adaptive solver's rtol is bisected until its final position error at sim_time
matches (is <=) the ML run's own final position error, and only then are
right-hand-side evaluations and wall-clock time compared.

Limitations (read before citing results):
  - The adaptive baselines are scipy's solve_ivp implementations, NOT
    numba-compiled like the fixed-step integrators. Wall-clock comparisons
    therefore mix algorithmic cost with Python/scipy overhead and are biased
    against the adaptive solvers on speed; RHS-evaluation counts (nfev vs
    steps * evals_per_step) are the implementation-independent comparison and
    should be the headline metric. A fully like-for-like wall-clock test would
    need an adaptive RK45/DOP853 written in numba, which is not implemented.
  - Accuracy is matched on FINAL POSITION ONLY (same metric as the
    accuracy-based label search), not on the whole trajectory or on velocity.
  - Bisection assumes error is roughly monotone in rtol; adaptive step control
    is not perfectly monotone, so treat the matched rtol as approximate.
  - The "truth" is a DOP853 run at rtol=atol=1e-13; error is measured against
    that, so errors near ~1e-9 km or below are not meaningful.

UPDATED cost accounting (read before citing results):
  - The "_spec" adaptive baseline tunes rtol PER ORBIT using the known truth
    (an oracle no real user has), i.e. a LOWER BOUND on adaptive cost. The new
    "_cal" baseline fixes ONE rtol per method from a separate calibration set
    with the same binomial fixed-sequence test used to certify the ML safety
    scale (calibrate_adaptive_rtol), i.e. the same "violation rate <= alpha with
    confidence 1-delta" standard the ML gets. Report both.
  - ML cost now includes the FTLE feature: ml_feature_steps is the number of
    RK4 steps of the 42-variable variational system, ml_feature_evals_equiv
    converts them to plain-RK4-evaluation equivalents using the measured
    per-step wall-time ratio of the compiled kernels (measure_cost_ratios), and
    ml_total_evals_equiv = ml_evals + ml_feature_evals_equiv. ml_total_wall_time
    = feature + prediction + integration wall time. The conversion is a
    measured wall-time ratio, not a flop count.
  - The ML recommendation is restricted to candidate_integrators (default
    RK4 and leapfrog; Euler fails ~70% of orbits and is kept only as a
    negative control elsewhere).
"""
from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd

from scipy.stats import binom

from .audit_3d import ltt_choose, LTT_METHOD
from .config import MU, POSITION_TOL_KM, du_km_for_mu
from .physics_3d import (
    INTEGRATORS_3D,
    EVALS_PER_STEP_3D,
    cr3bp_derivatives_3d,
    integrate_endpoint_3d,
    reference_trajectory_dense_3d,
    compute_ftle_3d_variational_details,
    step_rk4_3d,
)

TRUTH_RTOL = 1e-13
TRUTH_ATOL = 1e-13


def _final_pos_error_km(prev, curr, t_prev, t_actual, sim_time, ref_pos, du_km):
    if t_actual > sim_time and t_actual > t_prev:
        alpha = (sim_time - t_prev) / (t_actual - t_prev)
        final_pos = np.asarray(prev[:3]) + alpha * (np.asarray(curr[:3]) - np.asarray(prev[:3]))
    else:
        final_pos = np.asarray(curr[:3])
    return float(np.linalg.norm(final_pos - ref_pos)) * du_km


# An adaptive solve that needs more than this many right-hand-side evaluations is abandoned (about a minute with the Python
# RHS). Orbits that pass very close to a body can need tens of millions at tight rtol, which stalled the comparison for hours.
# _run_adaptive then returns None: a violation during rtol calibration, and the orbit is skipped (n_skipped) in the comparison.
MAX_ADAPTIVE_NFEV = 2_000_000
ADAPTIVE_CAP_HITS = {"n": 0}


class _NfevCap(Exception):
    pass


def _run_adaptive(state, sim_time, ref_pos, rtol, mu, method, atol_ratio=1e-3):
    from scipy.integrate import solve_ivp

    count = [0]

    def rhs(t, y):
        count[0] += 1
        if count[0] > MAX_ADAPTIVE_NFEV:
            raise _NfevCap()
        return cr3bp_derivatives_3d(y.tolist(), mu)

    atol = rtol * atol_ratio
    t0 = time.perf_counter()
    try:
        sol = solve_ivp(rhs, (0.0, sim_time), state, method=method, rtol=rtol, atol=atol)
    except _NfevCap:
        ADAPTIVE_CAP_HITS["n"] += 1
        return None
    wall = time.perf_counter() - t0
    if not sol.success or not np.all(np.isfinite(sol.y[:, -1])):
        return None
    err = float(np.linalg.norm(sol.y[:3, -1] - ref_pos)) * du_km_for_mu(mu)
    return {"rtol": rtol, "nfev": int(sol.nfev), "wall_time": wall, "err_km": err}


def scipy_matched_accuracy(
    state,
    sim_time: float,
    ref_pos: np.ndarray,
    target_err_km: float,
    mu: float = MU,
    method: str = "RK45",
    rtol_tight: float = 1e-13,
    rtol_loose: float = 1e-1,
    max_iter: int = 30,
    atol_ratio: float = 1e-3,
):
    """
    Find (approximately) the loosest rtol for which `method` reaches
    final-position error <= target_err_km, and return its cost at that rtol.
    Returns None if even rtol_tight cannot reach the target (or a solve
    fails). Log-scale bisection between rtol_tight (assumed accurate) and
    rtol_loose (assumed too coarse); if rtol_loose already meets the target,
    it is used directly.
    """

    def run(rtol):
        return _run_adaptive(state, sim_time, ref_pos, rtol, mu, method, atol_ratio)

    res_loose = run(rtol_loose)
    if res_loose is not None and res_loose["err_km"] <= target_err_km:
        return res_loose

    res_tight = run(rtol_tight)
    if res_tight is None or res_tight["err_km"] > target_err_km:
        return None

    lo, hi = math.log10(rtol_tight), math.log10(rtol_loose)
    best = res_tight
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        res = run(10.0 ** mid)
        if res is not None and res["err_km"] <= target_err_km:
            best = res
            lo = mid
        else:
            hi = mid
        if hi - lo < 0.02:
            break
    return best


RTOL_GRID_CAL = tuple(10.0 ** e for e in np.arange(-12.0, -1.9, 0.5))


def calibrate_adaptive_rtol(data_cal, mu: float = MU, method: str = "RK45", tol_km: float | None = None,
                            alpha: float = 0.05, delta: float = 0.1, rtol_grid=RTOL_GRID_CAL,
                            max_rows: int = 300, random_state: int = 42, atol_ratio: float = 1e-3,
                            method_ltt: str = LTT_METHOD):
    """
    Choose ONE rtol for an adaptive solver without per-orbit knowledge of the
    truth: reject H_rtol "violation rate > alpha" when
    P(Binomial(n, alpha) <= k) <= threshold, with the same selection rule as
    audit_3d.calibrate_scale_ltt (Bonferroni over the rtol grid by default; the
    loosest certified rtol), whose exchangeability caveat applies equally here. A run that fails or exceeds
    tol_km counts as a violation. data_cal must be held-out rows with
    ftle_nan/reference_divergent already removed. Returns a dict; chosen_rtol is
    None if even the tightest rtol is not certified.
    """
    if tol_km is None:
        tol_km = POSITION_TOL_KM
    n_take = min(max_rows, len(data_cal))
    sample = data_cal.sample(n=n_take, random_state=random_state)
    items = []
    for _, row in sample.iterrows():
        state = [row["x"], row["y"], row["z"], row["vx"], row["vy"], row["vz"]]
        T = float(row["sim_time"])
        truth, reached = reference_trajectory_dense_3d(state, T, mu, rtol=TRUTH_RTOL, atol=TRUTH_ATOL)
        if reached:
            items.append((state, T, truth.sol(T)[:3]))
    chain = []
    ordered = sorted(rtol_grid)
    for rtol in ordered:
        k = 0
        hits0 = ADAPTIVE_CAP_HITS["n"]
        for state, T, ref_pos in items:
            res = _run_adaptive(state, T, ref_pos, rtol, mu, method, atol_ratio)
            if res is None or res["err_km"] > tol_km:
                k += 1
        n = len(items)
        if n == 0:
            break
        p = float(binom.cdf(k, n, alpha))
        chain.append({"rtol": float(rtol), "n": n, "violations": k, "violation_rate": k / n, "p_value": p,
                      "capped": ADAPTIVE_CAP_HITS["n"] - hits0})
        if method_ltt == "fixed_sequence" and p > delta:
            break
    chosen, flags, thr = ltt_choose([c["rtol"] for c in chain], [c["p_value"] for c in chain], delta, method_ltt,
                                    m=len(ordered))
    for c, f in zip(chain, flags):
        c["certified"] = f
    return {"method": method, "chosen_rtol": chosen, "alpha": alpha, "delta": delta, "ltt_method": method_ltt,
            "p_threshold": thr, "n_cal": len(items), "tol_km": tol_km, "chain": chain}


_COST_RATIO_CACHE = {}


def measure_cost_ratios(mu: float = MU, repeats: int = 5):
    """
    Wall-time per RK4 step of the 42-variable variational system (the FTLE
    feature) divided by wall time per RK4 step of the plain 6-variable system,
    measured on a benign orbit with the compiled kernels (a warm-up call is
    made first so JIT compilation is excluded). If numba is not installed the
    ratio reflects pure-Python overhead and is not representative. Cached per mu.
    """
    if mu in _COST_RATIO_CACHE:
        return _COST_RATIO_CACHE[mu]
    state = [0.4878 + 0.03, 0.866 + 0.01, 0.01, 0.01, -0.02, 0.005]
    T = 1.0
    compute_ftle_3d_variational_details(state, mu, T)
    integrate_endpoint_3d(step_rk4_3d, state, 0.01, T, mu)
    t_var, steps_var, t_rk4 = [], 1, []
    for _ in range(repeats):
        t0 = time.perf_counter()
        d = compute_ftle_3d_variational_details(state, mu, T)
        t_var.append(time.perf_counter() - t0)
        steps_var = max(int(d["steps"]), 1)
        t0 = time.perf_counter()
        integrate_endpoint_3d(step_rk4_3d, state, 0.01, T, mu)
        t_rk4.append(time.perf_counter() - t0)
    per_var = min(t_var) / steps_var
    per_rk4 = min(t_rk4) / math.ceil(T / 0.01)
    ratio = per_var / per_rk4 if per_rk4 > 0 else float("nan")
    out = {"variational_step_over_rk4_step": float(ratio), "sec_per_variational_step": float(per_var),
           "sec_per_rk4_step": float(per_rk4)}
    _COST_RATIO_CACHE[mu] = out
    return out


CHEAP_FEATURE_EVALS_EQUIV = 1.0   # jacobi, r1, r2, ratios, c - c_L1: one acceleration-sized computation


def measure_prediction_cost(models: dict, sample: pd.DataFrame, integrators=("rk4", "leapfrog"),
                            mu: float = MU, n_rows: int = 1000, repeats: int = 3):
    """
    Amortized (batched) model-inference cost per orbit, in plain-RK4-RHS-evaluation equivalents:
    wall time of the candidate integrators' classifier + regressor predict calls on up to n_rows
    rows, divided by the rows and by the measured wall time of one compiled RHS evaluation
    (measure_cost_ratios). Like the FTLE conversion this is a measured wall-time ratio, not a flop
    count, and it depends on the model library (sklearn trees) as much as on the model.
    """
    ratios = measure_cost_ratios(mu)
    sec_per_rhs = ratios["sec_per_rk4_step"] / EVALS_PER_STEP_3D["rk4"]
    x = sample[models["feature_columns"]].to_numpy(dtype=float)[:n_rows]
    for k in integrators:
        models[k]["classifier"].predict(x[:5])
        if models[k]["regressor"] is not None:
            models[k]["regressor"].predict(x[:5])
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        for k in integrators:
            models[k]["classifier"].predict(x)
            if models[k]["regressor"] is not None:
                models[k]["regressor"].predict(x)
        best = min(best, time.perf_counter() - t0)
    per_orbit = best / max(len(x), 1)
    return {"sec_per_orbit": float(per_orbit), "evals_equiv": float(per_orbit / sec_per_rhs),
            "sec_per_rhs": float(sec_per_rhs), "n_rows": int(len(x))}


def matched_accuracy_baseline_3d(
    data: pd.DataFrame,
    models: dict,
    mu: float = MU,
    num_samples: int = 50,
    random_state: int = 42,
    methods: tuple = ("RK45", "DOP853"),
    default_sim_time: float = 20.0,
    tol_km: float | None = None,
    dt_scale=None,
    match: str = "ml_error",
    candidate_integrators: tuple = ("rk4", "leapfrog"),
    adaptive_calibrated: dict | None = None,
    measure_feature_cost: bool = True,
    charge_prediction: bool = False,
    global_dt: dict | None = None,
    adaptive_spec_cache: dict | None = None,
):
    """
    dt_scale: None (use predicted dt as is), a float, or a dict
    {integrator: float | None} multiplying each integrator's predicted dt
    before it is used (e.g. the outcome-calibrated scales from
    audit_3d.calibrate_scale_ltt); a None entry removes that integrator from
    the recommendation. match: "ml_error" (adaptive solver tuned to the ML
    run's own final error, original behavior), "tolerance" (tuned to the
    spec tol_km -- the cost of simply meeting the requirement), or "both"
    (the latter adds columns/summary keys with a "_spec" infix).
    adaptive_calibrated: {method: rtol | None} from calibrate_adaptive_rtol;
    adds columns/summary keys with a "_cal" infix (a fixed, calibrated rtol per
    method instead of per-orbit oracle tuning). candidate_integrators: which
    integrators the ML may recommend. measure_feature_cost: time the FTLE
    feature and the model prediction for each sampled orbit and add total-cost
    columns (see module docstring).

    For each sampled row of `data` (should be held-out, ftle_nan/
    reference_divergent rows already dropped): take the model's
    cost-normalized recommended (integrator, dt), integrate to sim_time,
    measure its own final position error vs a tight DOP853 truth, then find
    each adaptive method's cost at matched error.

    Returns (per_orbit_df, summary_dict). Rows are skipped (not counted)
    when: no integrator is predicted feasible, the truth reference collides/
    ejects before sim_time, the ML-recommended run itself diverges, or an
    adaptive method cannot reach the target error. The adaptive target is the
    ML run's OWN error (matched accuracy), so if the ML run overshoots tol_km
    (frac_ml_meets_tol < 1) the adaptive solvers are being asked for a
    looser-than-spec accuracy; report frac_ml_meets_tol next to any cost ratio.

    H1' additions (all default off, so the original behaviour is unchanged):
      - If "ftle_3d" is not among models["feature_columns"] the FTLE integration is skipped and the
        feature cost is CHEAP_FEATURE_EVALS_EQUIV instead.
      - charge_prediction: add ml_predict_evals_equiv (measure_prediction_cost) and
        ml_total_evals_equiv_pred = ml_total_evals_equiv + ml_predict_evals_equiv, and summarize cost
        ratios on that basis (keys with "totalpred").
      - global_dt: {integrator: dt | None}; every kept orbit is also run at that one fixed step
        (columns global_<integ>_*), the no-model baseline a user would otherwise have to pick.
      - adaptive_spec_cache: dict reused across calls on the SAME sampled orbits so the model-independent
        per-orbit "_spec" adaptive solves are done once.
    `n_skipped` in the summary
    reports how many -- if that is large, the comparison is only over the
    easy subset and should be reported as such.
    """
    from .modeling_3d import predict_hurdle, cost_normalized_choice

    if tol_km is None:
        tol_km = POSITION_TOL_KM
    du_km = du_km_for_mu(mu)
    sample = data.sample(n=min(num_samples, len(data)), random_state=random_state)
    feature_cols = models["feature_columns"]
    rows = []
    n_skipped = 0
    cost_ratio = measure_cost_ratios(mu)["variational_step_over_rk4_step"] if measure_feature_cost else float("nan")
    adaptive_calibrated = adaptive_calibrated or {}
    uses_ftle = "ftle_3d" in feature_cols
    pred_cost = measure_prediction_cost(models, data, candidate_integrators, mu) if charge_prediction else None

    for idx, row in sample.iterrows():
        state = [row["x"], row["y"], row["z"], row["vx"], row["vy"], row["vz"]]
        sim_time = float(row["sim_time"]) if np.isfinite(row.get("sim_time", np.nan)) else default_sim_time

        t0 = time.perf_counter()
        pred = predict_hurdle(models, pd.DataFrame([row[feature_cols].to_dict()]))
        predict_wall = time.perf_counter() - t0
        dts = {}
        for i in candidate_integrators:
            if not pred[i]["feasible"][0]:
                continue
            sc = dt_scale.get(i, 1.0) if isinstance(dt_scale, dict) else (1.0 if dt_scale is None else float(dt_scale))
            if sc is None:
                continue
            dts[i] = float(pred[i]["dt"][0]) * float(sc)
        choice, _ = cost_normalized_choice(dts)
        if choice is None:
            n_skipped += 1
            continue
        dt_pred = float(dts[choice])
        if not np.isfinite(dt_pred) or dt_pred <= 0:
            n_skipped += 1
            continue

        truth, reached = reference_trajectory_dense_3d(state, sim_time, mu, rtol=TRUTH_RTOL, atol=TRUTH_ATOL)
        if not reached:
            n_skipped += 1
            continue
        ref_pos = truth.sol(sim_time)[:3]

        t0 = time.perf_counter()
        ok, prev, curr, t_prev, t_actual, _reason = integrate_endpoint_3d(
            INTEGRATORS_3D[choice], state, dt_pred, sim_time, mu, 4_000_000)
        ml_wall = time.perf_counter() - t0
        if not ok:
            n_skipped += 1
            continue
        ml_err_km = _final_pos_error_km(prev, curr, t_prev, t_actual, sim_time, ref_pos, du_km)
        n_steps = int(math.ceil(t_actual / dt_pred - 1e-9))
        record = {
            "orbit_idx": idx, "sim_time": sim_time, "ml_integrator": choice, "ml_dt": dt_pred,
            "ml_scale": dt_pred / float(pred[choice]["dt"][0]),
            "ml_err_km": ml_err_km, "ml_meets_tol": bool(ml_err_km <= tol_km),
            "ml_evals": n_steps * EVALS_PER_STEP_3D[choice],
            "ml_wall_time": ml_wall,
            "ml_predict_wall_time": predict_wall,
        }
        if measure_feature_cost:
            if uses_ftle:
                t0 = time.perf_counter()
                fd = compute_ftle_3d_variational_details(state, mu, sim_time)
                feat_wall = time.perf_counter() - t0
                feat_steps = fd["steps"]
                feat_equiv = feat_steps * cost_ratio * EVALS_PER_STEP_3D["rk4"]
            else:
                feat_wall, feat_steps, feat_equiv = 0.0, 0, CHEAP_FEATURE_EVALS_EQUIV
            record.update({
                "ml_feature_wall_time": feat_wall,
                "ml_feature_steps": feat_steps,
                "ml_feature_evals_equiv": feat_equiv,
                "ml_total_evals_equiv": record["ml_evals"] + feat_equiv,
                "ml_total_wall_time": feat_wall + predict_wall + ml_wall,
            })
            if pred_cost is not None:
                record["ml_predict_evals_equiv"] = pred_cost["evals_equiv"]
                record["ml_total_evals_equiv_pred"] = record["ml_total_evals_equiv"] + pred_cost["evals_equiv"]

        for gi, gdt in (global_dt or {}).items():
            if gdt is None:
                continue
            ok_g, prev_g, curr_g, tp_g, ta_g, _rg = integrate_endpoint_3d(
                INTEGRATORS_3D[gi], state, float(gdt), sim_time, mu, 4_000_000)
            record[f"global_{gi}_dt"] = float(gdt)
            record[f"global_{gi}_evals"] = int(math.ceil(sim_time / float(gdt) - 1e-9)) * EVALS_PER_STEP_3D[gi]
            gerr = _final_pos_error_km(prev_g, curr_g, tp_g, ta_g, sim_time, ref_pos, du_km) if ok_g else float("nan")
            record[f"global_{gi}_err_km"] = gerr
            record[f"global_{gi}_meets_tol"] = bool(ok_g and gerr <= tol_km)

        targets = {}
        if match in ("ml_error", "both"):
            targets[""] = max(ml_err_km, 1e-9)
        if match in ("tolerance", "both"):
            targets["_spec"] = tol_km
        if not targets:
            raise ValueError("match must be 'ml_error', 'tolerance' or 'both'")

        all_ok = True
        for method in methods:
            for suffix, target in targets.items():
                cache_key = (idx, method, suffix)
                use_cache = adaptive_spec_cache is not None and suffix == "_spec"
                if use_cache and cache_key in adaptive_spec_cache:
                    res = adaptive_spec_cache[cache_key]
                else:
                    res = scipy_matched_accuracy(state, sim_time, ref_pos, target_err_km=target,
                                                  mu=mu, method=method)
                    if use_cache:
                        adaptive_spec_cache[cache_key] = res
                if res is None:
                    all_ok = False
                    break
                record[f"{method}{suffix}_rtol"] = res["rtol"]
                record[f"{method}{suffix}_evals"] = res["nfev"]
                record[f"{method}{suffix}_wall_time"] = res["wall_time"]
                record[f"{method}{suffix}_err_km"] = res["err_km"]
            if not all_ok:
                break
        if not all_ok:
            n_skipped += 1
            continue

        for method in methods:
            rtol_cal = adaptive_calibrated.get(method)
            if rtol_cal is None:
                continue
            res = _run_adaptive(state, sim_time, ref_pos, float(rtol_cal), mu, method)
            if res is None:
                record[f"{method}_cal_rtol"] = float(rtol_cal)
                record[f"{method}_cal_evals"] = np.nan
                record[f"{method}_cal_wall_time"] = np.nan
                record[f"{method}_cal_err_km"] = np.nan
                record[f"{method}_cal_meets_tol"] = False
            else:
                record[f"{method}_cal_rtol"] = res["rtol"]
                record[f"{method}_cal_evals"] = res["nfev"]
                record[f"{method}_cal_wall_time"] = res["wall_time"]
                record[f"{method}_cal_err_km"] = res["err_km"]
                record[f"{method}_cal_meets_tol"] = bool(res["err_km"] <= tol_km)
        rows.append(record)

    df = pd.DataFrame(rows)
    summary = {"n_used": len(df), "n_skipped": n_skipped, "adaptive_nfev_cap": MAX_ADAPTIVE_NFEV,
               "adaptive_cap_hits_total": ADAPTIVE_CAP_HITS["n"]}
    if len(df) == 0:
        return df, summary
    summary["tol_km"] = tol_km
    summary["candidate_integrators"] = list(candidate_integrators)
    summary["frac_ml_meets_tol"] = float(df["ml_meets_tol"].mean())
    summary["mean_ml_evals"] = float(df["ml_evals"].mean())
    summary["mean_ml_wall_time"] = float(df["ml_wall_time"].mean())
    summary["n_ml_meets_tol"] = int(df["ml_meets_tol"].sum())
    has_total = "ml_total_evals_equiv" in df.columns
    if has_total:
        summary["mean_ml_total_evals_equiv"] = float(df["ml_total_evals_equiv"].mean())
        summary["mean_ml_total_wall_time"] = float(df["ml_total_wall_time"].mean())
        summary["variational_step_over_rk4_step"] = float(cost_ratio)
    for method in methods:
        for suffix in ("", "_spec", "_cal"):
            col = f"{method}{suffix}_evals"
            if col not in df.columns:
                continue
            if suffix == "":
                d = df
            elif suffix == "_spec":
                d = df[df["ml_meets_tol"]]
            else:
                d = df[df["ml_meets_tol"] & df[col].notna()]
                summary[f"frac_{method}_cal_meets_tol"] = float(df[f"{method}_cal_meets_tol"].mean())
            if len(d) == 0:
                continue
            summary[f"mean_{method}{suffix}_evals"] = float(d[col].mean())
            summary[f"mean_{method}{suffix}_wall_time"] = float(d[f"{method}{suffix}_wall_time"].mean())
            ratio = d["ml_evals"] / d[col]
            summary[f"median_ml_over_{method}{suffix}_evals"] = float(ratio.median())
            summary[f"frac_orbits_ml_cheaper_than_{method}{suffix}"] = float((ratio < 1.0).mean())
            if has_total:
                ratio_t = d["ml_total_evals_equiv"] / d[col]
                summary[f"median_ml_total_over_{method}{suffix}_evals"] = float(ratio_t.median())
                summary[f"frac_orbits_ml_total_cheaper_than_{method}{suffix}"] = float((ratio_t < 1.0).mean())
                if charge_prediction:
                    ratio_p = d["ml_total_evals_equiv_pred"] / d[col]
                    summary[f"median_ml_totalpred_over_{method}{suffix}_evals"] = float(ratio_p.median())
                    summary[f"frac_orbits_ml_totalpred_cheaper_than_{method}{suffix}"] = float((ratio_p < 1.0).mean())
    if charge_prediction and pred_cost is not None:
        summary["prediction_evals_equiv"] = pred_cost["evals_equiv"]
    for gi in (global_dt or {}):
        col = f"global_{gi}_evals"
        if col not in df.columns:
            continue
        summary[f"frac_GLOBAL_{gi}_meets_tol"] = float(df[f"global_{gi}_meets_tol"].mean())
        summary[f"global_{gi}_dt"] = float(df[f"global_{gi}_dt"].iloc[0])
        d = df[df["ml_meets_tol"]]
        if len(d) == 0:
            continue
        summary[f"mean_global_{gi}_evals"] = float(d[col].mean())
        base = "ml_total_evals_equiv_pred" if (charge_prediction and "ml_total_evals_equiv_pred" in d.columns) \
            else ("ml_total_evals_equiv" if has_total else "ml_evals")
        ratio_g = d[base] / d[col]
        summary[f"median_{base}_over_GLOBAL_{gi}_evals"] = float(ratio_g.median())
        summary[f"frac_orbits_ml_totalpred_cheaper_than_GLOBAL_{gi}"] = float((ratio_g < 1.0).mean())
    return df, summary