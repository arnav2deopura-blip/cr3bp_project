"""
Position-error replacements for the Jacobi-drift audits in modeling_3d.py /
validation_3d.py, plus an outcome-calibrated step-size scale.

All "stable" decisions here mean: final position at sim_time within tol_km of
a DOP853 reference (config.POSITION_TOL_KM by default), the same criterion the
label_mode="accuracy" data uses.

calibrate_scale_ltt: Learn-then-Test style certification with binomial tail
p-values (verify the construction against the original reference before
citing it). For a candidate scale s multiplying the predicted dt,
H_s: P(violation | predicted feasible) > alpha is rejected when
P(Binomial(n, alpha) <= k) <= threshold. A violation is an error above
tol_km, a collision/escape/non-finite run, or a run stopped by the step cap.
LTT_METHOD = "bonferroni" (default): every scale on the grid is tested at
delta / m (m = grid size) and the LARGEST rejected scale is chosen; with
probability >= 1 - delta every rejected scale has risk <= alpha, with no
assumption that risk is monotone in the scale. method="fixed_sequence"
(the earlier rule) tests from the smallest scale up at delta and stops at the
first non-rejection; it is more powerful if risk is monotone but a single bad
smallest scale (for instance step-cap failures from a tiny dt over a long
horizon) then certifies nothing. Both are valid only PROVIDED the calibration
orbits are exchangeable with deployment orbits: calibrate on rows drawn the
same way as the evaluation rows and never used for training (run_rigor_suite
`gate`/`final` use a random split for exactly this reason; a grouped/regional
split breaks exchangeability with random locked rows).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binom, beta
from sklearn.model_selection import train_test_split

from .config import MU, POSITION_TOL_KM, du_km_for_mu
from .physics_3d import (
    INTEGRATORS_3D,
    EVALS_PER_STEP_3D,
    integrate_endpoint_3d,
    reference_trajectory_dense_3d,
    add_orbit_features_3d,
    REASON_STEP_CAP,
)
from .search_3d import (find_dt_for_integrator_accuracy_3d, find_dt_for_integrator_accuracy_monotone_3d,
                        _dt_upper_bound_3d)

INTEGRATOR_NAMES = ("euler", "rk4", "leapfrog")
STATE_COLS = ["x", "y", "z", "vx", "vy", "vz"]
LTT_SCALE_GRID = (0.02, 0.025, 0.03, 0.04, 0.05, 0.065, 0.08, 0.1, 0.125, 0.16, 0.2, 0.25, 0.3, 0.4, 0.5,
                  0.6, 0.8, 1.0)
DT_MIN = 0.0001
AUDIT_MAX_STEPS = 4_000_000
LTT_METHOD = "bonferroni"


def ltt_choose(values, p_values, delta, method=LTT_METHOD, m=None):
    """
    values: grid values ordered safest-first (ascending scale / dt / rtol); p_values: same order.
    Returns (chosen or None, certified flags, p-value threshold used).
    bonferroni: every value tested at delta / m, chosen = the largest certified value.
    fixed_sequence: tested in order at delta, stopping at the first non-rejection; chosen = the last rejected.
    """
    m = m or len(values)
    if method == "fixed_sequence":
        flags, stopped = [], False
        for p in p_values:
            ok = (not stopped) and p <= delta
            flags.append(bool(ok))
            stopped = stopped or not ok
        chosen = None
        for v, f in zip(values, flags):
            if f:
                chosen = float(v)
        return chosen, flags, float(delta)
    if method != "bonferroni":
        raise ValueError("method must be 'bonferroni' or 'fixed_sequence'")
    thr = float(delta) / m
    flags = [bool(p <= thr) for p in p_values]
    cert = [float(v) for v, f in zip(values, flags) if f]
    return (max(cert) if cert else None), flags, thr


class ReferenceCache:
    """DOP853 references keyed by row position; recomputed if the state or horizon changes."""

    def __init__(self, mu: float = MU):
        self.mu = mu
        self._d = {}

    def get(self, key, state, sim_time):
        s = tuple(float(v) for v in state)
        T = float(sim_time)
        item = self._d.get(key)
        if item is None or item[0] != s or item[1] != T:
            sol, reached = reference_trajectory_dense_3d(list(s), T, self.mu)
            item = (s, T, sol, reached)
            self._d[key] = item
        return item[2], item[3]


def final_position_error_km(integrator, state, dt, sim_time, ref_pos, mu=MU, max_steps=AUDIT_MAX_STEPS):
    """
    max_steps is raised from integrate_endpoint_3d's default 250,000 to
    AUDIT_MAX_STEPS: after a safety scale (e.g. 0.02) the step can drop below
    sim_time/250000, and a run stopped by the step cap used to be counted as an
    accuracy VIOLATION even though a smaller step is more accurate. The label
    search (dt >= 1e-4) never needs more than 200,000 steps and is unchanged.
    """
    ok, prev, curr, t_prev, t_act, reason = integrate_endpoint_3d(
        INTEGRATORS_3D[integrator], state, dt, sim_time, mu, max_steps)
    if not ok:
        return None, reason
    if t_act > sim_time and t_act > t_prev:
        a = (sim_time - t_prev) / (t_act - t_prev)
        pos = np.asarray(prev[:3]) + a * (np.asarray(curr[:3]) - np.asarray(prev[:3]))
    else:
        pos = np.asarray(curr[:3])
    return float(np.linalg.norm(pos - np.asarray(ref_pos))) * du_km_for_mu(mu), 0


def clopper_pearson(k, n, conf=0.95):
    if n <= 0:
        return float("nan"), float("nan")
    a = 1.0 - conf
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def position_error_audit_3d(data, pred, tol_km=None, scale=1.0, mu=MU, cache=None,
                            integrators=INTEGRATOR_NAMES, rows=None):
    """
    Re-integrate every predicted-feasible row at scale * predicted dt and
    record the final position error. `pred` must come from predict_hurdle on
    exactly `data` (positional alignment); `rows` restricts to positions.
    scale: float or {integrator: float | None} (None skips that integrator).
    A run that collides/ejects/goes non-finite counts as a violation.
    """
    if tol_km is None:
        tol_km = POSITION_TOL_KM
    if cache is None:
        cache = ReferenceCache(mu)
    if rows is None:
        rows = np.arange(len(data))
    states = data[STATE_COLS].to_numpy(dtype=float)
    horizons = data["sim_time"].to_numpy(dtype=float)
    records = []
    for integ in integrators:
        s = scale.get(integ, 1.0) if isinstance(scale, dict) else scale
        if s is None:
            continue
        feas = pred[integ]["feasible"]
        dts = pred[integ]["dt"]
        for i in rows:
            if not feas[i] or not np.isfinite(dts[i]) or dts[i] <= 0:
                continue
            sol, reached = cache.get(i, states[i], horizons[i])
            if not reached:
                continue
            dt = float(dts[i]) * float(s)
            err, reason = final_position_error_km(integ, list(states[i]), dt, horizons[i],
                                                  sol.sol(horizons[i])[:3], mu)
            completed = err is not None
            records.append({
                "integrator": integ, "row": int(i), "dt": dt, "scale": float(s),
                "pos_err_km": float("nan") if err is None else err,
                "completed": completed, "reason": int(reason), "stable": bool(completed and err <= tol_km),
            })
    return pd.DataFrame(records, columns=["integrator", "row", "dt", "scale", "pos_err_km",
                                           "completed", "reason", "stable"])


def summarize_audit(audit_df):
    out = {}
    for integ, g in audit_df.groupby("integrator"):
        err = g.loc[g["completed"], "pos_err_km"]
        out[integ] = {
            "n": int(len(g)),
            "violation_rate": float((~g["stable"]).mean()),
            "completed_rate": float(g["completed"].mean()),
            "median_err_km": float(err.median()) if len(err) else float("nan"),
            "p90_err_km": float(err.quantile(0.9)) if len(err) else float("nan"),
            "max_err_km": float(err.max()) if len(err) else float("nan"),
        }
    return out


def scale_frontier_3d(data, pred, integrators=INTEGRATOR_NAMES, tol_km=None, scales=None, rows=None,
                      mu=MU, cache=None):
    """
    Violation rate and relative cost of EVERY scale in the grid (not just the
    first one the fixed-sequence test looks at), with the violations split into
    "exceeded tol" (the run finished but was off by more than tol_km) and
    "incomplete" (the run collided/escaped/hit the step cap). step_cost_multiplier
    = 1/scale is how many times more steps than the unscaled prediction.
    This is descriptive, not a certificate.
    """
    if scales is None:
        scales = LTT_SCALE_GRID
    if cache is None:
        cache = ReferenceCache(mu)
    records = []
    for integ in integrators:
        for sc in scales:
            a = position_error_audit_3d(data, pred, tol_km, scale=sc, mu=mu, cache=cache,
                                        integrators=(integ,), rows=rows)
            n = len(a)
            if n == 0:
                continue
            exceeded = int((a["completed"] & ~a["stable"]).sum())
            incomplete = int((~a["completed"]).sum())
            records.append({"integrator": integ, "scale": float(sc), "n": n,
                            "violation_rate": (exceeded + incomplete) / n,
                            "exceeded_tol_rate": exceeded / n, "incomplete_rate": incomplete / n,
                            "step_cap_rate": float((a["reason"] == REASON_STEP_CAP).mean()),
                            "step_cost_multiplier": 1.0 / sc})
    return pd.DataFrame(records)


def calibrate_scale_ltt(data, pred, integrator, tol_km=None, alpha=0.05, delta=0.1,
                        scales=LTT_SCALE_GRID, rows=None, mu=MU, cache=None, method=LTT_METHOD):
    chain = []
    ordered = sorted(scales)
    for s in ordered:
        a = position_error_audit_3d(data, pred, tol_km, scale=s, mu=mu, cache=cache,
                                    integrators=(integrator,), rows=rows)
        n = len(a)
        if n == 0:
            break
        k = int((~a["stable"]).sum())
        p = float(binom.cdf(k, n, alpha))
        chain.append({"scale": float(s), "n": n, "violations": k, "violation_rate": k / n, "p_value": p,
                      "incomplete": int((~a["completed"]).sum()),
                      "step_cap": int((a["reason"] == REASON_STEP_CAP).sum())})
        if method == "fixed_sequence" and p > delta:
            break
    chosen, flags, thr = ltt_choose([c["scale"] for c in chain], [c["p_value"] for c in chain], delta, method,
                                    m=len(ordered))
    for c, f in zip(chain, flags):
        c["certified"] = f
    return {"integrator": integrator, "chosen_scale": chosen, "alpha": alpha, "delta": delta, "method": method,
            "p_threshold": thr, "chain": chain}


GLOBAL_DT_GRID = tuple(1e-4 * 2 ** (k / 4) for k in range(50))   # 1e-4 .. ~0.49


def calibrate_global_dt_ltt(data, integrator, tol_km=None, alpha=0.05, delta=0.1,
                            dt_grid=GLOBAL_DT_GRID, rows=None, mu=MU, cache=None, method=LTT_METHOD):
    """
    ONE fixed step for every orbit (no model, no per-orbit information): the baseline a user
    without the recommender would have to pick. Same binomial test and selection rule as
    calibrate_scale_ltt (Bonferroni over the dt grid by default; the largest certified dt). Every row counts
    (no feasibility gate); a run that collides, escapes or hits the step cap is a violation.
    Rows whose DOP853 reference diverges are skipped, as everywhere else.
    """
    if tol_km is None:
        tol_km = POSITION_TOL_KM
    if cache is None:
        cache = ReferenceCache(mu)
    if rows is None:
        rows = np.arange(len(data))
    states = data[STATE_COLS].to_numpy(dtype=float)
    horizons = data["sim_time"].to_numpy(dtype=float)
    chain = []
    ordered = sorted(dt_grid)
    for dt in ordered:
        n = k = incomplete = 0
        for i in rows:
            sol, reached = cache.get(int(i), states[i], horizons[i])
            if not reached:
                continue
            err, _reason = final_position_error_km(integrator, list(states[i]), float(dt), horizons[i],
                                                   sol.sol(horizons[i])[:3], mu)
            n += 1
            if err is None:
                incomplete += 1
                k += 1
            elif err > tol_km:
                k += 1
        if n == 0:
            break
        p = float(binom.cdf(k, n, alpha))
        chain.append({"dt": float(dt), "n": n, "violations": k, "violation_rate": k / n, "p_value": p,
                      "incomplete": incomplete})
        if method == "fixed_sequence" and p > delta:
            break
    chosen, flags, thr = ltt_choose([c["dt"] for c in chain], [c["p_value"] for c in chain], delta, method,
                                    m=len(ordered))
    for c, f in zip(chain, flags):
        c["certified"] = f
    return {"integrator": integrator, "chosen_dt": chosen, "alpha": alpha, "delta": delta, "method": method,
            "p_threshold": thr, "chain": chain}


def fit_and_validate_scale_ltt(data_test, pred_test, tol_km=None, alpha=0.05, delta=0.1,
                               cal_frac=0.5, random_state=42, integrators=INTEGRATOR_NAMES,
                               scales=LTT_SCALE_GRID, mu=MU, max_rows=None):
    """
    Split the held-out rows into calibration / evaluation halves, certify a
    per-integrator scale on the calibration half, and measure the realized
    violation rate (with a Clopper-Pearson interval) on the untouched half,
    next to the unscaled (scale=1) rate. max_rows caps the number of rows used.
    """
    if tol_km is None:
        tol_km = POSITION_TOL_KM
    idx = np.arange(len(data_test))
    if max_rows is not None and max_rows < len(idx):
        idx = np.random.RandomState(random_state).choice(idx, size=max_rows, replace=False)
    cal_rows, eval_rows = train_test_split(idx, test_size=1.0 - cal_frac, random_state=random_state)
    cache = ReferenceCache(mu)
    report = {"alpha": alpha, "delta": delta, "tol_km": tol_km,
              "n_cal_rows": int(len(cal_rows)), "n_eval_rows": int(len(eval_rows)), "per_integrator": {}}
    for integ in integrators:
        cal = calibrate_scale_ltt(data_test, pred_test, integ, tol_km, alpha, delta, scales,
                                  rows=cal_rows, mu=mu, cache=cache)
        raw = position_error_audit_3d(data_test, pred_test, tol_km, 1.0, mu, cache, (integ,), eval_rows)
        entry = {"chosen_scale": cal["chosen_scale"], "calibration_chain": cal["chain"],
                 "eval_unscaled_n": int(len(raw)),
                 "eval_unscaled_violation_rate": float((~raw["stable"]).mean()) if len(raw) else float("nan")}
        if cal["chosen_scale"] is not None:
            sc = position_error_audit_3d(data_test, pred_test, tol_km, cal["chosen_scale"], mu, cache,
                                         (integ,), eval_rows)
            k, n = int((~sc["stable"]).sum()), int(len(sc))
            lo, hi = clopper_pearson(k, n)
            entry.update({"eval_scaled_n": n, "eval_scaled_violations": k,
                          "eval_scaled_violation_rate": k / n if n else float("nan"),
                          "eval_scaled_violation_ci95": [lo, hi],
                          "eval_scaled_meets_alpha": bool(n > 0 and k / n <= alpha),
                          "step_cost_multiplier": 1.0 / cal["chosen_scale"]})
        report["per_integrator"][integ] = entry
    return report


def tolerance_sensitivity_accuracy_3d(data, tolerances_km=(10.0, 30.0, 100.0, 300.0, 1000.0),
                                      num_samples=50, mu=MU, random_state=42,
                                      integrators=INTEGRATOR_NAMES):
    sample = np.random.RandomState(random_state).choice(len(data), size=min(num_samples, len(data)), replace=False)
    states = data[STATE_COLS].to_numpy(dtype=float)
    horizons = data["sim_time"].to_numpy(dtype=float)
    cache = ReferenceCache(mu)
    records = []
    for i in sample:
        sol, reached = cache.get(i, states[i], horizons[i])
        if not reached:
            continue
        for integ in integrators:
            for tol in tolerances_km:
                r = find_dt_for_integrator_accuracy_3d(integ, list(states[i]), horizons[i], mu, tol_km=tol,
                                                       reference=(sol, reached), return_details=True)
                records.append({"orbit_row": int(i), "integrator": integ, "tol_km": float(tol),
                                "dt": r["dt"], "censored": r["censored"], "success": bool(r["dt"] > DT_MIN)})
    return pd.DataFrame(records)


def selection_agreement(sens_df, base_tol_km):
    d = sens_df[sens_df["success"]].copy()
    d["score"] = d["dt"] / d["integrator"].map(EVALS_PER_STEP_3D)
    best = d.loc[d.groupby(["orbit_row", "tol_km"])["score"].idxmax()][["orbit_row", "tol_km", "integrator"]]
    base = best[best["tol_km"] == base_tol_km].set_index("orbit_row")["integrator"]
    rows = []
    for tol, g in best.groupby("tol_km"):
        j = g.set_index("orbit_row")["integrator"]
        common = j.index.intersection(base.index)
        rows.append({"tol_km": float(tol), "n": int(len(common)),
                     "agree_with_base": float((j[common] == base[common]).mean()) if len(common) else float("nan")})
    return pd.DataFrame(rows)


def dense_sweep_accuracy_3d(data, tol_km=None, integrator="rk4", num_samples=200, n_steps=150,
                            mu=MU, random_state=42):
    """
    Compare the halving-search label (search_dt) and the locally-monotone label
    (monotone_dt) with a brute-force dense linear sweep of dt (dense_dt), and
    flag orbits whose pass/fail pattern over the sweep is non-monotone (a pass
    at a larger dt after a fail at a smaller one). Default sample raised from
    20 to 200 orbits so the non-monotone fraction has a usable interval.
    """
    if tol_km is None:
        tol_km = POSITION_TOL_KM
    sample = np.random.RandomState(random_state).choice(len(data), size=min(num_samples, len(data)), replace=False)
    states = data[STATE_COLS].to_numpy(dtype=float)
    horizons = data["sim_time"].to_numpy(dtype=float)
    cache = ReferenceCache(mu)
    records = []
    for i in sample:
        sol, reached = cache.get(i, states[i], horizons[i])
        if not reached:
            continue
        state = list(states[i])
        ref_pos = sol.sol(horizons[i])[:3]
        top = max(_dt_upper_bound_3d(state, mu), 0.005)
        grid = np.linspace(DT_MIN, top, n_steps)
        ok = []
        for dt in grid:
            err, _ = final_position_error_km(integrator, state, float(dt), horizons[i], ref_pos, mu)
            ok.append(err is not None and err <= tol_km)
        ok = np.array(ok)
        dense_dt = float(grid[ok].max()) if ok.any() else DT_MIN
        search_dt = find_dt_for_integrator_accuracy_3d(integrator, state, horizons[i], mu, tol_km=tol_km,
                                                       reference=(sol, reached))
        monotone_dt = find_dt_for_integrator_accuracy_monotone_3d(integrator, state, horizons[i], mu, tol_km=tol_km,
                                                                  reference=(sol, reached))
        fails = np.where(~ok)[0]
        non_monotone = bool(len(fails) > 0 and ok[fails[0] + 1:].any())
        records.append({"orbit_row": int(i), "search_dt": search_dt, "monotone_dt": monotone_dt,
                        "dense_dt": dense_dt,
                        "relative_error": abs(search_dt - dense_dt) / max(dense_dt, 1e-9),
                        "monotone_relative_error": abs(monotone_dt - dense_dt) / max(dense_dt, 1e-9),
                        "non_monotone": non_monotone})
    return pd.DataFrame(records)


def summarize_dense_sweep(dn):
    n = int(len(dn))
    k = int(dn["non_monotone"].sum()) if n else 0
    lo, hi = clopper_pearson(k, n)
    out = {"n": n, "frac_non_monotone": k / n if n else float("nan"), "frac_non_monotone_ci95": [lo, hi]}
    for col in ("relative_error", "monotone_relative_error"):
        if col in dn.columns and n:
            out[f"median_{col}"] = float(dn[col].median())
    return out


def calibration_stability_3d(data_test, pred_test, tol_km=None, alpha=0.05, delta=0.1, n_repeats=10,
                             cal_size=None, random_state=0, integrators=INTEGRATOR_NAMES,
                             scales=LTT_SCALE_GRID, mu=MU):
    """
    Re-run the scale calibration on n_repeats random calibration subsets of
    the held-out rows and report the chosen scale each time. The scale swung
    between runs (leapfrog 0.05 vs 0.2) when calibrated on only 300-400 rows
    on a coarse grid; a stable calibration should give (near-)identical scales
    across repeats. Returns (per_repeat_df, summary_df).
    """
    cache = ReferenceCache(mu)
    n = len(data_test)
    cal_size = min(cal_size or n // 2, n)
    rng = np.random.RandomState(random_state)
    records = []
    for r in range(n_repeats):
        rows = np.sort(rng.choice(n, size=cal_size, replace=False))
        for integ in integrators:
            res = calibrate_scale_ltt(data_test, pred_test, integ, tol_km, alpha, delta, scales,
                                      rows=rows, mu=mu, cache=cache)
            records.append({"repeat": r, "integrator": integ, "n_cal_rows": int(cal_size),
                            "chosen_scale": np.nan if res["chosen_scale"] is None else res["chosen_scale"]})
    per = pd.DataFrame(records)
    summ = per.groupby("integrator")["chosen_scale"].agg(
        frac_certified=lambda v: float(v.notna().mean()),
        min_scale="min", median_scale="median", max_scale="max",
        n_distinct=lambda v: int(v.dropna().nunique())).reset_index()
    return per, summ


def halo_validation_accuracy_3d(models, mu=MU, points=("L1", "L2"), az_values=(0.01, 0.02, 0.03, 0.04),
                                tol_km=None, scale=1.0, horizon_periods=1.0, min_horizon=1.0):
    """
    Differentially-corrected halo orbits (richardson.py, verified Az range
    only) integrated for horizon_periods halo periods (at least min_horizon
    TU) at scale * predicted dt, scored by final position error.
    """
    from .modeling_3d import predict_hurdle
    from .richardson import differential_corrector_halo_3d

    if tol_km is None:
        tol_km = POSITION_TOL_KM
    records = []
    for point in points:
        for az in az_values:
            try:
                res = differential_corrector_halo_3d(point, az, mu=mu)
            except Exception as e:
                print(f"Halo failed for {point} Az={az}: {e}")
                continue
            state0 = res["state0"]
            T = max(horizon_periods * res["T_full"], min_horizon)
            df = pd.DataFrame([{**dict(zip(STATE_COLS, state0)), "sim_time": T}])
            df = add_orbit_features_3d(df, mu, use_variational_ftle=True, ftle_T=T)
            pred = predict_hurdle(models, df)
            sol, reached = reference_trajectory_dense_3d(state0, T, mu)
            if not reached:
                continue
            ref_pos = sol.sol(T)[:3]
            for integ in INTEGRATOR_NAMES:
                s = scale.get(integ, 1.0) if isinstance(scale, dict) else scale
                if s is None or not pred[integ]["feasible"][0]:
                    continue
                dt = float(pred[integ]["dt"][0]) * float(s)
                err, _ = final_position_error_km(integ, state0, dt, T, ref_pos, mu)
                records.append({"point": point, "Az": az, "T_full": res["T_full"], "sim_time": T,
                                "integrator": integ, "dt": dt, "scale": float(s),
                                "pos_err_km": float("nan") if err is None else err,
                                "stable": bool(err is not None and err <= tol_km)})
    return pd.DataFrame(records)