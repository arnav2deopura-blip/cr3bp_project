"""
Error-model recommender and convergence-order check.

Instead of learning the label "largest safe dt" (which is ill-defined when
position error is not monotone in dt), learn the quantity directly:
log10(final position error in km) as a function of the starting state, the
horizon, log10(dt) and the integrator. The step is then CHOSEN, not predicted:
the largest dt on a geometric ladder whose conformal upper error bound -- and
the bound at every finer ladder step -- is below the tolerance.

error_ladder_table integrates sampled orbits at a geometric ladder of steps
anchored at each orbit's dt ceiling and records the true final position error
against the DOP853 reference. A fixed-step run that collides/escapes is
recorded with completed=False and an imputed error of one length unit (DU) so
hard regions are not silently dropped.

convergence_order_study fits log10(error) vs log10(dt) per orbit and integrator
over the points inside the asymptotic window and compares the slope with the
textbook order (Euler 1, leapfrog 2, RK4 4). Departures (in particular near
the Moon, or in chaotic regions where error saturates) are results, not bugs;
the theoretical orders are expectations being tested.

Caveats: the conformal margin is calibrated on rows of held-out ORBITS but
rows from the same orbit are correlated, and it assumes new orbits are
exchangeable with the calibration orbits (not true under regional shift, e.g.
CAPSTONE-like NRHO states if they are absent from training). Orbits with no
admissible step on the ladder get no recommendation (reported as coverage).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .config import MU, POSITION_TOL_KM, PHYSICS_FEATURE_COLUMNS_3D, du_km_for_mu
from .physics_3d import EVALS_PER_STEP_3D
from .search_3d import _dt_upper_bound_3d
from .audit_3d import ReferenceCache, final_position_error_km, STATE_COLS, INTEGRATOR_NAMES, clopper_pearson

DT_MIN = 1e-4
LADDER_RATIO = 2.0 ** 0.5
LADDER_MAX_LEVELS = 17
ERR_FLOOR_KM = 1e-9
THEORETICAL_ORDER = {"euler": 1.0, "leapfrog": 2.0, "rk4": 4.0}


def _ladder(dt_max, ratio=LADDER_RATIO, dt_min=DT_MIN, max_levels=LADDER_MAX_LEVELS):
    out = []
    for k in range(max_levels):
        dt = dt_max * ratio ** (-k)
        if dt < dt_min:
            break
        out.append(dt)
    return out


def error_ladder_table(data, num_orbits=300, integrators=INTEGRATOR_NAMES, ratio=LADDER_RATIO,
                       max_levels=LADDER_MAX_LEVELS, mu=MU, random_state=42, cache=None, rows=None):
    if cache is None:
        cache = ReferenceCache(mu)
    states = data[STATE_COLS].to_numpy(dtype=float)
    horizons = data["sim_time"].to_numpy(dtype=float)
    labels = data["orbit_label"].to_numpy() if "orbit_label" in data.columns else np.array(["?"] * len(data))
    if rows is None:
        rows = np.random.RandomState(random_state).choice(len(data), size=min(num_orbits, len(data)), replace=False)
    du = du_km_for_mu(mu)
    records = []
    for i in rows:
        state = list(states[i])
        T = float(horizons[i])
        sol, reached = cache.get(int(i), state, T)
        if not reached:
            continue
        ref_pos = sol.sol(T)[:3]
        dt_max = max(_dt_upper_bound_3d(state, mu), 0.005)
        ladder = _ladder(dt_max, ratio, max_levels=max_levels)
        for integ in integrators:
            for k, dt in enumerate(ladder):
                err, _reason = final_position_error_km(integ, state, float(dt), T, ref_pos, mu)
                records.append({
                    "orbit_row": int(i), "orbit_label": labels[i], "integrator": integ, "level": k,
                    "dt": float(dt), "dt_max": float(dt_max), "sim_time": T,
                    "pos_err_km": float(du) if err is None else float(err), "completed": err is not None,
                })
    return pd.DataFrame(records, columns=["orbit_row", "orbit_label", "integrator", "level", "dt", "dt_max",
                                          "sim_time", "pos_err_km", "completed"])


def _design(table, data, feature_cols):
    feats = data[feature_cols].to_numpy(dtype=float)[table["orbit_row"].to_numpy()]
    dt = table["dt"].to_numpy(dtype=float)
    dt_max = table["dt_max"].to_numpy(dtype=float)
    onehot = np.stack([(table["integrator"] == k).to_numpy(dtype=float) for k in INTEGRATOR_NAMES], axis=1)
    X = np.column_stack([feats, np.log10(dt), np.log10(dt / dt_max), onehot])
    y = np.log10(np.clip(table["pos_err_km"].to_numpy(dtype=float), ERR_FLOOR_KM, None))
    return X, y


def fit_error_model(table, data, feature_cols=None, tol_km=None, alpha=0.1, test_frac=0.3, cal_frac=0.3,
                    random_state=42, n_estimators=200):
    if tol_km is None:
        tol_km = POSITION_TOL_KM
    if feature_cols is None:
        feature_cols = [c for c in PHYSICS_FEATURE_COLUMNS_3D if c in data.columns]
    orbits = np.array(sorted(table["orbit_row"].unique()))
    rng = np.random.RandomState(random_state)
    rng.shuffle(orbits)
    n_te, n_cal = int(round(test_frac * len(orbits))), int(round(cal_frac * len(orbits)))
    te_o, cal_o, tr_o = orbits[:n_te], orbits[n_te:n_te + n_cal], orbits[n_te + n_cal:]
    if min(len(te_o), len(cal_o), len(tr_o)) < 2:
        raise ValueError("too few orbits for a train/calibration/test split")
    X, y = _design(table, data, feature_cols)
    m_tr = table["orbit_row"].isin(tr_o).to_numpy()
    m_cal = table["orbit_row"].isin(cal_o).to_numpy()
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1, min_samples_leaf=2)
    rf.fit(X[m_tr], y[m_tr])
    res = y[m_cal] - rf.predict(X[m_cal])
    n = len(res)
    k = int(math.ceil((n + 1) * (1 - alpha)))
    q = float(np.sort(res)[k - 1]) if k <= n else float("inf")
    return {"rf": rf, "q_log10": q, "alpha": alpha, "tol_km": tol_km, "feature_cols": list(feature_cols),
            "train_orbits": tr_o, "cal_orbits": cal_o, "test_orbits": te_o, "n_cal_rows": int(n)}


def recommend_for_orbit(em, data, row, mu=MU, integrators=INTEGRATOR_NAMES, ratio=LADDER_RATIO,
                        max_levels=LADDER_MAX_LEVELS):
    """
    Returns {integrator: dt or None}. dt is the largest ladder step whose upper
    error bound is <= tol_km and so is the bound at EVERY finer ladder step.
    """
    state = list(data[STATE_COLS].to_numpy(dtype=float)[row])
    dt_max = max(_dt_upper_bound_3d(state, mu), 0.005)
    ladder = _ladder(dt_max, ratio, max_levels=max_levels)
    feats = data[em["feature_cols"]].to_numpy(dtype=float)[row]
    K = len(ladder)
    blocks = []
    for ii in range(len(integrators)):
        onehot = np.zeros(len(INTEGRATOR_NAMES))
        onehot[INTEGRATOR_NAMES.index(integrators[ii])] = 1.0
        for dt in ladder:
            blocks.append(np.concatenate([feats, [math.log10(dt), math.log10(dt / dt_max)], onehot]))
    pred = em["rf"].predict(np.array(blocks))
    thr = math.log10(em["tol_km"])
    out = {}
    for ii, integ in enumerate(integrators):
        upper = pred[ii * K:(ii + 1) * K] + em["q_log10"]
        ok = upper <= thr
        suffix_ok = np.logical_and.accumulate(ok[::-1])[::-1]
        idx = np.where(suffix_ok)[0]
        out[integ] = float(ladder[idx[0]]) if len(idx) else None
    return out


def evaluate_recommender(em, data, orbit_rows=None, mu=MU, integrators=("rk4", "leapfrog"), cache=None,
                         include_feature_cost=True, adaptive_methods=("RK45", "DOP853")):
    """
    For each (test) orbit: recommend a step per integrator, take the cheapest
    recommendation, integrate, and score the real final error against the same
    DOP853 reference used for the labels. Adaptive solvers are run at the
    per-orbit ORACLE rtol that just meets tol_km (a lower bound on their cost;
    see fair_baseline_3d). Returns (per_orbit_df, summary_dict).
    """
    from .fair_baseline_3d import scipy_matched_accuracy, measure_cost_ratios
    from .physics_3d import compute_ftle_3d_variational_details, INTEGRATORS_3D

    tol_km = em["tol_km"]
    if cache is None:
        cache = ReferenceCache(mu)
    if orbit_rows is None:
        orbit_rows = em["test_orbits"]
    states = data[STATE_COLS].to_numpy(dtype=float)
    horizons = data["sim_time"].to_numpy(dtype=float)
    ratio_cost = measure_cost_ratios(mu)["variational_step_over_rk4_step"] if include_feature_cost else float("nan")
    records = []
    n_ref_fail = 0
    for i in orbit_rows:
        state = list(states[i])
        T = float(horizons[i])
        sol, reached = cache.get(int(i), state, T)
        if not reached:
            n_ref_fail += 1
            continue
        ref_pos = sol.sol(T)[:3]
        rec = recommend_for_orbit(em, data, int(i), mu, integrators)
        cands = {k: v for k, v in rec.items() if v is not None}
        rec_row = {"orbit_row": int(i), "sim_time": T, "covered": bool(cands)}
        if not cands:
            records.append(rec_row)
            continue
        costs = {k: int(math.ceil(T / v - 1e-9)) * EVALS_PER_STEP_3D[k] for k, v in cands.items()}
        choice = min(costs, key=costs.get)
        dt = cands[choice]
        err, _ = final_position_error_km(choice, state, dt, T, ref_pos, mu)
        rec_row.update({"integrator": choice, "dt": dt, "ml_evals": costs[choice],
                        "err_km": np.nan if err is None else err,
                        "violation": bool(err is None or err > tol_km)})
        if include_feature_cost:
            fd = compute_ftle_3d_variational_details(state, mu, T)
            rec_row["ml_total_evals_equiv"] = costs[choice] + fd["steps"] * ratio_cost * EVALS_PER_STEP_3D["rk4"]
        for m in adaptive_methods:
            res = scipy_matched_accuracy(state, T, ref_pos, target_err_km=tol_km, mu=mu, method=m)
            rec_row[f"{m}_spec_evals"] = np.nan if res is None else res["nfev"]
        records.append(rec_row)
    df = pd.DataFrame(records)
    summary = {"n_orbits_evaluated": int(len(df)), "n_reference_failed": n_ref_fail, "tol_km": tol_km,
               "q_log10": em["q_log10"], "alpha": em["alpha"]}
    if len(df) == 0:
        return df, summary
    df["covered"] = df["covered"].astype(bool)
    if "violation" in df.columns:
        df["violation"] = df["violation"].fillna(False).astype(bool)
    summary["coverage"] = float(df["covered"].mean())
    cov = df[df["covered"]]
    if len(cov):
        k, n = int(cov["violation"].sum()), int(len(cov))
        lo, hi = clopper_pearson(k, n)
        summary.update({"n_covered": n, "violations": k, "violation_rate": k / n, "violation_ci95": [lo, hi],
                        "median_ml_evals": float(cov["ml_evals"].median())})
        good = cov[~cov["violation"]]
        for m in adaptive_methods:
            col = f"{m}_spec_evals"
            d = good[good[col].notna()]
            if len(d):
                summary[f"median_ml_over_{m}_spec_evals"] = float((d["ml_evals"] / d[col]).median())
                if include_feature_cost:
                    r = d["ml_total_evals_equiv"] / d[col]
                    summary[f"median_ml_total_over_{m}_spec_evals"] = float(r.median())
                    summary[f"frac_ml_total_cheaper_than_{m}_spec"] = float((r < 1.0).mean())
    return df, summary


def convergence_order_study(table, data, tol_km=None, err_lo_km=1e-3, err_hi_factor=10.0, min_points=4):
    """
    Per orbit and integrator, fit log10(error) vs log10(dt) over the ladder
    points that completed and have err_lo_km <= error <= err_hi_factor*tol_km
    (the window where truncation error should dominate: above round-off, below
    saturation). Returns (per_orbit_df, summary_df); summary slopes are
    medians with quartiles, and frac_within_half_order is the share of fits
    within +/-0.5 of the textbook order.
    """
    if tol_km is None:
        tol_km = POSITION_TOL_KM
    hi = tol_km * err_hi_factor
    r2 = data["r2_3d"].to_numpy(dtype=float) if "r2_3d" in data.columns else None
    t = table[table["completed"]]
    rows = []
    for (orbit, integ), g in t.groupby(["orbit_row", "integrator"]):
        g = g[(g["pos_err_km"] >= err_lo_km) & (g["pos_err_km"] <= hi)]
        if len(g) < min_points:
            continue
        x = np.log10(g["dt"].to_numpy(dtype=float))
        y = np.log10(g["pos_err_km"].to_numpy(dtype=float))
        if np.ptp(x) == 0:
            continue
        slope, intercept = np.polyfit(x, y, 1)
        resid = y - (slope * x + intercept)
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        rows.append({"orbit_row": int(orbit), "integrator": integ, "orbit_label": g["orbit_label"].iloc[0],
                     "r2_3d": float(r2[int(orbit)]) if r2 is not None else np.nan,
                     "n_points": int(len(g)), "slope": float(slope),
                     "fit_r2": float(1.0 - np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else np.nan,
                     "theory": THEORETICAL_ORDER[integ]})
    per = pd.DataFrame(rows, columns=["orbit_row", "integrator", "orbit_label", "r2_3d", "n_points", "slope",
                                      "fit_r2", "theory"])
    if len(per) == 0:
        return per, pd.DataFrame()
    per["region"] = pd.cut(per["r2_3d"], bins=[-np.inf, 0.1, 0.5, np.inf], labels=["near_moon_r2<0.1", "mid_0.1-0.5",
                                                                                    "far_r2>=0.5"]).astype(str)

    def agg(g):
        return pd.Series({
            "n": int(len(g)), "theory": float(g["theory"].iloc[0]),
            "slope_median": float(g["slope"].median()), "slope_q25": float(g["slope"].quantile(0.25)),
            "slope_q75": float(g["slope"].quantile(0.75)),
            "frac_within_half_order": float((np.abs(g["slope"] - g["theory"]) <= 0.5).mean()),
        })

    pieces = []
    for keys in (["integrator"], ["integrator", "region"], ["integrator", "orbit_label"]):
        s = per.groupby(keys).apply(agg, include_groups=False).reset_index()
        s.insert(0, "grouping", "+".join(keys))
        pieces.append(s)
    return per, pd.concat(pieces, ignore_index=True)