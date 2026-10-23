"""
How well does energy (Jacobi) drift predict trajectory (position) error?

proxy_vs_error_table integrates sampled orbits at a ladder of fixed steps
(fractions of each orbit's dt_ceiling) and records, for the SAME run, the max
relative Jacobi drift and the final position error vs a DOP853 reference.
proxy_failure_summary quantifies where "drift <= tol" gives false assurance.
fit_error_predictor trains a regressor of log10(position error) from orbit
features + step size + integrator only (no integration, no reference), and
compares it with a Jacobi-drift linear fit and a dt/horizon scaling law;
split by ORBIT so rows of one orbit never straddle train/test.

Caveats: the exchangeability behind the conformal bound holds across orbits
drawn the same way as the table's orbits, not under regional distribution
shift; leapfrog's Jacobi behavior is structural (symplectic), so its
correlation should be read separately from RK4/Euler.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score

from .config import MU, POSITION_TOL_KM, PHYSICS_FEATURE_COLUMNS_3D
from .physics_3d import INTEGRATORS_3D, max_jacobi_drift_3d
from .search_3d import _dt_upper_bound_3d
from .audit_3d import ReferenceCache, final_position_error_km, STATE_COLS, INTEGRATOR_NAMES

DT_FRACS = (0.03, 0.06, 0.12, 0.25, 0.5, 1.0)
DRIFT_FLOOR = 1e-16
ERR_FLOOR_KM = 1e-9


def proxy_vs_error_table(data, mu=MU, num_orbits=100, dt_fracs=DT_FRACS,
                         integrators=INTEGRATOR_NAMES, random_state=42):
    rng = np.random.RandomState(random_state)
    rows = rng.choice(len(data), size=min(num_orbits, len(data)), replace=False)
    states = data[STATE_COLS].to_numpy(dtype=float)
    horizons = data["sim_time"].to_numpy(dtype=float)
    labels = data["orbit_label"].to_numpy() if "orbit_label" in data.columns else np.array(["?"] * len(data))
    cache = ReferenceCache(mu)
    records = []
    for i in rows:
        state = list(states[i])
        T = horizons[i]
        sol, reached = cache.get(i, state, T)
        if not reached:
            continue
        ref_pos = sol.sol(T)[:3]
        ceiling = float(data["dt_ceiling"].iloc[i]) if "dt_ceiling" in data.columns else max(_dt_upper_bound_3d(state, mu), 0.005)
        for integ in integrators:
            for f in dt_fracs:
                dt = ceiling * f
                ok, drift, _t, _r = max_jacobi_drift_3d(INTEGRATORS_3D[integ], state, dt, T, mu)
                err, _ = final_position_error_km(integ, state, dt, T, ref_pos, mu)
                records.append({
                    "orbit_row": int(i), "orbit_label": labels[i], "integrator": integ, "dt": dt,
                    "dt_frac": f, "sim_time": T,
                    "jacobi_drift": float(drift) if ok else float("inf"),
                    "pos_err_km": float("nan") if err is None else err,
                    "completed": bool(ok and err is not None),
                })
    return pd.DataFrame(records)


def _group_stats(g, drift_tol, tol_km):
    n = len(g)
    ld = np.log10(np.clip(g["jacobi_drift"].to_numpy(), DRIFT_FLOOR, None))
    le = np.log10(np.clip(g["pos_err_km"].to_numpy(), ERR_FLOOR_KM, None))
    rho = float(spearmanr(ld, le)[0]) if n >= 5 and np.ptp(ld) > 0 and np.ptp(le) > 0 else float("nan")
    drift_ok = g["jacobi_drift"] <= drift_tol
    bad = g["pos_err_km"] > tol_km
    auc = float("nan")
    if bad.nunique() == 2:
        auc = float(roc_auc_score(bad, g["jacobi_drift"].clip(upper=1e6)))
    return pd.Series({
        "n": n, "spearman_log_drift_vs_log_err": rho,
        "n_drift_ok": int(drift_ok.sum()),
        "false_assurance_rate": float(bad[drift_ok].mean()) if drift_ok.any() else float("nan"),
        "n_drift_bad": int((~drift_ok).sum()),
        "false_alarm_rate": float((~bad[~drift_ok]).mean()) if (~drift_ok).any() else float("nan"),
        "auc_drift_detects_err": auc,
    })


def proxy_failure_summary(table, drift_tol=1e-5, tol_km=None):
    """false_assurance_rate = P(pos_err > tol_km | drift <= drift_tol); false_alarm_rate = P(pos_err <= tol_km | drift > drift_tol)."""
    if tol_km is None:
        tol_km = POSITION_TOL_KM
    t = table[table["completed"]]
    by_integ = t.groupby("integrator").apply(lambda g: _group_stats(g, drift_tol, tol_km), include_groups=False)
    by_type = t.groupby(["orbit_label", "integrator"]).apply(lambda g: _group_stats(g, drift_tol, tol_km),
                                                              include_groups=False)
    return by_integ, by_type


def _design(table, data, feature_cols):
    t = table[table["completed"]].reset_index(drop=True)
    feats = data[feature_cols].to_numpy(dtype=float)[t["orbit_row"].to_numpy()]
    onehot = np.stack([(t["integrator"] == k).to_numpy(dtype=float) for k in INTEGRATOR_NAMES], axis=1)
    X = np.column_stack([feats, np.log10(t["dt"].to_numpy()), t["dt_frac"].to_numpy(), onehot])
    y = np.log10(np.clip(t["pos_err_km"].to_numpy(), ERR_FLOOR_KM, None))
    return t, X, y, onehot


def fit_error_predictor(table, data, tol_km=None, feature_cols=None, alpha=0.1, test_frac=0.3,
                        cal_frac=0.3, random_state=42, n_estimators=200):
    if tol_km is None:
        tol_km = POSITION_TOL_KM
    if feature_cols is None:
        feature_cols = [c for c in PHYSICS_FEATURE_COLUMNS_3D if c in data.columns]
    t, X, y, onehot = _design(table, data, feature_cols)
    orbits = np.array(sorted(t["orbit_row"].unique()))
    rng = np.random.RandomState(random_state)
    rng.shuffle(orbits)
    n_te, n_cal = int(round(test_frac * len(orbits))), int(round(cal_frac * len(orbits)))
    te_o, cal_o, tr_o = orbits[:n_te], orbits[n_te:n_te + n_cal], orbits[n_te + n_cal:]
    if min(len(te_o), len(cal_o), len(tr_o)) < 2:
        raise ValueError("too few orbits for a train/calibration/test split")
    m_tr, m_cal, m_te = (t["orbit_row"].isin(s).to_numpy() for s in (tr_o, cal_o, te_o))

    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    rf.fit(X[m_tr], y[m_tr])
    pred_te = rf.predict(X[m_te])

    drift_feat = lambda m: np.column_stack([np.log10(np.clip(t.loc[m, "jacobi_drift"].to_numpy(), DRIFT_FLOOR, None)), onehot[m]])
    lin_j = LinearRegression().fit(drift_feat(m_tr), y[m_tr])
    pred_j = lin_j.predict(drift_feat(m_te))
    scal_feat = lambda m: np.column_stack([np.log10(t.loc[m, "dt"].to_numpy()), np.log10(t.loc[m, "sim_time"].to_numpy()), onehot[m]])
    lin_s = LinearRegression().fit(scal_feat(m_tr), y[m_tr])
    pred_s = lin_s.predict(scal_feat(m_te))

    y_te = y[m_te]
    thr = np.log10(tol_km)
    bad = y_te > thr

    def metrics(p):
        return {"r2_log_err": float(r2_score(y_te, p)), "mae_log10": float(mean_absolute_error(y_te, p)),
                "auc_detects_err_over_tol": float(roc_auc_score(bad, p)) if bad.any() and (~bad).any() else float("nan")}

    res_cal = y[m_cal] - rf.predict(X[m_cal])
    n = len(res_cal)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    q = float(np.sort(res_cal)[k - 1]) if k <= n else float("inf")
    upper = pred_te + q
    certified = upper <= thr
    return {
        "n_rows": {"train": int(m_tr.sum()), "cal": int(m_cal.sum()), "test": int(m_te.sum())},
        "n_orbits": {"train": int(len(tr_o)), "cal": int(len(cal_o)), "test": int(len(te_o))},
        "random_forest_no_reference": metrics(pred_te),
        "jacobi_drift_linear": metrics(pred_j),
        "dt_horizon_scaling_law": metrics(pred_s),
        "conformal_upper_bound": {
            "alpha": alpha, "q_log10": q,
            "coverage_on_test": float(np.mean(y_te <= upper)),
            "frac_certified_under_tol": float(certified.mean()),
            "precision_certified_truly_under_tol": float((~bad[certified]).mean()) if certified.any() else float("nan"),
            "recall_of_truly_under_tol": float(certified[~bad].mean()) if (~bad).any() else float("nan"),
        },
        "test_predictions": pd.DataFrame({"orbit_row": t.loc[m_te, "orbit_row"].to_numpy(),
                                          "integrator": t.loc[m_te, "integrator"].to_numpy(),
                                          "y_true_log10_km": y_te, "y_pred_log10_km": pred_te,
                                          "y_upper_log10_km": upper}),
    }