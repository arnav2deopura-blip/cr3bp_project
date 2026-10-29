"""
Physics-rule baselines for the Sundman-time step (Part D, D1). The Part C policy predicts log10 ds with a 25-tree model on 25 features
(15 RHS-equivalents). Is that needed? A rule with 2-5 coefficients on a few free (pre-simulation) quantities is fitted on the SAME training rows
with the same quantile loss, certified with the same end-to-end rule and charged its own measured overhead.

Features (all computable from the initial state and the horizon; the three-body quantities are analytic, no integration):
  lT    log10 of the horizon T
  reg   the Hill regime as an ordinal (I = 1 ... V = 5): 1 + number of the thresholds C_L1 > C_L2 > C_L3 > C_L4 that C is at or below
  tau   osc_logtau_hat: log10 of the shortest two-body timescale (periapsis timescale of the osculating orbit about either body if the
        periapsis is reached before T, the local sqrt(r^3 / GM) otherwise)
  lrp   log10 of the smallest osculating periapsis radius over the two bodies (the current distance where no periapsis is reached before T)
  dC    C - C_L1
A rule is  log10 ds = b + w . features  with a subset of the features switched on; `RULE_SETS` pins the subsets before any fit.
The rule enters Part C's machinery as models["rule"] (see policy_c_3d.predict_raw_c), so it is certified, run and scored by the same code.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

from .config import MU
from .compiled_model_3d import c_l1_for_mu, _median_ns, _loop_rhs
from .osculating_3d import osc_row
from .diagnostics_3d import jacobi_at_points

RULE_FEATURES = ("lT", "reg", "tau", "lrp", "dC")
# name -> features switched on (pinned before any fit; PREREGISTRATION_D.md section 3)
RULE_SETS = {
    "rule_T": ("lT",),
    "rule_TH": ("lT", "reg"),
    "rule_TP": ("lT", "tau"),
    "rule_THP": ("lT", "reg", "tau"),
    "rule_THR": ("lT", "reg", "lrp"),
    "rule_THPD": ("lT", "reg", "tau", "dC"),
}
RULE_QUANTILES = (0.1, 0.2, 0.3, 0.5)


def hill_thresholds(mu):
    """(C_L1, C_L2, C_L3, C_L4) as an array, strictly decreasing."""
    c = jacobi_at_points(mu)
    return np.array([c["L1"], c["L2"], c["L3"], c["L4"]], dtype=np.float64)


@njit(cache=True)
def rule_features_row(st, T, mu, c_l1, thr, need_osc, out, obuf, flags):
    """The five rule features of one state into out[0:5]; the periapsis kernel runs only if need_osc."""
    x, y, z, vx, vy, vz = st[0], st[1], st[2], st[3], st[4], st[5]
    r1 = max(math.sqrt((x + mu) ** 2 + y ** 2 + z ** 2), 1e-9)
    r2 = max(math.sqrt((x + mu - 1.0) ** 2 + y ** 2 + z ** 2), 1e-9)
    C = x * x + y * y + 2.0 * (1.0 - mu) / r1 + 2.0 * mu / r2 - (vx * vx + vy * vy + vz * vz)
    out[0] = math.log10(T)
    reg = 1.0
    for j in range(4):
        if C <= thr[j]:
            reg += 1.0
    out[1] = reg
    out[4] = C - c_l1
    if need_osc:
        osc_row(x, y, z, vx, vy, vz, T, mu, obuf, flags)
        out[2] = obuf[12]
        l1 = obuf[0] if obuf[3] > 0.5 else math.log10(r1)
        l2 = obuf[6] if obuf[9] > 0.5 else math.log10(r2)
        out[3] = min(l1, l2)
    else:
        out[2] = 0.0
        out[3] = 0.0


@njit(cache=True)
def rule_features_batch(states, T, mu, c_l1, thr, need_osc):
    n = states.shape[0]
    out = np.empty((n, 5))
    obuf = np.zeros(13)
    flags = np.zeros(2)
    for i in range(n):
        rule_features_row(states[i], T[i], mu, c_l1, thr, need_osc, out[i], obuf, flags)
    return out


@njit(cache=True)
def rule_log_ds_one(st, T, mu, c_l1, thr, w, b, need_osc, out, obuf, flags):
    rule_features_row(st, T, mu, c_l1, thr, need_osc, out, obuf, flags)
    s = b
    for j in range(5):
        s += w[j] * out[j]
    return s


@njit(cache=True)
def _loop_rule(states, T, reps, mu, c_l1, thr, w, b, need_osc):
    n = states.shape[0]
    out = np.zeros(5)
    obuf = np.zeros(13)
    flags = np.zeros(2)
    acc = 0.0
    for r in range(reps):
        i = r % n
        acc += 10.0 ** rule_log_ds_one(states[i], T[i], mu, c_l1, thr, w, b, need_osc, out, obuf, flags)
    return acc


def _needs_osc(cols):
    return bool({"tau", "lrp"} & set(cols))


def fit_rule(train, cols, quantile, mu=MU, label_col="dt_rk4s", success_col="rk4s_success", sample_weight=None, max_rows=None, seed=42):
    """Weighted linear quantile regression of log10(label) on the chosen features (HiGHS), coefficients folded back to raw features.
    Returns the rule dict used as models["rule"]."""
    from sklearn.linear_model import QuantileRegressor
    cols = tuple(cols)
    c_l1 = c_l1_for_mu(mu)
    thr = hill_thresholds(mu)
    ok = train[success_col].to_numpy() == 1
    tr = train[ok]
    w = None if sample_weight is None else np.asarray(sample_weight, dtype=float)[ok]
    if max_rows and len(tr) > max_rows:
        idx = np.random.RandomState(seed).choice(len(tr), max_rows, replace=False)
        tr = tr.iloc[idx]
        w = None if w is None else w[idx]
    F = rule_features_batch(tr[["x", "y", "z", "vx", "vy", "vz"]].to_numpy(dtype=np.float64), tr["sim_time"].to_numpy(dtype=np.float64), mu, c_l1,
                            thr, _needs_osc(cols))
    X = F[:, [RULE_FEATURES.index(c) for c in cols]]
    m, sd = X.mean(axis=0), X.std(axis=0) + 1e-12
    reg = QuantileRegressor(quantile=quantile, alpha=0.0, solver="highs-ipm").fit((X - m) / sd, np.log10(tr[label_col].to_numpy(dtype=float)), sample_weight=w)
    wfull = np.zeros(5)
    for c, coef, s in zip(cols, reg.coef_, sd):
        wfull[RULE_FEATURES.index(c)] = coef / s
    b = float(reg.intercept_ - np.dot(reg.coef_, m / sd))
    return {"cols": cols, "w": wfull, "b": b, "quantile": float(quantile), "mu": float(mu), "c_l1": float(c_l1), "thr": thr, "need_osc": _needs_osc(cols),
            "n_params": len(cols) + 1, "n_fit_rows": int(len(tr))}


def predict_rule(rule, df):
    """log10 ds of every row of df (needs x..vz and sim_time)."""
    F = rule_features_batch(df[["x", "y", "z", "vx", "vy", "vz"]].to_numpy(dtype=np.float64), df["sim_time"].to_numpy(dtype=np.float64), rule["mu"],
                            rule["c_l1"], rule["thr"], rule["need_osc"])
    return F @ rule["w"] + rule["b"]


def rule_models(rule):
    """The models dict policy_c_3d.predict_raw_c / fit-free deployment understands."""
    return {"features": list(rule["cols"]), "candidates": ("rk4s",), "rule": rule, "lf_gate": "none", "structured": False, "quantile": rule["quantile"]}


def measure_rule_cost(rule, states, T, n_policy=100_000, n_rhs=1_000_000, repeats=5):
    """RHS-equivalents per orbit of the compiled rule, by the cost protocol of compiled_model_3d.measure_single_call_cost (one orbit at a time in a
    compiled loop over 1,000 distinct states, the RHS timed the same way, medians over repeats). Run on an idle machine."""
    states = np.ascontiguousarray(states[:1000], dtype=np.float64)
    T = np.ascontiguousarray(T[:1000], dtype=np.float64)
    a = (rule["mu"], rule["c_l1"], rule["thr"], rule["w"], rule["b"], rule["need_osc"])
    _loop_rule(states, T, 10, *a)
    _loop_rhs(states, 10, rule["mu"])
    pol_ns, pol_rng = _median_ns(lambda n: _loop_rule(states, T, n, *a), n_policy, repeats)
    rhs_ns, rhs_rng = _median_ns(lambda n: _loop_rhs(states, n, rule["mu"]), n_rhs, repeats)
    return {"policy_ns_per_call": pol_ns, "policy_ns_range": pol_rng, "rhs_ns_per_call": rhs_ns, "rhs_ns_range": rhs_rng, "rhs_equiv": pol_ns / rhs_ns,
            "rhs_equiv_range": [pol_rng[0] / rhs_rng[1], pol_rng[1] / rhs_rng[0]], "repeats": repeats, "kind": "rule"}


def fit_lin33(train, quantile=0.1, mu=MU, label_col="dt_rk4s", success_col="rk4s_success", sample_weight=None, max_rows=60000, seed=42):
    """Part C's M0 class for the Sundman step (one linear quantile regressor on the 33 engineered features of compiled_model_3d), refitted with the interior-point
    solver on up to max_rows rows. A descriptive 'ML-lite' arm of D1: the same 25 free quantities as the trees, no interactions beyond the engineered products."""
    from sklearn.linear_model import QuantileRegressor
    from .compiled_model_3d import linear_features_batch
    c_l1 = c_l1_for_mu(mu)
    ok = train[success_col].to_numpy() == 1
    tr = train[ok]
    w = None if sample_weight is None else np.asarray(sample_weight, dtype=float)[ok]
    if max_rows and len(tr) > max_rows:
        idx = np.random.RandomState(seed).choice(len(tr), max_rows, replace=False)
        tr = tr.iloc[idx]
        w = None if w is None else w[idx]
    X = linear_features_batch(tr[["x", "y", "z", "vx", "vy", "vz"]].to_numpy(dtype=np.float64), tr["sim_time"].to_numpy(dtype=np.float64), mu, c_l1)
    m, sd = X.mean(axis=0), X.std(axis=0) + 1e-12
    reg = QuantileRegressor(quantile=quantile, alpha=0.0, solver="highs-ipm").fit((X - m) / sd, np.log10(tr[label_col].to_numpy(dtype=float)), sample_weight=w)
    return {"mu": mu, "c_l1": c_l1, "w": np.ascontiguousarray(reg.coef_ / sd, dtype=np.float64), "b": float(reg.intercept_ - np.dot(reg.coef_, m / sd))}
