"""
Compiled policy models (Part C plan 5.2.1): the trained gradient-boosted trees are exported to flat arrays and evaluated in Numba, so
the model costs a few right-hand-side evaluations instead of ~1,100.

  export_forest(est)       flatten a fitted HistGradientBoosting{Regressor,Classifier} into (feat, thr, left, right, val, miss_left, roots,
                           baseline). These come from PRIVATE scikit-learn attributes (est._predictors[i][0].nodes, est._baseline_prediction):
                           the version is pinned (scikit-learn 1.9.0) and tests/test_part_c.py asserts that the compiled predictions equal
                           est.predict / est.predict_proba to <= 1e-12, so a different layout fails loudly instead of mispredicting.
  forest_raw / forest_batch  Numba evaluation (x[feat] <= thr goes left; leaves have left = -1)
  policy_hgb_one           ONE orbit end to end: state (+ periapsis) features -> RK4 and leapfrog quantile models -> leapfrog gate ->
                           cost-normalized choice -> raw dt. This is what is timed.
  measure_single_call_cost the cost protocol of the plan (single orbit at a time, median over repeats) in RHS-equivalents

"Single call" here means one orbit evaluated alone inside a compiled loop that cycles through 1,000 distinct states, so there is no batching
benefit, no Python dispatch and no repeated-state cache effect; the RHS reference is timed the same way (a compiled loop over the same
states calling _cr3bp_derivatives_3d_numba, the function the integrators use). Python-dispatched single calls are reported next to it.
"""
from __future__ import annotations

import math
import time

import numpy as np
from numba import njit

from .config import MU
from .osculating_3d import osc_row
from .physics_3d import _cr3bp_derivatives_3d_numba, jacobi_constant_3d, libration_point_location

GATE_NONE, GATE_CLF, GATE_CENSORED = 0, 1, 2
DT_MIN = 1e-4
EVALS_RK4, EVALS_LF = 4.0, 1.0


# ---------------------------------------------------------------------------------------------------------------------
# export / evaluate
# ---------------------------------------------------------------------------------------------------------------------

def export_forest(est):
    """Flatten a fitted HistGradientBoosting estimator (one tree per iteration: regression or binary classification)."""
    preds = est._predictors
    feat, thr, left, right, val, miss, roots = [], [], [], [], [], [], []
    off = 0
    for it in preds:
        if len(it) != 1:
            raise ValueError("only one tree per iteration is supported (regression / binary classification)")
        nodes = it[0].nodes
        if np.any(nodes["is_categorical"]):
            raise ValueError("categorical splits are not supported")
        n = len(nodes)
        roots.append(off)
        isleaf = nodes["is_leaf"].astype(bool)
        feat.append(np.where(isleaf, 0, nodes["feature_idx"]).astype(np.int64))
        thr.append(np.where(isleaf, 0.0, nodes["num_threshold"]).astype(np.float64))
        left.append(np.where(isleaf, -1, nodes["left"].astype(np.int64) + off))
        right.append(np.where(isleaf, -1, nodes["right"].astype(np.int64) + off))
        val.append(nodes["value"].astype(np.float64))
        miss.append(nodes["missing_go_to_left"].astype(np.uint8))
        off += n
    base = float(np.asarray(est._baseline_prediction).ravel()[0])
    return (np.concatenate(feat), np.concatenate(thr), np.concatenate(left), np.concatenate(right), np.concatenate(val),
            np.concatenate(miss), np.asarray(roots, dtype=np.int64), base)


@njit(cache=True)
def forest_raw(x, feat, thr, left, right, val, miss, roots, base):
    s = base
    for t in range(roots.shape[0]):
        i = roots[t]
        while left[i] >= 0:
            v = x[feat[i]]
            if v != v:
                i = left[i] if miss[i] == 1 else right[i]
            elif v <= thr[i]:
                i = left[i]
            else:
                i = right[i]
        s += val[i]
    return s


@njit(cache=True)
def forest_batch(X, feat, thr, left, right, val, miss, roots, base):
    out = np.empty(X.shape[0])
    for r in range(X.shape[0]):
        out[r] = forest_raw(X[r], feat, thr, left, right, val, miss, roots, base)
    return out


def predict_forest(forest, X):
    return forest_batch(np.ascontiguousarray(X, dtype=np.float64), *forest)


def forest_nodes_visited(forest):
    """Rough per-call work: number of trees x mean depth is what costs; returned for the log."""
    left = forest[2]
    roots = forest[6]
    return {"n_trees": int(len(roots)), "n_nodes": int(len(left))}


# ---------------------------------------------------------------------------------------------------------------------
# features and the single-orbit policy
# ---------------------------------------------------------------------------------------------------------------------

def c_l1_for_mu(mu):
    x_l1, _ = libration_point_location(mu, "L1")
    return float(jacobi_constant_3d([x_l1, 0.0, 0.0, 0.0, 0.0, 0.0], mu))


@njit(cache=True)
def state_features_row(st, T, mu, c_l1, out):
    """policy_b_3d.FEATS_STATE (12 values) into out[0:12]; same arithmetic as physics_3d.add_orbit_features_3d."""
    x, y, z, vx, vy, vz = st[0], st[1], st[2], st[3], st[4], st[5]
    r1 = max(math.sqrt((x + mu) ** 2 + y ** 2 + z ** 2), 1e-9)
    r2 = max(math.sqrt((x + mu - 1.0) ** 2 + y ** 2 + z ** 2), 1e-9)
    C = x * x + y * y + 2.0 * (1.0 - mu) / r1 + 2.0 * mu / r2 - (vx * vx + vy * vy + vz * vz)
    out[0], out[1], out[2], out[3], out[4], out[5] = x, y, z, vx, vy, vz
    out[6] = T
    out[7] = C
    out[8] = r1
    out[9] = r2
    out[10] = r1 / r2
    out[11] = C - c_l1


@njit(cache=True)
def policy_hgb_one(st, T, mu, c_l1, use_osc, fr, fl, fg, gate_kind, gate_thr, xbuf, flags):
    """fr, fl, fg: forests (tuples) of the RK4 and leapfrog log10-dt regressors and of the leapfrog gate classifier (ignored unless
    gate_kind == GATE_CLF). gate_thr: logit(p*) for the classifier, margin g* (dex above dt_min) for the censored regressor.
    Returns (choice, raw_dt): choice 0 = RK4, 1 = leapfrog (-1 never: RK4 is always feasible)."""
    state_features_row(st, T, mu, c_l1, xbuf)
    if use_osc:
        osc_row(st[0], st[1], st[2], st[3], st[4], st[5], T, mu, xbuf[12:25], flags)
    log_rk4 = forest_raw(xbuf, fr[0], fr[1], fr[2], fr[3], fr[4], fr[5], fr[6], fr[7])
    log_lf = forest_raw(xbuf, fl[0], fl[1], fl[2], fl[3], fl[4], fl[5], fl[6], fl[7])
    allow = True
    if gate_kind == 1:
        allow = forest_raw(xbuf, fg[0], fg[1], fg[2], fg[3], fg[4], fg[5], fg[6], fg[7]) >= gate_thr
    elif gate_kind == 2:
        allow = (log_lf - math.log10(1e-4)) >= gate_thr
    d_rk4 = 10.0 ** log_rk4
    d_lf = 10.0 ** log_lf
    if allow and d_lf / 1.0 > d_rk4 / 4.0:
        return 1, d_lf
    return 0, d_rk4


@njit(cache=True)
def policy_hgb_batch(states, T, mu, c_l1, use_osc, fr, fl, fg, gate_kind, gate_thr):
    n = states.shape[0]
    ch = np.empty(n, dtype=np.int64)
    dt = np.empty(n)
    xbuf = np.zeros(25)
    flags = np.zeros(2)
    for i in range(n):
        c, d = policy_hgb_one(states[i], T[i], mu, c_l1, use_osc, fr, fl, fg, gate_kind, gate_thr, xbuf, flags)
        ch[i] = c
        dt[i] = d
    return ch, dt


class CompiledHGBPolicy:
    """Everything the compiled policy needs, built from the models of policy_c_3d.fit_policy_c."""

    def __init__(self, models, p_star=0.5, g_star=0.0, mu=MU):
        self.mu = float(mu)
        self.c_l1 = c_l1_for_mu(mu)
        feats = models["features"]
        self.use_osc = len(feats) > 12
        if self.use_osc and len(feats) != 25:
            raise ValueError("expected FEATS_STATE (12) or FEATS_OSC (25) columns")
        self.fr = export_forest(models["reg_rk4"])
        self.fl = export_forest(models["reg_leapfrog"])
        self.gate = models["lf_gate"]
        if self.gate == "clf":
            self.fg = export_forest(models["clf_leapfrog"])
            self.gate_kind = GATE_CLF
            p = min(max(float(p_star), 1e-12), 1.0 - 1e-12)
            self.gate_thr = math.log(p / (1.0 - p))
        else:
            self.fg = self.fl                    # unused placeholder with the right type
            self.gate_kind = GATE_CENSORED if self.gate == "censored" else GATE_NONE
            self.gate_thr = float(g_star)

    def _args(self):
        return (self.mu, self.c_l1, self.use_osc, self.fr, self.fl, self.fg, self.gate_kind, self.gate_thr)

    def batch(self, states, T):
        return policy_hgb_batch(np.ascontiguousarray(states, dtype=np.float64), np.ascontiguousarray(T, dtype=np.float64), *self._args())

    def one(self, state, T):
        xbuf = np.zeros(25)
        flags = np.zeros(2)
        return policy_hgb_one(np.ascontiguousarray(state, dtype=np.float64), float(T), *self._args(), xbuf, flags)

    def size(self):
        return {"rk4": forest_nodes_visited(self.fr), "leapfrog": forest_nodes_visited(self.fl),
                "gate": forest_nodes_visited(self.fg) if self.gate == "clf" else None}


# ---------------------------------------------------------------------------------------------------------------------
# cost protocol
# ---------------------------------------------------------------------------------------------------------------------

@njit(cache=True)
def _loop_policy(states, T, reps, mu, c_l1, use_osc, fr, fl, fg, gate_kind, gate_thr):
    n = states.shape[0]
    xbuf = np.zeros(25)
    flags = np.zeros(2)
    acc = 0.0
    for r in range(reps):
        i = r % n
        c, d = policy_hgb_one(states[i], T[i], mu, c_l1, use_osc, fr, fl, fg, gate_kind, gate_thr, xbuf, flags)
        acc += d + c
    return acc


@njit(cache=True)
def _loop_rhs(states, reps, mu):
    n = states.shape[0]
    acc = 0.0
    for r in range(reps):
        d = _cr3bp_derivatives_3d_numba(states[r % n], mu)
        acc += d[3] + d[4] + d[5]
    return acc


@njit(cache=True)
def _loop_features_only(states, T, reps, mu, c_l1, use_osc):
    n = states.shape[0]
    xbuf = np.zeros(25)
    flags = np.zeros(2)
    acc = 0.0
    for r in range(reps):
        i = r % n
        state_features_row(states[i], T[i], mu, c_l1, xbuf)
        if use_osc:
            osc_row(states[i, 0], states[i, 1], states[i, 2], states[i, 3], states[i, 4], states[i, 5], T[i], mu, xbuf[12:25], flags)
        acc += xbuf[11] + xbuf[min(24, 12 + 12 * int(use_osc))]
    return acc


def _median_ns(fn, reps_inner, repeats):
    fn(reps_inner)                    # warm-up (already compiled by the caller, this touches the caches)
    out = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        fn(reps_inner)
        out.append((time.perf_counter_ns() - t0) / reps_inner)
    return float(np.median(out)), [float(min(out)), float(max(out))]


def measure_single_call_cost(policy, states, T, n_policy=100_000, n_rhs=1_000_000, repeats=5):
    """RHS-equivalents per orbit of the compiled policy (see the module docstring for what 'single call' means)."""
    states = np.ascontiguousarray(states[:1000], dtype=np.float64)
    T = np.ascontiguousarray(T[:1000], dtype=np.float64)
    a = policy._args()
    _loop_policy(states, T, 10, *a)
    _loop_rhs(states, 10, policy.mu)
    _loop_features_only(states, T, 10, policy.mu, policy.c_l1, policy.use_osc)
    pol_ns, pol_rng = _median_ns(lambda n: _loop_policy(states, T, n, *a), n_policy, repeats)
    rhs_ns, rhs_rng = _median_ns(lambda n: _loop_rhs(states, n, policy.mu), n_rhs, repeats)
    feat_ns, feat_rng = _median_ns(lambda n: _loop_features_only(states, T, n, policy.mu, policy.c_l1, policy.use_osc), n_policy, repeats)
    # Python-dispatched single calls (a deployment that calls the policy from Python pays this on top)
    py = []
    for k in range(2000):
        t0 = time.perf_counter_ns()
        policy.one(states[k % len(states)], T[k % len(T)])
        py.append(time.perf_counter_ns() - t0)
    py_rhs = []
    for k in range(2000):
        t0 = time.perf_counter_ns()
        _cr3bp_derivatives_3d_numba(states[k % len(states)], policy.mu)
        py_rhs.append(time.perf_counter_ns() - t0)
    return {"policy_ns_per_call": pol_ns, "policy_ns_range": pol_rng, "rhs_ns_per_call": rhs_ns, "rhs_ns_range": rhs_rng,
            "features_ns_per_call": feat_ns, "features_ns_range": feat_rng,
            "rhs_equiv": pol_ns / rhs_ns, "rhs_equiv_range": [pol_rng[0] / rhs_rng[1], pol_rng[1] / rhs_rng[0]],
            "features_rhs_equiv": feat_ns / rhs_ns,
            "python_dispatch_ns_policy": float(np.median(py)), "python_dispatch_ns_rhs": float(np.median(py_rhs)),
            "python_dispatch_rhs_equiv": float(np.median(py) / np.median(py_rhs)), "repeats": repeats,
            "n_policy_calls": n_policy, "n_rhs_calls": n_rhs}


def batched_cost(policy, states, T, n=1000, repeats=5):
    """Batch evaluation of n rows (the cheaper number a batched deployment would see), RHS-equivalents per orbit."""
    s = np.ascontiguousarray(states[:n], dtype=np.float64)
    t = np.ascontiguousarray(T[:n], dtype=np.float64)
    policy.batch(s, t)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        policy.batch(s, t)
        ts.append((time.perf_counter_ns() - t0) / n)
    return float(np.median(ts))


# ---------------------------------------------------------------------------------------------------------------------
# M0: linear quantile policy on engineered features (one dot product per integrator)
# ---------------------------------------------------------------------------------------------------------------------

@njit(cache=True)
def linear_features_row(st, T, mu, c_l1, out):
    """33 engineered features: b = [log10 r1, log10 r2, log10 T, C - C_L1, the 13 periapsis features] (17 values) and the products
    b_i * log10 T for every i except log10 T itself (16 values)."""
    sf = np.empty(12)
    state_features_row(st, T, mu, c_l1, sf)
    of = np.empty(13)
    fl = np.zeros(2)
    osc_row(st[0], st[1], st[2], st[3], st[4], st[5], T, mu, of, fl)
    lt = math.log10(T)
    out[0] = math.log10(sf[8])
    out[1] = math.log10(sf[9])
    out[2] = lt
    out[3] = sf[11]
    for j in range(13):
        out[4 + j] = of[j]
    k = 17
    for j in range(17):
        if j == 2:
            continue
        out[k] = out[j] * lt
        k += 1


@njit(cache=True)
def linear_features_batch(states, T, mu, c_l1):
    out = np.empty((states.shape[0], 33))
    for i in range(states.shape[0]):
        linear_features_row(states[i], T[i], mu, c_l1, out[i])
    return out


@njit(cache=True)
def policy_linear_one(st, T, mu, c_l1, w_rk4, b_rk4, w_lf, b_lf, gate_thr, xbuf):
    """Censored-regression gate only (the leapfrog regressor was trained on all rows with the label 0.5 * dt_min where no step works)."""
    linear_features_row(st, T, mu, c_l1, xbuf)
    log_rk4 = b_rk4
    log_lf = b_lf
    for j in range(33):
        log_rk4 += w_rk4[j] * xbuf[j]
        log_lf += w_lf[j] * xbuf[j]
    d_rk4 = 10.0 ** log_rk4
    d_lf = 10.0 ** log_lf
    if (log_lf - math.log10(1e-4)) >= gate_thr and d_lf > d_rk4 / 4.0:
        return 1, d_lf
    return 0, d_rk4


@njit(cache=True)
def policy_linear_batch(states, T, mu, c_l1, w_rk4, b_rk4, w_lf, b_lf, gate_thr):
    n = states.shape[0]
    ch = np.empty(n, dtype=np.int64)
    dt = np.empty(n)
    xbuf = np.zeros(33)
    for i in range(n):
        c, d = policy_linear_one(states[i], T[i], mu, c_l1, w_rk4, b_rk4, w_lf, b_lf, gate_thr, xbuf)
        ch[i] = c
        dt[i] = d
    return ch, dt


@njit(cache=True)
def _loop_linear(states, T, reps, mu, c_l1, w_rk4, b_rk4, w_lf, b_lf, gate_thr):
    n = states.shape[0]
    xbuf = np.zeros(33)
    acc = 0.0
    for r in range(reps):
        i = r % n
        c, d = policy_linear_one(states[i], T[i], mu, c_l1, w_rk4, b_rk4, w_lf, b_lf, gate_thr, xbuf)
        acc += d + c
    return acc


def fit_linear_policy(train, quantile=0.1, mu=MU, alpha=0.0, max_rows=None, seed=0):
    """Two linear quantile regressors (HiGHS) on the 33 engineered features, leapfrog trained on ALL rows with the censored label."""
    from sklearn.linear_model import QuantileRegressor
    from .policy_c_3d import LOG_FLOOR_CENSORED
    c_l1 = c_l1_for_mu(mu)
    tr = train if max_rows is None or max_rows >= len(train) else train.sample(max_rows, random_state=seed)
    X = linear_features_batch(tr[["x", "y", "z", "vx", "vy", "vz"]].to_numpy(dtype=np.float64), tr["sim_time"].to_numpy(dtype=np.float64), mu, c_l1)
    mu_x, sd_x = X.mean(axis=0), X.std(axis=0) + 1e-12
    Z = (X - mu_x) / sd_x
    out = {"mu": mu, "c_l1": c_l1, "quantile": quantile}
    for k in ("rk4", "leapfrog"):
        succ = tr[f"{k}_success"].to_numpy() == 1
        y = np.where(succ, np.log10(np.where(succ, tr[f"dt_{k}"].to_numpy(dtype=float), 1.0)), LOG_FLOOR_CENSORED)
        use = succ if k == "rk4" else np.ones(len(tr), dtype=bool)
        reg = QuantileRegressor(quantile=quantile, alpha=alpha, solver="highs").fit(Z[use], y[use])
        w = reg.coef_ / sd_x                      # fold the standardization into the weights: raw features in, log10 dt out
        out[f"w_{k}"] = np.ascontiguousarray(w, dtype=np.float64)
        out[f"b_{k}"] = float(reg.intercept_ - np.dot(reg.coef_, mu_x / sd_x))
    return out


class CompiledLinearPolicy:
    def __init__(self, lin, g_star=0.0):
        self.lin, self.g_star = lin, float(g_star)
        self.mu, self.c_l1 = lin["mu"], lin["c_l1"]

    def _args(self):
        l = self.lin
        return (self.mu, self.c_l1, l["w_rk4"], l["b_rk4"], l["w_leapfrog"], l["b_leapfrog"], self.g_star)

    def batch(self, states, T):
        return policy_linear_batch(np.ascontiguousarray(states, dtype=np.float64), np.ascontiguousarray(T, dtype=np.float64), *self._args())

    def one(self, state, T):
        return policy_linear_one(np.ascontiguousarray(state, dtype=np.float64), float(T), *self._args(), np.zeros(33))

    def raw_log_dt(self, states, T):
        l = self.lin
        X = linear_features_batch(np.ascontiguousarray(states, dtype=np.float64), np.ascontiguousarray(T, dtype=np.float64), self.mu, self.c_l1)
        return X @ l["w_rk4"] + l["b_rk4"], X @ l["w_leapfrog"] + l["b_leapfrog"]


def measure_linear_cost(policy, states, T, n_policy=100_000, n_rhs=1_000_000, repeats=5):
    states = np.ascontiguousarray(states[:1000], dtype=np.float64)
    T = np.ascontiguousarray(T[:1000], dtype=np.float64)
    a = policy._args()
    _loop_linear(states, T, 10, *a)
    _loop_rhs(states, 10, policy.mu)
    pol_ns, pol_rng = _median_ns(lambda n: _loop_linear(states, T, n, *a), n_policy, repeats)
    rhs_ns, rhs_rng = _median_ns(lambda n: _loop_rhs(states, n, policy.mu), n_rhs, repeats)
    return {"policy_ns_per_call": pol_ns, "policy_ns_range": pol_rng, "rhs_ns_per_call": rhs_ns, "rhs_ns_range": rhs_rng,
            "rhs_equiv": pol_ns / rhs_ns, "rhs_equiv_range": [pol_rng[0] / rhs_rng[1], pol_rng[1] / rhs_rng[0]], "repeats": repeats}
