"""
Study E pilots (posthoc/README.md P24-P25; exploratory): the 8th-order fixed step in Sundman time as a learned policy. Nothing here is used by a
sealed kernel; the sealed modules (search_3d, sundman_3d, compiled_solvers_d, fair_baseline_3d) are imported, never changed.

  Rk8sRunner                 the RK8 Sundman-time kernel of defense_runs (P23) at given per-orbit fictitious steps, with the interface of
                             compiled_solvers_d.SundmanRunner (.run_ds, .errors_km), so deploy_d.run_rows / certify_scale work unchanged
  monotone_search            the sealed monotone label search of search_3d.find_dt_for_integrator_accuracy_monotone_3d (grid ratio 2^(1/4),
                             window 8, six refinement points), written over a generic error function; tests/test_study_e.py checks that it
                             reproduces the sealed RK4 Sundman labels
  rk8s_label                 the E7 label of one orbit: largest RK8 fictitious step that meets the tolerance, ds in [1e-4, 64]
  sundman_matched_accuracy   the oracle-tuned rtol search of fair_baseline_3d.scipy_matched_accuracy (the rule of RK45_spec / DOP853_spec),
                             applied to the Sundman-time adaptive kernel of P20 (defense_runs.run_sundman_adaptive)
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from .config import MU, EJECTION_RADIUS, ADAPTIVE_ATOL_RATIO, POSITION_TOL_KM, collision_radii, du_km_for_mu
from .physics_3d import REASON_OK, REASON_ACCURACY_FLOOR
from .sundman_3d import SUNDMAN_TAU_C
from . import defense_runs as dr

__all__ = ["RK8_DS_MIN", "RK8_DS_MAX", "RK8_STAGES", "Rk8sRunner", "rk8s_batch", "monotone_search", "rk8s_label", "rk8s_label_job",
           "sundman_matched_accuracy", "sundman_matched_job"]

RK8_DS_MIN = 1e-4
RK8_DS_MAX = 64.0           # P23 found 40 of 300 labels censored at 4; P24 / P25 search up to 64 (2^6)
RK8_STAGES = dr.RK8_STAGES  # 12 RHS evaluations per step


@njit(cache=True)
def _rk8s_one(st, ds, T, mu, tau_c, max_steps, rc1, rc2, rej, A, B):
    r = dr._integrate_endpoint_rk8s_numba(st.copy(), ds, T, mu, max_steps, rc1, rc2, rej, tau_c, 0.0, A, B)
    return r[0], r[2], A.shape[0] * (r[6] + r[7]), r[5]


@njit(parallel=True, cache=True)
def rk8s_batch(states, T, ds, mu, tau_c, max_steps, rc1, rc2, rej, A, B):
    """Parallel batch of RK8 Sundman-time runs: (ok, y_end (n, 6), evals, reason), as compiled_solvers_d.rk4s_fixed_batch_par."""
    n = states.shape[0]
    ok = np.zeros(n, dtype=np.bool_)
    y = np.empty((n, 6))
    ev = np.zeros(n)
    rs = np.zeros(n, dtype=np.int64)
    for i in prange(n):
        o, yy, e, r = _rk8s_one(states[i], ds[i], T[i], mu, tau_c, max_steps, rc1, rc2, rej, A, B)
        ok[i] = o
        for j in range(6):
            y[i, j] = yy[j]
        ev[i] = e
        rs[i] = r
    return ok, y, ev, rs


class Rk8sRunner:
    """RK8 in Sundman time at given per-orbit fictitious steps: .run_ds(states, T, ds, parallel) -> (ok, y_end, evals, reason)."""

    def __init__(self, mu=MU, tau_c=SUNDMAN_TAU_C, max_steps=4_000_000):
        self.mu, self.tau_c, self.max_steps = float(mu), float(tau_c), int(max_steps)
        self.rc1, self.rc2 = collision_radii(mu)
        self.rej = EJECTION_RADIUS
        self.du_km = du_km_for_mu(mu)

    def run_ds(self, states, T, ds, parallel=True):
        return rk8s_batch(np.ascontiguousarray(states, dtype=np.float64), np.ascontiguousarray(T, dtype=np.float64),
                          np.ascontiguousarray(ds, dtype=np.float64), self.mu, self.tau_c, self.max_steps, self.rc1, self.rc2, self.rej,
                          dr._A8, dr._B8)

    def errors_km(self, y, ref_pos):
        return np.linalg.norm(y[:, :3] - ref_pos, axis=1) * self.du_km


def monotone_search(err_fn, dt_min, dt_max, tol_km=POSITION_TOL_KM, grid_ratio=2.0 ** 0.25, window=8, n_refine=6):
    """search_3d.find_dt_for_integrator_accuracy_monotone_3d without the reference integration: err_fn(dt) -> (err_km or None, reason).
    Returns dict(dt, censored, reason, pos_err_km)."""
    cache = {}

    def _eval(d):
        k = float(d)
        if k not in cache:
            cache[k] = err_fn(k)
        return cache[k]

    def _ok(d):
        e, _r = _eval(d)
        return e is not None and e <= tol_km

    n_grid = max(int(np.floor(np.log(dt_max / dt_min) / np.log(grid_ratio))) + 1, 1)
    grid = [dt_max * grid_ratio ** (-k) for k in range(n_grid)]
    best_k, k = None, 0
    while k < n_grid:
        fail_j = None
        for j in range(k, min(k + window, n_grid - 1) + 1):
            if not _ok(grid[j]):
                fail_j = j
                break
        if fail_j is None:
            best_k = k
            break
        k = fail_j + 1
    if best_k is None:
        last = _eval(grid[-1])[1]
        return {"dt": dt_min, "censored": False, "reason": REASON_ACCURACY_FLOOR if last == REASON_OK else last, "pos_err_km": float("nan")}
    chosen = grid[best_k]
    if best_k > 0:
        upper = grid[best_k - 1]
        for i in range(1, n_refine + 1):
            cand = chosen * (upper / grid[best_k]) ** (i / (n_refine + 1.0))
            if _ok(cand):
                chosen = cand
            else:
                break
    chosen = float(chosen)
    return {"dt": chosen, "censored": bool(best_k == 0), "reason": REASON_OK, "pos_err_km": float(_eval(chosen)[0])}


def rk8s_label(state, T, ref_pos, mu=MU, tol_km=POSITION_TOL_KM, ds_min=RK8_DS_MIN, ds_max=RK8_DS_MAX):
    """dict(dt_rk8s, rk8s_success, rk8s_censored, rk8s_reason, rk8s_evals) for one orbit (evals at the label)."""
    du = du_km_for_mu(mu)
    st = np.asarray(state, dtype=np.float64)
    ref_pos = np.asarray(ref_pos, dtype=np.float64)
    evals = {}

    def err_fn(ds):
        r = dr.integrate_endpoint_rk8s(st, ds, T, mu)
        evals[ds] = RK8_STAGES * (r[6] + r[7])
        if not r[0]:
            return None, int(r[5])
        return float(np.linalg.norm(np.asarray(r[2])[:3] - ref_pos)) * du, REASON_OK

    s = monotone_search(err_fn, ds_min, ds_max, tol_km)
    ok = s["reason"] == REASON_OK
    return {"dt_rk8s": s["dt"], "rk8s_success": int(ok), "rk8s_censored": bool(s["censored"]), "rk8s_reason": int(s["reason"]),
            "rk8s_err_km": s["pos_err_km"], "rk8s_evals": float(evals[s["dt"]]) if ok else float("nan")}


def rk8s_label_job(a):
    state, T, ref = a
    return rk8s_label(state, float(T), ref)


def sundman_matched_accuracy(state, T, ref_pos, target_err_km=POSITION_TOL_KM, mu=MU, method="RK45", rtol_tight=1e-13, rtol_loose=1e-1,
                             max_iter=30, atol_ratio=ADAPTIVE_ATOL_RATIO):
    """fair_baseline_3d.scipy_matched_accuracy with the Sundman-time kernel: the loosest rtol (log bisection) whose run meets the target.
    Returns dict(rtol, nfev, err_km) or None if even rtol_tight fails."""

    def run(rt):
        r = dr.run_sundman_adaptive(state, T, ref_pos, rt, mu, method, atol_ratio=atol_ratio)
        return None if not r["ok"] else {"rtol": rt, "nfev": r["nfev"], "err_km": r["err_km"]}

    res = run(rtol_loose)
    if res is not None and res["err_km"] <= target_err_km:
        return res
    best = run(rtol_tight)
    if best is None or best["err_km"] > target_err_km:
        return None
    lo, hi = math.log10(rtol_tight), math.log10(rtol_loose)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        res = run(10.0 ** mid)
        if res is not None and res["err_km"] <= target_err_km:
            best, lo = res, mid
        else:
            hi = mid
        if hi - lo < 0.02:
            break
    return best


def sundman_matched_job(a):
    state, T, ref, method, atol_ratio = a
    return sundman_matched_accuracy(state, float(T), ref, method=method, atol_ratio=atol_ratio)
