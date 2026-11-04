"""
Kernels for the exploratory defense-plan comparators (posthoc/README.md P20-P23). Nothing here is used by a sealed kernel.

  run_sundman_adaptive     SciPy RK45 / DOP853 on the seven-dimensional Sundman-time system (X, t) in s, ending at the event t(s) = T (P20)
  sundman_adaptive_rows    the same on many rows in a process pool
  scipy_rows_per_row_rtol  SciPy physical-time solver with one rtol per row (P21's learned tolerance; any SciPy method, e.g. LSODA for P22)
  integrate_endpoint_rk8s  fixed-step 8th-order Runge-Kutta (DOP853's 12-stage propagation formula, no error control) in Sundman time, with the
                           RK4 kernel's landing rule (sundman_3d.py), landing steps by the same formula in physical time (P23)
  run_fixed_s, oracle_ds   one fixed-step run (rk4 = the sealed sundman_3d kernel, rk8) and the two-octave oracle step on a 2^(1/4) grid
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit
from scipy.integrate._ivp import dop853_coefficients as _d853

from .config import MU, EJECTION_RADIUS, MAX_ADAPTIVE_NFEV, ADAPTIVE_ATOL_RATIO, POSITION_TOL_KM, collision_radii, du_km_for_mu
from .physics_3d import REASON_OK, REASON_STEP_CAP
from .sundman_3d import SUNDMAN_TAU_C, _G, _bad_state, integrate_endpoint_rk4s

__all__ = ["run_sundman_adaptive", "sundman_adaptive_rows", "scipy_rows_per_row_rtol", "integrate_endpoint_rk8s", "run_fixed_s", "oracle_ds",
           "S_MAX", "ORACLE_GRID"]

S_MAX = 1e7                                             # upper end of the fictitious-time span (never reached: the event ends every run)
ORACLE_GRID = tuple(4.0 * 2.0 ** (-k / 4.0) for k in range(62))     # 4 down to 1.07e-4, ratio 2^(1/4)
RK8_STAGES = _d853.N_STAGES
_A8 = np.ascontiguousarray(_d853.A[:RK8_STAGES, :RK8_STAGES], dtype=np.float64)
_B8 = np.ascontiguousarray(_d853.B, dtype=np.float64)


class _Cap(Exception):
    pass


def run_sundman_adaptive(state, T, ref_pos, rtol, mu=MU, method="RK45", atol_ratio=ADAPTIVE_ATOL_RATIO, tau_c=SUNDMAN_TAU_C, cap=MAX_ADAPTIVE_NFEV):
    """dict(ok, err_km, nfev, capped, t_end). ok is False when the solve failed, hit the cap, ended without reaching t = T or is non-finite."""
    from scipy.integrate import solve_ivp
    count = [0]

    def fun(s, Y):
        count[0] += 1
        if count[0] > cap:
            raise _Cap()
        out = np.empty(7)
        _G(Y, mu, tau_c, 0.0, out)
        return out

    def ev(s, Y):
        return Y[6] - T
    ev.terminal = True
    ev.direction = 1.0
    Y0 = np.concatenate([np.asarray(state, dtype=np.float64), [0.0]])
    try:
        sol = solve_ivp(fun, (0.0, S_MAX), Y0, method=method, rtol=rtol, atol=rtol * atol_ratio, events=ev)
    except _Cap:
        return {"ok": False, "err_km": float("nan"), "nfev": float("nan"), "capped": True, "t_end": float("nan")}
    if sol.status != 1 or len(sol.t_events[0]) == 0:
        return {"ok": False, "err_km": float("nan"), "nfev": float("nan"), "capped": False, "t_end": float("nan")}
    Ye = sol.y_events[0][0]
    if not np.all(np.isfinite(Ye)):
        return {"ok": False, "err_km": float("nan"), "nfev": float("nan"), "capped": False, "t_end": float("nan")}
    err = float(np.linalg.norm(Ye[:3] - np.asarray(ref_pos, dtype=float))) * du_km_for_mu(mu)
    return {"ok": True, "err_km": err, "nfev": float(count[0]), "capped": False, "t_end": float(Ye[6]), "nfev_solver": int(sol.nfev)}


def _sa_job(a):
    state, T, ref, rtol, mu, method = a
    return run_sundman_adaptive(state, T, ref, rtol, mu, method)


def _collect(res):
    ok = np.array([r["ok"] for r in res], dtype=bool)
    err = np.array([r["err_km"] for r in res], dtype=float)
    nfev = np.array([r["nfev"] for r in res], dtype=float)
    capped = np.array([r["capped"] for r in res], dtype=bool)
    return {"ok": ok, "err": err, "nfev": nfev, "capped": capped, "met": ok & (err <= POSITION_TOL_KM)}


def sundman_adaptive_rows(pool, states, T, ref_pos, rtol, method, mu=MU, n_jobs=8):
    jobs = [(states[i], float(T[i]), ref_pos[i], float(rtol), mu, method) for i in range(len(states))]
    res = pool.map(_sa_job, jobs, chunksize=max(1, len(jobs) // (8 * n_jobs))) if pool is not None else [_sa_job(j) for j in jobs]
    return _collect(res)


def scipy_rows_per_row_rtol(pool, states, T, ref_pos, rtols, method, atol_ratio=ADAPTIVE_ATOL_RATIO, mu=MU, n_jobs=8):
    """symmetric_cert.scipy_rows with one rtol per row (physical time)."""
    from .symmetric_cert import _job
    jobs = [(states[i], float(T[i]), ref_pos[i], float(rtols[i]), mu, method, atol_ratio) for i in range(len(states))]
    res = pool.map(_job, jobs, chunksize=max(1, len(jobs) // (8 * n_jobs))) if pool is not None else [_job(j) for j in jobs]
    ok = np.array([r[0] for r in res], dtype=bool)
    err = np.array([r[1] for r in res], dtype=float)
    nfev = np.array([r[2] for r in res], dtype=float)
    capped = np.array([r[3] for r in res], dtype=bool)
    return {"ok": ok, "err": err, "nfev": nfev, "capped": capped, "met": ok & (err <= POSITION_TOL_KM)}


@njit(cache=True)
def _rk8_step(Y, h, mu, tau_c, gfix, A, B, K, tmp, out):
    """One step of the 12-stage 8th-order formula on the 7-vector Y (gfix = 1: physical time)."""
    ns = B.shape[0]
    _G(Y, mu, tau_c, gfix, K[0])
    for s in range(1, ns):
        for i in range(7):
            acc = 0.0
            for j in range(s):
                acc += A[s, j] * K[j, i]
            tmp[i] = Y[i] + h * acc
        _G(tmp, mu, tau_c, gfix, K[s])
    for i in range(7):
        acc = 0.0
        for j in range(ns):
            acc += B[j] * K[j, i]
        out[i] = Y[i] + h * acc


@njit(cache=True)
def _integrate_endpoint_rk8s_numba(state, ds, limit_t, mu, max_steps, rc1, rc2, rej, tau_c, gfix, A, B):
    """Returns (ok, prev, curr, t_prev, t, reason, n_s, n_f), with the landing rule of sundman_3d._integrate_endpoint_rk4s_numba."""
    ns = B.shape[0]
    X = state.copy()
    prev = state.copy()
    t = 0.0
    t_prev = 0.0
    n_s = 0
    n_f = 0
    Y = np.empty(7)
    tmp = np.empty(7)
    out = np.empty(7)
    K = np.empty((ns, 7))
    g0 = np.empty(7)
    while t < limit_t:
        if n_s + n_f >= max_steps:
            return False, prev, X, t_prev, t, REASON_STEP_CAP, n_s, n_f
        for i in range(6):
            Y[i] = X[i]
        Y[6] = t
        _G(Y, mu, tau_c, gfix, g0)
        dt_pred = g0[6] * ds
        land = (t + 1.5 * dt_pred >= limit_t - 1e-12)
        if not land:
            _rk8_step(Y, ds, mu, tau_c, gfix, A, B, K, tmp, out)
            n_s += 1
            bad = _bad_state(out, mu, rc1, rc2, rej)
            if bad != 0:
                return False, prev, X, t_prev, t, bad, n_s, n_f
            if out[6] >= limit_t - 1e-12:
                land = True
            else:
                for i in range(6):
                    prev[i] = X[i]
                    X[i] = out[i]
                t_prev = t
                t = out[6]
        if land:
            remaining = limit_t - t
            nf = max(1, int(math.ceil(remaining / dt_pred - 1e-9)))
            h = remaining / nf
            for j in range(nf):
                if n_s + n_f >= max_steps:
                    return False, prev, X, t_prev, t, REASON_STEP_CAP, n_s, n_f
                for i in range(6):
                    Y[i] = X[i]
                Y[6] = t
                _rk8_step(Y, h, mu, tau_c, 1.0, A, B, K, tmp, out)
                n_f += 1
                bad = _bad_state(out, mu, rc1, rc2, rej)
                if bad != 0:
                    return False, prev, X, t_prev, t, bad, n_s, n_f
                for i in range(6):
                    prev[i] = X[i]
                    X[i] = out[i]
                t_prev = t
                t = t + h
            t = limit_t
            break
    return True, prev, X, t_prev, t, REASON_OK, n_s, n_f


def integrate_endpoint_rk8s(state, ds, limit_t, mu=MU, max_steps=4_000_000, tau_c=None, gfix=0.0):
    """(ok, prev, curr, t_prev, t, reason, n_s, n_f); evals = 12 * (n_s + n_f). gfix = 1 integrates in physical time with step ds."""
    rc1, rc2 = collision_radii(mu)
    return _integrate_endpoint_rk8s_numba(np.array(state, dtype=np.float64), float(ds), float(limit_t), float(mu), int(max_steps), rc1, rc2,
                                          EJECTION_RADIUS, float(SUNDMAN_TAU_C if tau_c is None else tau_c), float(gfix), _A8, _B8)


def run_fixed_s(kind, state, ds, T, mu=MU, max_steps=4_000_000):
    """One Sundman-time fixed-step run: dict(ok, final (6,), evals)."""
    if kind == "rk4":
        r = integrate_endpoint_rk4s(state, ds, T, mu, max_steps=max_steps)
        stages = 4
    elif kind == "rk8":
        r = integrate_endpoint_rk8s(state, ds, T, mu, max_steps=max_steps)
        stages = RK8_STAGES
    else:
        raise ValueError(kind)
    return {"ok": bool(r[0]), "final": np.asarray(r[2]), "evals": int(stages * (r[6] + r[7]))}


def oracle_ds(kind, state, T, ref_pos, mu=MU, grid=ORACLE_GRID, tol_km=POSITION_TOL_KM, window=8):
    """The largest grid value that passes together with its `window` finer neighbours (grid ordered from coarse to fine); walks from the
    coarsest value and stops at the first such run. Returns dict(ds, evals at ds, censored = the coarsest value is the label, n_runs)."""
    du = du_km_for_mu(mu)
    passed = {}

    def ok(k):
        if k not in passed:
            r = run_fixed_s(kind, state, grid[k], T, mu)
            e = float(np.linalg.norm(r["final"][:3] - ref_pos)) * du if r["ok"] else float("inf")
            passed[k] = (r["ok"] and e <= tol_km, r["evals"])
        return passed[k][0]

    for k in range(len(grid) - window):
        if all(ok(j) for j in range(k, k + window + 1)):
            return {"ds": float(grid[k]), "evals": int(passed[k][1]), "censored": k == 0, "n_runs": len(passed)}
    return {"ds": float("nan"), "evals": float("nan"), "censored": False, "n_runs": len(passed)}
