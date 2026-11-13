"""
Study E, draft section 7 item 4: event times of every method on doomed orbits, and the strict event rule of E4. Nothing here is used by a
sealed kernel.

Every method reports, when its run ends in a collision, an ejection or a non-finite state, the last good state and the first bad one with
their physical times. The event time is then located the same way for every method (policy and comparators alike): a cubic Hermite
interpolant of the position between the two states (the velocity is the derivative of the position, so both end states define it), and the
first root of the event function on it (collision with body k: r_k - rc_k; ejection: R_ej - |r|), by a 32-point scan and bisection.

  rk4s_event / rk8s_event      the Sundman-time fixed-step kernels (the sealed RK4 kernel of sundman_3d and the RK8 kernel of defense_runs),
                               copied line for line with the bad state kept (tests/test_study_e.py checks they equal the originals)
  rk_p_endpoint                physical-time RK45 (SciPy controller or the DOPRI5 PI controller) and DOP853, generic tableau, with the same
                               information (tests check the nfev equals compiled_solvers_d's kernels)
  hermite_event_time           the event location above
  reference_event              the reference event of an orbit: code, time and speed (DOP853 at 1e-13, physics_3d.reference_trajectory_dense_3d)
  strict_success               E4: same event type and |t_run - t_ref| <= tol / |v_ref|
  run_with_event               one orbit, any of the methods, -> dict(ok, final, evals, reason, event, t_event)

Event codes as the reference: 0 none (reached T), 1 primary collision, 2 secondary collision, 3 ejection, 9 other failure (non-finite,
step cap, too small a step, no event).
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

from .config import MU, EJECTION_RADIUS, ADAPTIVE_ATOL_RATIO, MAX_ADAPTIVE_NFEV, POSITION_TOL_KM, collision_radii, du_km_for_mu
from .physics_3d import REASON_OK, REASON_NONFINITE, REASON_STEP_CAP, REASON_COLLISION, REASON_EJECTION, _step_rk4_3d_numba
from .sundman_3d import _G, _bad_state, SUNDMAN_TAU_C
from .compiled_solvers_d import _rhs
from . import compiled_sundman_e as cs
from . import defense_runs as dr

__all__ = ["rk4s_event", "rk8s_event", "rk_p_endpoint", "hermite_event_time", "bad_code", "reference_event", "strict_success",
           "run_with_event", "METHODS"]

REASON_TOO_SMALL = cs.REASON_TOO_SMALL


# ---------------------------------------------------------------------------------------------------------------------
# fixed-step Sundman-time kernels with the bad state kept
# ---------------------------------------------------------------------------------------------------------------------

@njit(cache=True)
def _keep(info, t_good, Xg, t_bad, Xb):
    info[0] = t_good
    info[7] = t_bad
    for i in range(6):
        info[1 + i] = Xg[i]
        info[8 + i] = Xb[i]


@njit(cache=True)
def rk4s_event(state, ds, limit_t, mu, max_steps, rc1, rc2, rej, tau_c, info):
    """sundman_3d._integrate_endpoint_rk4s_numba (gfix = 0) with `info` filled on a bad state. Returns (ok, X, t, reason, n_s, n_f)."""
    X = state.copy()
    t = 0.0
    n_s = 0
    n_f = 0
    Y = np.empty(7)
    tmp = np.empty(7)
    k1 = np.empty(7)
    k2 = np.empty(7)
    k3 = np.empty(7)
    k4 = np.empty(7)
    while t < limit_t:
        if n_s + n_f >= max_steps:
            return False, X, t, REASON_STEP_CAP, n_s, n_f
        for i in range(6):
            Y[i] = X[i]
        Y[6] = t
        _G(Y, mu, tau_c, 0.0, k1)
        dt_pred = k1[6] * ds
        land = (t + 1.5 * dt_pred >= limit_t - 1e-12)
        if not land:
            for i in range(7):
                tmp[i] = Y[i] + 0.5 * ds * k1[i]
            _G(tmp, mu, tau_c, 0.0, k2)
            for i in range(7):
                tmp[i] = Y[i] + 0.5 * ds * k2[i]
            _G(tmp, mu, tau_c, 0.0, k3)
            for i in range(7):
                tmp[i] = Y[i] + ds * k3[i]
            _G(tmp, mu, tau_c, 0.0, k4)
            for i in range(7):
                tmp[i] = Y[i] + ds / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
            n_s += 1
            bad = _bad_state(tmp, mu, rc1, rc2, rej)
            if bad != 0:
                _keep(info, t, X, tmp[6], tmp)
                return False, X, t, bad, n_s, n_f
            if tmp[6] >= limit_t - 1e-12:
                land = True
            else:
                for i in range(6):
                    X[i] = tmp[i]
                t = tmp[6]
        if land:
            remaining = limit_t - t
            nf = max(1, int(math.ceil(remaining / dt_pred - 1e-9)))
            h = remaining / nf
            for j in range(nf):
                if n_s + n_f >= max_steps:
                    return False, X, t, REASON_STEP_CAP, n_s, n_f
                s_new = _step_rk4_3d_numba(X, h, mu)
                n_f += 1
                bad = _bad_state(s_new, mu, rc1, rc2, rej)
                if bad != 0:
                    _keep(info, t, X, t + h, s_new)
                    return False, X, t, bad, n_s, n_f
                for i in range(6):
                    X[i] = s_new[i]
                t = t + h
            t = limit_t
            break
    return True, X, t, REASON_OK, n_s, n_f


@njit(cache=True)
def rk8s_event(state, ds, limit_t, mu, max_steps, rc1, rc2, rej, tau_c, A, B, info):
    """defense_runs._integrate_endpoint_rk8s_numba (gfix = 0) with `info` filled on a bad state. Returns (ok, X, t, reason, n_s, n_f)."""
    ns = B.shape[0]
    X = state.copy()
    t = 0.0
    n_s = 0
    n_f = 0
    Y = np.empty(7)
    tmp = np.empty(7)
    out = np.empty(7)
    K = np.empty((ns, 7))
    g0 = np.empty(7)
    while t < limit_t:
        if n_s + n_f >= max_steps:
            return False, X, t, REASON_STEP_CAP, n_s, n_f
        for i in range(6):
            Y[i] = X[i]
        Y[6] = t
        _G(Y, mu, tau_c, 0.0, g0)
        dt_pred = g0[6] * ds
        land = (t + 1.5 * dt_pred >= limit_t - 1e-12)
        if not land:
            dr._rk8_step(Y, ds, mu, tau_c, 0.0, A, B, K, tmp, out)
            n_s += 1
            bad = _bad_state(out, mu, rc1, rc2, rej)
            if bad != 0:
                _keep(info, t, X, out[6], out)
                return False, X, t, bad, n_s, n_f
            if out[6] >= limit_t - 1e-12:
                land = True
            else:
                for i in range(6):
                    X[i] = out[i]
                t = out[6]
        if land:
            remaining = limit_t - t
            nf = max(1, int(math.ceil(remaining / dt_pred - 1e-9)))
            h = remaining / nf
            for j in range(nf):
                if n_s + n_f >= max_steps:
                    return False, X, t, REASON_STEP_CAP, n_s, n_f
                for i in range(6):
                    Y[i] = X[i]
                Y[6] = t
                dr._rk8_step(Y, h, mu, tau_c, 1.0, A, B, K, tmp, out)
                n_f += 1
                bad = _bad_state(out, mu, rc1, rc2, rej)
                if bad != 0:
                    _keep(info, t, X, t + h, out)
                    return False, X, t, bad, n_s, n_f
                for i in range(6):
                    X[i] = out[i]
                t = t + h
            t = limit_t
            break
    return True, X, t, REASON_OK, n_s, n_f


# ---------------------------------------------------------------------------------------------------------------------
# physical-time adaptive kernels (generic tableau) with the bad state kept
# ---------------------------------------------------------------------------------------------------------------------

@njit(cache=True)
def _initial_step_p(y, f0, T, rtol, atol, mu, order, tmp, f1):
    n = 6
    d0 = 0.0
    d1 = 0.0
    for i in range(n):
        sc = atol + abs(y[i]) * rtol
        d0 += (y[i] / sc) ** 2
        d1 += (f0[i] / sc) ** 2
    d0 = math.sqrt(d0 / n)
    d1 = math.sqrt(d1 / n)
    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * d0 / d1
    h0 = min(h0, T)
    for i in range(n):
        tmp[i] = y[i] + h0 * f0[i]
    _rhs(tmp, mu, f1)
    d2 = 0.0
    for i in range(n):
        sc = atol + abs(y[i]) * rtol
        d2 += ((f1[i] - f0[i]) / sc) ** 2
    d2 = math.sqrt(d2 / n) / h0
    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1.0 / (order + 1.0))
    return min(100.0 * h0, h1, T)


@njit(cache=True)
def rk_p_endpoint(y0, T, rtol, atol, mu, method, ctrl, max_nfev, rc1, rc2, rej, A, B, E, A8, B8, E3, E5, info):
    """Physical-time RK45 (method 0; ctrl 0 SciPy, 1 PI) or DOP853 (method 1) from 0 to T. Returns (ok, y_end, nfev, reason); `info` as above."""
    n = 6
    if method == 0:
        ns = B.shape[0]
        order = 4.0
        expo = -1.0 / 5.0
    else:
        ns = B8.shape[0]
        order = 7.0
        expo = -1.0 / 8.0
    K = np.empty((ns + 1, n))
    y = y0.copy()
    ynew = np.empty(n)
    tmp = np.empty(n)
    f1 = np.empty(n)
    _rhs(y, mu, K[0])
    nfev = 1
    h_abs = _initial_step_p(y, K[0], T, rtol, atol, mu, order, tmp, f1)
    nfev += 1
    t = 0.0
    facold = 1e-4
    while t < T:
        if nfev > max_nfev:
            return False, y, nfev, REASON_STEP_CAP
        min_step = 10.0 * abs(np.nextafter(t, np.inf) - t)
        if h_abs < min_step:
            h_abs = min_step
        rejected = False
        accepted = False
        t_new = t
        while not accepted:
            if h_abs < min_step:
                return False, y, nfev, REASON_TOO_SMALL
            t_new = t + h_abs
            if t_new > T:
                t_new = T
            h = t_new - t
            h_abs = abs(h)
            for st in range(1, ns):
                for i in range(n):
                    acc = 0.0
                    for j in range(st):
                        acc += K[j, i] * (A[st, j] if method == 0 else A8[st, j])
                    tmp[i] = y[i] + acc * h
                _rhs(tmp, mu, K[st])
            for i in range(n):
                acc = 0.0
                for j in range(ns):
                    acc += K[j, i] * (B[j] if method == 0 else B8[j])
                ynew[i] = y[i] + h * acc
            _rhs(ynew, mu, K[ns])
            nfev += ns
            if method == 0:
                en = 0.0
                for i in range(n):
                    acc = 0.0
                    for j in range(ns + 1):
                        acc += K[j, i] * E[j]
                    sc = atol + max(abs(y[i]), abs(ynew[i])) * rtol
                    en += (acc * h / sc) ** 2
                en = math.sqrt(en / n)
            else:
                e5 = 0.0
                e3 = 0.0
                for i in range(n):
                    a5 = 0.0
                    a3 = 0.0
                    for j in range(ns + 1):
                        a5 += K[j, i] * E5[j]
                        a3 += K[j, i] * E3[j]
                    sc = atol + max(abs(y[i]), abs(ynew[i])) * rtol
                    e5 += (a5 / sc) ** 2
                    e3 += (a3 / sc) ** 2
                if e5 == 0.0 and e3 == 0.0:
                    en = 0.0
                else:
                    en = abs(h) * e5 / math.sqrt((e5 + 0.01 * e3) * n)
            if en != en:
                return False, y, nfev, REASON_NONFINITE
            if ctrl == 0 or method != 0:
                if en < 1.0:
                    factor = 10.0 if en == 0.0 else min(10.0, 0.9 * en ** expo)
                    if rejected:
                        factor = min(1.0, factor)
                    h_abs *= factor
                    accepted = True
                else:
                    h_abs *= max(0.2, 0.9 * en ** expo)
                    rejected = True
            else:
                fac11 = en ** (0.2 - 0.04 * 0.75)
                if en <= 1.0:
                    fac = fac11 / facold ** 0.04
                    fac = max(0.1, min(5.0, fac / 0.9))
                    hn = h_abs / fac
                    facold = max(en, 1e-4)
                    if rejected:
                        hn = min(hn, h_abs)
                    h_abs = hn
                    accepted = True
                else:
                    h_abs = h_abs / min(5.0, fac11 / 0.9)
                    rejected = True
            if not accepted and nfev > max_nfev:
                return False, y, nfev, REASON_STEP_CAP
        bad = _bad_state(ynew, mu, rc1, rc2, rej)
        if bad != 0:
            _keep(info, t, y, t_new, ynew)
            return False, ynew.copy(), nfev, bad
        for i in range(n):
            y[i] = ynew[i]
            K[0, i] = K[ns, i]
        t = t_new
    return True, y, nfev, REASON_OK


# ---------------------------------------------------------------------------------------------------------------------
# event location and the reference event
# ---------------------------------------------------------------------------------------------------------------------

@njit(cache=True)
def bad_code(X, mu, rc1, rc2, rej):
    """The event code of a bad state, in _bad_state's order: 9 non-finite, 1 / 2 collision with the primary / secondary, 3 ejection, 0 none."""
    for i in range(6):
        if not math.isfinite(X[i]):
            return 9
    if math.sqrt((X[0] + mu) ** 2 + X[1] ** 2 + X[2] ** 2) < rc1:
        return 1
    if math.sqrt((X[0] + mu - 1.0) ** 2 + X[1] ** 2 + X[2] ** 2) < rc2:
        return 2
    if math.sqrt(X[0] ** 2 + X[1] ** 2 + X[2] ** 2) > rej:
        return 3
    return 0


@njit(cache=True)
def _event_fn(code, p, mu, rc1, rc2, rej):
    if code == 1:
        return math.sqrt((p[0] + mu) ** 2 + p[1] ** 2 + p[2] ** 2) - rc1
    if code == 2:
        return math.sqrt((p[0] + mu - 1.0) ** 2 + p[1] ** 2 + p[2] ** 2) - rc2
    return rej - math.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2)


@njit(cache=True)
def _hermite_pos(X0, X1, dt, th, out):
    h00 = 2.0 * th ** 3 - 3.0 * th ** 2 + 1.0
    h10 = th ** 3 - 2.0 * th ** 2 + th
    h01 = -2.0 * th ** 3 + 3.0 * th ** 2
    h11 = th ** 3 - th ** 2
    for i in range(3):
        out[i] = h00 * X0[i] + h10 * dt * X0[3 + i] + h01 * X1[i] + h11 * dt * X1[3 + i]


@njit(cache=True)
def hermite_event_time(code, t0, X0, t1, X1, mu, rc1, rc2, rej):
    """First root in [t0, t1] of the event function of `code` on the cubic Hermite interpolant of the position. Returns t1 if the interpolant
    has no sign change (the endpoint test fired but the interpolant stays outside: the step end is then the best estimate)."""
    dt = t1 - t0
    if dt <= 0.0:
        return t1
    p = np.empty(3)
    prev_th = 0.0
    _hermite_pos(X0, X1, dt, 0.0, p)
    if _event_fn(code, p, mu, rc1, rc2, rej) <= 0.0:
        return t0
    lo = -1.0
    hi = -1.0
    for k in range(1, 33):
        th = k / 32.0
        _hermite_pos(X0, X1, dt, th, p)
        if _event_fn(code, p, mu, rc1, rc2, rej) <= 0.0:
            lo = prev_th
            hi = th
            break
        prev_th = th
    if hi < 0.0:
        return t1
    for _it in range(80):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        _hermite_pos(X0, X1, dt, mid, p)
        if _event_fn(code, p, mu, rc1, rc2, rej) <= 0.0:
            hi = mid
        else:
            lo = mid
    return t0 + hi * dt


def reference_event(state, T, mu=MU):
    """(code, t_event, speed at the event) of the DOP853 reference (code 0: reached T, t_event = T)."""
    from .physics_3d import reference_trajectory_dense_3d
    sol, _reached = reference_trajectory_dense_3d(np.asarray(state, dtype=np.float64), float(T), mu)
    te = float(sol.t[-1])
    v = float(np.linalg.norm(np.asarray(sol.sol(te))[3:6]))
    return int(sol.reference_event), te, v


def strict_success(run_event, run_t, ref_code, ref_t, ref_speed, tol_km=POSITION_TOL_KM, mu=MU):
    """E4's rule for a doomed orbit: the same event type and |t_run - t_ref| <= tol / |v_ref| (tol in the system's length unit)."""
    if ref_code in (0, 9) or run_event != ref_code:
        return False
    return bool(abs(run_t - ref_t) <= (tol_km / du_km_for_mu(mu)) / max(ref_speed, 1e-12))


# ---------------------------------------------------------------------------------------------------------------------
# one orbit, any method
# ---------------------------------------------------------------------------------------------------------------------

METHODS = ("rk4s", "rk8s", "DP54PI", "DOP853", "RK45", "RK45s", "DP54PIs", "DOP853s")
_TAB = None


def _tab():
    global _TAB
    if _TAB is None:
        rk, d8 = cs.tableaux()
        _TAB = (rk, d8)
    return _TAB


def run_with_event(method, state, T, param, mu=MU, tau_c=SUNDMAN_TAU_C, atol_ratio=ADAPTIVE_ATOL_RATIO, max_nfev=MAX_ADAPTIVE_NFEV,
                   max_steps=4_000_000):
    """param = fictitious step ds (rk4s, rk8s) or rtol (adaptive). Returns dict(ok, final (6), evals, reason, event, t_event); event 0 and
    t_event = T when the run reached the horizon."""
    rc1, rc2 = collision_radii(mu)
    rej = EJECTION_RADIUS
    st = np.ascontiguousarray(state, dtype=np.float64)
    info = np.full(14, np.nan)
    rk, d8 = _tab()
    if method == "rk4s":
        ok, X, _t, reason, n_s, n_f = rk4s_event(st, float(param), float(T), mu, int(max_steps), rc1, rc2, rej, float(tau_c), info)
        evals = 4 * (n_s + n_f)
    elif method == "rk8s":
        ok, X, _t, reason, n_s, n_f = rk8s_event(st, float(param), float(T), mu, int(max_steps), rc1, rc2, rej, float(tau_c), dr._A8, dr._B8, info)
        evals = dr.RK8_STAGES * (n_s + n_f)
    elif method in ("DP54PI", "DOP853", "RK45"):
        m, c = {"DP54PI": (0, 1), "DOP853": (1, 0), "RK45": (0, 0)}[method]
        rt = float(param)
        ok, X, evals, reason = rk_p_endpoint(st, float(T), rt, rt * atol_ratio, mu, m, c, int(max_nfev), rc1, rc2, rej,
                                             rk["A"], rk["B"], rk["E"], np.ascontiguousarray(d8["A"][:12, :12]), d8["B"], d8["E3"], d8["E5"], info)
    elif method in cs.SundmanAdaptive.SPECS:
        m, c = cs.SundmanAdaptive.SPECS[method]
        rt = float(param)
        Y0 = np.concatenate([st, [0.0]])
        ok, X, evals, _a, _r, _s, reason = cs.rk_s_endpoint(Y0, float(T), rt, rt * atol_ratio, mu, float(tau_c), m, c, int(max_nfev), True,
                                                             rc1, rc2, rej, rk["A"], rk["B"], rk["C"], rk["E"], rk["P"], d8["A"], d8["B"], d8["C"],
                                                             d8["E3"], d8["E5"], d8["D"], info)
    else:
        raise ValueError(method)
    ok = bool(ok)
    out = {"ok": ok, "final": np.asarray(X, dtype=float).copy(), "evals": float(evals), "reason": int(reason), "event": 0, "t_event": float(T)}
    if ok:
        return out
    if reason in (REASON_COLLISION, REASON_EJECTION) and np.isfinite(info[0]):
        code = int(bad_code(info[8:14], mu, rc1, rc2, rej))
        out["event"] = code
        out["t_event"] = float(hermite_event_time(code, info[0], info[1:7].copy(), info[7], info[8:14].copy(), mu, rc1, rc2, rej))
    else:
        out["event"] = 9
        out["t_event"] = float("nan")
    return out
