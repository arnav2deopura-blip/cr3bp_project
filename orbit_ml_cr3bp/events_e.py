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
from .compiled_sundman_e import ck_record
from . import defense_runs as dr

__all__ = ["rk4s_event", "rk8s_event", "rk_p_endpoint", "hermite_event_time", "bad_code", "reference_event", "strict_success",
           "run_with_event", "METHODS"]

REASON_TOO_SMALL = cs.REASON_TOO_SMALL


# ---------------------------------------------------------------------------------------------------------------------
# fixed-step Sundman-time kernels with the bad state kept
# ---------------------------------------------------------------------------------------------------------------------

@njit(cache=True)
def seg_hit(Xa, Xb, mu, rc1, rc2):
    """Item 7 (collision_segment.py, compiled): 1 / 2 if the straight chord between the positions of Xa and Xb comes within rc1 / rc2 of the
    primary / secondary, else 0. Costs no RHS evaluation."""
    d0 = Xb[0] - Xa[0]
    d1 = Xb[1] - Xa[1]
    d2 = Xb[2] - Xa[2]
    dd = d0 * d0 + d1 * d1 + d2 * d2
    for body in (1, 2):
        cx = -mu if body == 1 else 1.0 - mu
        rc = rc1 if body == 1 else rc2
        tt = 0.0
        if dd > 0.0:
            tt = ((cx - Xa[0]) * d0 + (0.0 - Xa[1]) * d1 + (0.0 - Xa[2]) * d2) / dd
            tt = min(1.0, max(0.0, tt))
        qx = Xa[0] + tt * d0 - cx
        qy = Xa[1] + tt * d1
        qz = Xa[2] + tt * d2
        if math.sqrt(qx * qx + qy * qy + qz * qz) < rc:
            return body
    return 0


@njit(cache=True)
def _keep(info, t_good, Xg, t_bad, Xb):
    info[0] = t_good
    info[7] = t_bad
    info[14] = np.nan
    for i in range(6):
        info[1 + i] = Xg[i]
        info[8 + i] = Xb[i]


@njit(cache=True)
def rk4s_event(state, ds, limit_t, mu, max_steps, rc1, rc2, rej, tau_c, seg, info, ck_t, ck_out):
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
            if seg and tmp[6] < limit_t - 1e-12:
                hb = seg_hit(X, tmp, mu, rc1, rc2)
                if hb != 0:
                    _keep(info, t, X, tmp[6], tmp)
                    info[14] = hb
                    return False, X, t, REASON_COLLISION, n_s, n_f
            if tmp[6] >= limit_t - 1e-12:
                land = True
            else:
                ck_record(t, X, tmp[6], tmp, mu, ck_t, ck_out)
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
                if seg:
                    hb = seg_hit(X, s_new, mu, rc1, rc2)
                    if hb != 0:
                        _keep(info, t, X, t + h, s_new)
                        info[14] = hb
                        return False, X, t, REASON_COLLISION, n_s, n_f
                t_next = limit_t if j == nf - 1 else t + h
                ck_record(t, X, t_next, s_new, mu, ck_t, ck_out)
                for i in range(6):
                    X[i] = s_new[i]
                t = t + h
            t = limit_t
            break
    return True, X, t, REASON_OK, n_s, n_f


@njit(cache=True)
def rk8s_event(state, ds, limit_t, mu, max_steps, rc1, rc2, rej, tau_c, A, B, seg, info, ck_t, ck_out):
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
            if seg and out[6] < limit_t - 1e-12:
                hb = seg_hit(X, out, mu, rc1, rc2)
                if hb != 0:
                    _keep(info, t, X, out[6], out)
                    info[14] = hb
                    return False, X, t, REASON_COLLISION, n_s, n_f
            if out[6] >= limit_t - 1e-12:
                land = True
            else:
                ck_record(t, X, out[6], out, mu, ck_t, ck_out)
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
                if seg:
                    hb = seg_hit(X, out, mu, rc1, rc2)
                    if hb != 0:
                        _keep(info, t, X, t + h, out)
                        info[14] = hb
                        return False, X, t, REASON_COLLISION, n_s, n_f
                t_next = limit_t if j == nf - 1 else t + h
                ck_record(t, X, t_next, out, mu, ck_t, ck_out)
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
def rk_p_endpoint(y0, T, rtol, atol, mu, method, ctrl, max_nfev, rc1, rc2, rej, A, B, E, A8, B8, E3, E5, seg, info, ck_t, ck_out):
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
        if seg:
            hb = seg_hit(y, ynew, mu, rc1, rc2)
            if hb != 0:
                _keep(info, t, y, t_new, ynew)
                info[14] = hb
                return False, ynew.copy(), nfev, REASON_COLLISION
        ck_record(t, y, t_new, ynew, mu, ck_t, ck_out)
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


def checkpoint_times(T, K=20):
    """E3's K checkpoints: t_k = k T / K, k = 1 ... K (the last is the horizon)."""
    t = np.arange(1, K + 1, dtype=float) * (float(T) / K)
    t[-1] = float(T)                      # exactly the horizon (K * (T / K) can exceed T by one ulp and would never be recorded)
    return t


def reference_checkpoints(state, T, K=20, mu=MU):
    """(K, 3) reference positions at checkpoint_times(T, K), or None if the reference does not reach T."""
    from .physics_3d import reference_trajectory_dense_3d
    sol, reached = reference_trajectory_dense_3d(np.asarray(state, dtype=np.float64), float(T), mu)
    if not reached:
        return None
    return np.asarray(sol.sol(checkpoint_times(T, K)))[:3].T.copy()


def path_error_km(ck_pos, ref_ck, mu=MU):
    """E3's statistic: the maximum position error over the checkpoints (inf if any checkpoint is missing, i.e. the run failed first)."""
    e = np.linalg.norm(np.asarray(ck_pos) - np.asarray(ref_ck), axis=1) * du_km_for_mu(mu)
    return float(np.inf) if not np.all(np.isfinite(e)) else float(e.max())


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
                   max_steps=4_000_000, seg=True, checkpoints=None):
    """param = fictitious step ds (rk4s, rk8s) or rtol (adaptive). Returns dict(ok, final (6), evals, reason, event, t_event, seg_hit);
    event 0 and t_event = T when the run reached the horizon. seg = True (the Part E default) adds the between-node collision test (item 7);
    seg = False is the endpoint-only check of the sealed kernels. checkpoints: physical times (item 3); the result then has "ck_pos" (K, 3),
    the Hermite-interpolated positions at those times (NaN for checkpoints after a failure)."""
    rc1, rc2 = collision_radii(mu)
    rej = EJECTION_RADIUS
    st = np.ascontiguousarray(state, dtype=np.float64)
    info = np.full(15, np.nan)
    ck_t = np.empty(0) if checkpoints is None else np.ascontiguousarray(checkpoints, dtype=np.float64)
    ck_out = np.full((len(ck_t), 3), np.nan)
    rk, d8 = _tab()
    if method == "rk4s":
        ok, X, _t, reason, n_s, n_f = rk4s_event(st, float(param), float(T), mu, int(max_steps), rc1, rc2, rej, float(tau_c), bool(seg), info, ck_t, ck_out)
        evals = 4 * (n_s + n_f)
    elif method == "rk8s":
        ok, X, _t, reason, n_s, n_f = rk8s_event(st, float(param), float(T), mu, int(max_steps), rc1, rc2, rej, float(tau_c), dr._A8, dr._B8, bool(seg), info, ck_t, ck_out)
        evals = dr.RK8_STAGES * (n_s + n_f)
    elif method in ("DP54PI", "DOP853", "RK45"):
        m, c = {"DP54PI": (0, 1), "DOP853": (1, 0), "RK45": (0, 0)}[method]
        rt = float(param)
        ok, X, evals, reason = rk_p_endpoint(st, float(T), rt, rt * atol_ratio, mu, m, c, int(max_nfev), rc1, rc2, rej,
                                             rk["A"], rk["B"], rk["E"], np.ascontiguousarray(d8["A"][:12, :12]), d8["B"], d8["E3"], d8["E5"], bool(seg), info, ck_t, ck_out)
    elif method in cs.SundmanAdaptive.SPECS:
        m, c = cs.SundmanAdaptive.SPECS[method]
        rt = float(param)
        Y0 = np.concatenate([st, [0.0]])
        ok, X, evals, _a, _r, _s, reason = cs.rk_s_endpoint(Y0, float(T), rt, rt * atol_ratio, mu, float(tau_c), m, c, int(max_nfev), True,
                                                             rc1, rc2, rej, rk["A"], rk["B"], rk["C"], rk["E"], rk["P"], d8["A"], d8["B"], d8["C"],
                                                             d8["E3"], d8["E5"], d8["D"], bool(seg), info, ck_t, ck_out)
    else:
        raise ValueError(method)
    ok = bool(ok)
    out = {"ok": ok, "final": np.asarray(X, dtype=float).copy(), "evals": float(evals), "reason": int(reason), "event": 0, "t_event": float(T),
           "seg_hit": False}
    if checkpoints is not None:
        out["ck_pos"] = ck_out
    if ok:
        return out
    if reason in (REASON_COLLISION, REASON_EJECTION) and np.isfinite(info[0]):
        X0, X1 = info[1:7].copy(), info[8:14].copy()
        if np.isfinite(info[14]):              # a chord crossing with both endpoints outside the body
            code = int(info[14])
            out["seg_hit"] = True
        else:
            code = int(bad_code(X1, mu, rc1, rc2, rej))
        te = float(hermite_event_time(code, info[0], X0, info[7], X1, mu, rc1, rc2, rej))
        if out["seg_hit"] and te == info[7]:   # the Hermite curve stays outside: the chord's closest approach
            c = np.array([-mu if code == 1 else 1.0 - mu, 0.0, 0.0])
            d = X1[:3] - X0[:3]
            dd = float(d @ d)
            th = 0.0 if dd == 0 else float(np.clip((c - X0[:3]) @ d / dd, 0.0, 1.0))
            te = float(info[0] + th * (info[7] - info[0]))
        out["event"] = code
        out["t_event"] = te
    else:
        out["event"] = 9
        out["t_event"] = float("nan")
    return out


# ---------------------------------------------------------------------------------------------------------------------
# many rows (item 3: the run_rows variant with the path error; used by the Part E runner and the path-certified calibration)
# ---------------------------------------------------------------------------------------------------------------------

def _rows_chunk(a):
    method, S, T, P, R, RC, seg, mu = a
    out = []
    for i in range(len(S)):
        ck = None if RC is None else checkpoint_times(T[i], RC.shape[1])
        r = run_with_event(method, S[i], T[i], P[i], mu=mu, seg=seg, checkpoints=ck)
        e = float(np.linalg.norm(r["final"][:3] - R[i]) * du_km_for_mu(mu)) if (r["ok"] and R is not None) else float("nan")
        pe = path_error_km(r["ck_pos"], RC[i], mu) if (RC is not None and r["ok"]) else float("nan")
        out.append((r["ok"], e, pe, r["evals"], r["reason"], r["event"], r["t_event"], r["seg_hit"]))
    return out


def rows_with_event(method, states, T, params, ref_pos=None, ref_ck=None, seg=True, tol_km=POSITION_TOL_KM, mu=MU, pool=None, n_chunks=64):
    """One run per row (params: per-row ds or rtol, or a scalar). Returns dict of arrays: ok, err (final error, km), path (max checkpoint error,
    km; NaN without ref_ck), evals, reason, event, t_event, seg_hit, met (final error <= tol), met_path (path error <= tol)."""
    S = np.ascontiguousarray(states, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    P = np.broadcast_to(np.asarray(params, dtype=np.float64), (len(S),)).copy()
    R = None if ref_pos is None else np.asarray(ref_pos, dtype=np.float64)
    RC = None if ref_ck is None else np.asarray(ref_ck, dtype=np.float64)
    idx = np.array_split(np.arange(len(S)), max(1, min(n_chunks, len(S))))
    jobs = [(method, S[j], T[j], P[j], None if R is None else R[j], None if RC is None else RC[j], bool(seg), float(mu)) for j in idx if len(j)]
    res = pool.map(_rows_chunk, jobs) if pool is not None else [_rows_chunk(j) for j in jobs]
    flat = [x for chunk in res for x in chunk]
    ok, err, pe, ev, rs, evn, te, sh = (np.array(c) for c in zip(*flat))
    ok = ok.astype(bool)
    return {"ok": ok, "err": err.astype(float), "path": pe.astype(float), "evals": ev.astype(float), "reason": rs.astype(int),
            "event": evn.astype(int), "t_event": te.astype(float), "seg_hit": sh.astype(bool),
            "met": ok & (err <= tol_km), "met_path": ok & (pe <= tol_km)}


def reference_checkpoints_rows(states, T, K=20, mu=MU, pool=None):
    """(n, K, 3) reference checkpoints for many usable rows (NaN rows where the reference does not reach T)."""
    jobs = [(np.asarray(states[i], dtype=float), float(T[i]), int(K), float(mu)) for i in range(len(states))]
    res = pool.map(_ref_ck_job, jobs, chunksize=max(1, len(jobs) // 256)) if pool is not None else [_ref_ck_job(j) for j in jobs]
    return np.stack([np.full((K, 3), np.nan) if r is None else r for r in res])


def _ref_ck_job(a):
    s, T, K, mu = a
    return reference_checkpoints(s, T, K, mu)
