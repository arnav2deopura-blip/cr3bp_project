"""
Study E, draft section 7 item 8: compiled adaptive solvers in Sundman time. Nothing here is used by a sealed kernel.

The seven-dimensional system Y = (X, t) in the fictitious time s, dY/ds = G(Y) = (g F(X), g) (sundman_3d._G, same g and tau_c = 2), is
integrated from s = 0 by an adaptive Runge-Kutta method until the event t(s) = T. Everything follows scipy.integrate.solve_ivp as used by the
SciPy version of P20 (defense_runs.run_sundman_adaptive): t_span (0, S_MAX), max_step = inf, the same initial step, error norm over all seven
components, step-size controller, and event location on the method's own dense output (RK45: the quartic interpolant of RK45.P; DOP853: the
seventh-order interpolant, which costs three extra right-hand-side evaluations on the last step, counted as SciPy counts them). The final state
is the dense output at the root of t(s) - T (bisection to machine precision; SciPy uses brentq with xtol = 4 eps).

  rk_s_endpoint(Y0, T, rtol, atol, mu, tau_c, method, ctrl, ...)   one orbit; method 0 = RK45 / DP5(4), 1 = DOP853
      ctrl = CTRL_SCIPY: SciPy's controller (the SciPy version's nfev is reproduced; tests/test_study_e.py)
      ctrl = CTRL_PI:    the Hairer-Wanner DOPRI5 PI controller of compiled_solvers_d (DP5(4)-PI), RK45 only
  rk_s_batch / rk_s_batch_par                                        many orbits at one rtol
  SundmanAdaptive                                                    a small wrapper with the interface of the Part D compiled solvers

check_bad = True applies the collision / ejection / non-finite check of the other compiled kernels after every accepted step (the comparators
of Part E); check_bad = False reproduces P20, which had no such check.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from .config import MU, EJECTION_RADIUS, ADAPTIVE_ATOL_RATIO, MAX_ADAPTIVE_NFEV, collision_radii, du_km_for_mu, POSITION_TOL_KM
from .physics_3d import REASON_OK, REASON_NONFINITE, REASON_STEP_CAP
from .sundman_3d import _G, _bad_state, SUNDMAN_TAU_C

__all__ = ["S_MAX", "CTRL_SCIPY", "CTRL_PI", "METHOD_RK45", "METHOD_DOP853", "REASON_NO_EVENT", "REASON_TOO_SMALL", "tableaux",
           "rk_s_endpoint", "rk_s_batch", "rk_s_batch_par", "SundmanAdaptive"]

S_MAX = 1e7                 # as defense_runs.S_MAX
CTRL_SCIPY, CTRL_PI = 0, 1
METHOD_RK45, METHOD_DOP853 = 0, 1
REASON_TOO_SMALL = 7        # as compiled_solvers_d
REASON_NO_EVENT = 8         # s reached S_MAX without t = T


def tableaux():
    """(A, B, C, E, P) of RK45 padded to the DOP853 shapes and (A, B, C, E3, E5, D) of DOP853, float64, from SciPy."""
    from scipy.integrate._ivp import dop853_coefficients as dc
    from scipy.integrate._ivp.rk import RK45
    rk = {"A": np.ascontiguousarray(RK45.A, dtype=np.float64), "B": np.ascontiguousarray(RK45.B, dtype=np.float64),
          "C": np.ascontiguousarray(RK45.C, dtype=np.float64), "E": np.ascontiguousarray(RK45.E, dtype=np.float64),
          "P": np.ascontiguousarray(RK45.P, dtype=np.float64)}
    d8 = {"A": np.ascontiguousarray(dc.A, dtype=np.float64), "B": np.ascontiguousarray(dc.B, dtype=np.float64),
          "C": np.ascontiguousarray(dc.C, dtype=np.float64), "E3": np.ascontiguousarray(dc.E3, dtype=np.float64),
          "E5": np.ascontiguousarray(dc.E5, dtype=np.float64), "D": np.ascontiguousarray(dc.D, dtype=np.float64)}
    return rk, d8


@njit(cache=True)
def _rms(v, n):
    a = 0.0
    for i in range(n):
        a += v[i] * v[i]
    return math.sqrt(a / n)


@njit(cache=True)
def _initial_step(y, f0, rtol, atol, mu, tau_c, order, tmp, f1, sc):
    """scipy.integrate._ivp.common.select_initial_step with t_bound - t0 = S_MAX and max_step = inf. One evaluation (into f1)."""
    n = 7
    for i in range(n):
        sc[i] = atol + abs(y[i]) * rtol
    d0 = 0.0
    d1 = 0.0
    for i in range(n):
        d0 += (y[i] / sc[i]) ** 2
        d1 += (f0[i] / sc[i]) ** 2
    d0 = math.sqrt(d0 / n)
    d1 = math.sqrt(d1 / n)
    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * d0 / d1
    h0 = min(h0, S_MAX)
    for i in range(n):
        tmp[i] = y[i] + h0 * f0[i]
    _G(tmp, mu, tau_c, 0.0, f1)
    d2 = 0.0
    for i in range(n):
        d2 += ((f1[i] - f0[i]) / sc[i]) ** 2
    d2 = math.sqrt(d2 / n) / h0
    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1.0 / (order + 1.0))
    return min(100.0 * h0, h1, S_MAX)


@njit(cache=True)
def _dense_rk45(y_old, h, Q, x, out):
    """RkDenseOutput: y_old + h * Q @ (x, x^2, x^3, x^4)."""
    for i in range(7):
        acc = 0.0
        p = x
        for j in range(Q.shape[1]):
            acc += Q[i, j] * p
            p *= x
        out[i] = y_old[i] + h * acc


@njit(cache=True)
def _dense_dop853(y_old, F, x, out):
    """Dop853DenseOutput: Horner-like evaluation over reversed(F), alternately multiplying by x and by 1 - x."""
    m = F.shape[0]
    for i in range(7):
        out[i] = 0.0
    for k in range(m):
        f = F[m - 1 - k]
        for i in range(7):
            out[i] += f[i]
            if k % 2 == 0:
                out[i] *= x
            else:
                out[i] *= 1.0 - x
    for i in range(7):
        out[i] += y_old[i]


@njit(cache=True)
def rk_s_endpoint(Y0, T, rtol, atol, mu, tau_c, method, ctrl, max_nfev, check_bad, rc1, rc2, rej, A, B, C, E, P, A8, B8, C8, E3, E5, D8, info):
    """Returns (ok, X_end (6), nfev, n_accepted, n_rejected, s_end, reason). When a collision / ejection / non-finite check fails, `info`
    (length 14) receives t_good, X_good (6), t_bad, X_bad (6): the last good state and the first bad one, in physical time (events_e.py)."""
    n = 7
    if method == METHOD_RK45:
        ns = B.shape[0]               # 6 stages + FSAL
        order = 4.0
        expo = -1.0 / 5.0
    else:
        ns = B8.shape[0]              # 12 stages + FSAL
        order = 7.0
        expo = -1.0 / 8.0
    K = np.empty((16, n))
    y = Y0.copy()
    ynew = np.empty(n)
    tmp = np.empty(n)
    f1 = np.empty(n)
    sc = np.empty(n)
    err = np.empty(n)
    err3 = np.empty(n)
    Xout = np.empty(6)
    _G(y, mu, tau_c, 0.0, K[0])
    nfev = 1
    h_abs = _initial_step(y, K[0], rtol, atol, mu, tau_c, order, tmp, f1, sc)
    nfev += 1
    s = 0.0
    n_acc = 0
    n_rej = 0
    facold = 1e-4
    while s < S_MAX:
        if nfev > max_nfev:
            return False, Xout, nfev, n_acc, n_rej, s, REASON_STEP_CAP
        min_step = 10.0 * abs(np.nextafter(s, np.inf) - s)
        if h_abs < min_step:
            h_abs = min_step
        rejected = False
        accepted = False
        s_new = s
        h = 0.0
        while not accepted:
            if h_abs < min_step:
                return False, Xout, nfev, n_acc, n_rej, s, REASON_TOO_SMALL
            s_new = s + h_abs
            if s_new > S_MAX:
                s_new = S_MAX
            h = s_new - s
            h_abs = abs(h)
            for st in range(1, ns):
                for i in range(n):
                    acc = 0.0
                    for j in range(st):
                        acc += K[j, i] * (A[st, j] if method == METHOD_RK45 else A8[st, j])
                    tmp[i] = y[i] + acc * h
                _G(tmp, mu, tau_c, 0.0, K[st])
            for i in range(n):
                acc = 0.0
                for j in range(ns):
                    acc += K[j, i] * (B[j] if method == METHOD_RK45 else B8[j])
                ynew[i] = y[i] + h * acc
            _G(ynew, mu, tau_c, 0.0, K[ns])
            nfev += ns
            for i in range(n):
                sc[i] = atol + max(abs(y[i]), abs(ynew[i])) * rtol
            if method == METHOD_RK45:
                for i in range(n):
                    acc = 0.0
                    for j in range(ns + 1):
                        acc += K[j, i] * E[j]
                    err[i] = acc * h / sc[i]
                en = _rms(err, n)
            else:
                e5 = 0.0
                e3 = 0.0
                for i in range(n):
                    a5 = 0.0
                    a3 = 0.0
                    for j in range(ns + 1):
                        a5 += K[j, i] * E5[j]
                        a3 += K[j, i] * E3[j]
                    err[i] = a5 / sc[i]
                    err3[i] = a3 / sc[i]
                    e5 += err[i] * err[i]
                    e3 += err3[i] * err3[i]
                if e5 == 0.0 and e3 == 0.0:
                    en = 0.0
                else:
                    en = abs(h) * e5 / math.sqrt((e5 + 0.01 * e3) * n)
            if en != en:
                return False, Xout, nfev, n_acc, n_rej, s, REASON_NONFINITE
            if ctrl == CTRL_SCIPY or method != METHOD_RK45:
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
            if not accepted:
                n_rej += 1
                if nfev > max_nfev:
                    return False, Xout, nfev, n_acc, n_rej, s, REASON_STEP_CAP
        n_acc += 1
        if ynew[6] - T >= 0.0:
            # the event t(s) = T lies in (s, s_new]: locate it on the dense output of this step
            dense = np.empty(n)
            if method == METHOD_RK45:
                Q = np.empty((n, P.shape[1]))
                for i in range(n):
                    for j in range(P.shape[1]):
                        acc = 0.0
                        for k in range(ns + 1):
                            acc += K[k, i] * P[k, j]
                        Q[i, j] = acc
            else:
                for st in range(ns + 1, 16):
                    for i in range(n):
                        acc = 0.0
                        for j in range(st):
                            acc += K[j, i] * A8[st, j]
                        tmp[i] = y[i] + acc * h
                    _G(tmp, mu, tau_c, 0.0, K[st])
                nfev += 16 - (ns + 1)
                m = D8.shape[0] + 3
                F = np.empty((m, n))
                for i in range(n):
                    dy = ynew[i] - y[i]
                    F[0, i] = dy
                    F[1, i] = h * K[0, i] - dy
                    F[2, i] = 2.0 * dy - h * (K[ns, i] + K[0, i])
                for r in range(D8.shape[0]):
                    for i in range(n):
                        acc = 0.0
                        for k in range(16):
                            acc += D8[r, k] * K[k, i]
                        F[3 + r, i] = h * acc
            lo = 0.0
            hi = 1.0
            for _it in range(200):
                mid = 0.5 * (lo + hi)
                if mid <= lo or mid >= hi:
                    break
                if method == METHOD_RK45:
                    _dense_rk45(y, h, Q, mid, dense)
                else:
                    _dense_dop853(y, F, mid, dense)
                if dense[6] - T >= 0.0:
                    hi = mid
                else:
                    lo = mid
            if method == METHOD_RK45:
                _dense_rk45(y, h, Q, hi, dense)
            else:
                _dense_dop853(y, F, hi, dense)
            for i in range(6):
                Xout[i] = dense[i]
            if check_bad:
                bad = _bad_state(Xout, mu, rc1, rc2, rej)
                if bad != 0:
                    info[0] = y[6]
                    info[7] = T
                    for i in range(6):
                        info[1 + i] = y[i]
                        info[8 + i] = Xout[i]
                    return False, Xout, nfev, n_acc, n_rej, s + hi * h, bad
            for i in range(6):
                if not math.isfinite(Xout[i]):
                    return False, Xout, nfev, n_acc, n_rej, s + hi * h, REASON_NONFINITE
            return True, Xout, nfev, n_acc, n_rej, s + hi * h, REASON_OK
        if check_bad:
            bad = _bad_state(ynew, mu, rc1, rc2, rej)
            if bad != 0:
                info[0] = y[6]
                info[7] = ynew[6]
                for i in range(6):
                    Xout[i] = ynew[i]
                    info[1 + i] = y[i]
                    info[8 + i] = ynew[i]
                return False, Xout, nfev, n_acc, n_rej, s_new, bad
        for i in range(n):
            y[i] = ynew[i]
            K[0, i] = K[ns, i]
        s = s_new
    return False, Xout, nfev, n_acc, n_rej, s, REASON_NO_EVENT


@njit(cache=True)
def rk_s_batch(states, T, rtol, atol_ratio, mu, tau_c, method, ctrl, max_nfev, check_bad, rc1, rc2, rej, A, B, C, E, P, A8, B8, C8, E3, E5, D8):
    n = states.shape[0]
    ok = np.zeros(n, dtype=np.bool_)
    y = np.empty((n, 6))
    nf = np.zeros(n)
    rs = np.zeros(n, dtype=np.int64)
    Y0 = np.empty(7)
    info = np.empty(14)
    for i in range(n):
        for j in range(6):
            Y0[j] = states[i, j]
        Y0[6] = 0.0
        o, yy, e, _a, _r, _s, r = rk_s_endpoint(Y0, T[i], rtol, rtol * atol_ratio, mu, tau_c, method, ctrl, max_nfev, check_bad, rc1, rc2, rej,
                                                A, B, C, E, P, A8, B8, C8, E3, E5, D8, info)
        ok[i] = o
        for j in range(6):
            y[i, j] = yy[j]
        nf[i] = e
        rs[i] = r
    return ok, y, nf, rs


@njit(parallel=True, cache=True)
def rk_s_batch_par(states, T, rtol, atol_ratio, mu, tau_c, method, ctrl, max_nfev, check_bad, rc1, rc2, rej, A, B, C, E, P, A8, B8, C8, E3, E5, D8):
    n = states.shape[0]
    ok = np.zeros(n, dtype=np.bool_)
    y = np.empty((n, 6))
    nf = np.zeros(n)
    rs = np.zeros(n, dtype=np.int64)
    for i in prange(n):
        Y0 = np.empty(7)
        info = np.empty(14)
        for j in range(6):
            Y0[j] = states[i, j]
        Y0[6] = 0.0
        o, yy, e, _a, _r, _s, r = rk_s_endpoint(Y0, T[i], rtol, rtol * atol_ratio, mu, tau_c, method, ctrl, max_nfev, check_bad, rc1, rc2, rej,
                                                A, B, C, E, P, A8, B8, C8, E3, E5, D8, info)
        ok[i] = o
        for j in range(6):
            y[i, j] = yy[j]
        nf[i] = e
        rs[i] = r
    return ok, y, nf, rs


class SundmanAdaptive:
    """name in {"RK45s" (SciPy controller), "DP54PIs" (PI controller), "DOP853s"}. .rows(states, T, ref_pos, rtol) -> dict(ok, err, nfev,
    capped, met) like symmetric_cert.scipy_rows; .one(state, T, rtol) for a single orbit."""

    SPECS = {"RK45s": (METHOD_RK45, CTRL_SCIPY), "DP54PIs": (METHOD_RK45, CTRL_PI), "DOP853s": (METHOD_DOP853, CTRL_SCIPY)}

    def __init__(self, name, mu=MU, tau_c=SUNDMAN_TAU_C, atol_ratio=ADAPTIVE_ATOL_RATIO, max_nfev=MAX_ADAPTIVE_NFEV, check_bad=True):
        self.name = name
        self.method, self.ctrl = self.SPECS[name]
        self.mu, self.tau_c, self.atol_ratio, self.max_nfev, self.check_bad = float(mu), float(tau_c), float(atol_ratio), int(max_nfev), bool(check_bad)
        self.rc1, self.rc2 = collision_radii(mu)
        self.rej = EJECTION_RADIUS
        self.du_km = du_km_for_mu(mu)
        rk, d8 = tableaux()
        self.tab = (rk["A"], rk["B"], rk["C"], rk["E"], rk["P"], d8["A"], d8["B"], d8["C"], d8["E3"], d8["E5"], d8["D"])

    def run(self, states, T, rtol, parallel=True):
        f = rk_s_batch_par if parallel else rk_s_batch
        return f(np.ascontiguousarray(states, dtype=np.float64), np.ascontiguousarray(T, dtype=np.float64), float(rtol), self.atol_ratio, self.mu,
                 self.tau_c, self.method, self.ctrl, self.max_nfev, self.check_bad, self.rc1, self.rc2, self.rej, *self.tab)

    def rows(self, states, T, ref_pos, rtol, tol_km=POSITION_TOL_KM, parallel=True):
        ok, y, nf, rs = self.run(states, T, rtol, parallel)
        err = np.where(ok, np.linalg.norm(y[:, :3] - ref_pos, axis=1) * self.du_km, np.nan)
        capped = rs == REASON_STEP_CAP
        nfev = np.where(ok, nf, np.nan)
        return {"ok": ok, "err": err, "nfev": nfev, "capped": capped, "met": ok & (err <= tol_km), "reason": rs}
