"""
Compiled solvers for the wall-clock and batch benchmark of Part D (D2). The Part A-C comparisons count right-hand-side evaluations and the
adaptive baselines were SciPy (a Python call per step); this module puts the policy and the adaptive solvers on the same footing: all of them
are Numba kernels over the same right-hand side with the same collision / ejection / non-finite checks.

  dp54_endpoint        Dormand-Prince 5(4), unrolled, FSAL. ctrl = CTRL_SCIPY reproduces scipy.integrate.RK45 step for step (same initial
                       step, same I-controller, same error norm; the compiled nfev equals SciPy's); ctrl = CTRL_PI is the Hairer-Wanner DOPRI5
                       controller (PI, beta = 0.04) with the same error norm and initial step.
  dop853_endpoint      DOP853 from SciPy's own tableau (A, B, E3, E5), same controller as scipy.integrate.DOP853.
  sundman_policy_one   one orbit end to end: state -> 25 periapsis / state features -> 25-tree forest -> certified scale -> Sundman-time RK4.
  *_batch              the same kernels over many orbits, serial or parallel (prange).
  certify_rtol_compiled   the project's rtol certification (Bonferroni over the grid, equal-risk refinement) for a compiled solver.

Costs are still reported in right-hand-side evaluations (nfev) next to the wall times, so the two conclusions can be compared orbit by orbit.
The tableau coefficients and the controller constants come from SciPy (scipy.integrate._ivp, pinned: 1.17.1); tests/test_part_d.py checks the compiled
kernels against solve_ivp and fails loudly if a future SciPy changes them.
"""
from __future__ import annotations

import math
import time

import numpy as np
from numba import njit, prange
from scipy.stats import binom

from .config import MU, EJECTION_RADIUS, collision_radii, du_km_for_mu
from .physics_3d import REASON_OK, REASON_NONFINITE, REASON_STEP_CAP
from .sundman_3d import _integrate_endpoint_rk4s_numba, _bad_state, SUNDMAN_TAU_C
from . import compiled_model_3d as cm

CTRL_SCIPY, CTRL_PI = 0, 1
MAX_NFEV = 2_000_000          # fair_baseline_3d.MAX_ADAPTIVE_NFEV: the same abandonment cap as the SciPy comparators
ATOL_RATIO = 1e-3             # atol = 1e-3 * rtol, as in fair_baseline_3d._run_adaptive_ex
RTOL_GRID_B = tuple(10.0 ** e for e in np.linspace(-12.0, -3.0, 18))      # fair_baseline_3d.RTOL_GRID_B
REASON_TOO_SMALL = 7          # step below 10 ulp(t): SciPy's TOO_SMALL_STEP failure


def scipy_tableaux():
    """(A, B, E) of RK45 and (A, B, E3, E5) of DOP853 as float64 arrays, straight from SciPy."""
    from scipy.integrate._ivp import dop853_coefficients as dc
    from scipy.integrate._ivp.rk import RK45
    n = dc.N_STAGES
    return {"rk45": (np.asarray(RK45.A), np.asarray(RK45.B), np.asarray(RK45.E)),
            "dop853": (np.ascontiguousarray(dc.A[:n, :n]), np.asarray(dc.B), np.asarray(dc.E3), np.asarray(dc.E5))}


# ---------------------------------------------------------------------------------------------------------------------
# right-hand side (same arithmetic as physics_3d._cr3bp_derivatives_3d_numba, written into a buffer)
# ---------------------------------------------------------------------------------------------------------------------

@njit(cache=True)
def _rhs(y, mu, out):
    x, yy, z, vx, vy, vz = y[0], y[1], y[2], y[3], y[4], y[5]
    r1 = max(math.sqrt((x + mu) ** 2 + yy ** 2 + z ** 2), 1e-9)
    r2 = max(math.sqrt((x + mu - 1.0) ** 2 + yy ** 2 + z ** 2), 1e-9)
    r13 = r1 ** 3
    r23 = r2 ** 3
    out[0] = vx
    out[1] = vy
    out[2] = vz
    out[3] = 2.0 * vy + x - (1.0 - mu) * (x + mu) / r13 - mu * (x + mu - 1.0) / r23
    out[4] = -2.0 * vx + yy - (1.0 - mu) * yy / r13 - mu * yy / r23
    out[5] = -(1.0 - mu) * z / r13 - mu * z / r23


@njit(cache=True)
def _rhs_fast(y, mu, out):
    """The same right-hand side with one reciprocal per body instead of six divisions (rounding differs from _rhs in the last digit)."""
    x, yy, z, vx, vy, vz = y[0], y[1], y[2], y[3], y[4], y[5]
    dx1 = x + mu
    dx2 = x + mu - 1.0
    yz2 = yy * yy + z * z
    r1 = max(math.sqrt(dx1 * dx1 + yz2), 1e-9)
    r2 = max(math.sqrt(dx2 * dx2 + yz2), 1e-9)
    a1 = (1.0 - mu) / (r1 * r1 * r1)
    a2 = mu / (r2 * r2 * r2)
    out[0] = vx
    out[1] = vy
    out[2] = vz
    out[3] = 2.0 * vy + x - a1 * dx1 - a2 * dx2
    out[4] = -2.0 * vx + yy - (a1 + a2) * yy
    out[5] = -(a1 + a2) * z


class _Kernels:
    """Namespace of the kernels compiled for one right-hand side."""


def make_adaptive_kernels(rhs):
    """Compile the adaptive-solver kernels with `rhs` (a jitted (y, mu, out) function) as a compile-time constant, so neither variant pays a dispatch cost: SciPy-faithful
    arithmetic (_rhs) or the micro-optimized one (_rhs_fast). Not cached on disk (closures), compiled lazily at the first call."""
    @njit
    def initial_step(y, f0, T, rtol, atol, mu, order, tmp, f1):
        """scipy.integrate._ivp.common.select_initial_step for a forward integration of length T with max_step = inf. One evaluation (into f1)."""
        d0 = 0.0
        d1 = 0.0
        for i in range(6):
            sc = atol + abs(y[i]) * rtol
            d0 += (y[i] / sc) ** 2
            d1 += (f0[i] / sc) ** 2
        d0 = math.sqrt(d0 / 6.0)
        d1 = math.sqrt(d1 / 6.0)
        if d0 < 1e-5 or d1 < 1e-5:
            h0 = 1e-6
        else:
            h0 = 0.01 * d0 / d1
        h0 = min(h0, T)
        for i in range(6):
            tmp[i] = y[i] + h0 * f0[i]
        rhs(tmp, mu, f1)
        d2 = 0.0
        for i in range(6):
            sc = atol + abs(y[i]) * rtol
            d2 += ((f1[i] - f0[i]) / sc) ** 2
        d2 = math.sqrt(d2 / 6.0) / h0
        if d1 <= 1e-15 and d2 <= 1e-15:
            h1 = max(1e-6, h0 * 1e-3)
        else:
            h1 = (0.01 / max(d1, d2)) ** (1.0 / (order + 1.0))
        return min(100.0 * h0, h1, T)

    @njit
    def dp54_endpoint(y0, T, rtol, atol, mu, ctrl, max_nfev, rc1, rc2, rej):
        """Returns (ok, y_end, nfev, n_accepted, n_rejected, reason). nfev counts right-hand-side evaluations like solve_ivp's nfev
        (2 for the initial step, 6 per attempted step)."""
        y = y0.copy()
        ynew = np.empty(6)
        tmp = np.empty(6)
        k1 = np.empty(6)
        k2 = np.empty(6)
        k3 = np.empty(6)
        k4 = np.empty(6)
        k5 = np.empty(6)
        k6 = np.empty(6)
        k7 = np.empty(6)
        rhs(y, mu, k1)
        nfev = 1
        h_abs = initial_step(y, k1, T, rtol, atol, mu, 4.0, tmp, k2)
        nfev += 1
        t = 0.0
        n_acc = 0
        n_rej = 0
        facold = 1e-4
        while t < T:
            if nfev > max_nfev:
                return False, y, nfev, n_acc, n_rej, REASON_STEP_CAP,
            min_step = 10.0 * abs(np.nextafter(t, np.inf) - t)
            if h_abs < min_step:
                h_abs = min_step
            rejected = False
            accepted = False
            t_new = t
            while not accepted:
                if h_abs < min_step:
                    return False, y, nfev, n_acc, n_rej, REASON_TOO_SMALL
                t_new = t + h_abs
                if t_new > T:
                    t_new = T
                h = t_new - t
                h_abs = abs(h)
                for i in range(6):
                    tmp[i] = y[i] + (0.2 * k1[i]) * h
                rhs(tmp, mu, k2)
                for i in range(6):
                    tmp[i] = y[i] + (3.0 / 40.0 * k1[i] + 9.0 / 40.0 * k2[i]) * h
                rhs(tmp, mu, k3)
                for i in range(6):
                    tmp[i] = y[i] + (44.0 / 45.0 * k1[i] - 56.0 / 15.0 * k2[i] + 32.0 / 9.0 * k3[i]) * h
                rhs(tmp, mu, k4)
                for i in range(6):
                    tmp[i] = y[i] + (19372.0 / 6561.0 * k1[i] - 25360.0 / 2187.0 * k2[i] + 64448.0 / 6561.0 * k3[i] - 212.0 / 729.0 * k4[i]) * h
                rhs(tmp, mu, k5)
                for i in range(6):
                    tmp[i] = y[i] + (9017.0 / 3168.0 * k1[i] - 355.0 / 33.0 * k2[i] + 46732.0 / 5247.0 * k3[i] + 49.0 / 176.0 * k4[i]
                                     - 5103.0 / 18656.0 * k5[i]) * h
                rhs(tmp, mu, k6)
                for i in range(6):
                    ynew[i] = y[i] + h * (35.0 / 384.0 * k1[i] + 500.0 / 1113.0 * k3[i] + 125.0 / 192.0 * k4[i] - 2187.0 / 6784.0 * k5[i]
                                          + 11.0 / 84.0 * k6[i])
                rhs(ynew, mu, k7)
                nfev += 6
                en = 0.0
                for i in range(6):
                    e = (-71.0 / 57600.0 * k1[i] + 71.0 / 16695.0 * k3[i] - 71.0 / 1920.0 * k4[i] + 17253.0 / 339200.0 * k5[i]
                         - 22.0 / 525.0 * k6[i] + 1.0 / 40.0 * k7[i]) * h
                    sc = atol + max(abs(y[i]), abs(ynew[i])) * rtol
                    en += (e / sc) ** 2
                en = math.sqrt(en / 6.0)
                if en != en:
                    return False, y, nfev, n_acc, n_rej, REASON_NONFINITE
                if ctrl == CTRL_SCIPY:
                    if en < 1.0:
                        factor = 10.0 if en == 0.0 else min(10.0, 0.9 * en ** -0.2)
                        if rejected:
                            factor = min(1.0, factor)
                        h_abs *= factor
                        accepted = True
                    else:
                        h_abs *= max(0.2, 0.9 * en ** -0.2)
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
                        return False, y, nfev, n_acc, n_rej, REASON_STEP_CAP
            bad = _bad_state(ynew, mu, rc1, rc2, rej)
            if bad != 0:
                return False, ynew.copy(), nfev, n_acc, n_rej, bad
            for i in range(6):
                y[i] = ynew[i]
                k1[i] = k7[i]
            t = t_new
            n_acc += 1
        return True, y, nfev, n_acc, n_rej, REASON_OK

    @njit
    def dop853_endpoint(y0, T, rtol, atol, mu, A, B, E3, E5, max_nfev, rc1, rc2, rej):
        """Same return as dp54_endpoint. SciPy's DOP853: 12 stages, FSAL, error norm from E5 / E3, exponent -1/8, initial step order 7."""
        ns = B.shape[0]
        K = np.empty((ns + 1, 6))
        y = y0.copy()
        ynew = np.empty(6)
        tmp = np.empty(6)
        f1 = np.empty(6)
        rhs(y, mu, K[0])
        nfev = 1
        h_abs = initial_step(y, K[0], T, rtol, atol, mu, 7.0, tmp, f1)
        nfev += 1
        t = 0.0
        n_acc = 0
        n_rej = 0
        while t < T:
            if nfev > max_nfev:
                return False, y, nfev, n_acc, n_rej, REASON_STEP_CAP
            min_step = 10.0 * abs(np.nextafter(t, np.inf) - t)
            if h_abs < min_step:
                h_abs = min_step
            rejected = False
            accepted = False
            t_new = t
            while not accepted:
                if h_abs < min_step:
                    return False, y, nfev, n_acc, n_rej, REASON_TOO_SMALL
                t_new = t + h_abs
                if t_new > T:
                    t_new = T
                h = t_new - t
                h_abs = abs(h)
                for s in range(1, ns):
                    for i in range(6):
                        acc = 0.0
                        for j in range(s):
                            acc += K[j, i] * A[s, j]
                        tmp[i] = y[i] + acc * h
                    rhs(tmp, mu, K[s])
                for i in range(6):
                    acc = 0.0
                    for j in range(ns):
                        acc += K[j, i] * B[j]
                    ynew[i] = y[i] + h * acc
                rhs(ynew, mu, K[ns])
                nfev += ns
                e5 = 0.0
                e3 = 0.0
                for i in range(6):
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
                    en = abs(h) * e5 / math.sqrt((e5 + 0.01 * e3) * 6.0)
                if en != en:
                    return False, y, nfev, n_acc, n_rej, REASON_NONFINITE
                if en < 1.0:
                    factor = 10.0 if en == 0.0 else min(10.0, 0.9 * en ** (-1.0 / 8.0))
                    if rejected:
                        factor = min(1.0, factor)
                    h_abs *= factor
                    accepted = True
                else:
                    h_abs *= max(0.2, 0.9 * en ** (-1.0 / 8.0))
                    rejected = True
                    n_rej += 1
                    if nfev > max_nfev:
                        return False, y, nfev, n_acc, n_rej, REASON_STEP_CAP
            bad = _bad_state(ynew, mu, rc1, rc2, rej)
            if bad != 0:
                return False, ynew.copy(), nfev, n_acc, n_rej, bad
            for i in range(6):
                y[i] = ynew[i]
                K[0, i] = K[ns, i]
            t = t_new
            n_acc += 1
        return True, y, nfev, n_acc, n_rej, REASON_OK

    @njit
    def dp54_batch(states, T, rtol, atol_ratio, mu, ctrl, max_nfev, rc1, rc2, rej):
        n = states.shape[0]
        ok = np.zeros(n, dtype=np.bool_)
        y = np.empty((n, 6))
        nf = np.zeros(n)
        rs = np.zeros(n, dtype=np.int64)
        for i in range(n):
            o, yy, e, _a, _r, r = dp54_endpoint(states[i], T[i], rtol, rtol * atol_ratio, mu, ctrl, max_nfev, rc1, rc2, rej)
            ok[i] = o
            for j in range(6):
                y[i, j] = yy[j]
            nf[i] = e
            rs[i] = r
        return ok, y, nf, rs

    @njit(parallel=True)
    def dp54_batch_par(states, T, rtol, atol_ratio, mu, ctrl, max_nfev, rc1, rc2, rej):
        n = states.shape[0]
        ok = np.zeros(n, dtype=np.bool_)
        y = np.empty((n, 6))
        nf = np.zeros(n)
        rs = np.zeros(n, dtype=np.int64)
        for i in prange(n):
            o, yy, e, _a, _r, r = dp54_endpoint(states[i], T[i], rtol, rtol * atol_ratio, mu, ctrl, max_nfev, rc1, rc2, rej)
            ok[i] = o
            for j in range(6):
                y[i, j] = yy[j]
            nf[i] = e
            rs[i] = r
        return ok, y, nf, rs

    @njit
    def dop853_batch(states, T, rtol, atol_ratio, mu, A, B, E3, E5, max_nfev, rc1, rc2, rej):
        n = states.shape[0]
        ok = np.zeros(n, dtype=np.bool_)
        y = np.empty((n, 6))
        nf = np.zeros(n)
        rs = np.zeros(n, dtype=np.int64)
        for i in range(n):
            o, yy, e, _a, _r, r = dop853_endpoint(states[i], T[i], rtol, rtol * atol_ratio, mu, A, B, E3, E5, max_nfev, rc1, rc2, rej)
            ok[i] = o
            for j in range(6):
                y[i, j] = yy[j]
            nf[i] = e
            rs[i] = r
        return ok, y, nf, rs

    @njit(parallel=True)
    def dop853_batch_par(states, T, rtol, atol_ratio, mu, A, B, E3, E5, max_nfev, rc1, rc2, rej):
        n = states.shape[0]
        ok = np.zeros(n, dtype=np.bool_)
        y = np.empty((n, 6))
        nf = np.zeros(n)
        rs = np.zeros(n, dtype=np.int64)
        for i in prange(n):
            o, yy, e, _a, _r, r = dop853_endpoint(states[i], T[i], rtol, rtol * atol_ratio, mu, A, B, E3, E5, max_nfev, rc1, rc2, rej)
            ok[i] = o
            for j in range(6):
                y[i, j] = yy[j]
            nf[i] = e
            rs[i] = r
        return ok, y, nf, rs

    @njit
    def loop_dp54(st, T, reps, rtol, atol_ratio, mu, ctrl, max_nfev, rc1, rc2, rej):
        s = st.copy()
        acc = 0.0
        for r in range(reps):
            s[0] = st[0] + 1e-15 * (r % 7)
            _o, y, nf, _a, _r, _re = dp54_endpoint(s, T, rtol, rtol * atol_ratio, mu, ctrl, max_nfev, rc1, rc2, rej)
            acc += y[0] + nf
        return acc

    @njit
    def loop_dop853(st, T, reps, rtol, atol_ratio, mu, A, B, E3, E5, max_nfev, rc1, rc2, rej):
        s = st.copy()
        acc = 0.0
        for r in range(reps):
            s[0] = st[0] + 1e-15 * (r % 7)
            _o, y, nf, _a, _r, _re = dop853_endpoint(s, T, rtol, rtol * atol_ratio, mu, A, B, E3, E5, max_nfev, rc1, rc2, rej)
            acc += y[0] + nf
        return acc

    k = _Kernels()
    for name in ("initial_step", "dp54_endpoint", "dop853_endpoint", "dp54_batch", "dp54_batch_par", "dop853_batch", "dop853_batch_par", "loop_dp54", "loop_dop853"):
        setattr(k, name, locals()[name])
    return k


KR = make_adaptive_kernels(_rhs)           # R: SciPy-faithful arithmetic (the compiled nfev equals solve_ivp's)
KF = make_adaptive_kernels(_rhs_fast)      # F: micro-optimized arithmetic, used by the wall-clock arm
dp54_endpoint, dop853_endpoint = KR.dp54_endpoint, KR.dop853_endpoint



# ---------------------------------------------------------------------------------------------------------------------
# Dormand-Prince 5(4), unrolled
# ---------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------
# DOP853 from SciPy's tableau
# ---------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------
# Sundman-time RK4 policy, one orbit end to end
# ---------------------------------------------------------------------------------------------------------------------

@njit(cache=True)
def sundman_run(st, ds, T, mu, tau_c, max_steps, rc1, rc2, rej):
    """(ok, y_end, evals, reason) of the Sundman-time RK4 at fictitious step ds."""
    ok, _prev, curr, _tp, _t, reason, ns, nf = _integrate_endpoint_rk4s_numba(st, ds, T, mu, max_steps, rc1, rc2, rej, tau_c, 0.0)
    return ok, curr, 4 * (ns + nf), reason


@njit(cache=True)
def sundman_policy_one(st, T, scale, mu, c_l1, fr, tau_c, max_steps, rc1, rc2, rej, xbuf, flags):
    """The deployed `sund` policy for one orbit: features and forest (compiled_model_3d.policy_hgb_one with the single Sundman-step regressor),
    ds = scale * raw, then the integration. Returns (ok, y_end, evals, reason, ds)."""
    _c, raw = cm.policy_hgb_one(st, T, mu, c_l1, True, fr, fr, fr, 3, 0.0, False, xbuf, flags)
    ds = raw * scale
    ok, y, ev, reason = sundman_run(st, ds, T, mu, tau_c, max_steps, rc1, rc2, rej)
    return ok, y, ev, reason, ds


@njit(cache=True)
def rk4s_fixed_batch(states, T, ds, mu, tau_c, max_steps, rc1, rc2, rej):
    """Serial batch of Sundman runs at given per-orbit ds (used by the certification loops): (ok, y_end (n, 6), evals, reason)."""
    n = states.shape[0]
    ok = np.zeros(n, dtype=np.bool_)
    y = np.empty((n, 6))
    ev = np.zeros(n)
    rs = np.zeros(n, dtype=np.int64)
    for i in range(n):
        o, yy, e, r = sundman_run(states[i], ds[i], T[i], mu, tau_c, max_steps, rc1, rc2, rej)
        ok[i] = o
        for j in range(6):
            y[i, j] = yy[j]
        ev[i] = e
        rs[i] = r
    return ok, y, ev, rs


@njit(parallel=True, cache=True)
def rk4s_fixed_batch_par(states, T, ds, mu, tau_c, max_steps, rc1, rc2, rej):
    n = states.shape[0]
    ok = np.zeros(n, dtype=np.bool_)
    y = np.empty((n, 6))
    ev = np.zeros(n)
    rs = np.zeros(n, dtype=np.int64)
    for i in prange(n):
        o, yy, e, r = sundman_run(states[i], ds[i], T[i], mu, tau_c, max_steps, rc1, rc2, rej)
        ok[i] = o
        for j in range(6):
            y[i, j] = yy[j]
        ev[i] = e
        rs[i] = r
    return ok, y, ev, rs


@njit(cache=True)
def policy_batch(states, T, scale, mu, c_l1, fr, tau_c, max_steps, rc1, rc2, rej):
    n = states.shape[0]
    ok = np.zeros(n, dtype=np.bool_)
    y = np.empty((n, 6))
    ev = np.zeros(n)
    rs = np.zeros(n, dtype=np.int64)
    xbuf = np.zeros(25)
    flags = np.zeros(2)
    for i in range(n):
        o, yy, e, r, _ds = sundman_policy_one(states[i], T[i], scale, mu, c_l1, fr, tau_c, max_steps, rc1, rc2, rej, xbuf, flags)
        ok[i] = o
        for j in range(6):
            y[i, j] = yy[j]
        ev[i] = e
        rs[i] = r
    return ok, y, ev, rs


@njit(parallel=True, cache=True)
def policy_batch_par(states, T, scale, mu, c_l1, fr, tau_c, max_steps, rc1, rc2, rej):
    n = states.shape[0]
    ok = np.zeros(n, dtype=np.bool_)
    y = np.empty((n, 6))
    ev = np.zeros(n)
    rs = np.zeros(n, dtype=np.int64)
    for i in prange(n):
        xbuf = np.zeros(25)
        flags = np.zeros(2)
        o, yy, e, r, _ds = sundman_policy_one(states[i], T[i], scale, mu, c_l1, fr, tau_c, max_steps, rc1, rc2, rej, xbuf, flags)
        ok[i] = o
        for j in range(6):
            y[i, j] = yy[j]
        ev[i] = e
        rs[i] = r
    return ok, y, ev, rs


# ---------------------------------------------------------------------------------------------------------------------
# micro-optimized Sundman-time RK4 (same algorithm and control flow as sundman_3d._integrate_endpoint_rk4s_numba; the wall-clock arm of D2)
# ---------------------------------------------------------------------------------------------------------------------

@njit(cache=True)
def _G_fast(Y, mu, itau_c, out):
    """out = (g F(X), g): one reciprocal per body shared by the acceleration and the time scaling, no allocation."""
    x, yy, z, vx, vy, vz = Y[0], Y[1], Y[2], Y[3], Y[4], Y[5]
    dx1 = x + mu
    dx2 = x + mu - 1.0
    yz2 = yy * yy + z * z
    r1 = max(math.sqrt(dx1 * dx1 + yz2), 1e-9)
    r2 = max(math.sqrt(dx2 * dx2 + yz2), 1e-9)
    a1 = (1.0 - mu) / (r1 * r1 * r1)
    a2 = mu / (r2 * r2 * r2)
    g = 1.0 / (math.sqrt(a1) + math.sqrt(a2) + itau_c)
    out[0] = g * vx
    out[1] = g * vy
    out[2] = g * vz
    out[3] = g * (2.0 * vy + x - a1 * dx1 - a2 * dx2)
    out[4] = g * (-2.0 * vx + yy - (a1 + a2) * yy)
    out[5] = g * (-(a1 + a2) * z)
    out[6] = g


@njit(cache=True)
def sundman_run_fast(st, ds, T, mu, tau_c, max_steps, rc1, rc2, rej):
    """(ok, y_end, evals, reason): the Sundman-time RK4 at fictitious step ds with the optimized kernels (landing steps in physical time included)."""
    X = st.copy()
    t = 0.0
    n_s = 0
    n_f = 0
    itau = 1.0 / tau_c
    Y = np.empty(7)
    tmp = np.empty(7)
    k1 = np.empty(7)
    k2 = np.empty(7)
    k3 = np.empty(7)
    k4 = np.empty(7)
    l1 = np.empty(6)
    l2 = np.empty(6)
    l3 = np.empty(6)
    l4 = np.empty(6)
    lt = np.empty(6)
    sn = np.empty(6)
    while t < T:
        if n_s + n_f >= max_steps:
            return False, X, 4 * (n_s + n_f), REASON_STEP_CAP
        for i in range(6):
            Y[i] = X[i]
        Y[6] = t
        _G_fast(Y, mu, itau, k1)
        dt_pred = k1[6] * ds
        land = (t + 1.5 * dt_pred >= T - 1e-12)
        if not land:
            for i in range(7):
                tmp[i] = Y[i] + 0.5 * ds * k1[i]
            _G_fast(tmp, mu, itau, k2)
            for i in range(7):
                tmp[i] = Y[i] + 0.5 * ds * k2[i]
            _G_fast(tmp, mu, itau, k3)
            for i in range(7):
                tmp[i] = Y[i] + ds * k3[i]
            _G_fast(tmp, mu, itau, k4)
            for i in range(7):
                tmp[i] = Y[i] + ds / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
            n_s += 1
            bad = _bad_state(tmp, mu, rc1, rc2, rej)
            if bad != 0:
                return False, X, 4 * (n_s + n_f), bad
            if tmp[6] >= T - 1e-12:
                land = True
            else:
                for i in range(6):
                    X[i] = tmp[i]
                t = tmp[6]
        if land:
            remaining = T - t
            nf = max(1, int(math.ceil(remaining / dt_pred - 1e-9)))
            h = remaining / nf
            for j in range(nf):
                if n_s + n_f >= max_steps:
                    return False, X, 4 * (n_s + n_f), REASON_STEP_CAP
                _rhs_fast(X, mu, l1)
                for i in range(6):
                    lt[i] = X[i] + 0.5 * h * l1[i]
                _rhs_fast(lt, mu, l2)
                for i in range(6):
                    lt[i] = X[i] + 0.5 * h * l2[i]
                _rhs_fast(lt, mu, l3)
                for i in range(6):
                    lt[i] = X[i] + h * l3[i]
                _rhs_fast(lt, mu, l4)
                for i in range(6):
                    sn[i] = X[i] + h / 6.0 * (l1[i] + 2.0 * l2[i] + 2.0 * l3[i] + l4[i])
                n_f += 1
                bad = _bad_state(sn, mu, rc1, rc2, rej)
                if bad != 0:
                    return False, X, 4 * (n_s + n_f), bad
                for i in range(6):
                    X[i] = sn[i]
                t = t + h
            t = T
            break
    return True, X, 4 * (n_s + n_f), REASON_OK


@njit(cache=True)
def sundman_policy_one_fast(st, T, scale, mu, c_l1, fr, tau_c, max_steps, rc1, rc2, rej, xbuf, flags):
    _c, raw = cm.policy_hgb_one(st, T, mu, c_l1, True, fr, fr, fr, 3, 0.0, False, xbuf, flags)
    ds = raw * scale
    ok, y, ev, reason = sundman_run_fast(st, ds, T, mu, tau_c, max_steps, rc1, rc2, rej)
    return ok, y, ev, reason, ds


@njit(cache=True)
def rk4s_fixed_batch_fast(states, T, ds, mu, tau_c, max_steps, rc1, rc2, rej):
    n = states.shape[0]
    ok = np.zeros(n, dtype=np.bool_)
    y = np.empty((n, 6))
    ev = np.zeros(n)
    rs = np.zeros(n, dtype=np.int64)
    for i in range(n):
        o, yy, e, r = sundman_run_fast(states[i], ds[i], T[i], mu, tau_c, max_steps, rc1, rc2, rej)
        ok[i] = o
        for j in range(6):
            y[i, j] = yy[j]
        ev[i] = e
        rs[i] = r
    return ok, y, ev, rs


@njit(parallel=True, cache=True)
def rk4s_fixed_batch_fast_par(states, T, ds, mu, tau_c, max_steps, rc1, rc2, rej):
    n = states.shape[0]
    ok = np.zeros(n, dtype=np.bool_)
    y = np.empty((n, 6))
    ev = np.zeros(n)
    rs = np.zeros(n, dtype=np.int64)
    for i in prange(n):
        o, yy, e, r = sundman_run_fast(states[i], ds[i], T[i], mu, tau_c, max_steps, rc1, rc2, rej)
        ok[i] = o
        for j in range(6):
            y[i, j] = yy[j]
        ev[i] = e
        rs[i] = r
    return ok, y, ev, rs


@njit(cache=True)
def policy_batch_fast(states, T, scale, mu, c_l1, fr, tau_c, max_steps, rc1, rc2, rej):
    n = states.shape[0]
    ok = np.zeros(n, dtype=np.bool_)
    y = np.empty((n, 6))
    ev = np.zeros(n)
    rs = np.zeros(n, dtype=np.int64)
    xbuf = np.zeros(25)
    flags = np.zeros(2)
    for i in range(n):
        o, yy, e, r, _ds = sundman_policy_one_fast(states[i], T[i], scale, mu, c_l1, fr, tau_c, max_steps, rc1, rc2, rej, xbuf, flags)
        ok[i] = o
        for j in range(6):
            y[i, j] = yy[j]
        ev[i] = e
        rs[i] = r
    return ok, y, ev, rs


@njit(parallel=True, cache=True)
def policy_batch_fast_par(states, T, scale, mu, c_l1, fr, tau_c, max_steps, rc1, rc2, rej):
    n = states.shape[0]
    ok = np.zeros(n, dtype=np.bool_)
    y = np.empty((n, 6))
    ev = np.zeros(n)
    rs = np.zeros(n, dtype=np.int64)
    for i in prange(n):
        xbuf = np.zeros(25)
        flags = np.zeros(2)
        o, yy, e, r, _ds = sundman_policy_one_fast(states[i], T[i], scale, mu, c_l1, fr, tau_c, max_steps, rc1, rc2, rej, xbuf, flags)
        ok[i] = o
        for j in range(6):
            y[i, j] = yy[j]
        ev[i] = e
        rs[i] = r
    return ok, y, ev, rs


# ---------------------------------------------------------------------------------------------------------------------
# adaptive batches
# ---------------------------------------------------------------------------------------------------------------------









# ---------------------------------------------------------------------------------------------------------------------
# the solver set the benchmark compares (name -> callable(states, T, rtol) -> (ok, y, nfev))
# ---------------------------------------------------------------------------------------------------------------------

class Solvers:
    """Bundle of the compiled adaptive solvers for one mu: .run(name, states, T, rtol, parallel=False) -> (ok, y_end, nfev, reason)."""
    NAMES = ("DP54_scipy", "DP54_PI", "DOP853")

    def __init__(self, mu=MU, fast=False):
        """fast=False: the SciPy-faithful right-hand side (the compiled nfev equals SciPy's). fast=True: the micro-optimized right-hand side (_rhs_fast)."""
        self.mu = float(mu)
        self.fast = bool(fast)
        self.rc1, self.rc2 = collision_radii(mu)
        self.rej = EJECTION_RADIUS
        t = scipy_tableaux()
        self.A, self.B, self.E3, self.E5 = (np.ascontiguousarray(a) for a in t["dop853"])
        self.du_km = du_km_for_mu(mu)

    def run(self, name, states, T, rtol, parallel=False):
        s = np.ascontiguousarray(states, dtype=np.float64)
        T = np.ascontiguousarray(T, dtype=np.float64)
        K = KF if self.fast else KR
        if name in ("DP54_scipy", "DP54_PI"):
            ctrl = CTRL_SCIPY if name == "DP54_scipy" else CTRL_PI
            f = K.dp54_batch_par if parallel else K.dp54_batch
            ok, y, nf, rs = f(s, T, float(rtol), ATOL_RATIO, self.mu, ctrl, MAX_NFEV, self.rc1, self.rc2, self.rej)
        elif name == "DOP853":
            f = K.dop853_batch_par if parallel else K.dop853_batch
            ok, y, nf, rs = f(s, T, float(rtol), ATOL_RATIO, self.mu, self.A, self.B, self.E3, self.E5, MAX_NFEV, self.rc1, self.rc2, self.rej)
        else:
            raise ValueError(name)
        return ok, y, nf, rs

    def errors_km(self, y, ref_pos):
        return np.linalg.norm(y[:, :3] - ref_pos, axis=1) * self.du_km


class SundmanRunner:
    """The Sundman-time RK4 integrator at given per-orbit fictitious steps (the model is somewhere else): .run_ds(states, T, ds, parallel)."""

    def __init__(self, mu=MU, tau_c=SUNDMAN_TAU_C, max_steps=4_000_000, fast=False):
        """fast=False: Part C's kernel (the one every accuracy result is computed with). fast=True: the optimized kernel of the wall-clock arm."""
        self.mu, self.tau_c, self.max_steps = float(mu), float(tau_c), int(max_steps)
        self.fast = bool(fast)
        self.rc1, self.rc2 = collision_radii(mu)
        self.rej = EJECTION_RADIUS
        self.du_km = du_km_for_mu(mu)

    def run_ds(self, states, T, ds, parallel=False):
        f = (rk4s_fixed_batch_fast_par if parallel else rk4s_fixed_batch_fast) if self.fast else (rk4s_fixed_batch_par if parallel else rk4s_fixed_batch)
        return f(np.ascontiguousarray(states, dtype=np.float64), np.ascontiguousarray(T, dtype=np.float64), np.ascontiguousarray(ds, dtype=np.float64),
                 self.mu, self.tau_c, self.max_steps, self.rc1, self.rc2, self.rej)

    def errors_km(self, y, ref_pos):
        return np.linalg.norm(y[:, :3] - ref_pos, axis=1) * self.du_km


class SundmanPolicy(SundmanRunner):
    """The frozen `sund` policy as one compiled pipeline. `models` as stored in part_c/frozen/sund/policy_frozen.joblib."""

    def __init__(self, models, mu=MU, tau_c=SUNDMAN_TAU_C, max_steps=4_000_000, fast=False):
        super().__init__(mu, tau_c, max_steps, fast)
        self.cp = cm.CompiledHGBPolicy(models, mu=mu)
        if not (self.cp.use_osc and self.cp.gate_kind == 3):
            raise ValueError("expected the single-regressor Sundman policy with the periapsis features")
        self.c_l1 = self.cp.c_l1

    def run(self, states, T, scale, parallel=False):
        s = np.ascontiguousarray(states, dtype=np.float64)
        T = np.ascontiguousarray(T, dtype=np.float64)
        f = (policy_batch_fast_par if parallel else policy_batch_fast) if self.fast else (policy_batch_par if parallel else policy_batch)
        return f(s, T, float(scale), self.mu, self.c_l1, self.cp.fr, self.tau_c, self.max_steps, self.rc1, self.rc2, self.rej)

    def raw_ds(self, states, T):
        return self.cp.batch(states, T)[1]


# ---------------------------------------------------------------------------------------------------------------------
# rtol certification for a compiled solver (the project's rule, certify_rtol_lazy, on the compiled kernels)
# ---------------------------------------------------------------------------------------------------------------------

def certify_rtol_compiled(solvers, name, states, T, ref_pos, tol_km=100.0, alpha=0.05, delta=0.1, grid=RTOL_GRID_B, equal_risk_target=None,
                          refine_steps=10, parallel=True):
    """Loosest certified rtol (Bonferroni over the grid, scanning from the loosest value, stopping at the first rejection) and, if
    equal_risk_target is given, the loosest rtol whose realized violation rate on the rows is <= the target (grid scan, then log10 bisection
    between the loosest value that meets it and the next looser one). A run that fails or ends more than tol_km from the reference is a violation."""
    n = len(states)
    thr = delta / len(grid)
    chain, chosen, eqr = [], None, None

    def viol(rt):
        ok, y, nf, _rs = solvers.run(name, states, T, rt, parallel)
        err = solvers.errors_km(y, ref_pos)
        bad = ~ok | ~(err <= tol_km)
        return int(bad.sum()), float(np.mean(nf[ok])) if ok.any() else float("nan")

    for rt in sorted(grid, reverse=True):
        k, mnf = viol(rt)
        p = float(binom.cdf(k, n, alpha))
        chain.append({"rtol": float(rt), "n": n, "violations": k, "violation_rate": k / n, "p_value": p, "certified": bool(p <= thr), "mean_nfev": mnf})
        if chosen is None and p <= thr:
            chosen = float(rt)
        if eqr is None and equal_risk_target is not None and k / n <= equal_risk_target:
            eqr = float(rt)
        if chosen is not None and (equal_risk_target is None or eqr is not None):
            break
    if equal_risk_target is not None and eqr is not None and refine_steps > 0:
        looser = [c for c in chain if c["rtol"] > eqr]
        if looser and looser[-1]["violation_rate"] > equal_risk_target:
            lo, hi = math.log10(eqr), math.log10(min(c["rtol"] for c in looser))
            for _ in range(refine_steps):
                mid = 0.5 * (lo + hi)
                k, mnf = viol(10.0 ** mid)
                chain.append({"rtol": 10.0 ** mid, "n": n, "violations": k, "violation_rate": k / n, "p_value": float(binom.cdf(k, n, alpha)),
                              "certified": None, "refine": True, "mean_nfev": mnf})
                if k / n <= equal_risk_target:
                    lo, eqr = mid, 10.0 ** mid
                else:
                    hi = mid
    return {"method": name, "chosen_rtol": chosen, "equal_risk_rtol": eqr, "equal_risk_target": equal_risk_target, "p_threshold": thr, "n_cal": n,
            "chain": chain}


# ---------------------------------------------------------------------------------------------------------------------
# wall-clock: single orbit repeated inside a compiled loop (no Python dispatch in the timed region)
# ---------------------------------------------------------------------------------------------------------------------

@njit(cache=True)
def _loop_policy(st, T, reps, scale, mu, c_l1, fr, tau_c, max_steps, rc1, rc2, rej):
    xbuf = np.zeros(25)
    flags = np.zeros(2)
    s = st.copy()
    acc = 0.0
    for r in range(reps):
        s[0] = st[0] + 1e-15 * (r % 7)             # keeps the repeated runs distinct for the optimizer; changes nothing measurable
        _o, y, ev, _re, _ds = sundman_policy_one(s, T, scale, mu, c_l1, fr, tau_c, max_steps, rc1, rc2, rej, xbuf, flags)
        acc += y[0] + ev
    return acc


@njit(cache=True)
def _loop_policy_fast(st, T, reps, scale, mu, c_l1, fr, tau_c, max_steps, rc1, rc2, rej):
    xbuf = np.zeros(25)
    flags = np.zeros(2)
    s = st.copy()
    acc = 0.0
    for r in range(reps):
        s[0] = st[0] + 1e-15 * (r % 7)
        _o, y, ev, _re, _ds = sundman_policy_one_fast(s, T, scale, mu, c_l1, fr, tau_c, max_steps, rc1, rc2, rej, xbuf, flags)
        acc += y[0] + ev
    return acc






def time_per_orbit(fn, target_ns=1.0e6, repeats=3):
    """fn(reps) -> checksum. Median-free robust estimate: the minimum over `repeats` of (elapsed / reps), reps chosen so one repeat takes about target_ns."""
    t0 = time.perf_counter_ns()
    fn(1)
    t1 = max(time.perf_counter_ns() - t0, 50)
    reps = int(max(1, min(20000, target_ns / t1)))
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        fn(reps)
        best = min(best, (time.perf_counter_ns() - t0) / reps)
    return best
