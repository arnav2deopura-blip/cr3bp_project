"""
RK4 in a Sundman-transformed time ("rk4s", Part C plan 6.2): classical RK4 with a FIXED step ds in a fictitious time s, where
the physical step is dt = g(x) ds and g shrinks near the bodies. One number per orbit (ds) therefore buys small physical steps exactly
where the dynamical timescale is short, which is what a single fixed physical step cannot do (the paradigm gap tracks the timescale
range, T1 of Part B).

  g(x) = 1 / (1/tau_1 + 1/tau_2 + 1/tau_c),   tau_i = sqrt(r_i^3 / GM_i),  GM_1 = 1 - mu, GM_2 = mu
  augmented system   dX/ds = g(X) F(X),  dt/ds = g(X)    (classical RK4 on the 7-vector (X, t); g reuses the r_1, r_2 of F)
tau_c caps g far from both bodies (g -> tau_c, not infinity). The harmonic sum is used instead of min() so g is smooth and RK4 keeps
its order.

Landing on T: before each step dt_pred = g(X_n) ds. If t_n + 1.5 dt_pred >= T the run finishes in PHYSICAL time with
n_f = ceil((T - t_n)/dt_pred) classical RK4 steps of size (T - t_n)/n_f (so the last steps are no larger than the local transformed step
and no root-finding is needed). An s-step whose realized dt would overshoot T is discarded (its evaluations stay counted) and the run
finishes the same way. Cost = 4 right-hand-side evaluations per step, s-steps and landing steps alike.

A plain Sundman transform is not symplectic, so there is no leapfrog version in Part C.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

from .config import MU, EJECTION_RADIUS, collision_radii
from .physics_3d import (REASON_OK, REASON_COLLISION, REASON_EJECTION, REASON_NONFINITE, REASON_STEP_CAP, _step_rk4_3d_numba)

SUNDMAN_TAU_C = 1.0          # default cap; the pilot chooses among {0.5, 1, 2}
DS_MIN = 1e-4
DS_MAX = 4.0                 # g <= tau_c, so the physical step stays <= DS_MAX * tau_c


@njit(cache=True)
def _G(Y, mu, tau_c, gfix, out):
    """out = (g F(X), g) for Y = (X, t). gfix > 0 replaces g by that constant (used by the tests: gfix = 1 is plain RK4)."""
    x, y, z, vx, vy, vz = Y[0], Y[1], Y[2], Y[3], Y[4], Y[5]
    r1 = max(math.sqrt((x + mu) ** 2 + y ** 2 + z ** 2), 1e-9)
    r2 = max(math.sqrt((x + mu - 1.0) ** 2 + y ** 2 + z ** 2), 1e-9)
    r13 = r1 ** 3
    r23 = r2 ** 3
    ax = 2.0 * vy + x - (1.0 - mu) * (x + mu) / r13 - mu * (x + mu - 1.0) / r23
    ay = -2.0 * vx + y - (1.0 - mu) * y / r13 - mu * y / r23
    az = -(1.0 - mu) * z / r13 - mu * z / r23
    if gfix > 0.0:
        g = gfix
    else:
        g = 1.0 / (math.sqrt((1.0 - mu) / r13) + math.sqrt(mu / r23) + 1.0 / tau_c)
    out[0] = g * vx
    out[1] = g * vy
    out[2] = g * vz
    out[3] = g * ax
    out[4] = g * ay
    out[5] = g * az
    out[6] = g


@njit(cache=True)
def _g_only(state, mu, tau_c):
    r1 = max(math.sqrt((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2), 1e-9)
    r2 = max(math.sqrt((state[0] + mu - 1.0) ** 2 + state[1] ** 2 + state[2] ** 2), 1e-9)
    return 1.0 / (math.sqrt((1.0 - mu) / r1 ** 3) + math.sqrt(mu / r2 ** 3) + 1.0 / tau_c)


@njit(cache=True)
def _bad_state(s, mu, rc1, rc2, rej):
    """0 ok, else the REASON code: collision / ejection / non-finite."""
    xn, yn, zn = s[0], s[1], s[2]
    if not (math.isfinite(xn) and math.isfinite(yn) and math.isfinite(zn) and math.isfinite(s[3]) and math.isfinite(s[4]) and math.isfinite(s[5])):
        return REASON_NONFINITE
    r1n = math.sqrt((xn + mu) ** 2 + yn * yn + zn * zn)
    r2n = math.sqrt((xn + mu - 1.0) ** 2 + yn * yn + zn * zn)
    if r1n < rc1 or r2n < rc2:
        return REASON_COLLISION
    if math.sqrt(xn * xn + yn * yn + zn * zn) > rej:
        return REASON_EJECTION
    return 0


@njit(cache=True)
def _integrate_endpoint_rk4s_numba(state, ds, limit_t, mu, max_steps, rc1, rc2, rej, tau_c, gfix):
    """Returns (ok, prev, curr, t_prev, t, reason, n_s, n_f): n_s s-steps (including discarded overshoots), n_f landing steps."""
    X = state.copy()
    prev = state.copy()
    t = 0.0
    t_prev = 0.0
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
            return False, prev, X, t_prev, t, REASON_STEP_CAP, n_s, n_f
        for i in range(6):
            Y[i] = X[i]
        Y[6] = t
        _G(Y, mu, tau_c, gfix, k1)
        dt_pred = k1[6] * ds
        land = (t + 1.5 * dt_pred >= limit_t - 1e-12)
        if not land:
            for i in range(7):
                tmp[i] = Y[i] + 0.5 * ds * k1[i]
            _G(tmp, mu, tau_c, gfix, k2)
            for i in range(7):
                tmp[i] = Y[i] + 0.5 * ds * k2[i]
            _G(tmp, mu, tau_c, gfix, k3)
            for i in range(7):
                tmp[i] = Y[i] + ds * k3[i]
            _G(tmp, mu, tau_c, gfix, k4)
            for i in range(7):
                tmp[i] = Y[i] + ds / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
            n_s += 1
            bad = _bad_state(tmp, mu, rc1, rc2, rej)
            if bad != 0:
                return False, prev, X, t_prev, t, bad, n_s, n_f
            if tmp[6] >= limit_t - 1e-12:         # the step would overshoot T: discard it and land in physical time
                land = True
            else:
                for i in range(6):
                    prev[i] = X[i]
                    X[i] = tmp[i]
                t_prev = t
                t = tmp[6]
        if land:
            remaining = limit_t - t
            nf = max(1, int(math.ceil(remaining / dt_pred - 1e-9)))
            h = remaining / nf
            for j in range(nf):
                if n_s + n_f >= max_steps:
                    return False, prev, X, t_prev, t, REASON_STEP_CAP, n_s, n_f
                s_new = _step_rk4_3d_numba(X, h, mu)
                n_f += 1
                bad = _bad_state(s_new, mu, rc1, rc2, rej)
                if bad != 0:
                    return False, prev, X, t_prev, t, bad, n_s, n_f
                for i in range(6):
                    prev[i] = X[i]
                    X[i] = s_new[i]
                t_prev = t
                t = t + h
            t = limit_t
            break
    return True, prev, X, t_prev, t, REASON_OK, n_s, n_f


def integrate_endpoint_rk4s(state, ds, limit_t, mu=MU, max_steps=250_000, tau_c=None, gfix=0.0):
    """Python wrapper: (ok, prev, curr, t_prev, t, reason, n_s, n_f). evals = 4 * (n_s + n_f)."""
    rc1, rc2 = collision_radii(mu)
    return _integrate_endpoint_rk4s_numba(np.array(state, dtype=np.float64), float(ds), float(limit_t), float(mu), int(max_steps),
                                          rc1, rc2, EJECTION_RADIUS, float(SUNDMAN_TAU_C if tau_c is None else tau_c), float(gfix))


def rk4s_evals(n_s, n_f):
    return 4 * (int(n_s) + int(n_f))
