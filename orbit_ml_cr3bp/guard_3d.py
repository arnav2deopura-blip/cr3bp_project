"""
Runtime step-over guard (Part C plan 5.2.3b). The fixed-step kernels already compute r1, r2 at every step for the collision check; the
guard adds tau = min(sqrt(r1^3 / (1 - mu)), sqrt(r2^3 / mu)) at the START of every step and aborts the run when dt / tau > g
(REASON_GUARD). E9 of Part B: violations concentrate at large dt / tau_min (2 % below 0.02, 40 % at 0.2-0.5, 86 % at 0.5-1, 100 % above 1).
The aborted run is then replaced by an adaptive solve from t = 0 (policy_c_3d.run_with_fallback); its cost is the steps already taken x
evaluations per step plus the adaptive nfev, and the fallback's own violations count, so the certificate covers the composite policy.
The guard is evaluated at step starts only, so a pass between two step starts can be missed; the certificate counts whatever happens.
g = inf reproduces the unguarded kernels bit for bit (tests).

Kernels return (ok, prev, curr, t_prev, t, reason, steps).
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

from .config import MU, EJECTION_RADIUS, collision_radii
from .physics_3d import (REASON_OK, REASON_COLLISION, REASON_EJECTION, REASON_NONFINITE, REASON_STEP_CAP, _step_rk4_3d_numba,
                         _step_leapfrog_fsal_3d_numba, _accel_3d_numba)

REASON_GUARD = 7


@njit(cache=True)
def _tau_local(x, y, z, mu):
    r1 = max(math.sqrt((x + mu) ** 2 + y * y + z * z), 1e-9)
    r2 = max(math.sqrt((x + mu - 1.0) ** 2 + y * y + z * z), 1e-9)
    return min(math.sqrt(r1 ** 3 / (1.0 - mu)), math.sqrt(r2 ** 3 / mu))


@njit(cache=True)
def _integrate_endpoint_rk4_guard_numba(state, dt, limit_t, mu, max_steps, collision_radius1, collision_radius2, ejection_radius, g):
    prev = state.copy()
    curr = state.copy()
    t_prev = 0.0
    t = 0.0
    steps = 0
    while t < limit_t and steps < max_steps:
        h = dt
        last = False
        if t + dt >= limit_t - 1e-12:
            h = limit_t - t
            last = True
        if h > g * _tau_local(curr[0], curr[1], curr[2], mu):
            return False, prev, curr, t_prev, t, REASON_GUARD, steps
        s_new = _step_rk4_3d_numba(curr, h, mu)
        xn, yn, zn = s_new[0], s_new[1], s_new[2]
        r1n = math.sqrt((xn + mu) ** 2 + yn * yn + zn * zn)
        r2n = math.sqrt((xn + mu - 1.0) ** 2 + yn * yn + zn * zn)
        r_origin = math.sqrt(xn * xn + yn * yn + zn * zn)
        if r1n < collision_radius1 or r2n < collision_radius2:
            return False, prev, curr, t_prev, t, REASON_COLLISION, steps
        if r_origin > ejection_radius:
            return False, prev, curr, t_prev, t, REASON_EJECTION, steps
        if not (math.isfinite(xn) and math.isfinite(yn) and math.isfinite(zn) and
                math.isfinite(s_new[3]) and math.isfinite(s_new[4]) and math.isfinite(s_new[5])):
            return False, prev, curr, t_prev, t, REASON_NONFINITE, steps
        prev = curr
        t_prev = t
        curr = s_new
        if last:
            t = limit_t
        else:
            t += h
        steps += 1
    if t < limit_t:
        return False, prev, curr, t_prev, t, REASON_STEP_CAP, steps
    return True, prev, curr, t_prev, t, REASON_OK, steps


@njit(cache=True)
def _integrate_endpoint_leapfrog_guard_numba(state, dt, limit_t, mu, max_steps, collision_radius1, collision_radius2, ejection_radius, g):
    prev = state.copy()
    curr = state.copy()
    t_prev = 0.0
    t = 0.0
    steps = 0
    Ux, Uy, Uz = _accel_3d_numba(curr[0], curr[1], curr[2], mu)
    while t < limit_t and steps < max_steps:
        h = dt
        last = False
        if t + dt >= limit_t - 1e-12:
            h = limit_t - t
            last = True
        if h > g * _tau_local(curr[0], curr[1], curr[2], mu):
            return False, prev, curr, t_prev, t, REASON_GUARD, steps
        s_new, Ux, Uy, Uz = _step_leapfrog_fsal_3d_numba(curr, h, mu, Ux, Uy, Uz)
        xn, yn, zn = s_new[0], s_new[1], s_new[2]
        r1n = math.sqrt((xn + mu) ** 2 + yn * yn + zn * zn)
        r2n = math.sqrt((xn + mu - 1.0) ** 2 + yn * yn + zn * zn)
        r_origin = math.sqrt(xn * xn + yn * yn + zn * zn)
        if r1n < collision_radius1 or r2n < collision_radius2:
            return False, prev, curr, t_prev, t, REASON_COLLISION, steps
        if r_origin > ejection_radius:
            return False, prev, curr, t_prev, t, REASON_EJECTION, steps
        if not (math.isfinite(xn) and math.isfinite(yn) and math.isfinite(zn) and
                math.isfinite(s_new[3]) and math.isfinite(s_new[4]) and math.isfinite(s_new[5])):
            return False, prev, curr, t_prev, t, REASON_NONFINITE, steps
        prev = curr
        t_prev = t
        curr = s_new
        if last:
            t = limit_t
        else:
            t += h
        steps += 1
    if t < limit_t:
        return False, prev, curr, t_prev, t, REASON_STEP_CAP, steps
    return True, prev, curr, t_prev, t, REASON_OK, steps


def integrate_endpoint_guard(integrator, state, dt, limit_t, mu=MU, max_steps=4_000_000, g=math.inf):
    """(ok, prev, curr, t_prev, t, reason, steps) of the guarded fixed-step run; reason == REASON_GUARD when the guard fired."""
    rc1, rc2 = collision_radii(mu)
    f = {"rk4": _integrate_endpoint_rk4_guard_numba, "leapfrog": _integrate_endpoint_leapfrog_guard_numba}[integrator]
    return f(np.array(state, dtype=np.float64), float(dt), float(limit_t), float(mu), int(max_steps), rc1, rc2, EJECTION_RADIUS, float(g))
