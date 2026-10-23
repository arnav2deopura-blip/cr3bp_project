from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pandas as pd

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from .config import (
    MU,
    EJECTION_RADIUS,
    FTLE_INTEGRATION_TIME,
    FTLE_PERTURBATION,
    FTLE_MAX_STEP,
    FTLE_STEP_SAFETY,
    FTLE_DT_FLOOR,
    FTLE_MAX_STEPS,
    NEAR_MOON_R2_MIN_FACTOR,
    NEAR_MOON_R2_MAX,
    REFERENCE_RTOL,
    REFERENCE_ATOL,
    REFERENCE_MAX_STEP,
    collision_radii,
    system_params,
)

State3D = list[float]  # [x, y, z, vx, vy, vz]
Integrator3D = Callable[[State3D, float, float], State3D]

REASON_OK = 0
REASON_COLLISION = 1
REASON_EJECTION = 2
REASON_NONFINITE = 3
REASON_STEP_CAP = 4
# Added for the accuracy-based (DOP853-reference) search in search_3d.py;
# unused by the original Jacobi-drift search (max_jacobi_drift_3d) above.
REASON_ACCURACY_FLOOR = 5       # trajectory itself was fine (finite, no
                                 # collision/ejection) but no tested dt down
                                 # to dt_min matched the reference closely
                                 # enough -- read as "infeasible: needs finer
                                 # resolution than this budget allows", not a
                                 # genuine integrator/physical failure.
REASON_REFERENCE_DIVERGENT = 6  # the DOP853 reference trajectory itself
                                 # collided/ejected before reaching sim_time,
                                 # so "accuracy vs reference at sim_time" is
                                 # undefined -- a property of the orbit, not
                                 # of any one integrator.


# =============================================================================
# NUMBA-OPTIMIZED CORE FUNCTIONS
# =============================================================================

@njit(cache=True)
def _cr3bp_derivatives_3d_numba(state, mu):
    x, y, z, vx, vy, vz = state
    r1 = math.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = math.sqrt((x + mu - 1.0)**2 + y**2 + z**2)
    r1 = max(r1, 1e-9)
    r2 = max(r2, 1e-9)
    ax = 2.0 * vy + x - (1.0 - mu) * (x + mu) / r1**3 - mu * (x + mu - 1.0) / r2**3
    ay = -2.0 * vx + y - (1.0 - mu) * y / r1**3 - mu * y / r2**3
    az = -(1.0 - mu) * z / r1**3 - mu * z / r2**3
    return np.array([vx, vy, vz, ax, ay, az])

@njit(cache=True)
def _step_euler_3d_numba(state, dt, mu):
    deriv = _cr3bp_derivatives_3d_numba(state, mu)
    return state + dt * deriv

@njit(cache=True)
def _step_rk4_3d_numba(state, dt, mu):
    k1 = _cr3bp_derivatives_3d_numba(state, mu)
    s2 = state + 0.5 * dt * k1
    k2 = _cr3bp_derivatives_3d_numba(s2, mu)
    s3 = state + 0.5 * dt * k2
    k3 = _cr3bp_derivatives_3d_numba(s3, mu)
    s4 = state + dt * k3
    k4 = _cr3bp_derivatives_3d_numba(s4, mu)
    return state + dt / 6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4)

@njit(cache=True)
def _accel_3d_numba(x, y, z, mu):
    r1 = math.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = math.sqrt((x + mu - 1.0)**2 + y**2 + z**2)
    r1 = max(r1, 1e-9)
    r2 = max(r2, 1e-9)
    Ux = -(1.0 - mu) * (x + mu) / r1**3 - mu * (x + mu - 1.0) / r2**3
    Uy = -(1.0 - mu) * y / r1**3 - mu * y / r2**3
    Uz = -(1.0 - mu) * z / r1**3 - mu * z / r2**3
    return Ux, Uy, Uz


@njit(cache=True)
def _step_leapfrog_symplectic_3d_numba(state, dt, mu):
    x, y, z, vx, vy, vz = state
    px = vx - y
    py = vy+ x
    pz = vz

    Ux, Uy, Uz = _accel_3d_numba(x, y, z, mu)
    px += 0.5 * dt * Ux
    py += 0.5 * dt * Uy
    pz += 0.5 * dt * Uz

    c = math.cos(dt)
    s = math.sin(dt)
    wr = x + px * dt
    wi = y + py * dt
    x = wr * c + wi * s
    y = -wr * s + wi * c
    pxn = px * c + py * s
    pyn = -px * s + py * c
    px = pxn
    py = pyn
    z = z + pz * dt

    Ux, Uy, Uz = _accel_3d_numba(x, y, z, mu)
    px += 0.5 * dt * Ux
    py += 0.5 * dt * Uy
    pz += 0.5 * dt * Uz

    return np.array([x, y, z, px + y, py - x, pz])


# -----------------------------------------------------------------------------
# Numba versions of max_jacobi_drift for each integrator
#
# FIXED: _max_jacobi_drift_euler_numba and _max_jacobi_drift_leapfrog_numba
# previously returned 3-tuples (success, max_drift, t) with no reason code,
# while _max_jacobi_drift_rk4_numba returned a 4-tuple (success, max_drift,
# t, reason). Every caller of max_jacobi_drift_3d() unpacks 4 values, so
# calling this with euler or leapfrog raised
# "ValueError: not enough values to unpack (expected 4, got 3)" -- this hit
# on literally the first orbit of any data-generation run, since
# label_orbit_3d() calls euler first. Both functions now return the same
# 4-tuple shape as rk4, with the same REASON_* granularity (collision vs.
# ejection vs. non-finite vs. step-cap) instead of one combined check.
#
# Also standardized the step-cap check from "t < 0.9 * limit_t" to
# "t < limit_t" to match rk4 exactly. The 0.9 slack was inert in practice
# (max_steps=5_000_000 means the step cap is essentially never hit before
# t reaches limit_t), but it was a real inconsistency across integrators;
# removed rather than left as silent divergence between integrators.
# -----------------------------------------------------------------------------

@njit(cache=True)
def _max_jacobi_drift_euler_numba(state, dt, limit_t, mu, max_steps, collision_radius1, collision_radius2, ejection_radius):
    x, y, z, vx, vy, vz = state
    r1 = math.sqrt((x+mu)**2 + y*y + z*z); r1 = max(r1, 1e-9)
    r2 = math.sqrt((x+mu-1.0)**2 + y*y + z*z); r2 = max(r2, 1e-9)
    C0 = x*x + y*y + 2.0*(1.0 - mu)/r1 + 2.0*mu/r2 - (vx*vx + vy*vy + vz*vz)
    max_drift = 0.0
    t = 0.0
    steps = 0
    while t < limit_t and steps < max_steps:
        dx = vx
        dy = vy
        dz = vz
        r1 = math.sqrt((x+mu)**2 + y*y + z*z); r1 = max(r1, 1e-9)
        r2 = math.sqrt((x+mu-1.0)**2 + y*y + z*z); r2 = max(r2, 1e-9)
        ax = 2.0*vy + x - (1.0 - mu)*(x+mu)/r1**3 - mu*(x+mu-1.0)/r2**3
        ay = -2.0*vx + y - (1.0 - mu)*y/r1**3 - mu*y/r2**3
        az = -(1.0 - mu)*z/r1**3 - mu*z/r2**3
        x = x + dx*dt
        y = y + dy*dt
        z = z + dz*dt
        vx = vx + ax*dt
        vy = vy + ay*dt
        vz = vz + az*dt

        r1 = math.sqrt((x+mu)**2 + y*y + z*z)
        r2 = math.sqrt((x+mu-1.0)**2 + y*y + z*z)
        r_origin = math.sqrt(x*x + y*y + z*z)
        if r1 < collision_radius1 or r2 < collision_radius2:
            return False, np.inf, t, REASON_COLLISION
        if r_origin > ejection_radius:
            return False, np.inf, t, REASON_EJECTION
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z) and
                math.isfinite(vx) and math.isfinite(vy) and math.isfinite(vz)):
            return False, np.inf, t, REASON_NONFINITE

        C = x*x + y*y + 2.0*(1.0 - mu)/r1 + 2.0*mu/r2 - (vx*vx + vy*vy + vz*vz)
        if abs(C0) < 1e-12:
            drift = abs(C - C0)
        else:
            drift = abs((C - C0)/C0)
        max_drift = max(max_drift, drift)
        t += dt
        steps += 1
    if t < limit_t:
        return False, np.inf, t, REASON_STEP_CAP
    return True, max_drift, t, REASON_OK

@njit(cache=True)
def _max_jacobi_drift_rk4_numba(state, dt, limit_t, mu, max_steps, collision_radius1, collision_radius2, ejection_radius):
    x, y, z, vx, vy, vz = state
    r1 = math.sqrt((x+mu)**2 + y*y + z*z); r1 = max(r1, 1e-9)
    r2 = math.sqrt((x+mu-1.0)**2 + y*y + z*z); r2 = max(r2, 1e-9)
    C0 = x*x + y*y + 2.0*(1.0 - mu)/r1 + 2.0*mu/r2 - (vx*vx + vy*vy + vz*vz)
    max_drift = 0.0
    t = 0.0
    steps = 0
    while t < limit_t and steps < max_steps:
        s = np.array([x, y, z, vx, vy, vz])
        s = _step_rk4_3d_numba(s, dt, mu)
        x, y, z, vx, vy, vz = s[0], s[1], s[2], s[3], s[4], s[5]

        r1 = math.sqrt((x+mu)**2 + y*y + z*z)
        r2 = math.sqrt((x+mu-1.0)**2 + y*y + z*z)
        r_origin = math.sqrt(x*x + y*y + z*z)
        if r1 < collision_radius1 or r2 < collision_radius2:
            return False, np.inf, t, REASON_COLLISION
        if r_origin > ejection_radius:
            return False, np.inf, t, REASON_EJECTION
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z) and
                math.isfinite(vx) and math.isfinite(vy) and math.isfinite(vz)):
            return False, np.inf, t, REASON_NONFINITE

        C = x*x + y*y + 2.0*(1.0 - mu)/r1 + 2.0*mu/r2 - (vx*vx + vy*vy + vz*vz)
        if abs(C0) < 1e-12:
            drift = abs(C - C0)
        else:
            drift = abs((C - C0)/C0)
        max_drift = max(max_drift, drift)
        t += dt
        steps += 1
    if t < limit_t:
        return False, np.inf, t, REASON_STEP_CAP
    return True, max_drift, t, REASON_OK

@njit(cache=True)
def _max_jacobi_drift_leapfrog_numba(state, dt, limit_t, mu, max_steps, collision_radius1, collision_radius2, ejection_radius):
    x, y, z, vx, vy, vz = state
    r1 = math.sqrt((x+mu)**2 + y*y + z*z); r1 = max(r1, 1e-9)
    r2 = math.sqrt((x+mu-1.0)**2 + y*y + z*z); r2 = max(r2, 1e-9)
    C0 = x*x + y*y + 2.0*(1.0 - mu)/r1 + 2.0*mu/r2 - (vx*vx + vy*vy + vz*vz)
    max_drift = 0.0
    t = 0.0
    steps = 0
    while t < limit_t and steps < max_steps:
        s = np.array([x, y, z, vx, vy, vz])
        s = _step_leapfrog_symplectic_3d_numba(s, dt, mu)
        x, y, z, vx, vy, vz = s[0], s[1], s[2], s[3], s[4], s[5]

        r1 = math.sqrt((x+mu)**2 + y*y + z*z)
        r2 = math.sqrt((x+mu-1.0)**2 + y*y + z*z)
        r_origin = math.sqrt(x*x + y*y + z*z)
        if r1 < collision_radius1 or r2 < collision_radius2:
            return False, np.inf, t, REASON_COLLISION
        if r_origin > ejection_radius:
            return False, np.inf, t, REASON_EJECTION
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z) and
                math.isfinite(vx) and math.isfinite(vy) and math.isfinite(vz)):
            return False, np.inf, t, REASON_NONFINITE

        C = x*x + y*y + 2.0*(1.0 - mu)/r1 + 2.0*mu/r2 - (vx*vx + vy*vy + vz*vz)
        if abs(C0) < 1e-12:
            drift = abs(C - C0)
        else:
            drift = abs((C - C0)/C0)
        max_drift = max(max_drift, drift)
        t += dt
        steps += 1
    if t < limit_t:
        return False, np.inf, t, REASON_STEP_CAP
    return True, max_drift, t, REASON_OK

# -----------------------------------------------------------------------------
# Numba endpoint-tracking integration, one per integrator (same dispatch
# pattern as the max_jacobi_drift_* functions above, for the same numba
# nopython-mode reason: no runtime function-pointer dispatch). Used by the
# accuracy-based (DOP853-reference) search in search_3d.py instead of the
# Jacobi-drift search above. Each step is delegated to this project's
# already-verified step_*_3d_numba functions (no physics formulas are
# reimplemented here). FIXED: the final step is now SHORTENED so the run
# lands exactly on limit_t. Previously the last full step overshot limit_t
# and the caller linearly interpolated positions, which injects an O(dt^2)
# error (km-scale at dt~0.1 even when the integrator error is ~0.01 km) that
# depends on whether dt divides limit_t -- making position error
# artificially non-monotone in dt. prev/t_prev are still returned for API
# compatibility; with exact landing curr is the state at limit_t.
# -----------------------------------------------------------------------------

@njit(cache=True)
def _integrate_endpoint_euler_numba(state, dt, limit_t, mu, max_steps, collision_radius1, collision_radius2, ejection_radius):
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
        s_new = _step_euler_3d_numba(curr, h, mu)
        xn, yn, zn = s_new[0], s_new[1], s_new[2]
        vxn, vyn, vzn = s_new[3], s_new[4], s_new[5]
        r1n = math.sqrt((xn + mu)**2 + yn*yn + zn*zn)
        r2n = math.sqrt((xn + mu - 1.0)**2 + yn*yn + zn*zn)
        r_origin = math.sqrt(xn*xn + yn*yn + zn*zn)
        if r1n < collision_radius1 or r2n < collision_radius2:
            return False, prev, curr, t_prev, t, REASON_COLLISION
        if r_origin > ejection_radius:
            return False, prev, curr, t_prev, t, REASON_EJECTION
        if not (math.isfinite(xn) and math.isfinite(yn) and math.isfinite(zn) and
                math.isfinite(vxn) and math.isfinite(vyn) and math.isfinite(vzn)):
            return False, prev, curr, t_prev, t, REASON_NONFINITE
        prev = curr
        t_prev = t
        curr = s_new
        if last:
            t = limit_t
        else:
            t += h
        steps += 1
    if t < limit_t:
        return False, prev, curr, t_prev, t, REASON_STEP_CAP
    return True, prev, curr, t_prev, t, REASON_OK

@njit(cache=True)
def _integrate_endpoint_rk4_numba(state, dt, limit_t, mu, max_steps, collision_radius1, collision_radius2, ejection_radius):
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
        s_new = _step_rk4_3d_numba(curr, h, mu)
        xn, yn, zn = s_new[0], s_new[1], s_new[2]
        vxn, vyn, vzn = s_new[3], s_new[4], s_new[5]
        r1n = math.sqrt((xn + mu)**2 + yn*yn + zn*zn)
        r2n = math.sqrt((xn + mu - 1.0)**2 + yn*yn + zn*zn)
        r_origin = math.sqrt(xn*xn + yn*yn + zn*zn)
        if r1n < collision_radius1 or r2n < collision_radius2:
            return False, prev, curr, t_prev, t, REASON_COLLISION
        if r_origin > ejection_radius:
            return False, prev, curr, t_prev, t, REASON_EJECTION
        if not (math.isfinite(xn) and math.isfinite(yn) and math.isfinite(zn) and
                math.isfinite(vxn) and math.isfinite(vyn) and math.isfinite(vzn)):
            return False, prev, curr, t_prev, t, REASON_NONFINITE
        prev = curr
        t_prev = t
        curr = s_new
        if last:
            t = limit_t
        else:
            t += h
        steps += 1
    if t < limit_t:
        return False, prev, curr, t_prev, t, REASON_STEP_CAP
    return True, prev, curr, t_prev, t, REASON_OK

@njit(cache=True)
def _integrate_endpoint_leapfrog_numba(state, dt, limit_t, mu, max_steps, collision_radius1, collision_radius2, ejection_radius):
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
        s_new = _step_leapfrog_symplectic_3d_numba(curr, h, mu)
        xn, yn, zn = s_new[0], s_new[1], s_new[2]
        vxn, vyn, vzn = s_new[3], s_new[4], s_new[5]
        r1n = math.sqrt((xn + mu)**2 + yn*yn + zn*zn)
        r2n = math.sqrt((xn + mu - 1.0)**2 + yn*yn + zn*zn)
        r_origin = math.sqrt(xn*xn + yn*yn + zn*zn)
        if r1n < collision_radius1 or r2n < collision_radius2:
            return False, prev, curr, t_prev, t, REASON_COLLISION
        if r_origin > ejection_radius:
            return False, prev, curr, t_prev, t, REASON_EJECTION
        if not (math.isfinite(xn) and math.isfinite(yn) and math.isfinite(zn) and
                math.isfinite(vxn) and math.isfinite(vyn) and math.isfinite(vzn)):
            return False, prev, curr, t_prev, t, REASON_NONFINITE
        prev = curr
        t_prev = t
        curr = s_new
        if last:
            t = limit_t
        else:
            t += h
        steps += 1
    if t < limit_t:
        return False, prev, curr, t_prev, t, REASON_STEP_CAP
    return True, prev, curr, t_prev, t, REASON_OK

# -----------------------------------------------------------------------------
# Numba variational equations integration
# -----------------------------------------------------------------------------

@njit(cache=True)
def _cr3bp_variational_equations_3d_numba(state_and_phi, mu):
    x, y, z, vx, vy, vz = state_and_phi[:6]
    phi = state_and_phi[6:].reshape((6,6)).copy()

    r1 = math.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = math.sqrt((x + mu - 1.0)**2 + y**2 + z**2)
    r1 = max(r1, 1e-9)
    r2 = max(r2, 1e-9)
    r1_3 = r1**3
    r2_3 = r2**3
    r1_5 = r1**5
    r2_5 = r2**5

    mu1 = 1.0 - mu

    dax_dx = 1.0 - mu1*(1.0/r1_3 - 3.0*(x+mu)**2/r1_5) - mu*(1.0/r2_3 - 3.0*(x+mu-1.0)**2/r2_5)
    dax_dy = 3.0*mu1*(x+mu)*y/r1_5 + 3.0*mu*(x+mu-1.0)*y/r2_5
    dax_dz = 3.0*mu1*(x+mu)*z/r1_5 + 3.0*mu*(x+mu-1.0)*z/r2_5

    day_dx = dax_dy
    day_dy = 1.0 - mu1*(1.0/r1_3 - 3.0*y**2/r1_5) - mu*(1.0/r2_3 - 3.0*y**2/r2_5)
    day_dz = 3.0*mu1*y*z/r1_5 + 3.0*mu*y*z/r2_5

    daz_dx = 3.0*mu1*z*(x+mu)/r1_5 + 3.0*mu*z*(x+mu-1.0)/r2_5
    daz_dy = 3.0*mu1*z*y/r1_5 + 3.0*mu*z*y/r2_5
    daz_dz = -mu1*(1.0/r1_3 - 3.0*z**2/r1_5) - mu*(1.0/r2_3 - 3.0*z**2/r2_5)

    A = np.zeros((6,6))
    A[0,3] = 1.0
    A[1,4] = 1.0
    A[2,5] = 1.0
    A[3,0] = dax_dx
    A[3,1] = dax_dy
    A[3,2] = dax_dz
    A[3,4] = 2.0
    A[4,0] = day_dx
    A[4,1] = day_dy
    A[4,2] = day_dz
    A[4,3] = -2.0
    A[5,0] = daz_dx
    A[5,1] = daz_dy
    A[5,2] = daz_dz

    phi_dot = A @ phi
    deriv = np.concatenate((_cr3bp_derivatives_3d_numba(state_and_phi[:6], mu), phi_dot.flatten()))
    return deriv

@njit(cache=True)
def _integrate_variational_rk4_numba(state, mu, T, dt, collision_radius1, collision_radius2, ejection_radius,
                                     step_safety, dt_floor, max_steps):
    phi0 = np.eye(6).flatten()
    full_state = np.concatenate((state, phi0))
    t = 0.0
    steps = 0
    while t < T and steps < max_steps:
        x = full_state[0]
        y = full_state[1]
        z = full_state[2]
        r1c = math.sqrt((x + mu)**2 + y*y + z*z)
        r2c = math.sqrt((x + mu - 1.0)**2 + y*y + z*z)
        h = min(dt, max(step_safety * min(r1c, r2c)**1.5, dt_floor))
        last = False
        if t + h >= T - 1e-12:
            h = T - t
            last = True
        k1 = _cr3bp_variational_equations_3d_numba(full_state, mu)
        s2 = full_state + 0.5*h*k1
        k2 = _cr3bp_variational_equations_3d_numba(s2, mu)
        s3 = full_state + 0.5*h*k2
        k3 = _cr3bp_variational_equations_3d_numba(s3, mu)
        s4 = full_state + h*k3
        k4 = _cr3bp_variational_equations_3d_numba(s4, mu)
        new_state = full_state + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        xn = new_state[0]
        yn = new_state[1]
        zn = new_state[2]
        r1 = math.sqrt((xn + mu)**2 + yn*yn + zn*zn)
        r2 = math.sqrt((xn + mu - 1.0)**2 + yn*yn + zn*zn)
        r_origin = math.sqrt(xn*xn + yn*yn + zn*zn)
        if r1 < collision_radius1 or r2 < collision_radius2:
            return full_state, t, 1, steps
        if r_origin > ejection_radius:
            return full_state, t, 2, steps
        finite = True
        for i in range(new_state.shape[0]):
            if not math.isfinite(new_state[i]):
                finite = False
        if not finite:
            return full_state, t, 3, steps
        full_state = new_state
        if last:
            t = T
        else:
            t += h
        steps += 1
    if t < T:
        return full_state, t, 4, steps
    return full_state, t, 0, steps


FTLE_STATUS_OK = 0
FTLE_STATUS_COLLISION = 1
FTLE_STATUS_EJECTION = 2
FTLE_STATUS_NONFINITE = 3
FTLE_STATUS_STEP_CAP = 4


def compute_ftle_3d_variational_details(state: State3D, mu: float, T: float, dt: float = FTLE_MAX_STEP,
                                        step_safety: float = FTLE_STEP_SAFETY,
                                        dt_floor: float = FTLE_DT_FLOOR) -> dict:
    """
    Largest singular value of the state-transition matrix, as a finite-time
    Lyapunov exponent. FIXED relative to the old fixed-step (dt=0.01) version:
      - the RK4 step is shrunk near either body (step <= step_safety * r^1.5,
        floored at dt_floor), because a fixed 0.01 step is far coarser than the
        local dynamical time near the Moon and could produce spurious
        "collisions" (-> NaN features -> the orbit was silently dropped);
      - if the trajectory really does collide/escape (or hits the step cap)
        before T, the exponent is computed up to the last valid time
        (t_reached) instead of returning NaN. status/steps/t_reached report
        what happened; a NaN is now returned only if the matrix itself is
        non-finite or nothing was integrated.
    """
    r1c, r2c = collision_radii(mu)
    state_arr = np.array(state, dtype=np.float64)
    final_full, t_reached, status, steps = _integrate_variational_rk4_numba(
        state_arr, mu, T, dt, r1c, r2c, EJECTION_RADIUS, step_safety, dt_floor, FTLE_MAX_STEPS)
    out = {"ftle": float("nan"), "t_reached": float(t_reached), "status": int(status), "steps": int(steps)}
    if not np.all(np.isfinite(final_full)) or t_reached < 1e-6:
        return out
    phi = final_full[6:].reshape(6, 6)
    sigma_max = np.linalg.norm(phi, ord=2)
    if not np.isfinite(sigma_max):
        return out
    if sigma_max < 1e-15:
        out["ftle"] = 0.0
        return out
    ftle = math.log(sigma_max) / t_reached
    out["ftle"] = float(max(0.0, ftle)) if np.isfinite(ftle) else float("nan")
    return out


def compute_ftle_3d_variational(state: State3D, mu: float, T: float, dt: float = FTLE_MAX_STEP) -> float:
    return compute_ftle_3d_variational_details(state, mu, T, dt)["ftle"]

# =============================================================================
# PUBLIC API (wrappers)
# =============================================================================

def cr3bp_derivatives_3d(state: State3D, mu: float = MU) -> State3D:
    arr = np.array(state, dtype=np.float64)
    deriv = _cr3bp_derivatives_3d_numba(arr, mu)
    return deriv.tolist()

def step_euler_3d(state: State3D, dt: float, mu: float = MU) -> State3D:
    arr = np.array(state, dtype=np.float64)
    new_arr = _step_euler_3d_numba(arr, dt, mu)
    return new_arr.tolist()

def step_rk4_3d(state: State3D, dt: float, mu: float = MU) -> State3D:
    arr = np.array(state, dtype=np.float64)
    new_arr = _step_rk4_3d_numba(arr, dt, mu)
    return new_arr.tolist()

def step_leapfrog_symplectic_3d(state: State3D, dt: float, mu: float = MU) -> State3D:
    arr = np.array(state, dtype=np.float64)
    new_arr = _step_leapfrog_symplectic_3d_numba(arr, dt, mu)
    return new_arr.tolist()

def max_jacobi_drift_3d(
    step_func: Integrator3D,
    start_state: State3D,
    dt_val: float,
    limit_t: float,
    mu: float = MU,
    max_steps: int = 5_000_000,
) -> tuple[bool, float, float, int]:
    r1_crash, r2_crash = collision_radii(mu)
    if step_func is step_euler_3d:
        result = _max_jacobi_drift_euler_numba(
            np.array(start_state, dtype=np.float64), dt_val, limit_t, mu, max_steps,
            r1_crash, r2_crash, EJECTION_RADIUS)
    elif step_func is step_rk4_3d:
        result = _max_jacobi_drift_rk4_numba(
            np.array(start_state, dtype=np.float64), dt_val, limit_t, mu, max_steps,
            r1_crash, r2_crash, EJECTION_RADIUS)
    elif step_func is step_leapfrog_symplectic_3d:
        result = _max_jacobi_drift_leapfrog_numba(
            np.array(start_state, dtype=np.float64), dt_val, limit_t, mu, max_steps,
            r1_crash, r2_crash, EJECTION_RADIUS)
    else:
        raise ValueError("Unknown step function; must be one of the 3D integrators.")
    return result


def integrate_endpoint_3d(
    step_func: Integrator3D,
    start_state: State3D,
    dt_val: float,
    limit_t: float,
    mu: float = MU,
    max_steps: int = 250_000,
) -> tuple[bool, np.ndarray, np.ndarray, float, float, int]:
    """
    Fixed-step integrate start_state to limit_t, tracking only the last two
    states rather than the whole trajectory or a running Jacobi-drift
    statistic. Used by the accuracy-based (DOP853-reference) search in
    search_3d.py. The nominal step is dt_val; the FINAL step is shortened so
    the run lands exactly on limit_t (no overshoot, no interpolation).

    Returns (success, prev_state, curr_state, t_prev, t_actual, reason):
      - success: False on collision/ejection/non-finite/step-cap (same
        REASON_* codes as max_jacobi_drift_3d above); True if the loop
        reached limit_t.
      - prev_state, t_prev: the state/time before the last step.
      - curr_state, t_actual: the state/time after the last step; on success
        t_actual == limit_t, so curr_state is the state at limit_t.
    max_steps defaults to 250,000 -- comfortably above the 200,000 steps a
    20 TU horizon (config.ACCURACY_SIM_TIME_LOG_MAX) needs at the search's
    dt_min=1e-4 floor. Raise this if you widen either of those.
    """
    r1_crash, r2_crash = collision_radii(mu)
    if step_func is step_euler_3d:
        result = _integrate_endpoint_euler_numba(
            np.array(start_state, dtype=np.float64), dt_val, limit_t, mu, max_steps,
            r1_crash, r2_crash, EJECTION_RADIUS)
    elif step_func is step_rk4_3d:
        result = _integrate_endpoint_rk4_numba(
            np.array(start_state, dtype=np.float64), dt_val, limit_t, mu, max_steps,
            r1_crash, r2_crash, EJECTION_RADIUS)
    elif step_func is step_leapfrog_symplectic_3d:
        result = _integrate_endpoint_leapfrog_numba(
            np.array(start_state, dtype=np.float64), dt_val, limit_t, mu, max_steps,
            r1_crash, r2_crash, EJECTION_RADIUS)
    else:
        raise ValueError("Unknown step function; must be one of the 3D integrators.")
    return result

# =============================================================================
# Feature engineering, orbit generators, reference, diagnostics
# =============================================================================

def jacobi_constant_3d(state: State3D, mu: float = MU) -> float:
    x, y, z, vx, vy, vz = state
    r1 = math.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = math.sqrt((x + mu - 1.0)**2 + y**2 + z**2)
    r1 = max(r1, 1e-9)
    r2 = max(r2, 1e-9)
    C = x*x + y*y + 2.0*(1.0 - mu)/r1 + 2.0*mu/r2 - (vx*vx + vy*vy + vz*vz)
    return C

def radius_to_primary_3d(state: State3D, primary: int, mu: float = MU) -> float:
    x, y, z, _, _, _ = state
    if primary == 1:
        return math.sqrt((x + mu)**2 + y**2 + z**2)
    elif primary == 2:
        return math.sqrt((x + mu - 1.0)**2 + y**2 + z**2)
    else:
        raise ValueError("primary must be 1 or 2")

def compute_ftle_3d_divergence(
    state: State3D,
    mu: float,
    T: float,
    step_func: Integrator3D,
    dt: float,
    perturb: float = FTLE_PERTURBATION,
) -> float:
    perturb_vec = np.random.randn(6)
    perturb_vec = perturb_vec / np.linalg.norm(perturb_vec) * perturb
    state_pert = np.array(state) + perturb_vec

    def integrate(start):
        curr = np.array(start)
        t = 0.0
        while t < T:
            curr = np.array(step_func(curr.tolist(), dt, mu))
            t += dt
        return curr

    state_final = integrate(state)
    state_pert_final = integrate(state_pert.tolist())
    sep_final = np.linalg.norm(state_final - state_pert_final)
    sep_initial = perturb
    if sep_final < 1e-15 or sep_initial < 1e-15:
        return 0.0
    ftle = (1.0 / T) * math.log(sep_final / sep_initial)
    return max(0.0, ftle)

# Alias
compute_ftle_3d = compute_ftle_3d_divergence

def libration_point_location(mu: float, point: str) -> tuple[float, float]:
    if point in ("L4", "L5"):
        x = 0.5 - mu
        y = math.sqrt(3.0) / 2.0 if point == "L4" else -math.sqrt(3.0) / 2.0
        return x, y

    if point == "L1":
        x0 = 0.5 - mu + 0.1
    elif point == "L2":
        x0 = 1.0 - mu + 0.1
    elif point == "L3":
        x0 = -mu - 0.1
    else:
        raise ValueError("point must be L1, L2, L3, L4, L5")

    def f(x):
        r1 = abs(x + mu)
        r2 = abs(x + mu - 1.0)
        return x - (1.0 - mu) * (x + mu) / r1**3 - mu * (x + mu - 1.0) / r2**3

    def df(x):
        r1 = abs(x + mu)
        r2 = abs(x + mu - 1.0)
        term1 = (1.0 - mu) * (1.0 / r1**3 - 3.0 * (x + mu)**2 / r1**5)
        term2 = mu * (1.0 / r2**3 - 3.0 * (x + mu - 1.0)**2 / r2**5)
        return 1.0 - term1 - term2

    for _ in range(100):
        fx = f(x0)
        dfx = df(x0)
        if abs(dfx) < 1e-12:
            break
        x1 = x0 - fx / dfx
        if abs(x1 - x0) < 1e-14:
            x0 = x1
            break
        x0 = x1

    # FIXED: for small mu (e.g. Sun-Earth ~3e-6, Jupiter-Europa ~2.5e-5) the
    # Newton iteration above, started from the Earth-Moon-style guesses, can
    # converge to a DIFFERENT collinear point (L2 came back equal to L1),
    # silently corrupting every L2-based sample and C_L1 -related feature for
    # those mu. Verify the root lies in its own interval and satisfies f(x)=0;
    # otherwise fall back to a bracketed solve (f is strictly monotone on each
    # interval, so each has exactly one root). For Earth-Moon mu the Newton
    # result is valid and returned unchanged.
    def _valid(x):
        if not math.isfinite(x) or abs(f(x)) > 1e-9:
            return False
        if point == "L1":
            return -mu < x < 1.0 - mu
        if point == "L2":
            return x > 1.0 - mu
        return x < -mu

    if not _valid(x0):
        from scipy.optimize import brentq
        eps = 1e-9
        if point == "L1":
            lo, hi = -mu + eps, 1.0 - mu - eps
        elif point == "L2":
            lo, hi = 1.0 - mu + eps, 2.0
        else:
            lo, hi = -2.0, -mu - eps
        x0 = brentq(f, lo, hi, xtol=1e-15, rtol=1e-15)
    return x0, 0.0

def add_orbit_features_3d(dataframe: pd.DataFrame, mu: float = MU, use_variational_ftle: bool = False,
                          ftle_T: float = FTLE_INTEGRATION_TIME) -> pd.DataFrame:
    df = dataframe.copy()
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    z = df["z"].to_numpy(dtype=float)
    vx = df["vx"].to_numpy(dtype=float)
    vy = df["vy"].to_numpy(dtype=float)
    vz = df["vz"].to_numpy(dtype=float)

    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x + mu - 1.0)**2 + y**2 + z**2)
    r1 = np.maximum(r1, 1e-9)
    r2 = np.maximum(r2, 1e-9)

    jacobi = x*x + y*y + 2.0*(1.0 - mu)/r1 + 2.0*mu/r2 - (vx*vx + vy*vy + vz*vz)
    df["jacobi_3d"] = jacobi
    df["r1_3d"] = r1
    df["r2_3d"] = r2
    df["r1_r2_ratio_3d"] = r1 / r2

    x_l1, _ = libration_point_location(mu, "L1")
    C_L1 = jacobi_constant_3d([x_l1, 0.0, 0.0, 0.0, 0.0, 0.0], mu)
    df["c_minus_c_l1_3d"] = jacobi - C_L1

    df["ftle_3d"] = 0.0
    df["ftle_status"] = FTLE_STATUS_OK
    df["ftle_t_frac"] = 1.0
    if use_variational_ftle:
        ftle_values, ftle_status, ftle_frac = [], [], []
        for _, row in df.iterrows():
            state = [row["x"], row["y"], row["z"], row["vx"], row["vy"], row["vz"]]
            d = compute_ftle_3d_variational_details(state, mu, T=ftle_T)
            ftle_values.append(d["ftle"])
            ftle_status.append(d["status"])
            ftle_frac.append(d["t_reached"] / ftle_T)
        df["ftle_3d"] = ftle_values
        df["ftle_status"] = ftle_status
        df["ftle_t_frac"] = ftle_frac
    return df

def reference_solution_3d(
    state: State3D,
    sim_time: float,
    mu: float = MU,
    rtol: float = 1e-12,
    atol: float = 1e-12,
):
    from scipy.integrate import solve_ivp

    def rhs(t, y):
        return cr3bp_derivatives_3d(y.tolist(), mu)

    sol = solve_ivp(rhs, (0.0, sim_time), state, method='DOP853',
                    rtol=rtol, atol=atol, dense_output=False,
                    max_step=0.01)
    return sol.t, sol.y.T


REFERENCE_EVENT_NAMES = {0: "ok", 1: "primary_collision", 2: "secondary_collision", 3: "ejection",
                         4: "initial_inside_body", 9: "solver_failure"}


def initial_inside_body(state: State3D, mu: float = MU) -> bool:
    r1c, r2c = collision_radii(mu)
    x, y, z = state[0], state[1], state[2]
    r1 = math.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = math.sqrt((x + mu - 1.0)**2 + y**2 + z**2)
    return bool(r1 < r1c or r2 < r2c)


def reference_trajectory_dense_3d(
    state: State3D,
    sim_time: float,
    mu: float = MU,
    rtol: float = REFERENCE_RTOL,
    atol: float = REFERENCE_ATOL,
    max_step: float = REFERENCE_MAX_STEP,
    collision_radius=None,
    ejection_radius: float = EJECTION_RADIUS,
):
    """
    Dense-output DOP853 reference trajectory for the accuracy-based
    (position-error) search in search_3d.py. Unlike reference_solution_3d
    above (used by validation_3d.accuracy_comparison for a fixed 100-orbit
    validation sample), this: (a) uses dense_output=True so the caller can
    evaluate the state at any t via the returned sol.sol(t), rather than
    only at solver-chosen grid points, and (b) terminates early via events
    if the reference trajectory itself collides or is ejected before
    sim_time, since "position error vs reference at sim_time" is undefined
    in that case.

    collision_radius: None (default) uses the physical per-body radii for
    mu (config.collision_radii); a float applies to both bodies; a
    (primary, secondary) tuple sets them separately.

    Returns (sol, reached_full_horizon):
      - sol: the scipy OdeSolution/OdeResult. sol.sol(t) gives the
        interpolated 6-state at any t in [0, sol.t[-1]]. sol.reference_event
        is an int code (see REFERENCE_EVENT_NAMES): 0 = reached the horizon,
        1/2 = primary/secondary collision, 3 = ejection, 9 = solver stopped
        without an event.
      - reached_full_horizon: True iff integration reached sim_time without
        a collision/ejection event firing first. If False, sol.t[-1] < sim_time
        is where the reference trajectory itself stopped being physically
        meaningful under this project's collision/ejection convention; the
        caller (search_3d.find_dt_for_integrator_accuracy_3d /
        data_3d.label_orbit_accuracy_3d) is responsible for treating this as
        REASON_REFERENCE_DIVERGENT rather than attempting a comparison.
    """
    from scipy.integrate import solve_ivp

    if collision_radius is None:
        cr1, cr2 = collision_radii(mu)
    elif isinstance(collision_radius, (tuple, list)):
        cr1, cr2 = float(collision_radius[0]), float(collision_radius[1])
    else:
        cr1 = cr2 = float(collision_radius)

    def rhs(t, y):
        return cr3bp_derivatives_3d(y.tolist(), mu)

    def _event_collision1(t, y):
        return math.sqrt((y[0] + mu)**2 + y[1]**2 + y[2]**2) - cr1
    _event_collision1.terminal = True
    _event_collision1.direction = -1

    def _event_collision2(t, y):
        return math.sqrt((y[0] + mu - 1.0)**2 + y[1]**2 + y[2]**2) - cr2
    _event_collision2.terminal = True
    _event_collision2.direction = -1

    def _event_ejection(t, y):
        return math.sqrt(y[0]**2 + y[1]**2 + y[2]**2) - ejection_radius
    _event_ejection.terminal = True
    _event_ejection.direction = 1

    sol = solve_ivp(rhs, (0.0, sim_time), state, method="DOP853",
                     rtol=rtol, atol=atol, max_step=max_step, dense_output=True,
                     events=[_event_collision1, _event_collision2, _event_ejection])
    reached_full_horizon = bool(sol.t[-1] >= sim_time - 1e-9)
    code = 0
    if not reached_full_horizon:
        code = 9
        for i, te in enumerate(sol.t_events):
            if len(te) > 0:
                code = i + 1
                break
    sol.reference_event = code
    return sol, reached_full_horizon


def generate_3d_orbit(
    mu: float = MU,
    point: str = "L1",
    offset: float = 0.02,
    velocity_perturb: float = 0.01,
    sampling_mode: str = "mixed",
    return_label: bool = False,
) -> State3D:
    if sampling_mode == "lagrange":
        x_eq, y_eq = libration_point_location(mu, point)
        x = x_eq + np.random.uniform(-offset, offset)
        y = y_eq + np.random.uniform(-offset, offset)
        z = np.random.uniform(-offset, offset)
        vx = np.random.uniform(-velocity_perturb, velocity_perturb)
        vy = np.random.uniform(-velocity_perturb, velocity_perturb)
        vz = np.random.uniform(-velocity_perturb, velocity_perturb)
        state = [x, y, z, vx, vy, vz]
        return (state, point) if return_label else state

    elif sampling_mode == "wide":
        x = np.random.uniform(-0.5, 1.5)
        y = np.random.uniform(-0.8, 0.8)
        z = np.random.uniform(-0.3, 0.3)
        vx = np.random.uniform(-0.5, 0.5)
        vy = np.random.uniform(-0.5, 0.5)
        vz = np.random.uniform(-0.5, 0.5)
        state = [x, y, z, vx, vy, vz]
        return (state, "wide") if return_label else state

    elif sampling_mode == "mixed":
        if np.random.rand() < 0.8:
            points = ["L1", "L2", "L3", "L4", "L5"]
            chosen = np.random.choice(points)
            x_eq, y_eq = libration_point_location(mu, chosen)
            off = np.random.uniform(0.0, 0.2)
            vpert = np.random.uniform(0.0, 0.2)
            x = x_eq + np.random.uniform(-off, off)
            y = y_eq + np.random.uniform(-off, off)
            z = np.random.uniform(-off, off)
            vx = np.random.uniform(-vpert, vpert)
            vy = np.random.uniform(-vpert, vpert)
            vz = np.random.uniform(-vpert, vpert)
            state = [x, y, z, vx, vy, vz]
            return (state, str(chosen)) if return_label else state
        else:
            return generate_3d_orbit(mu=mu, sampling_mode="wide", return_label=return_label)

    elif sampling_mode == "near_moon":
        # Densifies coverage close to the secondary body (r2_3d small), a
        # region "lagrange" and "wide" essentially never reach in practice
        # (checked against the existing cr3bp_3d_training_data.csv: min
        # r2_3d there is ~0.037, with <0.15% of rows even under r2_3d=0.1),
        # which is far outside the r2 range a near-rectilinear halo orbit
        # (NRHO) or a lunar-flyby trajectory would occupy. Position is drawn
        # at a LOG-uniform random distance r2 between NEAR_MOON_R2_MIN_FACTOR
        # x the secondary's physical radius and NEAR_MOON_R2_MAX (config.py)
        # from the secondary, in a uniformly random direction on
        # the sphere -- NOT constrained to be an actual periodic or even
        # bound orbit; the goal is state-space coverage for training, not
        # physically "nice" trajectories (exact periodic near-Moon orbits
        # are handled separately by richardson.py's differential corrector).
        x_moon = 1.0 - mu
        sysp = system_params(mu)
        r2_lo = NEAR_MOON_R2_MIN_FACTOR * sysp["r2_km"] / sysp["du_km"]
        r2 = float(np.exp(np.random.uniform(np.log(r2_lo), np.log(NEAR_MOON_R2_MAX))))
        u = np.random.normal(size=3)
        norm_u = np.linalg.norm(u)
        if norm_u < 1e-12:
            u = np.array([1.0, 0.0, 0.0])
        else:
            u = u / norm_u
        x = x_moon + r2 * u[0]
        y = 0.0 + r2 * u[1]
        z = 0.0 + r2 * u[2]
        vx = np.random.uniform(-0.6, 0.6)
        vy = np.random.uniform(-0.6, 0.6)
        vz = np.random.uniform(-0.6, 0.6)
        state = [x, y, z, vx, vy, vz]
        return (state, "near_moon") if return_label else state

    else:
        raise ValueError("sampling_mode must be 'lagrange', 'wide', 'mixed', or 'near_moon'")

# =============================================================================
# DICTIONARIES (must come after function definitions)
# =============================================================================

INTEGRATORS_3D: dict[str, Integrator3D] = {
    "euler": step_euler_3d,
    "rk4": step_rk4_3d,
    "leapfrog": step_leapfrog_symplectic_3d,
}

EVALS_PER_STEP_3D = {"euler": 1, "rk4": 4, "leapfrog": 2}