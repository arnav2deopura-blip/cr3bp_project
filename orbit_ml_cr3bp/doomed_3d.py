
"""
Doomed orbits (Part C plan 4.1.3, Track 1.3). 19 % of the sampled orbits (53.5 % of regime I in Part A) have a reference trajectory that
collides with a body or escapes before the horizon; every Part A / B number excludes them, which is survivorship: a deployed policy meets
them. Here the policy runs on them exactly as on a usable orbit and is judged on the OUTCOME, in a separate ledger that is never pooled
silently with the usable-orbit results.

  run_until_event      fixed-step run (RK4 or leapfrog) that reports WHICH event ended it (1 primary collision, 2 secondary collision,
                       3 escape, 0 none), the time of the last good state, the nominal step and the speed there
  agreement            plan rule: same outcome type and |t_run - t_ref| <= tol / |v_event| (tol in DU, v the speed at the last good
                       state); relaxed rule: same type and the reference event inside the step that detected it
  crash_flag_physical  the near-free two-body flag: the osculating periapsis about a body is inside the body's collision radius and is
                       reached before T (osc_* columns), per body
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

from .config import MU, EJECTION_RADIUS, collision_radii
from .physics_3d import _step_rk4_3d_numba, _step_leapfrog_fsal_3d_numba, _accel_3d_numba

EVENT_NAMES = {0: "none", 1: "primary_collision", 2: "secondary_collision", 3: "ejection"}
NAME_TO_CODE = {"ok": 0, "primary_collision": 1, "secondary_collision": 2, "ejection": 3}


@njit(cache=True)
def _run_until_event_numba(code, state, dt, limit_t, mu, max_steps, rc1, rc2, rej):
    """code 0 = RK4, 1 = leapfrog. Returns (event, t_last_good, steps, speed_at_last_good, finite)."""
    curr = state.copy()
    t = 0.0
    steps = 0
    Ux, Uy, Uz = 0.0, 0.0, 0.0
    if code == 1:
        Ux, Uy, Uz = _accel_3d_numba(curr[0], curr[1], curr[2], mu)
    while t < limit_t and steps < max_steps:
        h = dt
        last = False
        if t + dt >= limit_t - 1e-12:
            h = limit_t - t
            last = True
        if code == 0:
            s_new = _step_rk4_3d_numba(curr, h, mu)
        else:
            s_new, Ux, Uy, Uz = _step_leapfrog_fsal_3d_numba(curr, h, mu, Ux, Uy, Uz)
        xn, yn, zn = s_new[0], s_new[1], s_new[2]
        if not (math.isfinite(xn) and math.isfinite(yn) and math.isfinite(zn) and math.isfinite(s_new[3]) and math.isfinite(s_new[4])
                and math.isfinite(s_new[5])):
            return 0, t, steps, math.sqrt(curr[3] ** 2 + curr[4] ** 2 + curr[5] ** 2), False
        r1n = math.sqrt((xn + mu) ** 2 + yn * yn + zn * zn)
        r2n = math.sqrt((xn + mu - 1.0) ** 2 + yn * yn + zn * zn)
        if r1n < rc1:
            return 1, t, steps, math.sqrt(curr[3] ** 2 + curr[4] ** 2 + curr[5] ** 2), True
        if r2n < rc2:
            return 2, t, steps, math.sqrt(curr[3] ** 2 + curr[4] ** 2 + curr[5] ** 2), True
        if math.sqrt(xn * xn + yn * yn + zn * zn) > rej:
            return 3, t, steps, math.sqrt(curr[3] ** 2 + curr[4] ** 2 + curr[5] ** 2), True
        curr = s_new
        t = limit_t if last else t + h
        steps += 1
    return 0, t, steps, math.sqrt(curr[3] ** 2 + curr[4] ** 2 + curr[5] ** 2), True


def run_until_event(integrator, state, dt, limit_t, mu=MU, max_steps=4_000_000):
    rc1, rc2 = collision_radii(mu)
    code = {"rk4": 0, "leapfrog": 1}[integrator]
    ev, t, steps, v, finite = _run_until_event_numba(code, np.array(state, dtype=np.float64), float(dt), float(limit_t), float(mu),
                                                     int(max_steps), rc1, rc2, EJECTION_RADIUS)
    return {"event": int(ev), "t_last_good": float(t), "steps": int(steps), "speed": float(v), "finite": bool(finite), "dt": float(dt)}


def agreement(run, ref_event_code, ref_t_end, tol_du):
    """Plan rule (strict) and the relaxed one-step rule. `run` from run_until_event."""
    same_type = (run["event"] == ref_event_code) and run["event"] != 0
    strict = bool(same_type and abs(run["t_last_good"] - ref_t_end) <= tol_du / max(run["speed"], 1e-12))
    relaxed = bool(same_type and (run["t_last_good"] - 1e-12 <= ref_t_end <= run["t_last_good"] + run["dt"] + 1e-12))
    return {"same_type": bool(same_type), "strict": strict, "relaxed": relaxed}


def crash_flag_physical(osc, T, mu=MU, margin=1.0):
    """Per-body physical flag from the periapsis features (dict / DataFrame with osc_logrp_{1,2}, osc_tpn_{1,2}, osc_reach_{1,2}):
    the osculating periapsis radius is below margin x the body's collision radius and is reached before the horizon.
    Returns (flag_primary, flag_secondary, any)."""
    rc1, rc2 = collision_radii(mu)
    f1 = (10.0 ** np.asarray(osc["osc_logrp_1"]) < margin * rc1) & (np.asarray(osc["osc_reach_1"]) == 1.0)
    f2 = (10.0 ** np.asarray(osc["osc_logrp_2"]) < margin * rc2) & (np.asarray(osc["osc_reach_2"]) == 1.0)
    return f1, f2, f1 | f2
