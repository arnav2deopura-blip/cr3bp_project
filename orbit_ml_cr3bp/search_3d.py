from __future__ import annotations

import numpy as np

from .config import MU, INTEGRATOR_TOLERANCES_3D, POSITION_TOL_KM, du_km_for_mu
from .physics_3d import (
    INTEGRATORS_3D,
    max_jacobi_drift_3d,
    integrate_endpoint_3d,
    reference_trajectory_dense_3d,
    radius_to_primary_3d,
    jacobi_constant_3d,
    REASON_OK,
    REASON_ACCURACY_FLOOR,
    REASON_REFERENCE_DIVERGENT,
)


def _dt_upper_bound_3d(state, mu=MU):
    r2 = radius_to_primary_3d(state, 2, mu)
    local_time = r2 ** 1.5
    return max(0.001, min(0.5, 0.3 * local_time))


def find_dt_for_integrator_cr3bp_3d(
    integrator_name: str,
    state,
    sim_time: float,
    mu: float = MU,
    dt_min: float = 0.0001,
    tol_jacobi: float | None = None,
    return_details: bool = False,
):
    step_func = INTEGRATORS_3D[integrator_name]
    if tol_jacobi is None:
        tol_jacobi = INTEGRATOR_TOLERANCES_3D[integrator_name]

    dt_max = _dt_upper_bound_3d(state, mu)
    dt_max = max(dt_max, 0.005)

    dt = dt_max
    best_dt = None
    best_drift = None
    last_reason = None
    while dt >= dt_min:
        success, max_drift, _t, reason = max_jacobi_drift_3d(step_func, state, dt, sim_time, mu)
        last_reason = reason
        if success and max_drift <= tol_jacobi:
            best_dt = dt
            best_drift = max_drift
            break
        dt = round(dt / 2.0, 6)

    if best_dt is None:
        if return_details:
            return {
                "dt": dt_min, "censored": False, "reason": last_reason,
                "drift": float("nan"), "C0": jacobi_constant_3d(state, mu),
                "dt_ceiling": dt_max,
            }
        return dt_min

    lower = max(dt_min, best_dt * 0.5)
    upper = min(dt_max, best_dt * 1.5)
    largest_stable = best_dt
    largest_drift = best_drift

    if upper > lower:
        n_fine = 20
        dt_grid = np.linspace(lower, upper, n_fine)
        for dt_val in dt_grid:
            if dt_val <= largest_stable:
                continue
            success, max_drift, _t, _r = max_jacobi_drift_3d(step_func, state, dt_val, sim_time, mu)
            if success and max_drift <= tol_jacobi:
                largest_stable = dt_val
                largest_drift = max_drift

    expand_dt = min(dt_max, largest_stable * 1.1)
    while expand_dt <= dt_max and expand_dt > largest_stable:
        success, max_drift, _t, _r = max_jacobi_drift_3d(step_func, state, expand_dt, sim_time, mu)
        if success and max_drift <= tol_jacobi:
            largest_stable = expand_dt
            largest_drift = max_drift
            expand_dt = min(dt_max, expand_dt * 1.1)
        else:
            break

    largest_stable = round(float(largest_stable), 6)
    censored = largest_stable >= dt_max - 1e-6

    if return_details:
        return {
            "dt": largest_stable, "censored": bool(censored), "reason": REASON_OK,
            "drift": float(largest_drift), "C0": jacobi_constant_3d(state, mu),
            "dt_ceiling": dt_max,
        }
    return largest_stable


# =============================================================================
# ACCURACY-BASED (DOP853-REFERENCE) SEARCH
# =============================================================================
# find_dt_for_integrator_cr3bp_3d above defines "stable" as bounded Jacobi
# (energy) drift. That is a conservation criterion, not an accuracy one: our
# own validation (3d_accuracy_comparison.csv / run_3d_analysis_from_data.py's
# accuracy_comparison) shows a Jacobi-stable RK4 trajectory can still land
# with a median position error of ~1.7e-3 DU (roughly 650 km at DU_KM) from a
# DOP853 reference, worse in the tail -- nowhere near good enough for a
# real navigation/mission-design use case, and it says nothing about which
# choice of timestep actually gets you close to the true trajectory.
#
# This function instead defines "stable" (renamed here "accurate", to be
# explicit about the change) directly: the largest dt for which the
# integrator's own final position at sim_time is within tol_km of a DOP853
# reference trajectory computed once per orbit. It reuses the same
# coarse-halving -> refine -> expand search structure as the Jacobi search
# above (so the two are easy to compare), swapping only the per-candidate
# feasibility check.
#
# Two things this changes deliberately, both requested/expected:
#   1. Horizon: sim_time is meant to be capped by the caller (data_3d.py
#      draws it from config.ACCURACY_SIM_TIME_LOG_MIN/MAX, i.e. 1-20 TU) --
#      this function does not itself enforce a cap, since it is also usable
#      standalone (e.g. against a real mission horizon like CAPSTONE's
#      ~1.61 TU NRHO period).
#   2. "Infeasible" vs "failed": if no tested dt (down to dt_min) gets within
#      tol_km, but the trajectory itself stayed finite with no collision or
#      ejection, that is reported as REASON_ACCURACY_FLOOR ("would need finer
#      resolution than this dt floor allows, most likely local chaotic
#      sensitivity over this horizon") rather than being lumped in with a
#      genuine physical breakdown (REASON_COLLISION / REASON_EJECTION /
#      REASON_NONFINITE), which is a different, worse thing.
#
# A third, orbit-level (not integrator-level) case is NOT decided inside
# this function: if the DOP853 reference itself collides or is ejected
# before sim_time, "accuracy vs reference" is undefined for all three
# integrators at once, not a property of the one being searched here. If you
# call this function directly, check `reached_full_horizon` from
# reference_trajectory_dense_3d yourself first (data_3d.label_orbit_accuracy_3d
# does this for you, and is what generate_3d_data.py --label_mode accuracy
# actually calls).


def find_dt_for_integrator_accuracy_3d(
    integrator_name: str,
    state,
    sim_time: float,
    mu: float = MU,
    dt_min: float = 0.0001,
    tol_km: float | None = None,
    dt_max: float | None = None,
    reference=None,
    return_details: bool = False,
):
    """
    See module-level comment above. `reference`, if given, must be the
    (sol, reached_full_horizon) tuple already returned by
    reference_trajectory_dense_3d for this exact (state, sim_time, mu) --
    passing it in lets a caller compute the (expensive) reference once and
    reuse it across all three integrators, instead of recomputing it here
    per call. If omitted, this function computes it itself (convenient for
    standalone/one-off use, e.g. against a single real orbit).
    """
    if tol_km is None:
        tol_km = POSITION_TOL_KM

    step_func = INTEGRATORS_3D[integrator_name]

    if dt_max is None:
        dt_max = _dt_upper_bound_3d(state, mu)
        dt_max = max(dt_max, 0.005)

    if reference is None:
        sol, reached = reference_trajectory_dense_3d(state, sim_time, mu)
    else:
        sol, reached = reference

    if not reached:
        if return_details:
            return {
                "dt": dt_min, "censored": False, "reason": REASON_REFERENCE_DIVERGENT,
                "pos_err_km": float("nan"), "dt_ceiling": dt_max,
                "reference_final_t": float(sol.t[-1]),
                "reference_event": int(getattr(sol, "reference_event", 0)),
            }
        return dt_min

    ref_final_pos = sol.sol(sim_time)[:3]
    du_km = du_km_for_mu(mu)

    def _pos_err_km(dt_val):
        success, prev, curr, t_prev, t_actual, reason = integrate_endpoint_3d(
            step_func, state, dt_val, sim_time, mu)
        if not success:
            return None, reason
        if t_actual > sim_time and t_actual > t_prev:
            alpha = (sim_time - t_prev) / (t_actual - t_prev)
            final_pos = prev[:3] + alpha * (curr[:3] - prev[:3])
        else:
            final_pos = curr[:3]
        err_km = float(np.linalg.norm(np.asarray(final_pos) - ref_final_pos)) * du_km
        return err_km, REASON_OK

    dt = dt_max
    best_dt = None
    best_err = None
    last_reason = None
    while dt >= dt_min:
        err_km, reason = _pos_err_km(dt)
        last_reason = reason
        if err_km is not None and err_km <= tol_km:
            best_dt = dt
            best_err = err_km
            break
        dt = round(dt / 2.0, 6)

    if best_dt is None:
        reason_out = REASON_ACCURACY_FLOOR if last_reason == REASON_OK else last_reason
        if return_details:
            return {
                "dt": dt_min, "censored": False, "reason": reason_out,
                "pos_err_km": float("nan"), "dt_ceiling": dt_max,
                "reference_final_t": float(sol.t[-1]),
            }
        return dt_min

    lower = max(dt_min, best_dt * 0.5)
    upper = min(dt_max, best_dt * 1.5)
    largest_stable = best_dt
    largest_err = best_err

    if upper > lower:
        n_fine = 20
        for dt_val in np.linspace(lower, upper, n_fine):
            if dt_val <= largest_stable:
                continue
            err_km, _reason = _pos_err_km(dt_val)
            if err_km is not None and err_km <= tol_km:
                largest_stable = dt_val
                largest_err = err_km

    expand_dt = min(dt_max, largest_stable * 1.1)
    while expand_dt <= dt_max and expand_dt > largest_stable:
        err_km, _reason = _pos_err_km(expand_dt)
        if err_km is not None and err_km <= tol_km:
            largest_stable = expand_dt
            largest_err = err_km
            expand_dt = min(dt_max, expand_dt * 1.1)
        else:
            break

    largest_stable = round(float(largest_stable), 6)
    censored = largest_stable >= dt_max - 1e-6

    if return_details:
        return {
            "dt": largest_stable, "censored": bool(censored), "reason": REASON_OK,
            "pos_err_km": float(largest_err), "dt_ceiling": dt_max,
            "reference_final_t": float(sol.t[-1]),
        }
    return largest_stable


# =============================================================================
# MONOTONE-SAFE ACCURACY SEARCH
# =============================================================================
# find_dt_for_integrator_accuracy_3d above returns the first dt (scanning down
# from the ceiling) that meets the tolerance, then probes upward from it. If
# position error is not monotone in dt (a measured problem: the dense-sweep
# audit found non-monotone behavior in 45% of 20 orbits, partly an artifact
# of the old end-of-run interpolation that integrate_endpoint_3d no longer
# does), "the largest dt that passes" is not a stable target: a slightly
# different grid or a slightly different orbit can flip it between very
# different values, which is label noise for the ML model.
#
# This search instead defines the label as the largest dt on a geometric grid
# such that it AND the `window` next-finer grid points ALL meet the tolerance
# (default window=8 points at ratio 2^(1/4) = 2 octaves, i.e. the label is
# safe for every tested step down to dt/4), then refines upward between grid
# points, accepting a refinement point only while every finer refinement
# point also passes. It is a "locally monotone envelope", not a proof of
# monotonicity below the window: only dt/4 and below is unchecked.


def _make_pos_err_fn(step_func, state, sim_time, mu, ref_final_pos):
    du_km = du_km_for_mu(mu)

    def _pos_err_km(dt_val):
        success, prev, curr, t_prev, t_actual, reason = integrate_endpoint_3d(
            step_func, state, dt_val, sim_time, mu)
        if not success:
            return None, reason
        if t_actual > sim_time and t_actual > t_prev:
            alpha = (sim_time - t_prev) / (t_actual - t_prev)
            final_pos = prev[:3] + alpha * (curr[:3] - prev[:3])
        else:
            final_pos = curr[:3]
        return float(np.linalg.norm(np.asarray(final_pos) - ref_final_pos)) * du_km, REASON_OK

    return _pos_err_km


def find_dt_for_integrator_accuracy_monotone_3d(
    integrator_name: str,
    state,
    sim_time: float,
    mu: float = MU,
    dt_min: float = 0.0001,
    tol_km: float | None = None,
    dt_max: float | None = None,
    reference=None,
    return_details: bool = False,
    grid_ratio: float = 2.0 ** 0.25,
    window: int = 8,
    n_refine: int = 6,
):
    if tol_km is None:
        tol_km = POSITION_TOL_KM
    step_func = INTEGRATORS_3D[integrator_name]
    if dt_max is None:
        dt_max = max(_dt_upper_bound_3d(state, mu), 0.005)
    if reference is None:
        sol, reached = reference_trajectory_dense_3d(state, sim_time, mu)
    else:
        sol, reached = reference

    if not reached:
        if return_details:
            return {
                "dt": dt_min, "censored": False, "reason": REASON_REFERENCE_DIVERGENT,
                "pos_err_km": float("nan"), "dt_ceiling": dt_max,
                "reference_final_t": float(sol.t[-1]),
                "reference_event": int(getattr(sol, "reference_event", 0)),
            }
        return dt_min

    ref_final_pos = sol.sol(sim_time)[:3]
    err_fn = _make_pos_err_fn(step_func, state, sim_time, mu, ref_final_pos)
    cache = {}

    def _eval(dt_val):
        key = float(dt_val)
        if key not in cache:
            cache[key] = err_fn(key)
        return cache[key]

    def _ok(dt_val):
        err, _reason = _eval(dt_val)
        return err is not None and err <= tol_km

    n_grid = int(np.floor(np.log(dt_max / dt_min) / np.log(grid_ratio))) + 1
    n_grid = max(n_grid, 1)
    grid = [dt_max * grid_ratio ** (-k) for k in range(n_grid)]

    best_k = None
    k = 0
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
        last_reason = _eval(grid[-1])[1]
        reason_out = REASON_ACCURACY_FLOOR if last_reason == REASON_OK else last_reason
        if return_details:
            return {
                "dt": dt_min, "censored": False, "reason": reason_out,
                "pos_err_km": float("nan"), "dt_ceiling": dt_max,
                "reference_final_t": float(sol.t[-1]),
                "reference_event": 0,
            }
        return dt_min

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
    censored = best_k == 0
    chosen_err = _eval(chosen)[0]
    if return_details:
        return {
            "dt": chosen, "censored": bool(censored), "reason": REASON_OK,
            "pos_err_km": float(chosen_err), "dt_ceiling": dt_max,
            "reference_final_t": float(sol.t[-1]),
            "reference_event": 0,
        }
    return chosen

