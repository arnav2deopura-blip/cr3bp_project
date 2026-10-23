"""
Validation utilities for the 3D CR3BP hurdle model.

These all operate on an already-generated dataset and/or a trained model
(tolerance sensitivity, a dense-grid cross-check of the halving search, an
accuracy comparison against a high-precision DOP853 reference trajectory,
and validation against approximate Richardson analytic halo orbits), so
they belong with the analysis half of the two-file pipeline. They're
imported by run_3d_analysis_from_data.py; they used to live directly in
run_3d_pipeline.py, which made the analysis script import from a sibling
script instead of from the package.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import MU, INTEGRATOR_TOLERANCES_3D, EJECTION_RADIUS, collision_radii
from .physics_3d import (
    add_orbit_features_3d,
    INTEGRATORS_3D,
    max_jacobi_drift_3d,
    reference_solution_3d,
)
from .search_3d import find_dt_for_integrator_cr3bp_3d
from .richardson import richardson_halo_initial_conditions
from .modeling_3d import predict_hurdle
from .search_3d import _dt_upper_bound_3d


# =============================================================================
# TOLERANCE SENSITIVITY ANALYSIS
# =============================================================================

def tolerance_sensitivity_analysis(data, mu=MU, sim_time=20.0, num_samples=200, tolerances=(0.001, 0.01, 0.02)):
    results = []
    sample = data.sample(n=min(num_samples, len(data)), random_state=42)
    for idx, row in sample.iterrows():
        state = [row["x"], row["y"], row["z"], row["vx"], row["vy"], row["vz"]]
        for tol in tolerances:
            dt = find_dt_for_integrator_cr3bp_3d("rk4", state, sim_time, mu, tol_jacobi=tol)
            results.append({"orbit_idx": idx, "tolerance": tol, "dt_rk4": dt})
    return pd.DataFrame(results)


# =============================================================================
# DENSE SWEEP VALIDATION
# =============================================================================

def dense_sweep_validation(data, mu=MU, sim_time=20.0, num_samples=50, dt_min=1e-4, dt_max=0.5,
                           n_steps=500, tolerance=1e-5):
    sample = data.sample(n=min(num_samples, len(data)), random_state=42)
    results = []
    for idx, row in sample.iterrows():
        state = [row["x"], row["y"], row["z"], row["vx"], row["vy"], row["vz"]]
        grid_top = min(dt_max, max(_dt_upper_bound_3d(state, mu), 0.005))
        dt_grid = np.linspace(dt_min, grid_top, n_steps)
        largest_stable = dt_min
        for dt in dt_grid:
            success, drift, _, _ = max_jacobi_drift_3d(INTEGRATORS_3D["rk4"], state, dt, sim_time, mu)
            if success and drift <= tolerance:
                largest_stable = dt
        search_dt = find_dt_for_integrator_cr3bp_3d("rk4", state, sim_time, mu, tol_jacobi=tolerance)
        results.append({
            "orbit_idx": idx,
            "search_dt": search_dt,
            "dense_dt": largest_stable,
            "relative_error": abs(search_dt - largest_stable) / max(largest_stable, 1e-9),
        })
    return pd.DataFrame(results)


# =============================================================================
# ACCURACY COMPARISON
# =============================================================================

def accuracy_comparison(data, model, mu=MU, sim_time=20.0, num_samples=100):
    sample = data.sample(n=min(num_samples, len(data)), random_state=42)
    results = []
    for idx, row in sample.iterrows():
        row_sim_time = row["sim_time"] if "sim_time" in row and np.isfinite(row["sim_time"]) else sim_time
        state = [row["x"], row["y"], row["z"], row["vx"], row["vy"], row["vz"]]
        cr1, cr2 = collision_radii(mu)
        t_ref, y_ref = reference_solution_3d(state, row_sim_time, mu)
        pred = predict_hurdle(model, pd.DataFrame([row]))
        for integrator in ["euler", "rk4", "leapfrog"]:
            if not pred[integrator]["feasible"][0]:
                continue
            dt_pred = pred[integrator]["dt"][0]
            if not np.isfinite(dt_pred) or dt_pred <= 0:
                continue
            step_func = INTEGRATORS_3D[integrator]
            curr = state[:]
            t = 0.0
            traj = [curr]
            times = [0.0]
            max_steps = int(row_sim_time / dt_pred) + 10
            steps = 0
            diverged = False
            while t < row_sim_time and steps < max_steps:
                curr = step_func(curr, dt_pred, mu)
                r1 = math.sqrt((curr[0] + mu)**2 + curr[1]**2 + curr[2]**2)
                r2 = math.sqrt((curr[0] + mu - 1.0)**2 + curr[1]**2 + curr[2]**2)
                r_origin = math.sqrt(curr[0]**2 + curr[1]**2 + curr[2]**2)
                if (r1 < cr1 or r2 < cr2 or r_origin > EJECTION_RADIUS
                        or not all(math.isfinite(c) for c in curr)):
                    diverged = True
                    break
                t += dt_pred
                times.append(t)
                traj.append(curr)
                steps += 1
            if diverged or len(traj) < 2:
                continue
            if t > row_sim_time:
                prev_t = times[-2]
                alpha = (row_sim_time - prev_t) / dt_pred
                prev_state = np.array(traj[-2])
                curr_state = np.array(curr)
                final_state = prev_state + alpha * (curr_state - prev_state)
                traj[-1] = final_state.tolist()
                times[-1] = row_sim_time
            traj = np.array(traj)
            y_ref_interp = (
                np.interp(times, t_ref, y_ref[:, 0]),
                np.interp(times, t_ref, y_ref[:, 1]),
                np.interp(times, t_ref, y_ref[:, 2]),
            )
            final_pos_err = np.linalg.norm(
                traj[-1, :3] - np.array([y_ref_interp[0][-1], y_ref_interp[1][-1], y_ref_interp[2][-1]])
            )
            ref_angle = np.arctan2(y_ref_interp[1][-1], y_ref_interp[0][-1])
            pred_angle = np.arctan2(traj[-1, 1], traj[-1, 0])
            phase_err = abs(pred_angle - ref_angle)
            phase_err = min(phase_err, 2 * np.pi - phase_err)
            results.append({
                "orbit_idx": idx,
                "integrator": integrator,
                "dt_pred": dt_pred,
                "pos_err": final_pos_err,
                "phase_err": phase_err,
            })
    return pd.DataFrame(results)


# =============================================================================
# RICHARDSON HALO VALIDATION
# =============================================================================

def validate_on_richardson_halo(model, mu=MU, az_values=(0.01, 0.02, 0.03, 0.04)):
    # az_values kept within richardson.MAX_VERIFIED_AZ for both L1 and L2
    # (L2 was only verified up to Az~0.05; L1 up to Az~0.085). Going higher
    # will raise RuntimeError for some (point, Az) pairs -- caught below and
    # skipped -- rather than silently returning a bad orbit. See
    # richardson.py's module docstring for what "verified" means here and
    # how to extend the range if your paper needs larger Az.
    results = []
    for point in ["L1", "L2"]:
        for az in az_values:
            try:
                state0 = richardson_halo_initial_conditions(point, az, mu)
            except Exception as e:
                print(f"Richardson failed for {point} Az={az}: {e}")
                continue
            df = pd.DataFrame([{
                "x": state0[0], "y": state0[1], "z": state0[2],
                "vx": state0[3], "vy": state0[4], "vz": state0[5],
                "sim_time": 20.0,
            }])
            df = add_orbit_features_3d(df, mu, use_variational_ftle=True, ftle_T=20.0)
            pred = predict_hurdle(model, df)
            for integrator in ["euler", "rk4", "leapfrog"]:
                feasible = pred[integrator]["feasible"][0]
                if not feasible:
                    continue
                dt_pred = pred[integrator]["dt"][0]
                step_func = INTEGRATORS_3D[integrator]
                success, drift, _, _ = max_jacobi_drift_3d(step_func, state0, dt_pred, 20.0, mu)
                tol = INTEGRATOR_TOLERANCES_3D[integrator]
                # state0 is now a differentially-corrected halo IC (periodic
                # to ~1e-10 in the shooting residual, not a third-order
                # analytic approximation -- see richardson.py), so this
                # drift reflects the integrator's own stability at dt_pred,
                # not error inherited from an approximate IC.
                results.append({
                    "point": point,
                    "Az": az,
                    "integrator": integrator,
                    "dt_pred": dt_pred,
                    "jacobi_drift": drift,
                    "stable": success and drift <= tol,
                })
    return pd.DataFrame(results)