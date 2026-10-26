
"""
Endpoint-only error definition as a sensitivity (Part C plan 4.1.4, Issue 14). The label and the certificate judge a run by its FINAL position error; a run
can be wrong in the middle and right at the end ("lucky endpoint"). Here the chosen fixed-step runs are re-run with their trajectory recorded and compared
with the reference dense output at the integrator's own step times; the maximum error over 50 checkpoints is reported next to the final one.

Rule fixed in advance: if the share of orbits with (max error > tol and final error <= tol) exceeds 2 % in any merged cell, Part C reports BOTH labels for
that cell. It does not change the label definition (that would break comparability with Part B).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MU, du_km_for_mu
from .physics_3d import _step_rk4_3d_numba, _step_leapfrog_fsal_3d_numba, _accel_3d_numba, reference_trajectory_dense_3d

N_CHECK = 50


def trajectory_fixed(integrator, state, dt, T, mu=MU, max_steps=2_000_000):
    """(times, states) at every step end of the fixed-step run (the same arithmetic and landing as the endpoint kernels)."""
    X = np.array(state, dtype=np.float64)
    t = 0.0
    ts, xs = [0.0], [X.copy()]
    Ux = Uy = Uz = 0.0
    if integrator == "leapfrog":
        Ux, Uy, Uz = _accel_3d_numba(X[0], X[1], X[2], mu)
    while t < T and len(ts) < max_steps:
        h, last = dt, False
        if t + dt >= T - 1e-12:
            h, last = T - t, True
        if integrator == "rk4":
            X = _step_rk4_3d_numba(X, h, mu)
        else:
            X, Ux, Uy, Uz = _step_leapfrog_fsal_3d_numba(X, h, mu, Ux, Uy, Uz)
        t = T if last else t + h
        ts.append(t)
        xs.append(X.copy())
        if not np.all(np.isfinite(X)):
            break
    return np.array(ts), np.array(xs)


def trajectory_rk4s(state, ds, T, mu=MU, tau_c=None, max_steps=2_000_000):
    """(times, states) at every step end of the Sundman-time RK4 (s-steps then the physical landing steps), mirroring sundman_3d."""
    from .sundman_3d import _G, SUNDMAN_TAU_C
    tc = SUNDMAN_TAU_C if tau_c is None else tau_c
    X, t = np.array(state, dtype=np.float64), 0.0
    ts, xs = [0.0], [X.copy()]
    Y, tmp = np.empty(7), np.empty(7)
    k1, k2, k3, k4 = (np.empty(7) for _ in range(4))
    while t < T and len(ts) < max_steps:
        Y[:6], Y[6] = X, t
        _G(Y, mu, tc, 0.0, k1)
        dt_pred = k1[6] * ds
        land = t + 1.5 * dt_pred >= T - 1e-12
        if not land:
            _G(Y + 0.5 * ds * k1, mu, tc, 0.0, k2)
            _G(Y + 0.5 * ds * k2, mu, tc, 0.0, k3)
            _G(Y + ds * k3, mu, tc, 0.0, k4)
            Yn = Y + ds / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            if Yn[6] >= T - 1e-12:
                land = True
            else:
                X, t = Yn[:6].copy(), float(Yn[6])
                ts.append(t)
                xs.append(X.copy())
        if land:
            rem = T - t
            nf = max(1, int(np.ceil(rem / dt_pred - 1e-9)))
            h = rem / nf
            for _ in range(nf):
                X = _step_rk4_3d_numba(X, h, mu)
                t = t + h
                ts.append(t)
                xs.append(X.copy())
            ts[-1] = T
            break
        if not np.all(np.isfinite(X)):
            break
    return np.array(ts), np.array(xs)


def checkpoint_errors(ts, xs, sol, T, mu=MU, n_check=N_CHECK):
    """Position errors (km) at the step times nearest to n_check equally spaced times, against the reference dense output at those step times."""
    targets = np.linspace(T / n_check, T, n_check)
    idx = np.unique(np.clip(np.searchsorted(ts, targets), 0, len(ts) - 1))
    ref = sol.sol(ts[idx])[:3].T
    return np.linalg.norm(xs[idx, :3] - ref, axis=1) * du_km_for_mu(mu)


def max_error_audit(df, choices, steps, mu=MU, tol_km=100.0, n_check=N_CHECK):
    """df: usable rows (state columns, sim_time); choices: 'rk4' | 'leapfrog' | 'rk4s' per row; steps: dt (or ds for rk4s) per row. Returns one row per
    orbit with the final error, the maximum checkpoint error and the lucky-endpoint flag."""
    rows = []
    for i in range(len(df)):
        st = df[["x", "y", "z", "vx", "vy", "vz"]].to_numpy(dtype=float)[i]
        T = float(df["sim_time"].iloc[i])
        sol, ok = reference_trajectory_dense_3d(list(st), T, mu)
        if not ok:
            continue
        if choices[i] == "rk4s":
            ts, xs = trajectory_rk4s(st, steps[i], T, mu)
        else:
            ts, xs = trajectory_fixed(choices[i], st, steps[i], T, mu)
        if not np.all(np.isfinite(xs)) or ts[-1] < T - 1e-9:
            rows.append({"i": i, "final_km": np.nan, "max_km": np.nan, "lucky": False, "failed": True})
            continue
        fin = float(np.linalg.norm(xs[-1, :3] - sol.sol(T)[:3]) * du_km_for_mu(mu))
        err = checkpoint_errors(ts, xs, sol, T, mu, n_check)
        mx = float(max(err.max(), fin))
        rows.append({"i": i, "final_km": fin, "max_km": mx, "lucky": bool(mx > tol_km and fin <= tol_km), "failed": False})
    return pd.DataFrame(rows)
