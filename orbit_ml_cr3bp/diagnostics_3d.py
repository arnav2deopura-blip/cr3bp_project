"""
Per-orbit physics diagnostics for the failure study (Part B, plan section 5.2).

Everything here is computed from the reference trajectory that the labelling already needs (plus one cheap uncapped
tight-tolerance DOP853 run), saved as extra CSV columns, and used ONLY to describe or explain failures. None of these
columns may be a model input in the deployable model (they need the whole trajectory); the oracle-feature upper bound
(E3b) is the one place they are fed to a model, and it is labelled "not deployable".

Definitions (fixed in advance; `DIAGNOSTICS_VERSION` below is stored in the dataset metadata):
  tau(t)      local timescale, min( sqrt(r1^3 / (1-mu)), sqrt(r2^3 / mu) )
  delta_phys  T * max_t(1/tau) / int_0^T (1/tau) dt   (>= 1; 1 when uniformly paced)
  delta_emp   mean(h) / min(h) of the accepted steps of an UNCAPPED DOP853 run at rtol = atol = 1e-10, last (clipped) step
              dropped, because the reference itself is capped at max_step = 0.01 (Issue 20)
  transit     crossing of x = x_L1 or x = x_L2 with |y| < 0.1 DU between consecutive samples
  n_flyby     local minima of r2(t) below FLYBY_R2 = 0.05 DU
  frac_near   share of [0, T] with r2 < NEAR_R2 = 0.1 DU
Sampling: every accepted reference step plus four interior points of its dense output, so the closest approach is sampled
at a resolution finer than the solver's own steps (those shrink near a body).
"""
from __future__ import annotations

import math

import numpy as np

from .physics_3d import _cr3bp_derivatives_3d_numba, libration_point_location

DIAGNOSTICS_VERSION = 1
FLYBY_R2 = 0.05
NEAR_R2 = 0.1
TRANSIT_ABS_Y = 0.1
EMP_RTOL = 1e-10
_INTERIOR = (0.2, 0.4, 0.6, 0.8)

DIAGNOSTIC_COLUMNS = [
    "t_end", "min_r1", "min_r2", "t_min_r2", "n_flyby", "tau_min", "max_inv_tau", "int_inv_tau", "delta_phys",
    "delta_emp", "n_transit_l1", "n_transit_l2", "frac_near_secondary", "ref_check_km",
]


def _nan_diag():
    return {c: float("nan") for c in DIAGNOSTIC_COLUMNS}


def sample_reference(sol, n_interior=len(_INTERIOR)):
    """Times and states (6, n) at the accepted steps of `sol` and interior dense-output points."""
    t_nodes = np.asarray(sol.t, dtype=float)
    if len(t_nodes) < 2:
        return t_nodes, sol.sol(t_nodes)
    dt = np.diff(t_nodes)
    fr = np.array(_INTERIOR[:n_interior])
    inner = (t_nodes[:-1, None] + dt[:, None] * fr[None, :]).ravel()
    ts = np.unique(np.concatenate([t_nodes, inner]))
    return ts, sol.sol(ts)


def tau_local(r1, r2, mu):
    return np.minimum(np.sqrt(r1 ** 3 / (1.0 - mu)), np.sqrt(r2 ** 3 / mu))


def count_crossings(x, y, x_line, y_abs=TRANSIT_ABS_Y):
    s = np.sign(x - x_line)
    flip = (s[:-1] * s[1:]) < 0
    near = 0.5 * (np.abs(y[:-1]) + np.abs(y[1:])) < y_abs
    return int(np.sum(flip & near))


def empirical_dynamic_range(state, t_end, mu, rtol=EMP_RTOL):
    """mean(h)/min(h) of the accepted steps of an uncapped DOP853 run (last, clipped step dropped)."""
    from scipy.integrate import solve_ivp
    sol = solve_ivp(lambda t, y: _cr3bp_derivatives_3d_numba(y, mu), (0.0, t_end),
                    np.asarray(state, dtype=np.float64), method="DOP853", rtol=rtol, atol=rtol)
    h = np.diff(sol.t)
    if len(h) >= 3:
        h = h[:-1]
    if len(h) == 0 or h.min() <= 0:
        return float("nan")
    return float(h.mean() / h.min())


def reference_check_km(state, t_end, mu, ref_pos, rtol=1e-12, max_step=0.005):
    """Distance (km is applied by the caller via du) between the label reference's final position and a second DOP853 run with
    a different tolerance and half the max step. A cheap per-orbit validity flag (E1 does the full independent-method map)."""
    from scipy.integrate import solve_ivp
    sol = solve_ivp(lambda t, y: _cr3bp_derivatives_3d_numba(y, mu), (0.0, t_end), np.asarray(state, dtype=np.float64),
                    method="DOP853", rtol=rtol, atol=rtol, max_step=max_step)
    if not sol.success or sol.t[-1] < t_end - 1e-9:
        return float("nan")
    return float(np.linalg.norm(sol.y[:3, -1] - ref_pos))


def trajectory_diagnostics_3d(sol, state, mu, du_km, with_emp=True, with_check=True):
    """Diagnostics for one reference solution (sol from physics_3d.reference_trajectory_dense_3d), over [0, sol.t[-1]].
    For a reference-divergent orbit that is the part before the collision/escape, and t_end < sim_time says so."""
    t_end = float(sol.t[-1])
    out = _nan_diag()
    out["t_end"] = t_end
    if t_end <= 0 or len(sol.t) < 2:
        return out
    ts, Y = sample_reference(sol)
    x, y, z = Y[0], Y[1], Y[2]
    r1 = np.sqrt((x + mu) ** 2 + y ** 2 + z ** 2)
    r2 = np.sqrt((x + mu - 1.0) ** 2 + y ** 2 + z ** 2)
    tau = tau_local(r1, r2, mu)
    inv = 1.0 / tau
    integral = float(np.trapezoid(inv, ts)) if hasattr(np, "trapezoid") else float(np.trapz(inv, ts))
    i2 = int(np.argmin(r2))
    out.update({
        "min_r1": float(r1.min()), "min_r2": float(r2.min()), "t_min_r2": float(ts[i2]),
        "tau_min": float(tau.min()), "max_inv_tau": float(inv.max()), "int_inv_tau": integral,
        "delta_phys": float(t_end * inv.max() / integral) if integral > 0 else float("nan"),
    })
    low = r2 < FLYBY_R2
    mins = (r2[1:-1] <= r2[:-2]) & (r2[1:-1] <= r2[2:]) & low[1:-1]
    out["n_flyby"] = int(mins.sum())
    near = (r2 < NEAR_R2).astype(float)
    nint = float(np.trapezoid(near, ts)) if hasattr(np, "trapezoid") else float(np.trapz(near, ts))
    out["frac_near_secondary"] = nint / t_end
    x_l1, _ = libration_point_location(mu, "L1")
    x_l2, _ = libration_point_location(mu, "L2")
    out["n_transit_l1"] = count_crossings(x, y, x_l1)
    out["n_transit_l2"] = count_crossings(x, y, x_l2)
    if with_emp:
        out["delta_emp"] = empirical_dynamic_range(state, t_end, mu)
    if with_check:
        out["ref_check_km"] = reference_check_km(state, t_end, mu, sol.sol(t_end)[:3]) * du_km
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Hill / Jacobi regime (I..V), from the initial Jacobi constant
# ---------------------------------------------------------------------------------------------------------------------

def jacobi_at_points(mu):
    """C_L1..C_L5 for mass ratio mu (Jacobi constant of a body at rest at each libration point)."""
    from .physics_3d import jacobi_constant_3d
    c = {}
    for p in ("L1", "L2", "L3", "L4", "L5"):
        xl, yl = libration_point_location(mu, p)
        c[p] = jacobi_constant_3d([xl, yl, 0.0, 0.0, 0.0, 0.0], mu)
    return c


REGIME_LABELS = {"I": "C > C_L1", "II": "C_L2 < C < C_L1", "III": "C_L3 < C < C_L2", "IV": "C_L4 < C < C_L3",
                 "V": "C < C_L4"}


def hill_regime(C, mu):
    """Regime I (C > C_L1: necks closed) .. V (C < C_L4: no forbidden region). C may be an array."""
    cl = jacobi_at_points(mu)
    C = np.asarray(C, dtype=float)
    out = np.full(C.shape, "V", dtype=object)
    out[C > cl["L4"]] = "IV"
    out[C > cl["L3"]] = "III"
    out[C > cl["L2"]] = "II"
    out[C > cl["L1"]] = "I"
    return out
