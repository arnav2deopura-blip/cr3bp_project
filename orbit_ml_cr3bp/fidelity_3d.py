"""
How far does the CR3BP drift from real ephemeris states?

Takes spacecraft and Moon states (inertial, km and km/s, e.g. from JPL
Horizons via validate_on_capstone.query_vectors), converts each epoch to the
CR3BP synodic frame (same transform as validate_on_capstone.py), propagates
the CR3BP forward from epoch i with DOP853, and reports the position
difference to the observed state at later epochs, in km and vs lag.

Caveats (these affect the measured number, so state them with it):
  - The synodic frame is rebuilt at every epoch from the Moon's instantaneous
    position/velocity and scaled by its instantaneous distance, so part of the
    lunar orbit's eccentricity is absorbed into the scaling.
  - The CR3BP time unit is taken as tu_days (default: sidereal month / 2*pi, or
    moon_window_tu_days from the Moon's own angular rate). Measured here: a 1%
    change in tu_days alone moves the apparent "model error" by thousands of km
    at 1-2 TU lag, so ALWAYS report the +/-1% sensitivity next to the number.
  - Only Earth+Moon gravity; no Sun, SRP, or Earth/Moon non-sphericity, and
    the real spacecraft may have maneuvers inside the window.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import MU
from .physics_3d import cr3bp_derivatives_3d

SIDEREAL_MONTH_DAYS = 27.321661
TU_DAYS_DEFAULT = SIDEREAL_MONTH_DAYS / (2.0 * math.pi)
# TU_DAYS_DEFAULT is the CR3BP time unit by definition (1 / mean motion of the
# primaries). moon_window_tu_days() instead returns 1 / (the Moon's INSTANTANEOUS
# angular rate averaged over the queried window); because the lunar orbit is
# eccentric (rate varies by roughly +/-10% over a month) that value depends on
# orbital phase (4.72 days for Jan 1-15 2023 vs 4.348 nominal). Use
# TU_DAYS_DEFAULT as the single justified value and report the window-derived
# value only as a diagnostic, together with the +/-1% sensitivity.


def synodic_frame_from_moon_state(r_moon, v_moon):
    r_norm = np.linalg.norm(r_moon)
    x_hat = r_moon / r_norm
    h_vec = np.cross(r_moon, v_moon)
    h_norm = np.linalg.norm(h_vec)
    z_hat = h_vec / h_norm
    y_hat = np.cross(z_hat, x_hat)
    Q = np.vstack([x_hat, y_hat, z_hat])
    omega_z = h_norm / r_norm**2
    return Q, r_norm, omega_z


def inertial_to_cr3bp_synodic(r_sc, v_sc, r_moon, v_moon):
    """Mirror of validate_on_capstone.inertial_to_cr3bp_synodic (spacecraft state relative to the Earth-Moon barycenter)."""
    Q, r_norm, omega_z = synodic_frame_from_moon_state(r_moon, v_moon)
    r_rot = Q @ r_sc
    v_rot = Q @ v_sc - np.cross(np.array([0.0, 0.0, omega_z]), r_rot)
    return np.concatenate([r_rot / r_norm, v_rot / (r_norm * omega_z)])


def moon_window_tu_days(r_moon, v_moon):
    """CR3BP time unit (days) implied by the Moon's own mean angular rate over the queried window (independent of the spacecraft)."""
    omegas = []
    for r, v in zip(r_moon, v_moon):
        _Q, _rn, omega_z = synodic_frame_from_moon_state(np.asarray(r, float), np.asarray(v, float))
        omegas.append(omega_z)
    return 1.0 / (float(np.mean(omegas)) * 86400.0)


def synodic_states_from_ephemeris(r_sc, v_sc, r_moon, v_moon):
    states = np.array([inertial_to_cr3bp_synodic(r_sc[i], v_sc[i], r_moon[i], v_moon[i])
                       for i in range(len(r_sc))])
    scale_km = np.linalg.norm(np.asarray(r_moon), axis=1)
    return states, scale_km


def cr3bp_vs_ephemeris_error(jd, states_syn, scale_km, mu=MU, tu_days=TU_DAYS_DEFAULT,
                             max_lag_days=None, stride=1, rtol=1e-11, atol=1e-11):
    from scipy.integrate import solve_ivp

    jd = np.asarray(jd, dtype=float)
    states_syn = np.asarray(states_syn, dtype=float)
    scale_km = np.asarray(scale_km, dtype=float)

    def rhs(t, y):
        return cr3bp_derivatives_3d(y.tolist(), mu)

    records = []
    for i in range(0, len(jd) - 1, stride):
        lag_days = jd[i + 1:] - jd[i]
        j_idx = np.arange(i + 1, len(jd))
        if max_lag_days is not None:
            keep = lag_days <= max_lag_days
            lag_days, j_idx = lag_days[keep], j_idx[keep]
        if len(lag_days) == 0:
            continue
        t_eval = lag_days / tu_days
        sol = solve_ivp(rhs, (0.0, float(t_eval[-1])), states_syn[i], method="DOP853",
                        rtol=rtol, atol=atol, t_eval=t_eval)
        if not sol.success or sol.y.shape[1] != len(t_eval):
            continue
        err_du = np.linalg.norm(sol.y[:3].T - states_syn[j_idx, :3], axis=1)
        for k, j in enumerate(j_idx):
            records.append({"start_idx": i, "end_idx": int(j), "lag_days": float(lag_days[k]),
                            "lag_tu": float(t_eval[k]), "pos_err_du": float(err_du[k]),
                            "pos_err_km": float(err_du[k] * scale_km[j])})
    return pd.DataFrame(records, columns=["start_idx", "end_idx", "lag_days", "lag_tu", "pos_err_du", "pos_err_km"])


def summarize_fidelity(df, lag_edges_tu=(0.0, 0.25, 0.5, 1.0, 1.61, 3.0, 5.0)):
    if len(df) == 0:
        return pd.DataFrame()
    bins = pd.cut(df["lag_tu"], bins=list(lag_edges_tu) + [np.inf], include_lowest=True)
    g = df.groupby(bins, observed=True)["pos_err_km"]
    return pd.DataFrame({"n": g.size(), "median_km": g.median(), "p90_km": g.quantile(0.9), "max_km": g.max()})