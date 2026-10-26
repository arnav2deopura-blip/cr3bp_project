"""
E3c (exploratory, see the 2026-10-26 entry of PREREGISTRATION_B.md): features from integrating only the first fraction f of the horizon
with a cheap adaptive solve, and the cost of doing so.

For each fraction f: RK45 (rtol 1e-6, atol 1e-9) over [0, f T]; the cost is scipy's `nfev` (capped: a capped row gets NaN features and
the cap as cost). Features from the ACCEPTED steps (their density follows the local timescale, so close approaches are sampled finely):
  la_min_r1, la_min_r2, la_tau_min, la_max_inv_tau, la_delta = (f T) max(1/tau) / integral_0^{fT} (1/tau) dt   (trapezoid over steps)
Partial FTLE is not computed (a variational integration, and it leaks the outcome).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.integrate import solve_ivp

from .config import MU
from .physics_3d import _cr3bp_derivatives_3d_numba

FRACTIONS = (0.05, 0.1, 0.25, 0.5, 1.0)
LA_RTOL = 1e-6
LA_ATOL = 1e-9
LA_CAP = 300_000
STATE_COLS = ["x", "y", "z", "vx", "vy", "vz"]
FEATURE_SUFFIXES = ("min_r1", "min_r2", "tau_min", "max_inv_tau", "delta")


class _Cap(Exception):
    pass


def tag(f):
    return f"{int(round(f * 100)):03d}"


def feature_names(f):
    return [f"la{tag(f)}_{s}" for s in FEATURE_SUFFIXES]


def lookahead_one(state, t_end, mu=MU, rtol=LA_RTOL, atol=LA_ATOL, cap=LA_CAP):
    """Features and nfev of one cheap solve over [0, t_end]."""
    count = [0]

    def rhs(t, y):
        count[0] += 1
        if count[0] > cap:
            raise _Cap()
        return _cr3bp_derivatives_3d_numba(y, mu)

    try:
        sol = solve_ivp(rhs, (0.0, t_end), np.asarray(state, dtype=np.float64), method="RK45", rtol=rtol, atol=atol)
    except _Cap:
        return None, cap
    if not sol.success or not np.all(np.isfinite(sol.y)):
        return None, int(sol.nfev)
    y = sol.y
    r1 = np.sqrt((y[0] + mu) ** 2 + y[1] ** 2 + y[2] ** 2)
    r2 = np.sqrt((y[0] + mu - 1.0) ** 2 + y[1] ** 2 + y[2] ** 2)
    tau = np.minimum(np.sqrt(r1 ** 3 / (1.0 - mu)), np.sqrt(r2 ** 3 / mu))
    inv = 1.0 / tau
    t = sol.t
    integral = float(np.trapezoid(inv, t)) if len(t) > 1 else float("nan")
    delta = float(t_end * inv.max() / integral) if integral and np.isfinite(integral) and integral > 0 else float("nan")
    return {"min_r1": float(r1.min()), "min_r2": float(r2.min()), "tau_min": float(tau.min()),
            "max_inv_tau": float(inv.max()), "delta": delta}, int(sol.nfev)


def _job(a):
    rid, state, T, mu, fractions = a
    out = {"row_id": rid}
    for f in fractions:
        feats, nfev = lookahead_one(state, f * T, mu)
        for s in FEATURE_SUFFIXES:
            out[f"la{tag(f)}_{s}"] = np.nan if feats is None else feats[s]
        out[f"la{tag(f)}_nfev"] = nfev
        out[f"la{tag(f)}_capped"] = feats is None
    return out


def compute_lookahead(df, mu=MU, fractions=FRACTIONS, n_jobs=8, progress=True):
    """df needs row_id, the six state columns and sim_time. Returns one row per row_id."""
    states = df[STATE_COLS].to_numpy(dtype=float)
    T = df["sim_time"].to_numpy(dtype=float)
    ids = df["row_id"].to_numpy()
    jobs = [(int(ids[i]), states[i], float(T[i]), mu, fractions) for i in range(len(df))]
    rows = []
    with Pool(n_jobs) as pool:
        for k, r in enumerate(pool.imap_unordered(_job, jobs, chunksize=16)):
            rows.append(r)
            if progress and (k + 1) % 2000 == 0:
                print(f"  lookahead: {k + 1}/{len(jobs)}", flush=True)
    return pd.DataFrame(rows).sort_values("row_id").reset_index(drop=True)
