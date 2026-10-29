"""
The frozen Part C policy on structured, out-of-distribution orbits (Part D, D5): Earth-Moon periodic-orbit families from the JPL Three-Body Periodic
Orbits API (fetch_jpl_d.py saves the raw responses). The policy was trained on random states; halos, NRHOs, DROs, Lyapunov, vertical, axial, butterfly
and resonant orbits are what a mission designer actually propagates.

Units. JPL's normalization (mass ratio 1.215058560962404e-2, length unit 389,703.26 km) differs from the project's (mu = 1.2150585 35e-2, DU = 384,400 km). The
dynamics are the dimensionless CR3BP, so the states are used as given and integrated at JPL's mu (the orbits are periodic only for it; the policy takes mu as an
argument); the position tolerance is kept at the PROJECT's 100 km = 100 / 384,400 DU so that it is the tolerance the policy was calibrated for (in JPL's own
kilometres that is 101.4 km).

  load_stream / select_indices     one (family, libration point, branch) stream and an evenly spaced, reproducible sample of its members (two disjoint samples:
                                   `dry` for the expectations written before the seal, `confirm` for the test)
  member_diagnostics               one period at 1e-13: closure error, closest approach to each body, maximum radius -> member quality flags
  build_runs / reference_multi     horizons k x period (k = 1, 2, 5, 10, 20, at most T_CAP), one DOP853 reference per member (all horizons from one dense solution) and
                                   a second, differently stepped reference that decides whether the reference is trustworthy at each horizon
  run_policy / run_comparators     the frozen policy at the frozen scale, and SciPy RK45 / DOP853 at the Part C rtols, on the same runs
"""
from __future__ import annotations

import json
import math
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from .config import DU_KM, MU, POSITION_TOL_KM, collision_radii
from . import deploy_d as dd
from .physics_3d import _cr3bp_derivatives_3d_numba, reference_trajectory_dense_3d, jacobi_constant_3d

KS = (1, 2, 5, 10, 20)
T_CAP = 60.0                  # TU: three times the training horizon range, for the extrapolation rows
T_RANGE = (1.0, 20.0)         # TU: the training horizon range (the in-range rows are the test)
N_PER_STREAM = 50
CLOSURE_TOL = 1e-8            # DU: a member whose one-period closure error is larger is not periodic enough to be a test orbit
REF_AGREE_FRAC = 0.05         # reference trustworthy at a horizon if two differently stepped DOP853 references agree within 5 % of the tolerance
STAB_BANDS = (("stable (<=1.0001)", 0.0, 1.0001), ("1-10", 1.0001, 10.0), ("10-100", 10.0, 100.0), (">100", 100.0, np.inf))
GROUPS = {"halo_L1": ("halo_L1_N", "halo_L1_S"), "halo_L2": ("halo_L2_N", "halo_L2_S"), "lyapunov_L1": ("lyapunov_L1",), "lyapunov_L2": ("lyapunov_L2",),
          "vertical_L1": ("vertical_L1",), "vertical_L2": ("vertical_L2",), "axial_L1": ("axial_L1",), "axial_L2": ("axial_L2",), "dro": ("dro",),
          "butterfly": ("butterfly_N", "butterfly_S"),
          "resonant": ("resonant_12", "resonant_21", "resonant_13", "resonant_31", "resonant_23", "resonant_32", "resonant_34", "resonant_43")}
STREAM_TO_GROUP = {s: g for g, ss in GROUPS.items() for s in ss}


def load_stream(path):
    j = json.load(open(path))
    if j.get("signature", {}).get("version") != "1.0":
        raise ValueError(f"{path}: unexpected API version {j.get('signature')}")
    if "data" not in j:
        return None, float(j["system"]["mass_ratio"]) if "system" in j else None
    df = pd.DataFrame(np.array(j["data"], dtype=float), columns=j["fields"])
    return df, float(j["system"]["mass_ratio"])


def select_indices(n_total, n, which):
    """Evenly spaced member indices along the family. 'confirm': round(j * step); 'dry': round((j + 0.5) * step); the dry sample drops any index the
    confirm sample uses."""
    step = (n_total - 1) / n
    conf = sorted({int(round(j * step)) for j in range(n)} | {n_total - 1})
    if which == "confirm":
        return conf
    dry = sorted({int(round((j + 0.5) * step)) for j in range(n)} - set(conf))
    return dry


def member_diagnostics(job):
    """job = (state, period, mu). Integrates one period (DOP853, 1e-13, max step 0.01) and returns closure error (max-norm over the 6 components), closest
    approach to each body and the largest |r|; the integration is stopped by a body collision / escape event like the reference solver."""
    state, period, mu = job
    sol, reached = reference_trajectory_dense_3d(list(state), period, mu)
    ts = np.linspace(0.0, sol.t[-1], 4001)
    Y = sol.sol(ts)
    r1 = np.sqrt((Y[0] + mu) ** 2 + Y[1] ** 2 + Y[2] ** 2)
    r2 = np.sqrt((Y[0] + mu - 1.0) ** 2 + Y[1] ** 2 + Y[2] ** 2)
    closure = float(np.max(np.abs(sol.y[:, -1] - np.asarray(state)))) if reached else float("nan")
    return {"reached": bool(reached), "closure": closure, "min_r1": float(r1.min()), "min_r2": float(r2.min()),
            "max_r": float(np.sqrt(Y[0] ** 2 + Y[1] ** 2 + Y[2] ** 2).max())}


def stab_band(nu):
    for name, lo, hi in STAB_BANDS:
        if lo < nu <= hi or (lo == 0.0 and nu <= hi):
            return name
    return STAB_BANDS[-1][0]


def reference_multi(job):
    """job = (state, Ts (sorted), mu). Positions at every T from one DOP853 reference (rtol 1e-13, max step 0.01) and from a differently stepped second
    reference (rtol 1e-12, max step 0.004); NaN beyond the time either one stops (collision / escape / failure)."""
    state, Ts, mu = job
    Ts = np.asarray(Ts, dtype=float)
    out = []
    for kw in ({}, {"rtol": 1e-12, "atol": 1e-12, "max_step": 0.004}):
        sol, _ = reference_trajectory_dense_3d(list(state), float(Ts.max()), mu, **kw)
        pos = np.full((len(Ts), 3), np.nan)
        for i, t in enumerate(Ts):
            if t <= sol.t[-1] + 1e-9:
                pos[i] = sol.sol(min(t, sol.t[-1]))[:3]
        out.append(pos)
    return out[0], out[1]


def build_runs(members, mu, ks=KS, t_cap=T_CAP):
    """One row per (member, horizon). members: DataFrame with x..vz, period, jacobi, stability, stream, group, idx. Horizons below 1 TU or above t_cap are dropped."""
    rows = []
    for _, m in members.iterrows():
        for k in ks:
            T = k * float(m["period"])
            if T < 1.0 - 1e-9 or T > t_cap:
                continue
            rows.append({**{c: m[c] for c in members.columns}, "k": k, "sim_time": T, "in_range": bool(T_RANGE[0] <= T <= T_RANGE[1]),
                         "stab_band": stab_band(float(m["stability"]))})
    return pd.DataFrame(rows)


def attach_references(runs, mu, n_jobs=8):
    """Reference positions (ref1), the second reference (ref2) and the trust flag per run. One dense solution per member serves all its horizons."""
    members = {}
    for i, r in runs.iterrows():
        members.setdefault((r["stream"], int(r["idx"])), []).append(i)
    keys = list(members)
    jobs = [(runs.loc[members[k][0], ["x", "y", "z", "vx", "vy", "vz"]].to_numpy(dtype=float), sorted(runs.loc[members[k], "sim_time"]), mu) for k in keys]
    with Pool(n_jobs) as pool:
        res = pool.map(reference_multi, jobs, chunksize=1)
    ref1 = np.full((len(runs), 3), np.nan)
    ref2 = np.full((len(runs), 3), np.nan)
    for k, (p1, p2), job in zip(keys, res, jobs):
        Ts = job[1]
        for i in members[k]:
            j = int(np.argmin(np.abs(np.array(Ts) - runs.loc[i, "sim_time"])))
            ref1[runs.index.get_loc(i)], ref2[runs.index.get_loc(i)] = p1[j], p2[j]
    tol_du = POSITION_TOL_KM / DU_KM
    agree = np.linalg.norm(ref1 - ref2, axis=1)
    out = runs.copy()
    out["ref_x"], out["ref_y"], out["ref_z"] = ref1[:, 0], ref1[:, 1], ref1[:, 2]
    out["ref_disagree_km"] = agree * DU_KM
    out["ref_ok"] = np.isfinite(agree) & (agree <= REF_AGREE_FRAC * tol_du)
    out["ref_reached"] = np.isfinite(ref1).all(axis=1)
    return out


def run_policy(runs, policy, scale, mu, parallel=True):
    """The frozen Sundman policy at the frozen scale on every run: (ok, err_km, evals, ds)."""
    s = runs[["x", "y", "z", "vx", "vy", "vz"]].to_numpy(dtype=float)
    T = runs["sim_time"].to_numpy(dtype=float)
    ref = runs[["ref_x", "ref_y", "ref_z"]].to_numpy(dtype=float)
    raw = policy.raw_ds(s, T)
    r = dd.run_rows(policy, s, T, raw * scale, ref, POSITION_TOL_KM, parallel)
    return r, raw


def run_comparators(runs, rtols, mu, n_jobs=8, methods=("RK45", "DOP853"), tags=("cal", "eqr")):
    """Adaptive comparators at frozen rtols ({(method, tag): rtol}) on the runs with a finite reference."""
    ev = runs[runs["ref_reached"]]
    ref = ev[["ref_x", "ref_y", "ref_z"]].to_numpy(dtype=float)
    cols = {}
    for m in methods:
        for tag in tags:
            ok, err, nfev = dd.adaptive_rows(ev[["x", "y", "z", "vx", "vy", "vz"]].to_numpy(dtype=float), ev["sim_time"].to_numpy(dtype=float), ref,
                                              rtols[(m, tag)], m, mu, n_jobs)
            for name, val in ((f"{m}_{tag}_nfev", nfev), (f"{m}_{tag}_err_km", err), (f"{m}_{tag}_ok", ok & (err <= POSITION_TOL_KM))):
                full = pd.Series(np.nan if "ok" not in name else False, index=runs.index, dtype=float if "ok" not in name else bool)
                full.loc[ev.index] = val
                cols[name] = full
    return pd.DataFrame(cols)
