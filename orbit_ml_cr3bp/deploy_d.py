"""
Lean deployment of Sundman-time step policies for Part D. A policy here is anything that yields a raw log10 ds per orbit (the frozen `sund` model, a physics rule,
a constant); this module certifies ONE scale (or one per stratum) with Part C's exact fixed-sequence rule, runs the Sundman RK4 through the compiled batch
kernel and scores it against the adaptive comparators. The arithmetic is Part C's: the same integrator kernel (sundman_3d), the same grid and level
(policy_c_3d.FIXED_SEQ_GRID, alpha = 0.05, delta = 0.1), the same violation definition, the same adaptive comparators (SciPy RK45 / DOP853, atol = 1e-3 rtol);
tests/test_part_d.py checks that it reproduces a Part C ledger orbit for orbit.

  reference_positions   DOP853 (rtol 1e-13) final positions, cached on disk keyed by a hash of the states and horizons
  certify_scale         exact fixed-sequence certificate over the scale grid on given rows
  run_rows / score      one run of the policy at per-row ds, and the ledger columns (ml_ok, ml_err_km, ml_evals, meets_tol, C_ml_int, C_ml_tot)
  adaptive_rows         SciPy RK45 / DOP853 at a given rtol on rows (the Part C comparator job)
  comparator_frame      certified-rtol and equal-risk-rtol comparators on evaluation rows, columns named like Part C's ledger
"""
from __future__ import annotations

import hashlib
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from .config import MU, POSITION_TOL_KM
from .compiled_solvers_d import SundmanRunner
from . import policy_b_3d as pb
from . import policy_c_3d as pc

STATE_COLS = pb.STATE_COLS
ALPHA, DELTA = pb.ALPHA, pb.DELTA


def states_T(df):
    return df[STATE_COLS].to_numpy(dtype=np.float64), df["sim_time"].to_numpy(dtype=np.float64)


def _key(states, T):
    return hashlib.sha256(np.ascontiguousarray(np.hstack([states, T[:, None]])).tobytes()).hexdigest()


def reference_positions(df, mu=MU, cache=None, n_jobs=8):
    """(n, 3) reference positions at the horizon. With `cache` (a .npz path) the result is stored and reused if the rows are identical."""
    s, T = states_T(df)
    key = _key(s, T)
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        if str(z["key"]) == key:
            return z["ref"]
    ref = pc.reference_final_positions(df, mu, n_jobs)
    if cache:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        np.savez(cache, ref=ref, key=np.array(key))
    return ref


def run_rows(runner, states, T, ds, ref_pos, tol_km=POSITION_TOL_KM, parallel=True):
    """One Sundman-time run per row at fictitious step ds[i]. Returns dict(ok, err (km, NaN where the run failed), evals, met)."""
    ok, y, ev, rs = runner.run_ds(states, T, ds, parallel)
    err = np.where(ok, runner.errors_km(y, ref_pos), np.nan)
    return {"ok": ok, "err": err, "evals": ev, "met": ok & (err <= tol_km), "reason": rs}


def certify_scale(runner, states, T, raw_ds, ref_pos, tol_km=POSITION_TOL_KM, alpha=ALPHA, delta=DELTA, grid=pc.FIXED_SEQ_GRID, parallel=True):
    """The exact fixed-sequence certificate of policy_c_3d.certify_sequence (values tested safest-first at the full delta, stopping at the first
    non-rejection; the certified scale is the last rejected value) for the policy `raw_ds` on these rows. Returns its report dict plus the cache of
    violation counts per scale."""
    cache = {}

    def violations_at(s):
        if s not in cache:
            r = run_rows(runner, states, T, raw_ds * s, ref_pos, tol_km, parallel)
            cache[s] = int((~r["met"]).sum())
        return cache[s], len(states)

    rep = pc.certify_sequence(violations_at, grid, alpha, delta, "fixed_sequence")
    rep["violations_by_scale"] = {float(k): int(v) for k, v in cache.items()}
    return rep


def score(df, run, overhead):
    """Ledger columns of a policy run on the rows of df."""
    out = pd.DataFrame({"row_id": df["row_id"].to_numpy()})
    out["ml_ok"] = run["ok"]
    out["ml_err_km"] = run["err"]
    out["ml_evals"] = run["evals"]
    out["meets_tol"] = run["met"]
    out["violation"] = ~run["met"]
    out["C_ml_int"] = run["evals"]
    out["C_ml_tot"] = run["evals"] + float(overhead)
    return out


def adaptive_rows(states, T, ref_pos, rtol, method, mu=MU, n_jobs=8):
    """SciPy `method` at rtol on rows: (ok, err_km, nfev); nfev is NaN where the run failed or hit the evaluation cap (as in Part C's ledger)."""
    jobs = [(states[i], float(T[i]), ref_pos[i], mu, (float(rtol), method)) for i in range(len(states))]
    with Pool(n_jobs) as pool:
        res = pool.map(pc._adaptive_only_job, jobs, chunksize=max(1, len(jobs) // (8 * n_jobs)))
    ok = np.array([r[0] for r in res], dtype=bool)
    err = np.array([r[1] for r in res], dtype=float)
    nfev = np.array([r[2] for r in res], dtype=float)
    return ok, err, np.where(ok, nfev, np.nan)


def comparator_rtols(cal_df, ref_cal, target, methods=pb.METHODS, tol_km=POSITION_TOL_KM, mu=MU, n_jobs=8):
    """{(method, 'cal'|'eqr'): rtol} and the certification reports: the certified rtol of each method on the calibration rows (Bonferroni over the
    18-value grid) and the loosest rtol whose realized violation rate there is <= target (Part C's equal-risk comparator)."""
    s, T = states_T(cal_df)
    items = list(zip(s, T, ref_cal))
    rt, rep = {}, {}
    for m in methods:
        r = pb.certify_rtol_lazy(cal_df, m, tol_km=tol_km, mu=mu, equal_risk_target=target, n_jobs=n_jobs, items=items)
        rep[m] = r
        rt[(m, "cal")] = r["chosen_rtol"]
        rt[(m, "eqr")] = r["equal_risk_rtol"]
    return rt, rep


def comparator_frame(ev_df, ref_ev, rtols, methods=pb.METHODS, tags=("cal", "eqr"), tol_km=POSITION_TOL_KM, mu=MU, n_jobs=8):
    """Part C ledger columns <method>_<tag>_{rtol,nfev,err_km,ok} on the evaluation rows (NaN where no rtol was certified)."""
    s, T = states_T(ev_df)
    out = pd.DataFrame({"row_id": ev_df["row_id"].to_numpy()})
    for m in methods:
        for tag in tags:
            rt = rtols.get((m, tag))
            if rt is None:
                out[f"{m}_{tag}_rtol"], out[f"{m}_{tag}_nfev"], out[f"{m}_{tag}_err_km"], out[f"{m}_{tag}_ok"] = np.nan, np.nan, np.nan, False
                continue
            ok, err, nfev = adaptive_rows(s, T, ref_ev, rt, m, mu, n_jobs)
            out[f"{m}_{tag}_rtol"] = float(rt)
            out[f"{m}_{tag}_nfev"] = nfev
            out[f"{m}_{tag}_err_km"] = err
            out[f"{m}_{tag}_ok"] = ok & (err <= tol_km)
    return out


def default_runner(mu=MU):
    return SundmanRunner(mu)
