"""
Symmetric certification of the adaptive comparators (posthoc/README.md P7-P8). The sealed studies certified the policy's scale by fixed sequence
at the full delta on a grid 0.038 dex apart (policy_c_3d.FIXED_SEQ_GRID) but the comparators' rtol by Bonferroni at delta / 18 on a grid 0.53 dex
apart (fair_baseline_3d.RTOL_GRID_B). This module gives the comparators the policy's rule on a grid of at least the same cost resolution.

  RTOL_GRID_FINE              10^(-10 + k/8), k = 0..56: 1e-10 to 1e-3, 0.125 dex apart
  cost_resolution_dex         the cost step of a grid for a method whose cost scales as tol^(-1/order)
  certify_rtol_fixed_sequence policy_c_3d.certify_sequence applied to an rtol grid (tightest, i.e. safest, first; full delta)
  chosen_at                   the fixed-sequence choice at a smaller delta from an existing chain (a prefix of it)
  equal_risk_rtol             the sealed equal-risk rule of policy_b_3d.certify_rtol_lazy (grid scan + log bisection) for any violation function
  scipy_rows                  SciPy RK45 / DOP853 on rows at one rtol with any atol ratio (scalar, or one value per state component)
The statistics are the sealed ones (policy_c_3d.certify_sequence and the binomial p-value); only the grid and the comparator's rule change.
"""
from __future__ import annotations

import math
from multiprocessing import Pool

import numpy as np
from scipy.stats import binom

from .config import MU, POSITION_TOL_KM, ADAPTIVE_ATOL_RATIO, RTOL_GRID_B
from .fair_baseline_3d import _run_adaptive_ex
from . import policy_c_3d as pc

__all__ = ["RTOL_GRID_FINE", "ATOL_VARIANTS", "cost_resolution_dex", "certify_rtol_fixed_sequence", "chosen_at", "equal_risk_rtol", "scipy_rows"]

RTOL_GRID_FINE = tuple(float(10.0 ** (-10.0 + k / 8.0)) for k in range(57))
# P8: atol = ratio * rtol; "pos100" is 0.26 rtol on positions (2.6e-4 DU = 100 km at rtol 1e-3) and 1e3 rtol on velocities
ATOL_VARIANTS = {"r1e-1": 1e-1, "r1e-3": 1e-3, "r1e-6": 1e-6, "pos100": (0.26, 0.26, 0.26, 1e3, 1e3, 1e3)}
METHOD_ORDER = {"RK45": 5.0, "DOP853": 8.0, "DP54_scipy": 5.0, "DP54_PI": 5.0}


def cost_resolution_dex(grid, order):
    """Largest cost step (dex) between neighbours of a tolerance grid for a method of the given order (cost ~ tol^(-1/order))."""
    g = np.sort(np.log10(np.asarray(grid, dtype=float)))
    return float(np.max(np.diff(g)) / order)


def certify_rtol_fixed_sequence(violations_at, grid=RTOL_GRID_FINE, alpha=pc.ALPHA, delta=pc.DELTA):
    """Fixed-sequence certificate of an rtol: values tested from the tightest (safest) upwards at the full delta, stopping at the first
    non-rejection; the certified rtol is the last rejected value (None if the tightest value is not rejected). violations_at(rtol) -> (k, n)."""
    rep = pc.certify_sequence(violations_at, grid, alpha, delta, "fixed_sequence")
    chain = [{"rtol": c["scale"], **{k: v for k, v in c.items() if k != "scale"}} for c in rep["chain"]]
    return {"rule": "fixed_sequence", "chosen_rtol": rep["chosen_scale"], "alpha": alpha, "delta": delta, "chain": chain, "grid_size": len(grid),
            "grid_min": float(min(grid)), "grid_max": float(max(grid))}


def chosen_at(chain, delta, key="rtol"):
    """The fixed-sequence choice at `delta` from a chain run in ascending order at a delta >= this one. Valid only if the chain ran up to
    a value that fails at this delta, or to the end of the grid (both hold when the chain was run at a larger delta)."""
    chosen = None
    for c in sorted(chain, key=lambda c: c[key]):
        if c["p_value"] > delta:
            return chosen
        chosen = float(c[key])
    return chosen


def equal_risk_rtol(violations_at, target, grid=RTOL_GRID_B, refine_steps=4):
    """The sealed equal-risk rtol (policy_b_3d.certify_rtol_lazy): the loosest grid value whose violation rate is <= target, scanning from the
    loosest, then refine_steps log10 bisections towards the next looser grid value, keeping the loosest rtol whose rate is <= target.
    violations_at(rtol) -> (k, n). Returns (rtol or None, chain of {rtol, n, violations, violation_rate})."""
    chain, eqr = [], None
    for rt in sorted(grid, reverse=True):
        k, n = violations_at(float(rt))
        chain.append({"rtol": float(rt), "n": n, "violations": k, "violation_rate": k / n})
        if k / n <= target:
            eqr = float(rt)
            break
    if eqr is not None and refine_steps > 0:
        looser = [c for c in chain if c["rtol"] > eqr]
        if looser and looser[-1]["violation_rate"] > target:
            lo, hi = math.log10(eqr), math.log10(min(c["rtol"] for c in looser))
            for _ in range(refine_steps):
                mid = 0.5 * (lo + hi)
                k, n = violations_at(10.0 ** mid)
                chain.append({"rtol": 10.0 ** mid, "n": n, "violations": k, "violation_rate": k / n, "refine": True})
                if k / n <= target:
                    lo, eqr = mid, 10.0 ** mid
                else:
                    hi = mid
    return eqr, chain


def _job(a):
    state, T, ref_pos, rtol, mu, method, atol_ratio = a
    ratio = np.asarray(atol_ratio, dtype=float) if np.ndim(atol_ratio) else float(atol_ratio)
    res, capped = _run_adaptive_ex(np.asarray(state, dtype=float), float(T), ref_pos, float(rtol), mu, method, ratio)
    if res is None:
        return False, float("nan"), float("nan"), bool(capped)
    return True, float(res["err_km"]), float(res["nfev"]), False


def scipy_rows(pool, states, T, ref_pos, rtol, method, atol_ratio=ADAPTIVE_ATOL_RATIO, mu=MU, n_jobs=8):
    """SciPy `method` at rtol on every row. Returns dict(ok, err (km), nfev (NaN where the run failed or hit the cap), capped, met)."""
    jobs = [(states[i], float(T[i]), ref_pos[i], float(rtol), mu, method, atol_ratio) for i in range(len(states))]
    res = pool.map(_job, jobs, chunksize=max(1, len(jobs) // (8 * n_jobs))) if pool is not None else [_job(j) for j in jobs]
    ok = np.array([r[0] for r in res], dtype=bool)
    err = np.array([r[1] for r in res], dtype=float)
    nfev = np.array([r[2] for r in res], dtype=float)
    capped = np.array([r[3] for r in res], dtype=bool)
    return {"ok": ok, "err": err, "nfev": nfev, "capped": capped, "met": ok & (err <= POSITION_TOL_KM)}


def binom_p(k, n, alpha=pc.ALPHA):
    return float(binom.cdf(k, n, alpha))
