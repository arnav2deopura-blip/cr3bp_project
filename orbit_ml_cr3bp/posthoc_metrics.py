"""
Post hoc summaries of the sealed ledgers (posthoc/README.md, P1-P3). Pure functions on arrays / DataFrames; nothing here reads files
(run_posthoc.py does), so every statistic can be checked against hand-computed toy ledgers (tests/test_posthoc.py).

  cost_stats       share cheaper (the sealed win_share definition), aggregate cost ratio, mean log10 ratio and medians for one policy
                   ledger against one comparator column, each with an orbit bootstrap interval; optional per-row weights (mixtures)
  mixture_weights  per-row weights that move a ledger's family mix to a declared target mix
  breakdown        cost_stats per value of a grouping column
Conventions: costs are RHS evaluations, the policy total includes its overhead; ratios are comparator / policy, so values above 1 (or
log ratios above 0) favour the policy. Orbits without a comparator cost (failed or capped comparator runs) are left out of every
statistic of that comparator, exactly as in the sealed share, and counted in n_no_comp.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .audit_3d import clopper_pearson
from .confirm_b_3d import N_BOOT

__all__ = ["MIXTURES", "FAMILIES", "cost_stats", "mixture_weights", "breakdown", "stats_frame"]

FAMILIES = ("L1", "L2", "L3", "L4", "L5", "wide", "near_moon")
MIXTURES = {
    "as_sampled": None,
    "uniform_families": {f: 1.0 / 7.0 for f in FAMILIES},
    "cislunar": {"near_moon": 1.0 / 3.0, "L1": 1.0 / 3.0, "L2": 1.0 / 3.0},
}
_CHUNK = 250


def _boot_indices(n, n_boot, seed, strata=None):
    """Yields (chunk, n) index arrays: plain resampling of the n rows, or resampling within each stratum (stratum sizes kept fixed)."""
    rng = np.random.RandomState(seed)
    if strata is None:
        for b0 in range(0, n_boot, _CHUNK):
            yield rng.randint(0, n, (min(_CHUNK, n_boot - b0), n))
        return
    groups = [np.flatnonzero(strata == g) for g in pd.unique(strata)]
    for b0 in range(0, n_boot, _CHUNK):
        m = min(_CHUNK, n_boot - b0)
        yield np.hstack([g[rng.randint(0, len(g), (m, len(g)))] for g in groups])


def _ci(x):
    return [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))]


def cost_stats(policy_cost, comp_cost, met, weights=None, strata=None, n_boot=N_BOOT, seed=0, capped=None):
    """Statistics of one policy ledger against one comparator.
    policy_cost: policy total cost per orbit (overhead included); comp_cost: comparator RHS evaluations (NaN = no comparator cost);
    met: the policy run met the tolerance; weights: optional per-row weights (None = 1); strata: optional labels to resample within
    (use the family when weights reweight families); capped: optional bool array of comparator runs that hit the evaluation cap."""
    p = np.asarray(policy_cost, dtype=float)
    c = np.asarray(comp_cost, dtype=float)
    met = np.asarray(met, dtype=bool)
    n_all = len(p)
    if np.any(~np.isfinite(p)) or np.any(p <= 0):
        raise ValueError("policy costs must be finite and positive")
    has = np.isfinite(c)
    if np.any(c[has] <= 0):
        raise ValueError("comparator costs must be positive")
    w_all = np.ones(n_all) if weights is None else np.asarray(weights, dtype=float)
    keep = has & (w_all > 0)
    p, c, met, w = p[keep], c[keep], met[keep], w_all[keep]
    st = None if strata is None else np.asarray(strata)[keep]
    n = len(p)
    out = {"n": int(n_all), "n_with_comp": int(has.sum()), "n_no_comp": int((~has).sum()),
           "n_capped": None if capped is None else int(np.asarray(capped, dtype=bool).sum()), "weighted": weights is not None}
    if n == 0:
        return out
    win = (met & (p < c)).astype(float)
    lr = np.log10(c / p)
    sw = w.sum()
    out.update({
        "share": float((w * win).sum() / sw),
        "k": int(win.sum()),
        "aggregate_ratio": float((w * c).sum() / (w * p).sum()),
        "mean_log_ratio": float((w * lr).sum() / sw),
        "policy_mean": float((w * p).sum() / sw),
        "comp_mean": float((w * c).sum() / sw),
        "median_policy_all": float(np.median(p)) if weights is None else None,
        "median_policy_met": float(np.median(p[met])) if weights is None and met.any() else None,
        "median_comp": float(np.median(c)) if weights is None else None,
        "violation_rate": float((w * ~met).sum() / sw),
    })
    if weights is None:
        out["share_cp95"] = list(clopper_pearson(int(win.sum()), n))
    sh, ag, ml = [], [], []
    for idx in _boot_indices(n, n_boot, seed, st):
        ww = w[idx]
        s = ww.sum(axis=1)
        sh.append((ww * win[idx]).sum(axis=1) / s)
        ag.append((ww * c[idx]).sum(axis=1) / (ww * p[idx]).sum(axis=1))
        ml.append((ww * lr[idx]).sum(axis=1) / s)
    out["share_ci95"] = _ci(np.concatenate(sh))
    out["aggregate_ratio_ci95"] = _ci(np.concatenate(ag))
    out["mean_log_ratio_ci95"] = _ci(np.concatenate(ml))
    return out


def mixture_weights(family, target):
    """Per-row weights moving the observed family mix to `target` ({family: share}, families not named get 0). Within a family every row
    gets the same weight, target share / observed share; a family in the target that is absent from the rows raises."""
    family = np.asarray(family, dtype=object)
    if target is None:
        return np.ones(len(family))
    tot = sum(target.values())
    w = np.zeros(len(family))
    for f, t in target.items():
        m = family == f
        if t > 0 and not m.any():
            raise ValueError(f"family {f} has no rows")
        if m.any():
            w[m] = (t / tot) / (m.sum() / len(family))
    return w


def stats_frame(led, comp, policy_col="C_ml_tot", **kw):
    """cost_stats on a ledger DataFrame for comparator column prefix `comp` (e.g. 'RK45_eqr')."""
    cap = led[f"{comp}_capped"].fillna(False).astype(bool).to_numpy() if f"{comp}_capped" in led.columns else None
    return cost_stats(led[policy_col].to_numpy(dtype=float), led[f"{comp}_nfev"].to_numpy(dtype=float),
                      led["meets_tol"].astype(bool).to_numpy(), capped=cap, **kw)


def breakdown(led, comp, by, policy_col="C_ml_tot", **kw):
    """One row of cost_stats per value of column `by` (plus the column value and the group's share of the policy's total cost)."""
    rows = []
    tot = led[policy_col].sum()
    for g, d in led.groupby(by, sort=True):
        r = stats_frame(d, comp, policy_col, **kw)
        r[by] = g
        r["share_of_policy_cost"] = float(d[policy_col].sum() / tot)
        rows.append(r)
    return rows


def recovered_share(led_ml, led_rule, led_glob, n_boot=N_BOOT, seed=0):
    """Share of the global-step gap that a rule recovers, both gaps on the same orbits' ledgers: 1 - gap(rule) / gap(global), each gap the
    D1 / C5 statistic (mean log10 cost of the arm minus that of the ML policy, over orbits that met the tolerance under both). The bootstrap
    resamples orbits once for both gaps (rows are matched by row_id; every ledger must hold the same orbits)."""
    cols = ["row_id", "meets_tol", "C_ml_tot"]
    m = led_ml[cols].merge(led_rule[cols], on="row_id", suffixes=("", "_r")).merge(led_glob[cols].rename(columns={"meets_tol": "meets_tol_g", "C_ml_tot": "C_ml_tot_g"}), on="row_id")
    if not (len(m) == len(led_ml) == len(led_rule) == len(led_glob)):
        raise ValueError("the three ledgers must hold the same orbits")
    ok = m["meets_tol"].astype(bool).to_numpy()
    okr, okg = ok & m["meets_tol_r"].astype(bool).to_numpy(), ok & m["meets_tol_g"].astype(bool).to_numpy()
    lm = np.log10(m["C_ml_tot"].to_numpy(dtype=float))
    dr = np.log10(m["C_ml_tot_r"].to_numpy(dtype=float)) - lm
    dg = np.log10(m["C_ml_tot_g"].to_numpy(dtype=float)) - lm

    def stat(idx):
        gr = (dr[idx] * okr[idx]).sum(axis=-1) / okr[idx].sum(axis=-1)
        gg = (dg[idx] * okg[idx]).sum(axis=-1) / okg[idx].sum(axis=-1)
        return gr, gg, 1.0 - gr / gg
    gr, gg, rec = stat(np.arange(len(m)))
    boots = np.concatenate([stat(idx)[2] for idx in _boot_indices(len(m), n_boot, seed)])
    return {"gap_rule": float(gr), "gap_global": float(gg), "recovered_share": float(rec), "recovered_share_ci95": _ci(boots), "n_orbits": int(len(m))}
