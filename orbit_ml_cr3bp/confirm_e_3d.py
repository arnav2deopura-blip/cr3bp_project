"""
Test statistics of Study E (part_e/PREREGISTRATION_E.md section 4). Pure functions of the ledgers written by run_confirm_e_3d.py; nothing here
runs an integrator.

  ratio_test        E1a / E2 / E7a: ratio of totals Σ comparator / Σ policy over the same rows (violating runs included on both sides, at the
                    evaluations they spent; policy overhead included), orbit bootstrap (4,000, seed 0), one-sided p = share of resamples <= 1
  iut               intersection-union: p = the largest
  binom_le          E3 and the E5 abstention criterion: exact one-sided binomial, H0 rate > level, p = P[X <= k | n, level]
  noninferiority    E4: success difference policy - comparator > -margin (paired orbit bootstrap, p = share of resamples <= -margin)
  holm              Holm-Bonferroni over the family (E1-E5, E7), familywise 0.05
"""
from __future__ import annotations

import numpy as np
from scipy.stats import binom

__all__ = ["N_BOOT", "ratio_test", "iut", "binom_le", "noninferiority", "holm", "band_table"]

N_BOOT = 4000
# The policy-scale grid of Part E: the sealed FIXED_SEQ_GRID (0.1 * 2^(k/8), k = 0..26, and 1.0) continued upward in the same 1/8-octave steps
# to 9.8 (k = 53). Added 2026-11-13 after the E_select dry run, before the seal: at 1,000 km the policy reached the top of the sealed grid (1.0)
# with 3 / 2,900 calibration violations, i.e. it was capped while the comparators could loosen freely. Under the fixed-sequence rule the extra
# values cost no power, and where the scale is not capped (every 100 km step of the dry run stopped at 0.872) nothing changes.
SCALE_GRID_E = tuple(sorted(set([float(0.1 * 2.0 ** (k / 8.0)) for k in range(54)] + [1.0])))


def _boot_idx(n, n_boot, seed):
    return np.random.RandomState(seed).randint(0, n, size=(n_boot, n))


def ratio_test(policy_cost, comp_cost, n_boot=N_BOOT, seed=0):
    p = np.asarray(policy_cost, dtype=float)
    c = np.asarray(comp_cost, dtype=float)
    m = np.isfinite(p) & np.isfinite(c)
    if np.any(p[m] <= 0) or np.any(c[m] < 0):
        raise ValueError("costs must be positive")
    p, c = p[m], c[m]
    n = len(p)
    ratio = float(c.sum() / p.sum())
    idx = _boot_idx(n, n_boot, seed)
    bs = c[idx].sum(axis=1) / p[idx].sum(axis=1)
    return {"ratio": ratio, "p": float((bs <= 1.0).mean()), "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
            "n": int(n), "n_dropped_nonfinite": int((~m).sum()), "share_policy_cheaper": float((p < c).mean())}


def iut(results):
    """results: {name: dict with "p"}. Returns the largest p and which part gave it."""
    name = max(results, key=lambda k: results[k]["p"])
    return {"p": float(results[name]["p"]), "binding": name, "parts": {k: float(v["p"]) for k, v in results.items()}}


def binom_le(k, n, level):
    k, n = int(k), int(n)
    return {"k": k, "n": n, "rate": k / n if n else float("nan"), "level": float(level), "p": float(binom.cdf(k, n, level)) if n else 1.0}


def noninferiority(success_policy, success_comp, margin=0.01, n_boot=N_BOOT, seed=0):
    a = np.asarray(success_policy, dtype=float)
    b = np.asarray(success_comp, dtype=float)
    d = a - b
    n = len(d)
    idx = _boot_idx(n, n_boot, seed)
    bs = d[idx].mean(axis=1)
    return {"diff": float(d.mean()), "p": float((bs <= -margin).mean()), "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
            "margin": float(margin), "n": int(n), "success_policy": float(a.mean()), "success_comparator": float(b.mean())}


def holm(pvals, alpha=0.05):
    """pvals: {test: p}. Returns {test: {p, p_adj, reject}} (step-down, monotone adjusted p)."""
    names = sorted(pvals, key=lambda k: pvals[k])
    m = len(names)
    out, run = {}, 0.0
    for i, k in enumerate(names):
        adj = min(1.0, (m - i) * pvals[k])
        run = max(run, adj)
        out[k] = {"p": float(pvals[k]), "p_adj": float(run)}
    stop = False
    for k in names:
        if not stop and out[k]["p_adj"] <= alpha:
            out[k]["reject"] = True
        else:
            stop = True
            out[k]["reject"] = False
    return out


def band_table(bands, policy_cost, comp_cost, policy_viol, comp_viol, n_boot=1000):
    """Per horizon band: ratio of totals with interval and both realized violation rates (the per-band reporting of decision D6)."""
    bands = np.asarray(bands)
    rows = {}
    for b in sorted(set(bands)):
        s = (bands == b) & np.isfinite(np.asarray(comp_cost, dtype=float))
        if not s.any():
            rows[str(b)] = {"n": 0, "note": "no per-band comparator (fewer than 100 calibration rows in the band, or no equal-risk rtol)"}
            continue
        r = ratio_test(np.asarray(policy_cost)[s], np.asarray(comp_cost)[s], n_boot=n_boot)
        r.update({"policy_violation_rate": float(np.mean(np.asarray(policy_viol)[s])), "comparator_violation_rate": float(np.mean(np.asarray(comp_viol)[s]))})
        rows[str(b)] = r
    return rows
