"""
Post hoc statistics for the defense-plan analyses (posthoc/README.md P10-P19). Pure functions on arrays; run_posthoc.py reads the ledgers and
calls these. Tested on hand-computed toy inputs in tests/test_defense.py.

  comparator_violated  the comparator-violation rule of P10 (failed, capped or > tol), from a ledger's <comp>_ok / <comp>_err_km columns
  mcnemar_exact        paired exact McNemar test of two violation indicators over the same orbits
  severity             error quantiles of the violating runs (finite errors) and the count of violations without a finite error
  partial_spearman     Spearman correlation of y and x controlling z (Pearson correlation of rank residuals)
  ols                  least-squares coefficients (intercept first)
  ratio_of_totals_ci   percentile and BCa bootstrap intervals for sum(c) / sum(p)
  top_share            share of a total carried by the costliest fraction of rows
  dial_weights         per-row weights that put weight w on one family and keep the others' observed proportions
  floor_flags          which raw p-values sit at the floor of their resampling test
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import binom, norm, rankdata

from .config import POSITION_TOL_KM

__all__ = ["comparator_violated", "mcnemar_exact", "severity", "partial_spearman", "ols", "ratio_of_totals_ci", "top_share", "dial_weights",
           "floor_flags", "PERM_FLOOR", "BOOT_FLOOR"]

PERM_FLOOR = 1.0 / 10001.0          # 10,000 permutations, p = (1 + #) / (1 + B)
BOOT_FLOOR = 1.0 / 4001.0           # 4,000 bootstrap resamples, p = (1 + #) / (1 + B)


def comparator_violated(ok, err_km, tol_km=POSITION_TOL_KM):
    """True where the comparator run failed / was capped (ok false or no finite error) or ended more than tol_km from the reference."""
    ok = np.asarray(ok).astype(bool)
    err = np.asarray(err_km, dtype=float)
    return ~(ok & np.isfinite(err) & (err <= tol_km))


def mcnemar_exact(viol_a, viol_b):
    """Exact two-sided McNemar test. b = orbits where only A violates, c = only B. p = min(1, 2 P[Bin(b + c, 1/2) <= min(b, c)])."""
    a = np.asarray(viol_a, dtype=bool)
    b_ = np.asarray(viol_b, dtype=bool)
    if a.shape != b_.shape:
        raise ValueError("paired indicators must have the same shape")
    b = int((a & ~b_).sum())
    c = int((~a & b_).sum())
    n = b + c
    p = 1.0 if n == 0 else float(min(1.0, 2.0 * binom.cdf(min(b, c), n, 0.5)))
    return {"only_a": b, "only_b": c, "both": int((a & b_).sum()), "neither": int((~a & ~b_).sum()), "p_two_sided": p,
            "rate_a": float(a.mean()), "rate_b": float(b_.mean()), "n": int(a.size)}


def severity(err_km, violated, tol_km=POSITION_TOL_KM):
    """Among the violating runs: n, the number without a finite final error, and over those with one the median and 90th percentile of
    err / tol and the share above 10 x tol (both as a share of the finite-error violations and of all violations)."""
    v = np.asarray(violated, dtype=bool)
    e = np.asarray(err_km, dtype=float)[v]
    fin = np.isfinite(e)
    r = e[fin] / tol_km
    out = {"n_violations": int(v.sum()), "n_no_finite_error": int((~fin).sum()), "n_finite": int(fin.sum())}
    if fin.any():
        out.update({"median_err_over_tol": float(np.median(r)), "p90_err_over_tol": float(np.quantile(r, 0.9)),
                    "n_above_10tol": int((r > 10).sum()), "share_above_10tol_of_finite": float((r > 10).mean()),
                    "share_above_10tol_or_no_error_of_all": float(((r > 10).sum() + (~fin).sum()) / v.sum())})
    return out


def _resid(y, x):
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def partial_spearman(y, x, z):
    """Spearman correlation of y and x controlling z: ranks of all three, residuals of rank(y) and rank(x) after a linear fit on rank(z),
    Pearson correlation of the residuals."""
    ry, rx, rz = rankdata(y), rankdata(x), rankdata(z)
    ey, ex = _resid(ry, rz), _resid(rx, rz)
    return float(np.corrcoef(ey, ex)[0, 1])


def ols(y, *xs):
    """Least-squares coefficients of y on an intercept and the columns xs (intercept first)."""
    X = np.column_stack([np.ones(len(y))] + [np.asarray(x, dtype=float) for x in xs])
    beta, *_ = np.linalg.lstsq(X, np.asarray(y, dtype=float), rcond=None)
    return beta


def ratio_of_totals_ci(p, c, n_boot=4000, seed=0, chunk=250):
    """sum(c) / sum(p) over rows with a finite c, with the percentile and the BCa 95 % bootstrap intervals (same resamples). BCa: bias
    correction z0 = Phi^-1(share of resamples below the estimate), acceleration from the leave-one-out jackknife."""
    p = np.asarray(p, dtype=float)
    c = np.asarray(c, dtype=float)
    keep = np.isfinite(c)
    p, c = p[keep], c[keep]
    n = len(p)
    est = c.sum() / p.sum()
    rng = np.random.RandomState(seed)
    boots = []
    for b0 in range(0, n_boot, chunk):
        idx = rng.randint(0, n, (min(chunk, n_boot - b0), n))
        boots.append(c[idx].sum(axis=1) / p[idx].sum(axis=1))
    boots = np.concatenate(boots)
    pct = [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]
    prop = (np.sum(boots < est) + 0.5 * np.sum(boots == est)) / len(boots)
    z0 = norm.ppf(min(max(prop, 1e-12), 1 - 1e-12))
    jack = (c.sum() - c) / (p.sum() - p)
    d = jack.mean() - jack
    acc = float((d ** 3).sum() / (6.0 * ((d ** 2).sum()) ** 1.5)) if np.any(d != 0) else 0.0
    lims = []
    for q in (0.025, 0.975):
        zq = norm.ppf(q)
        a2 = norm.cdf(z0 + (z0 + zq) / (1.0 - acc * (z0 + zq)))
        lims.append(float(np.quantile(boots, a2)))
    return {"estimate": float(est), "n": int(n), "percentile_ci95": pct, "bca_ci95": lims, "z0": float(z0), "acceleration": acc}


def top_share(cost, frac):
    """Share of sum(cost) carried by the ceil(frac * n) costliest rows."""
    x = np.sort(np.asarray(cost, dtype=float))[::-1]
    k = max(1, int(math.ceil(frac * len(x))))
    return float(x[:k].sum() / x.sum())


def dial_weights(family, focus, w):
    """Weights giving family `focus` a total share w and the other families 1 - w in their observed proportions (normalized to mean 1)."""
    f = np.asarray(family, dtype=object)
    m = f == focus
    n, nf = len(f), int(m.sum())
    if nf == 0 or nf == n:
        raise ValueError("the focus family must be present and not alone")
    wt = np.where(m, w / (nf / n), (1.0 - w) / ((n - nf) / n))
    return wt


def floor_flags(pvals, floors=(PERM_FLOOR, BOOT_FLOOR), rel=1e-9):
    """{name: floor or None}: the resampling floor a raw p-value equals (within rel), if any."""
    out = {}
    for k, p in pvals.items():
        out[k] = next((f for f in floors if abs(p - f) <= rel * f), None)
    return out
