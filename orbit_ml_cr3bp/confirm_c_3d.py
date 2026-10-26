"""
Confirmatory tests of Part C (see part_c/PREREGISTRATION_C.md). Every function takes arrays / DataFrames and returns a dict with the estimate
and a one-sided p-value in the DIRECTION FIXED IN ADVANCE; nothing here reads files (run_confirm_c_3d.py does), so the functions can be dry-run on
D_select before the seal. Family (Holm, familywise 0.05): C1 prediction, C2 physical-time policy, C3 Sundman paradigm, C4 Sundman ML policy,
C5 ML adds value over one global certified step. C3 is an intersection-union test: its p is the larger of its two parts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binom, spearmanr

from .confirm_b_3d import holm, N_BOOT

__all__ = ["holm", "c1_prediction", "c2_policy_share", "c3_sundman_paradigm", "c5_ml_vs_global", "win_share"]


def _boot_p_le_zero(diff_stat, n, stat_fn, n_boot=N_BOOT, seed=0):
    """One-sided bootstrap p for H0: stat <= 0: (1 + #{b: stat_b <= 0}) / (B + 1)."""
    rng = np.random.RandomState(seed)
    boots = np.array([stat_fn(rng.randint(0, n, n)) for _ in range(n_boot)])
    return float((1 + np.sum(boots <= 0)) / (n_boot + 1)), boots


def c1_prediction(abs_resid_c, abs_resid_b, n_boot=N_BOOT, seed=0):
    """C1. The Part C model's median |log10 raw / label| is lower than the Part B-style state-only model refitted on the SAME D_train.
    Arrays are per orbit, same orbits, orbits with a valid RK4 label. Statistic: median(b) - median(c) > 0; paired bootstrap over orbits."""
    c, b = np.asarray(abs_resid_c, dtype=float), np.asarray(abs_resid_b, dtype=float)
    m = np.isfinite(c) & np.isfinite(b)
    c, b = c[m], b[m]
    stat = float(np.median(b) - np.median(c))
    p, boots = _boot_p_le_zero(stat, len(c), lambda i: np.median(b[i]) - np.median(c[i]), n_boot, seed)
    return {"p": p, "median_abs_resid_partC": float(np.median(c)), "median_abs_resid_partB_refit": float(np.median(b)), "difference": stat,
            "ci95": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))], "n": int(len(c))}


def win_share(led, comp):
    """The orbits where the policy met the tolerance AND its total cost (overhead included) is below the comparator's, out of every orbit that
    has a comparator cost (a violating orbit is not a win; this is the share Part B reported as 'cheaper than adaptive')."""
    col = f"{comp}_nfev"
    has = led[col].notna()
    k = int((led["meets_tol"].astype(bool) & has & (led["C_ml_tot"] < led[col])).sum())
    return k, int(has.sum())


def c2_policy_share(led, comp="RK45_eqr"):
    """C2 (physical-time policy) and C4 (Sundman policy): share of orbits where the policy, overhead included, is cheaper than `comp` is > 0.5.
    Exact one-sided binomial, H0 share <= 0.5."""
    k, n = win_share(led, comp)
    from orbit_ml_cr3bp.audit_3d import clopper_pearson
    return {"p": float(binom.sf(k - 1, n, 0.5)), "k": k, "n": n, "share": k / n, "ci95": list(clopper_pearson(k, n)), "comparator": comp}


def c3_sundman_paradigm(M, comp="RK45_spec", n_boot=2000, seed=0):
    """C3. (a) the oracle fixed-ds Sundman run (counted cost) is cheaper than `comp` on > 50 % of orbits (exact one-sided binomial);
    (b) Spearman(log delta_phys, P_s) < Spearman(log delta_phys, P) on the same orbits (paired bootstrap), P_s = log10(C_s* / C_ad),
    P = log10(C_fix* / C_ad). p = max of the two parts. M needs C_s_star, C_fix_star, <comp>_nfev, log_delta_phys."""
    col = f"{comp}_nfev"
    d = M[M[col].notna() & M["C_s_star"].notna() & M["C_fix_star"].notna()]
    k, n = int((d["C_s_star"] < d[col]).sum()), int(len(d))
    p_a = float(binom.sf(k - 1, n, 0.5))
    x = d["log_delta_phys"].to_numpy(dtype=float)
    ps = np.log10(d["C_s_star"].to_numpy(dtype=float) / d[col].to_numpy(dtype=float))
    pp = np.log10(d["C_fix_star"].to_numpy(dtype=float) / d[col].to_numpy(dtype=float))
    rho_s, rho_p = float(spearmanr(x, ps).statistic), float(spearmanr(x, pp).statistic)

    def stat(i):
        return spearmanr(x[i], pp[i]).statistic - spearmanr(x[i], ps[i]).statistic

    p_b, boots = _boot_p_le_zero(rho_p - rho_s, n, stat, n_boot, seed)
    from orbit_ml_cr3bp.audit_3d import clopper_pearson
    return {"p": float(max(p_a, p_b)), "a_share_cheaper": {"k": k, "n": n, "share": k / n, "p": p_a, "ci95": list(clopper_pearson(k, n))},
            "b_rho": {"rho_sundman": rho_s, "rho_fixed_physical": rho_p, "difference": rho_p - rho_s, "p": p_b,
                      "ci95": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))]},
            "median_P_s": float(np.median(ps)), "median_P": float(np.median(pp)), "comparator": comp}


def c5_ml_vs_global(led_ml, led_glob, n_boot=N_BOOT, seed=0):
    """C5. The Sundman ML policy's mean log10 total cost is below the one-global-ds policy's, at certified risk, paired per orbit, counting
    orbits that met the tolerance under BOTH (the overhead of the ML model is in its cost; the global policy has none)."""
    m = led_ml[["row_id", "meets_tol", "C_ml_tot"]].merge(led_glob[["row_id", "meets_tol", "C_ml_tot"]], on="row_id", suffixes=("_ml", "_g"))
    both = m[m["meets_tol_ml"].astype(bool) & m["meets_tol_g"].astype(bool)]
    diff = (np.log10(both["C_ml_tot_g"]) - np.log10(both["C_ml_tot_ml"])).to_numpy()       # > 0: ML cheaper
    p, boots = _boot_p_le_zero(float(diff.mean()), len(diff), lambda i: diff[i].mean(), n_boot, seed)
    return {"p": p, "mean_log10_saving": float(diff.mean()), "ci95": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))],
            "n_paired": int(len(diff)), "ml_cheaper_share": float((diff > 0).mean()),
            "violation_rate_ml": float((~led_ml["meets_tol"].astype(bool)).mean()), "violation_rate_global": float((~led_glob["meets_tol"].astype(bool)).mean())}
