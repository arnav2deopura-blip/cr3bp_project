"""
Confirmatory tests of Part B (see PREREGISTRATION_B.md). Every function takes DataFrames and returns a dict with the estimate, a
one- or two-sided p-value in the DIRECTION FIXED IN ADVANCE, and the inputs of the decision rule. Nothing here reads files; the
runner (run_confirm_b_3d.py) does, so these functions can be dry-run on D_explore before the seal.

Family (Holm, familywise 0.05): T1 paradigm (HB1), T2 information (HB2), T3 calibration shape (HB4), T4 where (HB5),
T5 horizon dial (HB6). Each is an intersection-union test: its p-value is the largest p of its parts.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import binom, chi2_contingency, mannwhitneyu, spearmanr

N_PERM = 10_000
N_BOOT = 4_000
ALPHA = 0.05


def holm(pvals: dict, alpha=ALPHA):
    """Holm-Bonferroni step-down. Returns {name: {"p": p, "p_holm": adjusted, "reject": bool}}."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        out[k] = {"p": float(p), "p_holm": float(running), "reject": bool(running <= alpha)}
    return out


def _gap(led, comp):
    d = led[led[f"{comp}_nfev"].notna() & led["C_fix_star"].notna()]
    return d, np.log10(d["C_fix_star"] / d[f"{comp}_nfev"])


def spearman_perm_onesided(x, y, n_perm=N_PERM, seed=0, alternative="greater"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    rho = float(spearmanr(x, y).statistic)
    rng = np.random.RandomState(seed)
    from scipy.stats import rankdata
    rx, ry = rankdata(x), rankdata(y)
    rx = (rx - rx.mean()) / rx.std()
    ry = (ry - ry.mean()) / ry.std()
    null = np.array([np.mean(rx * ry[rng.permutation(len(ry))]) for _ in range(n_perm)])
    p = (1 + np.sum(null >= rho)) / (n_perm + 1) if alternative == "greater" else (1 + np.sum(null <= rho)) / (n_perm + 1)
    return rho, float(p), int(len(x))


# ---------------------------------------------------------------------------------------------------------------------
def t1_paradigm(led, comparators=("RK45_spec", "DOP853_spec"), delta_col="log_delta_phys", seed=0):
    """HB1. (a) for each spec comparator: H0 share(C_fix* < C_ad) >= 0.5, one-sided exact binomial for < 0.5; (b) Spearman rho of
    log10 delta_phys with the paradigm gap vs RK45 spec, H0 rho <= 0 (permutation). p = max over all parts."""
    parts, rep = {}, {}
    for c in comparators:
        d, g = _gap(led, c)
        k, n = int((d["C_fix_star"] < d[f"{c}_nfev"]).sum()), int(len(d))
        p = float(binom.cdf(k, n, 0.5))      # P(X <= k | p = 0.5): small when the share is well below one half
        parts[f"share_{c}"] = p
        rep[f"share_{c}"] = {"k": k, "n": n, "share": k / n, "p": p}
    d, g = _gap(led, comparators[0])
    rho, p, n = spearman_perm_onesided(d[delta_col], g, seed=seed)
    b, a0 = np.polyfit(d[delta_col].to_numpy(dtype=float), g.to_numpy(dtype=float), 1)
    rng = np.random.RandomState(seed)
    xs, ys = d[delta_col].to_numpy(dtype=float), g.to_numpy(dtype=float)
    boots = [np.polyfit(xs[i], ys[i], 1)[0] for i in (rng.randint(0, len(xs), len(xs)) for _ in range(1000))]
    rep["rho_delta_gap"] = {"rho": rho, "p": p, "n": n, "slope": float(b), "intercept": float(a0),
                            "slope_ci95": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))]}
    parts["rho"] = p
    return {"p": float(max(parts.values())), "parts": rep}


def t2_information(sig, eps=1e-5, seed=0):
    """HB2. sig: E3a per-(base orbit, eps) table with lambda_T and sigma_chosen. H0: Spearman(lambda_T, sigma) <= 0 at the
    pre-registered eps (permutation). Also reported (not part of the p): median |residual| / sigma and the share of orbits whose
    residual is within sigma, overall and in the top lambda_T tercile."""
    g = sig[np.isclose(sig["eps"], eps)].dropna(subset=["sigma_chosen", "lambda_T"])
    rho, p, n = spearman_perm_onesided(g["lambda_T"], g["sigma_chosen"], seed=seed)
    top = g[g["lambda_T"] >= g["lambda_T"].quantile(2 / 3)]
    ratio = (top["abs_resid_dex"] / top["sigma_chosen"]).replace([np.inf, -np.inf], np.nan)
    return {"p": p, "rho": rho, "n": n, "eps": eps,
            "share_sigma_positive": float((g["sigma_chosen"] > 0).mean()),
            "max_sigma_dex": float(g["sigma_chosen"].max()), "median_abs_resid_dex": float(g["abs_resid_dex"].median()),
            "top_tercile_share_resid_within_sigma": float((top["abs_resid_dex"] <= top["sigma_chosen"]).mean()),
            "top_tercile_median_resid_over_sigma": float(ratio.median()) if ratio.notna().any() else None}


def t3_calibration_shape(led, alt_global, alt_mondrian, cell_col="cell", seed=0):
    """HB4. (a) chi-square homogeneity of the violation indicator across the pre-specified (merged) cells; (b) paired comparison,
    per orbit, of log10 cost under Mondrian-by-regime scales vs the global scale, H0: Mondrian mean log cost >= global (one-sided,
    paired bootstrap), counting only orbits that met the tolerance under BOTH. alt_*: DataFrames with row_id, viol, cost."""
    tab = pd.crosstab(led[cell_col], led["violation"])
    chi2, p_a, dof, _ = chi2_contingency(tab.to_numpy())
    m = alt_global.merge(alt_mondrian, on="row_id", suffixes=("_g", "_m"))
    both = m[~m["viol_g"] & ~m["viol_m"]]
    diff = np.log10(both["cost_m"]) - np.log10(both["cost_g"])
    rng = np.random.RandomState(seed)
    boots = np.array([diff.to_numpy()[rng.randint(0, len(diff), len(diff))].mean() for _ in range(N_BOOT)])
    p_b = float((1 + np.sum(boots >= 0)) / (N_BOOT + 1))     # evidence that Mondrian is cheaper = bootstrap mass below 0
    return {"p": float(max(p_a, p_b)),
            "chi2_homogeneity": {"chi2": float(chi2), "dof": int(dof), "p": float(p_a), "n_cells": int(tab.shape[0])},
            "mondrian_minus_global_mean_log10_cost": {"mean": float(diff.mean()), "ci95": [float(np.quantile(boots, .025)),
                                                       float(np.quantile(boots, .975))], "p_mondrian_cheaper": p_b, "n_paired": int(len(diff))},
            "violation_rate_global": float(alt_global["viol"].mean()), "violation_rate_mondrian": float(alt_mondrian["viol"].mean())}


def t4_where(led, comp="RK45_spec"):
    """HB5. Three one-sided contrasts, Holm within the three:
       C1 paradigm gap: min r2 < 0.17 DU  >  min r2 >= 0.17 DU          (Mann-Whitney)
       C2 paradigm gap: Hill regimes II, III  >  regimes IV, V          (Mann-Whitney)
       C3 violation rate: min r2 < 0.05 DU  >  min r2 >= 0.05 DU        (one-sided two-proportion z, pooled)"""
    d, g = _gap(led, comp)
    d = d.assign(gap=g)
    c1 = mannwhitneyu(d.loc[d["min_r2"] < 0.17, "gap"], d.loc[d["min_r2"] >= 0.17, "gap"], alternative="greater")
    a = d[d["regime"].isin(["II", "III"])]["gap"]
    b = d[d["regime"].isin(["IV", "V"])]["gap"]
    c2 = mannwhitneyu(a, b, alternative="greater")
    v = led[led["has_choice"]]
    x1, n1 = int(v.loc[v["min_r2"] < 0.05, "violation"].sum()), int((v["min_r2"] < 0.05).sum())
    x2, n2 = int(v.loc[v["min_r2"] >= 0.05, "violation"].sum()), int((v["min_r2"] >= 0.05).sum())
    p_pool = (x1 + x2) / (n1 + n2)
    z = (x1 / n1 - x2 / n2) / math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    from scipy.stats import norm
    p3 = float(1 - norm.cdf(z))
    ps = {"C1_gap_r2<0.17": float(c1.pvalue), "C2_gap_regimes_II_III_vs_IV_V": float(c2.pvalue), "C3_violation_r2<0.05": p3}
    h = holm(ps)
    return {"p": float(max(v["p_holm"] for v in h.values())), "contrasts": h,
            "medians": {"gap_r2<0.17": float(d.loc[d["min_r2"] < 0.17, "gap"].median()),
                        "gap_r2>=0.17": float(d.loc[d["min_r2"] >= 0.17, "gap"].median()),
                        "gap_II_III": float(a.median()), "gap_IV_V": float(b.median()),
                        "viol_r2<0.05": x1 / n1, "viol_r2>=0.05": x2 / n2}}


def paired_trend(df, metric, order_col="sim_time", id_col="orbit_idx", seed=0, n_boot=N_BOOT):
    """Within-IC Spearman rho between the ordering variable and `metric`, averaged over ICs that have every level; one-sided
    bootstrap over ICs for H0: mean rho <= 0. Respects the pairing (the same ICs appear at every horizon)."""
    piv = df.pivot_table(index=id_col, columns=order_col, values=metric)
    piv = piv.dropna()
    levels = piv.columns.to_numpy(dtype=float)
    rhos = np.array([spearmanr(levels, row).statistic for row in piv.to_numpy()])
    rhos = rhos[np.isfinite(rhos)]
    rng = np.random.RandomState(seed)
    boots = np.array([rhos[rng.randint(0, len(rhos), len(rhos))].mean() for _ in range(n_boot)])
    return {"mean_rho": float(rhos.mean()), "ci95": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))],
            "p": float((1 + np.sum(boots <= 0)) / (n_boot + 1)), "n_ics": int(len(rhos)),
            "by_level_median": {str(float(l)): float(piv[l].median()) for l in piv.columns}}


def t5_horizon_dial(led, comp="RK45_spec"):
    """HB6 (horizon dial). Both the paradigm gap log10(C_fix*/C_ad) and the integration-only ML gap log10(C_ml_int/C_ad) must rise
    with T, paired within initial state. The overhead term is excluded from the ML gap because a fixed model cost is a larger
    share of a cheaper (short) solve, which would work against the hypothesis for a reason unrelated to the dynamics."""
    d = led[led[f"{comp}_nfev"].notna() & led["C_fix_star"].notna()].copy()
    d["paradigm_gap"] = np.log10(d["C_fix_star"] / d[f"{comp}_nfev"])
    m = d[d["ml_ok"].fillna(False).astype(bool) & d["ml_evals"].notna()].copy()
    m["ml_int_gap"] = np.log10(m["ml_evals"] / m[f"{comp}_nfev"])
    r1 = paired_trend(d, "paradigm_gap")
    r2 = paired_trend(m, "ml_int_gap")
    viol = led.groupby("sim_time")["violation"].mean().to_dict()
    return {"p": float(max(r1["p"], r2["p"])), "paradigm_gap": r1, "ml_integration_gap": r2,
            "violation_rate_by_T": {str(float(k)): float(v) for k, v in viol.items()}}
