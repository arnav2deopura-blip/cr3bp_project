"""
Confirmatory tests of Part D (see part_d/PREREGISTRATION_D.md). Same conventions as confirm_c_3d: every function takes arrays / DataFrames and returns a dict
with the estimate and a one-sided p-value in the DIRECTION FIXED IN ADVANCE; nothing here reads files (run_confirm_d_3d.py does). Family (Holm, familywise
0.05): D1 ML versus the best physics rule, D2 compiled throughput, D3 replication of the Part C headline, D4 conditional safety, D5 out-of-distribution
periodic orbits. D4 is an intersection-union test (its p is the largest of its parts), as C3 was.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binom

from .audit_3d import clopper_pearson
from .confirm_b_3d import holm, N_BOOT
from .confirm_c_3d import c2_policy_share, c5_ml_vs_global, win_share, _boot_p_le_zero

__all__ = ["holm", "d1_ml_vs_rule", "d2_throughput", "d3_replication", "d4_conditional", "d5_ood", "win_share"]


def d1_ml_vs_rule(led_ml, led_rule, n_boot=N_BOOT, seed=0):
    """D1. The ML policy's mean log10 total cost (overhead included) is below the best physics rule's, paired per orbit, orbits that met the tolerance under
    BOTH. The same statistic as C5 (confirm_c_3d.c5_ml_vs_global) with the rule in place of the one global step."""
    r = c5_ml_vs_global(led_ml, led_rule, n_boot=n_boot, seed=seed)
    r["comparator"] = "best physics rule"
    return r


def d2_throughput(t_policy, t_comp, n_boot=N_BOOT, seed=0):
    """D2. Throughput ratio = (total wall time of the compiled adaptive solver at equal risk over the orbits) / (total wall time of the compiled policy
    pipeline over the same orbits) > 1, i.e. the policy processes more orbits per second on one thread. Orbit-level bootstrap of the ratio (paired);
    also the share of orbits where the policy is faster and the median per-orbit ratio. Times in ns, one value per orbit and arm."""
    tp, tc = np.asarray(t_policy, dtype=float), np.asarray(t_comp, dtype=float)
    m = np.isfinite(tp) & np.isfinite(tc) & (tp > 0) & (tc > 0)
    tp, tc = tp[m], tc[m]
    ratio = float(tc.sum() / tp.sum())
    p, boots = _boot_p_le_zero(ratio - 1.0, len(tp), lambda i: tc[i].sum() / tp[i].sum() - 1.0, n_boot, seed)
    k = int((tp < tc).sum())
    return {"p": p, "throughput_ratio": ratio, "ci95": [float(np.quantile(boots, .025) + 1.0), float(np.quantile(boots, .975) + 1.0)],
            "share_orbits_policy_faster": k / len(tp), "median_orbit_ratio": float(np.median(tc / tp)), "n": int(len(tp)),
            "mean_ns_policy": float(tp.mean()), "mean_ns_comparator": float(tc.mean())}


def d3_replication(led, comp="RK45_eqr"):
    """D3. The C4 test on an independent dataset with the frozen Part C Sundman policy: share cheaper than the equal-risk RK45 above 0.5 (exact one-sided
    binomial). The calibration of the scale and of the comparator tolerance is repeated on the new data's own 4,000 calibration rows (Part C's protocol)."""
    r = c2_policy_share(led, comp)
    r["violation_rate"] = float((~led["meets_tol"].astype(bool)).mean())
    r["viol_cp95"] = list(clopper_pearson(int((~led["meets_tol"].astype(bool)).sum()), len(led)))
    return r


def d4_conditional(table, led, level=0.07, min_report=300, comp="RK45_eqr"):
    """D4. (i) every reported stratum (>= min_report evaluation orbits) has a violation rate below `level`: exact one-sided binomial per stratum against
    H0 'rate >= level', and the intersection-union p is the largest of them; (ii) the policy (abstentions and overhead included) is cheaper than the
    equal-risk comparator on more than half of the orbits (exact binomial, as C4). p = max of the two."""
    rep = table[table["n"] >= min_report]
    p_viol = float(rep["p_rate_ge_level"].max()) if len(rep) else 1.0
    cost = c2_policy_share(led, comp)
    k = int((~led["meets_tol"].astype(bool)).sum())
    return {"p": float(max(p_viol, cost["p"])), "p_strata": p_viol, "p_cost": cost["p"], "n_reported_strata": int(len(rep)), "level": level,
            "worst_stratum": rep.sort_values("rate", ascending=False).iloc[0][["group", "n", "rate", "cp_hi"]].to_dict() if len(rep) else None,
            "max_cp_hi": float(rep["cp_hi"].max()) if len(rep) else float("nan"), "share_cheaper": cost["share"], "share_ci95": cost["ci95"],
            "overall_violation_rate": k / len(led), "abstain_share": float(led["ml_abstain"].mean()), "comparator": comp}


def d5_ood(runs, threshold=0.10, n_boot=N_BOOT, seed=0, group_col="group", cluster_col="cluster"):
    """D5. Family-balanced violation rate of the frozen policy on the in-range periodic-orbit runs whose reference is trustworthy (columns: group, cluster,
    violation). The statistic is the mean over families of each family's violation rate (equal weight per family). The runs of one orbit (several horizons) and the
    two members of a north / south mirror pair are not independent, so the bootstrap resamples CLUSTERS (member, mirror pair merged) within each family; one-sided
    p = share of resamples with a statistic >= threshold, H0: statistic >= threshold (twice the nominal 5 %)."""
    fam = {}
    for g, d in runs.groupby(group_col):
        cl = [x["violation"].to_numpy(dtype=float) for _, x in d.groupby(cluster_col)]
        fam[g] = (cl, np.array([len(c) for c in cl]), np.array([c.sum() for c in cl]))
    stat = float(np.mean([sum(c.sum() for c in cl) / sum(len(c) for c in cl) for cl, _, _ in fam.values()]))
    rng = np.random.RandomState(seed)

    def rate(n, k, idx):
        return k[idx].sum() / n[idx].sum()

    boots = np.array([np.mean([rate(n, k, rng.randint(0, len(n), len(n))) for _, n, k in fam.values()]) for _ in range(n_boot)])
    p = float((1 + np.sum(boots >= threshold)) / (n_boot + 1))
    per = {}
    for g, (cl, n, k) in fam.items():
        per[g] = {"n": int(n.sum()), "violations": int(k.sum()), "rate": float(k.sum() / n.sum()), "n_clusters": int(len(n)), "cp95": list(clopper_pearson(int(k.sum()), int(n.sum())))}
    return {"p": p, "family_balanced_rate": stat, "ci95": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))], "threshold": threshold,
            "n_runs": int(sum(v["n"] for v in per.values())), "n_families": len(fam), "per_family": per}
