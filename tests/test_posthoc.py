"""
Checks for the post hoc summaries (orbit_ml_cr3bp/posthoc_metrics.py). Run:  python tests/test_posthoc.py   (each test_* function is run; the first
failure stops)
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from orbit_ml_cr3bp import posthoc_metrics as pm  # noqa: E402


def _toy():
    # orbit 3 has no comparator cost; orbit 4 is cheaper but violated, so it is not a win
    return (np.array([100.0, 200.0, 300.0, 400.0]), np.array([150.0, 100.0, np.nan, 500.0]), np.array([True, True, True, False]))


def test_cost_stats_hand_computed():
    p, c, met = _toy()
    r = pm.cost_stats(p, c, met, n_boot=200)
    assert r["n"] == 4 and r["n_with_comp"] == 3 and r["n_no_comp"] == 1
    assert r["k"] == 1 and math.isclose(r["share"], 1 / 3)
    assert math.isclose(r["aggregate_ratio"], 750.0 / 700.0)
    assert math.isclose(r["mean_log_ratio"], (math.log10(1.5) + math.log10(0.5) + math.log10(1.25)) / 3)
    assert math.isclose(r["median_policy_all"], 200.0) and math.isclose(r["median_policy_met"], 150.0)
    assert math.isclose(r["violation_rate"], 1 / 3)
    for k in ("share", "aggregate_ratio", "mean_log_ratio"):
        lo, hi = r[f"{k}_ci95"]
        assert lo <= r[k] <= hi, k


def test_cost_stats_weighted():
    p, c, met = _toy()
    r = pm.cost_stats(p, c, met, weights=np.array([2.0, 1.0, 1.0, 1.0]), n_boot=50)
    assert math.isclose(r["share"], 2 / 4)
    assert math.isclose(r["aggregate_ratio"], (2 * 150 + 100 + 500) / (2 * 100 + 200 + 400))
    assert r["median_policy_all"] is None          # medians are not reweighted
    # weight 0 drops a row entirely
    r0 = pm.cost_stats(p, c, met, weights=np.array([1.0, 0.0, 1.0, 1.0]), n_boot=50)
    assert math.isclose(r0["aggregate_ratio"], 650.0 / 500.0)


def test_share_matches_sealed_win_share():
    from orbit_ml_cr3bp.confirm_c_3d import win_share
    rng = np.random.RandomState(3)
    n = 500
    led = pd.DataFrame({"C_ml_tot": rng.uniform(50, 500, n), "RK45_eqr_nfev": np.where(rng.rand(n) < 0.05, np.nan, rng.uniform(50, 500, n)),
                        "meets_tol": rng.rand(n) < 0.95})
    k, m = win_share(led, "RK45_eqr")
    r = pm.stats_frame(led, "RK45_eqr", n_boot=10)
    assert r["k"] == k and r["n_with_comp"] == m and math.isclose(r["share"], k / m)


def test_degenerate_bootstrap():
    p = np.full(30, 100.0)
    c = np.full(30, 120.0)
    r = pm.cost_stats(p, c, np.ones(30, bool), n_boot=100)
    assert r["aggregate_ratio_ci95"] == [1.2, 1.2] or np.allclose(r["aggregate_ratio_ci95"], 1.2)
    assert np.allclose(r["share_ci95"], 1.0)


def test_bad_costs_raise():
    for p in (np.array([0.0, 1.0]), np.array([np.nan, 1.0])):
        try:
            pm.cost_stats(p, np.array([1.0, 1.0]), np.ones(2, bool), n_boot=5)
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


def test_mixture_weights():
    fam = np.array(["a", "a", "b", "c"])
    w = pm.mixture_weights(fam, {"a": 0.5, "b": 0.5})
    assert np.allclose(w, [1.0, 1.0, 2.0, 0.0])
    # weighted family shares equal the target
    for f, t in {"a": 0.5, "b": 0.5, "c": 0.0}.items():
        assert math.isclose(w[fam == f].sum() / w.sum(), t)
    assert np.allclose(pm.mixture_weights(fam, None), 1.0)
    try:
        pm.mixture_weights(fam, {"z": 1.0})
    except ValueError:
        pass
    else:
        raise AssertionError("absent family must raise")
    for name, mix in pm.MIXTURES.items():
        if mix:
            assert math.isclose(sum(mix.values()), 1.0), name
            assert set(mix) <= set(pm.FAMILIES), name


def test_stratified_bootstrap_keeps_group_sizes():
    st = np.array(["a"] * 7 + ["b"] * 3)
    for idx in pm._boot_indices(10, 30, 0, st):
        assert idx.shape[1] == 10
        assert np.all((st[idx] == "a").sum(axis=1) == 7)


def test_breakdown_sums():
    rng = np.random.RandomState(1)
    n = 300
    led = pd.DataFrame({"C_ml_tot": rng.uniform(50, 500, n), "RK45_eqr_nfev": rng.uniform(50, 500, n), "meets_tol": True,
                        "h_band": rng.choice(["1-3", "3-8", "8-20"], n)})
    rows = pm.breakdown(led, "RK45_eqr", "h_band", n_boot=10)
    assert math.isclose(sum(r["share_of_policy_cost"] for r in rows), 1.0)
    assert sum(r["n"] for r in rows) == n
    assert sum(r["k"] for r in rows) == pm.stats_frame(led, "RK45_eqr", n_boot=10)["k"]


def test_recovered_share_matches_c5():
    from orbit_ml_cr3bp.confirm_c_3d import c5_ml_vs_global
    rng = np.random.RandomState(2)
    n = 400

    def led(scale, p_ok):
        return pd.DataFrame({"row_id": np.arange(n), "C_ml_tot": 100.0 * scale * rng.uniform(0.8, 1.25, n), "meets_tol": rng.rand(n) < p_ok})
    lm, lr, lg = led(1.0, 0.96), led(1.2, 0.95), led(2.0, 0.96)
    r = pm.recovered_share(lm, lr, lg, n_boot=200)
    gr = c5_ml_vs_global(lm, lr, n_boot=10)["mean_log10_saving"]
    gg = c5_ml_vs_global(lm, lg, n_boot=10)["mean_log10_saving"]
    assert math.isclose(r["gap_rule"], gr) and math.isclose(r["gap_global"], gg)
    assert math.isclose(r["recovered_share"], 1 - gr / gg)
    assert r["recovered_share_ci95"][0] <= r["recovered_share"] <= r["recovered_share_ci95"][1]


# ---------------------------------------------------------------------------------------------------------------------
# symmetric comparator certification (orbit_ml_cr3bp/symmetric_cert.py)
# ---------------------------------------------------------------------------------------------------------------------

def test_comparator_symmetry():
    """The policy and the comparators must be certified by the same rule on grids of matching cost resolution. The sealed comparator grid
    fails this check (plan F2); the post hoc fine grid passes it for every adaptive method."""
    from orbit_ml_cr3bp import symmetric_cert as sc, policy_c_3d as pc
    from orbit_ml_cr3bp.fair_baseline_3d import RTOL_GRID_B
    policy_res = float(np.max(np.diff(np.log10(sorted(pc.FIXED_SEQ_GRID)))))       # policy cost ~ 1 / scale
    assert math.isclose(policy_res, math.log10(2.0) / 8.0, rel_tol=1e-9)
    for m, order in sc.METHOD_ORDER.items():
        assert sc.cost_resolution_dex(sc.RTOL_GRID_FINE, order) <= policy_res, m
    assert sc.cost_resolution_dex(RTOL_GRID_B, 5.0) > policy_res                   # the sealed asymmetry
    # same rule: a synthetic monotone risk gives the same answer through the policy's certify_sequence and the comparator wrapper
    n = 4000

    def viol(x):
        return int(round(n * min(1.0, 0.03 * (x / 1e-6) ** 0.5))), n
    rep = sc.certify_rtol_fixed_sequence(viol, sc.RTOL_GRID_FINE)
    direct = pc.certify_sequence(viol, sc.RTOL_GRID_FINE, pc.ALPHA, pc.DELTA, "fixed_sequence")
    assert rep["chosen_rtol"] == direct["chosen_scale"] and rep["delta"] == pc.DELTA


def test_fixed_sequence_rtol_brute_force():
    from orbit_ml_cr3bp import symmetric_cert as sc
    rng = np.random.RandomState(0)
    n = 1000
    for trial in range(20):
        # non-monotone risk on purpose: the rule stops at the first failure even if a looser value would pass
        rates = np.clip(np.cumsum(rng.uniform(-0.002, 0.006, len(sc.RTOL_GRID_FINE))), 0, 1)
        k = {rt: int(round(r * n)) for rt, r in zip(sc.RTOL_GRID_FINE, rates)}
        rep = sc.certify_rtol_fixed_sequence(lambda rt: (k[rt], n), sc.RTOL_GRID_FINE)
        exp = None
        for rt in sorted(sc.RTOL_GRID_FINE):
            if sc.binom_p(k[rt], n) > 0.1:
                break
            exp = rt
        assert rep["chosen_rtol"] == exp, trial
        for d in (0.1, 0.05, 0.01):
            exp_d = None
            for rt in sorted(sc.RTOL_GRID_FINE):
                if sc.binom_p(k[rt], n) > d:
                    break
                exp_d = rt
            assert sc.chosen_at(rep["chain"], d) == exp_d, (trial, d)


def _toy_cal(n=14, seed=5):
    from orbit_ml_cr3bp.config import MU
    from orbit_ml_cr3bp import policy_c_3d as pc
    rng = np.random.RandomState(seed)
    rows = []
    while len(rows) < n:
        s = [rng.uniform(0.6, 1.3), rng.uniform(-0.4, 0.4), rng.uniform(-0.1, 0.1), rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), rng.uniform(-0.1, 0.1)]
        if min(np.hypot(s[0] + MU, s[1]), np.hypot(s[0] + MU - 1.0, s[1])) > 0.08:
            rows.append(s + [rng.uniform(1.0, 6.0)])
    df = pd.DataFrame(rows, columns=["x", "y", "z", "vx", "vy", "vz", "sim_time"])
    return df, pc.reference_final_positions(df, MU, n_jobs=1)


def test_equal_risk_matches_sealed():
    """The generic equal-risk rule with the sealed atol ratio gives exactly the sealed equal-risk rtol (policy_b_3d.certify_rtol_lazy)."""
    from orbit_ml_cr3bp import symmetric_cert as sc, policy_b_3d as pb
    df, ref = _toy_cal()
    S, T = df[pb.STATE_COLS].to_numpy(), df["sim_time"].to_numpy()
    items = list(zip(S, T, ref))
    for target in (0.0, 0.15, 0.4):
        sealed = pb.certify_rtol_lazy(df, "RK45", equal_risk_target=target, n_jobs=1, items=items)["equal_risk_rtol"]

        def viol(rt):
            r = sc.scipy_rows(None, S, T, ref, rt, "RK45", 1e-3)
            return int((~r["met"]).sum()), len(S)
        mine, _chain = sc.equal_risk_rtol(viol, target)
        assert mine == sealed, (target, mine, sealed)


def test_vector_atol_equals_scalar():
    from orbit_ml_cr3bp import symmetric_cert as sc
    df, ref = _toy_cal(4, seed=9)
    S, T = df[["x", "y", "z", "vx", "vy", "vz"]].to_numpy(), df["sim_time"].to_numpy()
    a = sc.scipy_rows(None, S, T, ref, 1e-6, "RK45", 1e-3)
    b = sc.scipy_rows(None, S, T, ref, 1e-6, "RK45", (1e-3,) * 6)
    c = sc.scipy_rows(None, S, T, ref, 1e-6, "RK45", sc.ATOL_VARIANTS["pos100"])
    assert np.array_equal(a["nfev"], b["nfev"]) and np.allclose(a["err"], b["err"])
    assert not np.array_equal(a["nfev"], c["nfev"])          # the variant really changes the error control


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for f in fns:
        f()
        print("ok", f.__name__)
    print(f"{len(fns)} tests passed")
