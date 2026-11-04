"""
Checks for the defense-plan statistics (orbit_ml_cr3bp/defense_metrics.py) and the exploratory kernels (orbit_ml_cr3bp/defense_runs.py).
Run:  python tests/test_defense.py   (each test_* function is run; the first failure stops)
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from orbit_ml_cr3bp import defense_metrics as dm  # noqa: E402


def test_comparator_violated():
    v = dm.comparator_violated([True, True, False, True], [50.0, 150.0, 10.0, np.nan])
    assert v.tolist() == [False, True, True, True]


def test_mcnemar_hand_computed():
    a = np.array([1, 1, 1, 0, 0, 0, 1, 0], dtype=bool)
    b = np.array([1, 0, 0, 1, 0, 0, 1, 0], dtype=bool)
    r = dm.mcnemar_exact(a, b)
    assert (r["only_a"], r["only_b"], r["both"], r["neither"]) == (2, 1, 2, 3)
    # 2 * P[Bin(3, 1/2) <= 1] = 2 * 4/8 = 1
    assert math.isclose(r["p_two_sided"], 1.0)
    r = dm.mcnemar_exact(np.ones(10, bool), np.zeros(10, bool))
    assert math.isclose(r["p_two_sided"], 2 * 0.5 ** 10)


def test_severity():
    err = np.array([50.0, 150.0, 2000.0, np.nan, 90.0])
    viol = np.array([False, True, True, True, False])
    s = dm.severity(err, viol)
    assert s["n_violations"] == 3 and s["n_no_finite_error"] == 1 and s["n_finite"] == 2
    assert math.isclose(s["median_err_over_tol"], (1.5 + 20.0) / 2)
    assert s["n_above_10tol"] == 1 and math.isclose(s["share_above_10tol_of_finite"], 0.5)
    assert math.isclose(s["share_above_10tol_or_no_error_of_all"], 2 / 3)


def test_partial_spearman_removes_a_common_driver():
    rng = np.random.RandomState(1)
    z = rng.randn(4000)
    x = z + 0.1 * rng.randn(4000)
    y = z + 0.1 * rng.randn(4000)
    from scipy.stats import spearmanr
    assert spearmanr(x, y)[0] > 0.9
    assert abs(dm.partial_spearman(y, x, z)) < 0.1


def test_ols():
    x = np.arange(10.0)
    z = (np.arange(10.0) % 3)
    y = 2.0 + 0.5 * x - 1.5 * z
    b = dm.ols(y, x, z)
    assert np.allclose(b, [2.0, 0.5, -1.5])


def test_ratio_of_totals_ci():
    rng = np.random.RandomState(0)
    p = rng.lognormal(5, 1, 3000)
    c = p * rng.lognormal(0, 0.3, 3000)
    c[:5] = np.nan
    r = dm.ratio_of_totals_ci(p, c, n_boot=500)
    keep = np.isfinite(c)
    assert r["n"] == 2995 and math.isclose(r["estimate"], c[keep].sum() / p[keep].sum())
    for k in ("percentile_ci95", "bca_ci95"):
        lo, hi = r[k]
        assert lo < r["estimate"] < hi, k


def test_top_share_and_dial_weights():
    assert math.isclose(dm.top_share([1, 1, 1, 1, 1, 1, 1, 1, 1, 91], 0.1), 0.91)
    fam = np.array(["a", "a", "b", "c"], dtype=object)
    w = dm.dial_weights(fam, "a", 0.25)
    assert math.isclose(w[fam == "a"].sum() / w.sum(), 0.25)
    assert math.isclose(w[2], w[3])                    # the other families keep their observed proportions
    w0 = dm.dial_weights(fam, "a", 0.5)                # the observed share reproduces equal weights
    assert np.allclose(w0, 1.0)


def test_floor_flags():
    f = dm.floor_flags({"perm": 1 / 10001, "boot": 1 / 4001, "exact": 1.2e-4})
    assert f["perm"] == dm.PERM_FLOOR and f["boot"] == dm.BOOT_FLOOR and f["exact"] is None


def test_sundman_adaptive_matches_physical_solution():
    """The adaptive Sundman-time solve ends at T, and at a tight tolerance its final position agrees with a tight physical-time solve."""
    from scipy.integrate import solve_ivp
    from orbit_ml_cr3bp import defense_runs as dr
    from orbit_ml_cr3bp.config import MU, DU_KM
    from orbit_ml_cr3bp.physics_3d import _cr3bp_derivatives_3d_numba
    s0 = np.array([0.5, 0.1, 0.02, 0.0, 0.6, 0.01])         # does not collide within T (the sealed kernels report ok)
    T = 3.0
    ref = solve_ivp(lambda t, y: _cr3bp_derivatives_3d_numba(y, MU), (0, T), s0, method="DOP853", rtol=1e-13, atol=1e-16).y[:3, -1]
    r = dr.run_sundman_adaptive(s0, T, ref, 1e-10, MU, "DOP853")
    assert r["ok"] and r["err_km"] < 0.01, r
    r5 = dr.run_sundman_adaptive(s0, T, ref, 1e-6, MU, "RK45")
    assert r5["ok"] and r5["nfev"] > 0 and abs(r5["t_end"] - T) < 1e-9, r5


def test_rk8_fixed_step_order():
    """The 12-stage DOP853 propagation formula used as a fixed-step method converges at about 8th order (physical time, g = 1)."""
    from orbit_ml_cr3bp import defense_runs as dr
    from orbit_ml_cr3bp.config import MU
    s0 = np.array([0.5, 0.1, 0.02, 0.0, 0.6, 0.01])
    T = 1.0
    errs = []
    ref = dr.integrate_endpoint_rk8s(s0, T / 512, T, MU, gfix=1.0)[2][:3]
    for n in (16, 32):
        errs.append(np.linalg.norm(dr.integrate_endpoint_rk8s(s0, T / n, T, MU, gfix=1.0)[2][:3] - ref))
    order = math.log2(errs[0] / errs[1])
    assert 7.0 < order < 9.5, order


def test_rk4s_wrapper_matches_sealed_kernel():
    """defense_runs' RK4 oracle helper uses the sealed kernel unchanged."""
    from orbit_ml_cr3bp import defense_runs as dr, sundman_3d as sd
    from orbit_ml_cr3bp.config import MU
    s0 = np.array([0.85, 0.0, 0.05, 0.0, 0.25, 0.0])
    s0 = np.array([0.5, 0.1, 0.02, 0.0, 0.6, 0.01])
    a = sd.integrate_endpoint_rk4s(s0, 0.05, 3.0, MU, max_steps=4_000_000)
    b = dr.run_fixed_s("rk4", s0, 0.05, 3.0, MU)
    assert a[0] and b["ok"] and np.allclose(a[2], b["final"]) and b["evals"] == 4 * (a[6] + a[7])


def test_oracle_ds_and_rk8_sundman():
    """On a non-colliding orbit both oracle steps exist, the RK8 step is longer, and RK8 in Sundman time agrees with a tight physical reference."""
    from scipy.integrate import solve_ivp
    from orbit_ml_cr3bp import defense_runs as dr
    from orbit_ml_cr3bp.config import MU, DU_KM
    from orbit_ml_cr3bp.physics_3d import _cr3bp_derivatives_3d_numba
    s0 = np.array([0.5, 0.1, 0.02, 0.0, 0.6, 0.01])
    T = 3.0
    ref = solve_ivp(lambda t, y: _cr3bp_derivatives_3d_numba(y, MU), (0, T), s0, method="DOP853", rtol=1e-13, atol=1e-16).y[:3, -1]
    r = dr.run_fixed_s("rk8", s0, 0.03, T, MU)
    assert r["ok"] and np.linalg.norm(r["final"][:3] - ref) * DU_KM < 1e-3
    o4, o8 = dr.oracle_ds("rk4", s0, T, ref, MU), dr.oracle_ds("rk8", s0, T, ref, MU)
    assert np.isfinite(o4["ds"]) and np.isfinite(o8["ds"]) and o8["ds"] > o4["ds"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
