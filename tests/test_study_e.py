"""
Checks for the Study E pilot kernels (orbit_ml_cr3bp/study_e_rk8.py; posthoc/README.md P24-P25).
Run:  python tests/test_study_e.py   (each test_* function is run; the first failure stops)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from orbit_ml_cr3bp import study_e_rk8 as e8, defense_runs as dr  # noqa: E402
from orbit_ml_cr3bp.config import MU, du_km_for_mu  # noqa: E402

DU_KM = du_km_for_mu(MU)
STATES = np.array([[0.5, 0.1, 0.02, 0.0, 0.6, 0.01], [0.8, 0.0, 0.05, 0.0, 0.3, 0.0], [-0.4, 0.3, 0.0, -0.5, -0.4, 0.05]])
HORIZONS = np.array([3.0, 2.0, 5.0])
C_CONFIRM = os.path.join(ROOT, "data_c/confirm/cr3bp_3d_training_data_accuracy_monotone_seed43.csv")


def _ref(s, T):
    from scipy.integrate import solve_ivp
    from orbit_ml_cr3bp.physics_3d import _cr3bp_derivatives_3d_numba
    return solve_ivp(lambda t, y: _cr3bp_derivatives_3d_numba(y, MU), (0, T), s, method="DOP853", rtol=1e-13, atol=1e-16).y[:3, -1]


def test_rk8_runner_matches_python_kernel():
    ds = np.array([0.05, 0.1, 0.02])
    ok, y, ev, rs = e8.Rk8sRunner(MU).run_ds(STATES, HORIZONS, ds)
    for i in range(len(STATES)):
        r = dr.run_fixed_s("rk8", STATES[i], ds[i], HORIZONS[i], MU)
        assert bool(ok[i]) == r["ok"] and int(ev[i]) == r["evals"]
        assert np.array_equal(y[i], r["final"])


def test_monotone_search_reproduces_sealed_rk4s_search():
    """The generic copy of the search, run with the sealed RK4 Sundman kernel, gives the sealed search's label."""
    from orbit_ml_cr3bp.search_3d import find_dt_for_integrator_accuracy_monotone_3d
    from orbit_ml_cr3bp.physics_3d import Rk4sStep, REASON_OK
    from orbit_ml_cr3bp.sundman_3d import DS_MIN, DS_MAX, integrate_endpoint_rk4s
    for s, T in zip(STATES, HORIZONS):
        sealed = find_dt_for_integrator_accuracy_monotone_3d("rk4s", s, T, MU, dt_min=DS_MIN, dt_max=DS_MAX, tol_km=100.0, return_details=True,
                                                            step_func=Rk4sStep())
        ref = _ref(s, T)

        def err_fn(ds, s=s, T=T, ref=ref):
            r = integrate_endpoint_rk4s(s, ds, T, MU, 4_000_000)
            return (None, int(r[5])) if not r[0] else (float(np.linalg.norm(np.asarray(r[2])[:3] - ref)) * DU_KM, REASON_OK)
        mine = e8.monotone_search(err_fn, DS_MIN, DS_MAX, 100.0)
        assert mine["reason"] == sealed["reason"] and mine["censored"] == sealed["censored"]
        assert np.isclose(mine["dt"], sealed["dt"], rtol=1e-12), (mine["dt"], sealed["dt"])


def test_monotone_search_reproduces_stored_labels():
    """On real Study C confirmation rows the copy reproduces the stored dt_rk4s (skipped if the data are not present)."""
    if not os.path.exists(C_CONFIRM):
        print("  skipped (no data_c/confirm)")
        return
    from orbit_ml_cr3bp.physics_3d import REASON_OK
    from orbit_ml_cr3bp.sundman_3d import DS_MIN, DS_MAX, integrate_endpoint_rk4s
    d = pd.read_csv(C_CONFIRM, nrows=200, low_memory=False)
    d = d[(d["reference_outcome"].astype(str) == "ok") & (d["rk4s_reason"] == REASON_OK)].head(4)
    for _i, row in d.iterrows():
        s = row[["x", "y", "z", "vx", "vy", "vz"]].to_numpy(float)
        T = float(row["sim_time"])
        ref = _ref(s, T)

        def err_fn(ds, s=s, T=T, ref=ref):
            r = integrate_endpoint_rk4s(s, ds, T, MU, 4_000_000)
            return (None, int(r[5])) if not r[0] else (float(np.linalg.norm(np.asarray(r[2])[:3] - ref)) * DU_KM, REASON_OK)
        mine = e8.monotone_search(err_fn, DS_MIN, DS_MAX, 100.0)
        assert np.isclose(mine["dt"], row["dt_rk4s"], rtol=1e-9), (mine["dt"], row["dt_rk4s"])


def test_rk8s_label_meets_tolerance_and_beats_rk4():
    for s, T in zip(STATES, HORIZONS):
        ref = _ref(s, T)
        lab = e8.rk8s_label(s, T, ref)
        assert lab["rk8s_success"] == 1 and lab["rk8s_err_km"] <= 100.0
        r = dr.run_fixed_s("rk8", s, lab["dt_rk8s"], T, MU)
        assert r["ok"] and r["evals"] == lab["rk8s_evals"]
        assert lab["dt_rk8s"] > dr.oracle_ds("rk4", s, T, ref, MU)["ds"]


def test_sundman_matched_accuracy():
    s, T = STATES[0], HORIZONS[0]
    ref = _ref(s, T)
    for m in ("RK45", "DOP853"):
        r = e8.sundman_matched_accuracy(s, T, ref, method=m)
        assert r is not None and r["err_km"] <= 100.0
        again = dr.run_sundman_adaptive(s, T, ref, r["rtol"], MU, m)
        assert again["nfev"] == r["nfev"]
        assert r["rtol"] == 0.1 or not (lambda x: x["ok"] and x["err_km"] <= 100.0)(dr.run_sundman_adaptive(s, T, ref, 0.1, MU, m))


def _real_rows(n=12, seed=1):
    if not os.path.exists(C_CONFIRM):
        return None
    d = pd.read_csv(C_CONFIRM, nrows=400, low_memory=False)
    d = d[d["reference_outcome"].astype(str) == "ok"].sample(n, random_state=seed)
    return d[["x", "y", "z", "vx", "vy", "vz"]].to_numpy(float), d["sim_time"].to_numpy(float)


def test_compiled_sundman_adaptive_matches_scipy_p20():
    """Item 8: the compiled Sundman-time RK45 / DOP853 (SciPy controller, no collision check) reproduce P20's SciPy solver: the same nfev on
    every orbit and the same endpoint to well under a metre."""
    from orbit_ml_cr3bp import compiled_sundman_e as cs
    rows = _real_rows()
    S, T = rows if rows is not None else (STATES, HORIZONS)
    for name, m in (("RK45s", "RK45"), ("DOP853s", "DOP853")):
        sa = cs.SundmanAdaptive(name, check_bad=False)
        for rt in (1e-5, 1e-9):
            ok, y, nf, rs = sa.run(S, T, rt, parallel=False)
            for i in range(len(S)):
                r = dr.run_sundman_adaptive(S[i], T[i], y[i, :3], rt, MU, m)
                assert r["ok"] == bool(ok[i]), (name, rt, i)
                if r["ok"]:
                    assert nf[i] == r["nfev"], (name, rt, i, nf[i], r["nfev"])
                    assert r["err_km"] < 1e-3, (name, rt, i, r["err_km"])


def test_compiled_sundman_adaptive_variants():
    """The parallel batch equals the serial one; the collision check changes nothing on orbits that stay clear; DP5(4)-PI meets a tight
    tolerance with an nfev of the same order as the SciPy controller."""
    from orbit_ml_cr3bp import compiled_sundman_e as cs
    rows = _real_rows(8, 2)
    S, T = rows if rows is not None else (STATES, HORIZONS)
    for name in ("RK45s", "DP54PIs", "DOP853s"):
        a, b = cs.SundmanAdaptive(name, check_bad=True), cs.SundmanAdaptive(name, check_bad=False)
        s1, s2, s3 = a.run(S, T, 1e-8, parallel=False), a.run(S, T, 1e-8, parallel=True), b.run(S, T, 1e-8, parallel=False)
        for k in range(4):
            assert np.array_equal(s1[k], s2[k])
        clear = s1[0]
        assert np.array_equal(s1[2][clear], s3[2][clear]) and np.array_equal(s1[1][clear], s3[1][clear])
    ref = np.array([_ref(s, t) for s, t in zip(S, T)])
    pi = cs.SundmanAdaptive("DP54PIs").rows(S, T, ref, 1e-10)
    sp = cs.SundmanAdaptive("RK45s").rows(S, T, ref, 1e-10)
    both = pi["ok"] & sp["ok"]
    assert both.sum() >= len(S) - 1 and np.all(pi["err"][both] < 100.0)
    r = pi["nfev"][both] / sp["nfev"][both]
    assert np.all((r > 0.3) & (r < 3.0)), r


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
