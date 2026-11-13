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


def _doomed_rows(n=10, seed=3):
    if not os.path.exists(C_CONFIRM):
        return None
    d = pd.read_csv(C_CONFIRM, nrows=3000, low_memory=False)
    d = d[d["reference_outcome"].astype(str).isin(["primary_collision", "secondary_collision", "ejection"])]
    d = d.sample(min(n, len(d)), random_state=seed)
    return d[["x", "y", "z", "vx", "vy", "vz"]].to_numpy(float), d["sim_time"].to_numpy(float)


def _mixed_rows():
    a, b = _real_rows(6, 4), _doomed_rows(6, 4)
    if a is None:
        return STATES, HORIZONS
    return np.vstack([a[0], b[0]]), np.concatenate([a[1], b[1]])


def test_event_kernels_equal_the_originals():
    """Item 4: the event-keeping copies of the fixed-step Sundman kernels and the generic physical-time adaptive kernel give the same run as
    the kernels they copy (usable and doomed orbits)."""
    from orbit_ml_cr3bp import events_e as ee, compiled_solvers_d as csd
    from orbit_ml_cr3bp.sundman_3d import integrate_endpoint_rk4s
    from orbit_ml_cr3bp.config import collision_radii, EJECTION_RADIUS
    rc1, rc2 = collision_radii(MU)
    S, T = _mixed_rows()
    tb = csd.scipy_tableaux()["dop853"]
    n_bad = 0
    for s, t in zip(S, T):
        for ds in (0.3, 0.05):
            a = ee.run_with_event("rk4s", s, t, ds)
            r = integrate_endpoint_rk4s(s, ds, t, MU, 4_000_000)
            assert a["ok"] == bool(r[0]) and a["reason"] == int(r[5]) and a["evals"] == 4 * (r[6] + r[7])
            assert np.array_equal(a["final"], np.asarray(r[2]))
            b = ee.run_with_event("rk8s", s, t, ds * 4)
            r = dr.integrate_endpoint_rk8s(s, ds * 4, t, MU)
            assert b["ok"] == bool(r[0]) and b["reason"] == int(r[5]) and b["evals"] == 12 * (r[6] + r[7])
            assert np.array_equal(b["final"], np.asarray(r[2]))
            n_bad += int(not a["ok"])
        for rt in (1e-6, 1e-10):
            for meth, ref_fn in (("DP54PI", lambda: csd.KR.dp54_endpoint(s, t, rt, rt * 1e-3, MU, csd.CTRL_PI, 2_000_000, rc1, rc2, EJECTION_RADIUS)),
                                 ("RK45", lambda: csd.KR.dp54_endpoint(s, t, rt, rt * 1e-3, MU, csd.CTRL_SCIPY, 2_000_000, rc1, rc2, EJECTION_RADIUS)),
                                 ("DOP853", lambda: csd.KR.dop853_endpoint(s, t, rt, rt * 1e-3, MU, *tb, 2_000_000, rc1, rc2, EJECTION_RADIUS))):
                a = ee.run_with_event(meth, s, t, rt)
                r = ref_fn()
                assert a["ok"] == bool(r[0]) and a["reason"] == int(r[5]) and a["evals"] == r[2], (meth, rt, a["evals"], r[2])
                assert np.allclose(a["final"], np.asarray(r[1]), rtol=0, atol=1e-12)
    assert n_bad > 0, "the doomed rows should produce some bad runs"


def test_event_times_converge_to_the_reference():
    """On doomed orbits a tight run of every method finds the reference's event type, and its Hermite event time agrees with the reference's
    event time to well within the strict rule's window."""
    from orbit_ml_cr3bp import events_e as ee
    rows = _doomed_rows(8, 5)
    if rows is None:
        print("  skipped (no data_c/confirm)")
        return
    for s, t in zip(*rows):
        code, te, v = ee.reference_event(s, t)
        assert code in (1, 2, 3)
        window = (100.0 / DU_KM) / v
        for meth, prm in (("rk4s", 0.01), ("rk8s", 0.05), ("DP54PI", 1e-11), ("DOP853", 1e-11), ("DP54PIs", 1e-11), ("DOP853s", 1e-11)):
            r = ee.run_with_event(meth, s, t, prm)
            assert not r["ok"] and r["event"] == code, (meth, r, code)
            assert abs(r["t_event"] - te) < 0.01 * window, (meth, r["t_event"], te, window)
            assert ee.strict_success(r["event"], r["t_event"], code, te, v)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
