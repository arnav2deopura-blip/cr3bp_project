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
            a = ee.run_with_event("rk4s", s, t, ds, seg=False)
            r = integrate_endpoint_rk4s(s, ds, t, MU, 4_000_000)
            assert a["ok"] == bool(r[0]) and a["reason"] == int(r[5]) and a["evals"] == 4 * (r[6] + r[7])
            assert np.array_equal(a["final"], np.asarray(r[2]))
            b = ee.run_with_event("rk8s", s, t, ds * 4, seg=False)
            r = dr.integrate_endpoint_rk8s(s, ds * 4, t, MU)
            assert b["ok"] == bool(r[0]) and b["reason"] == int(r[5]) and b["evals"] == 12 * (r[6] + r[7])
            assert np.array_equal(b["final"], np.asarray(r[2]))
            n_bad += int(not a["ok"])
        for rt in (1e-6, 1e-10):
            for meth, ref_fn in (("DP54PI", lambda: csd.KR.dp54_endpoint(s, t, rt, rt * 1e-3, MU, csd.CTRL_PI, 2_000_000, rc1, rc2, EJECTION_RADIUS)),
                                 ("RK45", lambda: csd.KR.dp54_endpoint(s, t, rt, rt * 1e-3, MU, csd.CTRL_SCIPY, 2_000_000, rc1, rc2, EJECTION_RADIUS)),
                                 ("DOP853", lambda: csd.KR.dop853_endpoint(s, t, rt, rt * 1e-3, MU, *tb, 2_000_000, rc1, rc2, EJECTION_RADIUS))):
                a = ee.run_with_event(meth, s, t, rt, seg=False)
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


def test_segment_check_in_the_kernels():
    """Item 7: the compiled chord test agrees with collision_segment.py; a run whose step jumps across the Moon is flagged only with seg = True;
    with seg = True nothing changes on runs that never pass near a body."""
    from orbit_ml_cr3bp import events_e as ee, compiled_sundman_e as cs
    from orbit_ml_cr3bp.collision_segment import crossed_bodies
    from orbit_ml_cr3bp.config import collision_radii
    rc1, rc2 = collision_radii(MU)
    rng = np.random.RandomState(0)
    moon, earth = np.array([1.0 - MU, 0, 0]), np.array([-MU, 0, 0])
    for _ in range(400):
        c = moon if rng.rand() < 0.5 else earth
        a = np.r_[c + rng.normal(scale=0.03, size=3), np.zeros(3)]
        b = np.r_[c + rng.normal(scale=0.03, size=3), np.zeros(3)]
        r = crossed_bodies(np.vstack([a[:3], b[:3]]), MU)
        want = 1 if r["primary"] == 0 else (2 if r["secondary"] == 0 else 0)
        assert ee.seg_hit(a, b, MU, rc1, rc2) == want == cs._seg_hit(a, b, MU, rc1, rc2)
    # a straight fast pass through the Moon: one long physical step (fixed-step RK8 landing with a huge ds) jumps across it
    s0 = np.array([1.0 - MU - 0.05, 0.0005, 0.0, 2.0, 0.0, 0.0])
    T = 0.05
    for meth, prm in (("DP54PI", 1e-1), ("rk8s", 50.0)):          # both step across the Moon: the endpoint check alone misses it
        plain = ee.run_with_event(meth, s0, T, prm, seg=False)
        checked = ee.run_with_event(meth, s0, T, prm, seg=True)
        assert plain["ok"], meth
        assert not checked["ok"] and checked["event"] == 2 and checked["seg_hit"] and 0.0 < checked["t_event"] < T, (meth, checked)
    S, Tt = _real_rows(6, 7) if _real_rows() is not None else (STATES, HORIZONS)
    for s, t in zip(S, Tt):
        for meth, prm in (("rk4s", 0.1), ("rk8s", 0.5), ("DP54PI", 1e-8), ("DOP853s", 1e-8)):
            a, b = ee.run_with_event(meth, s, t, prm, seg=False), ee.run_with_event(meth, s, t, prm, seg=True)
            if a["ok"] and b["ok"]:
                assert np.array_equal(a["final"], b["final"]) and a["evals"] == b["evals"]
            elif b["seg_hit"]:
                assert a["ok"] or a["evals"] >= b["evals"]


def test_checkpoints_path_error():
    """Item 3: recording checkpoints changes no run; a checkpoint at a node time returns that node (endpoint_audit_3d's node list); the last
    checkpoint is the final position; a tight run has a small path error and a loose one a path error at least its final error."""
    from orbit_ml_cr3bp import events_e as ee
    from orbit_ml_cr3bp.endpoint_audit_3d import trajectory_rk4s
    S, T = _real_rows(5, 8) if _real_rows() is not None else (STATES, HORIZONS)
    for s, t in zip(S, T):
        ts, xs = trajectory_rk4s(s, 0.2, t, MU)
        nodes = ts[1:-1][:: max(1, (len(ts) - 2) // 10)]
        r = ee.run_with_event("rk4s", s, t, 0.2, seg=False, checkpoints=nodes)
        idx = np.searchsorted(ts, nodes)
        assert np.allclose(r["ck_pos"], xs[idx, :3], rtol=0, atol=1e-13)
        ck = ee.checkpoint_times(t)
        ref = ee.reference_checkpoints(s, t)
        for meth, prm in (("rk4s", 0.2), ("rk8s", 0.5), ("DP54PI", 1e-7), ("DOP853", 1e-7), ("DP54PIs", 1e-7), ("DOP853s", 1e-7)):
            a = ee.run_with_event(meth, s, t, prm, seg=False)
            b = ee.run_with_event(meth, s, t, prm, seg=False, checkpoints=ck)
            assert a["ok"] == b["ok"] and a["evals"] == b["evals"] and np.array_equal(a["final"], b["final"])
            if b["ok"]:
                assert np.allclose(b["ck_pos"][-1], b["final"][:3], rtol=0, atol=1e-13), meth
                fin = float(np.linalg.norm(b["final"][:3] - ref[-1])) * DU_KM
                pe = ee.path_error_km(b["ck_pos"], ref)
                assert pe >= fin - 1e-9, meth
        tight = ee.run_with_event("DOP853", s, t, 1e-12, seg=False, checkpoints=ck)
        assert ee.path_error_km(tight["ck_pos"], ref) < 1.0


def test_rows_with_event_matches_single_runs():
    from multiprocessing import Pool
    from orbit_ml_cr3bp import events_e as ee
    S, T = _mixed_rows()
    ref = np.array([_ref(s, t) for s, t in zip(S, T)])
    rck = ee.reference_checkpoints_rows(S, T)
    with Pool(2) as pool:
        a = ee.rows_with_event("rk4s", S, T, 0.1, ref, rck, pool=pool)
    b = ee.rows_with_event("rk4s", S, T, 0.1, ref, rck)
    for k in a:
        assert np.array_equal(a[k], b[k], equal_nan=True) if a[k].dtype.kind == "f" else np.array_equal(a[k], b[k]), k
    for i in range(len(S)):
        r = ee.run_with_event("rk4s", S[i], T[i], 0.1)
        assert r["ok"] == a["ok"][i] and r["evals"] == a["evals"][i] and r["event"] == a["event"][i]
    assert np.all(a["path"][a["ok"] & np.isfinite(rck[:, 0, 0])] >= a["err"][a["ok"] & np.isfinite(rck[:, 0, 0])] - 1e-9)


def test_tolerance_relabelling_keeps_the_states():
    """Item 2: the same seed gives the same initial states at every tolerance and under a restricted horizon range (gen_part_e.sh relies on
    this for E_tol and E_confirm); the horizon range is respected; leaving it out reproduces the earlier generator."""
    from orbit_ml_cr3bp.data_3d import generate_one_3d_training_orbit as gen
    cols = ["x", "y", "z", "vx", "vy", "vz", "sampling_branch"]
    for idx in range(4):
        kw = dict(mu=MU, base_seed=999, label_mode="accuracy_monotone", pipeline="v6", tau_c=2.0)
        a = gen(idx, tol_km=100.0, **kw)["row"]
        b = gen(idx, tol_km=10.0, **kw)["row"]
        c = gen(idx, tol_km=100.0, sim_time_range=(1.0, 3.0), **kw)["row"]
        d = gen(idx, tol_km=100.0, sim_time_range=None, **kw)["row"]
        assert all(a[k] == b[k] == c[k] for k in cols) and a["sim_time"] == b["sim_time"]
        assert 1.0 <= c["sim_time"] <= 3.0 and 1.0 <= a["sim_time"] <= 20.0
        assert {k: v for k, v in a.items() if k == k} .keys() == d.keys() and all((a[k] == d[k]) or (a[k] != a[k] and d[k] != d[k]) for k in a)


def test_ood_gate():
    """Item 5: the split-conformal rank, the gate on synthetic data, the third JPL sample disjoint from Study D's two, the gated violation."""
    from orbit_ml_cr3bp import ood_e as oe
    from orbit_ml_cr3bp.periodic_d import select_indices
    sc = np.arange(1, 20, dtype=float)                       # n = 19: rank ceil(20 * 0.95) = 19
    assert oe.conformal_threshold(sc, 0.05) == 19.0 and oe.conformal_threshold(sc[:10], 0.05) == np.inf
    rng = np.random.RandomState(0)
    tr = pd.DataFrame(rng.randn(3000, 3), columns=list("abc"))
    cal = pd.DataFrame(rng.randn(2000, 3), columns=list("abc"))
    g = oe.KnnGate(k=10, alpha=0.05).fit(tr, list("abc"))
    g.calibrate(cal)
    new = pd.DataFrame(rng.randn(4000, 3), columns=list("abc"))
    assert abs(g.abstain(new).mean() - 0.05) < 0.015
    assert g.abstain(pd.DataFrame([[8.0, 8.0, 8.0]], columns=list("abc")))[0]
    for n_total in (1535, 2198, 5731, 19782):              # the smallest and some larger JPL stream sizes
        e = set(oe.select_indices_e(n_total, 50))
        assert not e & set(select_indices(n_total, 50, "confirm")) and not e & set(select_indices(n_total, 50, "dry"))
        assert len(e) == 50
    v = oe.gated_violations([True, False, True], [True, True, False], [False, False, True])
    assert v.tolist() == [False, True, True]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
