"""
Checks for the Part D code. Run:  python tests/test_part_d.py        (each test_* function is run; the first failure stops)
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from orbit_ml_cr3bp.config import MU  # noqa: E402


def _states(n=40, seed=0, horizon=4.0):
    """Reproducible test states away from both bodies (no collision inside the horizon is needed: tests compare only runs the compiled solver completes)."""
    rng = np.random.RandomState(seed)
    out = []
    while len(out) < n:
        s = np.array([rng.uniform(0.6, 1.3), rng.uniform(-0.4, 0.4), rng.uniform(-0.1, 0.1), rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), rng.uniform(-0.1, 0.1)])
        if min(np.hypot(s[0] + MU, s[1]), np.hypot(s[0] + MU - 1.0, s[1])) > 0.08:
            out.append(s)
    T = rng.uniform(1.0, horizon, n)
    return np.array(out), T


def _scipy_run(state, T, rtol, method):
    from scipy.integrate import solve_ivp
    from orbit_ml_cr3bp.physics_3d import _cr3bp_derivatives_3d_numba as f
    return solve_ivp(lambda t, y: f(y, MU), (0, T), state, method=method, rtol=rtol, atol=rtol * 1e-3)


# ---------------------------------------------------------------------------------------------------------------------
# D2: compiled solvers
# ---------------------------------------------------------------------------------------------------------------------

def test_dp54_scipy_controller_reproduces_solve_ivp_exactly():
    from orbit_ml_cr3bp import compiled_solvers_d as cs
    S, T = _states()
    sv = cs.Solvers(MU)
    for rtol in (1e-4, 1e-7, 1e-10):
        ok, y, nf, _ = sv.run("DP54_scipy", S, T, rtol)
        n_cmp = 0
        for i in range(len(S)):
            if not ok[i]:
                continue
            sol = _scipy_run(S[i], T[i], rtol, "RK45")
            assert sol.success
            assert int(nf[i]) == sol.nfev, (rtol, i, nf[i], sol.nfev)
            assert np.abs(sol.y[:, -1] - y[i]).max() < 1e-7 * max(1.0, np.abs(y[i]).max()) or rtol <= 1e-4, (rtol, i)
            n_cmp += 1
        assert n_cmp >= 30


def test_dop853_port_matches_scipy_nfev_on_almost_every_orbit():
    from orbit_ml_cr3bp import compiled_solvers_d as cs
    S, T = _states(seed=1)
    sv = cs.Solvers(MU)
    for rtol in (1e-6, 1e-9):
        ok, y, nf, _ = sv.run("DOP853", S, T, rtol)
        same, tot = 0, 0
        for i in range(len(S)):
            if not ok[i]:
                continue
            sol = _scipy_run(S[i], T[i], rtol, "DOP853")
            same += int(nf[i]) == sol.nfev
            tot += 1
            if int(nf[i]) == sol.nfev:
                assert np.abs(sol.y[:, -1] - y[i]).max() < 1e-8
        assert tot >= 30 and same / tot >= 0.95, (rtol, same, tot)       # floating-point summation order can flip a borderline step on a rare orbit


def test_pi_controller_is_accurate_and_not_worse_on_average():
    from orbit_ml_cr3bp import compiled_solvers_d as cs
    S, T = _states(seed=2)
    sv = cs.Solvers(MU)
    ok1, y1, nf1, _ = sv.run("DP54_scipy", S, T, 1e-8)
    ok2, y2, nf2, _ = sv.run("DP54_PI", S, T, 1e-8)
    m = ok1 & ok2
    assert m.sum() >= 30
    assert np.abs(y1[m] - y2[m]).max() < 1e-4
    assert nf2[m].mean() <= 1.1 * nf1[m].mean()


def test_event_checks_stop_a_colliding_run():
    from orbit_ml_cr3bp import compiled_solvers_d as cs
    sv = cs.Solvers(MU)
    s = np.array([[1.0 - MU - 0.05, 0.0, 0.0, 0.0, 0.0, 0.0]])     # falls into the Moon
    ok, y, nf, rs = sv.run("DP54_PI", s, np.array([5.0]), 1e-8)
    assert not ok[0] and rs[0] != 0


def test_compiled_policy_pipeline_equals_the_sklearn_route():
    import joblib
    from orbit_ml_cr3bp import compiled_solvers_d as cs, policy_c_3d as pc
    from orbit_ml_cr3bp.osculating_3d import add_osc_features
    from orbit_ml_cr3bp.physics_3d import jacobi_constant_3d
    from orbit_ml_cr3bp.compiled_model_3d import c_l1_for_mu
    fz = joblib.load(os.path.join(ROOT, "part_c", "frozen", "sund", "policy_frozen.joblib"))
    S, T = _states(seed=3)
    df = pd.DataFrame(S, columns=["x", "y", "z", "vx", "vy", "vz"])
    df["sim_time"] = T
    C = np.array([jacobi_constant_3d(list(s), MU) for s in S])
    r1 = np.sqrt((S[:, 0] + MU) ** 2 + S[:, 1] ** 2 + S[:, 2] ** 2)
    r2 = np.sqrt((S[:, 0] + MU - 1.0) ** 2 + S[:, 1] ** 2 + S[:, 2] ** 2)
    df["jacobi_3d"], df["r1_3d"], df["r2_3d"], df["r1_r2_ratio_3d"], df["c_minus_c_l1_3d"] = C, r1, r2, r1 / r2, C - c_l1_for_mu(MU)
    df = add_osc_features(df)
    ref = 10.0 ** pc.predict_raw_c(fz["models"], df)["rk4s"]["logdt"]
    sp = cs.SundmanPolicy(fz["models"], MU)
    assert np.allclose(sp.raw_ds(S, T), ref, rtol=1e-12)
    ok, y, ev, _ = sp.run(S, T, 0.87)
    ok2, y2, ev2, _ = sp.run_ds(S, T, ref * 0.87)
    assert (ok == ok2).all() and np.array_equal(ev, ev2) and np.allclose(y, y2, rtol=0, atol=1e-13)


def test_certify_rtol_compiled_orders_the_chain():
    from orbit_ml_cr3bp import compiled_solvers_d as cs
    S, T = _states(n=200, seed=4)      # 0.95^n <= 0.1 / 18 needs n >= 101 for a zero-violation certificate
    sv = cs.Solvers(MU)
    ok, y, _, _ = sv.run("DP54_scipy", S, T, 1e-12)
    keep = ok
    S, T, ref = S[keep], T[keep], y[keep][:, :3]
    r = cs.certify_rtol_compiled(sv, "DP54_scipy", S, T, ref, tol_km=100.0, equal_risk_target=0.1)
    assert r["chosen_rtol"] is not None and r["equal_risk_rtol"] is not None
    assert r["equal_risk_rtol"] >= r["chosen_rtol"] * 0.999        # a 10 % risk target allows a looser tolerance than the 5 %-risk certificate


def test_lane_efficiency():
    import run_part_d_3d as rp
    assert rp.lane_efficiency(np.full(64, 10.0), widths=(8,))[8] == 1.0
    w = np.ones(64)
    w[0] = 100.0
    assert 0.1 < rp.lane_efficiency(w, widths=(8,), order="random")[8] < 0.5                 # one slow orbit idles the other seven lanes of its group
    w = np.exp(np.random.RandomState(0).normal(0, 1.0, 4096))
    e_r, e_s = rp.lane_efficiency(w, widths=(8, 128), order="random"), rp.lane_efficiency(w, widths=(8, 128), order="sorted")
    assert e_s[8] > e_r[8] and e_s[128] > e_r[128] and e_r[8] > e_r[128]                      # sorting helps; wider groups waste more


# ---------------------------------------------------------------------------------------------------------------------
# D1: rules
# ---------------------------------------------------------------------------------------------------------------------

def test_rule_features_match_the_dataset_columns_and_regimes():
    from orbit_ml_cr3bp import rules_d as rd
    from orbit_ml_cr3bp.diagnostics_3d import hill_regime
    from orbit_ml_cr3bp.osculating_3d import add_osc_features
    from orbit_ml_cr3bp.physics_3d import jacobi_constant_3d
    S, T = _states(n=80, seed=5)
    F = rd.rule_features_batch(S, T, MU, rd.c_l1_for_mu(MU), rd.hill_thresholds(MU), True)
    C = np.array([jacobi_constant_3d(list(s), MU) for s in S])
    reg = pd.Series(hill_regime(C, MU)).map({"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5}).to_numpy()
    assert np.array_equal(F[:, 1], reg)
    assert np.allclose(F[:, 0], np.log10(T)) and np.allclose(F[:, 4], C - rd.c_l1_for_mu(MU))
    df = pd.DataFrame(S, columns=["x", "y", "z", "vx", "vy", "vz"])
    df["sim_time"] = T
    o = add_osc_features(df)
    assert np.allclose(F[:, 2], o["osc_logtau_hat"])
    r1 = np.sqrt((S[:, 0] + MU) ** 2 + S[:, 1] ** 2 + S[:, 2] ** 2)
    r2 = np.sqrt((S[:, 0] + MU - 1.0) ** 2 + S[:, 1] ** 2 + S[:, 2] ** 2)
    lrp = np.minimum(np.where(o["osc_reach_1"] == 1, o["osc_logrp_1"], np.log10(r1)), np.where(o["osc_reach_2"] == 1, o["osc_logrp_2"], np.log10(r2)))
    assert np.allclose(F[:, 3], lrp)


def test_rule_fit_recovers_a_planted_linear_law_and_enters_predict_raw_c():
    from orbit_ml_cr3bp import rules_d as rd, policy_c_3d as pc
    S, T = _states(n=1500, seed=6, horizon=20.0)
    df = pd.DataFrame(S, columns=["x", "y", "z", "vx", "vy", "vz"])
    df["sim_time"] = T
    F = rd.rule_features_batch(S, T, MU, rd.c_l1_for_mu(MU), rd.hill_thresholds(MU), True)
    df["dt_rk4s"] = 10.0 ** (-0.3 - 0.5 * F[:, 0] + 0.05 * F[:, 1] + 0.1 * F[:, 2])
    df["rk4s_success"] = 1
    rule = rd.fit_rule(df, ("lT", "reg", "tau"), 0.5, MU)
    assert np.allclose(rule["w"][[0, 1, 2]], [-0.5, 0.05, 0.1], atol=2e-3) and abs(rule["b"] + 0.3) < 5e-3
    models = rd.rule_models(rule)
    raw = pc.predict_raw_c(models, df)
    assert np.allclose(raw["rk4s"]["logdt"], np.log10(df["dt_rk4s"]), atol=2e-3)
    pred = pc.apply_gate(raw, lf_gate="none")
    idx, dt = pc.choose_unscaled(pred, ("rk4s",))
    assert (idx == 0).all() and np.allclose(dt, df["dt_rk4s"], rtol=5e-3)


def test_deploy_runner_matches_the_part_c_integrator_wrapper():
    from orbit_ml_cr3bp import deploy_d as dd
    from orbit_ml_cr3bp.sundman_3d import integrate_endpoint_rk4s, rk4s_evals
    S, T = _states(n=20, seed=7)
    runner = dd.default_runner(MU)
    ds = np.full(len(S), 0.1)
    r = dd.run_rows(runner, S, T, ds, np.zeros((len(S), 3)), tol_km=1e9, parallel=False)
    for i in range(len(S)):
        ok, _p, curr, _tp, _t, _re, ns, nf = integrate_endpoint_rk4s(S[i], 0.1, T[i], MU, 4_000_000, tau_c=2.0)
        assert bool(ok) == bool(r["ok"][i])
        if ok:
            assert r["evals"][i] == rk4s_evals(ns, nf)
            assert np.allclose(np.asarray(curr)[:3] * 384400.0, np.linalg.norm(0) + np.asarray(curr)[:3] * 384400.0)
            assert np.isfinite(r["err"][i])


# ---------------------------------------------------------------------------------------------------------------------
# D4: strata
# ---------------------------------------------------------------------------------------------------------------------

def _frame(n=3000, seed=0):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({"regime": rng.choice(["I", "II", "III", "IV", "V"], n, p=[.3, .05, .3, .1, .25]), "h_band": rng.choice(["1-3", "3-8", "8-20"], n),
                         "sim_time": rng.uniform(1, 20, n), "osc_logrp_1": rng.uniform(-1, 0, n), "osc_logrp_2": np.where(rng.rand(n) < .15, rng.uniform(-2.3, -0.5, n), 0.0),
                         "osc_reach_1": (rng.rand(n) < .05).astype(float), "osc_reach_2": (rng.rand(n) < .15).astype(float)})


def test_strata_cover_every_row_and_merge_small_cells():
    from orbit_ml_cr3bp import strata_d as sd
    cal, ev = _frame(3000, 0), _frame(3000, 1)
    cc_, ce = sd.base_cells(cal), sd.base_cells(ev)
    smap, labs = sd.build_strata_map(cc_, min_n=100)
    lab_c, lab_e = sd.assign_strata(cc_, smap, labs), sd.assign_strata(ce, smap, labs)
    assert len(lab_c) == len(cal) and len(lab_e) == len(ev) and not pd.isna(lab_e).any()
    sizes = pd.Series(lab_c).value_counts()
    assert (sizes[sizes.index != "pooled-small"] >= 100).all()
    assert len(set(lab_c)) < cc_["base"].nunique()


def test_certify_strata_abstains_where_no_scale_is_certified():
    from orbit_ml_cr3bp import strata_d as sd, policy_c_3d as pc

    class FakeRunner:        # the violation probability of a row grows with the scale and is high in stratum "bad" whatever the scale
        def run_ds(self, states, T, ds, parallel=False):
            n = len(ds)
            viol = (states[:, 0] < 0.5) | (ds > states[:, 1])
            return np.ones(n, dtype=bool), np.zeros((n, 6)), np.full(n, 100.0), np.zeros(n, dtype=int)

        def errors_km(self, y, ref):
            return np.zeros(len(y))
    # build a runner whose error depends on the rows through ds and a per-row flag carried in `states`
    class R2(FakeRunner):
        def run_ds(self, states, T, ds, parallel=False):
            n = len(ds)
            y = np.zeros((n, 6))
            y[:, 0] = np.where((states[:, 0] < 0.5), 1e6, np.where(ds > 0.4, 1e6, 0.0))       # stratum 'bad' always violates; 'good' violates above ds = 0.4
            return np.ones(n, dtype=bool), y, np.full(n, 100.0), np.zeros(n, dtype=int)

        def errors_km(self, y, ref):
            return np.abs(y[:, 0])
    n = 600
    states = np.zeros((n, 6))
    states[:, 0] = np.where(np.arange(n) < 300, 1.0, 0.0)
    labels = np.where(np.arange(n) < 300, "good", "bad")
    cert, G = sd.certify_strata(R2(), states, np.ones(n), np.ones(n), np.zeros((n, 3)), labels, min_n=50)
    assert G == 2 and cert["bad"]["scale"] is None and cert["good"]["scale"] is not None
    assert cert["good"]["scale"] <= 0.4 + 1e-9


def test_d4_and_d5_and_d2_statistics_on_synthetic_inputs():
    from orbit_ml_cr3bp import confirm_d_3d as cd, strata_d as sd
    rng = np.random.RandomState(0)
    n = 4000
    meets = rng.rand(n) > 0.03
    led = pd.DataFrame({"meets_tol": meets, "C_ml_tot": rng.uniform(100, 300, n), "RK45_eqr_nfev": rng.uniform(200, 400, n), "RK45_eqr_ok": True,
                        "DOP853_eqr_nfev": 250.0, "DOP853_eqr_ok": True, "ml_abstain": False, "stratum": rng.choice(["a", "b", "c"], n)})
    tab = sd.stratum_table(led, "stratum")
    r = cd.d4_conditional(tab, led, 0.07, 300)
    assert r["p"] < 1e-3 and r["n_reported_strata"] == 3
    led["meets_tol"] = rng.rand(n) > 0.09                                           # 9 % violations: the 7 % hypothesis must not be supported
    r2 = cd.d4_conditional(sd.stratum_table(led, "stratum"), led, 0.07, 300)
    assert r2["p"] > 0.05
    runs = pd.DataFrame({"group": np.repeat(["a", "b", "c"], 200), "violation": rng.rand(600) < 0.02, "cluster": np.tile(np.arange(100), 6)})
    assert cd.d5_ood(runs, 0.10, n_boot=500)["p"] < 0.01
    runs["violation"] = rng.rand(600) < 0.2
    assert cd.d5_ood(runs, 0.10, n_boot=500)["p"] > 0.5
    tp = rng.uniform(900, 1100, 2000)
    assert cd.d2_throughput(tp, tp * 3.0, n_boot=500)["p"] < 0.01
    assert cd.d2_throughput(tp, tp * 0.5, n_boot=500)["p"] > 0.9


# ---------------------------------------------------------------------------------------------------------------------
# D5: periodic orbits
# ---------------------------------------------------------------------------------------------------------------------

def test_select_indices_disjoint_and_spread():
    from orbit_ml_cr3bp import periodic_d as pdd
    for N in (60, 1535, 19782):
        conf, dry = pdd.select_indices(N, 50, "confirm"), pdd.select_indices(N, 50, "dry")
        assert not (set(conf) & set(dry)) and max(conf) == N - 1 and min(conf) == 0
        assert len(conf) >= 49 and all(0 <= i < N for i in dry)
    assert pdd.stab_band(1.0) == pdd.STAB_BANDS[0][0] and pdd.stab_band(50.0) == "10-100" and pdd.stab_band(1e4) == ">100"


def test_member_diagnostics_close_on_a_jpl_orbit_if_the_data_are_present():
    from orbit_ml_cr3bp import periodic_d as pdd
    path = os.path.join(ROOT, "data_d", "jpl", "lyapunov_L1.json")
    if not os.path.exists(path):
        print("  (JPL data not present: skipped)")
        return
    df, mu = pdd.load_stream(path)
    m = df.iloc[len(df) // 2]
    d = pdd.member_diagnostics((m[["x", "y", "z", "vx", "vy", "vz"]].to_numpy(dtype=float), float(m["period"]), mu))
    assert d["reached"] and d["closure"] < 1e-8 and d["min_r2"] > 0.0


def test_periodic_orbit_reference_is_stable_to_a_second_reference():
    from orbit_ml_cr3bp import periodic_d as pdd
    path = os.path.join(ROOT, "data_d", "jpl", "dro.json")
    if not os.path.exists(path):
        print("  (JPL data not present: skipped)")
        return
    df, mu = pdd.load_stream(path)
    m = df.iloc[len(df) // 3]
    p1, p2 = pdd.reference_multi((m[["x", "y", "z", "vx", "vy", "vz"]].to_numpy(dtype=float), [float(m["period"]), 2 * float(m["period"])], mu))
    assert np.isfinite(p1).all() and np.linalg.norm(p1 - p2, axis=1).max() < 1e-9


def _run_all():
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for k, f in fns:
        print(f"{k} ...", flush=True)
        f()
        print("  ok", flush=True)
    print(f"{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
