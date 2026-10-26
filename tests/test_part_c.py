"""
Checks for the Part C code. Run:  python tests/test_part_c.py        (each test_* function is run; the first failure stops)
Tests are added with the step that introduces the code they cover (see part_c/PART_C_LOG.md).
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from orbit_ml_cr3bp import failure_study_3d as fs  # noqa: E402


# ---------------------------------------------------------------------------------------------------------------------
# C0: lever surface
# ---------------------------------------------------------------------------------------------------------------------

def _synthetic_ledger(n=4000, seed=0):
    rng = np.random.RandomState(seed)
    c_int = np.exp(rng.normal(np.log(1500.0), 0.8, n))
    c_ad = np.exp(rng.normal(np.log(300.0), 0.6, n))
    meets = rng.rand(n) > 0.04
    return pd.DataFrame({"meets_tol": meets, "C_ml_int": c_int, "RK45_cal_nfev": c_ad})


def test_lever_threshold_is_the_exact_break_even():
    led = _synthetic_ledger()
    for O in (0.0, 10.0, 50.0):
        thr = fs.lever_thresholds(led, "RK45_cal", O)
        ok = led["meets_tol"]
        # just above the threshold the ML policy is cheaper, just below it is not (orbits with C_ad > O)
        fin = ok & np.isfinite(thr)
        above = led.loc[fin, "C_ml_int"] * 10 ** (-(thr[fin] + 1e-9)) + O
        below = led.loc[fin, "C_ml_int"] * 10 ** (-(thr[fin] - 1e-9)) + O
        assert (above < led.loc[fin, "RK45_cal_nfev"]).all()
        assert (below >= led.loc[fin, "RK45_cal_nfev"]).all()
        # orbits where C_ad <= O can never win
        assert np.isinf(thr[ok & (led["RK45_cal_nfev"] <= O)]).all()
        # violations carry no threshold
        assert thr[~ok].isna().all()


def test_lever_share_matches_direct_count_and_is_monotone():
    led = _synthetic_ledger()
    surf = fs.lever_surface(led, comparators=("RK45_cal",), O_grid=(0.0, 25.0), delta_grid=(0.0, 0.2, 0.5, 1.0))
    ok = led["meets_tol"]
    for _, r in surf.iterrows():
        cost = led.loc[ok, "C_ml_int"] * 10 ** (-r["delta_dex"]) + r["O"]
        direct = float((cost < led.loc[ok, "RK45_cal_nfev"]).mean())
        assert math.isclose(r["share"], direct, abs_tol=1e-12), (r["O"], r["delta_dex"], r["share"], direct)
        assert r["lo"] <= r["share"] <= r["hi"]
    for O, g in surf.groupby("O"):
        assert (np.diff(g.sort_values("delta_dex")["share"].to_numpy()) >= 0).all()      # more reduction never hurts
    s0 = surf[(surf["delta_dex"] == 0.0)].sort_values("O")["share"].to_numpy()
    assert s0[0] >= s0[1]                                                                # more overhead never helps


def test_lever_isoline_reaches_the_target_share():
    led = _synthetic_ledger()
    iso = fs.lever_isoline(led, comparators=("RK45_cal",), O_grid=(0.0, 50.0), n_boot=50)
    for _, r in iso.iterrows():
        if np.isfinite(r["delta_needed_dex"]):
            s = fs.lever_surface(led, comparators=("RK45_cal",), O_grid=(r["O"],), delta_grid=(r["delta_needed_dex"] + 1e-9,))
            assert s["share"].iloc[0] >= 0.5
            s = fs.lever_surface(led, comparators=("RK45_cal",), O_grid=(r["O"],), delta_grid=(r["delta_needed_dex"] - 1e-6,))
            assert s["share"].iloc[0] < 0.5
    # unreachable when more than half of the orbits have C_ad <= O
    iso = fs.lever_isoline(led, comparators=("RK45_cal",), O_grid=(1e6,), n_boot=20)
    assert np.isinf(iso["delta_needed_dex"].iloc[0])


# ---------------------------------------------------------------------------------------------------------------------
# C1: end-to-end certification and the leapfrog gate
# ---------------------------------------------------------------------------------------------------------------------

def _risk_curve(grid, lo=0.01, hi=0.15):
    """True violation risk, increasing in the scale (a monotone curve, the case the fixed sequence is built for)."""
    g = np.log(np.asarray(sorted(grid)))
    return np.sort(lo + (hi - lo) * (g - g.min()) / (g.max() - g.min()))


def test_fixed_sequence_false_certification_rate_is_at_most_delta():
    from orbit_ml_cr3bp import policy_c_3d as pc
    grid = pc.FIXED_SEQ_GRID
    assert len(grid) == 28 and abs(grid[0] - 0.1) < 1e-12 and abs(grid[26] - 0.1 * 2 ** (26 / 8)) < 1e-12 and grid[-1] == 1.0
    rng = np.random.RandomState(1)
    n, reps, alpha, delta = 1000, 10_000, 0.05, 0.1
    for lo, hi in ((0.01, 0.15), (0.02, 0.08), (0.049, 0.06)):
        risk = dict(zip(sorted(grid), _risk_curve(grid, lo, hi)))
        for rule in ("fixed_sequence", "bonferroni"):
            false_cert = 0
            ks = {s: rng.binomial(n, risk[s], size=reps) for s in grid}
            for r in range(reps):
                res = pc.certify_sequence(lambda s: (int(ks[s][r]), n), grid, alpha, delta, rule)
                c = res["chosen_scale"]
                false_cert += (c is not None and risk[c] > alpha)
            rate = false_cert / reps
            slack = 3 * math.sqrt(delta * (1 - delta) / reps)
            assert rate <= delta + slack, (rule, lo, hi, rate)


def test_fixed_sequence_stops_at_first_non_rejection():
    from orbit_ml_cr3bp import policy_c_3d as pc
    grid = (0.1, 0.2, 0.3, 0.4)
    # violations: safe, safe, too many, safe again -> the later safe value must NOT be certified
    ks = {0.1: 0, 0.2: 0, 0.3: 40, 0.4: 0}
    res = pc.certify_sequence(lambda s: (ks[s], 400), grid, 0.05, 0.1, "fixed_sequence")
    assert res["chosen_scale"] == 0.2 and len(res["chain"]) == 3
    res = pc.certify_sequence(lambda s: (ks[s], 400), grid, 0.05, 0.1, "bonferroni")
    assert res["chosen_scale"] == 0.4          # Bonferroni tests each value alone (Part B's behaviour)


def test_choice_does_not_depend_on_a_common_scale_and_matches_part_b():
    from orbit_ml_cr3bp import policy_c_3d as pc
    from orbit_ml_cr3bp import policy_b_3d as pb
    rng = np.random.RandomState(3)
    n = 400
    pred = {"rk4": {"feasible": rng.rand(n) > 0.02, "dt": 10 ** rng.uniform(-3, -0.5, n)},
            "leapfrog": {"feasible": rng.rand(n) > 0.3, "dt": 10 ** rng.uniform(-4, -1.5, n)}}
    idx, raw = pc.choose_unscaled(pred)
    for s in (0.1, 0.37, 0.8, 1.0):
        for i in range(n):
            ch, r, used = pb.policy_choice(pred, {"rk4": s, "leapfrog": s}, i)
            if idx[i] < 0:
                assert ch is None
            else:
                assert ch == pc.CANDIDATES[idx[i]] and math.isclose(r, raw[i], rel_tol=1e-12) and math.isclose(used, s * raw[i], rel_tol=1e-12)
    # the evaluation view makes the choice unique
    view = pc.policy_pred_for_evaluation(pred, idx)
    for i in range(0, n, 7):
        ch, r, used = pb.policy_choice(view, {"rk4": 0.4, "leapfrog": 0.4}, i)
        assert (ch is None) == (idx[i] < 0)


def test_end_to_end_risk_equals_realized_risk_of_the_chosen_runs():
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    from test_failure_study import _mini_dataset
    from orbit_ml_cr3bp import policy_c_3d as pc
    from orbit_ml_cr3bp import policy_b_3d as pb
    d = _mini_dataset()
    n = len(d)
    train, cal = d.iloc[: n // 2].reset_index(drop=True), d.iloc[n // 2:].reset_index(drop=True)
    models = pc.fit_policy_c(train, pb.FEATS_STATE, quantile=0.1, lf_gate="clf", seed=1)
    pred = pc.predict_policy_c(models, cal, p_star=0.5)
    ref = pc.reference_final_positions(cal, n_jobs=4)
    grid = (0.2, 0.4, 0.6, 0.8, 1.0)
    rep = pc.certify_end_to_end(cal, pred, ref, grid=grid, rule="bonferroni", n_jobs=4, delta=0.9)    # lenient delta: we test counting, not power
    s = rep["chosen_scale"]
    assert s is not None
    idx, raw = pc.choose_unscaled(pred)
    view = pc.policy_pred_for_evaluation(pred, idx)
    res = pb.evaluate_orbits(cal, view, {"rk4": s, "leapfrog": s}, rtols={}, do_spec=False, n_jobs=4, progress=False)
    has = res["ml_integrator"].notna().to_numpy()
    ok = res["ml_ok"].fillna(False).astype(bool).to_numpy() & (res["ml_err_km"].to_numpy() <= 100.0)
    realized = int((has & ~ok).sum())
    assert realized == rep["violations"], (realized, rep["violations"])
    assert int(has.sum()) == rep["n_policy_runs"]
    assert sum(v["violations"] for v in rep["conditional_risk"].values()) == rep["violations"]
    # ml_scale in the evaluation is the common scale for every chosen orbit
    assert np.allclose(res.loc[has, "ml_scale"], s)


def test_gate_threshold_search_meets_its_budget():
    from orbit_ml_cr3bp import policy_c_3d as pc
    rng = np.random.RandomState(5)
    n = 3000
    success = rng.rand(n) > 0.3
    # a calibrated-ish score: high where the label says feasible
    score = np.clip(np.where(success, rng.beta(5, 2, n), rng.beta(2, 5, n)), 0, 1)
    raw = {"rk4": {"logdt": np.full(n, -2.0), "score": np.full(n, np.inf)},
           "leapfrog": {"logdt": np.full(n, -1.0), "score": score}}          # leapfrog always looks 4x cheaper -> chosen when allowed
    sel = pd.DataFrame({"rk4_success": np.ones(n, dtype=int), "leapfrog_success": success.astype(int)})
    thr, tab = pc.select_gate_threshold(raw, sel, max_rate=0.005, lf_gate="clf")
    assert thr is not None
    rate, _ = pc.infeasible_choice_rate(raw, sel, p_star=thr)
    assert rate <= 0.005
    below = tab[tab["threshold"] < thr]
    assert (below["infeasible_choice_rate"] > 0.005).all()                  # the smallest threshold that meets the budget


# ---------------------------------------------------------------------------------------------------------------------
# C2: osculating periapsis features
# ---------------------------------------------------------------------------------------------------------------------

def _kepler_periapsis(r0, v0, GM, t_max=400.0):
    """Next periapsis (radius, time) of the two-body orbit by numerical integration of (r, v) inertial, event r.v = 0 from below."""
    from scipy.integrate import solve_ivp

    def f(t, y):
        r = y[:3]
        return np.concatenate([y[3:], -GM * r / np.linalg.norm(r) ** 3])

    def ev(t, y):
        return float(np.dot(y[:3], y[3:]))
    ev.terminal = True
    ev.direction = 1.0
    sol = solve_ivp(f, (0.0, t_max), np.concatenate([r0, v0]), method="DOP853", rtol=1e-13, atol=1e-14, events=ev)
    if sol.t_events[0].size == 0:
        return None, None
    return float(np.linalg.norm(sol.y_events[0][0][:3])), float(sol.t_events[0][0])


def _rot_state_from_inertial(r_rel, v_in, mu):
    """Rotating-frame state of a particle at position r_rel from the PRIMARY with inertial velocity v_in (omega = z-hat)."""
    x, y, z = r_rel[0] - mu, r_rel[1], r_rel[2]
    return np.array([x, y, z, v_in[0] + r_rel[1], v_in[1] - r_rel[0], v_in[2]])


def test_osculating_periapsis_matches_a_two_body_integration():
    from orbit_ml_cr3bp.osculating_3d import osc_features, OSC_FEATURE_COLS
    mu = 1e-11
    GM = 1.0 - mu
    ix = {c: i for i, c in enumerate(OSC_FEATURE_COLS)}
    cases = {
        "elliptic approaching": ([0.8, 0.2, 0.1], [-0.25, 0.35, 0.05]),
        "elliptic receding": ([0.5, -0.4, 0.2], [0.5, 0.6, -0.1]),
        "elliptic nearly radial": ([0.7, 0.1, 0.0], [-0.55, 0.12, 0.02]),
        "hyperbolic approaching": ([1.2, 0.3, -0.2], [-1.6, 0.3, 0.1]),
    }
    for name, (r0, v0) in cases.items():
        r0, v0 = np.array(r0), np.array(v0)
        rp_num, tp_num = _kepler_periapsis(r0, v0, GM)
        assert rp_num is not None, name
        T = 400.0
        f, fl = osc_features(_rot_state_from_inertial(r0, v0, mu)[None, :], np.array([T]), mu)
        rp, tp = 10 ** f[0, ix["osc_logrp_1"]], f[0, ix["osc_tpn_1"]] * T
        assert fl[0, 0] == 0
        assert abs(rp - rp_num) <= 1e-8 * rp_num, (name, rp, rp_num)
        assert abs(tp - tp_num) <= 1e-7 * max(tp_num, 1e-3) + 1e-9, (name, tp, tp_num)
        assert f[0, ix["osc_reach_1"]] == 1.0


def test_osculating_periapsis_in_the_cr3bp_at_tiny_mu():
    """The transform from the rotating frame is exercised end to end: integrate the CR3BP (mu = 1e-11) and watch |r1| reach its minimum."""
    from scipy.integrate import solve_ivp
    from orbit_ml_cr3bp.osculating_3d import osc_features, OSC_FEATURE_COLS
    from orbit_ml_cr3bp.physics_3d import _cr3bp_derivatives_3d_numba
    mu = 1e-11
    ix = {c: i for i, c in enumerate(OSC_FEATURE_COLS)}
    st = _rot_state_from_inertial(np.array([0.8, 0.2, 0.1]), np.array([-0.25, 0.35, 0.05]), mu)

    def f(t, y):
        return _cr3bp_derivatives_3d_numba(y, mu)

    def ev(t, y):                       # d|r1|/dt = 0 going from negative to positive
        r = y[:3] - np.array([-mu, 0.0, 0.0])
        return float(np.dot(r, y[3:6]))
    ev.terminal = True
    ev.direction = 1.0
    sol = solve_ivp(f, (0.0, 200.0), st, method="DOP853", rtol=1e-13, atol=1e-14, events=ev)
    rp_num, tp_num = np.linalg.norm(sol.y_events[0][0][:3] - np.array([-mu, 0, 0])), sol.t_events[0][0]
    # the rotating-frame r.v differs from the inertial r.v by omega . (r x r) = 0, so the event is the same periapsis
    T = 200.0
    feat, _ = osc_features(st[None, :], np.array([T]), mu)
    assert abs(10 ** feat[0, ix["osc_logrp_1"]] - rp_num) <= 1e-7 * rp_num
    assert abs(feat[0, ix["osc_tpn_1"]] * T - tp_num) <= 1e-6 * tp_num


def test_osculating_receding_hyperbola_has_no_future_periapsis_and_circular_is_flagged():
    from orbit_ml_cr3bp.osculating_3d import osc_features, OSC_FEATURE_COLS
    mu = 1e-11
    ix = {c: i for i, c in enumerate(OSC_FEATURE_COLS)}
    st = _rot_state_from_inertial(np.array([1.5, 0.2, 0.1]), np.array([1.4, 0.3, 0.0]), mu)       # fast and outward
    f, fl = osc_features(st[None, :], np.array([10.0]), mu)
    assert f[0, ix["osc_tpn_1"]] == 2.0 and f[0, ix["osc_reach_1"]] == 0.0 and f[0, ix["osc_ecc_1"]] > 1.0
    # a circular inertial orbit at r = 1 is stationary in the rotating frame (n = omega = 1): rotating velocity 0 -> e ~ 0, flag 1
    st = _rot_state_from_inertial(np.array([0.0, 1.0, 0.0]), np.array([-1.0, 0.0, 0.0]), mu)
    assert np.allclose(st[3:], 0.0, atol=1e-12)
    f, fl = osc_features(st[None, :], np.array([10.0]), mu)
    assert fl[0, 0] == 1 and f[0, ix["osc_ecc_1"]] < 1e-8 and abs(10 ** f[0, ix["osc_logrp_1"]] - 1.0) < 1e-8


def test_osculating_features_are_finite_on_random_states():
    from orbit_ml_cr3bp.osculating_3d import osc_features
    rng = np.random.RandomState(0)
    n = 5000
    st = np.hstack([rng.uniform(-1.5, 1.5, (n, 3)), rng.uniform(-2, 2, (n, 3))])
    f, fl = osc_features(st, rng.uniform(1, 20, n), 0.01215058535)
    assert np.isfinite(f).all() and np.isin(fl, (0, 1, 2)).all()


def main():
    names = [k for k in sorted(globals()) if k.startswith("test_")]
    for k in names:
        globals()[k]()
        print("ok  ", k)
    print(f"{len(names)} passed")


if __name__ == "__main__":
    main()
