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


# ---------------------------------------------------------------------------------------------------------------------
# C3: compiled models
# ---------------------------------------------------------------------------------------------------------------------

def _random_policy_frame(n, seed, mu=0.01215058535):
    """States in the study's box with the state features of the real pipeline and synthetic labels (no integration)."""
    from orbit_ml_cr3bp.physics_3d import add_orbit_features_3d
    rng = np.random.RandomState(seed)
    st = np.hstack([rng.uniform(-1.3, 1.3, (n, 2)), rng.uniform(-0.3, 0.3, (n, 1)), rng.uniform(-1.2, 1.2, (n, 3))])
    df = pd.DataFrame(st, columns=["x", "y", "z", "vx", "vy", "vz"])
    df["sim_time"] = np.exp(rng.uniform(0, math.log(20.0), n))
    df = add_orbit_features_3d(df, mu)
    from orbit_ml_cr3bp.osculating_3d import add_osc_features
    df = add_osc_features(df, mu)
    base = -1.2 + 0.6 * np.tanh(df["r2_3d"].to_numpy()) - 0.15 * np.log10(df["sim_time"].to_numpy()) + 0.2 * df["osc_logtau_hat"].to_numpy()
    df["dt_rk4"] = 10 ** (base + rng.normal(0, 0.15, n))
    df["rk4_success"] = (rng.rand(n) > 0.0001).astype(int)
    df["dt_leapfrog"] = 10 ** (base - 0.5 + rng.normal(0, 0.25, n))
    df["leapfrog_success"] = (rng.rand(n) > 0.25 + 0.2 * (df["r2_3d"].to_numpy() < 0.3)).astype(int)
    return df


def test_compiled_hgb_equals_sklearn_predict():
    from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
    from orbit_ml_cr3bp import compiled_model_3d as cm
    from orbit_ml_cr3bp.policy_b_3d import FEATS_STATE
    df = _random_policy_frame(6000, 11)
    X = df[FEATS_STATE].to_numpy(dtype=float)
    y = np.log10(df["dt_rk4"].to_numpy())
    for kw in ({"max_iter": 25, "max_depth": 3}, {"max_iter": 100, "max_depth": 5}, {"max_iter": 300, "learning_rate": 0.05}):
        reg = HistGradientBoostingRegressor(loss="quantile", quantile=0.1, random_state=0, **kw).fit(X[:5000], y[:5000])
        f = cm.export_forest(reg)
        Xt = np.tile(X, (2, 1))[:10_000]
        assert np.max(np.abs(cm.predict_forest(f, Xt) - reg.predict(Xt))) <= 1e-12, kw
    clf = HistGradientBoostingClassifier(max_iter=60, random_state=0).fit(X[:5000], df["leapfrog_success"].to_numpy()[:5000])
    raw = cm.predict_forest(cm.export_forest(clf), X[:3000])
    assert np.max(np.abs(1.0 / (1.0 + np.exp(-raw)) - clf.predict_proba(X[:3000])[:, 1])) <= 1e-12


def test_compiled_policy_equals_the_python_policy():
    from orbit_ml_cr3bp import compiled_model_3d as cm
    from orbit_ml_cr3bp import policy_c_3d as pc
    from orbit_ml_cr3bp import policy_b_3d as pb
    df = _random_policy_frame(5000, 12)
    train, te = df.iloc[:4000].reset_index(drop=True), df.iloc[4000:].reset_index(drop=True)
    for feats, gate in ((pb.FEATS_STATE, "clf"), (pc.FEATS_OSC, "clf"), (pc.FEATS_OSC, "censored"), (pb.FEATS_STATE, "none")):
        models = pc.fit_policy_c(train, feats, quantile=0.1, seed=3, lf_gate=gate, reg_kw={"max_iter": 40, "max_depth": 4}, clf_kw={"max_iter": 30})
        p_star, g_star = 0.6, 0.2
        py = pc.predict_policy_c(models, te, p_star=p_star, g_star=g_star)
        idx, raw = pc.choose_unscaled(py)
        cp = cm.CompiledHGBPolicy(models, p_star=p_star, g_star=g_star)
        ch, dt = cp.batch(te[["x", "y", "z", "vx", "vy", "vz"]].to_numpy(), te["sim_time"].to_numpy())
        assert (ch == idx).all(), (gate, int((ch != idx).sum()))
        assert np.allclose(dt, raw, rtol=1e-12, atol=0), gate
        for i in range(0, 1000, 97):          # the single-orbit entry point agrees with the batch
            c1, d1 = cp.one(te[["x", "y", "z", "vx", "vy", "vz"]].to_numpy()[i], te["sim_time"].to_numpy()[i])
            assert c1 == ch[i] and d1 == dt[i]


def test_compiled_linear_policy_equals_its_python_form():
    from orbit_ml_cr3bp import compiled_model_3d as cm
    df = _random_policy_frame(3000, 13)
    lin = cm.fit_linear_policy(df.iloc[:2500].reset_index(drop=True), quantile=0.1)
    cp = cm.CompiledLinearPolicy(lin, g_star=0.1)
    te = df.iloc[2500:].reset_index(drop=True)
    st, T = te[["x", "y", "z", "vx", "vy", "vz"]].to_numpy(), te["sim_time"].to_numpy()
    ch, dt = cp.batch(st, T)
    lr, ll = cp.raw_log_dt(st, T)
    d_rk4, d_lf = 10 ** lr, 10 ** ll
    allow = (ll - math.log10(1e-4) >= 0.1) & (d_lf > d_rk4 / 4.0)
    assert (ch == allow.astype(int)).all()
    assert np.allclose(dt, np.where(allow, d_lf, d_rk4), rtol=1e-10)
    # the quantile model is roughly calibrated on its own training distribution (q = 0.1: about 10 % of labels below the prediction)
    trn = df.iloc[:2500]
    lr_t, _ = cp.raw_log_dt(trn[["x", "y", "z", "vx", "vy", "vz"]].to_numpy(), trn["sim_time"].to_numpy())
    over = float((lr_t > np.log10(trn["dt_rk4"].to_numpy())).mean())
    assert 0.05 < over < 0.2, over


# ---------------------------------------------------------------------------------------------------------------------
# C4: RK4 in Sundman time
# ---------------------------------------------------------------------------------------------------------------------

def _smooth_states(n, seed, mu=0.01215058535):
    """Orbits near the triangular points: smooth, no close approach, so the RK4 order shows up cleanly."""
    from orbit_ml_cr3bp.physics_3d import generate_3d_orbit
    np.random.seed(seed)
    out = []
    for _ in range(n):
        st, _lab = generate_3d_orbit(mu=mu, point="L4", sampling_mode="lagrange", offset=0.03, velocity_perturb=0.03, return_label=True)
        out.append(np.asarray(st, dtype=float))
    return out


def _py_rk4s(state, ds, T, mu, tau_c, gfix=0.0):
    """Pure-Python statement of the algorithm in sundman_3d's docstring, counting right-hand-side evaluations independently."""
    from orbit_ml_cr3bp import sundman_3d as sd
    from orbit_ml_cr3bp.physics_3d import _step_rk4_3d_numba
    G = sd._G.py_func
    n_rhs = [0]

    def Gc(Y):
        n_rhs[0] += 1
        o = np.empty(7)
        G(Y, mu, tau_c, gfix, o)
        return o

    X, t = np.array(state, dtype=float), 0.0
    while t < T:
        Y = np.concatenate([X, [t]])
        k1 = Gc(Y)
        dt_pred = k1[6] * ds
        land = t + 1.5 * dt_pred >= T - 1e-12
        if not land:
            k2 = Gc(Y + 0.5 * ds * k1)
            k3 = Gc(Y + 0.5 * ds * k2)
            k4 = Gc(Y + ds * k3)
            Yn = Y + ds / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            if Yn[6] >= T - 1e-12:
                land = True
            else:
                X, t = Yn[:6], Yn[6]
        if land:
            rem = T - t
            nf = max(1, int(math.ceil(rem / dt_pred - 1e-9)))
            h = rem / nf
            for _ in range(nf):
                X = _step_rk4_3d_numba.py_func(X, h, mu)
                n_rhs[0] += 4
            t = T
            break
    return X, n_rhs[0] - 0  # the landing branch reuses k1's evaluation slot: 4 per step, so subtract nothing here


def test_sundman_lands_on_T_and_counts_evaluations_like_a_pure_python_reference():
    from orbit_ml_cr3bp import sundman_3d as sd
    mu = 0.01215058535
    cases = [(np.array([0.2, 0.5, 0.01, -0.3, 0.1, 0.02]), 0.05, 3.0), (np.array([0.8, 0.1, 0.05, -0.1, 0.4, 0.0]), 0.1, 2.0),
             (np.array([0.9, 0.0, 0.0, 0.0, 0.3, 0.1]), 0.02, 1.5), (np.array([-0.5, 0.6, 0.1, 0.3, 0.2, 0.0]), 0.7, 4.0)]
    for st, ds, T in cases:
        ok, prev, cur, t_prev, t, reason, ns, nf = sd.integrate_endpoint_rk4s(st, ds, T, mu, max_steps=10_000_000, tau_c=1.0)
        assert ok and reason == 0 and t == T and t_prev < T
        Xp, n_rhs = _py_rk4s(st, ds, T, mu, 1.0)
        assert np.allclose(cur, Xp, rtol=1e-11, atol=1e-12), (st, ds, T, np.max(np.abs(cur - Xp)))
        # the reference counts every evaluation of G: 4 per s-step (discarded overshoots included), 4 per landing step, plus the one at the
        # landing decision (k1), whose value the first landing step reuses (F = k1[:6] / g). The kernel's cost is 4 per step.
        assert n_rhs == sd.rk4s_evals(ns, nf) + 1, (n_rhs, ns, nf)


def test_sundman_with_constant_g_is_classical_rk4():
    from orbit_ml_cr3bp import sundman_3d as sd
    from orbit_ml_cr3bp.physics_3d import integrate_endpoint_3d, step_rk4_3d
    mu = 0.01215058535
    for st in _smooth_states(4, 5):
        for ds, T in ((0.0625, 2.0), (0.01, 1.0), (0.125, 4.0)):
            a = sd.integrate_endpoint_rk4s(st, ds, T, mu, max_steps=10_000_000, gfix=1.0)
            b = integrate_endpoint_3d(step_rk4_3d, list(st), ds, T, mu, 10_000_000)
            assert a[0] and b[0]
            assert np.max(np.abs(np.asarray(a[2]) - np.asarray(b[2]))) <= 1e-13, (ds, T)
            assert sd.rk4s_evals(a[6], a[7]) == 4 * int(math.ceil(T / ds - 1e-9))


def test_sundman_convergence_order_is_four_in_ds():
    from orbit_ml_cr3bp import sundman_3d as sd
    from orbit_ml_cr3bp.physics_3d import reference_trajectory_dense_3d
    mu = 0.01215058535
    slopes = []
    for st in _smooth_states(10, 7):
        T = 4.0
        sol, ok = reference_trajectory_dense_3d(list(st), T, mu)
        assert ok
        ref = sol.sol(T)[:3]
        dss = np.array([0.16, 0.08, 0.04, 0.02])
        errs = []
        for ds in dss:
            r = sd.integrate_endpoint_rk4s(st, ds, T, mu, max_steps=10_000_000)
            errs.append(np.linalg.norm(np.asarray(r[2])[:3] - ref))
        errs = np.array(errs)
        assert (errs > 1e-12).all()
        slopes.append(np.polyfit(np.log(dss), np.log(errs), 1)[0])
    assert 3.6 < np.median(slopes) < 4.4 and min(slopes) > 3.0, slopes


def test_sundman_step_shrinks_near_a_body():
    from orbit_ml_cr3bp import sundman_3d as sd
    mu = 0.01215058535
    far = np.array([0.2, 0.5, 0.0, 0.0, 0.0, 0.0])
    near = np.array([1 - mu + 0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
    gf, gn = sd._g_only(far, mu, 1.0), sd._g_only(near, mu, 1.0)
    assert gn < 0.05 * gf and gf <= 1.0


# ---------------------------------------------------------------------------------------------------------------------
# C5: step-over guard, fallback, abstention
# ---------------------------------------------------------------------------------------------------------------------

def test_guard_with_infinite_threshold_reproduces_the_unguarded_kernels():
    import math as m
    from orbit_ml_cr3bp.guard_3d import integrate_endpoint_guard, REASON_GUARD
    from orbit_ml_cr3bp.physics_3d import integrate_endpoint_3d, step_rk4_3d, step_leapfrog_symplectic_3d
    mu = 0.01215058535
    for st in _smooth_states(5, 21):
        for integ, fn in (("rk4", step_rk4_3d), ("leapfrog", step_leapfrog_symplectic_3d)):
            for dt, T in ((0.0123, 3.0), (0.05, 5.0)):
                a = integrate_endpoint_guard(integ, st, dt, T, mu, 4_000_000, m.inf)
                b = integrate_endpoint_3d(fn, list(st), dt, T, mu, 4_000_000)
                assert a[0] == b[0] and a[5] == b[5] and a[5] != REASON_GUARD
                assert np.array_equal(np.asarray(a[2]), np.asarray(b[2])), (integ, dt)         # bit for bit
                assert a[6] == m.ceil(T / dt - 1e-9)


def test_guard_fires_on_a_close_approach_and_the_fallback_is_counted():
    import math as m
    from orbit_ml_cr3bp import policy_c_3d as pc
    from orbit_ml_cr3bp.guard_3d import integrate_endpoint_guard, REASON_GUARD
    from orbit_ml_cr3bp.physics_3d import reference_trajectory_dense_3d
    mu = 0.01215058535
    # a fast pass at 0.0064 DU from the Moon (radius 0.0045): dt / tau is far above 1 at the pass for dt = 0.05
    st = np.array([1 - mu - 0.12, 0.05, 0.0, 0.4, 0.1, 0.0])
    sol, ok = reference_trajectory_dense_3d(list(st), 1.0, mu)
    assert ok
    g = integrate_endpoint_guard("rk4", st, 0.05, 1.0, mu, 4_000_000, 0.3)
    assert not g[0]
    assert g[5] == REASON_GUARD and g[6] >= 0
    ref = sol.sol(1.0)[:3]
    rows = np.array([0])
    states, T = st[None, :], np.array([1.0])
    idx, raw = np.array([0]), np.array([0.05])
    ok1, err1, ev1, fb1 = pc.run_chosen(None, states, T, ref[None, :], idx, raw, 1.0, rows, mu, g=0.3, fb=(1e-9, "RK45"))
    assert fb1[0] and ok1[0] and err1[0] < 100.0
    assert ev1[0] > g[6] * 4                    # steps taken x 4 + the adaptive nfev
    ok2, err2, ev2, fb2 = pc.run_chosen(None, states, T, ref[None, :], idx, raw, 1.0, rows, mu, g=m.inf, fb=(1e-9, "RK45"))
    assert not fb2[0] and ev2[0] == 4 * m.ceil(1.0 / 0.05 - 1e-9)


def test_abstention_flags_and_the_certified_risk_counts_them():
    from orbit_ml_cr3bp import policy_c_3d as pc
    df = pd.DataFrame({"osc_reach_2": [1.0, 1.0, 0.0, 1.0], "osc_logtau_hat": [-2.0, -0.2, -2.0, -1.5]})
    f = pc.abstain_flags(df, "physics", logtau_below=-1.0)
    assert f.tolist() == [True, False, False, True]
    disp = np.arange(10.0)
    assert pc.abstain_flags(pd.DataFrame(index=range(10)), "dispersion", dispersion=disp, frac=0.2).sum() == 2


def main():
    names = [k for k in sorted(globals()) if k.startswith("test_")]
    for k in names:
        globals()[k]()
        print("ok  ", k)
    print(f"{len(names)} passed")


if __name__ == "__main__":
    main()
