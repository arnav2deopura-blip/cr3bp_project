"""
Robustness checks added by the rigor plan of 2026-10-30 (Phase 4). Run:  python tests/test_hardening.py   (each test_* function is run; the first failure
stops). The comparator-symmetry and aggregate-metric checks are in tests/test_posthoc.py.
"""
from __future__ import annotations

import glob
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from orbit_ml_cr3bp.config import MU, collision_radii  # noqa: E402


# ---------------------------------------------------------------------------------------------------------------------
# segment-sphere collision test (orbit_ml_cr3bp/collision_segment.py)
# ---------------------------------------------------------------------------------------------------------------------

def test_segment_crossing_detected_between_endpoints():
    from orbit_ml_cr3bp.collision_segment import crossed_bodies, segment_min_distance
    rc1, rc2 = collision_radii(MU)
    moon = np.array([1.0 - MU, 0.0, 0.0])
    # one long step straight through the Moon: both endpoints far outside, the chord passes through the centre
    P = np.array([moon + [-0.05, 0.01, 0.0], moon + [0.05, -0.01, 0.0], moon + [0.2, 0.0, 0.0]])
    assert np.all(np.linalg.norm(P - moon, axis=1) > 10 * rc2)          # the endpoint check sees nothing
    r = crossed_bodies(P, MU)
    assert r["secondary"] == 0 and r["primary"] == -1
    # the same step offset sideways by 2 Moon radii misses
    Q = P + np.array([0.0, 2.0 * rc2 / math.cos(math.atan(0.2)) + 1e-6, 0.0])
    assert crossed_bodies(Q, MU)["secondary"] == -1
    # tangent at exactly 1.0001 radii: a miss; at 0.9999 radii: a hit
    for k, hit in ((1.0001, False), (0.9999, True)):
        T = np.array([moon + [-0.1, k * rc2, 0.0], moon + [0.1, k * rc2, 0.0]])
        assert (crossed_bodies(T, MU)["secondary"] == 0) == hit, k
    # zero-length segment = point test; the endpoint inside counts
    Z = np.array([moon + [0.5 * rc2, 0, 0], moon + [0.5 * rc2, 0, 0]])
    assert crossed_bodies(Z, MU)["secondary"] == 0
    # the Earth with its own radius
    earth = np.array([-MU, 0.0, 0.0])
    E = np.array([earth + [0.0, -0.3, 0.0], earth + [0.0, 0.3, 0.0]])
    assert crossed_bodies(E, MU)["primary"] == 0
    # distances: brute force on a fine subdivision agrees
    rng = np.random.RandomState(0)
    R = moon + rng.normal(scale=0.03, size=(30, 3))
    d = segment_min_distance(R, moon)
    t = np.linspace(0, 1, 2001)[:, None]
    for i in range(len(R) - 1):
        brute = np.min(np.linalg.norm(R[i] + t * (R[i + 1] - R[i]) - moon, axis=1))
        assert abs(brute - d[i]) <= 1e-6, i


# ---------------------------------------------------------------------------------------------------------------------
# usable_rows: the reference outcome alone selects the same rows as the sealed FTLE-based rule
# ---------------------------------------------------------------------------------------------------------------------

def _sealed_usable(df):
    m = ~df["ftle_nan"].astype(bool)
    if "reference_divergent" in df.columns:
        m &= ~df["reference_divergent"].fillna(False).astype(bool)
    return df[m]


def test_usable_rows_equivalence_on_every_dataset():
    from orbit_ml_cr3bp import policy_b_3d as pb
    files = sorted(glob.glob(os.path.join(ROOT, "data", "*.csv")) + glob.glob(os.path.join(ROOT, "data_b", "*.csv"))
                   + glob.glob(os.path.join(ROOT, "data_b", "*", "*.csv")) + glob.glob(os.path.join(ROOT, "data_c", "*", "*.csv"))
                   + glob.glob(os.path.join(ROOT, "data_d", "*", "*.csv")))
    if not files:
        print("  (no datasets on disk; skipped)")
        return
    for f in files:
        d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in ("ftle_nan", "reference_divergent", "reference_outcome"))
        for c in ("ftle_nan", "reference_divergent"):
            if c in d.columns:
                d[c] = d[c].map({True: True, False: False, "True": True, "False": False}).fillna(False).astype(bool)
        assert pb.usable_rows(d).index.equals(_sealed_usable(d).index), f
    # a frame without reference_outcome falls back to the sealed rule
    d = pd.DataFrame({"ftle_nan": [False, True, False], "reference_divergent": [False, False, True]})
    assert list(pb.usable_rows(d).index) == [0]
    # pipeline v6 rows: no FTLE, so ftle_nan is False and only the outcome decides
    d = pd.DataFrame({"ftle_nan": [False, False], "reference_divergent": [False, True], "reference_outcome": ["ok", "ejection"],
                      "ftle_3d": [np.nan, np.nan]})
    assert list(pb.usable_rows(d).index) == [0]


# ---------------------------------------------------------------------------------------------------------------------
# osculating periapsis features near e = 1 (the hyperbolic / elliptic semi-major axis diverges)
# ---------------------------------------------------------------------------------------------------------------------

def _kepler_periapsis(r0, v0, GM, t_max=400.0):
    from scipy.integrate import solve_ivp

    def f(t, y):
        r = y[:3]
        return np.concatenate([y[3:], -GM * r / np.linalg.norm(r) ** 3])

    def ev(t, y):
        return float(np.dot(y[:3], y[3:]))
    ev.terminal = True
    ev.direction = 1.0
    sol = solve_ivp(f, (0.0, t_max), np.concatenate([r0, v0]), method="DOP853", rtol=1e-13, atol=1e-15, events=ev)
    return float(np.linalg.norm(sol.y_events[0][0][:3])), float(sol.t_events[0][0])


def test_osculating_near_parabolic():
    """|1 - e| from 1e-6 to 1e-3 on both sides, approaching the periapsis: r_p and t_p agree with a two-body integration and are
    continuous across e = 1 (the elliptic and hyperbolic branches are used here, not the parabolic fallback, which starts at 1e-6)."""
    from orbit_ml_cr3bp.osculating_3d import osc_features, OSC_FEATURE_COLS
    mu = 1e-11
    GM = 1.0 - mu
    ix = {c: i for i, c in enumerate(OSC_FEATURE_COLS)}
    rp0, r0 = 0.05, 0.5
    res = {}
    for de in (-1e-3, -1e-4, -1e-5, -2e-6, 2e-6, 1e-5, 1e-4, 1e-3):
        e = 1.0 + de
        h = math.sqrt(GM * rp0 * (1.0 + e))
        v2 = GM * (e - 1.0) / rp0 + 2.0 * GM / r0          # vis-viva with 2 eps = GM (e - 1) / r_p
        vt = h / r0
        vr = -math.sqrt(v2 - vt * vt)                         # approaching
        rr = np.array([r0, 0.0, 0.0])
        vv = np.array([vr, vt, 0.0])
        rp_num, tp_num = _kepler_periapsis(rr, vv, GM)
        st = np.array([rr[0] - mu, rr[1], rr[2], vv[0] + rr[1], vv[1] - rr[0], vv[2]])   # rotating frame, primary at (-mu, 0, 0)
        T = 400.0
        f, fl = osc_features(st[None, :], np.array([T]), mu)
        assert np.all(np.isfinite(f[0])), de
        rp, tp = 10 ** f[0, ix["osc_logrp_1"]], f[0, ix["osc_tpn_1"]] * T
        assert abs(rp - rp_num) <= 1e-7 * rp_num, (de, rp, rp_num)
        assert abs(tp - tp_num) <= 1e-6 * tp_num, (de, tp, tp_num)
        assert abs(f[0, ix["osc_ecc_1"]] - e) <= 1e-8, de
        res[de] = tp
    assert abs(res[-2e-6] - res[2e-6]) <= 1e-5 * res[2e-6]


# ---------------------------------------------------------------------------------------------------------------------
# pipeline v6 = v5 without the FTLE
# ---------------------------------------------------------------------------------------------------------------------

def test_pipeline_v6_labels_equal_v5():
    from orbit_ml_cr3bp.data_3d import generate_one_3d_training_orbit, FTLE_STATUS_NOT_COMPUTED
    ftle_cols = {"ftle_3d", "ftle_status", "ftle_t_frac"}
    for idx in (0, 1, 2):
        a = generate_one_3d_training_orbit(idx, MU, base_seed=777, label_mode="accuracy_monotone", pipeline="v5")
        b = generate_one_3d_training_orbit(idx, MU, base_seed=777, label_mode="accuracy_monotone", pipeline="v6")
        ra, rb = a["row"], b["row"]
        assert set(ra) == set(rb), idx
        assert a["usable"] == b["usable"] and a["combo"] == b["combo"], idx
        for k in ra:
            if k in ftle_cols:
                continue
            va, vb = ra[k], rb[k]
            same = (va == vb) or (isinstance(va, float) and isinstance(vb, float) and math.isnan(va) and math.isnan(vb))
            assert same, (idx, k, va, vb)
        assert math.isnan(rb["ftle_3d"]) and rb["ftle_status"] == FTLE_STATUS_NOT_COMPUTED and rb["ftle_nan"] is False


def test_comparator_settings_defined_once():
    """The adaptive comparators' atol ratio, evaluation cap and sealed rtol grid live in config and are the sealed values."""
    from orbit_ml_cr3bp import config as c, fair_baseline_3d as fb, compiled_solvers_d as cs, policy_b_3d as pb
    assert c.ADAPTIVE_ATOL_RATIO == 1e-3 and c.MAX_ADAPTIVE_NFEV == 2_000_000
    assert c.RTOL_GRID_B == tuple(10.0 ** e for e in np.linspace(-12.0, -3.0, 18)) and len(c.RTOL_GRID_B) == 18
    assert fb.RTOL_GRID_B is c.RTOL_GRID_B and cs.RTOL_GRID_B is c.RTOL_GRID_B and pb.RTOL_GRID_B is c.RTOL_GRID_B
    assert fb.MAX_ADAPTIVE_NFEV == cs.MAX_NFEV == c.MAX_ADAPTIVE_NFEV and cs.ATOL_RATIO == c.ADAPTIVE_ATOL_RATIO
    assert fb._run_adaptive_ex.__defaults__ == (c.ADAPTIVE_ATOL_RATIO,)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for f in fns:
        f()
        print("ok", f.__name__)
    print(f"{len(fns)} tests passed")
