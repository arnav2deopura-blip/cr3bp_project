"""
Phase-0 checks for the failure study (Part B). Run:  python tests/test_phase0.py   (or pytest)
Each test states what it protects. They are cheap (seconds) except test_legacy_labels_reproduce, which relabels 24 orbits.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from orbit_ml_cr3bp.config import MU, du_km_for_mu  # noqa: E402
from orbit_ml_cr3bp import physics_3d as ph  # noqa: E402
from orbit_ml_cr3bp.diagnostics_3d import (  # noqa: E402
    hill_regime, jacobi_at_points, trajectory_diagnostics_3d, count_crossings)
from orbit_ml_cr3bp.search_3d import label_dt_max, find_dt_for_integrator_accuracy_monotone_3d  # noqa: E402

REF_CSV = os.path.join(ROOT, "data", "cr3bp_3d_training_data_accuracy_monotone_seed12.csv")


def test_fsal_leapfrog_bit_identical():
    """Issue 8: the FSAL endpoint kernel must give exactly the old two-acceleration-per-step result."""
    rng = np.random.RandomState(3)
    for _ in range(25):
        s0 = np.array([0.5 + 0.4 * rng.randn(), 0.3 * rng.randn(), 0.1 * rng.randn(),
                       0.2 * rng.randn(), 0.2 * rng.randn(), 0.1 * rng.randn()])
        dt = float(rng.choice([0.003, 0.02, 0.1, 0.37]))
        T = float(rng.uniform(1.0, 4.0))
        ok, _p, curr, _tp, t_act, reason = ph.integrate_endpoint_3d(ph.step_leapfrog_symplectic_3d, list(s0), dt, T, MU)
        if not ok:
            continue
        # reference loop with the original step
        s = s0.copy()
        t = 0.0
        while t < T:
            h = dt
            if t + dt >= T - 1e-12:
                h = T - t
                t = T
            else:
                t += h
            s = ph._step_leapfrog_symplectic_3d_numba(s, h, MU)
        assert np.array_equal(np.asarray(curr), s), (curr, s)


def test_eval_table():
    assert ph.EVALS_PER_STEP_3D["leapfrog"] == 1 and ph.EVALS_PER_STEP_3D["rk4"] == 4
    assert ph.EVALS_PER_STEP_3D_V3["leapfrog"] == 2


def test_hill_thresholds():
    """Plan section 2.4 values for the Earth-Moon system (computed with the project's own functions)."""
    c = jacobi_at_points(MU)
    assert abs(c["L1"] - 3.18834) < 2e-5 and abs(c["L2"] - 3.17216) < 2e-5
    assert abs(c["L3"] - 3.01215) < 2e-5 and abs(c["L4"] - 2.98800) < 1e-5 and abs(c["L5"] - c["L4"]) < 1e-12
    r = hill_regime([3.5, 3.18, 3.1, 3.0, 2.9], MU)
    assert list(r) == ["I", "II", "III", "IV", "V"], list(r)


def test_ceiling_modes():
    st = [0.8, 0.0, 0.0, 0.0, 0.1, 0.0]
    assert label_dt_max(st, 20.0, MU, "physical") == 2.0
    assert label_dt_max(st, 1.2, MU, "physical") == 1.2
    legacy = label_dt_max(st, 20.0, MU, "legacy")
    assert 0.005 <= legacy <= 0.5


def test_diagnostics_circular_orbit_is_uniformly_paced():
    """A near-circular orbit around the Moon is uniformly paced: delta_phys ~ 1, no transits, one pass, positive tau."""
    r = 0.02
    v = math.sqrt(MU / r)
    # prograde in the inertial frame; in the rotating frame subtract the frame rotation (v_rot = v - r*omega)
    state = [1.0 - MU + r, 0.0, 0.0, 0.0, v - r, 0.0]
    sol, ok = ph.reference_trajectory_dense_3d(state, 2.0, MU)
    assert ok
    d = trajectory_diagnostics_3d(sol, state, MU, du_km_for_mu(MU))
    assert 0.9 < d["min_r2"] / r < 1.1, d
    assert 1.0 <= d["delta_phys"] < 1.6, d
    assert d["frac_near_secondary"] > 0.99 and d["n_transit_l1"] == 0 and d["n_transit_l2"] == 0
    assert d["ref_check_km"] < 1e-3 and d["delta_emp"] >= 1.0


def test_crossing_counter():
    x = np.array([-1.0, -0.5, 0.5, 1.0, 0.4, -0.2])
    y = np.zeros_like(x)
    assert count_crossings(x, y, 0.0) == 2
    assert count_crossings(x, y + 1.0, 0.0) == 0


def test_dt_search_physical_not_censored_on_old_censored_rows():
    """Issue 1: rows that Part A pinned at the ceiling are no longer censored under the physical ceiling."""
    import pandas as pd
    d = pd.read_csv(REF_CSV)
    d = d[~d.ftle_nan & ~d.reference_divergent.fillna(False)]
    d = d[d.rk4_censored].sample(n=12, random_state=1)
    for _, r in d.iterrows():
        st = [r.x, r.y, r.z, r.vx, r.vy, r.vz]
        sol, ok = ph.reference_trajectory_dense_3d(st, r.sim_time, MU)
        res = find_dt_for_integrator_accuracy_monotone_3d("rk4", st, r.sim_time, MU, reference=(sol, ok),
                                                          ceiling_mode="physical", return_details=True)
        assert not res["censored"], (r.sim_time, res)
        assert res["dt"] >= r.dt_rk4 * 0.999, (res["dt"], r.dt_rk4)  # can only get larger than a censored lower bound


def test_legacy_labels_reproduce():
    """The legacy path (ceiling_mode='legacy', label_pipeline_version 3 behaviour) still gives the sealed labels.
    Reference rtol changed 1e-11 -> 1e-13, so allow a rare small difference where the error is within ~0.1 km of the tolerance."""
    import pandas as pd
    from orbit_ml_cr3bp.data_3d import label_orbit_accuracy_3d
    d = pd.read_csv(REF_CSV)
    d = d[~d.ftle_nan & ~d.reference_divergent.fillna(False)].sample(n=24, random_state=5)
    same = 0
    for _, r in d.iterrows():
        st = [r.x, r.y, r.z, r.vx, r.vy, r.vz]
        lab = label_orbit_accuracy_3d(st, r.sim_time, MU, monotone=True, ceiling_mode="legacy")
        same += int(abs(lab["dt_rk4"] - r.dt_rk4) <= 1e-3 * r.dt_rk4 and lab["rk4_censored"] == bool(r.rk4_censored))
    assert same >= 22, same


def test_convergence_orders():
    """H7-style sanity: leapfrog 2nd order, RK4 4th order still hold after the kernel change."""
    st = [0.45, 0.82, 0.01, 0.02, -0.03, 0.01]
    T = 2.0
    sol, ok = ph.reference_trajectory_dense_3d(st, T, MU)
    assert ok
    ref = sol.sol(T)[:3]
    for name, order, dts in (("leapfrog", 2.0, (0.04, 0.02)), ("rk4", 4.0, (0.1, 0.05))):
        errs = []
        for dt in dts:
            _ok, _p, curr, _tp, _ta, _r = ph.integrate_endpoint_3d(ph.INTEGRATORS_3D[name], st, dt, T, MU)
            errs.append(np.linalg.norm(np.asarray(curr)[:3] - ref))
        slope = math.log(errs[0] / errs[1]) / math.log(dts[0] / dts[1])
        assert abs(slope - order) < 0.4, (name, slope)


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                print("FAIL", name, type(e).__name__, e)
    sys.exit(1 if fails else 0)
