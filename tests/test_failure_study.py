"""
Checks for the Part B analysis library. Run:  python tests/test_failure_study.py
The integration-heavy tests generate a small v4 dataset in a temp folder (about 15 s).
"""
from __future__ import annotations

import math
import os
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from orbit_ml_cr3bp import failure_study_3d as fs  # noqa: E402
from orbit_ml_cr3bp.config import MU  # noqa: E402

_CACHE = {}


def _mini_dataset(n=260, seed=777):
    if "d" in _CACHE:
        return _CACHE["d"]
    from orbit_ml_cr3bp.data_3d import generate_3d_training_dataset
    cwd = os.getcwd()
    tmp = tempfile.mkdtemp()
    os.chdir(tmp)
    try:
        df, _, _ = generate_3d_training_dataset(num_orbits=n, n_workers=6, chunk_size=4, label_mode="accuracy_monotone",
                                                tol_km=100.0, base_seed=seed, pipeline="v4")
    finally:
        os.chdir(cwd)
    from orbit_ml_cr3bp import policy_b_3d as pb
    d = df.reset_index(drop=True)
    d["row_id"] = np.arange(len(d))
    d = pb.usable_rows(d).reset_index(drop=True)
    d = fs.add_derived(d)
    _CACHE["d"] = d
    return d


def test_decomposition_identity_synthetic():
    rng = np.random.RandomState(0)
    n = 500
    led = pd.DataFrame({
        "sim_time": rng.uniform(1, 20, n), "ml_raw_dt": 10 ** rng.uniform(-3, -1, n), "ml_scale": rng.choice([0.1, 0.5, 0.6], n),
        "dstar_rk4": 10 ** rng.uniform(-3, -0.5, n), "dstar_leapfrog": 10 ** rng.uniform(-3, -1, n),
        "ml_integrator": rng.choice(["rk4", "leapfrog"], n), "C_feat": 1.0, "C_pred": 600.0, "meets_tol": True,
    })
    led["ml_dt"] = led.ml_raw_dt * led.ml_scale
    led["e_chosen"] = led.ml_integrator.map({"rk4": 4, "leapfrog": 1})
    led["dstar_chosen"] = np.where(led.ml_integrator == "rk4", led.dstar_rk4, led.dstar_leapfrog)
    led["ml_evals"] = led.e_chosen * np.ceil(led.sim_time / led.ml_dt)
    led["C_ml_int"] = led.ml_evals
    led["C_ml_tot"] = led.ml_evals + led.C_feat + led.C_pred
    c_rk = 4 * np.ceil(led.sim_time / led.dstar_rk4)
    c_lf = 1 * np.ceil(led.sim_time / led.dstar_leapfrog)
    led["C_fix_star"] = np.minimum(c_rk, c_lf)
    led["k_star"] = np.where(c_rk <= c_lf, "rk4", "leapfrog")
    led["X_nfev"] = rng.uniform(50, 5000, n)
    dec = fs.decompose(led, "X")
    assert dec["total"].notna().all()
    assert np.allclose(dec[["P", "S", "Q", "K", "O", "R"]].sum(axis=1), dec["total"], atol=1e-9)
    assert (dec["S"] >= -1e-12).all(), "selection loss is non-negative by construction of k*"
    # R is pure ceil rounding: bounded by the log of one step's worth of cost
    steps = led.sim_time / led.ml_dt
    assert (dec["R"].abs() <= np.log10(1 + 1 / np.minimum(steps, led.sim_time / np.where(led.k_star == "rk4",
            led.dstar_rk4, led.dstar_leapfrog))) + 1e-9).all()


def test_cell_merge_rule():
    rng = np.random.RandomState(1)
    rows = []
    # regime I, r2 band '<0.05': three horizon cells of 30 each (all < 50, union 90 >= 50) -> 'I|<0.05|h*'
    for h, n in (("1-3", 30), ("3-8", 30), ("8-20", 30)):
        rows += [("I", "<0.05", h)] * n
    # regime I, band '0.05-0.17': 80 + 10 + 10 -> big cell kept, small two merged (20 < 50) -> r2 stage
    rows += [("I", "0.05-0.17", "1-3")] * 80 + [("I", "0.05-0.17", "3-8")] * 10 + [("I", "0.05-0.17", "8-20")] * 10
    # regime V: single big cell and one tiny leftover that cannot reach 50 anywhere
    rows += [("V", "0.17-0.8", "3-8")] * 120 + [("V", ">0.8", "1-3")] * 5
    d = pd.DataFrame(rows, columns=["regime", "r2_band", "h_band"])
    lab = fs.assign_cells(d)
    assert (lab[(d.regime == "I") & (d.r2_band == "<0.05")] == "I|<0.05|h*").all()
    assert (lab[(d.regime == "I") & (d.r2_band == "0.05-0.17") & (d.h_band == "1-3")] == "I|0.05-0.17|1-3").all()
    small = lab[(d.regime == "I") & (d.r2_band == "0.05-0.17") & (d.h_band != "1-3")]
    assert (small == "pooled-small").all() or (small == "I|r2*|h*").all()
    assert (lab[(d.regime == "V") & (d.r2_band == "0.17-0.8")] == "V|0.17-0.8|3-8").all()
    assert (lab[(d.regime == "V") & (d.r2_band == ">0.8")] == "pooled-small").all()
    assert lab.notna().all()


def test_lazy_scale_equals_full_chain():
    from orbit_ml_cr3bp import policy_b_3d as pb
    from orbit_ml_cr3bp.audit_3d import calibrate_scale_ltt, ReferenceCache
    from orbit_ml_cr3bp.modeling_3d import predict_hurdle
    d = _mini_dataset()
    n = len(d)
    ntr = n - 140
    models = pb.fit_policy_models(d, np.arange(ntr), n_estimators=40)
    cal = d.iloc[ntr:].reset_index(drop=True)
    pred = predict_hurdle(models, cal)
    cache = ReferenceCache(MU)
    for k in pb.CANDIDATES:
        full = calibrate_scale_ltt(cal, pred, k, tol_km=100.0, alpha=0.05, delta=0.1, mu=MU, cache=cache)
        lazy = pb.certify_scale_lazy(cal, pred, k, tol_km=100.0, mu=MU, cache=cache)
        assert full["chosen_scale"] == lazy["chosen_scale"], (k, full["chosen_scale"], lazy["chosen_scale"])


def test_spec_search_matches_original():
    from orbit_ml_cr3bp import policy_b_3d as pb
    from orbit_ml_cr3bp.fair_baseline_3d import scipy_matched_accuracy
    from orbit_ml_cr3bp.physics_3d import reference_trajectory_dense_3d
    d = _mini_dataset().iloc[:8]
    for _, r in d.iterrows():
        st = np.array([r.x, r.y, r.z, r.vx, r.vy, r.vz])
        sol, ok = reference_trajectory_dense_3d(list(st), r.sim_time, MU)
        ref = sol.sol(r.sim_time)[:3]
        for method in ("RK45", "DOP853"):
            mine = pb.spec_rtol_search(st, r.sim_time, ref, 100.0, MU, method)
            orig = scipy_matched_accuracy(list(st), r.sim_time, ref, 100.0, MU, method)
            assert mine is not None and orig is not None
            assert mine["err_km"] <= 100.0 and orig["err_km"] <= 100.0
            # both find "the loosest rtol that meets the tolerance" by bisection to 0.02 decade; the error is not monotone
            # in rtol so allow up to a decade of disagreement and require a similar cost within a factor 5
            assert abs(math.log10(mine["rtol"]) - math.log10(orig["rtol"])) <= 1.0
            assert 0.2 <= mine["nfev"] / orig["nfev"] <= 5.0, (method, mine["nfev"], orig["nfev"])


def test_classify_priorities():
    led = pd.DataFrame({
        "ref_check_km": [20.0, 0.0, 0.0, 0.0, 0.0, 0.0], "cens_chosen": [False, True, False, False, False, False],
        "abstain": [False, False, True, False, False, False], "violation": [False, False, False, True, False, False],
        "meets_tol": [True, True, False, False, True, True], "C_ml_tot": [10.0, 10, 10, 10, 10, 1000],
        "X_nfev": [5.0, 5, 5, 5, 20, 5], "C_fix_star": [1.0] * 6, "sim_time": [3.0] * 6, "ml_raw_dt": [0.1] * 6,
        "dstar_chosen": [0.1] * 6, "ml_scale": [0.5] * 6, "C_feat": [1.0] * 6, "C_pred": [5.0] * 6, "C_ml_int": [4.0] * 6,
        "e_chosen": [4] * 6, "k_star": ["rk4"] * 6, "dstar_rk4": [0.1] * 6, "dstar_leapfrog": [0.1] * 6,
        "over_pred": [0.0, 0, 0, 0.3, 0, 0],
    })
    cls, _ = fs.classify(led, "X")
    assert list(cls[:5]) == ["F0-ref", "F0-label", "abstain", "violation:unattributed", "ml-win"], list(cls)
    assert cls[5].startswith("cost-loss")


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                fails += 1
                print("FAIL", name, type(e).__name__, e)
    sys.exit(1 if fails else 0)
