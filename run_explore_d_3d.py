"""
Exploratory commands of Part D (pre-registered as exploratory in part_d/PREREGISTRATION_D.md; outside the Holm family; no seal guard, every number is labelled
exploratory in the results). Outputs go to a run folder under --out_dir.

  d6          grouped permutation importance and drop-group refits (Sundman and physical-time targets), FTLE as an offline explanatory group
  symplectic  the leapfrog step is symplectic: Phi^T J Phi = J by finite differences (Tier 3)
  d12         the highest lambda*T orbits recomputed with a tighter / different reference: which are 'unpredictable' (reference not converged)
  d7          FTLE field and policy violations on a planar surface of section at three Jacobi constants
  d10         tau_c sweep of the oracle Sundman cost and the non-monotone labels (what the 4.5 % are)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from orbit_ml_cr3bp import protocol_3d as proto  # noqa: E402
from orbit_ml_cr3bp.config import MU, POSITION_TOL_KM, du_km_for_mu  # noqa: E402
import run_part_d_3d as rp  # noqa: E402


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(proto.json_safe(obj), f, indent=1)


def cmd_d6(a):
    from orbit_ml_cr3bp import exploratory_d as ex
    out = rp.make_run_dir(a.out_dir, "d6", a)
    t0 = time.time()
    tr, w = rp.load_train()
    te, _ref = rp.load_set(a.test_set)
    if len(tr) > a.max_train:
        idx = np.random.RandomState(0).choice(len(tr), a.max_train, replace=False)
        tr, w = tr.iloc[idx].reset_index(drop=True), w[idx]
    rep = {"n_train": int(len(tr)), "n_test": int(len(te)), "test_set": a.test_set}
    for target, lab_col, ok_col in (("sundman_time (log10 ds*)", "dt_rk4s", "rk4s_success"), ("physical_time (log10 dt*)", "dt_rk4", "rk4_success")):
        trn, ten = tr[tr[ok_col] == 1], te[te[ok_col] == 1]
        wt = w[(tr[ok_col] == 1).to_numpy()]
        ytr, yte = np.log10(trn[lab_col].to_numpy(dtype=float)), np.log10(ten[lab_col].to_numpy(dtype=float))
        rep[target] = {"label_std_dex": float(yte.std()), "label_median_abs_dev_dex": float(np.median(np.abs(yte - np.median(yte)))),
                       "deployable_groups": ex.grouped_importance(trn, ytr, ten, yte, ex.GROUPS, w_train=wt, n_perm=a.n_perm, n_iter=a.n_iter),
                       "with_ftle_group": ex.grouped_importance(trn, ytr, ten, yte, {**ex.GROUPS, **ex.FTLE_GROUP}, w_train=wt, n_perm=a.n_perm, n_iter=a.n_iter)}
        print(target, "full median |err|", round(rep[target]["deployable_groups"]["full"]["median_abs_err"], 3), f"({time.time() - t0:.0f}s)", flush=True)
        for kind in ("deployable_groups", "with_ftle_group"):
            print(" ", kind, {g: (round(v["drop_delta_mse"], 4), round(v["perm_delta_mse"], 4)) for g, v in rep[target][kind]["groups"].items()}, flush=True)
    save_json(rep, os.path.join(out, "d6_report.json"))
    print("wrote", out)


def cmd_symplectic(a):
    from orbit_ml_cr3bp import exploratory_d as ex
    from orbit_ml_cr3bp.physics_3d import _step_leapfrog_symplectic_3d_numba as leap, _step_rk4_3d_numba as rk4
    out = rp.make_run_dir(a.out_dir, "symplectic", a)
    U, _ = rp.load_set("select")
    st = U[["x", "y", "z", "vx", "vy", "vz"]].sample(a.n, random_state=0).to_numpy(dtype=float)
    rep = {"n_states": int(len(st))}
    for dt in a.dts:
        rep[f"dt={dt}"] = {}
        for name, fn in (("leapfrog", leap), ("rk4", rk4)):
            d = ex.symplecticity_defect(fn, st, MU, h=1e-6, dt=dt)
            rep[f"dt={dt}"][name] = {"max_defect": float(np.nanmax(d)), "median_defect": float(np.nanmedian(d)), "finite": int(np.isfinite(d).sum())}
        print(dt, rep[f"dt={dt}"], flush=True)
    save_json(rep, os.path.join(out, "symplectic_report.json"))


def _ref_alt(job):
    from scipy.integrate import solve_ivp
    from orbit_ml_cr3bp.physics_3d import _cr3bp_derivatives_3d_numba as f, reference_trajectory_dense_3d
    state, T, mu = job
    res = {}
    sol, reached = reference_trajectory_dense_3d(list(state), T, mu, rtol=1e-14, atol=1e-14, max_step=0.002)
    res["tight"] = sol.sol(T)[:3] if reached else np.full(3, np.nan)
    try:
        s2 = solve_ivp(lambda t, y: f(y, mu), (0, T), state, method="LSODA", rtol=1e-13, atol=1e-13)
        res["lsoda"] = s2.y[:3, -1] if s2.success else np.full(3, np.nan)
    except Exception:
        res["lsoda"] = np.full(3, np.nan)
    return res


def cmd_d12(a):
    """D12. The top `frac` of lambda_T = FTLE * T of a dataset: final positions from (a) the standard reference (DOP853 1e-13, step <= 0.01), (b) DOP853 at 1e-14 with step <= 0.002,
    (c) LSODA at 1e-13 (a different method: Adams / BDF, variable order). An orbit is 'unpredictable' if the three disagree by more than a tenth of the tolerance. Then the policy's violation labels with and without them."""
    from multiprocessing import Pool
    import joblib
    from orbit_ml_cr3bp import deploy_d as dd
    out = rp.make_run_dir(a.out_dir, "d12", a)
    U, ref = rp.load_set(a.set)
    S, T = dd.states_T(U)
    lam = U["lambda_T"].to_numpy()
    thr = np.quantile(lam, 1.0 - a.frac)
    top = np.where(lam >= thr)[0]
    print(f"{len(top)} orbits with lambda*T >= {thr:.2f} of {len(U)}", flush=True)
    with Pool(a.n_jobs) as pool:
        alt = pool.map(_ref_alt, [(S[i], float(T[i]), MU) for i in top], chunksize=2)
    du = du_km_for_mu(MU)
    tight = np.array([r["tight"] for r in alt])
    lsoda = np.array([r["lsoda"] for r in alt])
    d_tight = np.linalg.norm(tight - ref[top], axis=1) * du
    d_radau = np.linalg.norm(lsoda - ref[top], axis=1) * du
    spread = np.nanmax(np.vstack([d_tight, d_radau]), axis=0)
    unpred = ~(spread <= 0.1 * POSITION_TOL_KM)
    df = pd.DataFrame({"row_id": U["row_id"].to_numpy()[top], "lambda_T": lam[top], "sim_time": T[top], "ref_vs_tight_km": d_tight, "ref_vs_lsoda_km": d_radau, "unpredictable": unpred})
    df.to_csv(os.path.join(out, "d12_orbits.csv"), index=False)
    # policy labels with the standard reference and with the tight one
    fz = joblib.load(os.path.join(ROOT, rp.FROZEN_SUND))
    runner = rp._runner()
    cal, ev = rp.standard_split(U)
    from orbit_ml_cr3bp import policy_c_3d as pc
    raw = 10.0 ** pc.predict_raw_c(fz["models"], U)["rk4s"]["logdt"]
    sc = dd.certify_scale(runner, S[cal], T[cal], raw[cal], ref[cal])["chosen_scale"]
    r = dd.run_rows(runner, S[top], T[top], raw[top] * sc, ref[top])
    r_t = dd.run_rows(runner, S[top], T[top], raw[top] * sc, np.where(np.isfinite(tight), tight, ref[top]))
    rep = {"set": a.set, "frac": a.frac, "n_top": int(len(top)), "lambda_T_threshold": float(thr), "scale": sc,
           "unpredictable": int(unpred.sum()), "unpredictable_share": float(unpred.mean()),
           "median_disagreement_km": {"tight": float(np.nanmedian(d_tight)), "lsoda": float(np.nanmedian(d_radau))}, "max_disagreement_km": {"tight": float(np.nanmax(d_tight)), "lsoda": float(np.nanmax(d_radau))},
           "policy_violation_rate_top": float((~r["met"]).mean()), "policy_violation_rate_top_vs_tight_reference": float((~r_t["met"]).mean()),
           "label_flips": int((r["met"] != r_t["met"]).sum()), "policy_violation_rate_top_excl_unpredictable": float((~r["met"][~unpred]).mean()) if (~unpred).any() else None,
           "policy_violation_rate_all": float((~dd.run_rows(runner, S[ev], T[ev], raw[ev] * sc, ref[ev])["met"]).mean())}
    save_json(rep, os.path.join(out, "d12_report.json"))
    print(json.dumps(proto.json_safe(rep), indent=1))


def _ftle_job(a):
    from orbit_ml_cr3bp.physics_3d import compute_ftle_3d_variational
    st, T, mu = a
    try:
        return float(compute_ftle_3d_variational(list(st), mu, T))
    except Exception:
        return float("nan")


def cmd_d7(a):
    """D7. Surface of section y = 0, z = vz = 0, vy > 0 at fixed Jacobi constants, a grid in (x, vx): FTLE over T, the reference outcome, and the frozen Sundman policy's
    violation at the frozen Part C scale. Does a violation sit on an FTLE ridge?"""
    from multiprocessing import Pool
    import joblib
    from orbit_ml_cr3bp import compiled_solvers_d as cs, deploy_d as dd, policy_c_3d as pc
    from orbit_ml_cr3bp.physics_3d import reference_trajectory_dense_3d
    from sklearn.metrics import roc_auc_score
    out = rp.make_run_dir(a.out_dir, "d7", a)
    fz = joblib.load(os.path.join(ROOT, rp.FROZEN_SUND))
    pol = cs.SundmanPolicy(fz["models"], MU)
    scale = float(json.load(open(os.path.join(ROOT, "part_c", "results", "confirm", "sund", "apply_policy_report.json")))["scale"])
    rep = {"T": a.T, "scale_part_c": scale, "maps": {}}
    xs = np.linspace(a.xlim[0], a.xlim[1], a.n)
    vxs = np.linspace(a.vlim[0], a.vlim[1], a.n)
    for C in a.C:
        X, VX = np.meshgrid(xs, vxs, indexing="ij")
        r1 = np.sqrt((X + MU) ** 2)
        r2 = np.sqrt((X + MU - 1.0) ** 2)
        v2 = X ** 2 + 2 * (1 - MU) / r1 + 2 * MU / r2 - C
        valid = (v2 > VX ** 2) & (r1 > 0.02) & (r2 > 0.01)
        vy = np.sqrt(np.where(valid, v2 - VX ** 2, 0.0))
        idx = np.argwhere(valid)
        st = np.array([[X[i, j], 0.0, 0.0, VX[i, j], vy[i, j], 0.0] for i, j in idx])
        T = np.full(len(st), a.T)
        with Pool(a.n_jobs) as pool:
            ftle = np.array(pool.map(_ftle_job, [(s, a.T, MU) for s in st], chunksize=32))
            refs = pool.map(_ref_job, [(s, a.T, MU) for s in st], chunksize=16)
        ref = np.array(refs)
        ok_ref = np.isfinite(ref).all(axis=1)
        raw = pol.raw_ds(st, T)
        run = dd.run_rows(pol, st, T, raw * scale, np.where(ok_ref[:, None], ref, 0.0))
        viol = ok_ref & ~run["met"]
        key = f"C={C}"
        df = pd.DataFrame({"x": st[:, 0], "vx": st[:, 3], "vy": st[:, 4], "ftle": ftle, "ref_ok": ok_ref, "violation": viol})
        df.to_csv(os.path.join(out, f"slice_C{C}.csv"), index=False)
        d = df[df["ref_ok"] & np.isfinite(df["ftle"])]
        m = {"n_points": int(len(df)), "n_ref_ok": int(d.shape[0]), "violation_rate": float(d["violation"].mean()), "n_violations": int(d["violation"].sum())}
        if d["violation"].nunique() == 2:
            m["auc_ftle_for_violation"] = float(roc_auc_score(d["violation"], d["ftle"]))
            q = pd.qcut(d["ftle"], 10, duplicates="drop")
            m["violation_rate_by_ftle_decile"] = [float(x) for x in d.groupby(q, observed=True)["violation"].mean().to_numpy()]
            top = d["ftle"] >= d["ftle"].quantile(0.9)
            m["violation_rate_top_decile_ftle"] = float(d.loc[top, "violation"].mean())
            m["violation_rate_rest"] = float(d.loc[~top, "violation"].mean())
        rep["maps"][key] = m
        print(key, {k: v for k, v in m.items() if k != "violation_rate_by_ftle_decile"}, flush=True)
    save_json(rep, os.path.join(out, "d7_report.json"))
    print("wrote", out)


def _ref_job(a):
    from orbit_ml_cr3bp.physics_3d import reference_trajectory_dense_3d
    st, T, mu = a
    sol, reached = reference_trajectory_dense_3d(list(st), T, mu)
    return sol.sol(T)[:3] if reached else np.full(3, np.nan)


def _latest_d2_times():
    import glob
    c = sorted(glob.glob(os.path.join(ROOT, "out_d", "confirm", "d2", "d2_*", "per_orbit_times.csv.gz")))
    if not c:
        raise SystemExit("no confirmation D2 per-orbit times found (run the d2 step first)")
    return c[-1]


def cmd_ias15_prep(a):
    """Tier 3. A random sample of the D2 evaluation orbits of D_confirm_D with their states, horizons and reference positions, for ias15_baseline_d.py (run with the interpreter that has REBOUND)."""
    out = rp.make_run_dir(a.out_dir, "ias15_prep", a)
    U, ref = rp.load_set("confirm")
    t = pd.read_csv(_latest_d2_times())
    pick = t.sample(min(a.n, len(t)), random_state=0)["row_id"].to_numpy()
    m = U["row_id"].isin(pick).to_numpy()
    df = U.loc[m, ["row_id", "x", "y", "z", "vx", "vy", "vz", "sim_time"]].copy()
    df[["ref_x", "ref_y", "ref_z"]] = ref[m]
    df.to_csv(os.path.join(out, "ias15_input.csv"), index=False)
    print("wrote", os.path.join(out, "ias15_input.csv"), len(df))


def cmd_ias15_combine(a):
    """Tier 3. IAS15 at its per-orbit ORACLE epsilon (steps and wall time) against the compiled policy (certified scale) and the compiled DP5(4)-PI at its equal-risk rtol, same orbits."""
    out = rp.make_run_dir(a.out_dir, "ias15_combine", a)
    t = pd.read_csv(_latest_d2_times())
    o = pd.read_csv(a.ias15_output)
    m = o.merge(t, on="row_id")
    m["ias15_ns"] = m["ias15_wall_s"] * 1e9
    ok = m["ias15_ns"].notna()
    rep = {"n": int(len(m)), "n_ias15_ok": int(ok.sum()), "median_ias15_steps": float(m.loc[ok, "ias15_steps"].median()), "median_log10_epsilon": float(m.loc[ok, "ias15_log10_eps"].median()),
           "median_ns": {"ias15_oracle_epsilon": float(m.loc[ok, "ias15_ns"].median()), "policy": float(m.loc[ok, "t_ns_policy"].median()), "DP54_PI_eqr": float(m.loc[ok, "t_ns_DP54_PI_eqr"].median())},
           "total_ns_ratio_ias15_over_policy": float(m.loc[ok, "ias15_ns"].sum() / m.loc[ok, "t_ns_policy"].sum()),
           "total_ns_ratio_ias15_over_DP54_PI_eqr": float(m.loc[ok, "ias15_ns"].sum() / m.loc[ok, "t_ns_DP54_PI_eqr"].sum()),
           "share_ias15_faster_than_policy": float((m.loc[ok, "ias15_ns"] < m.loc[ok, "t_ns_policy"]).mean()),
           "share_ias15_faster_than_DP54_PI_eqr": float((m.loc[ok, "ias15_ns"] < m.loc[ok, "t_ns_DP54_PI_eqr"]).mean())}
    save_json(rep, os.path.join(out, "ias15_report.json"))
    print(json.dumps(proto.json_safe(rep), indent=1))


def _tau_job(a):
    from orbit_ml_cr3bp.data_3d import _sundman_labels
    from orbit_ml_cr3bp.physics_3d import reference_trajectory_dense_3d
    rid, state, T, mu, tol_km, tau_c = a
    sol, reached = reference_trajectory_dense_3d(list(state), T, mu)
    if not reached:
        return {"row_id": rid}
    r = _sundman_labels(list(state), T, mu, tol_km, (sol, reached), tau_c)
    return {"row_id": rid, "rk4s_evals": r["rk4s_evals"], "rk4s_censored": r["rk4s_censored"], "rk4s_reason": r["rk4s_reason"]}


def cmd_d10(a):
    """D10. (a) tau_c in {1, 2, 4, 8}: the oracle fixed-ds Sundman cost against RK45 at the per-orbit oracle tolerance on a random subsample of D_replication (the oracle comparators
    come from Part C's apply_policy ledger of the replication run); (b) the dense sweep in ds on a smaller subsample: how often the set of passing steps is non-monotone, where, and
    whether the flips coincide with changes in the number of physical-time landing steps."""
    import glob
    from multiprocessing import Pool
    from orbit_ml_cr3bp import exploratory_d as ex, deploy_d as dd
    from orbit_ml_cr3bp.audit_3d import clopper_pearson
    out = rp.make_run_dir(a.out_dir, "d10", a)
    U, ref = rp.load_set("replication")
    ledf = sorted(glob.glob(os.path.join(ROOT, a.ledger_glob)))
    if not ledf:
        raise SystemExit("no replication ledger found")
    led = pd.read_csv(ledf[-1], low_memory=False)[["row_id", "RK45_spec_nfev"]]
    rng = np.random.RandomState(0)
    pick = rng.choice(len(U), min(a.n_tau, len(U)), replace=False)
    S, T = dd.states_T(U)
    rep = {"n_tau_sample": int(len(pick)), "tau_c": {}}
    for tc in a.tau_c:
        with Pool(a.n_jobs) as pool:
            rs = pd.DataFrame(pool.map(_tau_job, [(int(U["row_id"].iloc[i]), S[i], float(T[i]), MU, POSITION_TOL_KM, tc) for i in pick], chunksize=8))
        x = rs.merge(led, on="row_id")
        h = x["RK45_spec_nfev"].notna() & x["rk4s_evals"].notna()
        k = int((x.loc[h, "rk4s_evals"] < x.loc[h, "RK45_spec_nfev"]).sum())
        rep["tau_c"][str(tc)] = {"n": int(h.sum()), "oracle_cheaper_than_RK45_spec": k / max(h.sum(), 1), "cp95": list(clopper_pearson(k, int(h.sum()))),
                                 "median_evals": float(x.loc[h, "rk4s_evals"].median()), "censored_share": float(x["rk4s_censored"].mean())}
        print("tau_c", tc, rep["tau_c"][str(tc)], flush=True)
    sw = rng.choice(len(U), min(a.n_sweep, len(U)), replace=False)
    d = ex.dense_sweep_landing(S[sw], T[sw], ref[sw], U["dt_rk4s"].to_numpy()[sw])
    d = pd.concat([d, U.iloc[sw][["row_id", "min_r2", "regime", "h_band", "r2_band", "sim_time", "lambda_T"]].reset_index(drop=True)], axis=1)
    d.to_csv(os.path.join(out, "dense_sweep.csv"), index=False)
    k = int(d["nonmonotone"].sum())
    rep["dense_sweep"] = {"n": int(len(d)), "nonmonotone": k, "share": k / len(d), "cp95": list(clopper_pearson(k, len(d))),
                          "by_r2_band": {g: float(x["nonmonotone"].mean()) for g, x in d.groupby("r2_band")}, "by_regime": {g: float(x["nonmonotone"].mean()) for g, x in d.groupby("regime")},
                          "by_h_band": {g: float(x["nonmonotone"].mean()) for g, x in d.groupby("h_band")},
                          "median_lambda_T_nonmonotone": float(d.loc[d["nonmonotone"], "lambda_T"].median()) if k else None, "median_lambda_T_monotone": float(d.loc[~d["nonmonotone"], "lambda_T"].median()),
                          "landing_step_changes_at_flips": float(d["nf_change_at_flips"].mean()), "landing_step_changes_overall": float(d["nf_change_overall"].mean()),
                          "landing_share_at_flips": float(d["landing_share_at_flips"].mean()), "landing_share_overall": float(d["landing_share_overall"].mean())}
    save_json(rep, os.path.join(out, "d10_report.json"))
    print(json.dumps(proto.json_safe(rep["dense_sweep"]), indent=1))


def cmd_d4b(a):
    """D4b (exploratory, written after the confirmatory D4 had run). The D4 comparator, RK45 at the rtol that matches the policy's MARGINAL risk, violates in some strata far more often than the
    stratified policy (regime I at 8-20 TU: 24 %). The fair comparator for a stratified certificate is the adaptive solver certified in the same strata by the same rule: per stratum, the
    loosest rtol of the 18-value grid that passes the fixed-sequence test (safest first, level delta / G) on the calibration rows of D_confirm_D, run on D_replication. SciPy's RK45 is
    reproduced exactly (nfev) by the compiled DP5(4) with SciPy's controller, which makes the per-stratum calibration fast; DOP853 is the compiled port. A stratum with no certified rtol uses the tightest one."""
    import glob
    from scipy.stats import binom
    from orbit_ml_cr3bp import compiled_solvers_d as cs, deploy_d as dd, strata_d as sd, policy_c_3d as pc
    from orbit_ml_cr3bp.audit_3d import clopper_pearson
    out = rp.make_run_dir(a.out_dir, "d4b", a)
    (Uc, rc_), (Ue, re_) = rp.d4_sets("confirm")
    d4 = sorted(glob.glob(os.path.join(ROOT, "out_d", "confirm", "d4", "d4_*", "ledger_strat.csv.gz")))[-1]
    led = pd.read_csv(d4)
    cells_c, cells_e = sd.base_cells(Uc), sd.base_cells(Ue)
    smap, labs = sd.build_strata_map(cells_c)
    lab_c, lab_e = sd.assign_strata(cells_c, smap, labs), sd.assign_strata(cells_e, smap, labs)
    G = len(set(lab_c))
    Sc, Tc = dd.states_T(Uc)
    Se, Te = dd.states_T(Ue)
    sv = cs.Solvers(MU, fast=False)
    rep = {"n_strata": G, "methods": {}}
    for name, label in (("DP54_scipy", "RK45"), ("DOP853", "DOP853")):
        rt_by = {}
        for g in sorted(set(lab_c)):
            idx = np.where(lab_c == g)[0]
            cache = {}

            def viol(rt, idx=idx):
                if rt not in cache:
                    ok, y, nf, _ = sv.run(name, Sc[idx], Tc[idx], rt, parallel=True)
                    cache[rt] = int((~(ok & (sv.errors_km(y, rc_[idx]) <= POSITION_TOL_KM))).sum())
                return cache[rt], len(idx)
            cert = pc.certify_sequence(viol, cs.RTOL_GRID_B, 0.05, 0.1 / G, "fixed_sequence") if len(idx) >= sd.MIN_N_CERT else {"chosen_scale": None}
            rt_by[g] = cert["chosen_scale"] if cert["chosen_scale"] is not None else float(min(cs.RTOL_GRID_B))
        nfev = np.full(len(Ue), np.nan)
        met = np.zeros(len(Ue), dtype=bool)
        for g in sorted(set(lab_e)):
            idx = np.where(lab_e == g)[0]
            ok, y, nf, _ = sv.run(name, Se[idx], Te[idx], rt_by.get(g, min(cs.RTOL_GRID_B)), parallel=True)
            m = ok & (sv.errors_km(y, re_[idx]) <= POSITION_TOL_KM)
            met[idx], nfev[idx] = m, np.where(ok, nf, np.nan)
        e = pd.DataFrame({"row_id": Ue["row_id"].to_numpy(), "stratum": lab_e, "comp_met": met, "comp_nfev": nfev})
        m = led.merge(e[["row_id", "comp_met", "comp_nfev"]], on="row_id")
        has = m["comp_nfev"].notna()
        k = int((m["meets_tol"] & has & (m["C_ml_tot"] < m["comp_nfev"])).sum())
        per = []
        for g, x in m.merge(e[["row_id", "stratum"]].rename(columns={"stratum": "st"}), on="row_id").groupby("st"):
            kk = int((~x["comp_met"]).sum())
            per.append({"stratum": g, "n": int(len(x)), "comparator_rtol": rt_by.get(g), "comparator_violation_rate": kk / len(x), "comparator_cp95": list(clopper_pearson(kk, len(x))),
                        "policy_violation_rate": float((~x["meets_tol"]).mean()), "share_policy_cheaper": float((x["meets_tol"] & x["comp_nfev"].notna() & (x["C_ml_tot"] < x["comp_nfev"])).sum() / max(x["comp_nfev"].notna().sum(), 1))})
        rep["methods"][label] = {"overall_comparator_violation_rate": float((~m["comp_met"]).mean()), "overall_policy_violation_rate": float((~m["meets_tol"]).mean()),
                                 "share_policy_cheaper": k / int(has.sum()), "cp95": list(clopper_pearson(k, int(has.sum()))),
                                 "by_h_band": {g: float((x["meets_tol"] & x["comp_nfev"].notna() & (x["C_ml_tot"] < x["comp_nfev"])).sum() / max(x["comp_nfev"].notna().sum(), 1)) for g, x in m.groupby("h_band")},
                                 "by_regime": {g: float((x["meets_tol"] & x["comp_nfev"].notna() & (x["C_ml_tot"] < x["comp_nfev"])).sum() / max(x["comp_nfev"].notna().sum(), 1)) for g, x in m.groupby("regime")},
                                 "median_comparator_nfev_met": float(m.loc[m["comp_met"], "comp_nfev"].median()), "median_policy_total_met": float(m.loc[m["meets_tol"], "C_ml_tot"].median()),
                                 "strata": per}
        print(label, {k_: v for k_, v in rep["methods"][label].items() if k_ != "strata"}, flush=True)
    save_json(rep, os.path.join(out, "d4b_report.json"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("d6")
    sp.add_argument("--test_set", default="confirm"); sp.add_argument("--max_train", type=int, default=60000); sp.add_argument("--n_perm", type=int, default=5)
    sp.add_argument("--n_iter", type=int, default=200); sp.add_argument("--out_dir", default="out_d/explore"); sp.set_defaults(fn=cmd_d6)
    sp = sub.add_parser("symplectic")
    sp.add_argument("--n", type=int, default=100); sp.add_argument("--dts", type=float, nargs="+", default=[0.2, 0.05, 0.01])
    sp.add_argument("--out_dir", default="out_d/explore"); sp.set_defaults(fn=cmd_symplectic)
    sp = sub.add_parser("d12")
    sp.add_argument("--set", default="confirm"); sp.add_argument("--frac", type=float, default=0.01); sp.add_argument("--n_jobs", type=int, default=8)
    sp.add_argument("--out_dir", default="out_d/explore"); sp.set_defaults(fn=cmd_d12)
    sp = sub.add_parser("d7")
    sp.add_argument("--C", type=float, nargs="+", default=[3.19, 3.10, 2.95]); sp.add_argument("--n", type=int, default=120); sp.add_argument("--T", type=float, default=8.0)
    sp.add_argument("--xlim", type=float, nargs=2, default=[-1.2, 1.3]); sp.add_argument("--vlim", type=float, nargs=2, default=[-1.6, 1.6])
    sp.add_argument("--n_jobs", type=int, default=8); sp.add_argument("--out_dir", default="out_d/explore"); sp.set_defaults(fn=cmd_d7)
    sp = sub.add_parser("d10")
    sp.add_argument("--tau_c", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0]); sp.add_argument("--n_tau", type=int, default=1500); sp.add_argument("--n_sweep", type=int, default=500)
    sp.add_argument("--ledger_glob", default="out_d/confirm/d3/sund/apply_policy_*/ledger.csv"); sp.add_argument("--n_jobs", type=int, default=8)
    sp.add_argument("--out_dir", default="out_d/explore"); sp.set_defaults(fn=cmd_d10)
    sp = sub.add_parser("d4b")
    sp.add_argument("--out_dir", default="out_d/explore"); sp.set_defaults(fn=cmd_d4b)
    sp = sub.add_parser("ias15_prep")
    sp.add_argument("--n", type=int, default=300); sp.add_argument("--out_dir", default="out_d/explore"); sp.set_defaults(fn=cmd_ias15_prep)
    sp = sub.add_parser("ias15_combine")
    sp.add_argument("--ias15_output", required=True); sp.add_argument("--out_dir", default="out_d/explore"); sp.set_defaults(fn=cmd_ias15_combine)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
