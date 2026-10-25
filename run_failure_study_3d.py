"""
Failure study (Part B) command line. Every command writes into its own timestamped folder under --out_dir with a manifest.json.

  policy     fit the deployable model on a training slice, certify the ML scales and the adaptive rtols on a calibration slice,
             run the ML policy and the adaptive comparators (spec / cal / equal-risk) on an evaluation slice, and write the
             per-orbit ledger (ledger.csv) plus the calibration chains (policy_report.json)

Analysis commands (decompose, e1, e3a, ...) read a ledger and are added below as the study proceeds. Run with the project's
conda env:  /Users/arnavdeopura/miniforge3/envs/orbit-ml/bin/python run_failure_study_3d.py policy --data data_b/....csv
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orbit_ml_cr3bp.config import MU, POSITION_TOL_KM  # noqa: E402
from orbit_ml_cr3bp import protocol_3d as proto  # noqa: E402
from orbit_ml_cr3bp import policy_b_3d as pb  # noqa: E402
from orbit_ml_cr3bp import failure_study_3d as fs  # noqa: E402
from orbit_ml_cr3bp.audit_3d import ReferenceCache  # noqa: E402
from orbit_ml_cr3bp.fair_baseline_3d import measure_prediction_cost  # noqa: E402
from orbit_ml_cr3bp.modeling_3d import predict_hurdle  # noqa: E402


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        return None


def make_run_dir(base, name, args):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    d = os.path.join(base, f"{name}_{stamp}")
    os.makedirs(d, exist_ok=True)
    man = {"command": name, "utc": stamp, "args": proto.json_safe(vars(args)), "git_commit": _git_commit(),
           "python": sys.version, "argv": sys.argv}
    data = getattr(args, "data", None)
    if data and os.path.exists(data):
        man["data_sha256"] = proto.sha256_file(data)
        man["data_metadata"] = proto.read_metadata(data)
    with open(os.path.join(d, "manifest.json"), "w") as f:
        json.dump(man, f, indent=2)
    return d


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(proto.json_safe(obj), f, indent=2)


_BOOL_COLS = ("euler_censored", "rk4_censored", "leapfrog_censored", "ftle_nan", "reference_divergent")


def load_partb(path, mu=MU):
    d = pd.read_csv(path, low_memory=False)
    for c in _BOOL_COLS:   # a row without labels (ftle_nan) makes pandas read these as object; make them real bools
        if c in d.columns:
            d[c] = d[c].map({True: True, False: False, "True": True, "False": False}).fillna(False).astype(bool)
    d["row_id"] = np.arange(len(d))
    meta = proto.read_metadata(path) or {}
    if meta.get("label_pipeline_version") != 4:
        raise ValueError(f"{path}: label_pipeline_version={meta.get('label_pipeline_version')}; Part B needs 4 (--pipeline v4)")
    return d, meta


def split_rows(U, n_cal, n_eval, seed):
    perm = np.random.RandomState(seed).permutation(len(U))
    cal = np.sort(perm[:n_cal])
    ev = np.sort(perm[n_cal:n_cal + n_eval])
    train = np.sort(perm[n_cal + n_eval:])
    return train, cal, ev


def cmd_policy(a):
    out = make_run_dir(a.out_dir, "policy", a)
    t_start = time.time()
    d, meta = load_partb(a.data)
    if a.guard_lock:
        lock = proto.find_lock_for(a.data)
        if lock:
            raise ValueError(f"{a.data} has a lock file {lock}")
    U = pb.usable_rows(d).reset_index(drop=True)
    U = fs.add_derived(U)
    train, cal, ev = split_rows(U, a.n_cal, a.n_eval, a.seed)
    feats = pb.FEATS_FTLE if a.features == "ftle" else pb.FEATS_STATE
    print(f"usable {len(U)}; train {len(train)}, calibration {len(cal)}, evaluation {len(ev)}; features={a.features}", flush=True)
    models = pb.fit_policy_models(U, train, features=feats, quantile=a.quantile, seed=a.seed)
    cal_df = U.iloc[cal].reset_index(drop=True)
    ev_df = U.iloc[ev].reset_index(drop=True)
    cal_pred = predict_hurdle(models, cal_df)
    ev_pred = predict_hurdle(models, ev_df)

    # --- certified ML scales (lazy scan = the same "largest certified" value as the full Bonferroni chain)
    cache = ReferenceCache(MU)
    scale_rep, scales = {}, {}
    for k in pb.CANDIDATES:
        r = pb.certify_scale_lazy(cal_df, cal_pred, k, tol_km=a.tol_km, mu=MU, cache=cache)
        scale_rep[k] = r
        scales[k] = r["chosen_scale"]
        print(f"  scale[{k}] = {r['chosen_scale']}  (n={r['n']}, violations={r['violations']})", flush=True)

    # --- realized violation rate of the deployed policy on the calibration rows (target for the equal-risk adaptive rtol)
    cal_res = pb.evaluate_orbits(cal_df, cal_pred, scales, rtols={}, do_spec=False, tol_km=a.tol_km, n_jobs=a.n_jobs,
                                 progress=False)
    cal_ok = cal_res["ml_ok"].fillna(False).astype(bool) & (cal_res["ml_err_km"] <= a.tol_km)
    has = cal_res["ml_integrator"].notna()
    cal_rate = float(1.0 - cal_ok[has].mean()) if has.any() else float("nan")
    print(f"  deployed-policy violation rate on calibration rows: {cal_rate:.4f} (n={int(has.sum())})", flush=True)

    # --- adaptive rtols on the SAME calibration rows (equal n_cal, equal grid size)
    items = pb.reference_items(cal_df, MU)
    rtols, rtol_rep = {}, {}
    for m in pb.METHODS:
        r = pb.certify_rtol_lazy(cal_df, m, tol_km=a.tol_km, mu=MU, equal_risk_target=cal_rate, n_jobs=a.n_jobs, items=items)
        rtol_rep[m] = r
        rtols[(m, "cal")] = r["chosen_rtol"]
        rtols[(m, "eqr")] = r["equal_risk_rtol"]
        print(f"  {m}: certified rtol {r['chosen_rtol']}, equal-risk rtol {r['equal_risk_rtol']}", flush=True)

    # --- evaluation pass
    res = pb.evaluate_orbits(ev_df, ev_pred, scales, rtols=rtols, do_spec=not a.no_spec, tol_km=a.tol_km, n_jobs=a.n_jobs)
    cp = measure_prediction_cost(models, ev_df, pb.CANDIDATES, MU)
    raw = pb.candidate_raw_predictions(ev_df, ev_pred)
    led = fs.build_ledger(ev_df, res.merge(raw, on="row_id"), cp["evals_equiv"], c_feat=1.0, tol_km=a.tol_km)
    led.to_csv(os.path.join(out, "ledger.csv"), index=False)
    cal_df.assign(split="cal").to_csv(os.path.join(out, "cal_rows.csv"), index=False)
    pd.Series(U.iloc[train]["row_id"].to_numpy()).to_csv(os.path.join(out, "train_row_ids.csv"), index=False, header=["row_id"])
    import joblib
    joblib.dump({"models": models, "scales": scales, "rtols": {f"{m}|{t}": v for (m, t), v in rtols.items()},
                 "features": feats, "quantile": a.quantile}, os.path.join(out, "policy.joblib"))
    rep = {"args": vars(a), "n_usable": int(len(U)), "n_train": int(len(train)), "n_cal": int(len(cal)), "n_eval": int(len(ev)),
           "scales": scales, "scale_chains": {k: v["chain"] for k, v in scale_rep.items()},
           "cal_policy_violation_rate": cal_rate, "rtol_calibration": rtol_rep, "prediction_cost": cp,
           "risk_ledger": fs.risk_ledger(led, a.tol_km), "wall_seconds": time.time() - t_start}
    save_json(rep, os.path.join(out, "policy_report.json"))
    print(json.dumps(proto.json_safe(rep["risk_ledger"]), indent=1))
    print(f"wrote {out}  ({time.time() - t_start:.0f}s)")


def _cp(k, n):
    from orbit_ml_cr3bp.audit_3d import clopper_pearson
    return [float(x) for x in clopper_pearson(int(k), int(n))]


def _ols_slope(x, y, n_boot=2000, seed=0):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 10 or np.ptp(x) == 0:
        return {"n": int(len(x))}
    b, a0 = np.polyfit(x, y, 1)
    rng = np.random.RandomState(seed)
    bs = []
    for _ in range(n_boot):
        i = rng.randint(0, len(x), len(x))
        bs.append(np.polyfit(x[i], y[i], 1)[0])
    return {"n": int(len(x)), "slope": float(b), "intercept": float(a0),
            "slope_ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]}


def comparator_summary(led, comp, tol_km):
    col = f"{comp}_nfev"
    cls, dec = fs.classify(led, comp, tol_km)
    has = led[col].notna() & led["C_fix_star"].notna()
    sub = led[has]
    gap = np.log10(sub["C_fix_star"] / sub[col])
    k_cheaper = int((sub["C_fix_star"] < sub[col]).sum())
    rep = {"n_with_comparator": int(has.sum()),
           "oracle_fixed_cheaper": {"k": k_cheaper, "n": int(len(sub)), "share": k_cheaper / max(len(sub), 1),
                                    "ci95": _cp(k_cheaper, len(sub))},
           "paradigm_gap_dex_median": fs.bootstrap_ci(gap)}
    ok = dec["total"].notna()
    rep["n_decomposed"] = int(ok.sum())
    rep["decomposition_median_dex"] = {t: fs.bootstrap_ci(dec.loc[ok, t]) for t in ("P", "S", "Q", "K", "O", "R", "total")}
    # means are additive (P+S+Q+K+O+R = total exactly); medians of the terms do not add up to the median total
    rep["decomposition_mean_dex"] = {t: fs.bootstrap_ci(dec.loc[ok, t], stat=np.mean) for t in ("P", "S", "Q", "K", "O", "R", "total")}
    tot = dec.loc[ok, ["P", "S", "Q", "K", "O"]].clip(lower=0).sum(axis=1)
    rep["positive_term_share_of_gap"] = {t: float((dec.loc[ok, t].clip(lower=0) / tot.replace(0, np.nan)).mean())
                                         for t in ("P", "S", "Q", "K", "O")}
    rep["class_counts"] = cls.value_counts().to_dict()
    rep["ml_win"] = {"k": int((cls == "ml-win").sum()), "n": int(len(led)), "share": float((cls == "ml-win").mean())}
    loss = cls.str.startswith("cost-loss")
    rep["cost_loss_dominant_counts"] = cls[loss].str.split(":").str[1].value_counts().to_dict()
    rep["cost_loss_paradigm_bound_share"] = float(dec.loc[loss, "paradigm_bound"].astype(float).mean()) if loss.any() else None
    rep["paradigm_vs_delta"] = {
        "log_delta_phys": {"spearman": fs.spearman_bootstrap(sub["log_delta_phys"], gap), "ols": _ols_slope(sub["log_delta_phys"], gap)},
        "log_delta_emp": {"spearman": fs.spearman_bootstrap(sub["log_delta_emp"], gap), "ols": _ols_slope(sub["log_delta_emp"], gap)},
    }
    return rep, cls, dec


def group_table(led, cls, dec, comp, by):
    col = f"{comp}_nfev"
    rows = []
    for key, g in led.groupby(by, dropna=False):
        idx = g.index
        has = g[col].notna() & g["C_fix_star"].notna()
        gap = np.log10(g.loc[has, "C_fix_star"] / g.loc[has, col])
        d = dec.loc[idx]
        rows.append({by: key, "n": len(g), "n_viol": int(g["violation"].sum()),
                     "viol_rate": float(g["violation"].mean()),
                     "paradigm_gap_median": float(gap.median()) if len(gap) else np.nan,
                     "oracle_fixed_cheaper_share": float((g.loc[has, "C_fix_star"] < g.loc[has, col]).mean()) if has.any() else np.nan,
                     "P_med": float(d["P"].median()), "S_med": float(d["S"].median()), "Q_med": float(d["Q"].median()),
                     "K_med": float(d["K"].median()), "O_med": float(d["O"].median()), "total_med": float(d["total"].median()),
                     "ml_win_share": float((cls.loc[idx] == "ml-win").mean()),
                     "paradigm_bound_share_of_losses": float(d["paradigm_bound"].astype(float)[cls.loc[idx].str.startswith("cost-loss")].mean())
                     if cls.loc[idx].str.startswith("cost-loss").any() else np.nan})
    return pd.DataFrame(rows)


def cmd_decompose(a):
    out = make_run_dir(a.out_dir, "decompose", a)
    led = pd.read_csv(a.ledger)
    led["cell"] = fs.assign_cells(led)
    rep = {"ledger": a.ledger, "n_orbits": int(len(led)), "risk_ledger": fs.risk_ledger(led, a.tol_km),
           "cells": led["cell"].value_counts().to_dict()}
    for comp in fs.COMPARATORS:
        if f"{comp}_nfev" not in led.columns or led[f"{comp}_nfev"].notna().sum() == 0:
            continue
        r, cls, dec = comparator_summary(led, comp, a.tol_km)
        rep[comp] = r
        for by in ("regime", "cell", "r2_band", "h_band"):
            group_table(led, cls, dec, comp, by).to_csv(os.path.join(out, f"{comp}_by_{by}.csv"), index=False)
        pd.concat([led[["row_id", "regime", "cell"]], cls.rename("class"), dec], axis=1).to_csv(
            os.path.join(out, f"{comp}_orbit_classes.csv"), index=False)
        print(f"{comp}: oracle fixed cheaper on {r['oracle_fixed_cheaper']['share']:.3f} of {r['n_with_comparator']}; "
              f"median P {r['decomposition_median_dex']['P'][0]:.3f} S {r['decomposition_median_dex']['S'][0]:.3f} "
              f"Q {r['decomposition_median_dex']['Q'][0]:.3f} K {r['decomposition_median_dex']['K'][0]:.3f} "
              f"O {r['decomposition_median_dex']['O'][0]:.3f} total {r['decomposition_median_dex']['total'][0]:.3f}; "
              f"ML wins {r['ml_win']['share']:.3f}", flush=True)
    save_json(rep, os.path.join(out, "decompose_report.json"))
    print("wrote", out)


# ---------------------------------------------------------------------------------------------------------------------
# E1: reference validity map
# ---------------------------------------------------------------------------------------------------------------------

def _e1_job(a):
    from scipy.integrate import solve_ivp
    from orbit_ml_cr3bp.physics_3d import (reference_trajectory_dense_3d, _cr3bp_derivatives_3d_numba)
    from orbit_ml_cr3bp.config import collision_radii, EJECTION_RADIUS, du_km_for_mu
    rid, state, T = a
    state = np.asarray(state, dtype=float)
    out = {"row_id": rid}
    sol, reached = reference_trajectory_dense_3d(list(state), T, MU)
    out["ref_reached"] = bool(reached)
    out["ref_event"] = int(getattr(sol, "reference_event", 0))
    out["ref_t_end"] = float(sol.t[-1])
    cr1, cr2 = collision_radii(MU)
    n = [0]

    def rhs(t, y):
        n[0] += 1
        if n[0] > 3_000_000:
            raise RuntimeError("cap")
        return _cr3bp_derivatives_3d_numba(y, MU)

    def e1(t, y):
        return np.sqrt((y[0] + MU) ** 2 + y[1] ** 2 + y[2] ** 2) - cr1
    e1.terminal, e1.direction = True, -1

    def e2(t, y):
        return np.sqrt((y[0] + MU - 1.0) ** 2 + y[1] ** 2 + y[2] ** 2) - cr2
    e2.terminal, e2.direction = True, -1

    def e3(t, y):
        return np.sqrt(y[0] ** 2 + y[1] ** 2 + y[2] ** 2) - EJECTION_RADIUS
    e3.terminal, e3.direction = True, 1
    try:
        rad = solve_ivp(rhs, (0.0, T), state, method="Radau", rtol=1e-12, atol=1e-12, events=[e1, e2, e3])
    except RuntimeError:
        out["radau_capped"] = True
        return out
    out["radau_capped"] = False
    r_reached = bool(rad.t[-1] >= T - 1e-9)
    ev = 0
    if not r_reached:
        ev = 9
        for i, te in enumerate(rad.t_events):
            if len(te) > 0:
                ev = i + 1
                break
    out["radau_reached"] = r_reached
    out["radau_event"] = ev
    out["radau_t_end"] = float(rad.t[-1])
    if reached and r_reached:
        out["diff_km"] = float(np.linalg.norm(sol.sol(T)[:3] - rad.y[:3, -1])) * du_km_for_mu(MU)
    return out


def cmd_e1(a):
    from multiprocessing import Pool
    out = make_run_dir(a.out_dir, "e1", a)
    d, _ = load_partb(a.data)
    ok = d[~d["ftle_nan"] & ~d["reference_divergent"]]
    bad = d[~d["ftle_nan"] & d["reference_divergent"] & (d["reference_outcome"] != "initial_inside_body")]
    rng = np.random.RandomState(a.seed)
    pick_ok = ok.iloc[rng.choice(len(ok), size=min(a.n_ok, len(ok)), replace=False)]
    pick_bad = bad.iloc[rng.choice(len(bad), size=min(a.n_doomed, len(bad)), replace=False)]
    sel = pd.concat([pick_ok, pick_bad])
    jobs = [(int(r.row_id), [r.x, r.y, r.z, r.vx, r.vy, r.vz], float(r.sim_time)) for r in sel.itertuples()]
    with Pool(a.n_jobs) as pool:
        res = pd.DataFrame(pool.map(_e1_job, jobs, chunksize=2))
    res = res.merge(sel[["row_id", "sim_time", "reference_divergent", "reference_outcome", "ref_check_km", "min_r2",
                         "jacobi_3d", "orbit_label"]], on="row_id")
    res.to_csv(os.path.join(out, "e1_orbits.csv"), index=False)
    tol = a.tol_km
    comp = res[res["ref_reached"] & res["radau_reached"].fillna(False).astype(bool)]
    rep = {"n_usable_sampled": int(len(pick_ok)), "n_doomed_sampled": int(len(pick_bad)),
           "radau_capped": int(res["radau_capped"].sum()),
           "usable_pairs_compared": int(len(comp)),
           "diff_km_quantiles": {q: float(comp["diff_km"].quantile(q)) for q in (0.5, 0.9, 0.99, 1.0)} if len(comp) else {},
           "f0_ref_threshold_km": 0.1 * tol, "n_f0_ref": int((comp["diff_km"] > 0.1 * tol).sum()),
           "frac_within_1km": float((comp["diff_km"] <= 1.0).mean()) if len(comp) else None,
           "usable_orbits_where_radau_disagrees_on_outcome": int(((res["ref_reached"]) & (~res["radau_reached"].fillna(True).astype(bool))).sum()),
           "doomed_orbits_where_radau_reaches_horizon": int(((~res["ref_reached"]) & (res["radau_reached"].fillna(False).astype(bool))).sum()),
           "doomed_event_agreement": float((res.loc[~res["ref_reached"], "ref_event"] == res.loc[~res["ref_reached"], "radau_event"]).mean())
           if (~res["ref_reached"]).any() else None,
           "doomed_t_end_abs_diff_median": float((res.loc[~res["ref_reached"], "ref_t_end"] - res.loc[~res["ref_reached"], "radau_t_end"]).abs().median())
           if (~res["ref_reached"]).any() else None}
    save_json(rep, os.path.join(out, "e1_report.json"))
    print(json.dumps(proto.json_safe(rep), indent=1))
    print("wrote", out)


# ---------------------------------------------------------------------------------------------------------------------
# E3a: label sensitivity to tiny initial-condition perturbations (the information floor)
# ---------------------------------------------------------------------------------------------------------------------

def _clone_state(state, eps, rng):
    """Random perturbation: position part of norm eps (DU), velocity part of norm eps (DU/TU), uniform directions."""
    dp = rng.normal(size=3)
    dv = rng.normal(size=3)
    dp = eps * dp / np.linalg.norm(dp)
    dv = eps * dv / np.linalg.norm(dv)
    return np.asarray(state, dtype=float) + np.concatenate([dp, dv])


def _e3a_job(a):
    from orbit_ml_cr3bp.data_3d import label_orbit_accuracy_3d
    rid, state, T, eps, n_clones, seed = a
    rng = np.random.RandomState(seed)
    rows = []
    for c in range(n_clones):
        st = _clone_state(state, eps, rng)
        lab = label_orbit_accuracy_3d(list(st), T, MU, tol_km=POSITION_TOL_KM, monotone=True, ceiling_mode="physical")
        rows.append({"row_id": rid, "eps": eps, "clone": c, "diverged": bool(lab["reference_divergent"]),
                     "dt_rk4": lab["dt_rk4"], "dt_leapfrog": lab["dt_leapfrog"],
                     "rk4_ok": lab["rk4_reason"] == 0, "leapfrog_ok": lab["leapfrog_reason"] == 0})
    return rows


def cmd_e3a(a):
    from multiprocessing import Pool
    out = make_run_dir(a.out_dir, "e3a", a)
    led = pd.read_csv(a.ledger)
    led["cell"] = fs.assign_cells(led)
    rng = np.random.RandomState(a.seed)
    picks = []
    for cell, g in led.groupby("cell"):
        picks.append(g.iloc[rng.choice(len(g), size=min(a.per_cell, len(g)), replace=False)])
    base = pd.concat(picks)
    print(f"{len(base)} base orbits from {led['cell'].nunique()} cells; {a.clones} clones x {len(a.eps)} eps", flush=True)
    jobs = []
    for r in base.itertuples():
        for j, eps in enumerate(a.eps):
            jobs.append((int(r.row_id), [r.x, r.y, r.z, r.vx, r.vy, r.vz], float(r.sim_time), float(eps), a.clones,
                         int(a.seed * 1_000_003 + r.row_id * 7 + j)))
    t0 = time.time()
    rows = []
    with Pool(a.n_jobs) as pool:
        for k, rr in enumerate(pool.imap_unordered(_e3a_job, jobs, chunksize=2)):
            rows.extend(rr)
            if (k + 1) % 200 == 0:
                print(f"  {k + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    cl = pd.DataFrame(rows)
    cl.to_csv(os.path.join(out, "e3a_clones.csv"), index=False)
    recs = []
    for (rid, eps), g in cl.groupby(["row_id", "eps"]):
        rec = {"row_id": rid, "eps": eps, "n_clones": len(g), "flip_rate": float(g["diverged"].mean())}
        for k in ("rk4", "leapfrog"):
            gg = g[(~g["diverged"]) & g[f"{k}_ok"]]
            v = np.log10(gg[f"dt_{k}"])
            rec[f"sigma_{k}"] = float(v.std(ddof=1)) if len(v) >= 3 else np.nan
            rec[f"n_{k}"] = int(len(v))
            rec[f"median_dt_{k}"] = float(gg[f"dt_{k}"].median()) if len(gg) else np.nan
        recs.append(rec)
    sig = pd.DataFrame(recs)
    sig = sig.merge(led[["row_id", "cell", "regime", "lambda_T", "sim_time", "min_r2", "ml_integrator", "ml_raw_dt", "dstar_chosen",
                         "violation", "over_pred"]], on="row_id", how="left")
    sig["sigma_chosen"] = np.where(sig["ml_integrator"] == "rk4", sig["sigma_rk4"], sig["sigma_leapfrog"])
    sig["abs_resid_dex"] = np.abs(np.log10(sig["ml_raw_dt"] / sig["dstar_chosen"]))
    sig.to_csv(os.path.join(out, "e3a_sigma.csv"), index=False)
    rep = {"n_base": int(base["row_id"].nunique()), "eps_levels": a.eps, "clones": a.clones, "per_eps": {}}
    for eps, g in sig.groupby("eps"):
        s = g["sigma_chosen"].dropna()
        rho = fs.spearman_bootstrap(g["lambda_T"], g["sigma_chosen"])
        rep["per_eps"][str(eps)] = {
            "median_sigma_dex": float(s.median()), "p90_sigma_dex": float(s.quantile(0.9)),
            "median_flip_rate": float(g["flip_rate"].median()), "spearman_lambdaT_sigma": rho,
            "median_abs_resid_over_sigma": float((g["abs_resid_dex"] / g["sigma_chosen"]).replace([np.inf, -np.inf], np.nan).median()),
            "share_resid_below_sigma": float((g["abs_resid_dex"] <= g["sigma_chosen"]).mean())}
    save_json(rep, os.path.join(out, "e3a_report.json"))
    print(json.dumps(proto.json_safe(rep), indent=1))
    print("wrote", out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--data", required=True)
        sp.add_argument("--out_dir", default="out_b")
        sp.add_argument("--seed", type=int, default=42)
        sp.add_argument("--tol_km", type=float, default=POSITION_TOL_KM)
        sp.add_argument("--n_jobs", type=int, default=8)

    sp = sub.add_parser("policy")
    common(sp)
    sp.add_argument("--n_cal", type=int, default=2000)
    sp.add_argument("--n_eval", type=int, default=3000)
    sp.add_argument("--features", choices=["state", "ftle"], default="state")
    sp.add_argument("--quantile", type=float, default=pb.PINNED_QUANTILE)
    sp.add_argument("--no_spec", action="store_true")
    sp.add_argument("--guard_lock", action="store_true")
    sp.set_defaults(fn=cmd_policy)

    sp = sub.add_parser("decompose")
    sp.add_argument("--ledger", required=True)
    sp.add_argument("--out_dir", default="out_b")
    sp.add_argument("--tol_km", type=float, default=POSITION_TOL_KM)
    sp.set_defaults(fn=cmd_decompose)

    sp = sub.add_parser("e1")
    common(sp)
    sp.add_argument("--n_ok", type=int, default=600)
    sp.add_argument("--n_doomed", type=int, default=300)
    sp.set_defaults(fn=cmd_e1)

    sp = sub.add_parser("e3a")
    sp.add_argument("--ledger", required=True)
    sp.add_argument("--out_dir", default="out_b")
    sp.add_argument("--seed", type=int, default=7)
    sp.add_argument("--n_jobs", type=int, default=8)
    sp.add_argument("--per_cell", type=int, default=10)
    sp.add_argument("--clones", type=int, default=16)
    sp.add_argument("--eps", type=float, nargs="+", default=[1e-9, 1e-7, 1e-5])
    sp.set_defaults(fn=cmd_e3a)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
