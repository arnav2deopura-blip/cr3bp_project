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

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
