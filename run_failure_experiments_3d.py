"""
Failure-study experiments that read a `policy` run folder (run_failure_study_3d.py policy):

  e4   learning curves per cell (data-starved or at a floor), reliability of the quantile model, kNN out-of-distribution score
  e5   calibration alternatives at equal certified risk: global, Mondrian by regime, by predicted dispersion, abstention
  e9   step-over analysis of violations (error / tol against dt / tau_min) and a cheap runtime guard
  e10  physics-rule baseline, break-even inference cost per cell

All of these are exploratory on D_explore unless a pre-registration says otherwise.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orbit_ml_cr3bp.config import MU, POSITION_TOL_KM  # noqa: E402
from orbit_ml_cr3bp import failure_study_3d as fs  # noqa: E402
from orbit_ml_cr3bp import experiments_b_3d as xb  # noqa: E402
from orbit_ml_cr3bp import policy_b_3d as pb  # noqa: E402
from orbit_ml_cr3bp import protocol_3d as proto  # noqa: E402
from run_failure_study_3d import make_run_dir, save_json, load_partb  # noqa: E402


def load_policy_dir(a):
    d, meta = load_partb(a.data)
    U = fs.add_derived(pb.usable_rows(d).reset_index(drop=True))
    pdir = a.policy_dir
    led = pd.read_csv(os.path.join(pdir, "ledger.csv"), low_memory=False)
    led["cell"] = fs.assign_cells(led)
    train_ids = set(pd.read_csv(os.path.join(pdir, "train_row_ids.csv"))["row_id"])
    cal_ids = set(pd.read_csv(os.path.join(pdir, "cal_rows.csv"), low_memory=False)["row_id"])
    # row_id in the policy run is the position in the USABLE frame U (reset_index) -- see cmd_policy: U is rebuilt there and
    # row_id re-assigned inside evaluate_orbits from df["row_id"], which load_partb set on the FULL frame. Use that same id.
    cal = U[U["row_id"].isin(cal_ids)].reset_index(drop=True)
    if getattr(a, "train_data", None):      # frozen-model runs: the model's training rows live in another dataset
        dt, _ = load_partb(a.train_data)
        Ut = fs.add_derived(pb.usable_rows(dt).reset_index(drop=True))
        tids = set(pd.read_csv(os.path.join(a.train_policy_dir, "train_row_ids.csv"))["row_id"])
        train = Ut[Ut["row_id"].isin(tids)].reset_index(drop=True)
    else:
        train = U[U["row_id"].isin(train_ids)].reset_index(drop=True)
    rep = json.load(open(os.path.join(pdir, "policy_report.json")))
    return U, train, cal, led, rep


def cmd_e4(a):
    out = make_run_dir(a.out_dir, "e4", a)
    U, train, cal, led, rep = load_policy_dir(a)
    print(f"train {len(train)}, cal {len(cal)}, eval {len(led)}", flush=True)
    sizes = [s for s in (1000, 2000, 5000) if s < len(train)] + [len(train)]
    t0 = time.time()
    lc = xb.learning_curve(train, led, "rk4", sizes, range(a.seeds), q=a.quantile)
    lc.to_csv(os.path.join(out, "learning_curve_rk4.csv"), index=False)
    print(f"learning curves {time.time() - t0:.0f}s", flush=True)
    groups = ["ALL"] + sorted(led["cell"].unique())
    slopes = [xb.learning_slope(lc, g) for g in groups]
    pd.DataFrame(slopes).to_csv(os.path.join(out, "learning_slopes.csv"), index=False)
    rel = xb.reliability(train, led, "rk4", seed=a.seed)
    rel.to_csv(os.path.join(out, "reliability_rk4.csv"), index=False)
    dist, thr, flag = xb.knn_ood(train, cal, led)
    ood = pd.DataFrame({"row_id": led["row_id"], "ood_dist": dist, "ood_flag": flag})
    ood.to_csv(os.path.join(out, "ood.csv"), index=False)
    v = led["violation"].to_numpy()
    rep_out = {"n_train": len(train), "sizes": sizes, "slopes_all": slopes[0],
               "cells_still_improving": [s["group"] for s in slopes[1:] if s.get("slope_ci95") and s["slope_ci95"][1] < 0],
               "cells_flat": [s["group"] for s in slopes[1:] if s.get("slope_ci95") and s["slope_ci95"][0] <= 0 <= s["slope_ci95"][1]],
               "ood": {"threshold": thr, "flag_rate": float(flag.mean()), "violation_rate_flagged": float(v[flag].mean()) if flag.any() else None,
                       "violation_rate_unflagged": float(v[~flag].mean()), "auc_dist_for_violation": xb.auc_score(dist[v], dist[~v]),
                       "share_of_violations_flagged": float(flag[v].mean()) if v.any() else None},
               "reliability_all": rel[rel["group"] == "ALL"][["q", "n", "empirical"]].to_dict(orient="records")}
    save_json(rep_out, os.path.join(out, "e4_report.json"))
    print(json.dumps(proto.json_safe(rep_out), indent=1))
    print("wrote", out)


# ---------------------------------------------------------------------------------------------------------------------
# E5: calibration alternatives at equal certified risk
# ---------------------------------------------------------------------------------------------------------------------

def _merge_groups(keys, order, min_n):
    """Merge adjacent categories (in `order`) until every group has >= min_n rows. Returns {category: group_label}."""
    counts = {k: int((keys == k).sum()) for k in order}
    groups = [[k] for k in order if counts[k] > 0]
    while len(groups) > 1:
        sizes = [sum(counts[k] for k in g) for g in groups]
        i = int(np.argmin(sizes))
        if sizes[i] >= min_n:
            break
        j = i - 1 if i > 0 and (i == len(groups) - 1 or sizes[i - 1] <= sizes[i + 1]) else i + 1
        lo, hi = sorted((i, j))
        groups[lo] = groups[lo] + groups[hi]
        del groups[hi]
    return {k: "+".join(g) for g in groups for k in g}


def _group_scales(cal, cal_pred, ev, ev_pred, integ, cal_group, ev_group, cache, tol_km):
    """Per-group certified scale for one integrator (Bonferroni over grid x groups). Returns (scale per cal row, scale per eval
    row, report)."""
    labels = sorted(set(cal_group[cal_pred[integ]["feasible"]]))
    G = max(len(labels), 1)
    sc_group, rep = {}, {}
    for g in labels:
        rows = np.where((cal_group == g) & cal_pred[integ]["feasible"])[0]
        r = pb.certify_scale_lazy(cal, cal_pred, integ, tol_km=tol_km, mu=MU, cache=cache, n_tests=len(pb.LTT_SCALE_GRID) * G,
                                  rows=rows)
        sc_group[g] = r["chosen_scale"]
        rep[g] = {"n": r["n"], "violations": r["violations"], "scale": r["chosen_scale"]}
    fallback = min(pb.LTT_SCALE_GRID)
    cs = np.array([sc_group.get(g) if sc_group.get(g) is not None else fallback for g in cal_group], dtype=float)
    es = np.array([sc_group.get(g) if sc_group.get(g) is not None else fallback for g in ev_group], dtype=float)
    return cs, es, rep


def _summarize_alt(name, ev, res, c_pred, comp_cols, tol_km, deferred=None, defer_costs=None, defer_viol=None):
    r = ev[["row_id", "regime", "r2_band", "cell", "sim_time"]].merge(res, on="row_id", how="left")
    ok = r["ml_ok"].fillna(False).astype(bool) & (r["ml_err_km"] <= tol_km)
    has = r["ml_integrator"].notna()
    viol = has & ~ok
    cost = r["ml_evals"] + 1.0 + c_pred
    if deferred is not None:
        cost = np.where(deferred, defer_costs, cost)
        viol = np.where(deferred, defer_viol, viol)
        has = has | deferred
    viol = pd.Series(np.asarray(viol, dtype=bool), index=r.index)
    n = int(len(r))
    k = int(viol.sum())
    from orbit_ml_cr3bp.audit_3d import clopper_pearson
    out = {"alternative": name, "n": n, "violations": k, "violation_rate": k / n, "violation_ci95": clopper_pearson(k, n),
           "deferred_share": float(np.mean(deferred)) if deferred is not None else 0.0,
           "mean_cost": float(np.nanmean(np.where(viol, np.nan, cost))), "median_cost": float(np.nanmedian(np.where(viol, np.nan, cost))),
           "geomean_cost": float(np.exp(np.nanmean(np.log(np.where(viol, np.nan, cost)))))}
    for cc in comp_cols:
        out[f"median_cost_over_{cc}"] = float(np.nanmedian((np.where(viol, np.nan, cost) / ev[f"{cc}_nfev"].to_numpy())))
        out[f"share_cheaper_than_{cc}"] = float(np.nanmean(np.where(viol, np.nan, (cost < ev[f"{cc}_nfev"].to_numpy()).astype(float))))
    for by in ("regime", "r2_band"):
        out[f"violation_rate_by_{by}"] = {str(g): float(viol[r[by] == g].mean()) for g in sorted(r[by].dropna().unique())}
    orb = pd.DataFrame({"row_id": r["row_id"].to_numpy(), "viol": viol.to_numpy(), "cost": np.asarray(cost, dtype=float)})
    return out, orb


def cmd_e5(a):
    import joblib
    from orbit_ml_cr3bp.modeling_3d import predict_hurdle
    from orbit_ml_cr3bp.audit_3d import ReferenceCache
    out = make_run_dir(a.out_dir, "e5", a)
    U, train, cal, led, rep = load_policy_dir(a)
    pol = joblib.load(os.path.join(a.policy_dir, "policy.joblib"))
    models, scales = pol["models"], pol["scales"]
    c_pred = rep["prediction_cost"]["evals_equiv"] if a.c_pred is None else a.c_pred
    cal = fs.add_derived(cal) if "regime" not in cal.columns else cal
    cal_pred, ev_pred = predict_hurdle(models, cal), predict_hurdle(models, led)
    cache = ReferenceCache(MU)
    t0 = time.time()
    for i in range(len(cal)):   # prefetch the 2000 references once (the audits below reuse them)
        cache.get(i, cal[pb.STATE_COLS].to_numpy(dtype=float)[i], float(cal["sim_time"].iloc[i]))
    print(f"references {time.time() - t0:.0f}s", flush=True)
    cands = pb.CANDIDATES
    # q = 0.5 models for the dispersion of the predicted step
    disp_cal, disp_ev = {}, {}
    for k in cands:
        m50 = xb.fit_log_quantile(train, k, 0.5)
        for nm, df, pr, store in (("cal", cal, cal_pred, disp_cal), ("ev", led, ev_pred, disp_ev)):
            d50 = m50.predict(df[xb.FEATS_STATE].to_numpy(dtype=float))
            store[k] = np.where(pr[k]["feasible"], d50 - np.log10(np.where(pr[k]["dt"] > 0, pr[k]["dt"], np.nan)), np.inf)
    alts, rows_alt = [], {}
    comp_cols = ["RK45_spec", "RK45_cal", "RK45_eqr"]
    # --- 1. global scale (the policy as deployed)
    res = pb.evaluate_orbits(led, ev_pred, scales, rtols={}, do_spec=False, tol_km=a.tol_km, n_jobs=a.n_jobs, progress=False)
    o, orb = _summarize_alt("1_global", led, res, c_pred, comp_cols, a.tol_km)
    o["scales"] = scales
    alts.append(o)
    orb.to_csv(os.path.join(out, f"e5_orbits_{o['alternative']}.csv"), index=False)
    # --- 2. Mondrian by Hill regime (merged to >= n_min calibration rows per group and integrator)
    order = ["I", "II", "III", "IV", "V"]
    cs, es, grp_rep = {}, {}, {}
    for k in cands:
        mp = _merge_groups(cal["regime"].to_numpy()[cal_pred[k]["feasible"]], order, a.n_min)
        cg = np.array([mp.get(g, "none") for g in cal["regime"]])
        eg = np.array([mp.get(g, "none") for g in led["regime"]])
        cs[k], es[k], grp_rep[k] = _group_scales(cal, cal_pred, led, ev_pred, k, cg, eg, cache, a.tol_km)
    res = pb.evaluate_orbits(led, ev_pred, es, rtols={}, do_spec=False, tol_km=a.tol_km, n_jobs=a.n_jobs, progress=False)
    o, orb = _summarize_alt("2_mondrian_regime", led, res, c_pred, comp_cols, a.tol_km)
    o["groups"] = grp_rep
    alts.append(o)
    orb.to_csv(os.path.join(out, f"e5_orbits_{o['alternative']}.csv"), index=False)
    # --- 3. scale as a function of the model's own predicted dispersion (4 equal-count bins of the calibration dispersion)
    cs, es, grp_rep = {}, {}, {}
    for k in cands:
        dc = disp_cal[k]
        finite = np.isfinite(dc)
        edges = np.quantile(dc[finite], [0.25, 0.5, 0.75])
        cg = np.array([f"b{int(np.searchsorted(edges, x))}" for x in dc])
        eg = np.array([f"b{int(np.searchsorted(edges, x))}" for x in disp_ev[k]])
        cs[k], es[k], grp_rep[k] = _group_scales(cal, cal_pred, led, ev_pred, k, cg, eg, cache, a.tol_km)
    res = pb.evaluate_orbits(led, ev_pred, es, rtols={}, do_spec=False, tol_km=a.tol_km, n_jobs=a.n_jobs, progress=False)
    o, orb = _summarize_alt("3_dispersion_bins", led, res, c_pred, comp_cols, a.tol_km)
    o["groups"] = grp_rep
    alts.append(o)
    orb.to_csv(os.path.join(out, f"e5_orbits_{o['alternative']}.csv"), index=False)
    # --- 3b. split-conformal on the label (no re-integration): scale = 1 / q_(1-alpha) of d_hat / d*  on calibration rows
    sc_conf = {}
    for k in cands:
        m = cal_pred[k]["feasible"] & (cal[f"{k}_success"].to_numpy() == 1)
        ratio = cal_pred[k]["dt"][m] / cal[f"dt_{k}"].to_numpy()[m]
        n = int(m.sum())
        level = min(1.0, np.ceil((n + 1) * (1 - 0.05)) / n)
        sc_conf[k] = float(min(1.0, 1.0 / np.quantile(ratio, level)))
    res = pb.evaluate_orbits(led, ev_pred, sc_conf, rtols={}, do_spec=False, tol_km=a.tol_km, n_jobs=a.n_jobs, progress=False)
    o, orb = _summarize_alt("3b_label_conformal", led, res, c_pred, comp_cols, a.tol_km)
    o["scales"] = sc_conf
    alts.append(o)
    orb.to_csv(os.path.join(out, f"e5_orbits_{o['alternative']}.csv"), index=False)
    # --- 4. abstention: defer the most dispersed 20 % (by the least dispersed feasible candidate) to adaptive RK45 at the cal rtol
    dmin_cal = np.min(np.vstack([disp_cal[k] for k in cands]), axis=0)
    dmin_ev = np.min(np.vstack([disp_ev[k] for k in cands]), axis=0)
    thr = float(np.quantile(dmin_cal[np.isfinite(dmin_cal)], 1.0 - a.abstain_frac))
    keep_cal = np.where(dmin_cal <= thr)[0]
    sc_abs = {}
    for k in cands:
        r = pb.certify_scale_lazy(cal, cal_pred, k, tol_km=a.tol_km, mu=MU, cache=cache, rows=keep_cal)
        sc_abs[k] = r["chosen_scale"]
    defer = dmin_ev > thr
    res = pb.evaluate_orbits(led, ev_pred, sc_abs, rtols={}, do_spec=False, tol_km=a.tol_km, n_jobs=a.n_jobs, progress=False)
    ad_cost = 1.0 + c_pred + led["RK45_cal_nfev"].to_numpy()
    ad_viol = ~led["RK45_cal_ok"].fillna(False).astype(bool).to_numpy()
    o, orb = _summarize_alt("4_abstain_to_adaptive", led, res, c_pred, comp_cols, a.tol_km, deferred=defer,
                          defer_costs=ad_cost, defer_viol=ad_viol)
    o["scales"] = sc_abs
    o["dispersion_threshold_dex"] = thr
    alts.append(o)
    orb.to_csv(os.path.join(out, f"e5_orbits_{o['alternative']}.csv"), index=False)
    tab = pd.DataFrame([{k: v for k, v in o.items() if not isinstance(v, (dict, list))} for o in alts])
    tab.to_csv(os.path.join(out, "e5_alternatives.csv"), index=False)
    save_json({"c_pred": c_pred, "alternatives": alts}, os.path.join(out, "e5_report.json"))
    show = [c for c in tab.columns if c in ("alternative", "violation_rate", "deferred_share", "mean_cost", "median_cost",
                                            "median_cost_over_RK45_cal", "share_cheaper_than_RK45_cal", "share_cheaper_than_RK45_spec")]
    print(tab[show].round(4).to_string())
    print("wrote", out)


# ---------------------------------------------------------------------------------------------------------------------
# E9: step-over and a runtime guard
# ---------------------------------------------------------------------------------------------------------------------

def cmd_e9(a):
    import joblib
    from orbit_ml_cr3bp.modeling_3d import predict_hurdle
    out = make_run_dir(a.out_dir, "e9", a)
    U, train, cal, led, rep = load_policy_dir(a)
    pol = joblib.load(os.path.join(a.policy_dir, "policy.joblib"))
    models, scales = pol["models"], pol["scales"]
    ev_pred = predict_hurdle(models, led)
    frames = []
    for mult in a.multipliers:
        sc = {k: (scales[k] if mult == "cert" else float(mult)) for k in pb.CANDIDATES}
        res = pb.evaluate_orbits(led, ev_pred, sc, rtols={}, do_spec=False, tol_km=a.tol_km, n_jobs=a.n_jobs, progress=False)
        f = led[["row_id", "regime", "r2_band", "cell", "tau_min", "min_r2", "sim_time", "delta_phys"]].merge(res, on="row_id")
        f["multiplier"] = str(mult)
        frames.append(f)
    d = pd.concat(frames, ignore_index=True)
    d = d[d["ml_integrator"].notna()].copy()
    d["err_over_tol"] = d["ml_err_km"] / a.tol_km
    d["violation"] = ~(d["ml_ok"].fillna(False).astype(bool) & (d["err_over_tol"] <= 1.0))
    d["step_over"] = d["ml_dt"] / d["tau_min"]
    d["log_step_over"] = np.log10(d["step_over"])
    d.to_csv(os.path.join(out, "e9_runs.csv"), index=False)
    v = d[d["violation"] & d["ml_ok"].fillna(False).astype(bool)]
    cat = v["err_over_tol"] > fs.CATASTROPHIC
    from scipy.stats import spearmanr
    rep9 = {"n_runs": int(len(d)), "n_violations": int(d["violation"].sum()), "n_violations_finite_error": int(len(v)),
            "n_catastrophic": int(cat.sum()),
            "spearman_logstepover_logerr_among_violations": float(spearmanr(v["log_step_over"], np.log10(v["err_over_tol"])).statistic),
            "auc_catastrophic_vs_mild_logstepover": xb.auc_score(v.loc[cat, "log_step_over"], v.loc[~cat, "log_step_over"]),
            "auc_violation_vs_ok_logstepover": xb.auc_score(d.loc[d["violation"], "log_step_over"], d.loc[~d["violation"], "log_step_over"]),
            "median_step_over": {"ok": float(d.loc[~d["violation"], "step_over"].median()), "mild_violation": float(v.loc[~cat, "step_over"].median()),
                                 "catastrophic": float(v.loc[cat, "step_over"].median()) if cat.any() else None}}
    g = xb.guard_table(d["step_over"], d["violation"], a.thresholds)
    g.to_csv(os.path.join(out, "e9_guard.csv"), index=False)
    rep9["guard"] = g.to_dict(orient="records")
    # catastrophic share by step-over bin
    bins = pd.cut(d["step_over"], [0, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, np.inf])
    tab = d.groupby(bins, observed=True).agg(n=("violation", "size"), violation_rate=("violation", "mean"),
                                             median_err_over_tol=("err_over_tol", "median")).reset_index()
    tab["step_over"] = tab["step_over"].astype(str)
    tab.to_csv(os.path.join(out, "e9_by_stepover.csv"), index=False)
    rep9["by_stepover_bin"] = tab.to_dict(orient="records")
    save_json(rep9, os.path.join(out, "e9_report.json"))
    print(json.dumps(proto.json_safe({k: v for k, v in rep9.items() if k not in ("guard", "by_stepover_bin")}), indent=1))
    print(g.round(3).to_string())
    print(tab.round(3).to_string())
    print("wrote", out)


# ---------------------------------------------------------------------------------------------------------------------
# E10: physics-rule baseline and break-even inference cost
# ---------------------------------------------------------------------------------------------------------------------

def cmd_e10(a):
    import joblib
    from orbit_ml_cr3bp.modeling_3d import predict_hurdle
    from orbit_ml_cr3bp.audit_3d import ReferenceCache
    from orbit_ml_cr3bp.fair_baseline_3d import measure_prediction_cost
    out = make_run_dir(a.out_dir, "e10", a)
    U, train, cal, led, rep = load_policy_dir(a)
    pol = joblib.load(os.path.join(a.policy_dir, "policy.joblib"))
    models = pol["models"]
    cal = fs.add_derived(cal) if "regime" not in cal.columns else cal
    rules = {k: xb.fit_physics_rule(train, k) for k in pb.CANDIDATES}
    cal_pred, ev_pred = xb.physics_pred_dict(cal, rules), xb.physics_pred_dict(led, rules)
    cache = ReferenceCache(MU)
    scales = {}
    for k in pb.CANDIDATES:
        r = pb.certify_scale_lazy(cal, cal_pred, k, tol_km=a.tol_km, mu=MU, cache=cache)
        scales[k] = r["chosen_scale"]
        print(f"physics rule {k}: {rules[k]}  certified scale {r['chosen_scale']} (n={r['n']}, viol={r['violations']})", flush=True)
    res = pb.evaluate_orbits(led, ev_pred, scales, rtols={}, do_spec=False, tol_km=a.tol_km, n_jobs=a.n_jobs, progress=False)
    m = led[["row_id", "regime", "r2_band", "C_fix_star", "ml_evals", "meets_tol", "RK45_spec_nfev", "RK45_cal_nfev", "DOP853_cal_nfev",
             "RK45_eqr_nfev", "ml_integrator"]].rename(columns={"ml_evals": "ml_evals_ML", "meets_tol": "meets_ML",
                                                              "ml_integrator": "integ_ML"}).merge(res, on="row_id")
    ok = m["ml_ok"].fillna(False).astype(bool) & (m["ml_err_km"] <= a.tol_km)
    phys_cost = m["ml_evals"] + 1.0     # one acceleration-sized computation for tau(IC); the rule itself is closed form
    rep10 = {"rules": rules, "scales": scales, "n": int(len(m)),
             "physics_violation_rate": float((~ok).mean()), "ml_violation_rate": float((~m["meets_ML"]).mean()),
             "median_int_cost_physics": float(m.loc[ok, "ml_evals"].median()),
             "median_int_cost_ml": float(m.loc[m["meets_ML"], "ml_evals_ML"].median()),
             "median_total_cost_physics": float(phys_cost[ok].median()),
             "median_total_cost_ml": float(led.loc[led["meets_tol"], "C_ml_tot"].median())}
    both = ok & m["meets_ML"]
    rep10["median_ratio_physics_int_over_ml_int_both_ok"] = float((m.loc[both, "ml_evals"] / m.loc[both, "ml_evals_ML"]).median())
    for c in ("RK45_spec", "RK45_cal"):
        rep10[f"physics_total_cheaper_than_{c}"] = float((phys_cost[ok] < m.loc[ok, f"{c}_nfev"]).mean())
        rep10[f"physics_int_cheaper_than_{c}"] = float((m.loc[ok, "ml_evals"] < m.loc[ok, f"{c}_nfev"]).mean())
        rep10[f"ml_int_cheaper_than_{c}"] = float((m.loc[m["meets_ML"], "ml_evals_ML"] < m.loc[m["meets_ML"], f"{c}_nfev"]).mean())
    # --- break-even inference cost: headroom = C_ad - C_ml_int (what the features + model may cost for ML to still win)
    ok_ml = led[led["meets_tol"]]
    be = {}
    for c in ("RK45_spec", "DOP853_spec", "RK45_cal", "DOP853_cal", "RK45_eqr", "DOP853_eqr"):
        head = ok_ml[f"{c}_nfev"] - ok_ml["ml_evals"]
        pos = head[head > 0]
        be[c] = {"share_with_positive_headroom": float((head > 0).mean()), "median_headroom_when_positive": float(pos.median()) if len(pos) else None,
                 "p90_headroom": float(head.quantile(0.9)),
                 "share_ml_total_cheaper_at_zero_overhead": float((head > 0).mean()),
                 "share_ml_total_cheaper_with_measured_overhead": float((head > rep["prediction_cost"]["evals_equiv"] + 1.0).mean())}
    rep10["break_even"] = be
    rows = []
    for by in ("regime", "r2_band"):
        for g, gg in ok_ml.groupby(by):
            h = gg["RK45_cal_nfev"] - gg["ml_evals"]
            rows.append({"by": by, "group": g, "n": len(gg), "share_headroom_positive_vs_RK45_cal": float((h > 0).mean()),
                         "median_headroom_rk45_cal": float(h.median())})
    pd.DataFrame(rows).to_csv(os.path.join(out, "e10_break_even_by_group.csv"), index=False)
    # --- inference cost in isolation, repeated (the figure inside the ledger comes from one noisy wall-time measurement)
    meas = [measure_prediction_cost(models, led, pb.CANDIDATES, MU, n_rows=1000, repeats=5)["evals_equiv"] for _ in range(5)]
    rep10["prediction_cost_evals_equiv_repeats"] = meas
    rep10["prediction_cost_in_ledger"] = rep["prediction_cost"]["evals_equiv"]
    save_json(rep10, os.path.join(out, "e10_report.json"))
    print(json.dumps(proto.json_safe(rep10), indent=1))
    print("wrote", out)


# ---------------------------------------------------------------------------------------------------------------------
# E11: what the model is good for -- a deployable gate, and ML as a cost predictor for batching adaptive solves
# ---------------------------------------------------------------------------------------------------------------------

def _gate_cells(led, min_n=50):
    """Deployable cells = Hill regime x horizon band (both known from the initial state; min r2 is NOT known before integrating).
    Merged by the same hierarchical rule as the study cells: base cell if n >= min_n, else 'regime|h*', else 'pooled-small'."""
    base = led["regime"].astype(str) + "|" + led["h_band"].astype(str)
    n = base.map(base.value_counts())
    lab = pd.Series(np.where(n >= min_n, base, None), index=led.index, dtype=object)
    rest = lab.isna()
    g2 = led["regime"].astype(str) + "|h*"
    n2 = g2[rest].map(g2[rest].value_counts())
    lab[rest & (n2.reindex(led.index) >= min_n)] = g2[rest & (n2.reindex(led.index) >= min_n)]
    lab[lab.isna()] = "pooled-small"
    return lab


def cmd_e11(a):
    out = make_run_dir(a.out_dir, "e11", a)
    gate = pd.read_csv(a.gate_ledger, low_memory=False)
    lock = pd.read_csv(a.lock_ledger, low_memory=False)
    ad = a.adaptive
    gate["gcell"] = _gate_cells(gate)
    # a lock orbit takes its cell label from the gate ledger's cell set (same regime/horizon -> same label); unseen -> pooled-small
    lab_map = {(r, h): c for r, h, c in zip(gate["regime"], gate["h_band"], gate["gcell"])}
    lock["gcell"] = [lab_map.get((r, h), lab_map.get((r, "h*"), "pooled-small")) for r, h in zip(lock["regime"], lock["h_band"])]
    rows, detail = [], {}
    for om in a.overheads:
        oh = om + 1.0      # + one acceleration-sized computation for the cheap state features
        g = gate[gate["meets_tol"]]
        win = (g["ml_evals"] + oh < g[f"{ad}_cal_nfev"])
        viol = gate.groupby("gcell")["violation"].mean()
        share = win.groupby(g["gcell"]).mean().reindex(viol.index).fillna(0.0)
        n_c = gate.groupby("gcell").size()
        gated = [c for c in viol.index if share[c] >= a.min_win_share and viol[c] <= a.alpha and n_c[c] >= 50 and c != "pooled-small"]
        use_ml = lock["gcell"].isin(gated)
        ml_cost = lock["ml_evals"] + oh
        ml_viol = lock["violation"] | ~lock["has_choice"]
        ad_cost = lock[f"{ad}_eqr_nfev"]
        ad_viol = ~lock[f"{ad}_eqr_ok"].fillna(False).astype(bool)
        cost = np.where(use_ml, ml_cost, ad_cost)
        v = np.where(use_ml, ml_viol, ad_viol)
        # compare with pure adaptive at the equal-risk rtol, and with its per-orbit oracle (unattainable)
        base_cost, base_v = ad_cost, ad_viol
        ok = ~np.isnan(cost) & ~np.isnan(base_cost.to_numpy())
        rows.append({"overhead_rhs_equiv": om, "gated_cells": len(gated), "share_routed_to_ml": float(use_ml.mean()),
                     "gated_violation_rate": float(np.mean(v)), "adaptive_eqr_violation_rate": float(base_v.mean()),
                     "total_cost_gated_over_adaptive_eqr": float(np.nansum(cost[ok]) / np.nansum(base_cost.to_numpy()[ok])),
                     "median_cost_gated_over_adaptive_eqr": float(np.nanmedian(cost[ok] / base_cost.to_numpy()[ok])),
                     "total_cost_gated_over_adaptive_spec": float(np.nansum(cost[ok]) / np.nansum(lock[f"{ad}_spec_nfev"].to_numpy()[ok])),
                     "ml_win_share_on_routed": float((ml_cost[use_ml] < ad_cost[use_ml]).mean()) if use_ml.any() else float("nan")})
        detail[str(om)] = {"gated": gated}
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(out, "e11_gate.csv"), index=False)
    save_json({"adaptive": ad, "table": rows, "gated_cells": detail, "min_win_share": a.min_win_share, "alpha": a.alpha},
              os.path.join(out, "e11_report.json"))
    print(tab.round(4).to_string())
    print("wrote", out)


def cmd_e11b(a):
    from scipy.stats import spearmanr
    from sklearn.ensemble import HistGradientBoostingRegressor
    from orbit_ml_cr3bp.ensemble_regime_3d import _batch_cost
    out = make_run_dir(a.out_dir, "e11b", a)
    tr = pd.read_csv(a.train_ledger, low_memory=False)
    te = pd.read_csv(a.test_ledger, low_memory=False)
    col = f"{a.adaptive}_cal_nfev"
    tr, te = tr[tr[col].notna()], te[te[col].notna()]
    reg = HistGradientBoostingRegressor(random_state=0).fit(tr[xb.FEATS_STATE].to_numpy(dtype=float), np.log10(tr[col].to_numpy(dtype=float)))
    pred = reg.predict(te[xb.FEATS_STATE].to_numpy(dtype=float))
    true = te[col].to_numpy(dtype=float)
    rho = float(spearmanr(pred, true).statistic)
    rng = np.random.RandomState(0)
    rows = []
    for B in a.batch_sizes:
        ideal = true.sum()
        uns = np.mean([_batch_cost(true[rng.permutation(len(true))], B) for _ in range(200)])
        srt_pred = _batch_cost(true[np.argsort(pred)], B)
        srt_orac = _batch_cost(np.sort(true), B)
        rows.append({"batch_size": B, "n": int(len(true)), "lane_eff_unsorted": float(ideal / uns), "lane_eff_sorted_by_prediction": float(ideal / srt_pred),
                     "lane_eff_sorted_oracle": float(ideal / srt_orac), "cost_ratio_pred_sorted_over_unsorted": float(srt_pred / uns)})
    # the same exercise for the fixed-step integration count of the ML policy, which is known BEFORE launch (the sort key is exact)
    ml = te["ml_evals"].to_numpy(dtype=float)
    for B in a.batch_sizes:
        ml_ok = ~np.isnan(ml)
        uns = np.mean([_batch_cost(ml[ml_ok][rng.permutation(int(ml_ok.sum()))], B) for _ in range(200)])
        rows.append({"batch_size": B, "n": int(ml_ok.sum()), "what": "ML fixed-step counts (exactly known before launch)",
                     "lane_eff_unsorted": float(ml[ml_ok].sum() / uns), "lane_eff_sorted_exact": float(ml[ml_ok].sum() / _batch_cost(np.sort(ml[ml_ok]), B))})
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(out, "e11b_lane_efficiency.csv"), index=False)
    save_json({"spearman_pred_true_log_nfev": rho, "n_train": int(len(tr)), "n_test": int(len(te)), "rows": rows}, os.path.join(out, "e11b_report.json"))
    print("Spearman(predicted, true adaptive nfev) =", round(rho, 3))
    print(tab.round(3).to_string())
    print("wrote", out)


# ---------------------------------------------------------------------------------------------------------------------
# E3c (exploratory, pre-registered 2026-10-26): how much of the trajectory must be seen, and what seeing it costs
# ---------------------------------------------------------------------------------------------------------------------

def cmd_e3c_features(a):
    from orbit_ml_cr3bp import lookahead_3d as la
    out = make_run_dir(a.out_dir, "e3c_features", a)
    d, _ = load_partb(a.data)
    U = pb.usable_rows(d).reset_index(drop=True)
    print(f"lookahead features for {len(U)} usable rows, fractions {la.FRACTIONS}", flush=True)
    t0 = time.time()
    df = la.compute_lookahead(U, mu=a.mu, n_jobs=a.n_jobs)
    df.to_csv(os.path.join(out, "lookahead.csv"), index=False)
    cap = {la.tag(f): float(df[f"la{la.tag(f)}_capped"].mean()) for f in la.FRACTIONS}
    med = {la.tag(f): float(df[f"la{la.tag(f)}_nfev"].median()) for f in la.FRACTIONS}
    save_json({"n": int(len(df)), "capped_share": cap, "median_nfev": med, "wall_s": time.time() - t0}, os.path.join(out, "e3c_features_report.json"))
    print("capped share", cap, "median nfev", med, f"({time.time() - t0:.0f}s)\nwrote", out)


def cmd_e3c(a):
    from orbit_ml_cr3bp import lookahead_3d as la
    from orbit_ml_cr3bp.audit_3d import ReferenceCache, clopper_pearson
    out = make_run_dir(a.out_dir, "e3c", a)
    U, train, cal, led, rep = load_policy_dir(a)
    lk = pd.read_csv(a.lookahead, low_memory=False)
    train, cal, ev = (x.merge(lk, on="row_id", how="left") for x in (train, cal, led))
    c_pred = a.c_pred
    cache = ReferenceCache(MU)
    for i in range(len(cal)):
        cache.get(i, cal[pb.STATE_COLS].to_numpy(dtype=float)[i], float(cal["sim_time"].iloc[i]))
    comps = ["RK45_spec", "RK45_cal", "DOP853_cal"]
    rows, per = [], {}
    levels = [0.0] + list(la.FRACTIONS)
    for f in levels:
        feats = list(xb.FEATS_STATE) + ([] if f == 0 else la.feature_names(f))
        reg = xb.fit_log_quantile(train, "rk4", a.quantile, features=feats, seed=42)

        def predict(df):
            dt = 10.0 ** reg.predict(df[feats].to_numpy(dtype=float))
            n = len(df)
            return {"rk4": {"feasible": np.ones(n, dtype=bool), "dt": dt},
                    "leapfrog": {"feasible": np.zeros(n, dtype=bool), "dt": np.full(n, np.nan)}}
        cal_pred, ev_pred = predict(cal), predict(ev)
        r = pb.certify_scale_lazy(cal, cal_pred, "rk4", tol_km=a.tol_km, mu=MU, cache=cache)
        sc = r["chosen_scale"]
        y = np.log10(ev["dt_rk4"].to_numpy(dtype=float))
        yh = np.log10(ev_pred["rk4"]["dt"])
        res = pb.evaluate_orbits(ev, ev_pred, {"rk4": sc, "leapfrog": None}, rtols={}, do_spec=False, tol_km=a.tol_km, n_jobs=a.n_jobs,
                                 candidates=("rk4",), progress=False)
        m = ev[["row_id", "regime", "r2_band", "sim_time"] + [f"{c}_nfev" for c in comps]].rename(
            columns={f"{c}_nfev": c for c in comps}).merge(res, on="row_id", how="left")
        m["abs_resid"] = np.abs(yh - y)
        m["over"] = yh > y
        ok = m["ml_ok"].fillna(False).astype(bool) & (m["ml_err_km"] <= a.tol_km)
        m["meets"] = ok
        m["c_la"] = 0.0 if f == 0 else ev[f"la{la.tag(f)}_nfev"].to_numpy(dtype=float)
        m["c_int"] = m["ml_evals"]
        m["c_tot_nomodel"] = m["c_la"] + m["c_int"] + 1.0
        m["c_tot_model"] = m["c_tot_nomodel"] + c_pred
        per[f] = m
        n = len(m)
        k = int((~ok).sum())
        row = {"f": f, "scale": sc, "n": n, "violation_rate": k / n, "violation_ci95": clopper_pearson(k, n),
               "abs_resid_median": fs.bootstrap_ci(m["abs_resid"]), "abs_resid_p90": float(m["abs_resid"].quantile(0.9)),
               "pinball": float(xb.pinball(y, yh, a.quantile).mean()), "over_rate": float(m["over"].mean()),
               "median_c_int": float(m.loc[ok, "c_int"].median()), "median_c_la": float(m["c_la"].median()),
               "median_c_tot_nomodel": float(m.loc[ok, "c_tot_nomodel"].median())}
        for c in comps:
            for nm, col in (("nomodel", "c_tot_nomodel"), ("model1128", "c_tot_model")):
                w = ok & (m[col] < m[c])
                row[f"win_{c}_{nm}"] = float(w.mean())
        rows.append(row)
        print(f"f={f:<5} scale {sc}  viol {k / n:.3f}  |resid| med {row['abs_resid_median'][0]:.3f} p90 {row['abs_resid_p90']:.3f}  "
              f"C_int {row['median_c_int']:.0f} C_la {row['median_c_la']:.0f}  win vs RK45-cal (no model cost) {row['win_RK45_cal_nomodel']:.3f}", flush=True)
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(out, "e3c_by_fraction.csv"), index=False)
    # any lookahead fraction that makes the orbit cheaper than the comparator at its own certified scale
    any_win = {}
    for c in comps:
        for nm, col in (("nomodel", "c_tot_nomodel"), ("model1128", "c_tot_model")):
            best = np.min(np.vstack([np.where(per[f]["meets"], per[f][col], np.inf) for f in levels]), axis=0)
            any_win[f"{c}_{nm}"] = float((best < per[levels[0]][c].to_numpy()).mean())
    # by group: error curve, f_match (smallest f within 25 % of the f = 1 median error) and the win share there
    groups = []
    for by in ("regime", "r2_band"):
        for g in sorted(per[0.0][by].dropna().unique()):
            curve = {f: float(per[f].loc[per[f][by] == g, "abs_resid"].median()) for f in levels}
            target = 1.25 * curve[1.0]
            fm = next((f for f in levels if curve[f] <= target), None)
            wins = {f: float((per[f][by] == g)[per[f]["meets"] & (per[f]["c_tot_nomodel"] < per[f]["RK45_cal"])].sum() / max((per[f][by] == g).sum(), 1)) for f in levels}
            groups.append({"by": by, "group": g, "n": int((per[0.0][by] == g).sum()), "f_match": fm,
                           "win_RK45_cal_at_f_match_nomodel": wins[fm] if fm is not None else None,
                           **{f"resid_f{la.tag(f)}": curve[f] for f in levels}})
    gt = pd.DataFrame(groups)
    gt.to_csv(os.path.join(out, "e3c_by_group.csv"), index=False)
    save_json({"table": rows, "any_fraction_wins": any_win, "groups": groups, "levels": levels, "c_pred_pinned": c_pred,
               "note": "RK4 only; model cost excluded (nomodel) or the pinned 1,128 for the two-integrator hurdle model added (model1128)"},
              os.path.join(out, "e3c_report.json"))
    print(gt.round(3).to_string())
    print("any lookahead fraction wins:", {k: round(v, 4) for k, v in any_win.items()})
    print("wrote", out)


# ---------------------------------------------------------------------------------------------------------------------
# E7: the mu dial (exploratory, pre-registered 2026-10-26)
# ---------------------------------------------------------------------------------------------------------------------

def cmd_mu_dial(a):
    from orbit_ml_cr3bp.audit_3d import clopper_pearson
    out = make_run_dir(a.out_dir, "mu_dial", a)
    rows, detail = [], {}
    for spec in a.runs:
        mu_s, path = spec.split("=", 1)
        mu = float(mu_s)
        led = pd.read_csv(os.path.join(path, "ledger.csv"), low_memory=False)
        rep = json.load(open(os.path.join(path, "policy_report.json")))
        d = led[led["RK45_spec_nfev"].notna() & led["C_fix_star"].notna()].copy()
        gap = np.log10(d["C_fix_star"] / d["RK45_spec_nfev"])
        k = int((d["C_fix_star"] < d["RK45_spec_nfev"]).sum())
        dec = fs.decompose(led, "RK45_spec")
        okd = dec["total"].notna()
        ml = led[led["ml_ok"].fillna(False).astype(bool) & led["ml_evals"].notna() & led["RK45_spec_nfev"].notna()]
        ml_gap = np.log10(ml["ml_evals"] / ml["RK45_spec_nfev"])
        nv, n = int(led["violation"].sum()), int(led["has_choice"].sum())
        row = {"mu": mu, "n_usable": rep["n_usable"], "n_train": rep["n_train"], "n_eval": int(len(led)),
               "fixed_cheaper_share": k / len(d), "fixed_cheaper_ci95": clopper_pearson(k, len(d)),
               "paradigm_gap_median": fs.bootstrap_ci(gap, n_boot=1000), "paradigm_gap_mean": fs.bootstrap_ci(gap, stat=np.mean, n_boot=1000),
               "slope_on_log_delta_phys": _ols_slope(d["log_delta_phys"], gap, n_boot=1000),
               "spearman_delta_gap": fs.spearman_bootstrap(d["log_delta_phys"], gap, n_boot=500)[0],
               "median_delta_phys": float(d["delta_phys"].median()), "median_min_r2_over_rhill": float((d["min_r2"] / (mu / 3) ** (1 / 3)).median()),
               "ml_integration_gap_median": fs.bootstrap_ci(ml_gap, n_boot=1000),
               "violation_rate": nv / n, "violation_ci95": clopper_pearson(nv, n),
               "scale_rk4": rep["scales"]["rk4"], "scale_leapfrog": rep["scales"]["leapfrog"],
               "adaptive_spec_nfev_median": float(d["RK45_spec_nfev"].median()), "oracle_fixed_cost_median": float(d["C_fix_star"].median()),
               "terms_mean_dex": {t: fs.bootstrap_ci(dec.loc[okd, t], stat=np.mean, n_boot=1000) for t in ("P", "S", "Q", "K", "O", "total")}}
        rows.append(row)
        print(f"mu={mu:<10g} n_eval {len(led):5d}  fixed cheaper {row['fixed_cheaper_share']:.3f}  P med {row['paradigm_gap_median'][0]:+.3f}  "
              f"ML-int gap {row['ml_integration_gap_median'][0]:+.3f}  viol {row['violation_rate']:.3f}  scale {row['scale_rk4']}/{row['scale_leapfrog']}  "
              f"slope {row['slope_on_log_delta_phys'].get('slope', float('nan')):.2f}  median delta {row['median_delta_phys']:.2f}", flush=True)
    rows.sort(key=lambda r: r["mu"])
    order = lambda key: [r["mu"] for r in sorted(rows, key=lambda r: (r[key][0] if isinstance(r[key], (list, tuple)) else r[key]))]
    summary = {"rows": rows, "order_by_paradigm_gap_median_low_to_high": order("paradigm_gap_median"),
               "order_by_ml_integration_gap_low_to_high": order("ml_integration_gap_median"),
               "order_by_violation_rate_low_to_high": order("violation_rate"),
               "note": "descriptive only; the pre-registered expectation is that both gaps fall as mu falls"}
    save_json(summary, os.path.join(out, "mu_dial_report.json"))
    flat = pd.DataFrame([{"mu": r["mu"], "n_eval": r["n_eval"], "fixed_cheaper_share": r["fixed_cheaper_share"], "paradigm_gap_median": r["paradigm_gap_median"][0],
                          "paradigm_gap_lo": r["paradigm_gap_median"][1], "paradigm_gap_hi": r["paradigm_gap_median"][2],
                          "ml_int_gap_median": r["ml_integration_gap_median"][0], "ml_int_gap_lo": r["ml_integration_gap_median"][1],
                          "ml_int_gap_hi": r["ml_integration_gap_median"][2], "violation_rate": r["violation_rate"],
                          "violation_lo": r["violation_ci95"][0], "violation_hi": r["violation_ci95"][1], "slope": r["slope_on_log_delta_phys"].get("slope"),
                          "median_delta_phys": r["median_delta_phys"], "scale_rk4": r["scale_rk4"], "scale_leapfrog": r["scale_leapfrog"],
                          **{f"mean_{t}": r["terms_mean_dex"][t][0] for t in ("P", "S", "Q", "K", "O", "total")}} for r in rows])
    flat.to_csv(os.path.join(out, "mu_dial_table.csv"), index=False)
    print(flat.round(3).to_string())
    print("order by paradigm gap:", summary["order_by_paradigm_gap_median_low_to_high"])
    print("order by ML integration gap:", summary["order_by_ml_integration_gap_low_to_high"])
    print("wrote", out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--data", required=True)
        sp.add_argument("--policy_dir", required=True)
        sp.add_argument("--out_dir", default="out_b")
        sp.add_argument("--seed", type=int, default=42)
        sp.add_argument("--tol_km", type=float, default=POSITION_TOL_KM)
        sp.add_argument("--n_jobs", type=int, default=8)
        sp.add_argument("--train_data", default=None, help="dataset holding the (frozen) model's training rows")
        sp.add_argument("--train_policy_dir", default=None, help="policy folder of that model (train_row_ids.csv)")

    sp = sub.add_parser("e4")
    common(sp)
    sp.add_argument("--seeds", type=int, default=5)
    sp.add_argument("--quantile", type=float, default=0.1)
    sp.set_defaults(fn=cmd_e4)

    sp = sub.add_parser("e5")
    common(sp)
    sp.add_argument("--n_min", type=int, default=300)
    sp.add_argument("--abstain_frac", type=float, default=0.2)
    sp.add_argument("--c_pred", type=float, default=None)
    sp.set_defaults(fn=cmd_e5)

    sp = sub.add_parser("e9")
    common(sp)
    sp.add_argument("--multipliers", nargs="+", default=["cert", "1.0", "2.0"])
    sp.add_argument("--thresholds", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0])
    sp.set_defaults(fn=cmd_e9)

    sp = sub.add_parser("e10")
    common(sp)
    sp.set_defaults(fn=cmd_e10)

    sp = sub.add_parser("e11")
    sp.add_argument("--gate_ledger", required=True)
    sp.add_argument("--lock_ledger", required=True)
    sp.add_argument("--out_dir", default="out_b")
    sp.add_argument("--adaptive", default="DOP853")
    sp.add_argument("--overheads", type=float, nargs="+", default=[0.0, 10.0, 50.0, 200.0, 1000.0])
    sp.add_argument("--min_win_share", type=float, default=0.5)
    sp.add_argument("--alpha", type=float, default=0.05)
    sp.set_defaults(fn=cmd_e11)

    sp = sub.add_parser("e11b")
    sp.add_argument("--train_ledger", required=True)
    sp.add_argument("--test_ledger", required=True)
    sp.add_argument("--out_dir", default="out_b")
    sp.add_argument("--adaptive", default="DOP853")
    sp.add_argument("--batch_sizes", type=int, nargs="+", default=[32, 128])
    sp.set_defaults(fn=cmd_e11b)

    sp = sub.add_parser("e3c_features")
    common(sp)
    sp.add_argument("--mu", type=float, default=MU)
    sp.set_defaults(fn=cmd_e3c_features)

    sp = sub.add_parser("e3c")
    common(sp)
    sp.add_argument("--lookahead", required=True)
    sp.add_argument("--quantile", type=float, default=0.1)
    sp.add_argument("--c_pred", type=float, default=1128.0)
    sp.set_defaults(fn=cmd_e3c)

    sp = sub.add_parser("mu_dial")
    sp.add_argument("--runs", nargs="+", required=True, help="mu=policy_run_folder pairs")
    sp.add_argument("--out_dir", default="out_b")
    sp.set_defaults(fn=cmd_mu_dial)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
