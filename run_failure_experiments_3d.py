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
    train = U[U["row_id"].isin(train_ids)].reset_index(drop=True)
    cal = U[U["row_id"].isin(cal_ids)].reset_index(drop=True)
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
    return out, viol


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
    o, v = _summarize_alt("1_global", led, res, c_pred, comp_cols, a.tol_km)
    o["scales"] = scales
    alts.append(o)
    # --- 2. Mondrian by Hill regime (merged to >= n_min calibration rows per group and integrator)
    order = ["I", "II", "III", "IV", "V"]
    cs, es, grp_rep = {}, {}, {}
    for k in cands:
        mp = _merge_groups(cal["regime"].to_numpy()[cal_pred[k]["feasible"]], order, a.n_min)
        cg = np.array([mp.get(g, "none") for g in cal["regime"]])
        eg = np.array([mp.get(g, "none") for g in led["regime"]])
        cs[k], es[k], grp_rep[k] = _group_scales(cal, cal_pred, led, ev_pred, k, cg, eg, cache, a.tol_km)
    res = pb.evaluate_orbits(led, ev_pred, es, rtols={}, do_spec=False, tol_km=a.tol_km, n_jobs=a.n_jobs, progress=False)
    o, v = _summarize_alt("2_mondrian_regime", led, res, c_pred, comp_cols, a.tol_km)
    o["groups"] = grp_rep
    alts.append(o)
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
    o, v = _summarize_alt("3_dispersion_bins", led, res, c_pred, comp_cols, a.tol_km)
    o["groups"] = grp_rep
    alts.append(o)
    # --- 3b. split-conformal on the label (no re-integration): scale = 1 / q_(1-alpha) of d_hat / d*  on calibration rows
    sc_conf = {}
    for k in cands:
        m = cal_pred[k]["feasible"] & (cal[f"{k}_success"].to_numpy() == 1)
        ratio = cal_pred[k]["dt"][m] / cal[f"dt_{k}"].to_numpy()[m]
        n = int(m.sum())
        level = min(1.0, np.ceil((n + 1) * (1 - 0.05)) / n)
        sc_conf[k] = float(min(1.0, 1.0 / np.quantile(ratio, level)))
    res = pb.evaluate_orbits(led, ev_pred, sc_conf, rtols={}, do_spec=False, tol_km=a.tol_km, n_jobs=a.n_jobs, progress=False)
    o, v = _summarize_alt("3b_label_conformal", led, res, c_pred, comp_cols, a.tol_km)
    o["scales"] = sc_conf
    alts.append(o)
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
    o, v = _summarize_alt("4_abstain_to_adaptive", led, res, c_pred, comp_cols, a.tol_km, deferred=defer,
                          defer_costs=ad_cost, defer_viol=ad_viol)
    o["scales"] = sc_abs
    o["dispersion_threshold_dex"] = thr
    alts.append(o)
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

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
