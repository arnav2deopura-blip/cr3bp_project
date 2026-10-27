"""
Part C commands (see part_c/PART_C_LOG.md for the decisions behind each one). Every command writes into its own timestamped
folder under --out_dir with a manifest.json, as in Part B.

  lever          C0: lever surface (share of orbits where the ML policy would beat each adaptive comparator for a given overhead O
                 and reduction delta of its integration cost) and the 50 % iso-lines, from a policy ledger
  e3c_paired     Track 0.1 / 0.2: paired (McNemar) and risk-matched re-analysis of E3c, f = 0 against f = 0.05
  policy_c       Part C policy on one dataset: fit, tune the leapfrog gate on a selection slice, certify ONE end-to-end scale on the
                 calibration slice, evaluate on the evaluation slice, write the ledger (same columns as Part B's)
  osc_diag       C2: go / no-go diagnostic of the osculating periapsis features (exploratory, D_explore)
  sundman_pilot  C4: go / no-go pilot of the Sundman-time RK4 (oracle cost against the adaptive comparators, paradigm gap vs timescale
                 range, label quality, tau_c sweep, one global certified ds)
  freeze_policy  C7: fit the final policy (models + tuned thresholds) on D_train / D_select from a JSON config, write policy_frozen.joblib
  apply_policy   C8: apply a frozen policy to a new dataset: certify ONE end-to-end scale on n_cal rows, recalibrate the adaptive comparators on the
                 same rows, evaluate every other usable row, write the ledger (--no_cal: reuse the frozen scale, e.g. on D_lock_C)
  select         C7: one selection-set experiment per config (fit on D_train, certify one end-to-end scale on the calibration fold of D_select,
                 run on its evaluation fold); writes per-orbit costs so overheads can be added later
  compile_study  C3: accuracy and single-call cost (RHS-equivalents) of compiled model sizes M0 (linear), M1 (small trees), M2 (300 trees)

Run with the project's conda env:  /Users/arnavdeopura/miniforge3/envs/orbit-ml/bin/python run_part_c_3d.py <command> ...
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orbit_ml_cr3bp.config import MU, POSITION_TOL_KM  # noqa: E402
from orbit_ml_cr3bp import failure_study_3d as fs  # noqa: E402
from orbit_ml_cr3bp import policy_b_3d as pb  # noqa: E402
from orbit_ml_cr3bp import protocol_3d as proto  # noqa: E402
from run_failure_study_3d import make_run_dir, save_json, _ols_slope  # noqa: E402


def read_ledger(path):
    led = pd.read_csv(path, low_memory=False)
    for c in ("meets_tol", "violation", "has_choice", "abstain", "ml_ok"):
        if c in led.columns:
            led[c] = led[c].astype(bool)
    return led


# ---------------------------------------------------------------------------------------------------------------------
# C0: lever surface
# ---------------------------------------------------------------------------------------------------------------------

def cmd_lever(a):
    out = make_run_dir(a.out_dir, "lever", a)
    led = read_ledger(a.ledger)
    if "regime" not in led.columns or "h_band" not in led.columns:
        led = fs.add_derived(led)
    led["deploy_cell"] = led["regime"].astype(str) + "|" + led["h_band"].astype(str)
    surf = fs.lever_surface(led)
    surf.to_csv(os.path.join(out, "lever_surface.csv"), index=False)
    cells = fs.lever_surface(led, by="deploy_cell", O_grid=(0.0, 10.0, 50.0), delta_grid=(0.0, 0.3, 0.6))
    cells.to_csv(os.path.join(out, "lever_cells.csv"), index=False)
    iso = fs.lever_isoline(led)
    iso.to_csv(os.path.join(out, "lever_isoline.csv"), index=False)
    # oracle fixed step (delta = everything, O = 0 on the paradigm side): C_fix_star against each comparator
    orc = {}
    for comp in fs.COMPARATORS:
        col = f"{comp}_nfev"
        if col not in led.columns:
            continue
        m = led[col].notna() & led["C_fix_star"].notna()
        orc[comp] = {"n": int(m.sum()), "share_oracle_fixed_cheaper": float((led.loc[m, "C_fix_star"] < led.loc[m, col]).mean())}
    # the table of the plan's section 2.1, wide
    wide = surf[surf["group"] == "ALL"].pivot_table(index="comparator", columns=["O", "delta_dex"], values="share")
    save_json({"ledger": a.ledger, "n_orbits": int(len(led)), "n_met_tolerance": int(led["meets_tol"].sum()),
               "oracle_fixed": orc, "isoline": iso.to_dict(orient="records")}, os.path.join(out, "lever_report.json"))
    pd.set_option("display.width", 200)
    sel = [(0.0, 0.0), (0.0, 0.3), (0.0, 0.6), (10.0, 0.6), (50.0, 0.6), (200.0, 0.6)]
    print((100 * wide[[c for c in sel if c in wide.columns]]).round(1).to_string())
    print("\nDelta (dex) needed for a 50 % share, by overhead O:")
    print(iso.pivot_table(index="comparator", columns="O", values="delta_needed_dex").round(2).to_string())
    print("wrote", out)


# ---------------------------------------------------------------------------------------------------------------------
# Track 0.1 / 0.2: E3c re-analysed pairwise and at matched risk
# ---------------------------------------------------------------------------------------------------------------------

def _eval_rk4(reg_feats, reg, ev, cal_cache_pred, scale, comps, tol_km, n_jobs, c_extra):
    """Run the RK4-only policy on `ev` at one scale; returns per-orbit frame with meets, c_tot (no model cost) and comparator costs."""
    def predict(df):
        dt = 10.0 ** reg.predict(df[reg_feats].to_numpy(dtype=float))
        n = len(df)
        return {"rk4": {"feasible": np.ones(n, dtype=bool), "dt": dt},
                "leapfrog": {"feasible": np.zeros(n, dtype=bool), "dt": np.full(n, np.nan)}}
    ev_pred = predict(ev)
    res = pb.evaluate_orbits(ev, ev_pred, {"rk4": scale, "leapfrog": None}, rtols={}, do_spec=False, tol_km=tol_km,
                             n_jobs=n_jobs, candidates=("rk4",), progress=False)
    m = ev[["row_id"] + [f"{c}_nfev" for c in comps]].rename(columns={f"{c}_nfev": c for c in comps}).merge(res, on="row_id", how="left")
    m["meets"] = m["ml_ok"].fillna(False).astype(bool) & (m["ml_err_km"] <= tol_km)
    m["c_tot"] = m["ml_evals"] + 1.0 + c_extra
    return m


def mcnemar_exact(b, c):
    """Two-sided exact binomial test on the discordant pairs (b: only first wins, c: only second wins)."""
    from scipy.stats import binomtest
    n = b + c
    return 1.0 if n == 0 else float(binomtest(b, n, 0.5, alternative="two-sided").pvalue)


def cmd_e3c_paired(a):
    from orbit_ml_cr3bp import lookahead_3d as la
    from orbit_ml_cr3bp import experiments_b_3d as xb
    from orbit_ml_cr3bp.audit_3d import ReferenceCache, clopper_pearson
    from run_failure_experiments_3d import load_policy_dir
    out = make_run_dir(a.out_dir, "e3c_paired", a)
    U, train, cal, led, rep = load_policy_dir(a)
    lk = pd.read_csv(a.lookahead, low_memory=False)
    train, cal, ev = (x.merge(lk, on="row_id", how="left") for x in (train, cal, led))
    comps = ["RK45_cal", "DOP853_cal", "RK45_spec"]
    arms = {0.0: list(xb.FEATS_STATE)}
    for f in a.fractions:
        arms[f] = list(xb.FEATS_STATE) + la.feature_names(f)
    fr_all = list(a.fractions)
    if a.osc_arm:     # Track 4.2: the periapsis features as a free lookahead (arm key -1; charged 4 evaluations for the features)
        from orbit_ml_cr3bp.osculating_3d import add_osc_features, OSC_FEATURE_COLS
        train, ev = add_osc_features(train, MU), add_osc_features(ev, MU)
        arms[-1.0] = list(xb.FEATS_STATE) + OSC_FEATURE_COLS
        fr_all.append(-1.0)
    regs = {f: xb.fit_log_quantile(train, "rk4", a.quantile, features=ft, seed=42) for f, ft in arms.items()}
    scales = list(a.scales)
    per, rows = {}, []
    t0 = time.time()
    for f, ft in arms.items():
        c_la = 0.0 if f == 0 else (4.0 if f < 0 else ev[f"la{la.tag(f)}_nfev"].to_numpy(dtype=float))
        for s in scales:
            m = _eval_rk4(ft, regs[f], ev, None, s, comps, a.tol_km, a.n_jobs, c_la)
            per[(f, s)] = m
            n = len(m)
            k = int((~m["meets"]).sum())
            row = {"f": f, "scale": s, "n": n, "violations": k, "violation_rate": k / n, "viol_lo": clopper_pearson(k, n)[0],
                   "viol_hi": clopper_pearson(k, n)[1], "median_c_tot": float(m.loc[m["meets"], "c_tot"].median())}
            for c in comps:
                w = m["meets"] & (m["c_tot"] < m[c])
                row[f"win_{c}"] = float(w.mean())
            rows.append(row)
            print(f"f={f:<5} scale {s:<4} viol {k / n:.4f}  win vs RK45-cal {row['win_RK45_cal']:.4f}  median cost {row['median_c_tot']:.0f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(out, "e3c_scale_grid.csv"), index=False)
    # --- 0.1: paired test at the Part-B certified scales (0.5 for f = 0, 0.6 for f = 0.05)
    paired = {}
    for comp in comps:
        for (s0, s1) in a.paired_scales:
            for f1 in fr_all:
                m0, m1 = per[(0.0, s0)], per[(f1, s1)]
                w0 = (m0["meets"] & (m0["c_tot"] < m0[comp])).to_numpy()
                w1 = (m1["meets"] & (m1["c_tot"] < m1[comp])).to_numpy()
                b, c = int((w1 & ~w0).sum()), int((w0 & ~w1).sum())
                paired[f"{comp}|f0@{s0}|f{f1}@{s1}"] = {"wins_f0": int(w0.sum()), "wins_f1": int(w1.sum()), "only_f1": b,
                                                          "only_f0": c, "n": int(len(w0)), "diff": float(w1.mean() - w0.mean()),
                                                          "mcnemar_exact_p": mcnemar_exact(b, c)}
    # --- 0.2: equal realized risk. For each arm interpolate (violation rate -> win share, log median cost) along the scale grid.
    matched = {}
    for target in a.risk_targets:
        for f in arms:
            t = tab[tab["f"] == f].sort_values("violation_rate")
            x = t["violation_rate"].to_numpy()
            if not (x.min() <= target <= x.max()):
                matched[f"risk{target}|f{f}"] = None
                continue
            matched[f"risk{target}|f{f}"] = {
                "win_RK45_cal": float(np.interp(target, x, t["win_RK45_cal"].to_numpy())),
                "win_DOP853_cal": float(np.interp(target, x, t["win_DOP853_cal"].to_numpy())),
                "median_c_tot": float(10 ** np.interp(target, x, np.log10(t["median_c_tot"].to_numpy())))}
    save_json({"scales": scales, "paired_mcnemar": paired, "risk_matched": matched, "table": rows}, os.path.join(out, "e3c_paired_report.json"))
    for k, v in paired.items():
        if k.startswith("RK45_cal"):
            print(k, v)
    print("risk matched:")
    for k, v in matched.items():
        print(" ", k, v)
    print("wrote", out)


# ---------------------------------------------------------------------------------------------------------------------
# policy_c: the Part C policy on one dataset (development runs on D_explore; later steps add features, models and guards)
# ---------------------------------------------------------------------------------------------------------------------

def cmd_policy_c(a):
    import joblib
    from orbit_ml_cr3bp import policy_c_3d as pc
    from orbit_ml_cr3bp.audit_3d import clopper_pearson
    from run_failure_study_3d import load_partb, split_rows
    out = make_run_dir(a.out_dir, "policy_c", a)
    t_start = time.time()
    d, meta = load_partb(a.data)
    U = fs.add_derived(pb.usable_rows(d).reset_index(drop=True), a.mu)
    if a.features == "osc":
        from orbit_ml_cr3bp.osculating_3d import add_osc_features
        U = add_osc_features(U, a.mu)
    train, cal, ev = split_rows(U, a.n_cal, a.n_eval, a.seed)
    sel = np.array([], dtype=int)
    if a.n_sel:
        rng = np.random.RandomState(a.seed + 1)
        perm = rng.permutation(train)
        sel, train = np.sort(perm[:a.n_sel]), np.sort(perm[a.n_sel:])
    feats = pc.FEATS_OSC if a.features == "osc" else pb.FEATS_STATE
    print(f"usable {len(U)}; train {len(train)}, selection {len(sel)}, calibration {len(cal)}, evaluation {len(ev)}", flush=True)
    models = pc.fit_policy_c(U.iloc[train], feats, quantile=a.quantile, seed=a.seed, lf_gate=a.lf_gate, rk4_classifier=False)
    cal_df, ev_df = U.iloc[cal].reset_index(drop=True), U.iloc[ev].reset_index(drop=True)
    raw_cal, raw_ev = pc.predict_raw_c(models, cal_df), pc.predict_raw_c(models, ev_df)
    gate_info = {"lf_gate": a.lf_gate}
    p_star, g_star = a.p_star, a.g_star
    if a.lf_gate in ("clf", "censored") and len(sel) and a.tune_gate:
        sel_df = U.iloc[sel].reset_index(drop=True)
        thr, tab = pc.select_gate_threshold(pc.predict_raw_c(models, sel_df), sel_df, max_rate=a.gate_budget, lf_gate=a.lf_gate)
        tab.to_csv(os.path.join(out, "gate_table.csv"), index=False)
        if thr is None:
            raise RuntimeError("no gate threshold meets the infeasible-choice budget on the selection rows")
        if a.lf_gate == "clf":
            p_star = thr
        else:
            g_star = thr
        gate_info.update({"tuned_on": "selection slice", "n_sel": int(len(sel)), "budget": a.gate_budget, "threshold": thr})
        print(f"  gate threshold {thr} (infeasible-choice budget {a.gate_budget}) on {len(sel)} selection rows", flush=True)
    gate_info.update({"p_star": p_star, "g_star": g_star})
    pred_cal = pc.apply_gate(raw_cal, p_star=p_star, g_star=g_star, lf_gate=a.lf_gate)
    pred_ev = pc.apply_gate(raw_ev, p_star=p_star, g_star=g_star, lf_gate=a.lf_gate)
    if a.rk4_only:        # RK4 is the only candidate; leapfrog is never chosen (its conditional risk and gate are not part of the policy)
        for pr in (pred_cal, pred_ev):
            pr["leapfrog"]["feasible"] = np.zeros_like(pr["leapfrog"]["feasible"])
    # --- end-to-end certification on the calibration rows
    grid = pc.FIXED_SEQ_GRID if a.grid == "fine" else tuple(__import__("orbit_ml_cr3bp.audit_3d", fromlist=["x"]).LTT_SCALE_GRID)
    ref_cal = pc.reference_final_positions(cal_df, a.mu, a.n_jobs)
    guard_g = math.inf if a.guard_g is None else a.guard_g
    fallback, abstain_cal, abstain_ev = None, None, None
    items = pb.reference_items(cal_df, a.mu)
    rtols, rtol_rep = {}, {}
    if not math.isinf(guard_g) or a.abstain_rule != "none":
        # the adaptive fallback must exist BEFORE the policy is certified: certified rtol of RK45 on the same calibration rows
        r0 = pb.certify_rtol_lazy(cal_df, "RK45", tol_km=a.tol_km, mu=a.mu, equal_risk_target=None, n_jobs=a.n_jobs, items=items)
        fallback = (r0["chosen_rtol"], "RK45")
        print(f"  fallback: RK45 at the certified rtol {fallback[0]}", flush=True)
        if a.abstain_rule != "none":
            abstain_cal = pc.abstain_flags(cal_df, a.abstain_rule, a.abstain_logtau)
            abstain_ev = pc.abstain_flags(ev_df, a.abstain_rule, a.abstain_logtau)
    cert = pc.certify_end_to_end(cal_df, pred_cal, ref_cal, tol_km=a.tol_km, grid=grid, rule=a.rule, mu=a.mu, n_jobs=a.n_jobs,
                                 by_choice=a.by_choice, guard_g=guard_g, fallback=fallback, abstain=abstain_cal)
    print(f"  certified: scale {cert['scale']} by-choice {cert['scales_by_choice']}; cal violations {cert.get('violations')} of "
          f"{cert['n_policy_runs']}; conditional {json_brief(cert['conditional_risk'])}", flush=True)
    if cert["scale"] is None and not a.by_choice:
        raise RuntimeError("no scale on the grid is certified")
    cal_idx, cal_raw = pc.choose_unscaled(pred_cal)
    ev_idx, ev_raw = pc.choose_unscaled(pred_ev)
    names = list(pc.CANDIDATES)
    if not a.by_choice:
        scales = {k: cert["scale"] for k in names}
        cal_rate = cert["violation_rate"]
        sc_ev = np.full(len(ev_df), cert["scale"])
    else:
        scales = dict(cert["scales_by_choice"])
        sc_ev = np.array([scales.get(names[j], np.nan) if j >= 0 else np.nan for j in ev_idx])
        sc_cal = np.array([scales.get(names[j], np.nan) if j >= 0 else np.nan for j in cal_idx])
        rc = pc.evaluate_orbits_c(cal_df, cal_idx, cal_raw, sc_cal, rtols={}, do_spec=False, tol_km=a.tol_km, mu=a.mu, n_jobs=a.n_jobs,
                                  progress=False, ref_pos=ref_cal)
        has = rc["ml_integrator"].notna()
        cal_rate = float(1.0 - (rc["ml_ok"] & (rc["ml_err_km"] <= a.tol_km))[has].mean())
    print(f"  deployed-policy violation rate on calibration rows: {cal_rate:.4f}", flush=True)
    # --- adaptive comparators calibrated on the same rows (equal-risk target = the deployed policy's own rate)
    for m in pb.METHODS:
        r = pb.certify_rtol_lazy(cal_df, m, tol_km=a.tol_km, mu=a.mu, equal_risk_target=cal_rate, n_jobs=a.n_jobs, items=items)
        rtol_rep[m] = r
        rtols[(m, "cal")] = r["chosen_rtol"]
        rtols[(m, "eqr")] = r["equal_risk_rtol"]
        print(f"  {m}: certified rtol {r['chosen_rtol']}, equal-risk rtol {r['equal_risk_rtol']}", flush=True)
    # --- evaluation pass
    res = pc.evaluate_orbits_c(ev_df, ev_idx, ev_raw, sc_ev, rtols=rtols, do_spec=not a.no_spec, tol_km=a.tol_km, mu=a.mu, n_jobs=a.n_jobs,
                               guard_g=guard_g, fallback=fallback, abstain=abstain_ev)
    raw = pb.candidate_raw_predictions(ev_df, pred_ev)
    led = fs.build_ledger(ev_df, res.merge(raw, on="row_id"), a.c_pred_fixed, c_feat=1.0, tol_km=a.tol_km)
    led.to_csv(os.path.join(out, "ledger.csv"), index=False)
    cal_df.assign(split="cal").to_csv(os.path.join(out, "cal_rows.csv"), index=False)
    joblib.dump({"models": models, "scales": scales, "rtols": {f"{m}|{t}": v for (m, t), v in rtols.items()}, "gate": gate_info,
                 "features": feats, "quantile": a.quantile, "guard_g": None if math.isinf(guard_g) else guard_g, "fallback": fallback},
                os.path.join(out, "policy.joblib"))
    # --- summary of the deployed policy on the evaluation rows
    sumry = {"n_eval": int(len(led)), "risk_ledger": fs.risk_ledger(led, a.tol_km), "by_integrator": {}}
    for k in pc.CANDIDATES:
        g = led[led["ml_integrator"] == k]
        if len(g):
            nb = int(g["violation"].sum())
            sumry["by_integrator"][k] = {"n": int(len(g)), "share": float(len(g) / len(led)), "violations": nb, "rate": nb / len(g),
                                         "cp95": list(clopper_pearson(nb, len(g))),
                                         "infeasible_choice": int((g["dstar_chosen"].isna()).sum())}
    ok = led["meets_tol"]
    sumry["fallback_share"] = float(led["ml_fallback"].mean()) if "ml_fallback" in led.columns else 0.0
    sumry["abstain_share"] = float((led["ml_integrator"] == "adaptive").mean())
    sumry["mean_log10_C_ml_tot_met"] = float(np.log10(led.loc[ok, "C_ml_tot"]).mean())
    sumry["median_C_ml_tot_met"] = float(led.loc[ok, "C_ml_tot"].median())
    for comp in fs.COMPARATORS:
        if f"{comp}_nfev" in led.columns and led[f"{comp}_nfev"].notna().any():
            m_ = ok & led[f"{comp}_nfev"].notna()
            sumry[f"share_total_cheaper_{comp}"] = float((led.loc[m_, "C_ml_tot"] < led.loc[m_, f"{comp}_nfev"]).mean())
    sumry["median_C_ml_int_met"] = float(led.loc[ok, "C_ml_int"].median())
    sumry["median_C_oracle_fixed"] = float(led.loc[ok, "C_fix_star"].median())
    for comp in fs.COMPARATORS:
        if f"{comp}_nfev" in led.columns and led[f"{comp}_nfev"].notna().any():
            m_ = ok & led[f"{comp}_nfev"].notna()
            sumry[f"share_integration_cheaper_{comp}"] = float((led.loc[m_, "C_ml_int"] < led.loc[m_, f"{comp}_nfev"]).mean())
    save_json({"args": vars(a), "n_usable": int(len(U)), "n_train": int(len(train)), "n_sel": int(len(sel)), "n_cal": int(len(cal)),
               "n_eval": int(len(ev)), "gate": gate_info, "certificate": cert, "cal_policy_violation_rate": cal_rate,
               "rtol_calibration": rtol_rep, "summary": sumry, "wall_seconds": time.time() - t_start},
              os.path.join(out, "policy_c_report.json"))
    print(json.dumps(proto.json_safe(sumry), indent=1))
    print(f"wrote {out}  ({time.time() - t_start:.0f}s)")


def cmd_osc_diag(a):
    from scipy.stats import spearmanr
    from orbit_ml_cr3bp import experiments_b_3d as xb
    from orbit_ml_cr3bp.osculating_3d import add_osc_features, OSC_FEATURE_COLS
    from run_failure_study_3d import load_partb, split_rows
    out = make_run_dir(a.out_dir, "osc_diag", a)
    d, _ = load_partb(a.data)
    U = fs.add_derived(pb.usable_rows(d).reset_index(drop=True), a.mu)
    U = add_osc_features(U, a.mu)
    U["log_tau_min"] = np.log10(U["tau_min"])
    train, cal, ev = split_rows(U, a.n_cal, a.n_eval, a.seed)
    tr, te = U.iloc[train].reset_index(drop=True), U.iloc[ev].reset_index(drop=True)
    rep = {"n_usable": int(len(U)), "n_train": int(len(tr)), "n_eval": int(len(te))}
    # --- 1. does the analytic timescale track the reference's shortest timescale?
    m = np.isfinite(U["osc_logtau_hat"]) & np.isfinite(U["log_tau_min"])
    rep["spearman_logtauhat_logtaumin"] = {"all": float(spearmanr(U.loc[m, "osc_logtau_hat"], U.loc[m, "log_tau_min"]).statistic)}
    for by in ("regime", "r2_band", "h_band"):
        rep["spearman_logtauhat_logtaumin"][by] = {str(g): float(spearmanr(x["osc_logtau_hat"], x["log_tau_min"]).statistic)
                                                   for g, x in U[m].groupby(by) if len(x) >= 50}
    # --- 2. is the predicted secondary periapsis the real closest approach?
    r2 = U["osc_reach_2"] == 1.0
    rp2 = 10 ** U["osc_logrp_2"]
    close = U["min_r2"] < 0.05
    ratio = (rp2 / U["min_r2"])
    rep["periapsis_2"] = {"share_reach2": float(r2.mean()),
                          "within_factor2_given_reach2": float(((ratio > 0.5) & (ratio < 2.0))[r2].mean()),
                          "close_orbits_n": int(close.sum()), "close_with_reach2": float(r2[close].mean()),
                          "close_within_factor2": float((((ratio > 0.5) & (ratio < 2.0)) & r2)[close].mean()),
                          "far_orbits_with_reach2": float(r2[~close].mean())}
    # --- 3. the same model with and without the features (3 seeds), RK4 and leapfrog
    sets = {"state": list(xb.FEATS_STATE), "state+osc_all": list(xb.FEATS_STATE) + OSC_FEATURE_COLS,
            "state+tau_hat": list(xb.FEATS_STATE) + ["osc_logtau_hat"],
            "state+secondary": list(xb.FEATS_STATE) + [c for c in OSC_FEATURE_COLS if c.endswith("_2")] + ["osc_logtau_hat"]}
    rows = []
    for integ in ("rk4", "leapfrog"):
        e = te[te[f"{integ}_success"] == 1].reset_index(drop=True)
        y = np.log10(e[f"dt_{integ}"].to_numpy(dtype=float))
        for name, ft in sets.items():
            for seed in range(a.seeds):
                reg = xb.fit_log_quantile(tr, integ, a.quantile, features=ft, seed=seed)
                yh = reg.predict(e[ft].to_numpy(dtype=float))
                res = np.abs(yh - y)
                row = {"integrator": integ, "features": name, "seed": seed, "median_abs": float(np.median(res)), "p90_abs": float(np.quantile(res, 0.9)),
                       "pinball": float(xb.pinball(y, yh, a.quantile).mean()), "over_rate": float((yh > y).mean())}
                for band in ("<0.05", "0.05-0.17", "0.17-0.8", ">0.8"):
                    mm = (e["r2_band"] == band).to_numpy()
                    row[f"median_abs_r2{band}"] = float(np.median(res[mm])) if mm.sum() else float("nan")
                rows.append(row)
                print(integ, name, seed, round(row["median_abs"], 4), flush=True)
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(out, "osc_model_comparison.csv"), index=False)
    g = tab.groupby(["integrator", "features"]).mean(numeric_only=True).drop(columns="seed")
    print(g.round(4).to_string())
    summary = {}
    for integ in ("rk4", "leapfrog"):
        base = g.loc[(integ, "state")]
        for name in sets:
            if name == "state":
                continue
            cur = g.loc[(integ, name)]
            summary[f"{integ}|{name}"] = {"median_abs_change": float(cur["median_abs"] - base["median_abs"]),
                                          "pinball_change": float(cur["pinball"] - base["pinball"]),
                                          "over_rate": [float(base["over_rate"]), float(cur["over_rate"])],
                                          "close_cells_change": {b: float(cur[f"median_abs_r2{b}"] - base[f"median_abs_r2{b}"]) for b in ("<0.05", "0.05-0.17")}}
    best = summary["rk4|state+osc_all"]
    go = (best["median_abs_change"] <= -a.go_overall) or (min(best["close_cells_change"].values()) <= -a.go_close)
    rep["model_comparison"] = summary
    rep["go_rule"] = {"overall_dex": a.go_overall, "close_dex": a.go_close}
    rep["go"] = bool(go)
    save_json(rep, os.path.join(out, "osc_diag_report.json"))
    print(json.dumps(proto.json_safe({k: v for k, v in rep.items() if k != "model_comparison"}), indent=1))
    print(json.dumps(proto.json_safe(summary), indent=1))
    print("GO" if go else "NO-GO", "wrote", out)


COMPILE_CONFIGS = {
    # name: (reg_kw, clf_kw)   learning rates rise as the number of trees falls so a small ensemble still moves off its baseline
    "M1_10x3": ({"max_iter": 10, "max_depth": 3, "learning_rate": 0.3}, {"max_iter": 10, "max_depth": 3, "learning_rate": 0.3}),
    "M1_25x3": ({"max_iter": 25, "max_depth": 3, "learning_rate": 0.2}, {"max_iter": 25, "max_depth": 3, "learning_rate": 0.2}),
    "M1_50x4": ({"max_iter": 50, "max_depth": 4, "learning_rate": 0.15}, {"max_iter": 40, "max_depth": 4, "learning_rate": 0.15}),
    "M1_100x5": ({"max_iter": 100, "max_depth": 5, "learning_rate": 0.1}, {"max_iter": 60, "max_depth": 5, "learning_rate": 0.1}),
    "M2_300": ({"max_iter": 300, "learning_rate": 0.05}, {"max_iter": 100}),
}


def cmd_compile_study(a):
    from orbit_ml_cr3bp import policy_c_3d as pc
    from orbit_ml_cr3bp import compiled_model_3d as cm
    from orbit_ml_cr3bp import experiments_b_3d as xb
    from orbit_ml_cr3bp.osculating_3d import add_osc_features
    from run_failure_study_3d import load_partb, split_rows
    out = make_run_dir(a.out_dir, "compile_study", a)
    d, _ = load_partb(a.data)
    U = add_osc_features(fs.add_derived(pb.usable_rows(d).reset_index(drop=True), a.mu), a.mu)
    train, cal, ev = split_rows(U, a.n_cal, a.n_eval, a.seed)
    tr, te = U.iloc[train].reset_index(drop=True), U.iloc[ev].reset_index(drop=True)
    st = te[pb.STATE_COLS].to_numpy(dtype=float)
    Tt = te["sim_time"].to_numpy(dtype=float)
    rows = []

    def accuracy(pred_log, integ):
        m = te[f"{integ}_success"].to_numpy() == 1
        y = np.log10(te[f"dt_{integ}"].to_numpy(dtype=float)[m])
        yh = pred_log[m]
        return {f"{integ}_median_abs": float(np.median(np.abs(yh - y))), f"{integ}_p90_abs": float(np.quantile(np.abs(yh - y), 0.9)),
                f"{integ}_pinball": float(xb.pinball(y, yh, a.quantile).mean()), f"{integ}_over": float((yh > y).mean())}

    for feat_name, feats in (("osc", pc.FEATS_OSC), ("state", pb.FEATS_STATE)):
        for name, (rk, ck) in COMPILE_CONFIGS.items():
            if feat_name == "state" and name not in ("M1_50x4", "M2_300"):
                continue
            t0 = time.time()
            models = pc.fit_policy_c(tr, feats, quantile=a.quantile, seed=a.seed, lf_gate="clf", reg_kw=rk, clf_kw=ck)
            raw = pc.predict_raw_c(models, te)
            cp = cm.CompiledHGBPolicy(models, p_star=a.p_star)
            row = {"model": name, "features": feat_name, "fit_s": time.time() - t0, **accuracy(raw["rk4"]["logdt"], "rk4"),
                   **accuracy(raw["leapfrog"]["logdt"], "leapfrog"), "size": cp.size()}
            if a.time:
                c = cm.measure_single_call_cost(cp, st, Tt, repeats=a.repeats)
                row.update({k: c[k] for k in ("rhs_equiv", "rhs_equiv_range", "features_rhs_equiv", "policy_ns_per_call", "rhs_ns_per_call",
                                              "python_dispatch_rhs_equiv")})
                row["batched_rhs_equiv"] = cm.batched_cost(cp, st, Tt) / c["rhs_ns_per_call"]
            rows.append(row)
            print(name, feat_name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k not in ("size",)}, flush=True)
    # M0: linear quantile policy on engineered features (censored leapfrog gate)
    t0 = time.time()
    lin = cm.fit_linear_policy(tr, a.quantile, a.mu, max_rows=(a.lin_rows or None), seed=a.seed)
    cl = cm.CompiledLinearPolicy(lin, g_star=a.g_star)
    lr, ll = cl.raw_log_dt(st, Tt)
    row = {"model": "M0_linear", "features": "osc", "fit_s": time.time() - t0, **accuracy(lr, "rk4"), **accuracy(ll, "leapfrog")}
    if a.time:
        c = measure_linear = cm.measure_linear_cost(cl, st, Tt, repeats=a.repeats)
        row.update({k: c[k] for k in ("rhs_equiv", "rhs_equiv_range", "policy_ns_per_call", "rhs_ns_per_call")})
    rows.append(row)
    print("M0_linear", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(out, "compile_study.csv"), index=False)
    gate = a.gate_rhs
    elig = tab[tab.get("rhs_equiv", pd.Series(dtype=float)) <= gate] if "rhs_equiv" in tab else tab.iloc[0:0]
    save_json({"table": rows, "gate_rhs_equiv": gate, "eligible": elig["model"].tolist() + [""] if len(elig) else [],
               "machine": proto.json_safe({"platform": sys.platform, "python": sys.version}), "numba": __import__("numba").__version__,
               "sklearn": __import__("sklearn").__version__}, os.path.join(out, "compile_study_report.json"))
    pd.set_option("display.width", 250)
    cols = [c for c in ("model", "features", "rk4_median_abs", "rk4_pinball", "leapfrog_median_abs", "rhs_equiv", "batched_rhs_equiv") if c in tab.columns]
    print(tab[cols].round(3).to_string())
    print("wrote", out)


def _tau_label_job(a):
    from orbit_ml_cr3bp.data_3d import _sundman_labels
    from orbit_ml_cr3bp.physics_3d import reference_trajectory_dense_3d
    rid, state, T, mu, tol_km, tau_c = a
    sol, reached = reference_trajectory_dense_3d(list(state), T, mu)
    if not reached:
        return {"row_id": rid}
    r = _sundman_labels(list(state), T, mu, tol_km, (sol, reached), tau_c)
    r["row_id"] = rid
    return r


def _global_ds_job(a):
    from orbit_ml_cr3bp.sundman_3d import integrate_endpoint_rk4s
    from orbit_ml_cr3bp.config import du_km_for_mu
    state, T, ref_pos, ds, mu, tau_c = a
    ok, _p, curr, _tp, _t, _r, ns, nf = integrate_endpoint_rk4s(state, ds, T, mu, 4_000_000, tau_c=tau_c)
    if not ok:
        return False, float("nan"), 4 * (ns + nf)
    return True, float(np.linalg.norm(np.asarray(curr)[:3] - ref_pos)) * du_km_for_mu(mu), 4 * (ns + nf)


def cmd_sundman_pilot(a):
    from multiprocessing import Pool
    from scipy.stats import spearmanr, binom
    from orbit_ml_cr3bp import policy_c_3d as pc
    from orbit_ml_cr3bp.audit_3d import dense_sweep_log_3d, clopper_pearson
    from orbit_ml_cr3bp.sundman_3d import DS_MIN, DS_MAX
    out = make_run_dir(a.out_dir, "sundman_pilot", a)
    d = pd.read_csv(a.data, low_memory=False)
    for c in ("rk4_censored", "leapfrog_censored", "euler_censored", "ftle_nan", "reference_divergent", "rk4s_censored"):
        if c in d.columns:
            d[c] = d[c].map({True: True, False: False, "True": True, "False": False}).fillna(False).astype(bool)
    d["row_id"] = np.arange(len(d))
    meta = proto.read_metadata(a.data) or {}
    if meta.get("label_pipeline_version") != 5:
        raise ValueError("the Sundman pilot needs a v5 dataset")
    U = fs.add_derived(pb.usable_rows(d).reset_index(drop=True), a.mu)
    n_all = len(U)
    rep = {"n_candidates": int(len(d)), "n_usable_v4": int(n_all), "tau_c": meta.get("sundman_tau_c")}
    # --- label quality
    U = U[U["rk4s_success"] == 1].reset_index(drop=True)
    rep["rk4s_success_share"] = float(len(U) / n_all)
    rep["rk4s_censored_share"] = float(U["rk4s_censored"].mean())
    rep["rk4_censored_share"] = float(U["rk4_censored"].mean())
    # --- adaptive comparators on the same orbits (spec per orbit; cal / eqr at the Part B certified rtols)
    ref_led = read_ledger(a.rtol_ledger)
    rt = {(m, t): float(ref_led[f"{m}_{t}_rtol"].dropna().iloc[0]) for m in pb.METHODS for t in ("cal", "eqr")}
    rep["adaptive_rtols_from_part_b_explore"] = {f"{m}|{t}": v for (m, t), v in rt.items()}
    jobs = [(int(U["row_id"].iloc[i]), U[pb.STATE_COLS].to_numpy(dtype=float)[i], float(U["sim_time"].iloc[i]), None, None, a.tol_km, a.mu, rt, True,
             pb.METHODS) for i in range(len(U))]
    t0 = time.time()
    with Pool(a.n_jobs) as pool:
        res = pd.DataFrame(list(pool.imap_unordered(pb._orbit_job, jobs, chunksize=4)))
    print(f"adaptive comparators done ({time.time() - t0:.0f}s)", flush=True)
    M = U.merge(res, on="row_id", how="left")
    fx = fs.oracle_fixed(M)
    M = pd.concat([M, fx], axis=1)
    M["C_s"] = M["rk4s_evals"]
    comps = [f"{m}_{t}" for t in ("spec", "cal", "eqr") for m in pb.METHODS]
    rows = []
    for c in comps:
        col = f"{c}_nfev"
        h = M[col].notna() & M["C_s"].notna() & M["C_fix_star"].notna()
        g = M[h]
        Ps = np.log10(g["C_s"] / g[col])
        P = np.log10(g["C_fix_star"] / g[col])
        k1, k0 = int((g["C_s"] < g[col]).sum()), int((g["C_fix_star"] < g[col]).sum())
        row = {"comparator": c, "n": int(h.sum()), "sundman_cheaper": k1 / h.sum(), "sundman_ci": list(clopper_pearson(k1, int(h.sum()))),
               "fixed_cheaper": k0 / h.sum(), "median_P_s": float(Ps.median()), "median_P": float(P.median()),
               "rho_logdelta_Ps": fs.spearman_bootstrap(g["log_delta_phys"], Ps, n_boot=500),
               "rho_logdelta_P": fs.spearman_bootstrap(g["log_delta_phys"], P, n_boot=500),
               "slope_Ps": _ols_slope(g["log_delta_phys"], Ps, n_boot=500), "slope_P": _ols_slope(g["log_delta_phys"], P, n_boot=500)}
        rows.append(row)
        print(c, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k in ("n", "sundman_cheaper", "fixed_cheaper", "median_P_s", "median_P")},
              "rho Ps", round(row["rho_logdelta_Ps"][0], 3), "rho P", round(row["rho_logdelta_P"][0], 3), flush=True)
    rep["oracle_vs_adaptive"] = rows
    # --- by deployable cell (regime x horizon band), against RK45-spec
    g = M[M["RK45_spec_nfev"].notna() & M["C_s"].notna()].copy()
    g["cellkey"] = g["regime"].astype(str) + "|" + g["h_band"].astype(str)
    rep["by_cell_vs_RK45_spec"] = [{"cell": k, "n": int(len(x)), "sundman_cheaper": float((x["C_s"] < x["RK45_spec_nfev"]).mean()),
                                    "fixed_cheaper": float((x["C_fix_star"] < x["RK45_spec_nfev"]).mean()),
                                    "median_C_s": float(x["C_s"].median()), "median_C_fix": float(x["C_fix_star"].median()),
                                    "median_RK45_spec": float(x["RK45_spec_nfev"].median())} for k, x in g.groupby("cellkey") if len(x) >= 20]
    # --- dense sweep (monotonicity of the Sundman label in ds)
    rows_sw = np.random.RandomState(a.seed).choice(len(U), size=min(a.n_sweep, len(U)), replace=False)
    sw = _dense_sweep_rk4s(U, rows_sw, a.tol_km, a.mu, a.n_jobs)
    sw.to_csv(os.path.join(out, "dense_sweep_rk4s.csv"), index=False)
    k = int(sw["non_monotone"].sum())
    rep["dense_sweep"] = {"n": int(len(sw)), "non_monotone": k, "frac": k / max(len(sw), 1), "ci95": list(clopper_pearson(k, len(sw))),
                          "label_window_has_fail": float((sw["fails_in_label_window"] > 0).mean())}
    print("dense sweep", rep["dense_sweep"], flush=True)
    # --- one global certified ds (no ML): fixed-sequence on a calibration half
    perm = np.random.RandomState(a.seed + 7).permutation(len(M))
    cal_i, ev_i = perm[: len(M) // 2], perm[len(M) // 2:]
    ref = pc.reference_final_positions(M, a.mu, a.n_jobs)
    tau = float(meta.get("sundman_tau_c") or 1.0)
    st = M[pb.STATE_COLS].to_numpy(dtype=float)
    T = M["sim_time"].to_numpy(dtype=float)
    grid = [float(0.01 * 2 ** (j / 4)) for j in range(0, 33)]
    chosen, chain = None, []
    with Pool(a.n_jobs) as pool:
        def run(ds, idx):
            return pool.map(_global_ds_job, [(st[i], T[i], ref[i], ds, a.mu, tau) for i in idx], chunksize=8)
        for ds in grid:
            r = run(ds, cal_i)
            kk = int(sum((not x[0]) or (x[1] > a.tol_km) for x in r))
            p = float(binom.cdf(kk, len(cal_i), 0.05))
            chain.append({"ds": ds, "violations": kk, "n": len(cal_i), "p": p})
            if p > 0.1:
                break
            chosen = ds
        rep["global_ds"] = {"chosen": chosen, "chain": chain}
        if chosen is not None:
            r = run(chosen, ev_i)
            ok = np.array([x[0] and x[1] <= a.tol_km for x in r])
            cost = np.array([x[2] for x in r], dtype=float)
            Me = M.iloc[ev_i]
            gl = {"n_eval": int(len(ev_i)), "violation_rate": float(1 - ok.mean()), "median_cost": float(np.median(cost[ok]))}
            for c in comps:
                col = f"{c}_nfev"
                h = ok & Me[col].notna().to_numpy()
                gl[f"cheaper_than_{c}"] = float((cost[h] < Me[col].to_numpy()[h]).mean()) if h.any() else None
            rep["global_ds"]["eval"] = gl
            print("global ds", chosen, gl, flush=True)
    # --- tau_c sweep of the oracle label (cost only)
    sweep = {}
    for tc in a.tau_c_list:
        with Pool(a.n_jobs) as pool:
            rs = pd.DataFrame(pool.map(_tau_label_job, [(int(M["row_id"].iloc[i]), M[pb.STATE_COLS].to_numpy(dtype=float)[i], float(T[i]), a.mu,
                                                         a.tol_km, tc) for i in range(len(M))], chunksize=8))
        x = M[["row_id", "RK45_spec_nfev", "C_fix_star"]].merge(rs[["row_id", "rk4s_evals", "rk4s_censored"]], on="row_id")
        h = x["RK45_spec_nfev"].notna() & x["rk4s_evals"].notna()
        sweep[str(tc)] = {"n": int(h.sum()), "oracle_cheaper_than_RK45_spec": float((x.loc[h, "rk4s_evals"] < x.loc[h, "RK45_spec_nfev"]).mean()),
                          "median_evals": float(x.loc[h, "rk4s_evals"].median()), "censored": float(x["rk4s_censored"].mean())}
        print("tau_c", tc, sweep[str(tc)], flush=True)
    rep["tau_c_sweep"] = sweep
    M.to_csv(os.path.join(out, "pilot_table.csv"), index=False)
    spec = [r for r in rows if r["comparator"] == "RK45_spec"][0]
    rho_p, rho_ps = spec["rho_logdelta_P"][0], spec["rho_logdelta_Ps"][0]
    # plan 6.5: GO = oracle cheaper than RK45-spec on >= go_share of orbits AND the Spearman of the gap against log delta_phys is at least
    # go_rho_drop lower than for the physical-time fixed step. Plan 6.3 (label acceptance on the pilot) is reported separately: censored
    # < 2 %, non-monotone in ds < 2 % on the dense log sweep, convergence slope ~ 4 (the slope is a unit test).
    go = (spec["sundman_cheaper"] >= a.go_share) and (rho_ps <= rho_p - a.go_rho_drop)
    rep["go_rule"] = {"share_vs_RK45_spec": a.go_share, "rho_drop": a.go_rho_drop}
    rep["label_acceptance_63"] = {"censored_below_2pct": bool(rep["rk4s_censored_share"] < 0.02),
                                  "non_monotone_below_2pct": bool(rep["dense_sweep"]["frac"] < 0.02),
                                  "label_window_failures_below_2pct": bool(rep["dense_sweep"]["label_window_has_fail"] < 0.02),
                                  "censored": rep["rk4s_censored_share"], "non_monotone": rep["dense_sweep"]["frac"],
                                  "label_window_has_fail": rep["dense_sweep"]["label_window_has_fail"]}
    rep["go"] = bool(go)
    rep["go_detail"] = {"sundman_cheaper_RK45_spec": spec["sundman_cheaper"], "rho_P": rho_p, "rho_Ps": rho_ps}
    save_json(rep, os.path.join(out, "sundman_pilot_report.json"))
    print("GO" if go else "NO-GO", rep["go_detail"], "wrote", out)


def _dense_sweep_rk4s(U, rows, tol_km, mu, n_jobs):
    from orbit_ml_cr3bp.audit_3d import ReferenceCache, final_position_error_km
    from orbit_ml_cr3bp.sundman_3d import DS_MIN, DS_MAX
    from orbit_ml_cr3bp.physics_3d import Rk4sStep
    states = U[pb.STATE_COLS].to_numpy(dtype=float)
    cache = ReferenceCache(mu)
    recs = []
    ratio = 2.0 ** (1.0 / 16.0)
    n = int(np.floor(np.log(DS_MAX / DS_MIN) / np.log(ratio))) + 1
    grid = np.sort(DS_MAX * ratio ** (-np.arange(n)))
    for i in rows:
        T = float(U["sim_time"].iloc[i])
        sol, reached = cache.get(int(i), states[i], T)
        ref = sol.sol(T)[:3]
        ok = np.array([(lambda e: e is not None and e <= tol_km)(final_position_error_km("rk4s", list(states[i]), float(ds), T, ref, mu)[0]) for ds in grid])
        label = float(U["dt_rk4s"].iloc[i])
        fails = np.where(~ok)[0]
        win = (grid >= label / 4.0 - 1e-15) & (grid <= label + 1e-15)
        recs.append({"orbit_row": int(i), "label": label, "non_monotone": bool(len(fails) > 0 and ok[fails[0] + 1:].any()),
                     "n_fail": int(len(fails)), "fails_in_label_window": int((~ok[win]).sum()), "n_window": int(win.sum()),
                     "dense_ds": float(grid[ok].max()) if ok.any() else float("nan")})
    return pd.DataFrame(recs)


def load_c_frame(paths, mu=MU, enriched=None):
    """v5 datasets (concatenated) -> usable rows with derived columns; returns (frame, weights). `enriched`: set of paths that came from
    the enriched sampler (their rows get importance weights)."""
    from orbit_ml_cr3bp import policy_c_3d as pc
    frames, enr = [], []
    for k, pth in enumerate(paths):
        d = pd.read_csv(pth, low_memory=False)
        for c in ("rk4_censored", "leapfrog_censored", "euler_censored", "ftle_nan", "reference_divergent", "rk4s_censored"):
            if c in d.columns:
                d[c] = d[c].map({True: True, False: False, "True": True, "False": False}).fillna(False).astype(bool)
        meta = proto.read_metadata(pth) or {}
        if meta.get("label_pipeline_version") != 5:
            raise ValueError(f"{pth}: label_pipeline_version={meta.get('label_pipeline_version')}; Part C needs 5")
        d["src"] = k
        frames.append(d)
        enr.append(np.full(len(d), bool(enriched and pth in enriched)))
    D = pd.concat(frames, ignore_index=True)
    is_enr = np.concatenate(enr)
    D["row_id"] = np.arange(len(D))
    keep = pb.usable_rows(D).index.to_numpy()
    U = fs.add_derived(D.loc[keep].reset_index(drop=True), mu)
    w = pc.importance_weights(U["sampling_branch"].to_numpy(), is_enriched=is_enr[keep])
    return U, w


def _fit_for_config(cfg, tr, w, feats):
    """-> (kind, models dict) usable by policy_c_3d.predict_raw_c. kinds: 'trees' (HGB) or 'linear' (M0)."""
    from orbit_ml_cr3bp import policy_c_3d as pc
    from orbit_ml_cr3bp import compiled_model_3d as cm
    fam = cfg["family"]
    cands = ("rk4s",) if fam == "sundman" else (("rk4",) if cfg.get("rk4_only") else ("rk4", "leapfrog"))
    if cfg["cls"] == "linear":
        lin = {k: cm.fit_linear_single(tr, f"dt_{k}", f"{k}_success", cfg["q"], MU, max_rows=cfg.get("max_rows", 30000), seed=cfg.get("seed", 42),
                                       sample_weight=w) for k in cands}
        return "linear", {"features": ["x", "y", "z", "vx", "vy", "vz", "sim_time"], "candidates": cands, "linear": lin, "lf_gate": "none",
                          "structured": False, "quantile": cfg["q"]}
    if cfg["cls"] == "trees":
        size = {"reg_kw": {"max_iter": cfg["n_iter"], "max_depth": cfg.get("depth"), "learning_rate": cfg["lr"]},
                "clf_kw": {"max_iter": max(10, cfg["n_iter"] // 2), "max_depth": cfg.get("depth"), "learning_rate": cfg["lr"]}}
        models = pc.fit_policy_c(tr, feats, quantile=cfg["q"], seed=cfg.get("seed", 42), lf_gate=cfg.get("gate", "none") if "leapfrog" in cands else "none",
                                 sample_weight=w, structured=cfg.get("structured", False), candidates=cands, **size)
        return "trees", models
    raise ValueError(cfg["cls"])


def cmd_select(a):
    import joblib
    from orbit_ml_cr3bp import policy_c_3d as pc
    from orbit_ml_cr3bp import compiled_model_3d as cm
    out = make_run_dir(a.out_dir, "select", a)
    cfgs = json.load(open(a.configs))
    Utr, wtr = load_c_frame(a.train, a.mu, enriched=set(a.train_enriched))
    if a.max_train and len(Utr) > a.max_train:
        sub = np.random.RandomState(a.seed).choice(len(Utr), a.max_train, replace=False)
        Utr, wtr = Utr.iloc[sub].reset_index(drop=True), wtr[sub]
    Usel, _ = load_c_frame([a.select], a.mu)
    print(f"train {len(Utr)} usable rows, select {len(Usel)}", flush=True)
    cache_path = os.path.join(os.path.dirname(a.select), "select_ref_final.npy")
    if os.path.exists(cache_path) and len(np.load(cache_path)) == len(Usel):
        ref = np.load(cache_path)
    else:
        ref = pc.reference_final_positions(Usel, a.mu, a.n_jobs)
        np.save(cache_path, ref)
    cal_idx, ev_idx = pc.split_folds(len(Usel), a.seed)
    fitted, rows = {}, []
    fb_cache = {}
    for cfg in cfgs:
        t0 = time.time()
        fam = cfg["family"]
        feats = pc.FEATS_OSC if cfg.get("features", "osc") == "osc" else pb.FEATS_STATE
        key = json.dumps({k: cfg.get(k) for k in ("family", "cls", "n_iter", "depth", "lr", "structured", "q", "rk4_only", "features", "gate")}, sort_keys=True)
        if key not in fitted:
            fitted[key] = _fit_for_config(cfg, Utr, wtr, feats)
        kind, models = fitted[key]
        raw = pc.predict_raw_c(models, Usel)
        pred = pc.apply_gate(raw, p_star=cfg.get("p_star", 0.5), g_star=cfg.get("g_star", 0.0), lf_gate=models["lf_gate"])
        cands = tuple(models["candidates"])
        guard_g = math.inf if cfg.get("guard_g") is None else cfg["guard_g"]
        fallback, abstain = None, None
        if guard_g < math.inf or cfg.get("abstain_rule", "none") != "none":
            if "rtol" not in fb_cache:
                r0 = pb.certify_rtol_lazy(Usel.iloc[cal_idx].reset_index(drop=True), "RK45", tol_km=a.tol_km, mu=a.mu, n_jobs=a.n_jobs)
                fb_cache["rtol"] = r0["chosen_rtol"]
            fallback = (fb_cache["rtol"], "RK45")
            if cfg.get("abstain_rule", "none") != "none":
                abstain = pc.abstain_flags(Usel, cfg["abstain_rule"], cfg.get("abstain_logtau"))
        res = pc.evaluate_config(Usel, ref, pred, cal_idx, ev_idx, candidates=cands, tol_km=a.tol_km, rule=cfg.get("rule", "fixed_sequence_fast"),
                                 grid=pc.FIXED_SEQ_GRID if cfg.get("grid", "fine") == "fine" else pc.pb.LTT_SCALE_GRID, mu=a.mu, n_jobs=a.n_jobs,
                                 guard_g=guard_g, fallback=fallback, abstain=abstain)
        row = {"name": cfg["name"], **{k: v for k, v in cfg.items() if k != "name"}, "scale": res["scale"], "fit_plus_run_s": time.time() - t0}
        if res["scale"] is not None:
            row.update({k: res[k] for k in ("n_eval", "violations", "violation_rate", "viol_cp95", "median_c_int_met", "mean_log10_c_int_met")})
            row["cond_risk_cal"] = res["cert"]["conditional_risk"]
            row["cal_violation_rate"] = res["cert"].get("violation_rate")
            row["fallback_share"] = float(res["fallback"].mean())
            np.savez_compressed(os.path.join(out, f"orbits_{cfg['name']}.npz"), evals=res["evals"], met=res["met"], rows=res["rows"],
                                choice=res["choice"])
        rows.append(row)
        print(f"{cfg['name']:28s} scale {row['scale']}  viol {row.get('violation_rate', float('nan')):.4f}  median C_int {row.get('median_c_int_met', float('nan')):.0f}  "
              f"({row['fit_plus_run_s']:.0f}s)", flush=True)
        save_json({"rows": rows}, os.path.join(out, "select_rows.json"))
    pd.DataFrame(rows).to_csv(os.path.join(out, "select_rows.csv"), index=False)
    print("wrote", out)


def _config_features(cfg):
    from orbit_ml_cr3bp import policy_c_3d as pc
    return pc.FEATS_OSC if cfg.get("features", "osc") == "osc" else pb.FEATS_STATE


def cmd_freeze_policy(a):
    """Fit the final policy of one family on D_train (weighted), tune its gate on D_select, write the frozen spec."""
    import joblib
    from orbit_ml_cr3bp import policy_c_3d as pc
    out = make_run_dir(a.out_dir, "freeze_policy", a)
    cfg = json.load(open(a.config))
    Utr, wtr = load_c_frame(a.train, a.mu, enriched=set(a.train_enriched))
    Usel, _ = load_c_frame([a.select], a.mu)
    feats = _config_features(cfg)
    fam = cfg["family"]
    cands = ("rk4s",) if fam == "sundman" else (("rk4",) if cfg.get("rk4_only") else ("rk4", "leapfrog"))
    if cfg.get("cls") == "linear":
        from orbit_ml_cr3bp import compiled_model_3d as cm
        lin = {k: cm.fit_linear_single(Utr, f"dt_{k}", f"{k}_success", cfg["q"], a.mu, max_rows=cfg.get("max_rows", 30000), seed=cfg.get("seed", 42),
                                       sample_weight=wtr) for k in cands}
        models = {"features": ["x", "y", "z", "vx", "vy", "vz", "sim_time"], "candidates": cands, "linear": lin, "lf_gate": "none",
                  "structured": False, "quantile": cfg["q"]}
        spec = {"config": cfg, "gate": {"lf_gate": "none", "p_star": 0.5, "g_star": 0.0}, "features": models["features"], "candidates": list(cands),
                "n_train_rows": int(len(Utr)), "train_files": a.train}
        path = os.path.join(out, "policy_frozen.joblib")
        joblib.dump({"models": models, "spec": spec}, path)
        spec["model_file_sha256"] = proto.sha256_file(path)
        save_json(spec, os.path.join(out, "policy_frozen_spec.json"))
        print("wrote", out)
        return
    if cfg.get("cls") == "constant":      # the non-ML control: no model, no training data
        models = {"features": [], "candidates": cands, "constant": True, "lf_gate": "none", "structured": False, "quantile": None}
        spec = {"config": cfg, "gate": {"lf_gate": "none", "p_star": 0.5, "g_star": 0.0}, "features": [], "candidates": list(cands)}
        path = os.path.join(out, "policy_frozen.joblib")
        joblib.dump({"models": models, "spec": spec}, path)
        spec["model_file_sha256"] = proto.sha256_file(path)
        save_json(spec, os.path.join(out, "policy_frozen_spec.json"))
        print("wrote", out)
        return
    size = {"reg_kw": {"max_iter": cfg["n_iter"], "max_depth": cfg.get("depth"), "learning_rate": cfg["lr"]},
            "clf_kw": {"max_iter": max(10, cfg["n_iter"] // 2), "max_depth": cfg.get("depth"), "learning_rate": cfg["lr"]}}
    models = pc.fit_policy_c(Utr, feats, quantile=cfg["q"], seed=cfg.get("seed", 42), lf_gate=cfg.get("gate", "none") if "leapfrog" in cands else "none",
                             sample_weight=wtr, structured=cfg.get("structured", False), candidates=cands, **size)
    gate = {"lf_gate": models["lf_gate"], "p_star": cfg.get("p_star", 0.5), "g_star": cfg.get("g_star", 0.0)}
    if "leapfrog" in cands and models["lf_gate"] in ("clf", "censored"):
        thr, tab = pc.select_gate_threshold(pc.predict_raw_c(models, Usel), Usel, max_rate=cfg.get("gate_budget", 0.005), lf_gate=models["lf_gate"])
        tab.to_csv(os.path.join(out, "gate_table.csv"), index=False)
        if thr is None:
            raise RuntimeError("no gate threshold meets the budget on D_select")
        gate["p_star" if models["lf_gate"] == "clf" else "g_star"] = thr
    spec = {"config": cfg, "gate": gate, "features": feats, "candidates": list(cands), "n_train_rows": int(len(Utr)), "n_select_rows": int(len(Usel)),
            "train_files": a.train, "select_file": a.select,
            "sha256": {pth: proto.sha256_file(pth) for pth in list(a.train) + [a.select]}}
    path = os.path.join(out, "policy_frozen.joblib")
    joblib.dump({"models": models, "spec": spec}, path)
    spec["model_file_sha256"] = proto.sha256_file(path)
    save_json(spec, os.path.join(out, "policy_frozen_spec.json"))
    print(json.dumps(proto.json_safe({k: v for k, v in spec.items() if k not in ("sha256",)}), indent=1))
    print("wrote", out)


def deploy_and_evaluate(cfg, models, gate, cal_df, ev_df, out, a, scale_override=None, rtols_override=None, fallback_override=None):
    """Certify one end-to-end scale on cal_df (or reuse scale_override), recalibrate the adaptive comparators on the same rows, run the deployed
    policy and the comparators on ev_df, write the ledger and a summary. Returns (ledger, report)."""
    from orbit_ml_cr3bp import policy_c_3d as pc
    from orbit_ml_cr3bp.audit_3d import clopper_pearson
    cands = tuple(models["candidates"])
    fam = cfg["family"]
    raw_ev = pc.predict_raw_c(models, ev_df)
    pred_ev = pc.apply_gate(raw_ev, p_star=gate["p_star"], g_star=gate["g_star"], lf_gate=models["lf_gate"])
    ev_idx, ev_raw = pc.choose_unscaled(pred_ev, cands)
    guard_g = math.inf if cfg.get("guard_g") is None else float(cfg["guard_g"])
    grid = {"fine": pc.FIXED_SEQ_GRID, "global_ds": pc.GLOBAL_DS_GRID}.get(cfg.get("grid", "fine"), tuple(pb.LTT_SCALE_GRID))
    rule = cfg.get("rule", "fixed_sequence")
    fallback, abstain_ev, abstain_cal = fallback_override, None, None
    cert, rtols, rtol_rep = None, rtols_override or {}, {}
    if cal_df is not None and len(cal_df):
        raw_cal = pc.predict_raw_c(models, cal_df)
        pred_cal = pc.apply_gate(raw_cal, p_star=gate["p_star"], g_star=gate["g_star"], lf_gate=models["lf_gate"])
        ref_cal = pc.reference_final_positions(cal_df, a.mu, a.n_jobs)
        items = pb.reference_items(cal_df, a.mu)
        if guard_g < math.inf or cfg.get("abstain_rule", "none") != "none":
            r0 = pb.certify_rtol_lazy(cal_df, "RK45", tol_km=a.tol_km, mu=a.mu, equal_risk_target=None, n_jobs=a.n_jobs, items=items)
            fallback = (r0["chosen_rtol"], "RK45")
            print(f"  fallback: RK45 at the certified rtol {fallback[0]}", flush=True)
        if cfg.get("abstain_rule", "none") != "none":
            abstain_cal = pc.abstain_flags(cal_df, cfg["abstain_rule"], cfg.get("abstain_logtau"))
        cert = pc.certify_end_to_end(cal_df, pred_cal, ref_cal, tol_km=a.tol_km, grid=grid, rule=rule, mu=a.mu, n_jobs=a.n_jobs, candidates=cands,
                                     by_choice=bool(cfg.get("by_choice")), guard_g=guard_g, fallback=fallback, abstain=abstain_cal)
        print(f"  certified: scale {cert['scale']}; cal violations {cert.get('violations')} of {cert['n_policy_runs']}; "
              f"conditional {json_brief(cert['conditional_risk'])}", flush=True)
        if cert["scale"] is None:
            raise RuntimeError("no scale on the grid is certified")
        scale = cert["scale"]
        cal_rate = cert["violation_rate"]
        for m in pb.METHODS:
            r = pb.certify_rtol_lazy(cal_df, m, tol_km=a.tol_km, mu=a.mu, equal_risk_target=cal_rate, n_jobs=a.n_jobs, items=items)
            rtol_rep[m] = r
            rtols[(m, "cal")] = r["chosen_rtol"]
            rtols[(m, "eqr")] = r["equal_risk_rtol"]
            print(f"  {m}: certified rtol {r['chosen_rtol']}, equal-risk rtol {r['equal_risk_rtol']}", flush=True)
    else:
        scale = float(scale_override)
        cal_rate = None
    if cfg.get("abstain_rule", "none") != "none":
        abstain_ev = pc.abstain_flags(ev_df, cfg["abstain_rule"], cfg.get("abstain_logtau"))
    res = pc.evaluate_orbits_c(ev_df, ev_idx, ev_raw, np.full(len(ev_df), scale), rtols=rtols, do_spec=not a.no_spec, tol_km=a.tol_km, mu=a.mu,
                               n_jobs=a.n_jobs, candidates=cands, guard_g=guard_g, fallback=fallback, abstain=abstain_ev)
    raw = pb.candidate_raw_predictions(ev_df, pred_ev, candidates=cands) if set(cands) <= set(pc.CANDIDATES) else \
        pd.DataFrame({"row_id": ev_df["row_id"].to_numpy(), "raw_dt_rk4s": pred_ev["rk4s"]["dt"]})
    c_pred = float(cfg.get("overhead_rhs", 0.0))
    led = fs.build_ledger(ev_df, res.merge(raw, on="row_id"), c_pred, c_feat=0.0, tol_km=a.tol_km)
    if fam == "sundman":
        led["dstar_chosen"] = np.where(led["ml_integrator"] == "rk4s", led["dt_rk4s"], np.nan)
        led["cens_chosen"] = np.where(led["ml_integrator"] == "rk4s", led["rk4s_censored"].fillna(False), False)
        led["C_s_star"] = led["rk4s_evals"]
    led.to_csv(os.path.join(out, "ledger.csv"), index=False)
    ok = led["meets_tol"]
    sumry = {"n_eval": int(len(led)), "scale": scale, "risk_ledger": fs.risk_ledger(led, a.tol_km), "by_integrator": {}}
    for k in cands:
        g = led[led["ml_integrator"] == k]
        if len(g):
            nb = int(g["violation"].sum())
            sumry["by_integrator"][k] = {"n": int(len(g)), "share": float(len(g) / len(led)), "violations": nb, "rate": nb / len(g),
                                         "cp95": list(clopper_pearson(nb, len(g))), "infeasible_choice": int(g["dstar_chosen"].isna().sum())}
    sumry["fallback_share"] = float(led["ml_fallback"].mean()) if "ml_fallback" in led.columns else 0.0
    sumry["abstain_share"] = float((led["ml_integrator"] == "adaptive").mean())
    sumry["median_C_ml_int_met"] = float(led.loc[ok, "C_ml_int"].median())
    sumry["median_C_ml_tot_met"] = float(led.loc[ok, "C_ml_tot"].median())
    sumry["mean_log10_C_ml_tot_met"] = float(np.log10(led.loc[ok, "C_ml_tot"]).mean())
    for comp in fs.COMPARATORS:
        if f"{comp}_nfev" in led.columns and led[f"{comp}_nfev"].notna().any():
            m_ = ok & led[f"{comp}_nfev"].notna()
            w_ = led["meets_tol"] & led[f"{comp}_nfev"].notna() & (led["C_ml_tot"] < led[f"{comp}_nfev"])
            sumry[f"share_total_cheaper_{comp}"] = float(w_.sum() / led[f"{comp}_nfev"].notna().sum())
            sumry[f"share_integration_cheaper_{comp}"] = float((led.loc[m_, "C_ml_int"] < led.loc[m_, f"{comp}_nfev"]).mean())
    return led, {"certificate": cert, "scale": scale, "cal_policy_violation_rate": cal_rate, "rtol_calibration": rtol_rep, "summary": sumry,
                 "rtols": {f"{m}|{t}": v for (m, t), v in rtols.items()}, "fallback": fallback}


def cmd_apply_policy(a):
    import joblib
    from run_failure_study_3d import split_rows
    out = make_run_dir(a.out_dir, "apply_policy", a)
    t0 = time.time()
    fz = joblib.load(a.frozen)
    models, spec = fz["models"], fz["spec"]
    cfg, gate = dict(spec["config"]), spec["gate"]
    if models.get("constant"):
        cfg.update({"grid": "global_ds"})
    if a.overhead is not None:
        cfg["overhead_rhs"] = a.overhead
    U, _ = load_c_frame([a.data], a.mu)
    if a.no_cal:
        cal_df, ev_df = None, U
    else:
        _tr, cal, ev = split_rows(U, a.n_cal, 0, a.seed)
        cal_df, ev_df = U.iloc[cal].reset_index(drop=True), U.iloc[ev].reset_index(drop=True)
    print(f"usable {len(U)}; calibration {0 if cal_df is None else len(cal_df)}, evaluation {len(ev_df)}; family {cfg['family']}", flush=True)
    kw = {}
    if a.no_cal:
        prev = json.load(open(os.path.join(a.scale_from, "apply_policy_report.json")))
        kw = {"scale_override": prev["scale"], "rtols_override": {tuple(k.split("|")): v for k, v in prev["rtols"].items()},
              "fallback_override": tuple(prev["fallback"]) if prev.get("fallback") else None}
    led, rep = deploy_and_evaluate(cfg, models, gate, cal_df, ev_df, out, a, **kw)
    if cal_df is not None:
        cal_df.assign(split="cal")[["row_id"]].to_csv(os.path.join(out, "cal_row_ids.csv"), index=False)
    save_json({"args": vars(a), "spec": spec, "n_usable": int(len(U)), "n_cal": 0 if cal_df is None else int(len(cal_df)), **rep,
               "wall_seconds": time.time() - t0}, os.path.join(out, "apply_policy_report.json"))
    print(json.dumps(proto.json_safe(rep["summary"]), indent=1))
    print(f"wrote {out}  ({time.time() - t0:.0f}s)")


def cmd_c1_resid(a):
    """C1 inputs: |log10 raw prediction / label| of the Part C RK4 regressor and of the Part B-style state-only model refitted on the same D_train,
    on the evaluation orbits (every usable orbit that is not a calibration row) with a valid RK4 label."""
    import joblib
    from orbit_ml_cr3bp import policy_c_3d as pc
    out = make_run_dir(a.out_dir, "c1_resid", a)
    U, _ = load_c_frame([a.data], a.mu)
    if a.cal_ids and os.path.exists(a.cal_ids):
        cal = set(pd.read_csv(a.cal_ids)["row_id"])
        U = U[~U["row_id"].isin(cal)].reset_index(drop=True)
    U = U[U["rk4_success"] == 1].reset_index(drop=True)
    y = np.log10(U["dt_rk4"].to_numpy(dtype=float))
    res = {}
    for key, path in (("partC", a.phys), ("partB_refit", a.partb)):
        fz = joblib.load(path)
        raw = pc.predict_raw_c(fz["models"], U)["rk4"]["logdt"]
        res[f"abs_resid_{key}"] = np.abs(raw - y)
        res[f"over_{key}"] = raw > y
    df = pd.DataFrame({"row_id": U["row_id"].to_numpy(), **res})
    df.to_csv(os.path.join(out, "c1_resid.csv"), index=False)
    print({k: float(np.median(v)) for k, v in res.items() if k.startswith("abs")}, "n", len(df))
    print("wrote", out)


def cmd_mu_strata(a):
    """Track 4.1.1: the mu dial stratified by closest approach in Hill radii, from the existing per-mu policy ledgers (no new data)."""
    from orbit_ml_cr3bp import mu_dial_3d as md
    out = make_run_dir(a.out_dir, "mu_strata", a)
    frames = []
    for spec in a.runs:
        mu_s, path = spec.split("=", 1)
        mu = float(mu_s)
        f = os.path.join(path, "ledger.csv")
        led = read_ledger(f if os.path.exists(f) else f + ".gz")
        frames.append(md.stratify_by_hill(led, mu, comp=a.comp))
    tab = pd.concat(frames, ignore_index=True)
    flat = []
    for _, r in tab.iterrows():
        row = {"mu": r["mu"], "band": r["band"], "n": r["n"]}
        if "paradigm_gap_median" in r and isinstance(r["paradigm_gap_median"], (list, tuple)):
            row.update({"fixed_cheaper_share": r["fixed_cheaper_share"], "paradigm_gap": r["paradigm_gap_median"][0], "paradigm_lo": r["paradigm_gap_median"][1],
                        "paradigm_hi": r["paradigm_gap_median"][2], "ml_int_gap": r["ml_int_gap_median"][0], "ml_lo": r["ml_int_gap_median"][1],
                        "ml_hi": r["ml_int_gap_median"][2], "violation_rate": r["violation_rate"]})
        flat.append(row)
    flat = pd.DataFrame(flat)
    flat.to_csv(os.path.join(out, "mu_strata.csv"), index=False)
    pd.set_option("display.width", 220)
    print(flat.round(3).to_string())
    print("wrote", out)


def cmd_overhead(a):
    """Single-call cost of a frozen policy in RHS-equivalents (plan 2.1 protocol), median of --repeats. --pin writes the median into the frozen
    spec (config.overhead_rhs) so it is the value that is hashed at the seal. Run on an IDLE machine."""
    import joblib
    import platform
    import numba
    from orbit_ml_cr3bp import compiled_model_3d as cm
    fz = joblib.load(a.frozen)
    models, spec = fz["models"], fz["spec"]
    U, _ = load_c_frame([a.data], a.mu)
    U = U.sample(min(len(U), 2000), random_state=0)
    st, T = U[pb.STATE_COLS].to_numpy(dtype=float), U["sim_time"].to_numpy(dtype=float)
    if "linear" in models:
        k = models["candidates"][0]
        c = cm.measure_linear_single_cost(models["linear"][k], st, T, repeats=a.repeats)
        c["kind"] = "linear"
    elif models.get("constant"):
        c = {"rhs_equiv": 0.0, "kind": "constant"}
    else:
        pol = cm.CompiledHGBPolicy(models, p_star=spec["gate"]["p_star"], g_star=spec["gate"]["g_star"])
        c = cm.measure_single_call_cost(pol, st, T, repeats=a.repeats)
        c["kind"] = "trees"
        c["size"] = pol.size()
        c["batched_rhs_equiv"] = cm.batched_cost(pol, st, T) / c["rhs_ns_per_call"]
    c["machine"] = {"platform": platform.platform(), "python": platform.python_version(), "numba": numba.__version__,
                    "load_average_1min": os.getloadavg()[0], "cpu_count": os.cpu_count()}
    print(json.dumps(proto.json_safe(c), indent=1))
    if a.pin:
        spec["config"]["overhead_rhs"] = float(c["rhs_equiv"])
        spec["overhead_measurement"] = proto.json_safe(c)
        joblib.dump({"models": models, "spec": spec}, a.frozen)
        spec["model_file_sha256"] = proto.sha256_file(a.frozen)
        save_json(spec, os.path.join(os.path.dirname(a.frozen), "policy_frozen_spec.json"))
        print("pinned overhead_rhs =", spec["config"]["overhead_rhs"])


def cmd_doomed(a):
    """Track 1.3: run a frozen policy on the reference-divergent orbits of a v5 dataset and judge the OUTCOME (event type and time); separate ledger.
    Also the physical crash flag (osculating periapsis inside a body, reached before T) against the reference outcome."""
    import joblib
    from orbit_ml_cr3bp import policy_c_3d as pc
    from orbit_ml_cr3bp import doomed_3d as dd
    from orbit_ml_cr3bp.audit_3d import clopper_pearson
    from orbit_ml_cr3bp.sundman_3d import integrate_endpoint_rk4s, SUNDMAN_TAU_C
    from orbit_ml_cr3bp.config import collision_radii, du_km_for_mu
    out = make_run_dir(a.out_dir, "doomed", a)
    fz = joblib.load(a.frozen)
    models, spec = fz["models"], fz["spec"]
    cfg, gate = spec["config"], spec["gate"]
    scale = float(json.load(open(os.path.join(a.scale_from, "apply_policy_report.json")))["scale"])
    d = pd.read_csv(a.data, low_memory=False)
    for c in ("ftle_nan", "reference_divergent"):
        d[c] = d[c].map({True: True, False: False, "True": True, "False": False}).fillna(False).astype(bool)
    d["row_id"] = np.arange(len(d))
    D = d[d["reference_divergent"] & d["reference_outcome"].isin(["primary_collision", "secondary_collision", "ejection"])].reset_index(drop=True)
    if a.max_rows and len(D) > a.max_rows:
        D = D.sample(a.max_rows, random_state=0).reset_index(drop=True)
    cands = tuple(models["candidates"])
    raw = pc.predict_raw_c(models, D)
    pred = pc.apply_gate(raw, lf_gate=models["lf_gate"])
    idx, raw_dt = pc.choose_unscaled(pred, cands)
    abst = pc.abstain_flags(D, cfg["abstain_rule"], cfg.get("abstain_logtau")) if cfg.get("abstain_rule", "none") != "none" else np.zeros(len(D), bool)
    rc1, rc2 = collision_radii(a.mu)
    tol_du = a.tol_km / du_km_for_mu(a.mu)
    rows = []
    for i in range(len(D)):
        st = D[pb.STATE_COLS].to_numpy(dtype=float)[i]
        T = float(D["sim_time"].iloc[i])
        ref_code = dd.NAME_TO_CODE[D["reference_outcome"].iloc[i]]
        t_ref = float(D["t_end"].iloc[i])
        rec = {"row_id": int(D["row_id"].iloc[i]), "ref_outcome": D["reference_outcome"].iloc[i], "t_ref": t_ref, "abstained": bool(abst[i])}
        if abst[i]:
            rows.append(rec)
            continue
        step = float(raw_dt[i]) * scale
        if cands == ("rk4s",):
            ok, prev, curr, t_prev, t, reason, ns, nf = integrate_endpoint_rk4s(st, step, T, a.mu, 4_000_000, tau_c=SUNDMAN_TAU_C)
            if ok:
                run = {"event": 0, "t_last_good": T, "speed": float(np.linalg.norm(np.asarray(curr)[3:])), "dt": step, "finite": True, "steps": ns + nf}
            else:
                x = np.asarray(curr)
                r1 = np.sqrt((x[0] + a.mu) ** 2 + x[1] ** 2 + x[2] ** 2) / rc1
                r2 = np.sqrt((x[0] + a.mu - 1) ** 2 + x[1] ** 2 + x[2] ** 2) / rc2
                ev = 3 if reason == 2 else (1 if r1 < r2 else 2)
                run = {"event": ev if reason in (1, 2) else 0, "t_last_good": float(t), "speed": float(np.linalg.norm(x[3:])), "dt": float(step), "finite": reason != 3,
                       "steps": ns + nf}
        else:
            run = dd.run_until_event(cands[idx[i]], st, step, T, a.mu)
        ag = dd.agreement(run, ref_code, t_ref, tol_du)
        rec.update({"run_event": run["event"], "t_run": run["t_last_good"], "step": step, "steps": run["steps"], **ag})
        rows.append(rec)
    L = pd.DataFrame(rows)
    L.to_csv(os.path.join(out, "doomed_ledger.csv"), index=False)
    run = L[~L["abstained"]]
    rep = {"policy": cfg["name"], "scale": scale, "n_doomed": int(len(L)), "abstained_share": float(L["abstained"].mean()),
           "n_run": int(len(run)), "by_outcome": {}}
    for key, g in [("all", run)] + [(k, v) for k, v in run.groupby("ref_outcome")]:
        n = len(g)
        row = {"n": int(n)}
        for col in ("same_type", "strict", "relaxed"):
            k_ = int(g[col].sum())
            row[col] = {"k": k_, "share": k_ / max(n, 1), "cp95": list(clopper_pearson(k_, n))}
        row["not_detected"] = float((g["run_event"] == 0).mean()) if n else float("nan")
        rep["by_outcome"][key] = row
    # the free physical crash flag against the reference outcome, over usable + doomed orbits
    from orbit_ml_cr3bp.osculating_3d import add_osc_features
    A = add_osc_features(d[~d["ftle_nan"]].reset_index(drop=True), a.mu)
    f1, f2, anyf = dd.crash_flag_physical(A, A["sim_time"].to_numpy(), a.mu, margin=a.flag_margin)
    A["flag"] = anyf
    A["doomed"] = A["reference_divergent"]
    flag = {}
    for key, m in [("all", A["doomed"])] + [(o, A["reference_outcome"] == o) for o in ("primary_collision", "secondary_collision", "ejection")]:
        tp = int((A["flag"] & m).sum())
        flag[key] = {"n": int(m.sum()), "recall": tp / max(int(m.sum()), 1)}
    flag["precision_any_doomed"] = float((A["doomed"] & A["flag"]).sum() / max(int(A["flag"].sum()), 1))
    flag["flag_rate_usable"] = float(A.loc[~A["doomed"], "flag"].mean())
    rep["physical_crash_flag"] = flag
    save_json(rep, os.path.join(out, "doomed_report.json"))
    print(json.dumps(proto.json_safe(rep), indent=1))
    print("wrote", out)


def cmd_endpoint_audit(a):
    """Track 1.4: re-run the chosen fixed-step runs of a policy with the trajectory recorded and compare the maximum checkpoint error with the final one."""
    from orbit_ml_cr3bp import endpoint_audit_3d as ea
    from orbit_ml_cr3bp.audit_3d import clopper_pearson
    out = make_run_dir(a.out_dir, "endpoint_audit", a)
    led = read_ledger(os.path.join(a.apply_dir, "ledger.csv"))
    led["cell"] = fs.assign_cells(led)
    L = led[led["ml_integrator"].isin(["rk4", "leapfrog", "rk4s"]) & led["ml_ok"]].sample(min(a.n, int(led["ml_ok"].sum())), random_state=a.seed).reset_index(drop=True)
    res = ea.max_error_audit(L, L["ml_integrator"].tolist(), L["ml_dt"].tolist(), a.mu, a.tol_km)
    L = L.join(res.set_index("i"))
    L["max_over_tol"] = L["max_km"] / a.tol_km
    L["final_over_tol"] = L["final_km"] / a.tol_km
    L[["row_id", "ml_integrator", "cell", "regime", "h_band", "final_km", "max_km", "lucky"]].to_csv(os.path.join(out, "endpoint_audit.csv"), index=False)
    ok = L[~L["failed"].fillna(True).astype(bool)]
    k = int(ok["lucky"].sum())
    rep = {"n": int(len(ok)), "lucky_endpoint": {"k": k, "share": k / max(len(ok), 1), "cp95": list(clopper_pearson(k, len(ok)))},
           "max_over_tol_median": float(ok["max_over_tol"].median()), "final_over_tol_median": float(ok["final_over_tol"].median()),
           "share_max_above_tol": float((ok["max_km"] > a.tol_km).mean()), "share_final_above_tol": float((ok["final_km"] > a.tol_km).mean()), "by_cell": []}
    for c, g in ok.groupby("cell"):
        if len(g) >= a.min_cell:
            kk = int(g["lucky"].sum())
            rep["by_cell"].append({"cell": c, "n": int(len(g)), "lucky_share": kk / len(g), "cp95": list(clopper_pearson(kk, len(g))), "flag_both_labels": bool(kk / len(g) > 0.02)})
    rep["cells_flagged"] = [r["cell"] for r in rep["by_cell"] if r["flag_both_labels"]]
    save_json(rep, os.path.join(out, "endpoint_audit_report.json"))
    print(json.dumps(proto.json_safe({k_: v for k_, v in rep.items() if k_ != "by_cell"}), indent=1))
    print("wrote", out)


def json_brief(cond):
    return {k: f"{v['violations']}/{v['n']}={v['rate']:.3f}" for k, v in cond.items()}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("lever")
    sp.add_argument("--ledger", required=True)
    sp.add_argument("--out_dir", default="out_c")
    sp.set_defaults(fn=cmd_lever)

    sp = sub.add_parser("e3c_paired")
    sp.add_argument("--data", required=True)
    sp.add_argument("--policy_dir", required=True)
    sp.add_argument("--lookahead", required=True)
    sp.add_argument("--out_dir", default="out_c")
    sp.add_argument("--quantile", type=float, default=0.1)
    sp.add_argument("--tol_km", type=float, default=POSITION_TOL_KM)
    sp.add_argument("--n_jobs", type=int, default=8)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--fractions", type=float, nargs="+", default=[0.05])
    sp.add_argument("--scales", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    sp.add_argument("--paired_scales", type=float, nargs=2, action="append", default=[[0.5, 0.6], [0.5, 0.5], [0.6, 0.6]],
                    help="(scale of f = 0, scale of f > 0) pairs for the McNemar test")
    sp.add_argument("--risk_targets", type=float, nargs="+", default=[0.03, 0.035, 0.04])
    sp.add_argument("--osc_arm", action="store_true", help="Track 4.2: add the periapsis-feature arm (free lookahead)")
    sp.set_defaults(fn=cmd_e3c_paired)

    sp = sub.add_parser("osc_diag")
    sp.add_argument("--data", required=True)
    sp.add_argument("--out_dir", default="out_c")
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--seeds", type=int, default=3)
    sp.add_argument("--mu", type=float, default=MU)
    sp.add_argument("--n_cal", type=int, default=2000)
    sp.add_argument("--n_eval", type=int, default=3000)
    sp.add_argument("--quantile", type=float, default=pb.PINNED_QUANTILE)
    sp.add_argument("--go_overall", type=float, default=0.03, help="go if the median |log resid| falls by at least this (dex)")
    sp.add_argument("--go_close", type=float, default=0.05, help="... or a close-approach band improves by at least this (dex)")
    sp.set_defaults(fn=cmd_osc_diag)

    sp = sub.add_parser("sundman_pilot")
    sp.add_argument("--data", required=True)
    sp.add_argument("--out_dir", default="out_c")
    sp.add_argument("--rtol_ledger", default="part_b/results/explore/policy/ledger.csv", help="ledger whose certified cal / equal-risk rtols are reused")
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--mu", type=float, default=MU)
    sp.add_argument("--tol_km", type=float, default=POSITION_TOL_KM)
    sp.add_argument("--n_jobs", type=int, default=8)
    sp.add_argument("--n_sweep", type=int, default=200)
    sp.add_argument("--tau_c_list", type=float, nargs="*", default=[0.5, 2.0])
    sp.add_argument("--go_share", type=float, default=0.40)
    sp.add_argument("--go_rho_drop", type=float, default=0.30)
    sp.set_defaults(fn=cmd_sundman_pilot)

    sp = sub.add_parser("mu_strata")
    sp.add_argument("--runs", nargs="+", required=True, help="mu=policy_dir (a folder with ledger.csv), one per mu")
    sp.add_argument("--comp", default="RK45_spec")
    sp.add_argument("--out_dir", default="out_c")
    sp.set_defaults(fn=cmd_mu_strata)

    sp = sub.add_parser("endpoint_audit")
    sp.add_argument("--apply_dir", required=True)
    sp.add_argument("--n", type=int, default=1000)
    sp.add_argument("--min_cell", type=int, default=20)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--out_dir", default="out_c")
    sp.add_argument("--mu", type=float, default=MU)
    sp.add_argument("--tol_km", type=float, default=POSITION_TOL_KM)
    sp.set_defaults(fn=cmd_endpoint_audit)

    sp = sub.add_parser("doomed")
    sp.add_argument("--frozen", required=True)
    sp.add_argument("--data", required=True)
    sp.add_argument("--scale_from", required=True, help="the apply_policy run folder whose certified scale is used")
    sp.add_argument("--out_dir", default="out_c")
    sp.add_argument("--max_rows", type=int, default=0)
    sp.add_argument("--flag_margin", type=float, default=1.0)
    sp.add_argument("--mu", type=float, default=MU)
    sp.add_argument("--tol_km", type=float, default=POSITION_TOL_KM)
    sp.set_defaults(fn=cmd_doomed)

    sp = sub.add_parser("overhead")
    sp.add_argument("--frozen", required=True)
    sp.add_argument("--data", required=True, help="a v5 CSV whose states are used for the timing loop")
    sp.add_argument("--repeats", type=int, default=5)
    sp.add_argument("--pin", action="store_true")
    sp.add_argument("--mu", type=float, default=MU)
    sp.set_defaults(fn=cmd_overhead)

    sp = sub.add_parser("c1_resid")
    sp.add_argument("--data", required=True)
    sp.add_argument("--phys", required=True)
    sp.add_argument("--partb", required=True)
    sp.add_argument("--cal_ids", default=None)
    sp.add_argument("--out_dir", default="out_c")
    sp.add_argument("--mu", type=float, default=MU)
    sp.set_defaults(fn=cmd_c1_resid)

    sp = sub.add_parser("freeze_policy")
    sp.add_argument("--config", required=True)
    sp.add_argument("--train", nargs="+", required=True)
    sp.add_argument("--train_enriched", nargs="*", default=[])
    sp.add_argument("--select", required=True)
    sp.add_argument("--out_dir", default="out_c")
    sp.add_argument("--mu", type=float, default=MU)
    sp.set_defaults(fn=cmd_freeze_policy)

    sp = sub.add_parser("apply_policy")
    sp.add_argument("--frozen", required=True)
    sp.add_argument("--data", required=True)
    sp.add_argument("--out_dir", default="out_c")
    sp.add_argument("--n_cal", type=int, default=4000)
    sp.add_argument("--no_cal", action="store_true")
    sp.add_argument("--scale_from", default=None, help="with --no_cal: the apply_policy run folder whose scale, rtols and fallback are reused")
    sp.add_argument("--overhead", type=float, default=None, help="pinned model cost (RHS-equivalents per orbit, features included)")
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--mu", type=float, default=MU)
    sp.add_argument("--tol_km", type=float, default=POSITION_TOL_KM)
    sp.add_argument("--n_jobs", type=int, default=8)
    sp.add_argument("--no_spec", action="store_true")
    sp.set_defaults(fn=cmd_apply_policy)

    sp = sub.add_parser("select")
    sp.add_argument("--train", nargs="+", required=True, help="v5 CSVs (natural, enriched)")
    sp.add_argument("--train_enriched", nargs="*", default=[], help="the subset of --train that came from the enriched sampler")
    sp.add_argument("--select", required=True)
    sp.add_argument("--configs", required=True, help="JSON list of configs (name, family, cls, n_iter, depth, lr, structured, q, ...)")
    sp.add_argument("--out_dir", default="out_c")
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--mu", type=float, default=MU)
    sp.add_argument("--tol_km", type=float, default=POSITION_TOL_KM)
    sp.add_argument("--n_jobs", type=int, default=8)
    sp.add_argument("--max_train", type=int, default=0)
    sp.set_defaults(fn=cmd_select)

    sp = sub.add_parser("compile_study")
    sp.add_argument("--data", required=True)
    sp.add_argument("--out_dir", default="out_c")
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--mu", type=float, default=MU)
    sp.add_argument("--n_cal", type=int, default=2000)
    sp.add_argument("--n_eval", type=int, default=3000)
    sp.add_argument("--quantile", type=float, default=pb.PINNED_QUANTILE)
    sp.add_argument("--p_star", type=float, default=0.995)
    sp.add_argument("--g_star", type=float, default=0.3)
    sp.add_argument("--lin_rows", type=int, default=0, help="subsample the linear fit to this many rows (0 = all)")
    sp.add_argument("--time", action="store_true", help="measure the single-call cost (run on a quiet machine)")
    sp.add_argument("--repeats", type=int, default=5)
    sp.add_argument("--gate_rhs", type=float, default=10.0)
    sp.set_defaults(fn=cmd_compile_study)

    sp = sub.add_parser("policy_c")
    sp.add_argument("--data", required=True)
    sp.add_argument("--out_dir", default="out_c")
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--mu", type=float, default=MU)
    sp.add_argument("--tol_km", type=float, default=POSITION_TOL_KM)
    sp.add_argument("--n_jobs", type=int, default=8)
    sp.add_argument("--n_cal", type=int, default=2000)
    sp.add_argument("--n_eval", type=int, default=3000)
    sp.add_argument("--n_sel", type=int, default=2000, help="rows carved out of the training slice to tune the leapfrog gate")
    sp.add_argument("--quantile", type=float, default=pb.PINNED_QUANTILE)
    sp.add_argument("--features", choices=["state", "osc"], default="state")
    sp.add_argument("--lf_gate", choices=["clf", "censored", "none"], default="clf")
    sp.add_argument("--p_star", type=float, default=0.5)
    sp.add_argument("--g_star", type=float, default=0.0)
    sp.add_argument("--tune_gate", action="store_true", help="choose p* / g* on the selection slice (infeasible-choice budget)")
    sp.add_argument("--gate_budget", type=float, default=0.005)
    sp.add_argument("--rule", choices=["fixed_sequence", "bonferroni"], default="fixed_sequence")
    sp.add_argument("--grid", choices=["fine", "partb"], default="fine")
    sp.add_argument("--rk4_only", action="store_true", help="leapfrog is never chosen")
    sp.add_argument("--by_choice", action="store_true", help="one scale per selected integrator (Mondrian by selection)")
    sp.add_argument("--guard_g", type=float, default=None, help="step-over guard: abort the fixed-step run when dt / tau > g and fall back to RK45 (certified rtol)")
    sp.add_argument("--abstain_rule", choices=["none", "physics"], default="none")
    sp.add_argument("--abstain_logtau", type=float, default=-1.0, help="physics rule: abstain when reach_2 = 1 and log10 tau_hat is below this")
    sp.add_argument("--no_spec", action="store_true")
    sp.add_argument("--c_pred_fixed", type=float, default=0.0, help="model inference cost charged per orbit (RHS-equivalents); 0 = integration only")
    sp.set_defaults(fn=cmd_policy_c)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
