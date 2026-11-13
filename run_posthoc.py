"""
Post hoc analyses of the sealed Studies A-D (posthoc/README.md). Every command writes a run folder with a manifest under posthoc/results/ and
never touches a sealed file. Numbers produced here are POST HOC and are reported next to the sealed numbers, never in their place.

  aggregates     P1-P3: share, aggregate cost ratio and mean log ratio for every sealed policy ledger and comparator, by horizon band, Hill
                 regime and sampling family, under the declared mixtures, and with one overhead charged on every Table 3 column
  symmetric      P7 + the delta = 0.05 part of P8: RK45 / DOP853 certified by the policy's rule (fixed sequence, full delta) on the fine rtol
                 grid, on the sealed calibration rows of a Table 3 set, run on its evaluation rows; the policy and comparators at delta = 0.05
  three_quarters P4: the frozen global fictitious step on the Study D confirmation set with the D1 protocol, and the share of the global-step gap
                 that the best rule recovers, both gaps on that set
  fig4_check     P5: closest lunar approach of Hill-regime-I orbits of the Study B confirmation set (the empty Fig 4 cell)
  atol           P8: equal-risk RK45 with atol = r * rtol for r in {1e-1, 1e-3 (sealed), 1e-6} and a position-only variant, on the Study D
                 replication set; the setting cheapest on the CALIBRATION rows is the 'best atol' comparator
  d4b_fine       P7 for D4b: RK45 (compiled DP5(4), SciPy controller) and DOP853 certified per stratum by fixed sequence at delta / G on the coarse
                 (sealed, as a reproduction check) and on the fine grid; the stratified policy's D_replication ledger against each
  compiled       P7 for the D2 comparators: compiled DP5(4)-PI, DP5(4)-SciPy and DOP853 certified by fixed sequence on the fine grid on the Study D
                 confirmation set's calibration rows; RHS counts of policy and comparators on its evaluation rows (no timing)
  seeds          P6: the frozen Study C Sundman configuration refitted with model seeds 1-5 (and 42, which must reproduce the frozen model),
                 each certified and evaluated on the Study D confirmation set with the D1 protocol
Defense-plan analyses of 2026-11-03 (posthoc/README.md P10-P23; P20-P23 are exploratory comparator runs):
  risk P10, waterfall P11, overhead P12, concentration P13, timescale P14, mixture_curve P15, multiplicity P16, bca P17, refs P18, ood P19,
  sundman_adaptive P20, learned_rtol P21, multistep P22, rk8s P23
Study E pilots of 2026-11-13 (posthoc/README.md P24-P25; exploratory, Study D data only):
  rk8s_vs_sundman P24 (oracle RK8 Sundman step vs oracle-tuned Sundman-time RK45 / DOP853), rk8s_learned P25 (a learned RK8 Sundman step)
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from orbit_ml_cr3bp import posthoc_metrics as pm  # noqa: E402
from orbit_ml_cr3bp.config import MU  # noqa: E402
from run_part_d_3d import DATA, save_json, make_run_dir as _make_run_dir  # noqa: E402

OUT = "posthoc/results"
COMPARATORS = ("RK45_eqr", "RK45_cal", "DOP853_eqr", "DOP853_cal")
META_COLS = ["row_id", "sim_time", "orbit_label", "regime", "h_band", "r2_band", "min_r2"]
OVERHEAD_C, OVERHEAD_D = 15.036794812388587, 16.088492841244694      # pinned Study C value; Study D re-measurement (out_d/overheads)


def _one(pattern):
    f = sorted(glob.glob(os.path.join(ROOT, pattern)))
    if len(f) != 1:
        raise FileNotFoundError(f"{pattern}: {len(f)} matches")
    return f[0]


# name -> (ledger glob, dataset whose usable rows the ledger's row_id refers to (None: the ledger carries its metadata), Table 3 column)
LEDGERS = {
    "C_confirm_sund": ("out_c/confirm/sund/apply_policy_*/ledger.csv", None, True),
    "D_replication_sund": ("out_d/confirm/d3/sund/apply_policy_*/ledger.csv", None, True),
    "C_lock_sund": ("out_c/lock/sund/apply_policy_*/ledger.csv", None, True),
    "D_lock_sund": ("out_d/confirm/lock/lock_*/ledger_sund.csv.gz", "lock", True),
    "D_confirm_D1_sund": ("out_d/confirm/d1/d1_*/ledger_sund.csv.gz", "confirm", False),
    "D_confirm_D1_rule_THR": ("out_d/confirm/d1/d1_*/ledger_rule_THR.csv.gz", "confirm", False),
    "D_replication_D4_strat": ("out_d/confirm/d4/d4_*/ledger_strat.csv.gz", "replication", False),
    "D_replication_D4_onescale": ("out_d/confirm/d4/d4_*/ledger_marginal.csv.gz", "replication", False),
}


def load_ledger(name):
    """A sealed ledger with the metadata columns (joined from its dataset by row_id where the ledger does not carry them)."""
    pattern, dset, _t3 = LEDGERS[name]
    path = _one(pattern)
    led = pd.read_csv(path, low_memory=False)
    if dset is not None:
        import run_part_c_3d as rc
        U, _w = rc.load_c_frame([os.path.join(ROOT, DATA[dset])], MU)
        meta = U[[c for c in META_COLS if c in U.columns]]
        both = [c for c in meta.columns if c in led.columns and c != "row_id"]
        n0 = len(led)
        led = led.merge(meta, on="row_id", how="left", suffixes=("", "_meta"), indicator=True)
        if len(led) != n0 or (led["_merge"] != "both").any():
            raise ValueError(f"{name}: row_id join to {dset} failed")
        led = led.drop(columns="_merge")
        for c in both:                       # the ledger's own copy must agree with the dataset's
            if not (led[c].astype(str) == led[f"{c}_meta"].astype(str)).all():
                raise ValueError(f"{name}: column {c} disagrees with the dataset after the row_id join")
            led = led.drop(columns=f"{c}_meta")
    if "C_ml_int" not in led.columns:
        raise ValueError(f"{name}: no C_ml_int column")
    return led.copy(), os.path.relpath(path, ROOT)


def _flat(r, **keys):
    row = dict(keys)
    for k, v in r.items():
        if isinstance(v, list):
            row[f"{k}_lo"], row[f"{k}_hi"] = v
        else:
            row[k] = v
    return row


def cmd_aggregates(a):
    out = _make_run_dir(os.path.join(ROOT, a.out_dir), "aggregates", a)
    t0 = time.time()
    rows, by_rows, mix_rows, oh_rows, sources = [], [], [], [], {}
    for name in LEDGERS:
        led, src = load_ledger(name)
        sources[name] = src
        comps = [c for c in COMPARATORS if f"{c}_nfev" in led.columns]
        for comp in comps:
            rows.append(_flat(pm.stats_frame(led, comp, n_boot=a.n_boot), ledger=name, comparator=comp))
            for by in ("h_band", "regime", "orbit_label"):
                for r in pm.breakdown(led, comp, by, n_boot=a.n_boot):
                    g = r.pop(by)
                    by_rows.append(_flat(r, ledger=name, comparator=comp, by=by, group=g))
            for mix, target in pm.MIXTURES.items():
                if target is None:
                    continue
                w = pm.mixture_weights(led["orbit_label"].to_numpy(), target)
                r = pm.stats_frame(led, comp, weights=w, strata=led["orbit_label"].to_numpy(), n_boot=a.n_boot)
                mix_rows.append(_flat(r, ledger=name, comparator=comp, mixture=mix))
            if LEDGERS[name][2]:
                for h, tag in ((OVERHEAD_C, "15.04"), (OVERHEAD_D, "16.09")):
                    tmp = led.assign(C_alt=led["C_ml_int"] + h)
                    r = pm.stats_frame(tmp, comp, policy_col="C_alt", n_boot=a.n_boot)
                    oh_rows.append(_flat(r, ledger=name, comparator=comp, overhead=tag))
        print(f"{name}: {len(led)} orbits ({time.time() - t0:.0f}s)", flush=True)
    tabs = {"overall": rows, "by_group": by_rows, "mixtures": mix_rows, "overhead": oh_rows}
    for k, v in tabs.items():
        pd.DataFrame(v).to_csv(os.path.join(out, f"aggregates_{k}.csv"), index=False)
    save_json({"label": "POST HOC summaries of sealed ledgers (posthoc/README.md P1-P3)", "sources": sources, "n_boot": a.n_boot,
               "wall_seconds": time.time() - t0}, os.path.join(out, "aggregates_report.json"))
    ov = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(ov[["ledger", "comparator", "n", "n_no_comp", "share", "aggregate_ratio", "aggregate_ratio_ci95_lo", "aggregate_ratio_ci95_hi",
              "mean_log_ratio", "median_policy_all", "median_policy_met"]].round(4).to_string())
    print("wrote", out)


# P7 sets: dataset, sealed apply_policy run folder (ledger, report, calibration ids), reference cache
SYM_SETS = {
    "c_confirm": ("data_c/confirm/cr3bp_3d_training_data_accuracy_monotone_seed43.csv", "out_c/confirm/sund/apply_policy_*", "posthoc/cache/refs_c_confirm.npz"),
    "d_replication": (DATA["replication"], "out_d/confirm/d3/sund/apply_policy_*", "out_d/refs/replication.npz"),
}
METHODS = ("RK45", "DOP853")


def _sealed_set(name):
    """Usable rows, the sealed calibration / evaluation split (checked against the sealed run), reference positions, sealed ledger and report."""
    import json
    import joblib
    import run_part_c_3d as rc
    from orbit_ml_cr3bp import deploy_d as dd
    from run_part_d_3d import standard_split
    data, run_glob, cache = SYM_SETS[name]
    run = _one(run_glob)
    U, _w = rc.load_c_frame([os.path.join(ROOT, data)], MU)
    cal, ev = standard_split(U)
    sealed_cal = set(pd.read_csv(os.path.join(run, "cal_row_ids.csv"))["row_id"])
    if set(U.iloc[cal]["row_id"]) != sealed_cal:
        raise ValueError(f"{name}: the standard split does not reproduce the sealed calibration rows")
    led = pd.read_csv(os.path.join(run, "ledger.csv"), low_memory=False)
    if set(led["row_id"]) != set(U.iloc[ev]["row_id"]):
        raise ValueError(f"{name}: the evaluation rows differ from the sealed ledger's")
    rep = json.load(open(os.path.join(run, "apply_policy_report.json")))
    ref = dd.reference_positions(U, MU, cache=os.path.join(ROOT, cache), n_jobs=8)
    fz = joblib.load(os.path.join(ROOT, rep["args"]["frozen"]))
    return U, cal, ev, ref, led, rep, fz, os.path.relpath(run, ROOT)


def cmd_symmetric(a):
    from multiprocessing import Pool
    from orbit_ml_cr3bp import deploy_d as dd, symmetric_cert as sc
    from run_part_d_3d import raw_log_ds
    out = _make_run_dir(os.path.join(ROOT, a.out_dir), f"symmetric_{a.set}", a)
    t0 = time.time()
    U, cal, ev, ref, led, rep, fz, run = _sealed_set(a.set)
    S, T = dd.states_T(U)
    Sc, Tc, rc_ = S[cal], T[cal], ref[cal]
    Se, Te, re_ = S[ev], T[ev], ref[ev]
    ev_ids = U.iloc[ev]["row_id"].to_numpy()
    led = led.set_index("row_id").loc[ev_ids].reset_index()
    oh = float(rep["spec"]["config"]["overhead_rhs"])
    scale = float(rep["scale"])
    # the policy as sealed, re-run with the Part D kernel: must reproduce the sealed ledger orbit for orbit
    runner = dd.default_runner(MU)
    raw = 10.0 ** raw_log_ds(fz["models"], U)
    r0 = dd.run_rows(runner, Se, Te, raw[ev] * scale, re_)
    same = (r0["evals"] == led["ml_evals"].to_numpy()) & (r0["met"] == led["meets_tol"].astype(bool).to_numpy())
    report = {"label": "POST HOC (posthoc/README.md P7, P8 delta = 0.05)", "set": a.set, "sealed_run": run, "n_cal": int(len(cal)), "n_eval": int(len(ev)),
              "overhead_rhs": oh, "sealed_scale": scale, "policy_rerun_matches_sealed": int(same.sum()), "grid": {"min": min(sc.RTOL_GRID_FINE), "max": max(sc.RTOL_GRID_FINE),
              "n": len(sc.RTOL_GRID_FINE), "step_dex": 0.125}, "sealed_rtols": rep["rtols"]}
    if not same.all():
        save_json(report, os.path.join(out, "symmetric_report.json"))
        raise SystemExit(f"policy re-run differs from the sealed ledger on {int((~same).sum())} orbits; stopping")
    print(f"policy re-run reproduces the sealed ledger on all {len(ev)} orbits ({time.time() - t0:.0f}s)", flush=True)
    # delta = 0.05 for the policy, from the sealed calibration chain (a prefix of it)
    chain_p = rep["certificate"]["chain"]
    scale05 = sc.chosen_at(chain_p, 0.05, key="scale")
    viol05 = {c["scale"]: c["violations"] for c in chain_p}[scale05]
    report.update({"scale_delta05": scale05, "cal_violation_rate_delta05": viol05 / len(cal), "cal_violation_rate_sealed": rep["cal_policy_violation_rate"]})
    cols = {"row_id": ev_ids}
    if scale05 != scale:
        r5 = dd.run_rows(runner, Se, Te, raw[ev] * scale05, re_)
        cols.update({"pol05_met": r5["met"], "pol05_tot": r5["evals"] + oh})
    else:
        cols.update({"pol05_met": r0["met"], "pol05_tot": r0["evals"] + oh})
    print(f"policy scale at delta 0.05: {scale05} (sealed {scale} at 0.1)", flush=True)
    report["methods"] = {}
    with Pool(a.n_jobs) as pool:
        for m in METHODS:
            cache = {}

            def viol(rt, m=m, cache=cache):
                if rt not in cache:
                    r = sc.scipy_rows(pool, Sc, Tc, rc_, rt, m, n_jobs=a.n_jobs)
                    cache[rt] = (int((~r["met"]).sum()), float(np.nanmean(r["nfev"])), int(r["capped"].sum()))
                    print(f"  {m} rtol {rt:.3e}: {cache[rt][0]} violations, mean nfev {cache[rt][1]:.0f}, capped {cache[rt][2]} ({time.time() - t0:.0f}s)", flush=True)
                return cache[rt][0], len(cal)
            cert = sc.certify_rtol_fixed_sequence(viol, sc.RTOL_GRID_FINE)
            rt01 = cert["chosen_rtol"]
            rt05 = sc.chosen_at(cert["chain"], 0.05)
            eqr05, eqr_chain = (rep["rtols"][f"{m}|eqr"], None) if scale05 == scale else sc.equal_risk_rtol(viol, viol05 / len(cal))
            mrep = {"certified_rtol_delta01": rt01, "certified_rtol_delta05": rt05, "equal_risk_rtol_delta05": eqr05, "chain": cert["chain"],
                    "eqr05_chain": eqr_chain, "cal_mean_nfev": {f"{k:.6e}": v[1] for k, v in cache.items()}, "cal_capped": {f"{k:.6e}": v[2] for k, v in cache.items()},
                    "sealed_certified_rtol": rep["rtols"][f"{m}|cal"], "sealed_equal_risk_rtol": rep["rtols"][f"{m}|eqr"]}
            report["methods"][m] = mrep
            print(f"{m}: certified rtol (fine, fixed sequence) {rt01} [sealed {rep['rtols'][m + '|cal']}]; delta 0.05: {rt05}; equal-risk at delta 0.05 {eqr05}", flush=True)
            runs = {}
            for tag, rt in (("calsym", rt01), ("calsym05", rt05), ("eqr05", eqr05)):
                if rt is None:
                    continue
                key = float(rt)
                if key not in runs:
                    runs[key] = sc.scipy_rows(pool, Se, Te, re_, key, m, n_jobs=a.n_jobs)
                r = runs[key]
                cols.update({f"{m}_{tag}_rtol": key, f"{m}_{tag}_nfev": r["nfev"], f"{m}_{tag}_ok": r["met"], f"{m}_{tag}_capped": r["capped"]})
            save_json(report, os.path.join(out, "symmetric_report.json"))
    E = pd.DataFrame(cols)
    E.to_csv(os.path.join(out, "eval_columns.csv.gz"), index=False)
    # summaries: sealed and symmetric side by side
    L = led.merge(E, on="row_id")
    rows = []
    for m in METHODS:
        for label, pol, met, comp in ((f"{m} certified, sealed (Bonferroni, coarse grid, delta 0.1)", "C_ml_tot", "meets_tol", f"{m}_cal"),
                                      (f"{m} certified, symmetric (fixed sequence, fine grid, delta 0.1)", "C_ml_tot", "meets_tol", f"{m}_calsym"),
                                      (f"{m} equal-risk, sealed (delta 0.1)", "C_ml_tot", "meets_tol", f"{m}_eqr"),
                                      (f"{m} certified, symmetric, delta 0.05 (policy at delta 0.05)", "pol05_tot", "pol05_met", f"{m}_calsym05"),
                                      (f"{m} equal-risk, delta 0.05 (policy at delta 0.05)", "pol05_tot", "pol05_met", f"{m}_eqr05")):
            if f"{comp}_nfev" not in L.columns:
                continue
            r = pm.cost_stats(L[pol].to_numpy(float), L[f"{comp}_nfev"].to_numpy(float), L[met].astype(bool).to_numpy(), n_boot=a.n_boot,
                              capped=L[f"{comp}_capped"].fillna(False).astype(bool).to_numpy() if f"{comp}_capped" in L.columns else None)
            ok_col = f"{comp}_ok"
            r["comparator_violation_rate"] = float((~L[ok_col].astype(bool)).mean()) if ok_col in L.columns else None
            r["rtol"] = float(L[f"{comp}_rtol"].iloc[0]) if f"{comp}_rtol" in L.columns else None
            r["policy_violation_rate"] = float((~L[met].astype(bool)).mean())
            rows.append(_flat(r, set=a.set, comparison=label))
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(out, "symmetric_summary.csv"), index=False)
    report["wall_seconds"] = time.time() - t0
    save_json(report, os.path.join(out, "symmetric_report.json"))
    pd.set_option("display.width", 250)
    print(tab[["comparison", "rtol", "comparator_violation_rate", "policy_violation_rate", "share", "aggregate_ratio", "mean_log_ratio"]].round(4).to_string())
    print("wrote", out)


def cmd_three_quarters(a):
    import joblib
    from orbit_ml_cr3bp import deploy_d as dd, policy_c_3d as pc, confirm_d_3d as cd
    from run_part_d_3d import load_set, standard_split
    out = _make_run_dir(os.path.join(ROOT, a.out_dir), "three_quarters", a)
    fz = joblib.load(os.path.join(ROOT, "part_c/frozen/global/policy_frozen.joblib"))
    if not fz["models"].get("constant"):
        raise ValueError("part_c/frozen/global is not the constant (global step) policy")
    runner = dd.default_runner(MU)
    rep = {"label": "POST HOC (posthoc/README.md P4)", "grid": "policy_c_3d.GLOBAL_DS_GRID", "overhead_rhs": 0.0}
    leds = {}
    for name in ("replication", "confirm"):
        U, ref = load_set(name)
        S, T = dd.states_T(U)
        cal, ev = standard_split(U)
        raw = np.ones(len(U))                           # raw step 1: the certified scale IS the global fictitious step
        c = dd.certify_scale(runner, S[cal], T[cal], raw[cal], ref[cal], grid=pc.GLOBAL_DS_GRID)
        ds = c["chosen_scale"]
        led = dd.score(U.iloc[ev], dd.run_rows(runner, S[ev], T[ev], raw[ev] * ds, ref[ev]), 0.0)
        leds[name] = led
        rep[name] = {"global_ds": ds, "cal_violations": c["violations_by_scale"][ds], "violation_rate": float(led["violation"].mean())}
        print(name, rep[name], flush=True)
    # check: on the replication set this must reproduce the sealed D3 global arm
    sealed = pd.read_csv(_one("out_d/confirm/d3/global/apply_policy_*/ledger.csv"), usecols=["row_id", "ml_evals", "meets_tol"]).set_index("row_id")
    g = leds["replication"].set_index("row_id").loc[sealed.index]
    rep["replication_reproduces_sealed_D3_global"] = bool((g["ml_evals"].to_numpy() == sealed["ml_evals"].to_numpy()).all()
                                                          and (g["meets_tol"].to_numpy() == sealed["meets_tol"].astype(bool).to_numpy()).all())
    if not rep["replication_reproduces_sealed_D3_global"]:
        save_json(rep, os.path.join(out, "three_quarters_report.json"))
        raise SystemExit("the global arm does not reproduce the sealed D3 global ledger; stopping")
    d1 = os.path.dirname(_one("out_d/confirm/d1/d1_*/ledger_sund.csv.gz"))
    lm = pd.read_csv(os.path.join(d1, "ledger_sund.csv.gz"))
    lr = pd.read_csv(os.path.join(d1, "ledger_rule_THR.csv.gz"))
    lg = leds["confirm"]
    lg.to_csv(os.path.join(out, "ledger_global_confirm.csv.gz"), index=False)
    gap_rule = cd.d1_ml_vs_rule(lm, lr, n_boot=a.n_boot)
    gap_glob = cd.d1_ml_vs_rule(lm, lg, n_boot=a.n_boot)
    rec = pm.recovered_share(lm, lr, lg, n_boot=a.n_boot)
    rep.update({"gap_rule_THR_dex": gap_rule["mean_log10_saving"], "gap_rule_ci95": gap_rule["ci95"], "gap_global_dex": gap_glob["mean_log10_saving"],
                "gap_global_ci95": gap_glob["ci95"], "n_paired_rule": gap_rule["n_paired"], "n_paired_global": gap_glob["n_paired"], **rec,
                "sealed_sentence": "rule gap 0.0715 dex (confirmation set) over global gap 0.274 dex (replication set) = 0.74"})
    save_json(rep, os.path.join(out, "three_quarters_report.json"))
    print({k: v for k, v in rep.items() if k.startswith(("gap", "recovered", "n_paired"))})
    print("wrote", out)


def cmd_fig4_check(a):
    """Regime I is C > C_L1: the zero-velocity surface separates the Earth realm from a lunar bubble of radius about 0.15-0.17 DU (the Moon-L1 and
    Moon-L2 distances). Counts every regime-I orbit of the Study B confirmation set (reference-divergent ones included) by closest lunar approach,
    with the realm read from the initial distance to the Moon."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from orbit_ml_cr3bp.diagnostics_3d import hill_regime
    out = _make_run_dir(os.path.join(ROOT, a.out_dir), "fig4_check", a)
    d = pd.read_csv(os.path.join(ROOT, "data_b/cr3bp_3d_training_data_accuracy_monotone_seed22.csv"), low_memory=False)
    d["row_id"] = np.arange(len(d))
    d["regime"] = hill_regime(d["jacobi_3d"].to_numpy(dtype=float), MU)
    I = d[d["regime"] == "I"].copy()
    I["band"] = pd.cut(I["min_r2"], [0, 0.05, 0.17, 0.8, np.inf], right=False, labels=["<0.05", "0.05-0.17", "0.17-0.8", ">0.8"])
    I["realm"] = np.where(I["r2_3d"] < 0.17, "lunar (start within 0.17 DU of the Moon)", "earth (start beyond 0.17 DU)")
    by_out = pd.crosstab(I["reference_outcome"], I["band"])
    by_realm = pd.crosstab([I["realm"], I["orbit_label"]], I["band"])
    usable = I[I["reference_outcome"] == "ok"]
    led = pd.read_csv(os.path.join(ROOT, "part_b/results/confirm/policy/ledger.csv.gz"), usecols=["row_id", "regime", "r2_band"])
    in_cell = usable[usable["band"] == "0.05-0.17"]
    rep = {"label": "POST HOC (posthoc/README.md P5)", "n_regime_I": int(len(I)), "n_regime_I_usable": int(len(usable)),
           "n_usable_in_0.05-0.17": int(len(in_cell)), "rows_in_cell": in_cell[["row_id", "orbit_label", "min_r2", "r2_3d"]].to_dict("records"),
           "cell_rows_in_fig4_ledger": int(led["row_id"].isin(in_cell["row_id"]).sum()),
           "fig4_ledger_regime_I_0.05-0.17": int(((led["regime"] == "I") & (led["r2_band"] == "0.05-0.17")).sum()),
           "usable_min_r2_by_realm": {k: {"n": int(len(x)), "min": float(x["min_r2"].min()), "max": float(x["min_r2"].max()),
                                          "n_0.05-0.17": int((x["band"] == "0.05-0.17").sum())} for k, x in usable.groupby("realm")}}
    by_out.to_csv(os.path.join(out, "regime_I_by_outcome.csv"))
    by_realm.to_csv(os.path.join(out, "regime_I_by_realm.csv"))
    save_json(rep, os.path.join(out, "fig4_check_report.json"))
    fig, ax = plt.subplots(figsize=(6, 3.2))
    bins = np.logspace(np.log10(max(usable["min_r2"].min(), 1e-3)), np.log10(usable["min_r2"].max()), 60)
    for k, x in usable.groupby("realm"):
        ax.hist(x["min_r2"], bins=bins, histtype="step", lw=1.5, label=f"{k} (n = {len(x)})")
    for v in (0.05, 0.17):
        ax.axvline(v, ls="--", c="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("closest lunar approach along the reference (DU)")
    ax.set_ylabel("usable regime-I orbits")
    ax.legend(fontsize=7)
    ax.set_title("Study B confirmation set, Hill regime I (post hoc)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "regime_I_min_r2_hist.png"), dpi=150)
    print(by_out.to_string())
    print(by_realm.to_string())
    print({k: v for k, v in rep.items() if k != "rows_in_cell"}, rep["rows_in_cell"])
    print("wrote", out)


def cmd_seeds(a):
    import json
    import joblib
    from multiprocessing import Pool
    from orbit_ml_cr3bp import deploy_d as dd, policy_c_3d as pc, symmetric_cert as sc
    from run_part_d_3d import load_set, load_train, standard_split, raw_log_ds
    out = _make_run_dir(os.path.join(ROOT, a.out_dir), "seeds", a)
    t0 = time.time()
    fz = joblib.load(os.path.join(ROOT, "part_c/frozen/sund/policy_frozen.joblib"))
    cfg, feats = fz["spec"]["config"], list(fz["spec"]["features"])
    size = {"reg_kw": {"max_iter": cfg["n_iter"], "max_depth": cfg.get("depth"), "learning_rate": cfg["lr"]},
            "clf_kw": {"max_iter": max(10, cfg["n_iter"] // 2), "max_depth": cfg.get("depth"), "learning_rate": cfg["lr"]}}
    Utr, wtr = load_train()
    U, ref = load_set("confirm")
    S, T = dd.states_T(U)
    cal, ev = standard_split(U)
    Sc, Tc, rc_, Se, Te, re_ = S[cal], T[cal], ref[cal], S[ev], T[ev], ref[ev]
    d1 = os.path.dirname(_one("out_d/confirm/d1/d1_*/ledger_sund.csv.gz"))
    d1_led = pd.read_csv(os.path.join(d1, "ledger_sund.csv.gz"))
    d1_rep = json.load(open(os.path.join(d1, "d1_report.json")))
    runner = dd.default_runner(MU)
    report = {"label": "POST HOC (posthoc/README.md P6)", "config": cfg, "seeds": a.seeds, "overhead_rhs": OVERHEAD_D, "fits": {}}
    rows = []
    cal_cache, ev_cache = {}, {}
    with Pool(a.n_jobs) as pool:
        def viol(rt, m):
            if (m, rt) not in cal_cache:
                r = sc.scipy_rows(pool, Sc, Tc, rc_, rt, m, n_jobs=a.n_jobs)
                cal_cache[(m, rt)] = int((~r["met"]).sum())
            return cal_cache[(m, rt)], len(cal)
        for seed in a.seeds:
            models = pc.fit_policy_c(Utr, feats, quantile=cfg["q"], seed=seed, lf_gate="none", sample_weight=wtr, structured=cfg.get("structured", False),
                                     candidates=("rk4s",), **size)
            logds = raw_log_ds(models, U)
            if seed == 42:          # must be the frozen model
                same = np.array_equal(logds, raw_log_ds(fz["models"], U))
                report["seed42_reproduces_frozen_predictions"] = bool(same)
                if not same:
                    save_json(report, os.path.join(out, "seeds_report.json"))
                    raise SystemExit("the seed-42 refit does not reproduce the frozen model's predictions; stopping")
            raw = 10.0 ** logds
            c = dd.certify_scale(runner, Sc, Tc, raw[cal], rc_)
            scale = c["chosen_scale"]
            cal_rate = c["violations_by_scale"][scale] / len(cal)
            led = dd.score(U.iloc[ev], dd.run_rows(runner, Se, Te, raw[ev] * scale, re_), OVERHEAD_D)
            fit = {"scale": scale, "cal_violation_rate": cal_rate, "violation_rate": float(led["violation"].mean()), "rtols": {},
                   "median_abs_log_ds_change_vs_frozen": float(np.median(np.abs(logds - raw_log_ds(fz["models"], U))))}
            for m in METHODS:
                eqr, _ch = sc.equal_risk_rtol(lambda rt, m=m: viol(rt, m), cal_rate)
                fit["rtols"][m] = eqr
                if (m, eqr) not in ev_cache:
                    ev_cache[(m, eqr)] = sc.scipy_rows(pool, Se, Te, re_, eqr, m, n_jobs=a.n_jobs)["nfev"]
                led[f"{m}_eqr_nfev"] = ev_cache[(m, eqr)]
            led = led.merge(d1_led[["row_id", "RK45_cal_nfev", "DOP853_cal_nfev"]], on="row_id")       # the certified rtol does not depend on the policy
            if seed == 42:
                s0 = d1_led.set_index("row_id").loc[led["row_id"]]
                fit["seed42_reproduces_D1_ledger"] = bool((led["ml_evals"].to_numpy() == s0["ml_evals"].to_numpy()).all()
                                                          and np.allclose(led["RK45_eqr_nfev"].to_numpy(), s0["RK45_eqr_nfev"].to_numpy(), equal_nan=True)
                                                          and np.allclose(led["DOP853_eqr_nfev"].to_numpy(), s0["DOP853_eqr_nfev"].to_numpy(), equal_nan=True))
                fit["sealed_D1_rtols"] = d1_rep["arms"]["sund"].get("rtols")
            for comp in COMPARATORS:
                r = pm.stats_frame(led, comp, n_boot=a.n_boot)
                rows.append(_flat(r, seed=seed, comparator=comp, scale=scale, violation_rate=fit["violation_rate"]))
            report["fits"][str(seed)] = fit
            led.to_csv(os.path.join(out, f"ledger_seed{seed}.csv.gz"), index=False)
            save_json(report, os.path.join(out, "seeds_report.json"))
            print(f"seed {seed}: scale {scale}, violations {fit['violation_rate']:.4f}, rtols {fit['rtols']} ({time.time() - t0:.0f}s)", flush=True)
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(out, "seeds_summary.csv"), index=False)
    spread = tab.groupby("comparator").agg(share_min=("share", "min"), share_max=("share", "max"), share_sd=("share", "std"),
                                           agg_min=("aggregate_ratio", "min"), agg_max=("aggregate_ratio", "max"), agg_sd=("aggregate_ratio", "std"))
    spread.to_csv(os.path.join(out, "seeds_spread.csv"))
    report["wall_seconds"] = time.time() - t0
    save_json(report, os.path.join(out, "seeds_report.json"))
    pd.set_option("display.width", 250)
    print(tab[["seed", "comparator", "scale", "violation_rate", "share", "aggregate_ratio", "mean_log_ratio"]].round(4).to_string())
    print(spread.round(4).to_string())
    print("wrote", out)


def cmd_atol(a):
    from multiprocessing import Pool
    from orbit_ml_cr3bp import deploy_d as dd, symmetric_cert as sc
    out = _make_run_dir(os.path.join(ROOT, a.out_dir), "atol", a)
    t0 = time.time()
    U, cal, ev, ref, led, rep, _fz, run = _sealed_set("d_replication")
    S, T = dd.states_T(U)
    Sc, Tc, rc_, Se, Te, re_ = S[cal], T[cal], ref[cal], S[ev], T[ev], ref[ev]
    led = led.set_index("row_id").loc[U.iloc[ev]["row_id"].to_numpy()].reset_index()
    target = float(rep["cal_policy_violation_rate"])
    report = {"label": "POST HOC (posthoc/README.md P8)", "sealed_run": run, "target_cal_violation_rate": target, "variants": {}}
    cols, rows = {"row_id": led["row_id"].to_numpy()}, []
    with Pool(a.n_jobs) as pool:
        for name, ratio in sc.ATOL_VARIANTS.items():
            cache = {}

            def viol(rt, ratio=ratio, cache=cache):
                if rt not in cache:
                    r = sc.scipy_rows(pool, Sc, Tc, rc_, rt, "RK45", ratio, n_jobs=a.n_jobs)
                    cache[rt] = (int((~r["met"]).sum()), float(np.nanmean(r["nfev"])), int(r["capped"].sum()))
                return cache[rt][0], len(cal)
            eqr, chain = sc.equal_risk_rtol(viol, target)
            v = {"atol_ratio": ratio, "equal_risk_rtol": eqr, "cal_mean_nfev": cache[eqr][1] if eqr is not None else None,
                 "cal_violation_rate": cache[eqr][0] / len(cal) if eqr is not None else None, "chain": chain}
            if name == "r1e-3":
                v["reproduces_sealed_equal_risk_rtol"] = bool(eqr == rep["rtols"]["RK45|eqr"])
                if not v["reproduces_sealed_equal_risk_rtol"]:
                    report["variants"][name] = v
                    save_json(report, os.path.join(out, "atol_report.json"))
                    raise SystemExit(f"atol 1e-3 gives {eqr}, the sealed equal-risk rtol is {rep['rtols']['RK45|eqr']}; stopping")
            r = sc.scipy_rows(pool, Se, Te, re_, eqr, "RK45", ratio, n_jobs=a.n_jobs)
            cols.update({f"RK45_{name}_nfev": r["nfev"], f"RK45_{name}_ok": r["met"], f"RK45_{name}_capped": r["capped"]})
            st = pm.cost_stats(led["C_ml_tot"].to_numpy(float), r["nfev"], led["meets_tol"].astype(bool).to_numpy(), capped=r["capped"], n_boot=a.n_boot)
            st.update({"comparator_violation_rate": float((~r["met"]).mean()), "rtol": eqr, "cal_mean_nfev": v["cal_mean_nfev"]})
            rows.append(_flat(st, variant=name))
            report["variants"][name] = v
            save_json(report, os.path.join(out, "atol_report.json"))
            print(f"{name}: equal-risk rtol {eqr}, calibration mean nfev {v['cal_mean_nfev']:.0f}, share {st['share']:.4f}, ratio of totals "
                  f"{st['aggregate_ratio']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    best = min(report["variants"], key=lambda k: report["variants"][k]["cal_mean_nfev"])
    report["best_on_calibration"] = best
    tab = pd.DataFrame(rows)
    tab["best_on_calibration"] = tab["variant"] == best
    tab.to_csv(os.path.join(out, "atol_summary.csv"), index=False)
    pd.DataFrame(cols).to_csv(os.path.join(out, "eval_columns.csv.gz"), index=False)
    report["wall_seconds"] = time.time() - t0
    save_json(report, os.path.join(out, "atol_report.json"))
    pd.set_option("display.width", 250)
    print(tab[["variant", "rtol", "cal_mean_nfev", "comparator_violation_rate", "share", "aggregate_ratio", "mean_log_ratio", "best_on_calibration"]].round(4).to_string())
    print("wrote", out)


def cmd_d4b_fine(a):
    import json
    from orbit_ml_cr3bp import compiled_solvers_d as cs, deploy_d as dd, strata_d as sd, policy_c_3d as pc, symmetric_cert as sc
    from orbit_ml_cr3bp.config import POSITION_TOL_KM
    from run_part_d_3d import d4_sets
    out = _make_run_dir(os.path.join(ROOT, a.out_dir), "d4b_fine", a)
    t0 = time.time()
    (Uc, rc_), (Ue, re_) = d4_sets("confirm")
    led = pd.read_csv(_one("out_d/confirm/d4/d4_*/ledger_strat.csv.gz"))
    sealed = json.load(open(_one("out_d/explore/d4b_*/d4b_report.json")))
    cells_c, cells_e = sd.base_cells(Uc), sd.base_cells(Ue)
    smap, labs = sd.build_strata_map(cells_c)
    lab_c, lab_e = sd.assign_strata(cells_c, smap, labs), sd.assign_strata(cells_e, smap, labs)
    G = len(set(lab_c))
    Sc, Tc = dd.states_T(Uc)
    Se, Te = dd.states_T(Ue)
    sv = cs.Solvers(MU, fast=False)
    report = {"label": "POST HOC (posthoc/README.md P7, D4b)", "n_strata": G, "grids": {}}
    rows = []
    for gname, grid in (("coarse_sealed", cs.RTOL_GRID_B), ("fine", sc.RTOL_GRID_FINE)):
        report["grids"][gname] = {}
        for name, label in (("DP54_scipy", "RK45"), ("DOP853", "DOP853")):
            rt_by = {}
            for g in sorted(set(lab_c)):
                idx = np.where(lab_c == g)[0]
                cache = {}

                def viol(rt, idx=idx, cache=cache):
                    if rt not in cache:
                        ok, y, nf, _ = sv.run(name, Sc[idx], Tc[idx], rt, parallel=True)
                        cache[rt] = int((~(ok & (sv.errors_km(y, rc_[idx]) <= POSITION_TOL_KM))).sum())
                    return cache[rt], len(idx)
                cert = pc.certify_sequence(viol, grid, 0.05, 0.1 / G, "fixed_sequence") if len(idx) >= sd.MIN_N_CERT else {"chosen_scale": None}
                rt_by[g] = cert["chosen_scale"] if cert["chosen_scale"] is not None else float(min(grid))
            nfev = np.full(len(Ue), np.nan)
            met = np.zeros(len(Ue), dtype=bool)
            for g in sorted(set(lab_e)):
                idx = np.where(lab_e == g)[0]
                ok, y, nf, _ = sv.run(name, Se[idx], Te[idx], rt_by.get(g, float(min(grid))), parallel=True)
                met[idx] = ok & (sv.errors_km(y, re_[idx]) <= POSITION_TOL_KM)
                nfev[idx] = np.where(ok, nf, np.nan)
            e = pd.DataFrame({"row_id": Ue["row_id"].to_numpy(), "comp_met": met, "comp_nfev": nfev})
            m = led.merge(e, on="row_id")
            st = pm.cost_stats(m["C_ml_tot"].to_numpy(float), m["comp_nfev"].to_numpy(float), m["meets_tol"].astype(bool).to_numpy(), n_boot=a.n_boot)
            st.update({"comparator_violation_rate": float((~m["comp_met"]).mean()), "policy_violation_rate": float((~m["meets_tol"].astype(bool)).mean())})
            if gname == "coarse_sealed":
                st["sealed_share"] = sealed["methods"][label]["share_policy_cheaper"]
                st["reproduces_sealed_share"] = bool(abs(st["share"] - st["sealed_share"]) < 1e-12)
            report["grids"][gname][label] = {"rtol_by_stratum": rt_by, **{k: v for k, v in st.items() if not isinstance(v, list)}}
            rows.append(_flat(st, grid=gname, comparator=label))
            save_json(report, os.path.join(out, "d4b_fine_report.json"))
            print(f"{gname} {label}: share {st['share']:.4f}, ratio of totals {st['aggregate_ratio']:.3f}, comparator violations "
                  f"{st['comparator_violation_rate']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(out, "d4b_fine_summary.csv"), index=False)
    print(tab[["grid", "comparator", "share", "aggregate_ratio", "mean_log_ratio", "comparator_violation_rate"]].round(4).to_string())
    print("wrote", out)


def cmd_compiled(a):
    import json
    import joblib
    from orbit_ml_cr3bp import compiled_solvers_d as cs, deploy_d as dd, symmetric_cert as sc
    from orbit_ml_cr3bp.config import POSITION_TOL_KM
    from run_part_d_3d import load_set, standard_split, FROZEN_SUND
    out = _make_run_dir(os.path.join(ROOT, a.out_dir), "compiled", a)
    t0 = time.time()
    sealed = json.load(open(_one("out_d/confirm/d2/d2_*/d2_report.json")))
    U, ref = load_set("confirm")
    S, T = dd.states_T(U)
    cal, ev = standard_split(U)
    fz = joblib.load(os.path.join(ROOT, FROZEN_SUND))
    oh = float(sealed["overhead_rhs"])
    sp, sv = cs.SundmanPolicy(fz["models"], MU, fast=True), cs.Solvers(MU, fast=True)
    scale = float(sealed["scale"])
    okp, yp, evp, _ = sp.run(S[ev], T[ev], scale)
    metp = okp & (np.where(okp, sp.errors_km(yp, ref[ev]), np.nan) <= POSITION_TOL_KM)
    report = {"label": "POST HOC (posthoc/README.md P7, D2 comparators; RHS counts only, no timing)", "scale": scale, "overhead_rhs": oh,
              "policy_mean_evals": float(evp.mean()), "reproduces_sealed_policy_mean_evals": bool(abs(evp.mean() - sealed["outcome"]["policy|F"]["mean_evals"]) < 1e-9),
              "methods": {}}
    if not report["reproduces_sealed_policy_mean_evals"]:
        save_json(report, os.path.join(out, "compiled_report.json"))
        raise SystemExit("the compiled policy run does not reproduce the sealed D2 evaluation counts; stopping")
    rows = []
    for name in cs.Solvers.NAMES:
        cache = {}

        def viol(rt, cache=cache):
            if rt not in cache:
                ok, y, nf, _ = sv.run(name, S[cal], T[cal], rt, parallel=True)
                cache[rt] = int((~(ok & (sv.errors_km(y, ref[cal]) <= POSITION_TOL_KM))).sum())
            return cache[rt], len(cal)
        cert = sc.certify_rtol_fixed_sequence(viol, sc.RTOL_GRID_FINE)
        res = {}
        for tag, rt in (("cal_sealed", sealed["outcome"][f"{name}_cal|F"]["rtol"]), ("calsym", cert["chosen_rtol"]), ("eqr_sealed", sealed["outcome"][f"{name}_eqr|F"]["rtol"])):
            ok, y, nf, _ = sv.run(name, S[ev], T[ev], rt, parallel=True)
            met = ok & (np.where(ok, sv.errors_km(y, ref[ev]), np.nan) <= POSITION_TOL_KM)
            st = pm.cost_stats(evp + oh, np.where(ok, nf, np.nan).astype(float), metp, n_boot=a.n_boot)
            st.update({"rtol": rt, "comparator_violation_rate": float((~met).mean())})
            rows.append(_flat(st, comparator=name, tolerance=tag))
            res[tag] = {k: v for k, v in st.items() if not isinstance(v, list)}
        report["methods"][name] = {"certified_rtol_fixed_sequence_fine": cert["chosen_rtol"], "chain": cert["chain"], **res}
        save_json(report, os.path.join(out, "compiled_report.json"))
        print(f"{name}: fine fixed-sequence rtol {cert['chosen_rtol']} (sealed Bonferroni {sealed['outcome'][name + '_cal|F']['rtol']}) ({time.time() - t0:.0f}s)", flush=True)
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(out, "compiled_summary.csv"), index=False)
    print(tab[["comparator", "tolerance", "rtol", "comparator_violation_rate", "share", "aggregate_ratio", "mean_log_ratio"]].round(4).to_string())
    print("wrote", out)


# ---------------------------------------------------------------------------------------------------------------------
# Defense-plan analyses of 2026-11-03 (posthoc/README.md P10-P23)
# ---------------------------------------------------------------------------------------------------------------------

B_CONFIRM_LEDGER = "out_b/confirm/policy_*/ledger.csv"
C_PHYS_LEDGER = "out_c/confirm/phys/apply_policy_*/ledger.csv"
SYM_EVAL = {"C_confirm_sund": "posthoc/results/symmetric_c_confirm_*/eval_columns.csv.gz",
            "D_replication_sund": "posthoc/results/symmetric_d_replication_*/eval_columns.csv.gz"}
ALL_COMPS_B = ("RK45_spec", "RK45_cal", "RK45_eqr", "DOP853_spec", "DOP853_cal", "DOP853_eqr")


def _read(pattern):
    return pd.read_csv(_one(pattern), low_memory=False)


def _boot_mean(x, n_boot, seed=0):
    x = np.asarray(x, dtype=float)
    rng = np.random.RandomState(seed)
    b = np.array([x[rng.randint(0, len(x), len(x))].mean() for _ in range(n_boot)])
    return [float(np.quantile(b, .025)), float(np.quantile(b, .975))]


def cmd_risk(a):
    """P10: realized test violation rate of every comparator column with a paired exact McNemar test against the policy, and severities."""
    from orbit_ml_cr3bp import defense_metrics as dm
    from orbit_ml_cr3bp.audit_3d import clopper_pearson
    out = _make_run_dir(os.path.join(ROOT, a.out_dir), "risk", a)
    leds = {n: load_ledger(n)[0] for n in ("C_confirm_sund", "D_replication_sund", "C_lock_sund", "D_lock_sund", "D_replication_D4_strat")}
    leds["C_confirm_phys"] = _read(C_PHYS_LEDGER)
    rows, sev, src = [], [], {n: LEDGERS[n][0] if n in LEDGERS else C_PHYS_LEDGER for n in leds}
    for name, led in leds.items():
        pv = ~led["meets_tol"].astype(bool).to_numpy()
        if "ml_err_km" in led.columns:
            sev.append({"ledger": name, "method": "policy", **dm.severity(led["ml_err_km"].to_numpy(float), pv)})
        cols = {c: (led[f"{c}_ok"].to_numpy(), led[f"{c}_err_km"].to_numpy(float)) for c in COMPARATORS if f"{c}_ok" in led.columns}
        if name in SYM_EVAL:
            E = pd.read_csv(_one(SYM_EVAL[name])).set_index("row_id").loc[led["row_id"]]
            src[f"{name}_symmetric"] = SYM_EVAL[name]
            for m in METHODS:
                cols[f"{m}_calsym"] = (E[f"{m}_calsym_ok"].to_numpy(), np.zeros(len(E)))      # *_ok of P7 is already 'met' (error <= tol)
        for c, (ok, err) in cols.items():
            cv = dm.comparator_violated(ok, err)
            k = int(cv.sum())
            mc = dm.mcnemar_exact(pv, cv)
            rows.append({"ledger": name, "comparator": c, "n": len(cv), "comp_violations": k, "comp_rate": k / len(cv),
                         "comp_cp95_lo": clopper_pearson(k, len(cv))[0], "comp_cp95_hi": clopper_pearson(k, len(cv))[1],
                         "policy_violations": int(pv.sum()), "policy_rate": float(pv.mean()), "only_policy": mc["only_a"], "only_comp": mc["only_b"],
                         "both": mc["both"], "mcnemar_p_two_sided": mc["p_two_sided"]})
            if not c.endswith("calsym"):
                sev.append({"ledger": name, "method": c, **dm.severity(err, cv)})
    pd.DataFrame(rows).to_csv(os.path.join(out, "risk_summary.csv"), index=False)
    pd.DataFrame(sev).to_csv(os.path.join(out, "severity.csv"), index=False)
    save_json({"label": "POST HOC (posthoc/README.md P10)", "sources": src}, os.path.join(out, "risk_report.json"))
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).round(4).to_string())
    print(pd.DataFrame(sev).round(3).to_string())
    print("wrote", out)


def cmd_waterfall(a):
    """P11: the cost decomposition of the Study B policy, the Study C physical-time policy and the Study C Sundman policy on one common set and
    on all usable orbits, the physical paradigm term on the sealed subsets, and the share chain."""
    from orbit_ml_cr3bp.failure_study_3d import decompose, decompose_sundman
    from orbit_ml_cr3bp.confirm_c_3d import win_share
    out = _make_run_dir(os.path.join(ROOT, a.out_dir), "waterfall", a)
    B, P, S = _read(B_CONFIRM_LEDGER), _read(C_PHYS_LEDGER), load_ledger("C_confirm_sund")[0]
    if not (P["row_id"].to_numpy() == S["row_id"].to_numpy()).all():
        raise ValueError("the two Study C ledgers do not hold the same orbits in the same order")
    rep = {"label": "POST HOC (posthoc/README.md P11)", "sources": {"B": B_CONFIRM_LEDGER, "C_phys": C_PHYS_LEDGER, "C_sund": LEDGERS["C_confirm_sund"][0]}}
    rows = []
    terms_p, terms_s = ["P", "S", "Q", "K", "O", "R", "total"], ["P_s", "Q", "K", "O", "R", "total"]
    for comp in ("RK45_spec", "RK45_cal"):
        for c in (f"{comp}_nfev",):
            if not np.allclose(P[c].to_numpy(float), S[c].to_numpy(float), equal_nan=True):
                raise ValueError(f"{c} differs between the Study C ledgers")
        db, dp, ds = decompose(B, comp), decompose(P, comp), decompose_sundman(S, comp)
        r = {"comparator": comp}
        r["B_sealed_set"] = {t: float(db[t].mean()) for t in terms_p} | {"n": int(db["total"].notna().sum())}
        r["C_phys_sealed_set"] = {t: float(dp[t].mean()) for t in terms_p} | {"n": int(dp["total"].notna().sum())}
        r["C_sund_sealed_set"] = {t: float(ds[t].mean()) for t in terms_s} | {"n": int(ds["total"].notna().sum())}
        # (a) one common set within Study C
        abst = P["ml_abstain"].astype(bool).to_numpy()
        common = dp["total"].notna().to_numpy() & ~abst & ds["total"].notna().to_numpy() & P["meets_tol"].astype(bool).to_numpy() \
            & S["meets_tol"].astype(bool).to_numpy()
        r["common_n"] = int(common.sum())
        r["common_C_phys"] = {t: float(dp.loc[common, t].mean()) for t in terms_p}
        r["common_C_sund"] = {t: float(ds.loc[common, t].mean()) for t in terms_s}
        diff = dp.loc[common, "total"].to_numpy() - ds.loc[common, "total"].to_numpy()
        r["common_total_phys_minus_sund"] = {"mean": float(diff.mean()), "ci95": _boot_mean(diff, a.n_boot)}
        # (b) all usable orbits
        col = f"{comp}_nfev"
        ub = {}
        for nm, L, star in (("B", B, "C_fix_star"), ("C_phys", P, "C_fix_star"), ("C_sund_physical_oracle", S, "C_fix_star"), ("C_sund", S, "C_s_star")):
            pg = np.log10(L[star] / L[col])
            tot = np.log10(L["C_ml_tot"] / L[col])
            ub[nm] = {"paradigm_mean": float(pg.mean()), "paradigm_n": int(pg.notna().sum()), "paradigm_ci95": _boot_mean(pg.dropna(), a.n_boot),
                      "total_mean_all_usable": float(tot.mean()), "total_n": int(tot.notna().sum()), "total_ci95": _boot_mean(tot.dropna(), a.n_boot)}
        r["all_usable"] = ub
        both = np.log10(P["C_ml_tot"] / P[col]) - np.log10(S["C_ml_tot"] / S[col])
        r["all_usable_total_phys_minus_sund_paired"] = {"mean": float(both.mean()), "n": int(both.notna().sum()), "ci95": _boot_mean(both.dropna(), a.n_boot)}
        # the physical paradigm term of Study C on the sealed subsets
        pg = np.log10(P["C_fix_star"] / P[col])
        met = P["meets_tol"].astype(bool)
        r["C_physical_paradigm_by_subset"] = {"all_usable": float(pg.mean()), "met_by_phys": float(pg[met].mean()), "met_not_abstained": float(pg[met & ~abst].mean()),
                                               "abstained": float(pg[abst].mean()), "n_abstained": int(abst.sum()),
                                               "met_by_sund": float(pg[S["meets_tol"].astype(bool)].mean()), "common_set": float(pg[common].mean())}
        rep[comp] = r
        for t in terms_p:
            rows.append({"comparator": comp, "set": "common", "policy": "C_phys", "term": t, "mean": r["common_C_phys"][t], "n": r["common_n"]})
        for t in terms_s:
            rows.append({"comparator": comp, "set": "common", "policy": "C_sund", "term": t, "mean": r["common_C_sund"][t], "n": r["common_n"]})
    rep["share_chain"] = {comp: {nm: dict(zip(("k", "n"), win_share(L, comp))) for nm, L in (("B", B), ("C_phys", P), ("C_sund", S))}
                          for comp in ("RK45_eqr", "RK45_cal", "DOP853_eqr", "DOP853_cal")}
    for comp, d in rep["share_chain"].items():
        for nm, v in d.items():
            v["share"] = v["k"] / v["n"]
    pd.DataFrame(rows).to_csv(os.path.join(out, "waterfall_common_terms.csv"), index=False)
    save_json(rep, os.path.join(out, "waterfall_report.json"))
    import json
    print(json.dumps({k: v for k, v in rep.items() if k != "sources"}, indent=1, default=float)[:6000])
    print("wrote", out)


def cmd_overhead(a):
    """P12: Study B shares at overhead 0, 15.04 and the sealed value against all six comparators; Sundman ledgers at k x the sealed overhead."""
    from orbit_ml_cr3bp.audit_3d import clopper_pearson
    out = _make_run_dir(os.path.join(ROOT, a.out_dir), "overhead", a)
    B = _read(B_CONFIRM_LEDGER)
    rows = []
    # 15 Study B runs (collisions / escapes, all violations) have no recorded cost: they stay in the share's denominator as non-wins (the
    # sealed definition) and are left out of the ratio of totals and the mean log ratio
    fin = np.isfinite(B["C_ml_int"].to_numpy(float))
    for tag, h in (("0", 0.0), ("15.04", OVERHEAD_C), ("sealed", None)):
        tmp = B.assign(C_alt=B["C_ml_int"] + (B["C_feat"] + B["C_pred"] if h is None else h))
        for comp in ALL_COMPS_B:
            r = pm.stats_frame(tmp[fin], comp, policy_col="C_alt", n_boot=a.n_boot)
            has = tmp[f"{comp}_nfev"].notna().to_numpy()
            k = int((tmp["meets_tol"].astype(bool).to_numpy() & has & (tmp["C_alt"].to_numpy(float) < tmp[f"{comp}_nfev"].to_numpy(float))).sum())
            r.update({"k": k, "share": k / int(has.sum()), "n": int(len(tmp)), "n_with_comp": int(has.sum()), "n_policy_cost_missing": int((~fin).sum())})
            r.pop("share_ci95", None)
            r["share_cp95"] = list(clopper_pearson(k, int(has.sum())))
            rows.append(_flat(r, ledger="B_confirm", overhead=tag, comparator=comp))
    rep = {"label": "POST HOC (posthoc/README.md P12)", "B_sealed_overhead_median": float((B["C_feat"] + B["C_pred"]).median()),
           "B_C_ml_int_median": float(B["C_ml_int"].median()), "B_comparator_medians": {c: float(B[f"{c}_nfev"].median()) for c in ALL_COMPS_B}}
    by = []
    for name in ("C_confirm_sund", "D_replication_sund"):
        led, _src = load_ledger(name)
        oh = (led["C_ml_tot"] - led["C_ml_int"]).to_numpy()
        if np.ptp(oh) > 1e-9:
            raise ValueError(f"{name}: overhead is not one constant")
        rep[f"{name}_overhead"] = float(oh[0])
        rep[f"{name}_overhead_share_of_policy_total"] = float(oh.sum() / led["C_ml_tot"].sum())
        for k in (0, 1, 2, 5, 10):
            tmp = led.assign(C_alt=led["C_ml_int"] + k * oh[0])
            for comp in COMPARATORS:
                rows.append(_flat(pm.stats_frame(tmp, comp, policy_col="C_alt", n_boot=a.n_boot), ledger=name, overhead=f"{k}x", comparator=comp))
                for r in pm.breakdown(tmp, comp, "h_band", policy_col="C_alt", n_boot=a.n_boot):
                    g = r.pop("h_band")
                    by.append(_flat(r, ledger=name, overhead=f"{k}x", comparator=comp, h_band=g))
    pd.DataFrame(rows).to_csv(os.path.join(out, "overhead_overall.csv"), index=False)
    pd.DataFrame(by).to_csv(os.path.join(out, "overhead_by_h_band.csv"), index=False)
    save_json(rep, os.path.join(out, "overhead_report.json"))
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows)[["ledger", "overhead", "comparator", "k", "n", "share", "aggregate_ratio"]].round(4).to_string())
    print(pd.DataFrame(by).query("comparator == 'RK45_eqr'")[["ledger", "overhead", "h_band", "share", "aggregate_ratio"]].round(4).to_string())
    print(rep)
    print("wrote", out)


def cmd_concentration(a):
    """P13: share of total cost in the costliest 1 % / 10 % of orbits and by horizon band, policy and equal-risk RK45."""
    from orbit_ml_cr3bp import defense_metrics as dm
    out = _make_run_dir(os.path.join(ROOT, a.out_dir), "concentration", a)
    rows, rep = [], {"label": "POST HOC (posthoc/README.md P13)"}
    for name in ("C_confirm_sund", "D_replication_sund"):
        led, src = load_ledger(name)
        rep[name] = {"source": src}
        for meth, col in (("policy", "C_ml_tot"), ("RK45_eqr", "RK45_eqr_nfev")):
            x = led[col].dropna()
            rep[name][meth] = {"top1": dm.top_share(x, 0.01), "top10": dm.top_share(x, 0.10), "n": int(len(x))}
        for g, d in led.groupby("h_band"):
            rows.append({"ledger": name, "h_band": g, "share_of_orbits": len(d) / len(led),
                         "share_of_policy_total": d["C_ml_tot"].sum() / led["C_ml_tot"].sum(),
                         "share_of_RK45_eqr_total": d["RK45_eqr_nfev"].sum() / led["RK45_eqr_nfev"].sum()})
    pd.DataFrame(rows).to_csv(os.path.join(out, "concentration_by_h_band.csv"), index=False)
    save_json(rep, os.path.join(out, "concentration_report.json"))
    print(pd.DataFrame(rows).round(4).to_string())
    print(rep)
    print("wrote", out)


def cmd_timescale(a):
    """P14: rank and partial rank correlation of the oracle fixed-step gap with the timescale range controlling the horizon, and OLS slopes."""
    from scipy.stats import spearmanr
    from orbit_ml_cr3bp import defense_metrics as dm
    out = _make_run_dir(os.path.join(ROOT, a.out_dir), "timescale", a)
    B, S = _read(B_CONFIRM_LEDGER), load_ledger("C_confirm_sund")[0]
    rep, rows = {"label": "POST HOC (posthoc/README.md P14)"}, []
    for nm, L, star in (("B_physical", B, "C_fix_star"), ("C_physical", S, "C_fix_star"), ("C_sundman", S, "C_s_star")):
        d = L[L["RK45_spec_nfev"].notna() & L[star].notna() & L["log_delta_phys"].notna()]
        gap = np.log10(d[star] / d["RK45_spec_nfev"]).to_numpy()
        x = d["log_delta_phys"].to_numpy(float)
        z = np.log10(d["sim_time"].to_numpy(float))
        rng = np.random.RandomState(0)
        bs = {"partial": [], "slope1": [], "slope2_x": [], "slope2_z": []}
        for _ in range(a.n_boot):
            i = rng.randint(0, len(gap), len(gap))
            bs["partial"].append(dm.partial_spearman(gap[i], x[i], z[i]))
            bs["slope1"].append(dm.ols(gap[i], x[i])[1])
            b2 = dm.ols(gap[i], x[i], z[i])
            bs["slope2_x"].append(b2[1])
            bs["slope2_z"].append(b2[2])
        ci = {k: [float(np.quantile(v, .025)), float(np.quantile(v, .975))] for k, v in bs.items()}
        b2 = dm.ols(gap, x, z)
        r = {"n": int(len(gap)), "spearman_gap_logdelta": float(spearmanr(x, gap)[0]), "spearman_gap_logT": float(spearmanr(z, gap)[0]),
             "spearman_logdelta_logT": float(spearmanr(x, z)[0]), "partial_spearman_controlling_logT": dm.partial_spearman(gap, x, z),
             "partial_ci95": ci["partial"], "ols_slope_logdelta_alone": float(dm.ols(gap, x)[1]), "ols_slope_alone_ci95": ci["slope1"],
             "ols_with_logT": {"slope_logdelta": float(b2[1]), "slope_logdelta_ci95": ci["slope2_x"], "slope_logT": float(b2[2]), "slope_logT_ci95": ci["slope2_z"]}}
        for g, dd_ in d.assign(gap=gap).groupby("h_band"):
            rows.append({"set": nm, "h_band": g, "n": len(dd_), "ols_slope": float(dm.ols(dd_["gap"].to_numpy(), dd_["log_delta_phys"].to_numpy(float))[1]),
                         "spearman": float(spearmanr(dd_["log_delta_phys"], dd_["gap"])[0])})
        rep[nm] = r
    pd.DataFrame(rows).to_csv(os.path.join(out, "timescale_by_h_band.csv"), index=False)
    save_json(rep, os.path.join(out, "timescale_report.json"))
    import json
    print(json.dumps(rep, indent=1))
    print(pd.DataFrame(rows).round(3).to_string())
    print("wrote", out)


def cmd_mixture_curve(a):
    """P15: share and ratio of totals as a function of the near-Moon family weight."""
    from orbit_ml_cr3bp import defense_metrics as dm
    out = _make_run_dir(os.path.join(ROOT, a.out_dir), "mixture_curve", a)
    rows = []
    for name in ("C_confirm_sund", "D_replication_sund"):
        led, _src = load_ledger(name)
        fam = led["orbit_label"].to_numpy()
        obs = float(np.mean(fam == "near_moon"))
        for w in sorted(set([round(0.05 * k, 2) for k in range(21)] + [obs])):
            wt = dm.dial_weights(fam, "near_moon", w)
            for comp in ("RK45_eqr", "RK45_cal"):
                r = pm.stats_frame(led, comp, weights=wt, strata=fam, n_boot=a.n_boot)
                rows.append(_flat(r, ledger=name, comparator=comp, near_moon_weight=w, observed=bool(w == obs)))
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(out, "mixture_curve.csv"), index=False)
    save_json({"label": "POST HOC (posthoc/README.md P15)"}, os.path.join(out, "mixture_curve_report.json"))
    print(tab[["ledger", "comparator", "near_moon_weight", "observed", "share", "aggregate_ratio", "aggregate_ratio_ci95_lo", "aggregate_ratio_ci95_hi"]].round(4).to_string())
    print("wrote", out)


def cmd_multiplicity(a):
    """P16: Holm over all 15 B-D tests, D4 strata at level 0.05, resampling floors."""
    import json
    from scipy.stats import binom
    from orbit_ml_cr3bp.confirm_b_3d import holm
    from orbit_ml_cr3bp import defense_metrics as dm
    out = _make_run_dir(os.path.join(ROOT, a.out_dir), "multiplicity", a)
    files = {"B": "part_b/results/confirm/confirm_tests_confirm.json", "C": "part_c/results/confirm/confirm_tests_confirm.json",
             "D": "part_d/results/confirm/confirm_tests_confirm.json"}
    raw, within = {}, {}
    for s, f in files.items():
        h = json.load(open(os.path.join(ROOT, f)))["holm"]
        for k, v in h.items():
            key = f"{s}-{k.split('_')[0]}"
            raw[key], within[key] = float(v["p"]), float(v["p_holm"])
    if len(raw) != 15:
        raise ValueError(f"expected 15 tests, found {len(raw)}")
    h15 = holm(raw)
    floors = dm.floor_flags(raw)
    rows = [{"test": k, "p_raw": raw[k], "p_holm_within_study": within[k], "p_holm_all15": h15[k]["p_holm"], "reject_all15": h15[k]["reject"],
             "at_resampling_floor": floors[k]} for k in sorted(raw)]
    # D4 at 0.05
    led = load_ledger("D_replication_D4_strat")[0]
    sealed = json.load(open(os.path.join(ROOT, files["D"])))
    p_sealed = float(sealed["D4_conditional_safety"]["p_strata"])
    lv = {}
    for level in (0.07, 0.05):
        st = []
        for g, d in led.groupby("stratum"):
            n, k = len(d), int((~d["meets_tol"].astype(bool)).sum())
            if n >= 300:
                st.append({"stratum": g, "n": n, "k": k, "rate": k / n, "p": float(binom.cdf(k, n, level))})
        lv[level] = {"p_iut": max(x["p"] for x in st), "n_strata": len(st), "strata": st}
    lv["reproduces_sealed_at_0.07"] = bool(abs(lv[0.07]["p_iut"] - p_sealed) <= 1e-9 * max(p_sealed, 1e-300))
    rep = {"label": "POST HOC (posthoc/README.md P16)", "sealed_p_strata": p_sealed, "d4_levels": {str(k): v for k, v in lv.items()}}
    if not lv["reproduces_sealed_at_0.07"]:
        save_json(rep, os.path.join(out, "multiplicity_report.json"))
        raise SystemExit(f"D4 at 0.07 gives {lv[0.07]['p_iut']}, sealed {p_sealed}; stopping")
    pd.DataFrame(rows).to_csv(os.path.join(out, "holm_all15.csv"), index=False)
    save_json(rep, os.path.join(out, "multiplicity_report.json"))
    print(pd.DataFrame(rows).to_string())
    print({k: (v["p_iut"], v["n_strata"]) for k, v in lv.items() if isinstance(v, dict)}, "reproduces:", lv["reproduces_sealed_at_0.07"])
    print("wrote", out)


def cmd_bca(a):
    """P17: percentile and BCa intervals for the ratio of totals."""
    from orbit_ml_cr3bp import defense_metrics as dm
    out = _make_run_dir(os.path.join(ROOT, a.out_dir), "bca", a)
    rows = []
    for name in ("C_confirm_sund", "D_replication_sund"):
        led, _src = load_ledger(name)
        for comp in COMPARATORS:
            r = dm.ratio_of_totals_ci(led["C_ml_tot"].to_numpy(float), led[f"{comp}_nfev"].to_numpy(float), n_boot=a.n_boot)
            rows.append(_flat(r, ledger=name, comparator=comp))
    pd.DataFrame(rows).to_csv(os.path.join(out, "bca.csv"), index=False)
    save_json({"label": "POST HOC (posthoc/README.md P17)"}, os.path.join(out, "bca_report.json"))
    print(pd.DataFrame(rows).round(4).to_string())
    print("wrote", out)


def cmd_refs(a):
    """P18: agreement of the second DOP853 reference with the first on every usable row."""
    out = _make_run_dir(os.path.join(ROOT, a.out_dir), "refs", a)
    sets = {"B_confirm": "data_b/cr3bp_3d_training_data_accuracy_monotone_seed22.csv",
            "C_confirm": "data_c/confirm/cr3bp_3d_training_data_accuracy_monotone_seed43.csv",
            "C_lock": "data_c/lock/cr3bp_3d_training_data_accuracy_monotone_seed44.csv",
            "D_confirm": DATA["confirm"], "D_replication": DATA["replication"], "D_lock": DATA["lock"]}
    rows = []
    for nm, f in sets.items():
        d = pd.read_csv(os.path.join(ROOT, f), usecols=["reference_outcome", "ref_check_km"], low_memory=False)
        x = d.loc[d["reference_outcome"] == "ok", "ref_check_km"].to_numpy(float)
        rows.append({"set": nm, "n_usable": len(x), "n_nan": int(np.isnan(x).sum()), "max_km": float(np.nanmax(x)), "median_km": float(np.nanmedian(x)),
                     "p999_km": float(np.nanquantile(x, 0.999)), "n_above_1km": int((x > 1).sum()), "n_above_10km": int((x > 10).sum())})
    pd.DataFrame(rows).to_csv(os.path.join(out, "refs.csv"), index=False)
    save_json({"label": "POST HOC (posthoc/README.md P18)", "sets": sets}, os.path.join(out, "refs_report.json"))
    print(pd.DataFrame(rows).to_string())
    print("wrote", out)


def _policy_features(df):
    """The 25 inputs of the frozen Sundman policy computed from x..vz and sim_time."""
    from orbit_ml_cr3bp.physics_3d import add_orbit_features_3d
    from orbit_ml_cr3bp.osculating_3d import add_osc_features
    return add_osc_features(add_orbit_features_3d(df, MU), MU)


def cmd_ood(a):
    """P19: mean distance to the 10 nearest training rows in the standardized 25-feature space."""
    import joblib
    from sklearn.neighbors import NearestNeighbors
    from run_part_d_3d import load_train, load_set, standard_split
    out = _make_run_dir(os.path.join(ROOT, a.out_dir), "ood", a)
    feats = list(joblib.load(os.path.join(ROOT, "part_c/frozen/sund/policy_frozen.joblib"))["spec"]["features"])
    Utr, _w = load_train()
    Xtr = Utr[feats].to_numpy(float)
    mu_, sd_ = Xtr.mean(axis=0), Xtr.std(axis=0)
    sd_[sd_ == 0] = 1.0
    nn = NearestNeighbors(n_neighbors=10).fit((Xtr - mu_) / sd_)

    def dist(X):
        d, _ = nn.kneighbors((X - mu_) / sd_)
        return d.mean(axis=1)
    U, _ref = load_set("confirm")
    _cal, ev = standard_split(U)
    Uev = U.iloc[ev]
    rebuilt = _policy_features(Uev[["x", "y", "z", "vx", "vy", "vz", "sim_time"]].reset_index(drop=True))
    feat_check = float(np.max(np.abs(rebuilt[feats].to_numpy(float) - Uev[feats].to_numpy(float))))
    if feat_check > 1e-8:
        raise SystemExit(f"rebuilt features differ from the dataset's by {feat_check}; stopping")
    d_in = dist(Uev[feats].to_numpy(float))
    q95, q99 = np.quantile(d_in, [0.95, 0.99])
    groups = {"in_distribution_D_confirm_eval": (d_in, None, None)}
    d5 = pd.read_csv(_one("out_d/confirm/d5/d5_*/d5_runs.csv.gz"))
    d5 = d5[d5["in_range"].astype(bool) & d5["ref_ok"].astype(bool)].reset_index(drop=True)
    X5 = _policy_features(d5[["x", "y", "z", "vx", "vy", "vz", "sim_time"]])[feats].to_numpy(float)
    groups["D5_runs"] = (dist(X5), d5["group"].to_numpy(), d5["violation"].astype(bool).to_numpy())
    for f in sorted(glob.glob(os.path.join(ROOT, _one("out_d/explore/d7_*/d7_report.json").rsplit("/", 1)[0], "slice_C*.csv"))):
        s = pd.read_csv(f)
        s = s[s["ref_ok"].astype(bool)].reset_index(drop=True)
        df = pd.DataFrame({"x": s["x"], "y": 0.0, "z": 0.0, "vx": s["vx"], "vy": s["vy"], "vz": 0.0, "sim_time": 8.0})
        groups[f"Fig10_{os.path.basename(f)[6:-4]}"] = (dist(_policy_features(df)[feats].to_numpy(float)), None, s["violation"].astype(bool).to_numpy())
    rows = []
    for g, (d, fam, viol) in groups.items():
        rows.append({"group": g, "n": len(d), "median_dist": float(np.median(d)), "share_above_in_q95": float((d > q95).mean()),
                     "share_above_in_q99": float((d > q99).mean())})
        if fam is not None:
            for fm in sorted(set(fam)):
                m = fam == fm
                rows.append({"group": f"D5:{fm}", "n": int(m.sum()), "median_dist": float(np.median(d[m])), "share_above_in_q95": float((d[m] > q95).mean()),
                             "share_above_in_q99": float((d[m] > q99).mean())})
    pd.DataFrame(rows).to_csv(os.path.join(out, "ood_distance.csv"), index=False)
    save_json({"label": "POST HOC (posthoc/README.md P19)", "features": feats, "n_train": int(len(Xtr)), "in_q95": float(q95), "in_q99": float(q99),
               "feature_rebuild_max_abs_diff": feat_check}, os.path.join(out, "ood_report.json"))
    print(pd.DataFrame(rows).round(3).to_string())
    print("wrote", out)


def _comparator_summary(L, pol_tot, pol_met, comp_nfev, comp_met, capped, label, n_boot, extra=None):
    from orbit_ml_cr3bp import defense_metrics as dm
    st = pm.cost_stats(pol_tot, comp_nfev, pol_met, n_boot=n_boot, capped=capped)
    mc = dm.mcnemar_exact(~np.asarray(pol_met, bool), ~np.asarray(comp_met, bool))
    st.update({"comparison": label, "comparator_violation_rate": float((~np.asarray(comp_met, bool)).mean()), "policy_violation_rate": float((~np.asarray(pol_met, bool)).mean()),
               "mcnemar_only_policy": mc["only_a"], "mcnemar_only_comp": mc["only_b"], "mcnemar_p": mc["p_two_sided"], **(extra or {})})
    return _flat(st)


def _new_comparator(a, name, rows_fn):
    """Shared driver of P20 / P22: certify a comparator symmetrically and at equal risk on the Study D replication calibration rows, run it on the
    evaluation rows, and compare it with the sealed policy ledger. rows_fn(pool, states, T, ref, rtol) -> dict(ok, err, nfev, capped, met)."""
    from multiprocessing import Pool
    from orbit_ml_cr3bp import symmetric_cert as sc
    out = _make_run_dir(os.path.join(ROOT, a.out_dir), name, a)
    t0 = time.time()
    U, cal, ev, ref, led, rep, _fz, run = _sealed_set("d_replication")
    if a.smoke:                                         # code-path test on a few rows (written to a scratch out_dir; never reported)
        cal, ev = cal[:a.smoke], ev[:a.smoke]
    from orbit_ml_cr3bp import deploy_d as dd
    S, T = dd.states_T(U)
    Sc, Tc, rc_, Se, Te, re_ = S[cal], T[cal], ref[cal], S[ev], T[ev], ref[ev]
    led = led.set_index("row_id").loc[U.iloc[ev]["row_id"].to_numpy()].reset_index()
    target = float(rep["cal_policy_violation_rate"])
    report = {"label": f"POST HOC, EXPLORATORY (posthoc/README.md {name})", "sealed_run": run, "target_cal_violation_rate": target, "methods": {}}
    cols, rows = {"row_id": led["row_id"].to_numpy()}, []
    with Pool(a.n_jobs) as pool:
        for m in a.methods:
            cache = {}

            def viol(rt, m=m, cache=cache):
                if rt not in cache:
                    r = rows_fn(pool, Sc, Tc, rc_, rt, m)
                    cache[rt] = (int((~r["met"]).sum()), float(np.nanmean(r["nfev"])), int(r["capped"].sum()))
                    print(f"  {m} rtol {rt:.3e}: {cache[rt][0]} violations, mean nfev {cache[rt][1]:.0f}, capped {cache[rt][2]} ({time.time() - t0:.0f}s)", flush=True)
                return cache[rt][0], len(cal)
            eqr, eqr_chain = sc.equal_risk_rtol(viol, target)
            cert = sc.certify_rtol_fixed_sequence(viol, sc.RTOL_GRID_FINE)
            report["methods"][m] = {"equal_risk_rtol": eqr, "eqr_chain": eqr_chain, "certified_rtol": cert["chosen_rtol"], "cert_chain": cert["chain"]}
            save_json(report, os.path.join(out, f"{name}_report.json"))
            for tag, rt in (("eqr", eqr), ("calsym", cert["chosen_rtol"])):
                if rt is None:
                    continue
                r = rows_fn(pool, Se, Te, re_, rt, m)
                cols.update({f"{m}_{tag}_rtol": rt, f"{m}_{tag}_nfev": r["nfev"], f"{m}_{tag}_met": r["met"], f"{m}_{tag}_capped": r["capped"], f"{m}_{tag}_err_km": r["err"]})
                rows.append(_comparator_summary(led, led["C_ml_tot"].to_numpy(float), led["meets_tol"].astype(bool).to_numpy(), r["nfev"], r["met"], r["capped"],
                                                f"{name}:{m}_{tag}", a.n_boot, {"rtol": rt}))
                print(f"{m} {tag} rtol {rt:.3e}: comparator violations {(~r['met']).mean():.4f}, policy share {rows[-1]['share']:.4f}, ratio of totals {rows[-1]['aggregate_ratio']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    # the physical-time references at the same rules (sealed equal-risk columns; P7 symmetric columns)
    E7 = pd.read_csv(_one(SYM_EVAL["D_replication_sund"])).set_index("row_id").loc[led["row_id"]]
    for m in ("RK45", "DOP853"):
        cv = (~(led[f"{m}_eqr_ok"].astype(bool) & (led[f"{m}_eqr_err_km"] <= 100))).to_numpy()
        rows.append(_comparator_summary(led, led["C_ml_tot"].to_numpy(float), led["meets_tol"].astype(bool).to_numpy(), led[f"{m}_eqr_nfev"].to_numpy(float), ~cv,
                                        None, f"physical:{m}_eqr (sealed)", a.n_boot))
        rows.append(_comparator_summary(led, led["C_ml_tot"].to_numpy(float), led["meets_tol"].astype(bool).to_numpy(), E7[f"{m}_calsym_nfev"].to_numpy(float),
                                        E7[f"{m}_calsym_ok"].astype(bool).to_numpy(), None, f"physical:{m}_calsym (P7)", a.n_boot))
    pd.DataFrame(cols).to_csv(os.path.join(out, "eval_columns.csv.gz"), index=False)
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(out, f"{name}_summary.csv"), index=False)
    report["wall_seconds"] = time.time() - t0
    save_json(report, os.path.join(out, f"{name}_report.json"))
    pd.set_option("display.width", 250)
    print(tab[["comparison", "rtol", "comparator_violation_rate", "policy_violation_rate", "share", "aggregate_ratio", "aggregate_ratio_ci95_lo",
               "aggregate_ratio_ci95_hi", "mean_log_ratio", "mcnemar_p"]].round(4).to_string())
    print("wrote", out)


def cmd_sundman_adaptive(a):
    """P20: RK45 / DOP853 in Sundman time (exploratory)."""
    from orbit_ml_cr3bp import defense_runs as dr

    def rows_fn(pool, S, T, ref, rt, m):
        return dr.sundman_adaptive_rows(pool, S, T, ref, rt, m, n_jobs=a.n_jobs)
    _new_comparator(a, "sundman_adaptive", rows_fn)


def cmd_multistep(a):
    """P22: LSODA in physical time (exploratory)."""
    from orbit_ml_cr3bp import symmetric_cert as sc

    def rows_fn(pool, S, T, ref, rt, m):
        return sc.scipy_rows(pool, S, T, ref, rt, m, n_jobs=a.n_jobs)
    _new_comparator(a, "multistep", rows_fn)


M_GRID_LEARNED = tuple(float(10.0 ** (-3.0 + j / 8.0)) for j in range(25))


def cmd_learned_rtol(a):
    """P21: RK45 with a per-orbit rtol predicted by the policy's model class, certified like the policy (exploratory)."""
    from multiprocessing import Pool
    import joblib
    from orbit_ml_cr3bp import policy_c_3d as pc, symmetric_cert as sc, defense_runs as dr, deploy_d as dd
    out = _make_run_dir(os.path.join(ROOT, a.out_dir), "learned_rtol", a)
    t0 = time.time()
    feats = list(joblib.load(os.path.join(ROOT, "part_c/frozen/sund/policy_frozen.joblib"))["spec"]["features"])
    tr = load_ledger("C_confirm_sund")[0]
    tr = tr[np.isfinite(tr["RK45_spec_rtol"].to_numpy(float))]
    model = pc._hgb_regressor(0.2, 42, max_iter=25, learning_rate=0.2, max_depth=3)
    model.fit(tr[feats].to_numpy(float), np.log10(tr["RK45_spec_rtol"].to_numpy(float)))
    U, cal, ev, ref, led, rep, _fz, run = _sealed_set("d_replication")
    if a.smoke:
        cal, ev = cal[:a.smoke], ev[:a.smoke]
    S, T = dd.states_T(U)
    logp = model.predict(U[feats].to_numpy(float))
    led = led.set_index("row_id").loc[U.iloc[ev]["row_id"].to_numpy()].reset_index()
    diag = np.log10(led["RK45_spec_rtol"].to_numpy(float)) - logp[ev]
    oh = float(rep["spec"]["config"]["overhead_rhs"])
    target = float(rep["cal_policy_violation_rate"])
    report = {"label": "POST HOC, EXPLORATORY (posthoc/README.md P21)", "n_train": int(len(tr)), "sealed_run": run, "overhead_rhs": oh, "target_cal_violation_rate": target,
              "diagnostic_eval_log10_oracle_minus_pred": {"median": float(np.nanmedian(diag)), "median_abs": float(np.nanmedian(np.abs(diag))),
                                                          "share_pred_looser_than_oracle": float(np.nanmean(diag < 0))}}

    def rtols(m, idx):
        return np.clip(10.0 ** logp[idx] * m, 1e-13, 1e-1)
    cols, rows = {"row_id": led["row_id"].to_numpy()}, []
    with Pool(a.n_jobs) as pool:
        cache = {}

        def viol(m):
            if m not in cache:
                r = dr.scipy_rows_per_row_rtol(pool, S[cal], T[cal], ref[cal], rtols(m, cal), "RK45", n_jobs=a.n_jobs)
                cache[m] = (int((~r["met"]).sum()), float(np.nanmean(r["nfev"])))
                print(f"  m {m:.4e}: {cache[m][0]} violations, mean nfev {cache[m][1]:.0f} ({time.time() - t0:.0f}s)", flush=True)
            return cache[m][0], len(cal)
        eqr, eqr_chain = sc.equal_risk_rtol(viol, target, grid=M_GRID_LEARNED)
        cert = pc.certify_sequence(viol, M_GRID_LEARNED, pc.ALPHA, pc.DELTA, "fixed_sequence")
        report.update({"equal_risk_m": eqr, "eqr_chain": eqr_chain, "certified_m": cert["chosen_scale"], "cert_chain": cert["chain"]})
        save_json(report, os.path.join(out, "learned_rtol_report.json"))
        for tag, m in (("eqr", eqr), ("cert", cert["chosen_scale"])):
            if m is None:
                continue
            r = dr.scipy_rows_per_row_rtol(pool, S[ev], T[ev], ref[ev], rtols(m, ev), "RK45", n_jobs=a.n_jobs)
            cost = r["nfev"] + oh
            cols.update({f"learned_{tag}_nfev": r["nfev"], f"learned_{tag}_met": r["met"], f"learned_{tag}_err_km": r["err"]})
            rows.append(_comparator_summary(led, led["C_ml_tot"].to_numpy(float), led["meets_tol"].astype(bool).to_numpy(), cost, r["met"], r["capped"],
                                            f"learned-rtol RK45 ({tag}, overhead {oh:.2f} charged)", a.n_boot, {"m": m}))
            print(rows[-1]["comparison"], round(rows[-1]["share"], 4), round(rows[-1]["aggregate_ratio"], 4), round(rows[-1]["comparator_violation_rate"], 4), flush=True)
    pd.DataFrame(cols).to_csv(os.path.join(out, "eval_columns.csv.gz"), index=False)
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(out, "learned_rtol_summary.csv"), index=False)
    report["wall_seconds"] = time.time() - t0
    save_json(report, os.path.join(out, "learned_rtol_report.json"))
    pd.set_option("display.width", 250)
    print(tab[["comparison", "m", "comparator_violation_rate", "policy_violation_rate", "share", "aggregate_ratio", "aggregate_ratio_ci95_lo", "aggregate_ratio_ci95_hi", "mcnemar_p"]].round(4).to_string())
    print(report["diagnostic_eval_log10_oracle_minus_pred"])
    print("wrote", out)


def _rk8_job(args):
    from orbit_ml_cr3bp import defense_runs as dr
    s, T, ref = args
    return dr.oracle_ds("rk4", s, T, ref, MU), dr.oracle_ds("rk8", s, T, ref, MU)


def cmd_rk8s(a):
    """P23: oracle fixed steps of RK4 and RK8 in Sundman time on 300 orbits (exploratory)."""
    from multiprocessing import Pool
    from orbit_ml_cr3bp import deploy_d as dd
    out = _make_run_dir(os.path.join(ROOT, a.out_dir), "rk8s", a)
    t0 = time.time()
    U, cal, ev, ref, led, rep, _fz, run = _sealed_set("d_replication")
    S, T = dd.states_T(U)
    pick = np.sort(np.random.RandomState(0).choice(len(ev), a.n_orbits, replace=False))
    rows_ = ev[pick]
    with Pool(a.n_jobs) as pool:
        res = pool.map(_rk8_job, [(S[i], float(T[i]), ref[i]) for i in rows_], chunksize=1)
    L = led.set_index("row_id").loc[U.iloc[rows_]["row_id"].to_numpy()].reset_index()
    D = pd.DataFrame({"row_id": L["row_id"], "sim_time": L["sim_time"], "h_band": L["h_band"], "rk4_ds": [r[0]["ds"] for r in res], "rk4_evals": [r[0]["evals"] for r in res],
                      "rk8_ds": [r[1]["ds"] for r in res], "rk8_evals": [r[1]["evals"] for r in res], "rk4_censored": [r[0]["censored"] for r in res],
                      "rk8_censored": [r[1]["censored"] for r in res], "C_s_star_sealed": L["C_s_star"], "RK45_spec_nfev": L["RK45_spec_nfev"],
                      "DOP853_spec_nfev": L["DOP853_spec_nfev"]})
    D.to_csv(os.path.join(out, "rk8s_orbits.csv"), index=False)
    ok = D["rk4_evals"].notna() & D["rk8_evals"].notna()
    d = D[ok]
    rep_ = {"label": "POST HOC, EXPLORATORY (posthoc/README.md P23)", "n_orbits": int(len(D)), "n_both_labels": int(ok.sum()),
            "share_rk4_cheaper_than_RK45_spec": float((d["rk4_evals"] < d["RK45_spec_nfev"]).mean()),
            "share_rk8_cheaper_than_RK45_spec": float((d["rk8_evals"] < d["RK45_spec_nfev"]).mean()),
            "share_rk4_cheaper_than_DOP853_spec": float((d["rk4_evals"] < d["DOP853_spec_nfev"]).mean()),
            "share_rk8_cheaper_than_DOP853_spec": float((d["rk8_evals"] < d["DOP853_spec_nfev"]).mean()),
            "median_log10_rk8_over_rk4": float(np.median(np.log10(d["rk8_evals"] / d["rk4_evals"]))),
            "median_log10_rk4_over_sealed_C_s_star": float(np.median(np.log10(d["rk4_evals"] / d["C_s_star_sealed"]))),
            "share_rk4_cheaper_than_RK45_spec_sealed_label": float((d["C_s_star_sealed"] < d["RK45_spec_nfev"]).mean()),
            "by_h_band": {g: {"n": int(len(x)), "rk4_vs_RK45": float((x["rk4_evals"] < x["RK45_spec_nfev"]).mean()), "rk8_vs_RK45": float((x["rk8_evals"] < x["RK45_spec_nfev"]).mean()),
                              "rk8_vs_DOP853": float((x["rk8_evals"] < x["DOP853_spec_nfev"]).mean()), "median_log10_rk8_over_rk4": float(np.median(np.log10(x["rk8_evals"] / x["rk4_evals"])))}
                          for g, x in d.groupby("h_band")},
            "censored": {"rk4": int(D["rk4_censored"].sum()), "rk8": int(D["rk8_censored"].sum())}, "wall_seconds": time.time() - t0}
    save_json(rep_, os.path.join(out, "rk8s_report.json"))
    import json
    print(json.dumps(rep_, indent=1))
    print("wrote", out)


# ---------------------------------------------------------------------------------------------------------------------
# Study E pilots (posthoc/README.md P24-P25; exploratory, Study D data only)

RK8_WIDE_GRID = tuple(64.0 * 2.0 ** (-k / 4.0) for k in range(78))      # 64 down to 1.03e-4, ratio 2^(1/4) (P24)
C_TRAIN = ("data_c/train/cr3bp_3d_training_data_accuracy_monotone_seed41.csv",
           "data_c/train_enriched/cr3bp_3d_training_data_accuracy_monotone_seed141.csv")


def _rk8_wide_job(args):
    from orbit_ml_cr3bp import defense_runs as dr
    s, T, ref = args
    return dr.oracle_ds("rk8", s, T, ref, MU, grid=RK8_WIDE_GRID)


def _cost_cmp(a_cost, b_cost):
    """Oracle-vs-oracle comparison of two cost columns (no tolerance violations by construction): rows where both exist."""
    a_cost, b_cost = np.asarray(a_cost, float), np.asarray(b_cost, float)
    ok = np.isfinite(a_cost) & np.isfinite(b_cost)
    x, y = a_cost[ok], b_cost[ok]
    if len(x) == 0:
        return {"n": 0}
    return {"n": int(ok.sum()), "n_missing": int((~ok).sum()), "share_first_cheaper": float((x < y).mean()),
            "ratio_of_totals_second_over_first": float(y.sum() / x.sum()), "median_log10_second_over_first": float(np.median(np.log10(y / x)))}


def cmd_rk8s_vs_sundman(a):
    """P24: the oracle RK8 Sundman-time step (grid widened to 64) against oracle-tuned Sundman-time RK45 / DOP853, on the 300 orbits of P23."""
    from multiprocessing import Pool
    from orbit_ml_cr3bp import deploy_d as dd, study_e_rk8 as e8
    out = _make_run_dir(os.path.join(ROOT, a.out_dir), "rk8s_vs_sundman", a)
    t0 = time.time()
    U, cal, ev, ref, led, rep, _fz, run = _sealed_set("d_replication")
    S, T = dd.states_T(U)
    pick = np.sort(np.random.RandomState(0).choice(len(ev), a.n_orbits, replace=False))     # the P23 draw
    rows_ = ev[pick]
    L = led.set_index("row_id").loc[U.iloc[rows_]["row_id"].to_numpy()].reset_index()
    D = pd.DataFrame({"row_id": L["row_id"], "sim_time": L["sim_time"], "h_band": L["h_band"], "RK45_spec_nfev": L["RK45_spec_nfev"],
                      "DOP853_spec_nfev": L["DOP853_spec_nfev"]})
    with Pool(a.n_jobs) as pool:
        r8 = pool.map(_rk8_wide_job, [(S[i], float(T[i]), ref[i]) for i in rows_], chunksize=1)
        D["rk8_ds"] = [r["ds"] for r in r8]
        D["rk8_evals"] = [r["evals"] for r in r8]
        D["rk8_censored"] = [r["censored"] for r in r8]
        print(f"RK8 oracle done ({time.time() - t0:.0f}s); censored at 64: {int(D['rk8_censored'].sum())}", flush=True)
        for m in ("RK45", "DOP853"):
            for tag, ar in (("", 1e-3), ("_atol01", 0.1)):
                res = pool.map(e8.sundman_matched_job, [(S[i], float(T[i]), ref[i], m, ar) for i in rows_], chunksize=1)
                D[f"{m}s{tag}_spec_rtol"] = [np.nan if r is None else r["rtol"] for r in res]
                D[f"{m}s{tag}_spec_nfev"] = [np.nan if r is None else r["nfev"] for r in res]
                print(f"Sundman-time {m}{tag} oracle done ({time.time() - t0:.0f}s); no rtol found on {int(D[f'{m}s{tag}_spec_nfev'].isna().sum())}", flush=True)
    D.to_csv(os.path.join(out, "rk8s_vs_sundman_orbits.csv"), index=False)
    comps = ["RK45_spec", "DOP853_spec", "RK45s_spec", "DOP853s_spec", "RK45s_atol01_spec", "DOP853s_atol01_spec"]
    rep_ = {"label": "POST HOC, EXPLORATORY (posthoc/README.md P24)", "sealed_run": run, "n_orbits": int(len(D)),
            "rk8_censored_at_64": int(D["rk8_censored"].sum()), "rk8_no_label": int(D["rk8_evals"].isna().sum()),
            "overall": {c: _cost_cmp(D["rk8_evals"], D[f"{c}_nfev"]) for c in comps},
            "by_h_band": {g: {c: _cost_cmp(x["rk8_evals"], x[f"{c}_nfev"]) for c in comps} for g, x in D.groupby("h_band")},
            "note": "first = oracle RK8 Sundman-time fixed step; second = the comparator; *s_spec = oracle-tuned in Sundman time (atol 1e-3 rtol, "
                    "or 0.1 rtol for _atol01); RK45_spec / DOP853_spec = the sealed physical-time oracle-tuned columns",
            "wall_seconds": time.time() - t0}
    save_json(rep_, os.path.join(out, "rk8s_vs_sundman_report.json"))
    import json
    print(json.dumps({k: rep_[k] for k in ("rk8_censored_at_64", "overall")}, indent=1))
    print("wrote", out)


def _rk8_train_labels(a, Utr):
    """RK8 labels (study_e_rk8.rk8s_label) of the Study C training rows, cached under posthoc/cache/ keyed by the rows."""
    from multiprocessing import Pool
    from orbit_ml_cr3bp import deploy_d as dd, study_e_rk8 as e8
    tag = f"_smoke{a.smoke}" if a.smoke else ""
    ref = dd.reference_positions(Utr, MU, cache=os.path.join(ROOT, f"posthoc/cache/refs_c_train{tag}.npz"), n_jobs=a.n_jobs)
    S, T = dd.states_T(Utr)
    key = dd._key(S, T)
    path = os.path.join(ROOT, f"posthoc/cache/rk8s_labels_c_train{tag}.csv.gz")
    if os.path.exists(path) and os.path.exists(path + ".key") and open(path + ".key").read() == key:
        return pd.read_csv(path)
    with Pool(a.n_jobs) as pool:
        res = pool.map(e8.rk8s_label_job, [(S[i], float(T[i]), ref[i]) for i in range(len(Utr))], chunksize=16)
    L = pd.DataFrame(res)
    L.insert(0, "row_id", Utr["row_id"].to_numpy())
    L.to_csv(path, index=False)
    with open(path + ".key", "w") as f:
        f.write(key)
    return L


def cmd_rk8s_learned(a):
    """P25: a learned RK8 Sundman-time step (the sund model class and size, RK8 labels on the Study C training sets), certified and evaluated on the
    Study D replication set against equal-risk (to its own rate) and symmetric-certified adaptive solvers in physical and Sundman time."""
    from multiprocessing import Pool
    import joblib
    import run_part_c_3d as rc
    from orbit_ml_cr3bp import policy_c_3d as pc, symmetric_cert as sc, defense_runs as dr, deploy_d as dd, study_e_rk8 as e8
    out = _make_run_dir(os.path.join(ROOT, a.out_dir), "rk8s_learned", a)
    t0 = time.time()
    frozen = joblib.load(os.path.join(ROOT, "part_c/frozen/sund/policy_frozen.joblib"))
    feats = list(frozen["spec"]["features"])
    paths = [os.path.join(ROOT, p) for p in C_TRAIN]
    Utr, w = rc.load_c_frame(paths, MU, enriched={paths[1]})
    report = {"label": "POST HOC, EXPLORATORY (posthoc/README.md P25)", "n_train_usable": int(len(Utr)),
              "n_train_sealed_sund": int(frozen["spec"]["n_train_rows"])}
    if a.smoke:
        keep = np.sort(np.random.RandomState(0).choice(len(Utr), a.smoke, replace=False))
        Utr, w = Utr.iloc[keep].reset_index(drop=True), w[keep]
    L = _rk8_train_labels(a, Utr)
    print(f"training labels: {len(L)} rows, success {L['rk8s_success'].mean():.4f}, censored at 64 {L['rk8s_censored'].mean():.4f} ({time.time() - t0:.0f}s)", flush=True)
    tr = Utr.copy()
    tr["dt_rk8s"], tr["rk8s_success"] = L["dt_rk8s"].to_numpy(), L["rk8s_success"].to_numpy()
    models = pc.fit_policy_c(tr, feats, quantile=0.2, seed=42, lf_gate="none", sample_weight=w, candidates=("rk8s",),
                             reg_kw={"max_iter": 25, "max_depth": 3, "learning_rate": 0.2})
    joblib.dump(models, os.path.join(out, "policy_rk8s.joblib"))
    report.update({"train_success_rate": float(L["rk8s_success"].mean()), "train_censored_rate": float(L["rk8s_censored"].mean()),
                   "train_label_median_log10_ds": float(np.nanmedian(np.log10(L.loc[L["rk8s_success"] == 1, "dt_rk8s"])))})

    U, cal, ev, ref, led, rep, _fz, run = _sealed_set("d_replication")
    if a.smoke:
        cal, ev = cal[:a.smoke], ev[:a.smoke]
    S, T = dd.states_T(U)
    oh = float(rep["spec"]["config"]["overhead_rhs"])
    raw = 10.0 ** pc.predict_raw_c(models, U)["rk8s"]["logdt"]
    runner = e8.Rk8sRunner(MU)
    cert = dd.certify_scale(runner, S[cal], T[cal], raw[cal], ref[cal])
    scale = cert["chosen_scale"]
    report.update({"sealed_run": run, "overhead_rhs": oh, "certificate": {k: v for k, v in cert.items() if k != "violations_by_scale"},
                   "violations_by_scale": cert["violations_by_scale"], "scale": scale})
    save_json(report, os.path.join(out, "rk8s_learned_report.json"))
    if scale is None:
        print("no scale certified; stopping")
        return
    target = cert["violations_by_scale"][float(scale)] / len(cal)
    report["target_cal_violation_rate"] = target
    print(f"certified scale {scale:.4f}, calibration violation rate {target:.4f} (RK4 policy: {rep['cal_policy_violation_rate']:.4f}) ({time.time() - t0:.0f}s)", flush=True)
    ev_ids = U.iloc[ev]["row_id"].to_numpy()
    led = led.set_index("row_id").loc[ev_ids].reset_index()
    r = dd.run_rows(runner, S[ev], T[ev], raw[ev] * scale, ref[ev])
    pol_tot, pol_met = r["evals"] + oh, r["met"]
    cols = {"row_id": ev_ids, "h_band": led["h_band"].to_numpy(), "rk8_ds": raw[ev] * scale, "rk8_evals": r["evals"], "rk8_met": pol_met, "rk8_err_km": r["err"]}
    report["policy_eval_violation_rate"] = float((~pol_met).mean())
    comps = {}     # name -> (nfev, met, capped)
    with Pool(a.n_jobs) as pool:
        for kind, rows_fn in (("physical", lambda S_, T_, R_, rt, m: sc.scipy_rows(pool, S_, T_, R_, rt, m, n_jobs=a.n_jobs)),
                              ("sundman", lambda S_, T_, R_, rt, m: dr.sundman_adaptive_rows(pool, S_, T_, R_, rt, m, n_jobs=a.n_jobs))):
            for m in ("RK45", "DOP853"):
                cache = {}

                def viol(rt, m=m, cache=cache, rows_fn=rows_fn):
                    if rt not in cache:
                        q = rows_fn(S[cal], T[cal], ref[cal], rt, m)
                        cache[rt] = int((~q["met"]).sum())
                        print(f"  {kind} {m} rtol {rt:.3e}: {cache[rt]} violations ({time.time() - t0:.0f}s)", flush=True)
                    return cache[rt], len(cal)
                eqr, chain = sc.equal_risk_rtol(viol, target)
                report.setdefault("equal_risk", {})[f"{kind}:{m}"] = {"rtol": eqr, "chain": chain}
                save_json(report, os.path.join(out, "rk8s_learned_report.json"))
                if eqr is None:
                    continue
                q = rows_fn(S[ev], T[ev], ref[ev], eqr, m)
                comps[f"{kind}:{m}_eqr"] = (q["nfev"], q["met"], q["capped"])
                cols.update({f"{kind}_{m}_eqr_nfev": q["nfev"], f"{kind}_{m}_eqr_met": q["met"]})
    # symmetric-certified comparators do not depend on the policy: reuse P7 (physical) and P20 (Sundman) columns
    if not a.smoke:
        P7 = pd.read_csv(_one(SYM_EVAL["D_replication_sund"])).set_index("row_id").loc[ev_ids]
        P20 = pd.read_csv(_one("posthoc/results/sundman_adaptive_*/eval_columns.csv.gz")).set_index("row_id").loc[ev_ids]
        for m in ("RK45", "DOP853"):
            comps[f"physical:{m}_calsym (P7)"] = (P7[f"{m}_calsym_nfev"].to_numpy(float), P7[f"{m}_calsym_ok"].astype(bool).to_numpy(), None)
            comps[f"sundman:{m}_calsym (P20)"] = (P20[f"{m}_calsym_nfev"].to_numpy(float), P20[f"{m}_calsym_met"].astype(bool).to_numpy(), None)
    rk4_tot, rk4_met = led["C_ml_tot"].to_numpy(float), led["meets_tol"].astype(bool).to_numpy()
    rows = []
    bands = led["h_band"].to_numpy()
    for name, (nf, met, cap) in comps.items():
        for band in ["all"] + sorted(set(bands)):
            sel = np.ones(len(bands), bool) if band == "all" else bands == band
            cp = None if cap is None else cap[sel]
            rows.append(_comparator_summary(led[sel], pol_tot[sel], pol_met[sel], nf[sel], met[sel], cp, f"RK8 policy vs {name}", a.n_boot, {"h_band": band}))
            if "calsym" in name:      # the same comparator against the sealed RK4 policy, for a like-for-like view
                rows.append(_comparator_summary(led[sel], rk4_tot[sel], rk4_met[sel], nf[sel], met[sel], cp, f"sealed RK4 policy vs {name}", a.n_boot, {"h_band": band}))
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(out, "rk8s_learned_summary.csv"), index=False)
    pd.DataFrame(cols).to_csv(os.path.join(out, "eval_columns.csv.gz"), index=False)
    rel = pd.DataFrame({"rk8": pol_tot, "rk4": rk4_tot, "band": bands})
    report["rk8_vs_rk4_policy"] = {"ratio_of_totals_rk4_over_rk8": float(rel["rk4"].sum() / rel["rk8"].sum()),
                                   "by_band": {g: float(x["rk4"].sum() / x["rk8"].sum()) for g, x in rel.groupby("band")},
                                   "rk4_eval_violation_rate": float((~rk4_met).mean())}
    report["wall_seconds"] = time.time() - t0
    save_json(report, os.path.join(out, "rk8s_learned_report.json"))
    pd.set_option("display.width", 250)
    print(tab[tab["h_band"] == "all"][["comparison", "comparator_violation_rate", "policy_violation_rate", "share", "aggregate_ratio",
                                       "aggregate_ratio_ci95_lo", "aggregate_ratio_ci95_hi", "mcnemar_p"]].round(4).to_string())
    print(report["rk8_vs_rk4_policy"])
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("aggregates")
    sp.add_argument("--out_dir", default=OUT)
    sp.add_argument("--n_boot", type=int, default=pm.N_BOOT)
    sp.set_defaults(fn=cmd_aggregates)
    sp = sub.add_parser("symmetric")
    sp.add_argument("--set", choices=sorted(SYM_SETS), required=True)
    sp.add_argument("--out_dir", default=OUT)
    sp.add_argument("--n_boot", type=int, default=pm.N_BOOT)
    sp.add_argument("--n_jobs", type=int, default=8)
    sp.set_defaults(fn=cmd_symmetric)
    sp = sub.add_parser("three_quarters")
    sp.add_argument("--out_dir", default=OUT)
    sp.add_argument("--n_boot", type=int, default=pm.N_BOOT)
    sp.set_defaults(fn=cmd_three_quarters)
    sp = sub.add_parser("fig4_check")
    sp.add_argument("--out_dir", default=OUT)
    sp.set_defaults(fn=cmd_fig4_check)
    sp = sub.add_parser("atol")
    sp.add_argument("--out_dir", default=OUT)
    sp.add_argument("--n_boot", type=int, default=pm.N_BOOT)
    sp.add_argument("--n_jobs", type=int, default=8)
    sp.set_defaults(fn=cmd_atol)
    for nm, fn in (("d4b_fine", cmd_d4b_fine), ("compiled", cmd_compiled)):
        sp = sub.add_parser(nm)
        sp.add_argument("--out_dir", default=OUT)
        sp.add_argument("--n_boot", type=int, default=pm.N_BOOT)
        sp.set_defaults(fn=fn)
    for nm, fn in (("risk", cmd_risk), ("waterfall", cmd_waterfall), ("overhead", cmd_overhead), ("concentration", cmd_concentration),
                   ("timescale", cmd_timescale), ("mixture_curve", cmd_mixture_curve), ("multiplicity", cmd_multiplicity), ("bca", cmd_bca),
                   ("refs", cmd_refs), ("ood", cmd_ood)):
        sp = sub.add_parser(nm)
        sp.add_argument("--out_dir", default=OUT)
        sp.add_argument("--n_boot", type=int, default=pm.N_BOOT)
        sp.set_defaults(fn=fn)
    for nm, fn, methods in (("sundman_adaptive", cmd_sundman_adaptive, ["RK45", "DOP853"]), ("multistep", cmd_multistep, ["LSODA"])):
        sp = sub.add_parser(nm)
        sp.add_argument("--out_dir", default=OUT)
        sp.add_argument("--n_boot", type=int, default=pm.N_BOOT)
        sp.add_argument("--n_jobs", type=int, default=8)
        sp.add_argument("--methods", nargs="+", default=methods)
        sp.add_argument("--smoke", type=int, default=0)
        sp.set_defaults(fn=fn)
    sp = sub.add_parser("learned_rtol")
    sp.add_argument("--out_dir", default=OUT)
    sp.add_argument("--n_boot", type=int, default=pm.N_BOOT)
    sp.add_argument("--n_jobs", type=int, default=8)
    sp.add_argument("--smoke", type=int, default=0)
    sp.set_defaults(fn=cmd_learned_rtol)
    sp = sub.add_parser("rk8s")
    sp.add_argument("--out_dir", default=OUT)
    sp.add_argument("--n_orbits", type=int, default=300)
    sp.add_argument("--n_jobs", type=int, default=8)
    sp.set_defaults(fn=cmd_rk8s)
    sp = sub.add_parser("rk8s_vs_sundman")
    sp.add_argument("--out_dir", default=OUT)
    sp.add_argument("--n_orbits", type=int, default=300)
    sp.add_argument("--n_jobs", type=int, default=8)
    sp.set_defaults(fn=cmd_rk8s_vs_sundman)
    sp = sub.add_parser("rk8s_learned")
    sp.add_argument("--out_dir", default=OUT)
    sp.add_argument("--n_boot", type=int, default=pm.N_BOOT)
    sp.add_argument("--n_jobs", type=int, default=8)
    sp.add_argument("--smoke", type=int, default=0)
    sp.set_defaults(fn=cmd_rk8s_learned)
    sp = sub.add_parser("seeds")
    sp.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4, 5])
    sp.add_argument("--out_dir", default=OUT)
    sp.add_argument("--n_boot", type=int, default=pm.N_BOOT)
    sp.add_argument("--n_jobs", type=int, default=8)
    sp.set_defaults(fn=cmd_seeds)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
