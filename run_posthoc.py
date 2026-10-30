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
  seeds          P6: the frozen Study C Sundman configuration refitted with model seeds 1-5 (and 42, which must reproduce the frozen model),
                 each certified and evaluated on the Study D confirmation set with the D1 protocol
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
