"""Supporting-information tables S1-S4 (CSV, one file per table section) from the sealed result files and the post hoc run folders. Every number is
read from a result file or computed here; nothing is typed by hand. Output: manuscript/si/.

  S1 Table  D4 conditional certificate by stratum (stratified and one-scale policies, equal-risk RK45 / DOP853)
  S2 Table  D5 periodic-orbit families (by family, stability band, number of periods, and extrapolated horizons)
  S3 Table  physics rules of D1: coefficients with bootstrap 95 % intervals (--n_boot refits on resampled training rows), Koenker-Machado R1 on the
            training rows and mean pinball loss on the Study D confirmation set; the linear 33-feature arm (fit quality); the T1 least-squares fit
  S4 Table  post hoc cost summaries (posthoc/README.md): ratio of totals, mean log ratio and share for every sealed ledger and comparator, by horizon
            band, Hill regime and sampling family, the declared mixtures, and (when their runs exist) the symmetric comparator and fit-seed runs,
            and the defense-plan analyses of 2026-11-03 (posthoc/README.md P10-P23)

  python si_tables.py [--n_boot 100] [--skip_s3_boot]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "manuscript", "si")


def one(pattern):
    f = sorted(glob.glob(os.path.join(ROOT, pattern)))
    if len(f) != 1:
        raise FileNotFoundError(f"{pattern}: {len(f)} matches")
    return f[0]


def latest(pattern):
    f = sorted(glob.glob(os.path.join(ROOT, pattern)))
    return f[-1] if f else None


def write(df, name):
    p = os.path.join(OUT, name)
    df.to_csv(p, index=False)
    print("wrote", os.path.relpath(p, ROOT), len(df), "rows")


def s1():
    d = os.path.dirname(one("out_d/confirm/d4/d4_*/d4_report.json"))
    rep = json.load(open(os.path.join(d, "d4_report.json")))
    st = pd.read_csv(os.path.join(d, "table_strat_by_stratum.csv"))
    mg = pd.read_csv(os.path.join(d, "table_marginal_by_stratum.csv"))
    st.insert(1, "certified_scale", st["group"].map({k: v["scale"] for k, v in rep["strata"].items()}))
    st.insert(2, "n_cal", st["group"].map({k: v["n_cal"] for k, v in rep["strata"].items()}))
    keep = ["group", "n", "violations", "rate", "cp_lo", "cp_hi"]
    m = mg[keep].rename(columns={c: f"onescale_{c}" for c in keep if c != "group"})
    out = st.merge(m, on="group", how="left").rename(columns={"group": "stratum", "n": "n_eval"})
    out = out.drop(columns=["onescale_n"])
    write(out, "S1_Table_D4_strata.csv")


def s2():
    d = os.path.dirname(one("out_d/confirm/d5/d5_*/d5_report.json"))
    for f, tag in (("table_by_group.csv", "by_family"), ("table_by_stab_band.csv", "by_stability_band"), ("table_by_k.csv", "by_number_of_periods"),
                   ("table_extrapolated_by_group.csv", "extrapolated_beyond_20TU_by_family")):
        write(pd.read_csv(os.path.join(d, f)), f"S2_Table_D5_{tag}.csv")


def _pinball(y, pred, q, w=None):
    r = y - pred
    loss = np.maximum(q * r, (q - 1.0) * r)
    return float(np.average(loss, weights=w))


def s3(n_boot, skip_boot):
    import joblib
    from orbit_ml_cr3bp import rules_d as rd
    from orbit_ml_cr3bp.config import MU
    from run_part_d_3d import load_train, load_set
    Utr, wtr = load_train()
    Uc, _ref = load_set("confirm")
    ok_tr = Utr["rk4s_success"].to_numpy() == 1
    ok_c = Uc["rk4s_success"].to_numpy() == 1
    y_tr = np.log10(Utr.loc[ok_tr, "dt_rk4s"].to_numpy(dtype=float))
    y_c = np.log10(Uc.loc[ok_c, "dt_rk4s"].to_numpy(dtype=float))
    w_tr = np.asarray(wtr, dtype=float)[ok_tr]
    rows, fit = [], []
    rng = np.random.RandomState(0)
    boot_idx = [rng.randint(0, len(Utr), len(Utr)) for _ in range(0 if skip_boot else n_boot)]
    for d in sorted(glob.glob(os.path.join(ROOT, "part_d", "frozen", "rule_*"))):
        name = os.path.basename(d)
        spec = json.load(open(os.path.join(d, "policy_frozen_spec.json")))
        rule = joblib.load(os.path.join(d, "policy_frozen.joblib"))["models"]["rule"]
        cols, q = tuple(rule["cols"]), float(rule["quantile"])
        refit = rd.fit_rule(Utr, cols, q, MU, sample_weight=wtr)
        same = np.allclose(refit["w"], rule["w"], rtol=1e-6, atol=1e-9) and abs(refit["b"] - rule["b"]) <= 1e-6
        boots = []
        for bi in boot_idx:
            r = rd.fit_rule(Utr.iloc[bi].reset_index(drop=True), cols, q, MU, sample_weight=np.asarray(wtr, dtype=float)[bi])
            boots.append(np.concatenate([[r["b"]], [r["w"][rd.RULE_FEATURES.index(c)] for c in cols]]))
        boots = np.array(boots) if boots else None
        for j, c in enumerate(("intercept",) + cols):
            est = rule["b"] if c == "intercept" else float(rule["w"][rd.RULE_FEATURES.index(c)])
            rows.append({"rule": name, "quantile": q, "term": c, "estimate": est,
                         "ci95_lo": float(np.quantile(boots[:, j], 0.025)) if boots is not None else None,
                         "ci95_hi": float(np.quantile(boots[:, j], 0.975)) if boots is not None else None, "n_boot": len(boot_idx)})
        p_tr = rd.predict_rule(rule, Utr.loc[ok_tr])
        p_c = rd.predict_rule(rule, Uc.loc[ok_c])
        v_full = _pinball(y_tr, p_tr, q, w_tr)
        v_null = _pinball(y_tr, np.full(len(y_tr), _wquantile(y_tr, w_tr, q)), q, w_tr)       # intercept-only fit: the weighted q-quantile
        fit.append({"arm": name, "quantile": q, "n_params": len(cols) + 1, "refit_reproduces_frozen": bool(same), "pinball_train_weighted": v_full,
                    "R1_koenker_machado_train": 1.0 - v_full / v_null, "pinball_confirm_D": _pinball(y_c, p_c, q), "n_train_rows": int(ok_tr.sum()),
                    "n_confirm_rows": int(ok_c.sum()), "overhead_rhs": spec["config"].get("overhead_rhs")})
        print(name, "refit reproduces frozen:", same, flush=True)
    lin = joblib.load(os.path.join(ROOT, "part_d", "frozen", "lin33", "policy_frozen.joblib"))
    from orbit_ml_cr3bp import policy_c_3d as pc
    q = float(lin["models"]["quantile"])
    p_c = pc.predict_raw_c(lin["models"], Uc.loc[ok_c])["rk4s"]["logdt"]
    fit.append({"arm": "lin33", "quantile": q, "n_params": 34, "pinball_confirm_D": _pinball(y_c, p_c, q), "n_confirm_rows": int(ok_c.sum())})
    sund = joblib.load(os.path.join(ROOT, "part_c", "frozen", "sund", "policy_frozen.joblib"))
    p_c = pc.predict_raw_c(sund["models"], Uc.loc[ok_c])["rk4s"]["logdt"]
    fit.append({"arm": "sund (25 trees)", "quantile": float(sund["spec"]["config"]["q"]), "pinball_confirm_D": _pinball(y_c, p_c, float(sund["spec"]["config"]["q"])),
                "n_confirm_rows": int(ok_c.sum())})
    write(pd.DataFrame(rows), "S3_Table_rule_coefficients.csv")
    write(pd.DataFrame(fit), "S3_Table_fit_quality.csv")
    t1 = json.load(open(os.path.join(ROOT, "part_b", "results", "confirm", "confirm_tests_confirm.json")))["T1_paradigm"]["parts"]["rho_delta_gap"]
    write(pd.DataFrame([{"fit": "T1: paradigm gap (dex) on log10 timescale range, OLS, Study B confirmation set", "n": t1["n"], "slope": t1["slope"],
                         "slope_ci95_lo": t1["slope_ci95"][0], "slope_ci95_hi": t1["slope_ci95"][1], "intercept": t1["intercept"], "spearman_rho": t1["rho"]}]),
          "S3_Table_T1_fit.csv")


def _wquantile(y, w, q):
    """Weighted q-quantile (the smallest y whose cumulative weight reaches q): the minimiser of the weighted pinball loss over constants."""
    o = np.argsort(y)
    cw = np.cumsum(w[o]) / w.sum()
    return float(y[o][min(np.searchsorted(cw, q), len(y) - 1)])


def s4():
    d = latest("posthoc/results/aggregates_*")
    for f in ("overall", "by_group", "mixtures", "overhead"):
        write(pd.read_csv(os.path.join(d, f"aggregates_{f}.csv")), f"S4_Table_posthoc_cost_{f}.csv")
    sym = [latest(f"posthoc/results/symmetric_{s}_*") for s in ("c_confirm", "d_replication")]
    sym = [s for s in sym if s and os.path.exists(os.path.join(s, "symmetric_summary.csv"))]
    if sym:
        write(pd.concat([pd.read_csv(os.path.join(s, "symmetric_summary.csv")) for s in sym], ignore_index=True), "S4_Table_posthoc_symmetric_comparators.csv")
    sd = latest("posthoc/results/seeds_*")
    if sd and os.path.exists(os.path.join(sd, "seeds_summary.csv")):
        write(pd.read_csv(os.path.join(sd, "seeds_summary.csv")), "S4_Table_posthoc_fit_seeds.csv")
    for pat, f, name in (("atol_*", "atol_summary.csv", "S4_Table_posthoc_atol.csv"), ("d4b_fine_*", "d4b_fine_summary.csv", "S4_Table_posthoc_d4b_fine_grid.csv"),
                         ("compiled_*", "compiled_summary.csv", "S4_Table_posthoc_compiled_comparators.csv")):
        d = latest(f"posthoc/results/{pat}")
        if d and os.path.exists(os.path.join(d, f)):
            write(pd.read_csv(os.path.join(d, f)), name)
    tq = latest("posthoc/results/three_quarters_*")
    if tq:
        r = json.load(open(os.path.join(tq, "three_quarters_report.json")))
        write(pd.DataFrame([{k: (v if not isinstance(v, list) else f"{v[0]:.4f}-{v[1]:.4f}") for k, v in r.items() if not isinstance(v, dict)}]),
              "S4_Table_posthoc_three_quarters.csv")
    s4_defense()


def _kv(report, drop=("sources", "chain", "eqr_chain", "cert_chain", "strata", "features")):
    """A nested JSON report as a two-column key / value table (lists of numbers joined, chains and source lists left out)."""
    rows = []

    def walk(x, pre):
        if isinstance(x, dict):
            for k, v in x.items():
                if k not in drop:
                    walk(v, f"{pre}.{k}" if pre else str(k))
        elif isinstance(x, list):
            if all(isinstance(v, (int, float)) or v is None for v in x):
                rows.append({"quantity": pre, "value": " to ".join("" if v is None else f"{v:.6g}" for v in x)})
        else:
            rows.append({"quantity": pre, "value": x})
    walk(report, "")
    return pd.DataFrame(rows)


def s4_defense():
    """The defense-plan analyses of 2026-11-03 (posthoc/README.md P10-P23), when their run folders exist."""
    csvs = (("risk_*", "risk_summary.csv", "S4_Table_posthoc_comparator_risk.csv"), ("risk_*", "severity.csv", "S4_Table_posthoc_violation_severity.csv"),
            ("waterfall_*", "waterfall_common_terms.csv", "S4_Table_posthoc_waterfall_common_set.csv"),
            ("overhead_*", "overhead_overall.csv", "S4_Table_posthoc_overhead_multiples.csv"), ("overhead_*", "overhead_by_h_band.csv", "S4_Table_posthoc_overhead_by_h_band.csv"),
            ("concentration_*", "concentration_by_h_band.csv", "S4_Table_posthoc_cost_concentration_by_h_band.csv"),
            ("timescale_*", "timescale_by_h_band.csv", "S4_Table_posthoc_timescale_by_h_band.csv"),
            ("mixture_curve_*", "mixture_curve.csv", "S4_Table_posthoc_mixture_dial.csv"), ("multiplicity_*", "holm_all15.csv", "S4_Table_posthoc_holm_all15.csv"),
            ("bca_*", "bca.csv", "S4_Table_posthoc_bca_intervals.csv"), ("refs_*", "refs.csv", "S4_Table_posthoc_reference_agreement.csv"),
            ("ood_*", "ood_distance.csv", "S4_Table_posthoc_distance_to_training.csv"),
            ("sundman_adaptive_*", "sundman_adaptive_summary.csv", "S4_Table_posthoc_adaptive_in_sundman_time.csv"),
            ("learned_rtol_*", "learned_rtol_summary.csv", "S4_Table_posthoc_learned_rtol.csv"), ("multistep_*", "multistep_summary.csv", "S4_Table_posthoc_multistep.csv"))
    for pat, f, name in csvs:
        d = latest(f"posthoc/results/{pat}")
        if d and os.path.exists(os.path.join(d, f)):
            write(pd.read_csv(os.path.join(d, f)), name)
    reps = (("waterfall_*", "waterfall_report.json", "S4_Table_posthoc_waterfall.csv"), ("concentration_*", "concentration_report.json", "S4_Table_posthoc_cost_concentration.csv"),
            ("timescale_*", "timescale_report.json", "S4_Table_posthoc_timescale.csv"), ("multiplicity_*", "multiplicity_report.json", "S4_Table_posthoc_d4_level.csv"),
            ("rk8s_*", "rk8s_report.json", "S4_Table_posthoc_rk8_sundman_oracle.csv"), ("learned_rtol_*", "learned_rtol_report.json", "S4_Table_posthoc_learned_rtol_settings.csv"),
            ("sundman_adaptive_*", "sundman_adaptive_report.json", "S4_Table_posthoc_adaptive_in_sundman_time_settings.csv"),
            ("multistep_*", "multistep_report.json", "S4_Table_posthoc_multistep_settings.csv"))
    for pat, f, name in reps:
        d = latest(f"posthoc/results/{pat}")
        if d and os.path.exists(os.path.join(d, f)):
            write(_kv(json.load(open(os.path.join(d, f)))), name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_boot", type=int, default=100)
    ap.add_argument("--skip_s3_boot", action="store_true")
    ap.add_argument("--only", nargs="*", default=["s1", "s2", "s3", "s4"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if "s1" in a.only:
        s1()
    if "s2" in a.only:
        s2()
    if "s3" in a.only:
        s3(a.n_boot, a.skip_s3_boot)
    if "s4" in a.only:
        s4()


if __name__ == "__main__":
    main()
