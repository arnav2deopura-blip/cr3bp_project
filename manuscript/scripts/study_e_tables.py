"""S5 Table: the Study E pilots (P24, P25; exploratory, Study D data) and the Study E dry run on the selection set (not confirmatory).

Reads only tracked result files:
  posthoc/results/rk8s_vs_sundman_*/rk8s_vs_sundman_report.json     P24, oracle 8th-order step against oracle-tuned solvers
  posthoc/results/rk8s_learned_*/rk8s_learned_summary.csv, *_report.json   P25, learned 8th-order step at equal risk
  part_e/frozen/rk8s/selection.csv, train_labels_report.json        the E7 model selection on E_select
  part_e/frozen/overheads.json                                      overheads measured on the idle machine
  part_e/results/dry/tests/tests.json                               every dry-run test statistic
and writes manuscript/si/S5_Table_study_e_*.csv. It computes nothing new: every value is copied from those files, so a number in the
Study E paragraphs of main.tex can be traced to one row here. Run:  python manuscript/scripts/study_e_tables.py
"""
from __future__ import annotations

import glob
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SI = os.path.join(HERE, "..", "si")

BANDS = ["all", "1-3", "3-8", "8-20"]
COMP_NAMES = {
    "RK45_spec": "RK45, physical time", "DOP853_spec": "DOP853, physical time",
    "RK45s_spec": "RK45, Sundman time", "DOP853s_spec": "DOP853, Sundman time",
    "RK45s_atol01_spec": "RK45, Sundman time, atol = 0.1 rtol", "DOP853s_atol01_spec": "DOP853, Sundman time, atol = 0.1 rtol",
    "DP54PI": "DP5(4)-PI, physical time", "DP54PIs": "DP5(4)-PI, Sundman time", "DOP853s": "DOP853, Sundman time",
}


def latest(pattern):
    m = sorted(glob.glob(os.path.join(ROOT, pattern)))
    if not m:
        raise FileNotFoundError(pattern)
    return m[-1]


def write(df, name):
    os.makedirs(SI, exist_ok=True)
    p = os.path.join(SI, f"S5_Table_study_e_{name}.csv")
    df.to_csv(p, index=False, float_format="%.6g")
    print(f"{os.path.relpath(p, ROOT)}  {len(df)} rows")


def pilot_oracle():
    r = json.load(open(os.path.join(latest("posthoc/results/rk8s_vs_sundman_*"), "rk8s_vs_sundman_report.json")))
    rows = []
    for band, block in [("all", r["overall"])] + [(b, r["by_h_band"][b]) for b in BANDS[1:] if b in r["by_h_band"]]:
        for key, e in block.items():
            n = e["n"] - e.get("n_missing", 0)
            rows.append({
                "analysis": "P24 (exploratory)", "h_band_TU": band, "comparator_oracle_tuned": COMP_NAMES.get(key, key), "n": e["n"],
                "k_oracle_rk8_step_cheaper": int(round(e["share_first_cheaper"] * n)), "share_oracle_rk8_step_cheaper": e["share_first_cheaper"],
                "ratio_of_totals_comparator_over_rk8": e["ratio_of_totals_second_over_first"],
                "median_log10_comparator_over_rk8": e["median_log10_second_over_first"],
            })
    df = pd.DataFrame(rows)
    df["rk8_labels_at_grid_ceiling_64"] = r["rk8_censored_at_64"]
    write(df, "pilot_oracle_step")


def pilot_learned():
    d = latest("posthoc/results/rk8s_learned_*")
    s = pd.read_csv(os.path.join(d, "rk8s_learned_summary.csv"))
    r = json.load(open(os.path.join(d, "rk8s_learned_report.json")))
    keep = ["comparison", "h_band", "n", "k", "share", "share_cp95_lo", "share_cp95_hi", "aggregate_ratio", "aggregate_ratio_ci95_lo",
            "aggregate_ratio_ci95_hi", "policy_mean", "comp_mean", "policy_violation_rate", "comparator_violation_rate", "mcnemar_only_policy",
            "mcnemar_only_comp", "mcnemar_p"]
    s = s[keep].rename(columns={"aggregate_ratio": "ratio_of_totals_comparator_over_policy", "aggregate_ratio_ci95_lo": "ratio_ci95_lo",
                                "aggregate_ratio_ci95_hi": "ratio_ci95_hi", "k": "k_policy_cheaper", "share": "share_policy_cheaper",
                                "policy_mean": "policy_mean_evaluations", "comp_mean": "comparator_mean_evaluations"})
    s.insert(0, "analysis", "P25 (exploratory)")
    write(s, "pilot_learned_step")
    v = r["rk8_vs_rk4_policy"]
    meta = [
        ("training rows (usable, Study C fitting sets)", r["n_train_usable"]),
        ("training labels at the grid ceiling (share)", r["train_censored_rate"]),
        ("label search success rate", r["train_success_rate"]),
        ("overhead charged (RHS-equivalents, sealed Study C value)", r["overhead_rhs"]),
        ("certified scale (4,000 calibration rows, Study D replication set)", r["scale"]),
        ("calibration violation rate at the certified scale", r["target_cal_violation_rate"]),
        ("evaluation violation rate, learned RK8 policy", r["policy_eval_violation_rate"]),
        ("evaluation violation rate, sealed RK4 policy", v["rk4_eval_violation_rate"]),
        ("ratio of totals, RK4 policy over RK8 policy, all horizons", v["ratio_of_totals_rk4_over_rk8"]),
    ] + [(f"ratio of totals, RK4 policy over RK8 policy, {b} TU", v["by_band"][b]) for b in BANDS[1:]]
    for name, e in r["equal_risk"].items():
        meta.append((f"equal-risk relative tolerance, {name}", e["rtol"]))
    write(pd.DataFrame(meta, columns=["quantity", "value"]), "pilot_learned_step_settings")


def dry_selection():
    sel = pd.read_csv(os.path.join(ROOT, "part_e/frozen/rk8s/selection.csv"))
    sel.insert(0, "analysis", "E7 model selection on E_select (before the seal)")
    write(sel, "dry_run_model_selection")
    o = json.load(open(os.path.join(ROOT, "part_e/frozen/overheads.json")))
    lab = json.load(open(os.path.join(ROOT, "part_e/frozen/rk8s/train_labels_report.json")))
    rows = [("overhead, RK4 Sundman-time pipeline (RHS-equivalents)", o["rk4s"]), ("overhead, RK8 Sundman-time pipeline (RHS-equivalents)", o["rk8s"]),
            ("overhead, sealed Study C value (RHS-equivalents)", o["rk4s_sealed_study_c"]),
            ("RHS time in the overhead session, RK4 pipeline (ns)", o["measurements"]["rk4s"]["rhs_ns_per_call"]),
            ("1-minute load average at the measurement", o["machine"]["load_average"][0]),
            ("machine", f'{o["machine"]["cpu"]}, {o["machine"]["platform"]}, numba {o["machine"]["numba"]}'),
            ("E7 training labels (rows)", lab["n"]), ("E7 training labels at the ceiling ds = 64 (share)", lab["ceiling"]["share_at_ceiling"]),
            ("E7 training labels at the ceiling whose run takes more than one step", lab["ceiling"]["n_at_ceiling_multi_step"])]
    write(pd.DataFrame(rows, columns=["quantity", "value"]), "dry_run_settings")


def _stat_rows(path, node, out):
    """Every dict that carries a statistic (a ratio, a rate, a difference or only a p-value of an intersection-union test)."""
    if not isinstance(node, dict):
        return
    stat = next((k for k in ("ratio", "family_balanced_rate", "diff", "rate") if isinstance(node.get(k), (int, float))), None)
    if path[-1] == "parts":                      # a dict of part p-values, already carried by the rows of the parts themselves
        stat = None
    elif stat or (isinstance(node.get("p"), (int, float)) and "binding" in node):
        ci = node.get("ci95") or node.get("cp95") or [None, None]
        out.append({
            "path": "/".join(path), "statistic": stat or "intersection-union p", "value": node.get(stat) if stat else None,
            "ci95_lo": ci[0], "ci95_hi": ci[1], "p_one_sided": node.get("p"), "binding_part": node.get("binding"),
            "n": node.get("n", node.get("n_runs")), "k": node.get("k", node.get("violations")),
            "share_policy_cheaper": node.get("share_policy_cheaper"), "policy_violation_rate": node.get("policy_violation_rate"),
            "comparator_violation_rate": node.get("comparator_violation_rate"), "success_policy": node.get("success_policy"),
            "success_comparator": node.get("success_comparator"),
        })
    for k, v in node.items():
        if isinstance(v, dict):
            _stat_rows(path + [str(k)], v, out)


def _ledger_ratios():
    """The ratios of totals of E2 and E3, which tests.json stores only as p-values. Recomputed with the runner's own function
    (confirm_e_3d.ratio_test, seed 0, 4,000 resamples) from the dry-run ledgers in out_e/dry/ (git-ignored working data; deposited with
    the datasets). Skipped, with a message, when the ledgers are not on disk."""
    import sys
    sys.path.insert(0, ROOT)
    out = {}
    e2, e3 = (os.path.join(ROOT, "out_e", "dry", s, "ledger.csv.gz") for s in ("e2", "e3"))
    if not (os.path.exists(e2) and os.path.exists(e3)):
        print("out_e/dry ledgers not found: the E2 and E3 ratio rows are left out of S5_Table_study_e_dry_run_tests.csv")
        return out
    from orbit_ml_cr3bp import confirm_e_3d as ce
    led2 = pd.read_csv(e2, low_memory=False)
    for tol, x in led2.groupby("tol_km"):
        for c in ("DP54PI", "DP54PIs", "DOP853s"):
            if f"{c}_evals" in x:
                r = ce.ratio_test(x["pol_cost"], x[f"{c}_evals"])
                r["policy_violation_rate"] = float((~x["pol_met"].astype(bool)).mean())
                out[f"E2/ledger/{int(tol)}km:{c}"] = r
    led3 = pd.read_csv(e3, low_memory=False)
    for c in ("DP54PI", "DP54PIs", "DOP853s"):
        if f"{c}_evals" in led3:
            out[f"E3/ledger/ratio:{c}"] = ce.ratio_test(led3["pol_cost"], led3[f"{c}_evals"])
    return out


def dry_tests():
    t = json.load(open(os.path.join(ROOT, "part_e/results/dry/tests/tests.json")))
    rows = []
    for test, node in t.items():
        if test in ("holm", "note"):
            continue
        _stat_rows([test], node, rows)
    for path, node in _ledger_ratios().items():
        _stat_rows(path.split("/"), node, rows)
    df = pd.DataFrame(rows)
    df.insert(0, "analysis", "Study E dry run on E_select (selection data, before the seal; NOT a confirmatory result)")
    write(df, "dry_run_tests")
    h = t.get("holm")
    if h is not None:
        hd = pd.DataFrame(h) if isinstance(h, list) else pd.DataFrame([{"test": k, **(v if isinstance(v, dict) else {"value": v})} for k, v in h.items()])
        hd.insert(0, "analysis", "Holm over the dry-run p-values (illustration of the decision rule; NOT a confirmatory result)")
        write(hd, "dry_run_holm")


if __name__ == "__main__":
    pilot_oracle()
    pilot_learned()
    dry_selection()
    dry_tests()
