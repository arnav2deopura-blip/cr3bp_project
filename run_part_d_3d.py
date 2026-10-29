"""
Part D commands (part_d/PREREGISTRATION_D.md). Each command writes a run folder (manifest + outputs) under the given --out_dir.

  refs               DOP853 reference final positions of a dataset's usable rows, cached (out_d/refs/<set>.npz)
  overheads          single-call cost in RHS-equivalents of the frozen `sund` model and of every rule on an idle machine (part_d/selection)
  select_rules       fit the pre-specified rules on D_train, certify and score them on D_select_D (calibration / evaluation folds); writes the table
  freeze_rule        fit the chosen rule and write part_d/frozen/<name>/ (policy_frozen.joblib + spec) in the format of the Part C frozen policies
  (the confirmatory steps D1-D5 are run by run_confirm_d_3d.py, which calls the functions below)
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
from orbit_ml_cr3bp.config import MU, POSITION_TOL_KM  # noqa: E402

SEEDS = {"select": 52, "confirm": 53, "lock": 54, "replication": 55}
DATA = {k: f"data_d/{k}/cr3bp_3d_training_data_accuracy_monotone_seed{s}.csv" for k, s in SEEDS.items()}
TRAIN = ["data_c/train/cr3bp_3d_training_data_accuracy_monotone_seed41.csv", "data_c/train_enriched/cr3bp_3d_training_data_accuracy_monotone_seed141.csv"]
TRAIN_ENRICHED = ["data_c/train_enriched/cr3bp_3d_training_data_accuracy_monotone_seed141.csv"]
FROZEN_SUND = "part_c/frozen/sund/policy_frozen.joblib"
N_CAL = 4000          # calibration rows of the standard protocol (Part C: seed 42 draw of the usable rows)
CAL_SEED = 42         # run_failure_study_3d.split_rows(U, N_CAL, 0, seed), as in run_part_c_3d apply_policy (its --seed default)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(proto.json_safe(obj), f, indent=1)


def make_run_dir(out_dir, name, args):
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    d = os.path.join(ROOT, out_dir, f"{name}_{stamp}") if not os.path.isabs(out_dir) else os.path.join(out_dir, f"{name}_{stamp}")
    os.makedirs(d, exist_ok=True)
    save_json({"command": name, "args": vars(args), "utc": stamp, "python": sys.version, "argv": sys.argv}, os.path.join(d, "manifest.json"))
    return d


def load_set(name):
    """Usable rows of a Part D dataset with the derived columns (regime, bands, ...) and the cached reference final positions."""
    import run_part_c_3d as rc
    from orbit_ml_cr3bp import deploy_d as dd
    U, _w = rc.load_c_frame([os.path.join(ROOT, DATA[name])], MU)
    ref = dd.reference_positions(U, MU, cache=os.path.join(ROOT, "out_d", "refs", f"{name}.npz"), n_jobs=8)
    return U, ref


def load_train():
    import run_part_c_3d as rc
    return rc.load_c_frame([os.path.join(ROOT, p) for p in TRAIN], MU, enriched={os.path.join(ROOT, p) for p in TRAIN_ENRICHED})


def standard_split(U, n_cal=N_CAL, seed=CAL_SEED):
    """The Part C calibration / evaluation split of the usable rows (run_failure_study_3d.split_rows with n_eval <= 0)."""
    from run_failure_study_3d import split_rows
    _tr, cal, ev = split_rows(U, n_cal, 0, seed)
    return cal, ev


def cmd_refs(a):
    t0 = time.time()
    for name in a.sets:
        U, ref = load_set(name)
        print(f"{name}: {len(U)} usable rows, reference positions {ref.shape}, {time.time() - t0:.0f}s", flush=True)


def _runner():
    from orbit_ml_cr3bp import deploy_d as dd
    return dd.default_runner(MU)


def cmd_overheads(a):
    """Single-call cost (RHS-equivalents, features included) of the frozen `sund` model and of every rule, in ONE session on an idle machine. The Part C
    pinned value of `sund` (15.04) is kept for the Part C tests; D1 charges every arm the value measured here."""
    import joblib
    from orbit_ml_cr3bp import compiled_model_3d as cm, rules_d as rd
    out = make_run_dir(a.out_dir, "overheads", a)
    U, _ref = load_set(a.set)
    U = U.sample(min(len(U), 2000), random_state=0)
    st, T = U[["x", "y", "z", "vx", "vy", "vz"]].to_numpy(dtype=float), U["sim_time"].to_numpy(dtype=float)
    res = {"load_avg_start": os.getloadavg()}
    fz = joblib.load(os.path.join(ROOT, FROZEN_SUND))
    pol = cm.CompiledHGBPolicy(fz["models"])
    res["sund"] = cm.measure_single_call_cost(pol, st, T, repeats=a.repeats)
    res["sund"]["pinned_part_c"] = fz["spec"]["config"]["overhead_rhs"]
    train, w = load_train()
    for name, cols in rd.RULE_SETS.items():
        rule = rd.fit_rule(train.sample(20000, random_state=0), cols, 0.2, MU, sample_weight=None)      # coefficients do not change the cost
        res[name] = rd.measure_rule_cost(rule, st, T, repeats=a.repeats)
        print(name, round(res[name]["rhs_equiv"], 2), flush=True)
    lin = {"mu": MU, "c_l1": cm.c_l1_for_mu(MU), "w": np.linspace(-0.01, 0.01, 33), "b": -0.5}      # the cost does not depend on the coefficients
    res["lin33"] = cm.measure_linear_single_cost(lin, st, T, repeats=a.repeats)
    print("lin33", round(res["lin33"]["rhs_equiv"], 2), flush=True)
    res["load_avg_end"] = os.getloadavg()
    print("sund", round(res["sund"]["rhs_equiv"], 2), "(Part C pinned", res["sund"]["pinned_part_c"], ")")
    save_json(res, os.path.join(out, "overheads.json"))
    save_json(res, os.path.join(ROOT, "part_d", "selection", "overheads_idle.json"))


def _overheads():
    return json.load(open(os.path.join(ROOT, "part_d", "selection", "overheads_idle.json")))


def fit_all_rules(train, w, quantiles, mu=MU):
    from orbit_ml_cr3bp import rules_d as rd
    fitted = {}
    for name, cols in rd.RULE_SETS.items():
        for q in quantiles:
            fitted[(name, q)] = rd.fit_rule(train, cols, q, mu, sample_weight=w)
    return fitted


def eval_raw(runner, S, T, ref, raw_ds, cal, ev, overhead, tol_km=POSITION_TOL_KM):
    """Certify one scale on the cal rows, run on the eval rows. Returns summary dict and the eval-row arrays."""
    from orbit_ml_cr3bp import deploy_d as dd
    from orbit_ml_cr3bp.audit_3d import clopper_pearson
    rep = dd.certify_scale(runner, S[cal], T[cal], raw_ds[cal], ref[cal], tol_km)
    sc = rep["chosen_scale"]
    if sc is None:
        return {"scale": None}, None
    run = dd.run_rows(runner, S[ev], T[ev], raw_ds[ev] * sc, ref[ev], tol_km)
    tot = run["evals"] + overhead
    met = run["met"]
    k = int((~met).sum())
    lo, hi = clopper_pearson(k, len(ev))
    return {"scale": sc, "cal_violation_rate": rep["violations_by_scale"][sc] / len(cal), "n_eval": int(len(ev)), "violations": k, "violation_rate": k / len(ev),
            "viol_cp95": [lo, hi], "median_c_int_met": float(np.median(run["evals"][met])), "median_c_tot_met": float(np.median(tot[met])),
            "mean_log10_c_tot_met": float(np.mean(np.log10(tot[met])))}, {"met": met, "tot": tot, "evals": run["evals"]}


def cmd_select_rules(a):
    """D1 rule selection on D_select_D (every choice here, nothing on the confirmation set): each pre-specified rule at each quantile is fitted on D_train,
    one scale is certified on the calibration fold (half of the usable rows, seed 42) and the rule is scored on the evaluation fold by the median total
    cost per orbit that met the tolerance (overhead included). The winner: lowest median total cost among candidates with an evaluation-fold violation rate
    <= 6 %, ties within 2 % to the rule with fewer coefficients."""
    import joblib
    from orbit_ml_cr3bp import rules_d as rd, policy_c_3d as pc
    out = make_run_dir(a.out_dir, "select_rules", a)
    ov = _overheads()
    U, ref = load_set("select")
    S, T = U[["x", "y", "z", "vx", "vy", "vz"]].to_numpy(dtype=float), U["sim_time"].to_numpy(dtype=float)
    cal, ev = pc.split_folds(len(U), a.seed)
    runner = _runner()
    train, w = load_train()
    rows = []
    fz = joblib.load(os.path.join(ROOT, FROZEN_SUND))
    raw_ml = pc.predict_raw_c(fz["models"], U)["rk4s"]["logdt"]
    summ, _ = eval_raw(runner, S, T, ref, 10.0 ** raw_ml, cal, ev, ov["sund"]["rhs_equiv"])
    rows.append({"name": "sund (ML, Part C)", "n_params": None, "q": 0.2, "overhead": ov["sund"]["rhs_equiv"], **summ})
    print(rows[-1], flush=True)
    for (name, q), rule in fit_all_rules(train, w, a.quantiles).items():
        t0 = time.time()
        raw = rd.predict_rule(rule, U)
        summ, _ = eval_raw(runner, S, T, ref, 10.0 ** raw, cal, ev, ov[name]["rhs_equiv"])
        rows.append({"name": name, "n_params": rule["n_params"], "q": q, "overhead": ov[name]["rhs_equiv"], "coef": dict(zip(rule["cols"], np.round(rule["w"][[rd.RULE_FEATURES.index(c) for c in rule["cols"]]], 4))),
                     "b": rule["b"], "median_abs_err_dex_vs_label": float(np.median(np.abs(raw - np.log10(U["dt_rk4s"].to_numpy())))), **summ,
                     "s": round(time.time() - t0, 1)})
        print({k: rows[-1].get(k) for k in ("name", "q", "scale", "violation_rate", "median_c_tot_met")}, flush=True)
        save_json({"rows": rows}, os.path.join(out, "select_rules.json"))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "select_rules.csv"), index=False)
    cand = df[(df["name"].isin(list(rd.RULE_SETS))) & (df["violation_rate"] <= 0.06)].copy()
    best = cand["median_c_tot_met"].min()
    near = cand[cand["median_c_tot_met"] <= 1.02 * best].sort_values(["n_params", "median_c_tot_met"])
    win = near.iloc[0]
    sel = {"winner": {"name": win["name"], "q": float(win["q"])}, "criterion": "median total cost per met orbit; 2 % ties to fewer coefficients; eval-fold violation rate <= 6 %",
           "best_median": float(best), "within_2pct": near[["name", "q", "median_c_tot_met", "n_params"]].to_dict("records")}
    save_json(sel, os.path.join(out, "selection.json"))
    save_json(sel, os.path.join(ROOT, "part_d", "selection", "rule_selection.json"))
    print(json.dumps(proto.json_safe(sel), indent=1))


def cmd_freeze_lin33(a):
    """Fit and write the descriptive linear arm of D1 (part_d/frozen/lin33)."""
    import joblib
    from orbit_ml_cr3bp import rules_d as rd
    train, w = load_train()
    lin = rd.fit_lin33(train, a.q, MU, sample_weight=w)
    models = {"features": ["x", "y", "z", "vx", "vy", "vz", "sim_time"], "candidates": ("rk4s",), "linear": {"rk4s": lin}, "lf_gate": "none", "structured": False,
              "quantile": a.q}
    cfg = {"name": "lin33_final", "family": "sundman", "cls": "linear", "q": a.q, "rule": "fixed_sequence", "grid": "fine", "overhead_rhs": _overheads()["lin33"]["rhs_equiv"]}
    spec = {"config": cfg, "gate": {"lf_gate": "none", "p_star": 0.5, "g_star": 0.0}, "features": models["features"], "candidates": ["rk4s"], "n_train_rows": int(len(train)),
            "train_files": TRAIN}
    d = os.path.join(ROOT, "part_d", "frozen", "lin33")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "policy_frozen.joblib")
    joblib.dump({"models": models, "spec": spec}, path)
    spec["model_file_sha256"] = proto.sha256_file(path)
    save_json(spec, os.path.join(d, "policy_frozen_spec.json"))
    print("wrote", d)


def cmd_freeze_rule(a):
    """Fit the chosen rule (name, q from part_d/selection/rule_selection.json unless given) on D_train and write part_d/frozen/<out_name>/ in the Part C
    frozen format (joblib {models, spec} + a spec json), overhead pinned from part_d/selection/overheads_idle.json."""
    import joblib
    from orbit_ml_cr3bp import rules_d as rd
    sel = json.load(open(os.path.join(ROOT, "part_d", "selection", "rule_selection.json")))["winner"]
    name, q = a.rule or sel["name"], a.q if a.q is not None else sel["q"]
    train, w = load_train()
    rule = rd.fit_rule(train, rd.RULE_SETS[name], q, MU, sample_weight=w)
    ov = _overheads()
    cfg = {"name": f"{name}_final", "family": "sundman", "cls": "rule", "rule_name": name, "q": q, "features": list(rule["cols"]), "rule": "fixed_sequence",
           "grid": "fine", "overhead_rhs": ov[name]["rhs_equiv"]}
    models = rd.rule_models(rule)
    spec = {"config": cfg, "gate": {"lf_gate": "none", "p_star": 0.5, "g_star": 0.0}, "features": list(rule["cols"]), "candidates": ["rk4s"],
            "n_train_rows": int(len(train)), "coef": dict(zip(rule["cols"], rule["w"][[rd.RULE_FEATURES.index(c) for c in rule["cols"]]].tolist())), "b": rule["b"],
            "train_files": TRAIN}
    d = os.path.join(ROOT, "part_d", "frozen", a.out_name or name)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "policy_frozen.joblib")
    joblib.dump({"models": models, "spec": spec}, path)
    spec["model_file_sha256"] = proto.sha256_file(path)
    save_json(spec, os.path.join(d, "policy_frozen_spec.json"))
    print("wrote", d, json.dumps(proto.json_safe({k: spec[k] for k in ("coef", "b")})))


# ---------------------------------------------------------------------------------------------------------------------
# D1: the ML policy against the physics rules, on the standard calibration / evaluation split of one dataset
# ---------------------------------------------------------------------------------------------------------------------

TARGET_SET = {"dry": "select", "confirm": "confirm", "lock": "lock", "replication": "replication"}
ARMS_FROZEN = {"sund": FROZEN_SUND}          # + part_d/frozen/<name>/ for every rule (see arm_paths)


def arm_paths():
    """{arm name: frozen joblib path}: the ML policy, then every frozen rule in part_d/frozen (the first listed in rule_best.txt is the D1 test arm)."""
    arms = dict(ARMS_FROZEN)
    base = os.path.join(ROOT, "part_d", "frozen")
    if os.path.isdir(base):
        for d in sorted(os.listdir(base)):
            if os.path.exists(os.path.join(base, d, "policy_frozen.joblib")):
                arms[d] = os.path.join("part_d", "frozen", d, "policy_frozen.joblib")
    return arms


def best_rule_name():
    return json.load(open(os.path.join(ROOT, "part_d", "selection", "rule_selection.json")))["winner"]["name"]


def policy_overhead(arm, spec):
    """Overhead charged in D1: the value measured in the idle session (part_d/selection/overheads_idle.json) for the ML policy and every rule."""
    ov = _overheads()
    key = "sund" if arm == "sund" else spec["config"].get("rule_name", arm)
    return float(ov[key]["rhs_equiv"])


def raw_log_ds(arm_models, U):
    from orbit_ml_cr3bp import policy_c_3d as pc
    return pc.predict_raw_c(arm_models, U)["rk4s"]["logdt"]


def cmd_d1(a):
    """D1 on one target: every arm (ML, best rule, the other rules, the linear model if frozen) certified on the 4,000 calibration rows and scored on the
    evaluation rows; for the ML arm and the best rule also the adaptive comparators at the certified and the equal-risk rtol (their own, recomputed per arm).
    Writes one ledger per arm and d1_report.json."""
    import joblib
    from orbit_ml_cr3bp import deploy_d as dd
    from orbit_ml_cr3bp.audit_3d import clopper_pearson
    out = make_run_dir(a.out_dir, "d1", a)
    t0 = time.time()
    U, ref = load_set(TARGET_SET[a.target])
    S, T = dd.states_T(U)
    cal, ev = standard_split(U)
    cal_df, ev_df = U.iloc[cal].reset_index(drop=True), U.iloc[ev].reset_index(drop=True)
    runner = _runner()
    arms = arm_paths()
    best = best_rule_name()
    rep = {"target": a.target, "set": TARGET_SET[a.target], "n_usable": int(len(U)), "n_cal": int(len(cal)), "n_eval": int(len(ev)), "best_rule": best, "arms": {}}
    cal_rtols_cache = {}
    for arm, path in arms.items():
        fz = joblib.load(os.path.join(ROOT, path))
        models, spec = fz["models"], fz["spec"]
        oh = policy_overhead(arm, spec) if arm != "sund" else float(_overheads()["sund"]["rhs_equiv"])
        raw = 10.0 ** raw_log_ds(models, U)
        c = dd.certify_scale(runner, S[cal], T[cal], raw[cal], ref[cal])
        sc = c["chosen_scale"]
        r = {"overhead_rhs": oh, "scale": sc, "cal_violations": c["violations_by_scale"].get(sc) if sc else None, "chain": c["chain"]}
        if sc is None:
            rep["arms"][arm] = r
            continue
        r["cal_violation_rate"] = c["violations_by_scale"][sc] / len(cal)
        run = dd.run_rows(runner, S[ev], T[ev], raw[ev] * sc, ref[ev])
        led = dd.score(ev_df, run, oh)
        led["raw_ds"] = raw[ev]
        k = int(led["violation"].sum())
        r.update({"violations": k, "violation_rate": k / len(ev), "viol_cp95": list(clopper_pearson(k, len(ev))),
                  "median_C_int_met": float(led.loc[led["meets_tol"], "C_ml_int"].median()), "median_C_tot_met": float(led.loc[led["meets_tol"], "C_ml_tot"].median()),
                  "mean_log10_C_tot_met": float(np.log10(led.loc[led["meets_tol"], "C_ml_tot"]).mean())})
        if arm in ("sund", best):
            rt, rtrep = dd.comparator_rtols(cal_df, ref[cal], r["cal_violation_rate"], n_jobs=a.n_jobs)
            r["rtols"] = {f"{m}|{t}": v for (m, t), v in rt.items()}
            comp = dd.comparator_frame(ev_df, ref[ev], rt, n_jobs=a.n_jobs)
            led = led.merge(comp, on="row_id")
            for cname in ("RK45_cal", "DOP853_cal", "RK45_eqr", "DOP853_eqr"):
                has = led[f"{cname}_nfev"].notna()
                r[f"share_cheaper_{cname}"] = float((led["meets_tol"] & has & (led["C_ml_tot"] < led[f"{cname}_nfev"])).sum() / max(has.sum(), 1))
        led.to_csv(os.path.join(out, f"ledger_{arm}.csv.gz"), index=False)
        rep["arms"][arm] = r
        print(f"{arm:12s} scale {sc:.4f}  viol {r['violation_rate']:.4f}  median C_tot {r['median_C_tot_met']:.0f}  ({time.time() - t0:.0f}s)", flush=True)
        save_json(rep, os.path.join(out, "d1_report.json"))
    if "sund" in rep["arms"] and best in rep["arms"]:
        from orbit_ml_cr3bp import confirm_d_3d as cd
        lm = pd.read_csv(os.path.join(out, "ledger_sund.csv.gz"))
        lr = pd.read_csv(os.path.join(out, f"ledger_{best}.csv.gz"))
        rep["D1"] = cd.d1_ml_vs_rule(lm, lr)
        rep["pairwise_vs_ml"] = {}
        for arm in arms:
            if arm in ("sund",) or not os.path.exists(os.path.join(out, f"ledger_{arm}.csv.gz")):
                continue
            la = pd.read_csv(os.path.join(out, f"ledger_{arm}.csv.gz"))
            q = cd.d1_ml_vs_rule(lm, la)
            rep["pairwise_vs_ml"][arm] = {k: q[k] for k in ("mean_log10_saving", "ci95", "n_paired", "ml_cheaper_share", "p")}
    save_json(rep, os.path.join(out, "d1_report.json"))
    print(json.dumps(proto.json_safe(rep.get("D1")), indent=1))
    print(f"wrote {out}  ({time.time() - t0:.0f}s)")


# ---------------------------------------------------------------------------------------------------------------------
# D4: conditional safety by free strata, abstention to the adaptive solver
# ---------------------------------------------------------------------------------------------------------------------

def d4_sets(target):
    """((calibration frame, refs), (evaluation frame, refs)): confirm -> replication (independent datasets); dry -> two halves of D_select_D."""
    from orbit_ml_cr3bp import policy_c_3d as pc
    if target == "dry":
        U, ref = load_set("select")
        cal, ev = pc.split_folds(len(U), 42)
        return (U.iloc[cal].reset_index(drop=True), ref[cal]), (U.iloc[ev].reset_index(drop=True), ref[ev])
    return load_set("confirm"), load_set("replication")


def deploy_scales(runner, dd, U, ref, raw, scales, fallback_rtol, overhead, mu=MU):
    """Per-row scale (NaN = abstain to SciPy RK45 at fallback_rtol). Returns the ledger columns (meets_tol, C_ml_int, C_ml_tot, ml_abstain)."""
    S, T = dd.states_T(U)
    ab = ~np.isfinite(scales)
    met = np.zeros(len(U), dtype=bool)
    cint = np.full(len(U), np.nan)
    if (~ab).any():
        r = dd.run_rows(runner, S[~ab], T[~ab], raw[~ab] * scales[~ab], ref[~ab])
        met[~ab], cint[~ab] = r["met"], r["evals"]
    if ab.any():
        if fallback_rtol is None:
            cint[ab] = np.nan
        else:
            ok, err, nfev = dd.adaptive_rows(S[ab], T[ab], ref[ab], fallback_rtol, "RK45", mu)
            met[ab], cint[ab] = ok & (err <= POSITION_TOL_KM), nfev
    return pd.DataFrame({"row_id": U["row_id"].to_numpy(), "meets_tol": met, "violation": ~met, "C_ml_int": cint, "C_ml_tot": cint + overhead, "ml_abstain": ab})


def cmd_d4(a):
    """D4. Scales certified per stratum on the calibration dataset (confirm: all usable rows of D_confirm_D), evaluated on an independent dataset
    (D_replication). Also the unstratified Sundman policy on the same rows (the marginal certificate) for the 'where does the risk land' comparison, and the
    equal-risk RK45 / DOP853 of each policy (tolerance matched to that policy's calibration violation rate)."""
    import joblib
    from orbit_ml_cr3bp import deploy_d as dd, strata_d as sd, confirm_d_3d as cd, policy_c_3d as pc
    out = make_run_dir(a.out_dir, "d4", a)
    t0 = time.time()
    (Uc, rc_), (Ue, re_) = d4_sets(a.target)
    runner = _runner()
    fz = joblib.load(os.path.join(ROOT, FROZEN_SUND))
    OH = float(fz["spec"]["config"]["overhead_rhs"])
    raw_c, raw_e = (10.0 ** pc.predict_raw_c(fz["models"], U)["rk4s"]["logdt"] for U in (Uc, Ue))
    Sc, Tc = dd.states_T(Uc)
    cells_c, cells_e = sd.base_cells(Uc), sd.base_cells(Ue)
    smap, labs = sd.build_strata_map(cells_c)
    lab_c, lab_e = sd.assign_strata(cells_c, smap, labs), sd.assign_strata(cells_e, smap, labs)
    cert, G = sd.certify_strata(runner, Sc, Tc, raw_c, rc_, lab_c)
    marg = dd.certify_scale(runner, Sc, Tc, raw_c, rc_)
    rt0, _rep0 = dd.comparator_rtols(Uc, rc_, None, n_jobs=a.n_jobs)
    fb = rt0[("RK45", "cal")]
    rep = {"target": a.target, "n_cal": int(len(Uc)), "n_eval": int(len(Ue)), "n_strata": G, "fallback_rtol_RK45_cal": fb, "overhead_rhs": OH,
           "strata": {g: {"scale": c["scale"], "n_cal": c["n_cal"], "reason": c["reason"]} for g, c in cert.items()}, "marginal_scale": marg["chosen_scale"]}
    print(f"{G} strata; abstaining: {[g for g, c in cert.items() if c['scale'] is None]}; marginal scale {marg['chosen_scale']}", flush=True)
    arms = {"strat": {"sc_c": np.array([cert[l]["scale"] if cert[l]["scale"] is not None else np.nan for l in lab_c], dtype=float),
                      "sc_e": np.array([cert[l]["scale"] if cert[l]["scale"] is not None else np.nan for l in lab_e], dtype=float)},
            "marginal": {"sc_c": np.full(len(Uc), marg["chosen_scale"]), "sc_e": np.full(len(Ue), marg["chosen_scale"])}}
    ledgers = {}
    for arm, d in arms.items():
        cal_led = deploy_scales(runner, dd, Uc, rc_, raw_c, d["sc_c"], fb, OH)
        cal_rate = float(cal_led["violation"].mean())
        rt, rtrep = dd.comparator_rtols(Uc, rc_, cal_rate, n_jobs=a.n_jobs)
        ev_led = deploy_scales(runner, dd, Ue, re_, raw_e, d["sc_e"], fb, OH)
        comp = dd.comparator_frame(Ue, re_, rt, n_jobs=a.n_jobs)
        led = ev_led.merge(comp, on="row_id")
        led["stratum"] = lab_e
        led["moon_class"] = cells_e["mc"].to_numpy()
        led["r2_band"] = Ue["r2_band"].to_numpy()
        led["regime"] = Ue["regime"].to_numpy()
        led["h_band"] = Ue["h_band"].to_numpy()
        led.to_csv(os.path.join(out, f"ledger_{arm}.csv.gz"), index=False)
        ledgers[arm] = led
        rep[arm] = {"cal_violation_rate": cal_rate, "rtols": {f"{m}|{t}": v for (m, t), v in rt.items()}, "violation_rate": float(led["violation"].mean()),
                    "abstain_share": float(led["ml_abstain"].mean())}
        for g in ("stratum", "moon_class", "r2_band", "regime", "h_band"):
            sd.stratum_table(led, g).to_csv(os.path.join(out, f"table_{arm}_by_{g}.csv"), index=False)
        print(f"{arm}: cal rate {cal_rate:.4f}, eval violations {rep[arm]['violation_rate']:.4f}, abstain {rep[arm]['abstain_share']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    tab = sd.stratum_table(ledgers["strat"], "stratum")
    rep["D4"] = cd.d4_conditional(tab, ledgers["strat"], sd.UPPER_LEVEL, sd.MIN_N_REPORT)
    rep["D4_marginal_policy_for_contrast"] = cd.d4_conditional(sd.stratum_table(ledgers["marginal"], "stratum"), ledgers["marginal"], sd.UPPER_LEVEL, sd.MIN_N_REPORT)
    save_json(rep, os.path.join(out, "d4_report.json"))
    print(json.dumps(proto.json_safe(rep["D4"]), indent=1))
    print(f"wrote {out}  ({time.time() - t0:.0f}s)")


# ---------------------------------------------------------------------------------------------------------------------
# D5: the frozen policy on JPL periodic-orbit families
# ---------------------------------------------------------------------------------------------------------------------

def _label_job(a):
    """Oracle Sundman step ds* of one run (the Part C label: the largest ds, on the dense log grid, that meets the tolerance and does so for every finer ds down to ds/4)."""
    from orbit_ml_cr3bp.data_3d import _sundman_labels
    from orbit_ml_cr3bp.physics_3d import reference_trajectory_dense_3d
    state, T, mu, tol_km, tau_c = a
    sol, reached = reference_trajectory_dense_3d(list(state), T, mu)
    if not reached:
        return float("nan")
    r = _sundman_labels(list(state), T, mu, tol_km, (sol, reached), tau_c)
    return float(r["dt_rk4s"]) if r["rk4s_reason"] == 0 else float("nan")


def dd_run(runner, S, T, ds, ref):
    from orbit_ml_cr3bp import deploy_d as dd
    return dd.run_rows(runner, S, T, ds, ref)


def cmd_d5(a):
    """D5. The frozen Part C Sundman policy at the frozen Part C scale, and SciPy RK45 / DOP853 at the Part C certified and equal-risk rtols, on evenly
    spaced members of the JPL Earth-Moon families (target dry: one sample, confirm: the disjoint other), at 1, 2, 5, 10, 20 periods (<= 60 TU).
    Nothing is calibrated here. Members that collide, do not close, or whose reference stops are dropped BEFORE any policy run and listed."""
    import joblib
    from multiprocessing import Pool
    from orbit_ml_cr3bp import periodic_d as pdd, compiled_solvers_d as cs, confirm_d_3d as cd
    from orbit_ml_cr3bp.config import collision_radii
    out = make_run_dir(a.out_dir, "d5", a)
    t0 = time.time()
    which = "dry" if a.target == "dry" else "confirm"
    jpl = os.path.join(ROOT, "data_d", "jpl")
    mus, members = set(), []
    for stream, group in pdd.STREAM_TO_GROUP.items():
        df, mu_j = pdd.load_stream(os.path.join(jpl, stream + ".json"))
        if df is None or not len(df):
            print(f"{stream}: no data", flush=True)
            continue
        mus.add(mu_j)
        idx = pdd.select_indices(len(df), a.n_per_stream, which)
        sub = df.iloc[idx].copy()
        sub["stream"], sub["group"], sub["idx"] = stream, group, idx
        members.append(sub)
    if len(mus) != 1:
        raise RuntimeError(f"JPL responses disagree on the mass ratio: {mus}")
    mu = mus.pop()
    M = pd.concat(members, ignore_index=True)
    print(f"JPL mu {mu!r} vs config MU {MU!r} (relative difference {abs(mu - MU) / MU:.2e}); {len(M)} sampled members from {M['stream'].nunique()} streams", flush=True)
    rc1, rc2 = collision_radii(mu)
    jobs = [(M.loc[i, ["x", "y", "z", "vx", "vy", "vz"]].to_numpy(dtype=float), float(M.loc[i, "period"]), mu) for i in M.index]
    with Pool(a.n_jobs) as pool:
        diag = pool.map(pdd.member_diagnostics, jobs, chunksize=4)
    D = pd.DataFrame(diag)
    M = pd.concat([M, D], axis=1)
    M["quality_ok"] = M["reached"] & (M["closure"] <= pdd.CLOSURE_TOL) & (M["min_r1"] > rc1) & (M["min_r2"] > rc2)
    M.to_csv(os.path.join(out, "members.csv"), index=False)
    print(f"members kept {int(M['quality_ok'].sum())} of {len(M)}; dropped by reason: not reached {int((~M['reached']).sum())}, closure {int((M['reached'] & (M['closure'] > pdd.CLOSURE_TOL)).sum())}, "
          f"collision {int((M['reached'] & ((M['min_r1'] <= rc1) | (M['min_r2'] <= rc2))).sum())}", flush=True)
    runs = pdd.build_runs(M[M["quality_ok"]], mu)
    runs = pdd.attach_references(runs, mu, a.n_jobs)
    print(f"{len(runs)} runs, reference reached {int(runs['ref_reached'].sum())}, trusted {int(runs['ref_ok'].sum())}  ({time.time() - t0:.0f}s)", flush=True)
    fz = joblib.load(os.path.join(ROOT, FROZEN_SUND))
    pc_rep = json.load(open(os.path.join(ROOT, "part_c", "results", "confirm", "sund", "apply_policy_report.json")))
    scale = float(pc_rep["scale"])
    rtols = {tuple(k.split("|")): v for k, v in pc_rep["rtols"].items()}
    policy = cs.SundmanPolicy(fz["models"], mu=mu)
    ok_ref = runs["ref_reached"].to_numpy()
    sub = runs[ok_ref].copy()
    r, raw = pdd.run_policy(sub, policy, scale, mu)
    OH = float(fz["spec"]["config"]["overhead_rhs"])
    sub["ml_ok"], sub["ml_err_km"], sub["ml_evals"], sub["meets_tol"] = r["ok"], r["err"], r["evals"], r["met"]
    sub["violation"] = ~sub["meets_tol"]
    sub["C_ml_tot"] = sub["ml_evals"] + OH
    sub["raw_ds"] = raw
    sub["cluster"] = sub["stream"].str.replace("_N", "", regex=False).str.replace("_S", "", regex=False) + "#" + sub["idx"].astype(str)      # one orbit (all horizons) and its N/S mirror image
    comp = pdd.run_comparators(sub, rtols, mu, a.n_jobs)
    sub = pd.concat([sub, comp], axis=1)
    sub.to_csv(os.path.join(out, "d5_runs.csv.gz"), index=False)
    test = sub[sub["in_range"] & sub["ref_ok"]].copy()
    rep = {"target": a.target, "mu_jpl": mu, "mu_config": MU, "scale_part_c": scale, "rtols_part_c": {f"{m}|{t}": v for (m, t), v in rtols.items()},
           "n_members_sampled": int(len(M)), "n_members_kept": int(M["quality_ok"].sum()), "n_runs": int(len(runs)), "n_ref_reached": int(ok_ref.sum()),
           "n_in_range_trusted": int(len(test)), "tolerance_km_project": POSITION_TOL_KM, "tolerance_km_in_jpl_units": POSITION_TOL_KM * 389703.264829278 / 384400.0}
    if len(test):
        rep["D5"] = cd.d5_ood(test)
    def tab(d, col):
        rows = []
        for g, x in d.groupby(col):
            k = int(x["violation"].sum())
            r_ = {"group": g, "n": int(len(x)), "violations": k, "rate": k / len(x), "median_C_tot_met": float(x.loc[x["meets_tol"], "C_ml_tot"].median()) if x["meets_tol"].any() else None}
            for c in ("RK45_cal", "RK45_eqr", "DOP853_eqr"):
                r_[f"{c}_rate"] = float((~x[f"{c}_ok"].astype(bool)).mean())
                has = x[f"{c}_nfev"].notna()
                r_[f"share_cheaper_{c}"] = float((x["meets_tol"] & has & (x["C_ml_tot"] < x[f"{c}_nfev"])).sum() / max(has.sum(), 1))
            rows.append(r_)
        return pd.DataFrame(rows)
    for col in ("group", "stab_band", "k"):
        tab(test, col).to_csv(os.path.join(out, f"table_by_{col}.csv"), index=False)
    tab(sub[~sub["in_range"] & sub["ref_ok"]], "group").to_csv(os.path.join(out, "table_extrapolated_by_group.csv"), index=False)
    rep["not_trusted_reference"] = {"in_range_runs": int((sub["in_range"] & ~sub["ref_ok"]).sum()), "of": int(sub["in_range"].sum())}
    # explanatory (reported, not tested): the oracle step on the same runs, and a scale recalibrated on the other (dry) sample
    if a.labels and len(test):
        with Pool(a.n_jobs) as pool:
            lab = pool.map(_label_job, [(r_[["x", "y", "z", "vx", "vy", "vz"]].to_numpy(dtype=float), float(r_["sim_time"]), mu, POSITION_TOL_KM, 2.0)
                                        for _, r_ in test.iterrows()], chunksize=8)
        test["oracle_ds"] = lab
        test["log10_raw_over_oracle"] = np.log10(test["raw_ds"] / test["oracle_ds"])
        test.to_csv(os.path.join(out, "d5_test_runs_with_labels.csv.gz"), index=False)
        rep["oracle_label"] = {"n": int(test["oracle_ds"].notna().sum()), "median_log10_raw_over_oracle": float(test["log10_raw_over_oracle"].median()),
                               "share_policy_step_above_oracle": float((test["log10_raw_over_oracle"] > 0).mean()),
                               "by_group_median": {g: float(x["log10_raw_over_oracle"].median()) for g, x in test.groupby("group")}}
        print("oracle label:", json.dumps(proto.json_safe(rep["oracle_label"]))[:600], flush=True)
    if a.recal_from and len(test):
        from orbit_ml_cr3bp import policy_c_3d as pc
        dry = pd.read_csv(os.path.join(a.recal_from, "d5_runs.csv.gz"))
        dry = dry[dry["in_range"] & dry["ref_ok"]]
        runner = cs.SundmanRunner(mu)
        grid = tuple(0.1 * 2.0 ** (k / 8.0) for k in range(-48, 27)) + (1.0,)
        Sd, Td = dry[["x", "y", "z", "vx", "vy", "vz"]].to_numpy(dtype=float), dry["sim_time"].to_numpy(dtype=float)
        refd = dry[["ref_x", "ref_y", "ref_z"]].to_numpy(dtype=float)
        cache = {}

        def viol_at(sc):
            if sc not in cache:
                cache[sc] = int((~dd_run(runner, Sd, Td, dry["raw_ds"].to_numpy() * sc, refd)["met"]).sum())
            return cache[sc], len(dry)
        cert = pc.certify_sequence(viol_at, grid, 0.05, 0.1, "fixed_sequence")
        sc = cert["chosen_scale"]
        rec = {"scale": sc, "n_dry_runs": int(len(dry)), "note": "runs of one orbit are correlated, so this certificate is optimistic; descriptive"}
        if sc is not None:
            St, Tt = test[["x", "y", "z", "vx", "vy", "vz"]].to_numpy(dtype=float), test["sim_time"].to_numpy(dtype=float)
            r_ = dd_run(runner, St, Tt, test["raw_ds"].to_numpy() * sc, test[["ref_x", "ref_y", "ref_z"]].to_numpy(dtype=float))
            test["recal_met"], test["recal_C_tot"] = r_["met"], r_["evals"] + OH
            rec["test_violation_rate"] = float((~r_["met"]).mean())
            rec["test_violation_rate_family_balanced"] = float(np.mean([(~x["recal_met"]).mean() for _, x in test.groupby("group")]))
            for c in ("RK45_eqr", "DOP853_eqr", "RK45_cal"):
                has = test[f"{c}_nfev"].notna()
                rec[f"share_cheaper_{c}"] = float((test["recal_met"] & has & (test["recal_C_tot"] < test[f"{c}_nfev"])).sum() / max(has.sum(), 1))
            rec["median_C_tot_met"] = float(test.loc[test["recal_met"], "recal_C_tot"].median())
        rep["recalibrated_on_dry_sample"] = rec
        print("recalibrated:", json.dumps(proto.json_safe(rec)), flush=True)
    save_json(rep, os.path.join(out, "d5_report.json"))
    print(json.dumps(proto.json_safe(rep.get("D5", rep)), indent=1)[:3000])
    print(f"wrote {out}  ({time.time() - t0:.0f}s)")


# ---------------------------------------------------------------------------------------------------------------------
# D2: compiled wall-clock and batch benchmark
# ---------------------------------------------------------------------------------------------------------------------

def lane_efficiency(work, widths=(2, 4, 8, 32, 128), order="random", seed=0):
    """Share of lock-step lane-steps that do useful work when `work` (steps per orbit) is run in groups of `width` lanes that all advance until the longest
    orbit of the group is done: sum(work) / sum(width * max(work in group)). order 'sorted': orbits sorted by work first (the best a scheduler that knows the
    work could do)."""
    w = np.asarray(work, dtype=float)
    w = w[np.isfinite(w) & (w > 0)]
    if order == "random":
        w = w[np.random.RandomState(seed).permutation(len(w))]
    else:
        w = np.sort(w)
    res = {}
    for L in widths:
        n = (len(w) // L) * L
        g = w[:n].reshape(-1, L)
        res[L] = float(g.sum() / (L * g.max(axis=1)).sum())
    return res


def cmd_d2(a):
    """D2. The Sundman policy (features + 25-tree model + certified scale + RK4 in Sundman time) as ONE compiled pipeline against compiled adaptive solvers
    (Dormand-Prince 5(4) with a PI controller as the primary comparator, with SciPy's controller, and DOP853), all with the same right-hand side arithmetic and the same
    event checks, at tolerances recalibrated for the compiled solvers on the same 4,000 calibration rows (equal risk to the policy, and certified). Two kernel variants:
    F = micro-optimized kernels for BOTH arms (one reciprocal per body, no allocation in the landing steps; the primary D2), R = the kernels whose arithmetic is Part C's
    (policy) and SciPy's (comparators), reported for transparency. (a) single-thread wall time per orbit on the evaluation rows; (b) throughput of batches of 10^3 ... 10^5
    orbits, one thread and all threads; (c) modelled lane efficiency of a lock-step (SIMD / GPU) batch from the real step counts. RUN ON AN IDLE MACHINE."""
    import joblib
    import numba
    from scipy.stats import spearmanr
    from orbit_ml_cr3bp import compiled_solvers_d as cs, deploy_d as dd, confirm_d_3d as cd
    out = make_run_dir(a.out_dir, "d2", a)
    t0 = time.time()
    load0 = os.getloadavg()
    U, ref = load_set(TARGET_SET[a.target])
    S, T = dd.states_T(U)
    cal, ev = standard_split(U)
    fz = joblib.load(os.path.join(ROOT, FROZEN_SUND))
    OH = float(fz["spec"]["config"]["overhead_rhs"])
    spR, spF = cs.SundmanPolicy(fz["models"], MU, fast=False), cs.SundmanPolicy(fz["models"], MU, fast=True)
    svR, svF = cs.Solvers(MU, fast=False), cs.Solvers(MU, fast=True)
    raw = spR.raw_ds(S, T)
    c = dd.certify_scale(spR, S[cal], T[cal], raw[cal], ref[cal])
    scale = c["chosen_scale"]
    cal_rate = c["violations_by_scale"][scale] / len(cal)
    rep = {"target": a.target, "hardware": {"cpu": os.popen("sysctl -n machdep.cpu.brand_string").read().strip(), "ncpu": os.cpu_count(),
                                            "numba_threads": numba.config.NUMBA_NUM_THREADS, "numba": numba.__version__, "python": sys.version.split()[0]},
           "load_avg_start": load0, "scale": scale, "policy_cal_violation_rate": cal_rate, "overhead_rhs": OH, "n_cal": int(len(cal)), "n_eval": int(len(ev))}
    # recalibrate the compiled adaptive solvers (both kernel variants; the tolerances differ only at rounding level)
    rt = {"R": {}, "F": {}}
    for var, sv in (("R", svR), ("F", svF)):
        for name in cs.Solvers.NAMES:
            r = cs.certify_rtol_compiled(sv, name, S[cal], T[cal], ref[cal], equal_risk_target=cal_rate)
            rt[var][(name, "cal")], rt[var][(name, "eqr")] = r["chosen_rtol"], r["equal_risk_rtol"]
            rep.setdefault("rtol_calibration", {})[f"{name}|{var}"] = {"certified_rtol": r["chosen_rtol"], "equal_risk_rtol": r["equal_risk_rtol"], "chain": r["chain"]}
            print(f"{var} {name}: certified rtol {r['chosen_rtol']}, equal-risk rtol {r['equal_risk_rtol']}  ({time.time() - t0:.0f}s)", flush=True)
    # outcomes (risk and RHS counts) on the evaluation rows
    Se, Te, re_ = S[ev], T[ev], ref[ev]
    outcome, arrays = {}, {}
    for var, sp in (("R", spR), ("F", spF)):
        okp, yp, evp, _ = sp.run(Se, Te, scale)
        errp = np.where(okp, sp.errors_km(yp, re_), np.nan)
        metp = okp & (errp <= POSITION_TOL_KM)
        outcome[f"policy|{var}"] = {"violation_rate": float((~metp).mean()), "median_total_evals_met": float(np.median(evp[metp] + OH)), "mean_evals": float(evp.mean())}
        arrays[f"policy_{var}_evals"], arrays[f"policy_{var}_met"] = evp, metp
    metp = arrays["policy_F_met"]
    evp = arrays["policy_F_evals"]
    for name in cs.Solvers.NAMES:
        for tag in ("cal", "eqr"):
            rtv = rt["F"][(name, tag)]
            if rtv is None:
                continue
            ok, y, nf, _ = svF.run(name, Se, Te, rtv)
            err = np.where(ok, svF.errors_km(y, re_), np.nan)
            met = ok & (err <= POSITION_TOL_KM)
            outcome[f"{name}_{tag}|F"] = {"rtol": rtv, "violation_rate": float((~met).mean()), "median_nfev_met": float(np.median(nf[met])), "mean_nfev": float(nf.mean()),
                                          "share_policy_cheaper_nfev": float((metp & met & (evp + OH < nf)).sum() / len(Se))}
            arrays[f"{name}_{tag}_nfev"], arrays[f"{name}_{tag}_met"] = nf, met
    ok, y, nf, _ = svR.run("DP54_PI", Se, Te, rt["R"][("DP54_PI", "eqr")])
    arrays["DP54_PI_eqr_R_nfev"] = nf
    rep["outcome"] = outcome
    print(json.dumps(proto.json_safe(outcome), indent=1)[:3000], flush=True)
    # (a) single-thread wall time per orbit
    A, B, E3, E5 = svF.A, svF.B, svF.E3, svF.E5
    sF = dict(mu=MU, c_l1=spF.c_l1, fr=spF.cp.fr, tau_c=spF.tau_c, ms=spF.max_steps, rc1=spF.rc1, rc2=spF.rc2, rej=spF.rej)
    n_time = min(len(ev), a.n_time) if a.n_time else len(ev)
    pick = np.random.RandomState(0).permutation(len(ev))[:n_time]
    arms = ["policy", "DP54_PI_eqr", "DP54_scipy_eqr", "DOP853_eqr", "DP54_PI_cal", "policy_R", "DP54_PI_eqr_R"]
    tns = {k: np.full(len(ev), np.nan) for k in arms}
    cs._loop_policy(Se[0], Te[0], 2, scale, MU, sF["c_l1"], sF["fr"], sF["tau_c"], sF["ms"], sF["rc1"], sF["rc2"], sF["rej"])
    cs._loop_policy_fast(Se[0], Te[0], 2, scale, MU, sF["c_l1"], sF["fr"], sF["tau_c"], sF["ms"], sF["rc1"], sF["rc2"], sF["rej"])
    for K in (cs.KF, cs.KR):
        K.loop_dp54(Se[0], Te[0], 2, 1e-6, cs.ATOL_RATIO, MU, cs.CTRL_PI, cs.MAX_NFEV, sF["rc1"], sF["rc2"], sF["rej"])
    cs.KF.loop_dop853(Se[0], Te[0], 2, 1e-6, cs.ATOL_RATIO, MU, A, B, E3, E5, cs.MAX_NFEV, sF["rc1"], sF["rc2"], sF["rej"])
    tn = a.target_ns
    for n_done, i in enumerate(pick):
        st, Ti = Se[i], Te[i]
        P = (scale, MU, sF["c_l1"], sF["fr"], sF["tau_c"], sF["ms"], sF["rc1"], sF["rc2"], sF["rej"])
        tns["policy"][i] = cs.time_per_orbit(lambda reps: cs._loop_policy_fast(st, Ti, reps, *P), tn)
        tns["DP54_PI_eqr"][i] = cs.time_per_orbit(lambda reps: cs.KF.loop_dp54(st, Ti, reps, rt["F"][("DP54_PI", "eqr")], cs.ATOL_RATIO, MU, cs.CTRL_PI, cs.MAX_NFEV, sF["rc1"], sF["rc2"], sF["rej"]), tn)
        tns["DP54_scipy_eqr"][i] = cs.time_per_orbit(lambda reps: cs.KF.loop_dp54(st, Ti, reps, rt["F"][("DP54_scipy", "eqr")], cs.ATOL_RATIO, MU, cs.CTRL_SCIPY, cs.MAX_NFEV, sF["rc1"], sF["rc2"], sF["rej"]), tn)
        tns["DOP853_eqr"][i] = cs.time_per_orbit(lambda reps: cs.KF.loop_dop853(st, Ti, reps, rt["F"][("DOP853", "eqr")], cs.ATOL_RATIO, MU, A, B, E3, E5, cs.MAX_NFEV, sF["rc1"], sF["rc2"], sF["rej"]), tn)
        if rt["F"][("DP54_PI", "cal")] is not None:
            tns["DP54_PI_cal"][i] = cs.time_per_orbit(lambda reps: cs.KF.loop_dp54(st, Ti, reps, rt["F"][("DP54_PI", "cal")], cs.ATOL_RATIO, MU, cs.CTRL_PI, cs.MAX_NFEV, sF["rc1"], sF["rc2"], sF["rej"]), tn)
        tns["policy_R"][i] = cs.time_per_orbit(lambda reps: cs._loop_policy(st, Ti, reps, *P), tn)
        tns["DP54_PI_eqr_R"][i] = cs.time_per_orbit(lambda reps: cs.KR.loop_dp54(st, Ti, reps, rt["R"][("DP54_PI", "eqr")], cs.ATOL_RATIO, MU, cs.CTRL_PI, cs.MAX_NFEV, sF["rc1"], sF["rc2"], sF["rej"]), tn)
        if (n_done + 1) % 2000 == 0:
            print(f"  timed {n_done + 1}/{n_time} orbits ({time.time() - t0:.0f}s)", flush=True)
    tab = pd.DataFrame({**{f"t_ns_{k}": v for k, v in tns.items()}, "row_id": U.iloc[ev]["row_id"].to_numpy(), "sim_time": Te, "regime": U.iloc[ev]["regime"].to_numpy(),
                        "h_band": U.iloc[ev]["h_band"].to_numpy(), **arrays})
    tab.to_csv(os.path.join(out, "per_orbit_times.csv.gz"), index=False)
    tests = {}
    pairs = {"DP54_PI_eqr": ("policy", "DP54_PI_eqr", "DP54_PI_eqr_nfev", "policy_F_evals"), "DP54_scipy_eqr": ("policy", "DP54_scipy_eqr", "DP54_scipy_eqr_nfev", "policy_F_evals"),
             "DOP853_eqr": ("policy", "DOP853_eqr", "DOP853_eqr_nfev", "policy_F_evals"), "DP54_PI_cal": ("policy", "DP54_PI_cal", "DP54_PI_cal_nfev", "policy_F_evals"),
             "DP54_PI_eqr|reference_kernels": ("policy_R", "DP54_PI_eqr_R", "DP54_PI_eqr_R_nfev", "policy_R_evals")}
    for key, (pa, ca, nfa, evk) in pairs.items():
        m = np.isfinite(tns[ca]) & np.isfinite(tns[pa])
        tests[key] = cd.d2_throughput(tns[pa][m], tns[ca][m])
        lr_t, lr_n = np.log10(tns[ca][m] / tns[pa][m]), np.log10(arrays[nfa][m] / (arrays[evk][m] + OH))
        good = np.isfinite(lr_t) & np.isfinite(lr_n)
        tests[key]["spearman_log_time_ratio_vs_log_rhs_ratio"] = float(spearmanr(lr_t[good], lr_n[good]).statistic)
        tests[key]["share_faster_wall"], tests[key]["share_cheaper_rhs"] = float((lr_t[good] > 0).mean()), float((lr_n[good] > 0).mean())
        tests[key]["ns_per_rhs_policy_median"] = float(np.median(tns[pa][m] / (arrays[evk][m] + OH)))
        tests[key]["ns_per_rhs_comparator_median"] = float(np.median(tns[ca][m] / arrays[nfa][m]))
        # by horizon band and regime
        by = {}
        for col in ("h_band", "regime"):
            for g, idx in tab[m].groupby(col).groups.items():
                ii = np.asarray(tab[m].index.get_indexer(idx))
                by[f"{col}={g}"] = {"n": int(len(ii)), "throughput_ratio": float(tns[ca][m][ii].sum() / tns[pa][m][ii].sum()), "share_policy_faster": float((tns[pa][m][ii] < tns[ca][m][ii]).mean())}
        tests[key]["by_group"] = by
    rep["D2_single_thread"] = tests
    rep["D2"] = tests["DP54_PI_eqr"]
    print(json.dumps(proto.json_safe({k: {kk: vv for kk, vv in v.items() if kk != "by_group"} for k, v in tests.items()}), indent=1), flush=True)
    # (c) modelled lane efficiency from real step counts
    steps_pol = arrays["policy_F_evals"] / 4.0
    steps_ad = arrays["DP54_PI_eqr_nfev"] / 6.0
    rep["lane_efficiency_modelled"] = {"policy_random": lane_efficiency(steps_pol), "policy_sorted": lane_efficiency(steps_pol, order="sorted"),
                                       "DP54_PI_eqr_random": lane_efficiency(steps_ad), "DP54_PI_eqr_sorted": lane_efficiency(steps_ad, order="sorted")}
    # (b) batches
    numba.set_parallel_chunksize(a.chunk)
    nthreads = numba.config.NUMBA_NUM_THREADS
    rng = np.random.RandomState(1)
    batches = {}
    for size in a.sizes:
        idx = rng.randint(0, len(ev), size)
        Sb, Tb = np.ascontiguousarray(Se[idx]), np.ascontiguousarray(Te[idx])
        for thr in (1, nthreads):
            numba.set_num_threads(thr)
            fns = {"policy": lambda: cs.policy_batch_fast_par(Sb, Tb, scale, MU, sF["c_l1"], sF["fr"], sF["tau_c"], sF["ms"], sF["rc1"], sF["rc2"], sF["rej"]),
                   "DP54_PI_eqr": lambda: cs.KF.dp54_batch_par(Sb, Tb, rt["F"][("DP54_PI", "eqr")], cs.ATOL_RATIO, MU, cs.CTRL_PI, cs.MAX_NFEV, sF["rc1"], sF["rc2"], sF["rej"]),
                   "DP54_scipy_eqr": lambda: cs.KF.dp54_batch_par(Sb, Tb, rt["F"][("DP54_scipy", "eqr")], cs.ATOL_RATIO, MU, cs.CTRL_SCIPY, cs.MAX_NFEV, sF["rc1"], sF["rc2"], sF["rej"]),
                   "DOP853_eqr": lambda: cs.KF.dop853_batch_par(Sb, Tb, rt["F"][("DOP853", "eqr")], cs.ATOL_RATIO, MU, A, B, E3, E5, cs.MAX_NFEV, sF["rc1"], sF["rc2"], sF["rej"])}
            if size == a.sizes[1] if len(a.sizes) > 1 else True:
                fns["policy_R"] = lambda: cs.policy_batch_par(Sb, Tb, scale, MU, sF["c_l1"], sF["fr"], sF["tau_c"], sF["ms"], sF["rc1"], sF["rc2"], sF["rej"])
                fns["DP54_PI_eqr_R"] = lambda: cs.KR.dp54_batch_par(Sb, Tb, rt["R"][("DP54_PI", "eqr")], cs.ATOL_RATIO, MU, cs.CTRL_PI, cs.MAX_NFEV, sF["rc1"], sF["rc2"], sF["rej"])
            for arm, fn in fns.items():
                fn()
                ts = []
                for _ in range(a.repeats):
                    t1 = time.perf_counter()
                    fn()
                    ts.append(time.perf_counter() - t1)
                batches[f"{arm}|{size}|{thr}"] = {"seconds": ts, "orbits_per_s_median": size / float(np.median(ts)), "orbits_per_s_min_time": size / float(np.min(ts))}
        print(f"  batch {size} done ({time.time() - t0:.0f}s)", flush=True)
    numba.set_num_threads(nthreads)
    rep["batches"] = batches
    ratios = {}
    for size in a.sizes:
        for thr in (1, nthreads):
            for comp, pol in (("DP54_PI_eqr", "policy"), ("DP54_scipy_eqr", "policy"), ("DOP853_eqr", "policy"), ("DP54_PI_eqr_R", "policy_R")):
                if f"{comp}|{size}|{thr}" not in batches:
                    continue
                tp, tc = np.array(batches[f"{pol}|{size}|{thr}"]["seconds"]), np.array(batches[f"{comp}|{size}|{thr}"]["seconds"])
                ratios[f"{comp}|{size}|{thr}"] = {"median_ratio": float(np.median(tc) / np.median(tp)), "min_ratio_over_repeats": float(tc.min() / tp.max()), "pass": bool(tc.min() > tp.max())}
    rep["batch_throughput_ratios_adaptive_over_policy"] = ratios
    rep["load_avg_end"] = os.getloadavg()
    save_json(rep, os.path.join(out, "d2_report.json"))
    print(f"wrote {out}  ({time.time() - t0:.0f}s)")


def cmd_lock(a):
    """Locked final run (a number, not a test): the ML policy and the best rule with the scales and comparator tolerances certified on D_confirm_D (the d1
    report of the confirmation run), on every usable row of D_lock_D, once."""
    import joblib
    from orbit_ml_cr3bp import deploy_d as dd
    from orbit_ml_cr3bp.audit_3d import clopper_pearson
    out = make_run_dir(a.out_dir, "lock", a)
    prev = json.load(open(os.path.join(a.from_dir, "d1_report.json")))
    U, ref = load_set(TARGET_SET["lock"])
    S, T = dd.states_T(U)
    runner = _runner()
    arms = arm_paths()
    rep = {"from": a.from_dir, "n_usable": int(len(U)), "arms": {}}
    for arm in ("sund", prev["best_rule"]):
        fz = joblib.load(os.path.join(ROOT, arms[arm]))
        pr = prev["arms"][arm]
        raw = 10.0 ** raw_log_ds(fz["models"], U)
        run = dd.run_rows(runner, S, T, raw * pr["scale"], ref)
        led = dd.score(U, run, pr["overhead_rhs"])
        rt = {tuple(k.split("|")): v for k, v in pr["rtols"].items()}
        led = led.merge(dd.comparator_frame(U, ref, rt, n_jobs=a.n_jobs), on="row_id")
        k = int(led["violation"].sum())
        r = {"scale": pr["scale"], "violations": k, "violation_rate": k / len(U), "viol_cp95": list(clopper_pearson(k, len(U))),
             "median_C_tot_met": float(led.loc[led["meets_tol"], "C_ml_tot"].median())}
        for cname in ("RK45_cal", "DOP853_cal", "RK45_eqr", "DOP853_eqr"):
            has = led[f"{cname}_nfev"].notna()
            r[f"share_cheaper_{cname}"] = float((led["meets_tol"] & has & (led["C_ml_tot"] < led[f"{cname}_nfev"])).sum() / max(has.sum(), 1))
        rep["arms"][arm] = r
        led.to_csv(os.path.join(out, f"ledger_{arm}.csv.gz"), index=False)
        print(arm, json.dumps(proto.json_safe(r)), flush=True)
    save_json(rep, os.path.join(out, "lock_report.json"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("refs")
    sp.add_argument("--sets", nargs="+", default=list(SEEDS))
    sp.set_defaults(fn=cmd_refs)
    sp = sub.add_parser("overheads")
    sp.add_argument("--set", default="select")
    sp.add_argument("--repeats", type=int, default=7)
    sp.add_argument("--out_dir", default="out_d/overheads")
    sp.set_defaults(fn=cmd_overheads)
    sp = sub.add_parser("select_rules")
    sp.add_argument("--quantiles", type=float, nargs="+", default=list(__import__("orbit_ml_cr3bp.rules_d", fromlist=["x"]).RULE_QUANTILES))
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--out_dir", default="out_d/select_rules")
    sp.set_defaults(fn=cmd_select_rules)
    sp = sub.add_parser("freeze_rule")
    sp.add_argument("--rule", default=None)
    sp.add_argument("--q", type=float, default=None)
    sp.add_argument("--out_name", default=None)
    sp.set_defaults(fn=cmd_freeze_rule)
    sp = sub.add_parser("freeze_lin33")
    sp.add_argument("--q", type=float, default=0.1)
    sp.set_defaults(fn=cmd_freeze_lin33)
    sp = sub.add_parser("d1")
    sp.add_argument("--target", choices=list(TARGET_SET), default="dry")
    sp.add_argument("--out_dir", default="out_d/d1")
    sp.add_argument("--n_jobs", type=int, default=8)
    sp.set_defaults(fn=cmd_d1)
    sp = sub.add_parser("d4")
    sp.add_argument("--target", choices=["dry", "confirm"], default="dry")
    sp.add_argument("--out_dir", default="out_d/d4")
    sp.add_argument("--n_jobs", type=int, default=8)
    sp.set_defaults(fn=cmd_d4)
    sp = sub.add_parser("d5")
    sp.add_argument("--target", choices=["dry", "confirm"], default="dry")
    sp.add_argument("--n_per_stream", type=int, default=__import__("orbit_ml_cr3bp.periodic_d", fromlist=["x"]).N_PER_STREAM)
    sp.add_argument("--labels", action="store_true", help="also compute the oracle Sundman step of every test run (explanatory)")
    sp.add_argument("--recal_from", default=None, help="d5 run folder of the OTHER sample: certify a scale there and evaluate it on this sample (explanatory)")
    sp.add_argument("--out_dir", default="out_d/d5")
    sp.add_argument("--n_jobs", type=int, default=8)
    sp.set_defaults(fn=cmd_d5)
    sp = sub.add_parser("d2")
    sp.add_argument("--target", choices=["dry", "confirm"], default="dry")
    sp.add_argument("--out_dir", default="out_d/d2")
    sp.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000, 100000])
    sp.add_argument("--repeats", type=int, default=7)
    sp.add_argument("--chunk", type=int, default=16)
    sp.add_argument("--target_ns", type=float, default=5.0e5)
    sp.add_argument("--n_time", type=int, default=0, help="time only this many randomly chosen evaluation orbits (0 = all)")
    sp.set_defaults(fn=cmd_d2)
    sp = sub.add_parser("lock")
    sp.add_argument("--from_dir", required=True, help="the d1 run folder of the confirmation run")
    sp.add_argument("--out_dir", default="out_d/lock")
    sp.add_argument("--n_jobs", type=int, default=8)
    sp.set_defaults(fn=cmd_lock)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
