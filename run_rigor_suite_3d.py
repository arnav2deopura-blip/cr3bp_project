"""
Subcommands (all expect an accuracy-labeled CSV from generate_3d_data.py --label_mode accuracy
or accuracy_monotone; datasets generated before the label-pipeline fixes trigger a warning):

  gate        train, certify per-integrator step scales and a per-method adaptive rtol on held-out
              calibration orbits, run the matched-accuracy baseline (ML cost INCLUDES the FTLE feature
              and prediction; adaptive solvers both oracle-tuned "_spec" and calibrated "_cal"), apply the
              pre-registered pass/fail criteria, regime breakdown, lockstep ensemble cost model
  proxy       Jacobi-drift vs position-error study and reference-free error predictor
  audit       multi-seed metrics with CIs, position-error audit, outcome-calibrated scales (+ optional
              calibration-stability repeats), optional tolerance sensitivity / dense sweep / halo validation
  fidelity    CR3BP vs real ephemeris (needs network + astroquery; run from the folder containing
              validate_on_capstone.py)
  mu          within-mu, cross-mu transfer and pooled (mu as a feature) evaluation on several datasets
  replicate   same-mu independent datasets (different --base_seed): within and train-on-A/test-on-B
  ablate      feature ablation (full vs no_ftle vs state_only vs leave-one-out) with paired deltas
  crash       crash/escape (reference-divergent) classifier, with and without the FTLE feature
  errmodel    error-model recommender (predict log10 error, choose dt with a conformal bound)
  convergence convergence-order check against the textbook orders of the three integrators
  lock        draw the locked final test set once (stores row ids + CSV hash)
  final       evaluate on the locked set (logged; every call is recorded in --final_log)

Every command writes into its own timestamped run folder under --out_dir with a manifest.json
(arguments, package versions, git commit, input SHA-256 and generation metadata) unless --no_run_dir.
"""
import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd

from orbit_ml_cr3bp.config import MU, POSITION_TOL_KM, PHYSICS_FEATURE_COLUMNS_3D
from orbit_ml_cr3bp.modeling_3d import create_grouped_split_3d, fit_hurdle_models, predict_hurdle
from orbit_ml_cr3bp.audit_3d import (
    INTEGRATOR_NAMES, ReferenceCache, calibrate_scale_ltt, fit_and_validate_scale_ltt,
    position_error_audit_3d, summarize_audit, tolerance_sensitivity_accuracy_3d, selection_agreement,
    dense_sweep_accuracy_3d, summarize_dense_sweep, calibration_stability_3d, halo_validation_accuracy_3d,
    scale_frontier_3d, calibrate_global_dt_ltt,
)
from orbit_ml_cr3bp.fair_baseline_3d import matched_accuracy_baseline_3d, calibrate_adaptive_rtol
from orbit_ml_cr3bp.ensemble_regime_3d import lockstep_cost_model
from orbit_ml_cr3bp.proxy_analysis_3d import proxy_vs_error_table, proxy_failure_summary, fit_error_predictor
from orbit_ml_cr3bp.rigor_3d import (
    metrics_from_prediction, multiseed_evaluation, cross_mu_evaluation, replication_evaluation,
    feature_ablation_evaluation, default_feature_sets,
)
from orbit_ml_cr3bp.outcome_3d import crash_classifier_evaluation
from orbit_ml_cr3bp.error_model_3d import (
    error_ladder_table, fit_error_model, evaluate_recommender, convergence_order_study,
)
from orbit_ml_cr3bp import protocol_3d as proto

# Pre-registered decision criteria for `gate`. Set BEFORE looking at results; if they are
# changed afterwards, say so in the report. The thresholds below are unchanged from the first
# version; what changed is the cost basis (ML cost now includes the FTLE feature) and that the
# ML must also beat adaptive solvers at a CALIBRATED rtol, not only the per-orbit oracle rtol.
# ML_COMPETITIVE_ON_COST requires passing every evaluated comparison.
GATE_MIN_FRAC_ML_MEETS_TOL = 0.90
GATE_MIN_FRAC_ML_CHEAPER = 0.50
GATE_MIN_GROUP_N = 20
ADAPTIVE_METHODS = ("RK45", "DOP853")

# Pinned model for every command that fits hurdle models (see PREREGISTRATION.md, "Model"). Change these only
# together with a dated Deviations entry; `--regressor rf` is for a labelled sensitivity run only.
PINNED_REGRESSOR = "hgb_quantile"
PINNED_QUANTILE = 0.1


# Set from --ignore_lock in main(): knowingly use ALL rows of a CSV that has a lock (recorded in the manifest args).
LOCK_POLICY = {"ignore": False}


def load_accuracy_data(path, keep_divergent=False, locked_path=None, only_locked=False, check_lock=True):
    """check_lock: a CSV that has a lock file next to it (proto.find_lock_for) is only loaded with --locked pointing at
    that file, unless --ignore_lock. False for `lock` itself and for an independent --final_data CSV."""
    proto.check_dataset_pipeline(path)
    if check_lock:
        existing = proto.find_lock_for(path)
        if existing and not locked_path:
            if LOCK_POLICY["ignore"]:
                print(f"WARNING: --ignore_lock: using ALL rows of {path}, including the rows locked in {existing}. "
                      f"Do not use anything computed this way for a claim about the locked set.")
            else:
                raise ValueError(f"{path} has a locked test set ({existing}) but --locked was not given. Pass "
                                 f"--locked {existing}, or --ignore_lock to knowingly use all rows (it is recorded "
                                 f"in the manifest).")
        if existing and locked_path and os.path.realpath(existing) != os.path.realpath(locked_path):
            raise ValueError(f"--locked {locked_path} is not the lock for {path}; that is {existing}.")
    d = pd.read_csv(path)
    d["row_id"] = np.arange(len(d))
    if "ftle_nan" in d.columns:
        d = d[~d["ftle_nan"].astype(bool)]
    if "reference_divergent" in d.columns:
        d["reference_divergent"] = d["reference_divergent"].fillna(False).astype(bool)
        if not keep_divergent:
            d = d[~d["reference_divergent"]]
    d = d.reset_index(drop=True)
    for c in ("euler_censored", "rk4_censored", "leapfrog_censored"):
        if c in d.columns:
            d[c] = d[c].fillna(False).astype(bool)
    if locked_path:
        locked = proto.load_locked(locked_path, csv_path=path)
        dev, lk = proto.split_locked(d, locked)
        d = lk if only_locked else dev
    return d


def feature_columns(d):
    return [c for c in PHYSICS_FEATURE_COLUMNS_3D if c in d.columns]


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.DataFrame):
        return o.to_dict(orient="records")
    return str(o)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    print(f"Saved {path}")


def args_dict(a):
    return proto.json_safe(vars(a))


def regime_table(df, eval_data, method):
    d = df[df["ml_meets_tol"]].copy()
    if len(d) == 0:
        return pd.DataFrame()
    d["orbit_label"] = eval_data.loc[d["orbit_idx"].to_numpy(), "orbit_label"].to_numpy()
    d["sim_time_tercile"] = pd.qcut(d["sim_time"], 3, labels=["short", "mid", "long"], duplicates="drop").astype(str)
    cost_col = "ml_total_evals_equiv" if "ml_total_evals_equiv" in d.columns else "ml_evals"
    d["ratio"] = d[cost_col] / d[f"{method}_spec_evals"]
    rows = []
    for col in ("ml_integrator", "orbit_label", "sim_time_tercile"):
        for key, g in d.groupby(col):
            frac = float((g["ratio"] < 1.0).mean())
            rows.append({"grouping": col, "group": key, "n": int(len(g)), "frac_ml_cheaper": frac,
                         "median_eval_ratio": float(g["ratio"].median()),
                         "candidate_regime": bool(len(g) >= GATE_MIN_GROUP_N and frac >= GATE_MIN_FRAC_ML_CHEAPER)})
    return pd.DataFrame(rows)


def _model_kw(a):
    return {"regressor_kind": a.regressor, "quantile": a.quantile}


def _warn_if_unpinned(a):
    reg, q = getattr(a, "regressor", None), getattr(a, "quantile", None)
    if reg is None:
        return
    if reg != PINNED_REGRESSOR or (reg == "hgb_quantile" and q != PINNED_QUANTILE):
        print(f"WARNING: model is --regressor {reg} --quantile {q}, not the pinned {PINNED_REGRESSOR} / "
              f"{PINNED_QUANTILE} from PREREGISTRATION.md. Label this run as a sensitivity run; do not use it "
              f"for gate/final claims.")


def _freeze_guard(a):
    """If freeze_snapshot.py has sealed the code and the pre-registration, check they are unchanged. `final` refuses to
    run on a mismatch unless --allow_post_freeze_changes (recorded in the final log); other commands only warn."""
    st = proto.freeze_status(prereg_path=getattr(a, "prereg", None))
    if st is None:
        return {"sealed": False}
    info = {"sealed": True, "sealed_utc": st["sealed_utc"], "problems": st["problems"],
            "code_combined_sha256": st["code_combined_sha256"]}
    if not st["problems"]:
        print(f"Freeze check: code and pre-registration match the seal of {st['sealed_utc']}.")
        return info
    msg = "; ".join(st["problems"])
    if a.cmd == "final" and not getattr(a, "allow_post_freeze_changes", False):
        raise SystemExit(f"final refused: {msg}. Restore the sealed files, or pass --allow_post_freeze_changes "
                         f"(recorded in the final log) and list the change under '## Deviations'.")
    print(f"WARNING: {msg}. Anything run now is not covered by the seal; list the change under '## Deviations'.")
    return info


def _candidates(a):
    return ("euler", "rk4", "leapfrog") if getattr(a, "include_euler", False) else ("rk4", "leapfrog")


def _subset_pred(pred, rows):
    return {i: {k: v[rows] for k, v in pred[i].items()} for i in pred}


def _cost_gate(models, cal_data, cal_pred, eval_data, a, seed, spec_cache=None):
    cands = _candidates(a)
    cache = ReferenceCache(MU)
    scales, ltt = {}, {}
    for integ in INTEGRATOR_NAMES:
        if integ not in cands:
            scales[integ], ltt[integ] = None, {"skipped": "not a candidate integrator (negative control)"}
            continue
        r = calibrate_scale_ltt(cal_data, cal_pred, integ, a.tol_km, a.alpha, a.delta, cache=cache)
        scales[integ], ltt[integ] = r["chosen_scale"], r
        first = r["chain"][0] if r["chain"] else {}
        print(f"  {integ}: certified scale = {r['chosen_scale']} (alpha={a.alpha}, delta={a.delta}, "
              f"n_cal={r['chain'][-1]['n'] if r['chain'] else 0}; at the smallest scale {first.get('scale')}: "
              f"violation rate {first.get('violation_rate')}, of which incomplete runs {first.get('incomplete')} "
              f"(step cap {first.get('step_cap')}) of {first.get('n')})")
    frontier = None
    if a.frontier_rows > 0:
        fr_rows = np.arange(min(a.frontier_rows, len(cal_data)))
        frontier = scale_frontier_3d(cal_data, cal_pred, integrators=cands, tol_km=a.tol_km, rows=fr_rows, cache=cache)
        if len(frontier):
            frontier.to_csv(os.path.join(a.out_dir, "scale_frontier.csv"), index=False)
            print(frontier.round(4).to_string())
    fallback_used = []
    if a.fallback_scale is not None:
        for integ in cands:
            if scales.get(integ) is None:
                scales[integ] = float(a.fallback_scale)
                fallback_used.append(integ)
        if fallback_used:
            print(f"WARNING: no certified scale for {fallback_used}; using the UNCERTIFIED fallback scale "
                  f"{a.fallback_scale}. Cost numbers below carry no safety guarantee.")
    adaptive_cal, adaptive_info = {}, {}
    for m in ADAPTIVE_METHODS:
        r = calibrate_adaptive_rtol(cal_data, MU, m, a.tol_km, a.alpha, a.delta, max_rows=a.n_cal_adaptive,
                                    random_state=seed)
        adaptive_cal[m], adaptive_info[m] = r["chosen_rtol"], r
        print(f"  {m}: calibrated rtol = {r['chosen_rtol']} (n_cal={r['n_cal']})")
    n_use = min(a.n_baseline, len(eval_data))
    df, summ = matched_accuracy_baseline_3d(eval_data, models, num_samples=n_use, tol_km=a.tol_km,
                                            dt_scale=scales, match="both", random_state=seed,
                                            candidate_integrators=cands, adaptive_calibrated=adaptive_cal,
                                            adaptive_spec_cache=spec_cache)
    verdict = {}
    meets = summ.get("frac_ml_meets_tol")
    for m in ADAPTIVE_METHODS:
        for suffix in ("_spec", "_cal"):
            key = f"{m}{suffix}"
            cheaper = summ.get(f"frac_orbits_ml_total_cheaper_than_{m}{suffix}",
                               summ.get(f"frac_orbits_ml_cheaper_than_{m}{suffix}"))
            cheaper_evals_only = summ.get(f"frac_orbits_ml_cheaper_than_{m}{suffix}")
            if suffix == "_cal" and adaptive_cal.get(m) is None:
                verdict[key] = {"frac_ml_meets_tol": meets, "frac_ml_cheaper": None,
                                "frac_ml_cheaper_evals_only": None, "passes": None,
                                "note": "adaptive rtol could not be certified; comparison not evaluated"}
                continue
            if meets is None or cheaper is None:
                verdict[key] = {"frac_ml_meets_tol": meets, "frac_ml_cheaper": cheaper,
                                "frac_ml_cheaper_evals_only": cheaper_evals_only, "passes": None,
                                "note": "no ML recommendations were available to compare"}
                continue
            ok = bool(meets >= GATE_MIN_FRAC_ML_MEETS_TOL and cheaper >= GATE_MIN_FRAC_ML_CHEAPER)
            verdict[key] = {"frac_ml_meets_tol": meets, "frac_ml_cheaper": cheaper,
                            "frac_ml_cheaper_evals_only": cheaper_evals_only, "passes": ok,
                            "uncertified_scale": bool(fallback_used)}
    evaluated = [v["passes"] for v in verdict.values() if v["passes"] is not None]
    if not evaluated:
        overall = ("NOT_EVALUATED: no cost comparison could be run (no certified safety scale for any candidate "
                   "integrator and no --fallback_scale, or every orbit was skipped). This is NOT a verdict on cost.")
    elif all(evaluated):
        overall = "ML_COMPETITIVE_ON_COST"
    else:
        overall = "ML_DOES_NOT_BEAT_ADAPTIVE_ON_COST: do not claim a speed advantage"
    if fallback_used and evaluated:
        overall += f" [UNCERTIFIED fallback scale {a.fallback_scale} used for {fallback_used}: no safety guarantee]"
    part = {"certified_scales": scales, "ltt": ltt, "adaptive_calibration": adaptive_info,
            "baseline_summary": summ, "verdict": verdict, "overall": overall,
            "uncertified_fallback": fallback_used}
    if frontier is not None:
        part["frontier"] = frontier
    if len(df):
        part["regimes_vs_RK45_spec"] = regime_table(df, eval_data, "RK45")
        part["regime_note"] = ("Regimes are post-hoc subgroups; any 'candidate_regime' must be "
                               "re-tested on freshly generated orbits before it becomes a claim.")
        ens = {}
        sub = df[df["ml_meets_tol"]]
        for m in ADAPTIVE_METHODS:
            if len(sub):
                ens[m] = lockstep_cost_model(sub, f"{m}_spec_evals")
        part["lockstep_model"] = ens
    return part, df


def _judge_group(verdict, win_label, lose_label):
    """Strict: every comparison in the group must be evaluated; a pass needs all of them to pass."""
    if not verdict:
        return "NOT_EVALUATED: no comparison available"
    passes = [v["passes"] for v in verdict.values()]
    if any(p is None for p in passes):
        return "NOT_FULLY_EVALUATED: at least one comparison could not be evaluated; this is NOT a verdict"
    return win_label if all(passes) else lose_label


def _cost_gate_h1_prime(models, cal_data, cal_pred, eval_data, a, seed, adaptive_info, spec_cache):
    """
    H1' (see PREREGISTRATION.md): same orbits, calibration rows, tolerance and thresholds as H1, but
      - the model has NO FTLE feature (cheap features charged as CHEAP_FEATURE_EVALS_EQUIV),
      - model prediction cost is charged (ml_total_evals_equiv_pred),
      - a second comparator, the best single fixed step per integrator certified on the same
        calibration rows with the same alpha/delta (calibrate_global_dt_ltt).
    H1'a = vs RK45/DOP853 (spec and cal); H1'b = vs the global fixed step. Judged separately.
    The adaptive rtol calibration and the per-orbit "_spec" adaptive solves do not depend on the model,
    so they are reused from the H1 pass (adaptive_info, spec_cache).
    """
    cands = _candidates(a)
    cache = ReferenceCache(MU)
    scales, ltt = {}, {}
    for integ in INTEGRATOR_NAMES:
        if integ not in cands:
            scales[integ], ltt[integ] = None, {"skipped": "not a candidate integrator (negative control)"}
            continue
        r = calibrate_scale_ltt(cal_data, cal_pred, integ, a.tol_km, a.alpha, a.delta, cache=cache)
        scales[integ], ltt[integ] = r["chosen_scale"], r
        first = r["chain"][0] if r["chain"] else {}
        print(f"  [H1'] {integ}: certified scale = {r['chosen_scale']} (smallest scale {first.get('scale')}: "
              f"violation rate {first.get('violation_rate')}, incomplete {first.get('incomplete')} of {first.get('n')})")
    global_dt, global_ltt = {}, {}
    for integ in cands:
        g = calibrate_global_dt_ltt(cal_data, integ, a.tol_km, a.alpha, a.delta, cache=cache)
        global_dt[integ], global_ltt[integ] = g["chosen_dt"], g
        print(f"  [H1'] {integ}: certified GLOBAL fixed dt = {g['chosen_dt']} (alpha={a.alpha}, delta={a.delta}, "
              f"n_cal={g['chain'][-1]['n'] if g['chain'] else 0})")
    adaptive_cal = {m: adaptive_info[m]["chosen_rtol"] for m in ADAPTIVE_METHODS}
    n_use = min(a.n_baseline, len(eval_data))
    df, summ = matched_accuracy_baseline_3d(eval_data, models, num_samples=n_use, tol_km=a.tol_km,
                                            dt_scale=scales, match="tolerance", random_state=seed,
                                            candidate_integrators=cands, adaptive_calibrated=adaptive_cal,
                                            charge_prediction=True, global_dt=global_dt,
                                            adaptive_spec_cache=spec_cache)
    meets = summ.get("frac_ml_meets_tol")

    def judge(cheaper, extra=None):
        if extra is not None:
            return {"frac_ml_meets_tol": meets, "frac_ml_cheaper": None, "passes": None, "note": extra}
        if meets is None or cheaper is None:
            return {"frac_ml_meets_tol": meets, "frac_ml_cheaper": cheaper, "passes": None,
                    "note": "no ML recommendations were available to compare"}
        return {"frac_ml_meets_tol": meets, "frac_ml_cheaper": cheaper,
                "passes": bool(meets >= GATE_MIN_FRAC_ML_MEETS_TOL and cheaper >= GATE_MIN_FRAC_ML_CHEAPER)}

    v_adapt, v_global = {}, {}
    for m in ADAPTIVE_METHODS:
        for suffix in ("_spec", "_cal"):
            if suffix == "_cal" and adaptive_cal.get(m) is None:
                v_adapt[f"{m}{suffix}"] = judge(None, "adaptive rtol could not be certified; comparison not evaluated")
            else:
                v_adapt[f"{m}{suffix}"] = judge(summ.get(f"frac_orbits_ml_totalpred_cheaper_than_{m}{suffix}"))
    for integ in cands:
        key = f"GLOBAL_{integ}"
        if global_dt.get(integ) is None:
            v_global[key] = judge(None, "no global fixed dt could be certified; comparison not evaluated")
        else:
            v_global[key] = judge(summ.get(f"frac_orbits_ml_totalpred_cheaper_than_GLOBAL_{integ}"))
            v_global[key]["frac_global_meets_tol"] = summ.get(f"frac_GLOBAL_{integ}_meets_tol")
    part = {
        "feature_columns": models["feature_columns"],
        "certified_scales": scales, "ltt": ltt,
        "global_dt": global_dt, "global_ltt": global_ltt,
        "adaptive_calibration_reused_from_H1": True,
        "baseline_summary": summ,
        "verdict_vs_adaptive": v_adapt,
        "overall_vs_adaptive": _judge_group(v_adapt, "H1a_ML_COMPETITIVE_VS_ADAPTIVE",
                                            "H1a_ML_DOES_NOT_BEAT_ADAPTIVE_ON_COST"),
        "verdict_vs_global": v_global,
        "overall_vs_global": _judge_group(v_global, "H1b_ML_BEATS_GLOBAL_FIXED_DT",
                                          "H1b_ML_DOES_NOT_BEAT_GLOBAL_FIXED_DT"),
    }
    return part, df


def _print_verdict_h1_prime(part):
    summ = part.get("baseline_summary", {})
    print(f"[H1'] orbits compared: {summ.get('n_used')} (skipped {summ.get('n_skipped')}); "
          f"prediction charged at {summ.get('prediction_evals_equiv')} RHS-equivalents per orbit")
    for grp in ("vs_adaptive", "vs_global"):
        print(f"[H1'] {part['overall_' + grp]}")
        for key, v in part["verdict_" + grp].items():
            print(f"    vs {key}: ML meets tol {v['frac_ml_meets_tol']}, ML cheaper (incl. features+prediction) "
                  f"{v['frac_ml_cheaper']}, pass={v['passes']}")


def _print_verdict(part):
    summ = part.get("baseline_summary", {})
    print(f"orbits compared: {summ.get('n_used')} (skipped {summ.get('n_skipped')})")
    print("VERDICT:", part["overall"])
    for key, v in part["verdict"].items():
        print(f"  vs {key}: ML meets tol {v['frac_ml_meets_tol']}, ML cheaper (total cost) {v['frac_ml_cheaper']}, "
              f"cheaper (integration evals only) {v['frac_ml_cheaper_evals_only']}, pass={v['passes']}")


def _prereg_info(a):
    path = getattr(a, "prereg", None)
    if not path:
        print("WARNING: no --prereg file given; the criteria in this report are not tied to a written, "
              "hashed pre-registration.")
        return {"prereg_file": None}
    if not os.path.exists(path):
        raise FileNotFoundError(f"--prereg {path} not found")
    return {"prereg_file": path, "prereg_sha256": proto.sha256_file(path),
            "prereg_body_sha256": proto.prereg_body_sha256(path)}


def cmd_gate(a):
    data = load_accuracy_data(a.data, locked_path=a.locked)
    feats = feature_columns(data)
    tr, te = create_grouped_split_3d(data, test_size=0.3, random_state=a.seed, n_bins=5)
    models = fit_hurdle_models(data, feats, tr, random_state=a.seed, n_estimators=a.n_estimators,
                               regressor_kind=a.regressor, quantile=a.quantile)
    test = data.iloc[te].reset_index(drop=True)
    pred = predict_hurdle(models, test)
    report = {"args": args_dict(a), "n_train": int(len(tr)), "n_test": int(len(te)),
              "held_out_metrics": metrics_from_prediction(test, pred),
              "criteria": {"min_frac_ml_meets_tol": GATE_MIN_FRAC_ML_MEETS_TOL,
                           "min_frac_ml_cheaper": GATE_MIN_FRAC_ML_CHEAPER, "min_group_n": GATE_MIN_GROUP_N},
              **_prereg_info(a)}
    perm = np.random.RandomState(a.seed).permutation(len(test))
    n_cal = min(a.n_cal, len(test) // 2)
    cal_rows, base_rows = np.sort(perm[:n_cal]), np.sort(perm[n_cal:])
    if len(base_rows) < a.n_baseline:
        print(f"WARNING: only {len(base_rows)} held-out orbits left for the baseline (< {a.n_baseline}).")
        report["warning_small_baseline"] = int(len(base_rows))
    cal_data = test.iloc[cal_rows].reset_index(drop=True)
    cal_pred = _subset_pred(pred, cal_rows)
    eval_data = test.iloc[base_rows].reset_index(drop=True)
    spec_cache = {}
    part, df = _cost_gate(models, cal_data, cal_pred, eval_data, a, a.seed, spec_cache=spec_cache)
    report.update(part)
    if len(df):
        df.to_csv(os.path.join(a.out_dir, "gate_baseline.csv"), index=False)
    part2 = None
    if not a.no_h1_prime:
        feats_nf = [c for c in feats if c != "ftle_3d"]
        models_nf = fit_hurdle_models(data, feats_nf, tr, random_state=a.seed, n_estimators=a.n_estimators,
                                      regressor_kind=a.regressor, quantile=a.quantile)
        cal_pred_nf = _subset_pred(predict_hurdle(models_nf, test), cal_rows)
        part2, df2 = _cost_gate_h1_prime(models_nf, cal_data, cal_pred_nf, eval_data, a, a.seed,
                                         part["adaptive_calibration"], spec_cache)
        report["h1_prime"] = part2
        if len(df2):
            df2.to_csv(os.path.join(a.out_dir, "gate_baseline_h1_prime.csv"), index=False)
    save_json(report, os.path.join(a.out_dir, "gate_report.json"))
    _print_verdict(part)
    if part2 is not None:
        _print_verdict_h1_prime(part2)


def cmd_final(a):
    if a.final_data:
        proto.check_independent_datasets([a.data, a.final_data])
        dev = load_accuracy_data(a.data, locked_path=a.locked)
        locked = load_accuracy_data(a.final_data, check_lock=False)
    else:
        if not a.locked:
            raise ValueError("final needs --locked <locked_test_set.json> or --final_data <independent dataset>")
        dev = load_accuracy_data(a.data, locked_path=a.locked)
        locked = load_accuracy_data(a.data, locked_path=a.locked, only_locked=True)
    feats = feature_columns(dev)
    tr, te = create_grouped_split_3d(dev, test_size=0.3, random_state=a.seed, n_bins=5)
    models = fit_hurdle_models(dev, feats, tr, random_state=a.seed, n_estimators=a.n_estimators,
                               regressor_kind=a.regressor, quantile=a.quantile)
    cal_pool = dev.iloc[te].reset_index(drop=True)
    n_cal = min(a.n_cal, len(cal_pool))
    cal_rows = np.sort(np.random.RandomState(a.seed).choice(len(cal_pool), size=n_cal, replace=False))
    cal_data = cal_pool.iloc[cal_rows].reset_index(drop=True)
    cal_pred = _subset_pred(predict_hurdle(models, cal_pool), cal_rows)
    pred_locked = predict_hurdle(models, locked)
    report = {"args": args_dict(a), "n_dev_train": int(len(tr)), "n_calibration": int(n_cal),
              "n_locked": int(len(locked)),
              "locked_metrics": metrics_from_prediction(locked, pred_locked),
              "criteria": {"min_frac_ml_meets_tol": GATE_MIN_FRAC_ML_MEETS_TOL,
                           "min_frac_ml_cheaper": GATE_MIN_FRAC_ML_CHEAPER}, **_prereg_info(a)}
    spec_cache = {}
    part, df = _cost_gate(models, cal_data, cal_pred, locked, a, a.seed, spec_cache=spec_cache)
    report.update(part)
    part2 = None
    if not a.no_h1_prime:
        feats_nf = [c for c in feats if c != "ftle_3d"]
        models_nf = fit_hurdle_models(dev, feats_nf, tr, random_state=a.seed, n_estimators=a.n_estimators,
                                      regressor_kind=a.regressor, quantile=a.quantile)
        cal_pred_nf = _subset_pred(predict_hurdle(models_nf, cal_pool), cal_rows)
        part2, df2 = _cost_gate_h1_prime(models_nf, cal_data, cal_pred_nf, locked, a, a.seed,
                                         part["adaptive_calibration"], spec_cache)
        report["h1_prime"] = part2
        if len(df2):
            df2.to_csv(os.path.join(a.out_dir, "final_baseline_h1_prime.csv"), index=False)
    n_aud = min(a.audit_rows, len(locked))
    sub = np.sort(np.random.RandomState(a.seed).choice(len(locked), size=n_aud, replace=False))
    aud_cache = ReferenceCache(MU)
    raw = position_error_audit_3d(locked, pred_locked, a.tol_km, 1.0, MU, aud_cache, rows=sub)
    scaled = position_error_audit_3d(locked, pred_locked, a.tol_km, part["certified_scales"], MU, aud_cache, rows=sub)
    report["locked_audit_unscaled"] = summarize_audit(raw) if len(raw) else {}
    report["locked_audit_scaled"] = summarize_audit(scaled) if len(scaled) else {}
    if len(df):
        df.to_csv(os.path.join(a.out_dir, "final_baseline.csv"), index=False)
    n_eval = proto.log_final_evaluation(a.final_log, {"data": a.data, "locked": a.locked,
                                                       "final_data": a.final_data, "seed": a.seed,
                                                       "regressor": a.regressor, "quantile": a.quantile,
                                                       "freeze": getattr(a, "freeze_info", None),
                                                       "run_dir": a.out_dir, "overall": part["overall"],
                                                       "overall_h1_prime_vs_adaptive":
                                                           part2["overall_vs_adaptive"] if part2 else None,
                                                       "overall_h1_prime_vs_global":
                                                           part2["overall_vs_global"] if part2 else None})
    report["final_evaluation_number"] = n_eval
    print(f"This is evaluation #{n_eval} on the locked set (logged in {a.final_log}).")
    save_json(report, os.path.join(a.out_dir, "final_report.json"))
    _print_verdict(part)
    if part2 is not None:
        _print_verdict_h1_prime(part2)


def cmd_lock(a):
    data = load_accuracy_data(a.data, keep_divergent=True, check_lock=False)
    csv_dir = os.path.dirname(os.path.abspath(a.data))
    out = a.locked_out or os.path.join(csv_dir, "locked_test_set.json")
    if os.path.dirname(os.path.abspath(out)) != csv_dir or not os.path.basename(out).startswith("locked_test_set"):
        print(f"WARNING: {out} is not named locked_test_set*.json in {csv_dir}; the guard that makes development "
              f"commands refuse this CSV without --locked will not see it. Pass --locked {out} every time.")
    rec = proto.lock_test_set(data, out, csv_path=a.data, seed=a.lock_seed, test_frac=a.lock_frac)
    print(f"Locked {rec['n_locked']} of {rec['n_rows_in_split_pool']} rows to {out} "
          f"(csv sha256 {rec['csv_sha256'][:12]}...). Do not use these rows for development.")
    print(f"From now on pass --locked {out} to every command that reads {a.data}; without it they refuse "
          f"(--ignore_lock overrides, loudly).")


def cmd_proxy(a):
    data = load_accuracy_data(a.data, locked_path=a.locked)
    table = proxy_vs_error_table(data, num_orbits=a.n_orbits, random_state=a.seed)
    table.to_csv(os.path.join(a.out_dir, "proxy_table.csv"), index=False)
    by_integ, by_type = proxy_failure_summary(table, drift_tol=a.drift_tol, tol_km=a.tol_km)
    print(by_integ.round(3).to_string())
    print(by_type.round(3).to_string())
    by_integ.to_csv(os.path.join(a.out_dir, "proxy_by_integrator.csv"))
    by_type.to_csv(os.path.join(a.out_dir, "proxy_by_orbit_type.csv"))
    res = fit_error_predictor(table, data, tol_km=a.tol_km, random_state=a.seed)
    res["test_predictions"].to_csv(os.path.join(a.out_dir, "error_predictor_test.csv"), index=False)
    out = {k: v for k, v in res.items() if k != "test_predictions"}
    save_json(out, os.path.join(a.out_dir, "proxy_report.json"))
    print(json.dumps(out, indent=1, default=_json_default))


def cmd_audit(a):
    data = load_accuracy_data(a.data, locked_path=a.locked)
    feats = feature_columns(data)
    seeds = list(range(a.seeds))
    per_seed, summ = multiseed_evaluation(data, seeds, feats, n_estimators=a.n_estimators,
                                          audit_rows=a.audit_rows, tol_km=a.tol_km, model_kw=_model_kw(a))
    per_seed.to_csv(os.path.join(a.out_dir, "multiseed_per_seed.csv"), index=False)
    summ.to_csv(os.path.join(a.out_dir, "multiseed_summary.csv"), index=False)
    print(summ.round(4).to_string())

    tr, te = create_grouped_split_3d(data, test_size=0.3, random_state=0, n_bins=5)
    models = fit_hurdle_models(data, feats, tr, random_state=0, n_estimators=a.n_estimators,
                               regressor_kind=a.regressor, quantile=a.quantile)
    test = data.iloc[te].reset_index(drop=True)
    pred = predict_hurdle(models, test)
    report = {"args": args_dict(a)}
    if a.ltt_rows > 0:
        rep = fit_and_validate_scale_ltt(test, pred, a.tol_km, a.alpha, a.delta, random_state=0, max_rows=a.ltt_rows)
        report["scale_calibration"] = rep
        for integ, e in rep["per_integrator"].items():
            print(integ, "scale", e["chosen_scale"], "unscaled viol", e["eval_unscaled_violation_rate"],
                  "scaled viol", e.get("eval_scaled_violation_rate"), e.get("eval_scaled_violation_ci95"))
    if a.ltt_repeats > 0:
        per, ssum = calibration_stability_3d(test, pred, a.tol_km, a.alpha, a.delta, n_repeats=a.ltt_repeats,
                                             cal_size=a.stab_cal_size, random_state=0)
        per.to_csv(os.path.join(a.out_dir, "scale_stability_per_repeat.csv"), index=False)
        ssum.to_csv(os.path.join(a.out_dir, "scale_stability_summary.csv"), index=False)
        report["scale_stability"] = ssum
        print(ssum.to_string())
    if a.tol_sens:
        s = tolerance_sensitivity_accuracy_3d(test, num_samples=a.tol_sens)
        s.to_csv(os.path.join(a.out_dir, "tolerance_sensitivity.csv"), index=False)
        agree = selection_agreement(s, base_tol_km=100.0 if (s["tol_km"] == 100.0).any() else float(s["tol_km"].median()))
        report["selection_agreement"] = agree
        print(agree.to_string())
    if a.dense:
        dn = dense_sweep_accuracy_3d(test, a.tol_km, num_samples=a.dense)
        dn.to_csv(os.path.join(a.out_dir, "dense_sweep.csv"), index=False)
        report["dense_sweep"] = summarize_dense_sweep(dn)
        print(report["dense_sweep"])
    if a.halo:
        scale_sets = [1.0, 0.2, 0.1, 0.05]
        if a.ltt_rows > 0:
            certified = {i: report.get("scale_calibration", {}).get("per_integrator", {}).get(i, {}).get("chosen_scale")
                         for i in INTEGRATOR_NAMES}
            if any(v is not None for v in certified.values()):
                scale_sets.append(certified)
        parts = [halo_validation_accuracy_3d(models, tol_km=a.tol_km, scale=sc) for sc in scale_sets]
        parts = [p for p in parts if len(p)]
        h = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        h.to_csv(os.path.join(a.out_dir, "halo_validation.csv"), index=False)
        print(h.round(3).to_string() if len(h) else "halo validation produced no rows (no feasible predictions)")
    save_json(report, os.path.join(a.out_dir, "audit_report.json"))


def cmd_fidelity(a):
    from validate_on_capstone import query_vectors
    from orbit_ml_cr3bp.fidelity_3d import (synodic_states_from_ephemeris, cr3bp_vs_ephemeris_error,
                                            summarize_fidelity, moon_window_tu_days, TU_DAYS_DEFAULT)
    epochs = {"start": a.start, "stop": a.stop, "step": a.step}
    jd_m, r_m, v_m = query_vectors("301", "500@399", epochs)
    jd_s, r_s, v_s = query_vectors(a.target, "500@3", epochs)
    if len(jd_m) != len(jd_s):
        raise RuntimeError("Spacecraft/Moon epoch counts differ")
    states, scale = synodic_states_from_ephemeris(r_s, v_s, r_m, v_m)
    tu_window = moon_window_tu_days(r_m, v_m)
    if a.tu_days is not None:
        tu0, src = a.tu_days, "user supplied"
    elif a.tu_source == "window":
        tu0, src = tu_window, "from the Moon's instantaneous angular rate over the window (phase dependent)"
    else:
        tu0, src = TU_DAYS_DEFAULT, "nominal: sidereal month / 2 pi (the CR3BP definition)"
    print(f"time unit used: {tu0:.5f} days ({src}); window-derived diagnostic: {tu_window:.5f} days")
    out = {"tu_days_window_diagnostic": tu_window}
    for label, tu in (("nominal", tu0), ("minus1pct", tu0 * 0.99), ("plus1pct", tu0 * 1.01)):
        df = cr3bp_vs_ephemeris_error(jd_s, states, scale, tu_days=tu, max_lag_days=a.max_lag_days, stride=a.stride)
        df.to_csv(os.path.join(a.out_dir, f"fidelity_{label}_tu{tu:.4f}.csv"), index=False)
        s = summarize_fidelity(df)
        out[label] = {"tu_days": tu, "summary": s.reset_index().astype({s.index.name or "lag_tu": str}).to_dict(orient="records") if len(s) else []}
        print(f"--- tu_days={tu:.4f} ({label})")
        print(s.round(1).to_string())
    out["note"] = "Compare median_km at the horizons you care about with your integration tolerance (tol_km)."
    save_json(out, os.path.join(a.out_dir, f"fidelity_report_tu{tu0:.4f}.json"))


def _parse_datasets(specs, locked_path=None):
    datasets, paths = {}, []
    for spec in specs:
        name, path = spec.split("=", 1)
        use_locked = None
        if locked_path:
            try:
                proto.load_locked(locked_path, csv_path=path)
                use_locked = locked_path
            except ValueError:
                use_locked = None
        datasets[name] = load_accuracy_data(path, locked_path=use_locked)
        paths.append(path)
        note = " (locked rows excluded)" if use_locked else ""
        print(f"{name}: {len(datasets[name])} usable rows{note}")
    return datasets, paths


def _dataset_mu_and_tol(datasets, mu_overrides):
    mus, tol_du = {}, {}
    for name, d in datasets.items():
        if name in mu_overrides:
            mus[name] = mu_overrides[name]
        elif "mu" in d.columns:
            mus[name] = float(d["mu"].iloc[0])
        else:
            mus[name] = None
        if "tol_km" in d.columns and "du_km" in d.columns:
            tol_du[name] = float(d["tol_km"].iloc[0] / d["du_km"].iloc[0])
    return mus, tol_du


def cmd_mu(a):
    datasets, _paths = _parse_datasets(a.dataset, a.locked)
    overrides = {}
    for spec in a.dataset_mu or []:
        k, v = spec.split("=", 1)
        overrides[k] = float(v)
    mus, tol_du = _dataset_mu_and_tol(datasets, overrides)
    proto.check_independent_datasets(_paths)
    if tol_du:
        vals = list(tol_du.values())
        print("normalized tolerance (tol_km / du_km) per dataset:", {k: f"{v:.3e}" for k, v in tol_du.items()})
        if len(vals) > 1 and (max(vals) - min(vals)) / max(vals) > 0.01:
            print("WARNING: datasets use different NORMALIZED tolerances; within/transfer comparisons are not "
                  "like-for-like. Regenerate with the same --tol_du.")
    else:
        print("WARNING: tol_km/du_km columns missing; cannot verify the datasets share a normalized tolerance.")
    use_mus = mus if all(v is not None for v in mus.values()) else None
    if use_mus is None:
        print("pooled evaluation skipped: could not determine mu for every dataset (pass --dataset_mu name=value).")
    per_seed, summ = cross_mu_evaluation(datasets, list(range(a.seeds)), n_estimators=a.n_estimators, mus=use_mus,
                                          model_kw=_model_kw(a))
    per_seed.to_csv(os.path.join(a.out_dir, "cross_mu_per_seed.csv"), index=False)
    summ.to_csv(os.path.join(a.out_dir, "cross_mu_summary.csv"), index=False)
    keep = summ[summ["metric"].isin(["rk4_clf_acc", "rk4_r2_uncens", "selection_acc", "majority_baseline_acc",
                                     "selection_acc_no_euler", "majority_baseline_acc_no_euler"])]
    print(keep.round(3).to_string())


def cmd_replicate(a):
    datasets, _paths = _parse_datasets(a.dataset, a.locked)
    proto.check_independent_datasets(_paths)
    mus, tol_du = _dataset_mu_and_tol(datasets, {})
    known = [v for v in mus.values() if v is not None]
    if len(set(round(v, 12) for v in known)) > 1:
        raise ValueError(f"replicate needs datasets with the SAME mu; got {mus}")
    if len(set(tol_du.values())) > 1:
        print(f"WARNING: datasets differ in normalized tolerance: {tol_du}")
    per_seed, by_pair, across = replication_evaluation(datasets, list(range(a.seeds)), n_estimators=a.n_estimators,
                                                           model_kw=_model_kw(a))
    per_seed.to_csv(os.path.join(a.out_dir, "replication_per_seed.csv"), index=False)
    by_pair.to_csv(os.path.join(a.out_dir, "replication_by_pair.csv"), index=False)
    across.to_csv(os.path.join(a.out_dir, "replication_across_datasets.csv"), index=False)
    keep = across[across["metric"].isin(["rk4_clf_acc", "rk4_r2_uncens", "leapfrog_r2_uncens", "selection_acc",
                                         "selection_acc_no_euler", "majority_baseline_acc_no_euler"])]
    print(keep.round(3).to_string())


def cmd_ablate(a):
    data = load_accuracy_data(a.data, locked_path=a.locked)
    sets = default_feature_sets(data, leave_one_out=not a.no_loo)
    per_seed, summ, deltas = feature_ablation_evaluation(data, list(range(a.seeds)), sets,
                                                         n_estimators=a.n_estimators, model_kw=_model_kw(a))
    per_seed.to_csv(os.path.join(a.out_dir, "ablation_per_seed.csv"), index=False)
    summ.to_csv(os.path.join(a.out_dir, "ablation_summary.csv"), index=False)
    deltas.to_csv(os.path.join(a.out_dir, "ablation_deltas_vs_full.csv"), index=False)
    keep = deltas[deltas["metric"].isin(["rk4_r2_uncens", "leapfrog_r2_uncens", "selection_acc_no_euler"])]
    print(keep.round(4).to_string())


def cmd_crash(a):
    data = load_accuracy_data(a.data, keep_divergent=True, locked_path=a.locked)
    print(f"{len(data)} rows, {int(data['reference_divergent'].sum())} reference-divergent "
          f"({100 * data['reference_divergent'].mean():.1f}%)")
    per_seed, summ, by_outcome = crash_classifier_evaluation(data, list(range(a.seeds)), n_estimators=a.n_estimators)
    per_seed.to_csv(os.path.join(a.out_dir, "crash_per_seed.csv"), index=False)
    summ.to_csv(os.path.join(a.out_dir, "crash_summary.csv"), index=False)
    by_outcome.to_csv(os.path.join(a.out_dir, "crash_recall_by_outcome.csv"), index=False)
    print(summ[summ["metric"].isin(["accuracy", "majority_baseline_acc", "balanced_accuracy", "precision",
                                    "recall", "auc"])].round(3).to_string())
    print(by_outcome.round(3).to_string())


def cmd_errmodel(a):
    data = load_accuracy_data(a.data, locked_path=a.locked)
    table = error_ladder_table(data, num_orbits=a.n_orbits, random_state=a.seed)
    table.to_csv(os.path.join(a.out_dir, "error_ladder_table.csv"), index=False)
    em = fit_error_model(table, data, tol_km=a.tol_km, alpha=a.alpha_em, random_state=a.seed,
                         n_estimators=a.n_estimators)
    df, summ = evaluate_recommender(em, data)
    df.to_csv(os.path.join(a.out_dir, "error_model_eval.csv"), index=False)
    save_json({"args": args_dict(a), "n_train_orbits": int(len(em["train_orbits"])),
               "n_cal_orbits": int(len(em["cal_orbits"])), "n_test_orbits": int(len(em["test_orbits"])),
               "summary": summ}, os.path.join(a.out_dir, "error_model_report.json"))
    print(json.dumps(summ, indent=1, default=_json_default))


def cmd_convergence(a):
    data = load_accuracy_data(a.data, locked_path=a.locked)
    table = error_ladder_table(data, num_orbits=a.n_orbits, random_state=a.seed)
    table.to_csv(os.path.join(a.out_dir, "error_ladder_table.csv"), index=False)
    per, summ = convergence_order_study(table, data, tol_km=a.tol_km)
    per.to_csv(os.path.join(a.out_dir, "convergence_per_orbit.csv"), index=False)
    summ.to_csv(os.path.join(a.out_dir, "convergence_summary.csv"), index=False)
    print(summ.round(3).to_string())


def resolve_tol_km(a):
    path = getattr(a, "data", None)
    dataset_tol = None
    if path and os.path.exists(path):
        try:
            col = pd.read_csv(path, usecols=["tol_km"], nrows=1)
            dataset_tol = float(col["tol_km"].iloc[0])
        except Exception:
            dataset_tol = None
    if a.tol_km is None:
        return dataset_tol if dataset_tol is not None else POSITION_TOL_KM
    if dataset_tol is not None and abs(a.tol_km - dataset_tol) > 5e-3 * dataset_tol:
        raise ValueError(f"--tol_km {a.tol_km} disagrees with the tolerance the dataset was labeled with "
                         f"({dataset_tol:.6g} km). The labels, certified scales and audits are only "
                         f"meaningful at the labeling tolerance; regenerate the data or drop --tol_km.")
    return a.tol_km


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, data=True):
        if data:
            sp.add_argument("--data", required=True)
        sp.add_argument("--tol_km", type=float, default=None,
                        help="position tolerance in km; default = the tolerance stored in the dataset "
                             "(its tol_km column), else config.POSITION_TOL_KM. A value that disagrees with the "
                             "dataset's labeling tolerance is an error.")
        sp.add_argument("--seed", type=int, default=42)
        sp.add_argument("--n_estimators", type=int, default=150)
        sp.add_argument("--out_dir", default=".", help="root folder; each run gets its own timestamped subfolder")
        sp.add_argument("--no_run_dir", action="store_true", help="write directly into --out_dir (no run folder)")
        sp.add_argument("--locked", default=None, help="locked_test_set.json: exclude those rows from this command. "
                                                       "Required for any CSV that has a lock file next to it.")
        sp.add_argument("--ignore_lock", action="store_true",
                        help="knowingly use ALL rows of a CSV that has a lock (prints a warning; recorded in the manifest)")
        sp.add_argument("--prereg", default=None, help="pre-registration file; its SHA-256 is recorded")

    def model_opts(sp):
        sp.add_argument("--regressor", choices=["rf", "hgb_quantile"], default=PINNED_REGRESSOR,
                        help="pinned in the pre-registration; use rf only for a labelled sensitivity run")
        sp.add_argument("--quantile", type=float, default=PINNED_QUANTILE,
                        help="target percentile of the safe step for --regressor hgb_quantile")

    def gate_opts(sp):
        sp.add_argument("--n_baseline", type=int, default=1000)
        sp.add_argument("--n_cal", type=int, default=2000)
        sp.add_argument("--n_cal_adaptive", type=int, default=300)
        sp.add_argument("--alpha", type=float, default=0.05)
        sp.add_argument("--delta", type=float, default=0.1)
        sp.add_argument("--include_euler", action="store_true",
                        help="let the ML recommend Euler too (it is otherwise a negative control)")
        sp.add_argument("--frontier_rows", type=int, default=600,
                        help="rows used for the scale -> violation/cost table (0 = skip)")
        sp.add_argument("--fallback_scale", type=float, default=None,
                        help="if no scale is certified, run the cost comparison at this UNCERTIFIED scale "
                             "(clearly flagged in the report)")
        sp.add_argument("--no_h1_prime", action="store_true",
                        help="skip the H1' variant (no-FTLE model, prediction cost charged, global fixed-dt baseline)")
        model_opts(sp)

    g = sub.add_parser("gate"); common(g); gate_opts(g); g.set_defaults(fn=cmd_gate)

    fn = sub.add_parser("final"); common(fn); gate_opts(fn)
    fn.add_argument("--allow_post_freeze_changes", action="store_true",
                    help="run even though the code or pre-registration differs from the seal (FREEZE.json); logged")
    fn.add_argument("--audit_rows", type=int, default=500)
    fn.add_argument("--final_log", default="final_eval_log.jsonl")
    fn.add_argument("--final_data", default=None,
                    help="an independent, never-developed-on dataset CSV to evaluate on instead of locked rows "
                         "of --data (checked for seed overlap with --data)")
    fn.set_defaults(fn=cmd_final)

    lk = sub.add_parser("lock"); common(lk)
    lk.add_argument("--locked_out", default=None,
                    help="where to write the lock; default <folder of --data>/locked_test_set.json (the guard only "
                         "sees locked_test_set*.json next to the CSV)")
    lk.add_argument("--lock_seed", type=int, default=2024)
    lk.add_argument("--lock_frac", type=float, default=0.25,
                    help="fraction of ALL rows (reference-divergent ones included) locked at random with --lock_seed; "
                         "default 0.25")
    lk.set_defaults(fn=cmd_lock)

    pr = sub.add_parser("proxy"); common(pr)
    pr.add_argument("--n_orbits", type=int, default=100)
    pr.add_argument("--drift_tol", type=float, default=1e-5)
    pr.set_defaults(fn=cmd_proxy)

    au = sub.add_parser("audit"); common(au); model_opts(au)
    au.add_argument("--seeds", type=int, default=10)
    au.add_argument("--audit_rows", type=int, default=0)
    au.add_argument("--ltt_rows", type=int, default=0)
    au.add_argument("--ltt_repeats", type=int, default=0, help="calibration-stability repeats (0 = skip)")
    au.add_argument("--stab_cal_size", type=int, default=500)
    au.add_argument("--alpha", type=float, default=0.05)
    au.add_argument("--delta", type=float, default=0.1)
    au.add_argument("--tol_sens", type=int, default=0)
    au.add_argument("--dense", type=int, default=0)
    au.add_argument("--halo", action="store_true")
    au.set_defaults(fn=cmd_audit)

    fi = sub.add_parser("fidelity"); common(fi, data=False)
    fi.add_argument("--target", default="-1176")
    fi.add_argument("--start", default="2023-01-01")
    fi.add_argument("--stop", default="2023-01-15")
    fi.add_argument("--step", default="6h")
    fi.add_argument("--tu_days", type=float, default=None)
    fi.add_argument("--tu_source", choices=["nominal", "window"], default="nominal")
    fi.add_argument("--max_lag_days", type=float, default=15.0)
    fi.add_argument("--stride", type=int, default=1)
    fi.set_defaults(fn=cmd_fidelity)

    mu = sub.add_parser("mu"); common(mu, data=False); model_opts(mu)
    mu.add_argument("--dataset", action="append", required=True, help="name=path/to/accuracy.csv (repeat)")
    mu.add_argument("--dataset_mu", action="append", help="name=mu (only if the CSV has no mu column)")
    mu.add_argument("--seeds", type=int, default=5)
    mu.set_defaults(fn=cmd_mu)

    rp = sub.add_parser("replicate"); common(rp, data=False); model_opts(rp)
    rp.add_argument("--dataset", action="append", required=True,
                    help="name=path (repeat; same mu, different --base_seed)")
    rp.add_argument("--seeds", type=int, default=5)
    rp.set_defaults(fn=cmd_replicate)

    ab = sub.add_parser("ablate"); common(ab); model_opts(ab)
    ab.add_argument("--seeds", type=int, default=5)
    ab.add_argument("--no_loo", action="store_true", help="skip leave-one-feature-out sets")
    ab.set_defaults(fn=cmd_ablate)

    cr = sub.add_parser("crash"); common(cr)
    cr.add_argument("--seeds", type=int, default=5)
    cr.set_defaults(fn=cmd_crash)

    em = sub.add_parser("errmodel"); common(em)
    em.add_argument("--n_orbits", type=int, default=300)
    em.add_argument("--alpha_em", type=float, default=0.1)
    em.set_defaults(fn=cmd_errmodel)

    cv = sub.add_parser("convergence"); common(cv)
    cv.add_argument("--n_orbits", type=int, default=300)
    cv.set_defaults(fn=cmd_convergence)

    a = p.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    a.tol_km = resolve_tol_km(a)
    _warn_if_unpinned(a)
    LOCK_POLICY["ignore"] = bool(getattr(a, "ignore_lock", False))
    freeze_info = _freeze_guard(a)
    root = a.out_dir
    os.makedirs(root, exist_ok=True)
    data_paths = [getattr(a, "data", None)] if getattr(a, "data", None) else []
    if getattr(a, "final_data", None):
        data_paths.append(a.final_data)
    for spec in getattr(a, "dataset", None) or []:
        data_paths.append(spec.split("=", 1)[1])
    a.out_dir = proto.start_run(root, a.cmd, a, data_paths=data_paths, use_run_dir=not a.no_run_dir)
    print(f"Run folder: {a.out_dir}")
    a.freeze_info = freeze_info
    a.fn(a)


if __name__ == "__main__":
    main()