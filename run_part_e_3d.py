"""
Part E preparation (part_e/PREREGISTRATION_E.md sections 3 and 7, Step 3 of part_e/STUDY_E_PLAN.md). Everything here runs BEFORE the seal and
touches no confirmation data; its outputs are frozen inputs of the seal.

  python run_part_e_3d.py e7_labels            RK8 labels of the Study C fitting sets (reference positions reused from the P25 cache if identical)
  python run_part_e_3d.py e7_select            the E7 model and the label-ceiling check on E_select (seed 60)
  python run_part_e_3d.py e7_freeze            fit the chosen E7 model on the fitting sets -> part_e/frozen/rk8s/policy_frozen.joblib
  python run_part_e_3d.py overheads            RHS-equivalents of the RK4 and E7 pipelines, idle machine (load <= 1.0) -> part_e/frozen/overheads.json
  python run_part_e_3d.py gate                 the E5 gate: fit on the fitting sets, threshold on E_select -> part_e/frozen/gate/gate.joblib
  python run_part_e_3d.py frozen_list          part_e/FROZEN_FILES.json: every file the seal hashes (needs the confirmation data generated)

--smoke N: the Study D selection set (first N usable rows) stands in for E_select, the training sets are subsampled to 4N rows, and every
output goes to out_e/smoke/ (a code check; never a frozen input).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from orbit_ml_cr3bp import protocol_3d as proto  # noqa: E402
from orbit_ml_cr3bp.config import MU  # noqa: E402

C_TRAIN = ("data_c/train/cr3bp_3d_training_data_accuracy_monotone_seed41.csv",
           "data_c/train_enriched/cr3bp_3d_training_data_accuracy_monotone_seed141.csv")
SEL_E = "data_e/select/cr3bp_3d_training_data_accuracy_monotone_seed60.csv"
SEL_SMOKE = "data_d/select/cr3bp_3d_training_data_accuracy_monotone_seed52.csv"
RK4_FROZEN = "part_c/frozen/sund/policy_frozen.joblib"


def base(a):
    return os.path.join(ROOT, "out_e", "smoke", "frozen") if a.smoke else os.path.join(ROOT, "part_e", "frozen")


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(proto.json_safe(obj), open(path, "w"), indent=1)


def features():
    import joblib
    return list(joblib.load(os.path.join(ROOT, RK4_FROZEN))["spec"]["features"])


def load_training(a):
    import run_part_c_3d as rc
    paths = [os.path.join(ROOT, p) for p in C_TRAIN]
    U, w = rc.load_c_frame(paths, MU, enriched={paths[1]})
    if a.smoke:
        keep = np.sort(np.random.RandomState(0).choice(len(U), 4 * a.smoke, replace=False))
        U, w = U.iloc[keep].reset_index(drop=True), w[keep]
    return U, w


def load_select(a):
    import run_part_c_3d as rc
    U, _w = rc.load_c_frame([os.path.join(ROOT, SEL_SMOKE if a.smoke else SEL_E)], MU)
    return U.iloc[: a.smoke].reset_index(drop=True) if a.smoke else U


def ref_positions(df, tag, a):
    from orbit_ml_cr3bp import deploy_d as dd
    name = f"{'smoke_' if a.smoke else ''}{tag}.npz"
    cache = os.path.join(ROOT, "out_e", "refs", name)
    if tag == "c_train" and not os.path.exists(cache):
        old = os.path.join(ROOT, "posthoc", "cache", "refs_c_train.npz")       # P25's cache: used only if it is for exactly these rows
        if os.path.exists(old):
            z = np.load(old, allow_pickle=False)
            S, T = dd.states_T(df)
            if str(z["key"]) == dd._key(S, T):
                os.makedirs(os.path.dirname(cache), exist_ok=True)
                np.savez(cache, ref=z["ref"], key=z["key"])
    return dd.reference_positions(df, MU, cache=cache, n_jobs=8)


def labels_path(a):
    return os.path.join(base(a), "..", "e7_train_labels.csv.gz") if a.smoke else os.path.join(ROOT, "data_e", "e7_train_labels.csv.gz")


def rows_fn(pool):
    from orbit_ml_cr3bp import events_e as ee

    def f(method, S, T, param, ref):
        return ee.rows_with_event(method, S, T, param, ref, None, seg=True, pool=pool)
    return f


def cmd_e7_labels(a):
    from multiprocessing import Pool
    from orbit_ml_cr3bp import e7_policy as e7
    t0 = time.time()
    U, _w = load_training(a)
    ref = ref_positions(U, "c_train", a)
    print(f"{len(U)} training rows, references ready ({time.time() - t0:.0f}s)", flush=True)
    with Pool(a.n_jobs) as pool:
        L = e7.training_labels(U, ref, pool)
    path = labels_path(a)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    L.to_csv(path, index=False)
    rep = e7.ceiling_report(L)
    save_json({"n": int(len(L)), "success_rate": float(L["rk8s_success"].mean()), "ceiling": rep, "seconds": time.time() - t0},
              os.path.join(base(a), "rk8s", "train_labels_report.json"))
    print(json.dumps(rep, indent=1))


def _training_with_labels(a):
    U, w = load_training(a)
    L = pd.read_csv(labels_path(a))
    if not np.array_equal(L["row_id"].to_numpy(), U["row_id"].to_numpy()):
        raise SystemExit("the label file is not for these training rows; rerun e7_labels")
    U["dt_rk8s"], U["rk8s_success"] = L["dt_rk8s"].to_numpy(), L["rk8s_success"].to_numpy()
    return U, w


def cmd_e7_select(a):
    from multiprocessing import Pool
    from orbit_ml_cr3bp import e7_policy as e7
    t0 = time.time()
    tr, w = _training_with_labels(a)
    sel = load_select(a)
    ref = ref_positions(sel, "select", a)
    with Pool(a.n_jobs) as pool:
        # the label-ceiling rule on E_select
        L = e7.training_labels(sel, ref, pool)
        ceil = e7.ceiling_report(L)
        print("E_select ceiling:", ceil, flush=True)
        oh = a.overhead_guess
        chosen, tab = e7.select_e7(tr, w, features(), sel, ref, rows_fn(pool), oh, log=lambda m: print(m, flush=True))
    out = os.path.join(base(a), "rk8s")
    os.makedirs(out, exist_ok=True)
    tab.to_csv(os.path.join(out, "selection.csv"), index=False)
    save_json({"chosen": {"n_trees": chosen[0], "quantile": chosen[1]}, "overhead_used_for_selection": oh, "select_rows": int(len(sel)),
               "ceiling_on_select": ceil, "seconds": time.time() - t0}, os.path.join(out, "selection.json"))
    print("chosen:", chosen)


def cmd_e7_freeze(a):
    import joblib
    from orbit_ml_cr3bp import e7_policy as e7
    out = os.path.join(base(a), "rk8s")
    sel = json.load(open(os.path.join(out, "selection.json")))
    n, q = int(sel["chosen"]["n_trees"]), float(sel["chosen"]["quantile"])
    tr, w = _training_with_labels(a)
    models = e7.fit_e7(tr, w, features(), n, q)
    spec = {"config": {"name": "e7_rk8s", "n_iter": n, "q": q, **e7.MODEL_FIXED, "features": "osc", "label": "rk8s ds in [1e-4, 64]"},
            "features": features(), "candidates": ["rk8s"], "n_train_rows": int(len(tr)),
            "train_files": list(C_TRAIN), "sha256": {p: proto.sha256_file(os.path.join(ROOT, p)) for p in C_TRAIN},
            "labels_sha256": proto.sha256_file(labels_path(a))}
    joblib.dump({"models": models, "spec": spec}, os.path.join(out, "policy_frozen.joblib"))
    save_json(spec, os.path.join(out, "policy_frozen_spec.json"))
    print("frozen:", spec["config"])


def cmd_overheads(a):
    import joblib
    from orbit_ml_cr3bp import timing_e as te, compiled_model_3d as cm, e7_policy as e7
    if not a.smoke:
        te.require_idle()
    sel = load_select(a)
    S, T = sel[["x", "y", "z", "vx", "vy", "vz"]].to_numpy(float), sel["sim_time"].to_numpy(float)
    rk4 = cm.CompiledHGBPolicy(joblib.load(os.path.join(ROOT, RK4_FROZEN))["models"], mu=MU)
    m4 = cm.measure_single_call_cost(rk4, S, T, repeats=3 if a.smoke else 9)
    m8 = e7.CompiledE7(joblib.load(os.path.join(base(a), "rk8s", "policy_frozen.joblib"))["models"]).overhead(S, T, repeats=3 if a.smoke else 9)
    rec = {"rk4s": float(m4["rhs_equiv"]), "rk8s": float(m8["rhs_equiv"]), "rk4s_sealed_study_c": 15.036794812388587,
           "measurements": {"rk4s": m4, "rk8s": m8}, "machine": te.machine_info()}
    save_json(rec, os.path.join(base(a), "overheads.json"))
    print({k: rec[k] for k in ("rk4s", "rk8s", "rk4s_sealed_study_c")})


def cmd_gate(a):
    import joblib
    from orbit_ml_cr3bp import ood_e as oe
    tr, _w = load_training(a)
    g = oe.KnnGate().fit(tr, features())
    sel = load_select(a)
    thr = g.calibrate(sel)
    out = os.path.join(base(a), "gate")
    os.makedirs(out, exist_ok=True)
    joblib.dump(g, os.path.join(out, "gate.joblib"))
    save_json(g.spec(), os.path.join(out, "gate_spec.json"))
    print("gate threshold", thr, "on", len(sel), "E_select rows")


def cmd_frozen_list(a):
    files = ["part_c/frozen/sund/policy_frozen.joblib", "part_e/frozen/rk8s/policy_frozen.joblib", "part_e/frozen/gate/gate.joblib",
             "part_e/frozen/overheads.json"]
    for d in ("confirm", "confirm_8", "tol_10", "tol_1000", "lock", "confirm_20", "lock_20", "select"):
        for p in sorted(glob.glob(os.path.join(ROOT, "data_e", d, "*.csv")) + glob.glob(os.path.join(ROOT, "data_e", d, "*metadata.json"))):
            files.append(os.path.relpath(p, ROOT))
    files += sorted(os.path.relpath(p, ROOT) for p in glob.glob(os.path.join(ROOT, "data_d", "jpl", "*.json")))
    missing = [f for f in files if not os.path.exists(os.path.join(ROOT, f))]
    if missing:
        raise SystemExit("missing: " + ", ".join(missing))
    json.dump(files, open(os.path.join(ROOT, "part_e", "FROZEN_FILES.json"), "w"), indent=1)
    print(f"{len(files)} files listed in part_e/FROZEN_FILES.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm, fn in (("e7_labels", cmd_e7_labels), ("e7_select", cmd_e7_select), ("e7_freeze", cmd_e7_freeze), ("overheads", cmd_overheads),
                   ("gate", cmd_gate), ("frozen_list", cmd_frozen_list)):
        sp = sub.add_parser(nm)
        sp.add_argument("--smoke", type=int, default=0)
        sp.add_argument("--n_jobs", type=int, default=8)
        if nm == "e7_select":
            sp.add_argument("--overhead_guess", type=float, default=15.04,
                            help="overhead charged during selection (the E7 overhead is measured after the model is chosen; default: the RK4 one)")
        sp.set_defaults(fn=fn)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
