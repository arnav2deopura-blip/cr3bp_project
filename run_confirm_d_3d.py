"""
Confirmatory run of Part D (part_d/PREREGISTRATION_D.md). One command per step; each step runs ONCE per target and is logged in confirm_log_d.jsonl.
A second run of a step needs --allow_rerun "<reason>", recorded here and as a dated entry under '## Deviations' of the pre-registration.

  python run_confirm_d_3d.py steps                          list the steps and whether each has been run
  python run_confirm_d_3d.py run <step> [--target confirm|dry]
  python run_confirm_d_3d.py tests [--target confirm|dry]   D1-D5, Holm, descriptive tables (needs the steps before it)

Steps (target confirm = the D_confirm_D / D_replication / JPL-confirm-sample data of the pre-registration; target dry = D_select_D and the JPL dry sample, a code
check and the source of the written expectations: never counts, never logged, usable before the seal):
  d3_apply_sund, d3_apply_phys, d3_apply_global, d3_c1_resid   Part C's own commands on D_replication (seed 55): the replication of C1-C5
  d1                  ML versus the physics rules on D_confirm_D (seed 53)
  d4                  conditional safety: strata certified on D_confirm_D, evaluated on D_replication
  d5                  the frozen policy on the JPL periodic orbits
  d2                  compiled wall-clock and batch benchmark (REFUSES to start unless the machine is idle: 1-minute load average below --max_load)
  lock                final run of the ML policy and the best rule on D_lock_D with the scales calibrated in d1
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from orbit_ml_cr3bp import protocol_3d as proto  # noqa: E402
from orbit_ml_cr3bp import confirm_c_3d as cc  # noqa: E402
from orbit_ml_cr3bp import confirm_d_3d as cd  # noqa: E402
import freeze_d  # noqa: E402

PY = sys.executable
LOG = os.path.join(ROOT, "confirm_log_d.jsonl")
FROZEN_C = {"phys": "part_c/frozen/phys/policy_frozen.joblib", "sund": "part_c/frozen/sund/policy_frozen.joblib",
            "global": "part_c/frozen/global/policy_frozen.joblib", "partb_refit": "part_c/frozen/partb_refit/policy_frozen.joblib"}
REPL = "data_d/replication/cr3bp_3d_training_data_accuracy_monotone_seed55.csv"
SEL = "data_d/select/cr3bp_3d_training_data_accuracy_monotone_seed52.csv"
ORDER = ["d3_apply_sund", "d3_apply_phys", "d3_apply_global", "d3_c1_resid", "d1", "d4", "d5", "d2", "lock"]


def out_root(target):
    return "out_d/confirm" if target == "confirm" else "out_d/dry"


def _logged():
    return [json.loads(l) for l in open(LOG) if l.strip()] if os.path.exists(LOG) else []


def _record(step, target, cmd, rc, reason=None):
    rec = {"utc": datetime.now(timezone.utc).isoformat(), "step": step, "target": target, "cmd": cmd, "returncode": rc, "rerun_reason": reason,
           "code_sha256": proto.code_fingerprint(ROOT)["combined_sha256"]}
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")


def _latest(out_dir, prefix):
    c = sorted(x for x in glob.glob(os.path.join(ROOT, out_dir, prefix + "_*")) if os.path.isdir(x))
    return c[-1] if c else None


def step_commands(step, target, a):
    root = out_root(target)
    data = REPL if target == "confirm" else SEL
    if step.startswith("d3_apply_"):
        which = step.split("_", 2)[2]
        return [PY, "run_part_c_3d.py", "apply_policy", "--frozen", FROZEN_C[which], "--data", data, "--out_dir", os.path.join(root, "d3", which),
                "--n_cal", "4000", "--n_jobs", str(a.n_jobs)]
    if step == "d3_c1_resid":
        prev = _latest(os.path.join(root, "d3", "phys"), "apply_policy")
        return [PY, "run_part_c_3d.py", "c1_resid", "--data", data, "--phys", FROZEN_C["phys"], "--partb", FROZEN_C["partb_refit"],
                "--cal_ids", os.path.join(prev or "", "cal_row_ids.csv"), "--out_dir", os.path.join(root, "d3", "c1")]
    if step == "d5":
        extra = ["--labels"]
        if target == "confirm":
            extra += ["--recal_from", _latest(os.path.join(out_root("dry"), "d5"), "d5") or ""]
        return [PY, "run_part_d_3d.py", "d5", "--target", target, "--out_dir", os.path.join(root, "d5"), "--n_jobs", str(a.n_jobs)] + extra
    if step in ("d1", "d4", "d2"):
        return [PY, "run_part_d_3d.py", step, "--target", target, "--out_dir", os.path.join(root, step)] + (["--n_jobs", str(a.n_jobs)] if step != "d2" else [])
    if step == "lock":
        prev = _latest(os.path.join(out_root("confirm"), "d1"), "d1")
        return [PY, "run_part_d_3d.py", "lock", "--from_dir", prev, "--out_dir", os.path.join(root, "lock"), "--n_jobs", str(a.n_jobs)]
    raise SystemExit(f"unknown step {step}")


def cmd_steps(a):
    done = {(r["step"], r["target"]) for r in _logged() if r["returncode"] == 0}
    for s in ORDER:
        print(f"{s:16s} confirm:{'done' if (s, 'confirm') in done else '-'}")


def _guard(target):
    if target == "dry":
        return
    st = freeze_d.status()
    if st is None:
        raise SystemExit("Part D is not sealed (FREEZE_D.json missing): run `python freeze_d.py seal` first.")
    if st["problems"]:
        raise SystemExit("code or pre-registration changed since the seal:\n  - " + "\n  - ".join(st["problems"]))


def cmd_run(a):
    _guard(a.target)
    if a.target != "dry":
        prior = [r for r in _logged() if r["step"] == a.step and r["target"] == a.target and r["returncode"] == 0]
        if prior and not a.allow_rerun:
            raise SystemExit(f"step {a.step} on {a.target} already ran at {prior[-1]['utc']}. A rerun needs --allow_rerun '<reason>'.")
    if a.step == "d2":
        load = os.getloadavg()[0]
        if load > a.max_load and not a.allow_busy:
            raise SystemExit(f"d2 is a timing step: the 1-minute load average is {load:.2f} (> {a.max_load}). Stop other jobs, or pass --allow_busy '<reason>' (recorded).")
    cmd = step_commands(a.step, a.target, a)
    print("RUN", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=ROOT)
    if a.target != "dry":
        _record(a.step, a.target, cmd, rc, a.allow_rerun or a.allow_busy)
    if rc:
        raise SystemExit(rc)


def _report(target, step, name):
    d = _latest(os.path.join(out_root(target), step), step)
    return json.load(open(os.path.join(d, name)))


def _ledger_c(target, which):
    d = _latest(os.path.join(out_root(target), "d3", which), "apply_policy")
    return pd.read_csv(os.path.join(d, "ledger.csv"), low_memory=False)


def cmd_tests(a):
    _guard(a.target)
    root = os.path.join(ROOT, out_root(a.target))
    rep = {"target": a.target, "utc": datetime.now(timezone.utc).isoformat()}
    d1 = _report(a.target, "d1", "d1_report.json")
    rep["D1_ml_vs_best_rule"] = d1["D1"]
    rep["D1_arms"] = {k: {kk: v.get(kk) for kk in ("scale", "overhead_rhs", "violation_rate", "viol_cp95", "median_C_tot_met", "share_cheaper_RK45_eqr", "share_cheaper_DOP853_eqr")}
                      for k, v in d1["arms"].items()}
    rep["D1_pairwise_vs_ml"] = d1["pairwise_vs_ml"]
    rep["D2_compiled_throughput"] = _report(a.target, "d2", "d2_report.json")["D2"]
    sund = _ledger_c(a.target, "sund")
    rep["D3_replication"] = cd.d3_replication(sund, "RK45_eqr")
    # the rest of Part C's family on the new data, reported with the same functions (not part of the Holm family)
    phys, glob_ = _ledger_c(a.target, "phys"), _ledger_c(a.target, "global")
    r1 = pd.read_csv(os.path.join(_latest(os.path.join(out_root(a.target), "d3", "c1"), "c1_resid"), "c1_resid.csv"))
    rep["D3_also_C1_prediction"] = cc.c1_prediction(r1["abs_resid_partC"], r1["abs_resid_partB_refit"])
    rep["D3_also_C2_physical_policy"] = cc.c2_policy_share(phys, "RK45_eqr")
    rep["D3_also_C3_sundman_paradigm"] = cc.c3_sundman_paradigm(sund, "RK45_spec")
    rep["D3_also_C5_ml_vs_global"] = cc.c5_ml_vs_global(sund, glob_)
    rep["D4_conditional_safety"] = _report(a.target, "d4", "d4_report.json")["D4"]
    d5 = _report(a.target, "d5", "d5_report.json")
    rep["D5_out_of_distribution"] = d5.get("D5", {"p": 1.0, "note": "no evaluable runs"})
    p = {"D1": rep["D1_ml_vs_best_rule"]["p"], "D2": rep["D2_compiled_throughput"]["p"], "D3": rep["D3_replication"]["p"],
         "D4": rep["D4_conditional_safety"]["p"], "D5": rep["D5_out_of_distribution"]["p"]}
    rep["holm"] = cd.holm(p)
    path = os.path.join(root, f"confirm_tests_{a.target}.json")
    os.makedirs(root, exist_ok=True)
    with open(path, "w") as f:
        json.dump(proto.json_safe(rep), f, indent=2)
    print(json.dumps(proto.json_safe(rep["holm"]), indent=1))
    print("wrote", path)
    if a.target != "dry":
        _record("tests", a.target, ["tests"], 0, a.allow_rerun)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("steps"); s.set_defaults(fn=cmd_steps)
    r = sub.add_parser("run"); r.add_argument("step", choices=ORDER); r.add_argument("--target", choices=["confirm", "dry"], default="confirm")
    r.add_argument("--allow_rerun", default=None); r.add_argument("--allow_busy", default=None); r.add_argument("--max_load", type=float, default=1.5)
    r.add_argument("--n_jobs", type=int, default=8); r.set_defaults(fn=cmd_run)
    t = sub.add_parser("tests"); t.add_argument("--target", choices=["confirm", "dry"], default="confirm")
    t.add_argument("--allow_rerun", default=None); t.set_defaults(fn=cmd_tests)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
