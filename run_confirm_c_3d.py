"""
Confirmatory run of Part C (part_c/PREREGISTRATION_C.md). One command per step; each step runs ONCE per target and is logged in confirm_log_c.jsonl.
A second run of a step needs --allow_rerun "<reason>", recorded here and as a dated entry under '## Deviations' of the pre-registration.

  python run_confirm_c_3d.py steps                          list the steps and whether each has been run
  python run_confirm_c_3d.py run <step> [--target confirm|lock|dry]
  python run_confirm_c_3d.py tests [--target confirm|dry]   C1-C5, Holm, descriptive tables (needs the steps before it)

Targets: "confirm" = D_confirm_C (seed 43), "lock" = D_lock_C (seed 44), "dry" = D_select (seed 42; a code check on data the choices were made on; never counts,
never written to the log, usable before the seal).
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
import freeze_c  # noqa: E402

PY = sys.executable
LOG = os.path.join(ROOT, "confirm_log_c.jsonl")
FROZEN = {"phys": "part_c/frozen/phys/policy_frozen.joblib", "sund": "part_c/frozen/sund/policy_frozen.joblib",
          "global": "part_c/frozen/global/policy_frozen.joblib", "partb_refit": "part_c/frozen/partb_refit/policy_frozen.joblib"}
TARGETS = {
    "confirm": {"data": "data_c/confirm/cr3bp_3d_training_data_accuracy_monotone_seed43.csv", "out": "out_c/confirm"},
    "lock": {"data": "data_c/lock/cr3bp_3d_training_data_accuracy_monotone_seed44.csv", "out": "out_c/lock"},
    "dry": {"data": "data_c/select/cr3bp_3d_training_data_accuracy_monotone_seed42.csv", "out": "out_c/dry"},
}
ORDER = ["apply_phys", "apply_sund", "apply_global", "c1_resid", "doomed_sund", "doomed_phys", "audit_sund", "audit_phys"]


def _logged():
    return [json.loads(l) for l in open(LOG) if l.strip()] if os.path.exists(LOG) else []


def _record(step, target, cmd, rc, reason=None):
    rec = {"utc": datetime.now(timezone.utc).isoformat(), "step": step, "target": target, "cmd": cmd, "returncode": rc,
           "rerun_reason": reason, "code_sha256": proto.code_fingerprint(ROOT)["combined_sha256"]}
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")


def _latest(out_dir, prefix):
    c = sorted(x for x in glob.glob(os.path.join(ROOT, out_dir, prefix + "_*")) if os.path.isdir(x))
    return c[-1] if c else None


def step_commands(step, target, a):
    t = TARGETS[target]
    out = t["out"]
    if step.startswith("apply_"):
        which = step.split("_", 1)[1]
        cmd = [PY, "run_part_c_3d.py", "apply_policy", "--frozen", FROZEN[which], "--data", t["data"], "--out_dir", os.path.join(out, which),
               "--n_jobs", str(a.n_jobs)]
        if target == "lock":
            prev = _latest(os.path.join(TARGETS["confirm"]["out"], which), "apply_policy")
            cmd += ["--no_cal", "--scale_from", prev]
        elif target == "dry":
            cmd += ["--n_cal", "4000"]
        else:
            cmd += ["--n_cal", "4000"]
        return cmd
    if step.startswith("doomed_") or step.startswith("audit_"):
        which = step.split("_", 1)[1]
        prev = _latest(os.path.join(out, which), "apply_policy")
        if step.startswith("doomed_"):
            return [PY, "run_part_c_3d.py", "doomed", "--frozen", FROZEN[which], "--data", t["data"], "--scale_from", prev, "--out_dir", os.path.join(out, "doomed_" + which)]
        return [PY, "run_part_c_3d.py", "endpoint_audit", "--apply_dir", prev, "--n", "1000", "--out_dir", os.path.join(out, "audit_" + which)]
    if step == "c1_resid":
        return [PY, "run_part_c_3d.py", "c1_resid", "--data", t["data"], "--phys", FROZEN["phys"], "--partb", FROZEN["partb_refit"],
                "--cal_ids", os.path.join(_latest(os.path.join(out, "phys"), "apply_policy") or "", "cal_row_ids.csv"), "--out_dir", os.path.join(out, "c1")]
    raise SystemExit(f"unknown step {step}")


def cmd_steps(a):
    done = {(r["step"], r["target"]) for r in _logged() if r["returncode"] == 0}
    for s in ORDER:
        print(f"{s:13s} confirm:{'done' if (s, 'confirm') in done else '-':5s} lock:{'done' if (s, 'lock') in done else '-'}")


def _guard(target):
    if target == "dry":
        return
    st = freeze_c.status()
    if st is None:
        raise SystemExit("Part C is not sealed (FREEZE_C.json missing): run `python freeze_c.py seal` first.")
    if st["problems"]:
        raise SystemExit("code or pre-registration changed since the seal:\n  - " + "\n  - ".join(st["problems"]))


def cmd_run(a):
    _guard(a.target)
    if a.target != "dry":
        prior = [r for r in _logged() if r["step"] == a.step and r["target"] == a.target and r["returncode"] == 0]
        if prior and not a.allow_rerun:
            raise SystemExit(f"step {a.step} on {a.target} already ran at {prior[-1]['utc']}. A rerun needs --allow_rerun '<reason>'.")
    cmd = step_commands(a.step, a.target, a)
    print("RUN", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=ROOT)
    if a.target != "dry":
        _record(a.step, a.target, cmd, rc, a.allow_rerun)
    if rc:
        raise SystemExit(rc)


def _ledger(target, which):
    d = _latest(os.path.join(TARGETS[target]["out"], which), "apply_policy")
    return pd.read_csv(os.path.join(d, "ledger.csv"), low_memory=False)


def cmd_tests(a):
    _guard(a.target)
    out = os.path.join(ROOT, TARGETS[a.target]["out"])
    rep = {"target": a.target, "utc": datetime.now(timezone.utc).isoformat()}
    phys, sund, glob_ = _ledger(a.target, "phys"), _ledger(a.target, "sund"), _ledger(a.target, "global")
    rep["n_eval_orbits"] = int(len(sund))
    r1 = pd.read_csv(os.path.join(_latest(os.path.join(TARGETS[a.target]["out"], "c1"), "c1_resid"), "c1_resid.csv"))
    p = {}
    rep["C1_prediction"] = cc.c1_prediction(r1["abs_resid_partC"], r1["abs_resid_partB_refit"])
    rep["C2_physical_policy"] = cc.c2_policy_share(phys, "RK45_eqr")
    rep["C3_sundman_paradigm"] = cc.c3_sundman_paradigm(sund, "RK45_spec")
    rep["C4_sundman_policy"] = cc.c2_policy_share(sund, "RK45_eqr")
    rep["C5_ml_vs_global"] = cc.c5_ml_vs_global(sund, glob_)
    for k, name in (("C1", "C1_prediction"), ("C2", "C2_physical_policy"), ("C3", "C3_sundman_paradigm"), ("C4", "C4_sundman_policy"),
                    ("C5", "C5_ml_vs_global")):
        p[name] = rep[name]["p"]
    rep["holm"] = cc.holm(p)
    path = os.path.join(out, f"confirm_tests_{a.target}.json")
    os.makedirs(out, exist_ok=True)
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
    r = sub.add_parser("run"); r.add_argument("step", choices=ORDER); r.add_argument("--target", choices=list(TARGETS), default="confirm")
    r.add_argument("--allow_rerun", default=None); r.add_argument("--n_jobs", type=int, default=8); r.set_defaults(fn=cmd_run)
    t = sub.add_parser("tests"); t.add_argument("--target", choices=["confirm", "dry"], default="confirm")
    t.add_argument("--allow_rerun", default=None); t.set_defaults(fn=cmd_tests)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
