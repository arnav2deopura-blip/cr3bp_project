"""
Confirmatory run of Part B (PREREGISTRATION_B.md). One command per step; each step runs ONCE per target set and is logged in
confirm_log_b.jsonl (the same discipline as Part A's `final`). A second run of a step needs --allow_rerun "<reason>", which is
recorded and must also appear as a dated entry under '## Deviations' in the pre-registration.

  python run_confirm_b_3d.py steps                       list the steps and whether each has been run
  python run_confirm_b_3d.py run <step> [--target confirm|lock|dry]   run one step
  python run_confirm_b_3d.py tests                       T1-T5, Holm, descriptive tables (needs the steps before it)

Targets: "confirm" = D_confirm (seed 22), "lock" = D_lock (seed 23), "dry" = a dry run on D_explore (code check; never counts,
never written to the log; used before the seal).
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
from orbit_ml_cr3bp import confirm_b_3d as cb  # noqa: E402
from orbit_ml_cr3bp import failure_study_3d as fs  # noqa: E402
import freeze_b  # noqa: E402

PY = sys.executable
LOG = os.path.join(ROOT, "confirm_log_b.jsonl")
C_PRED_FIXED = 1128.0     # RHS-equivalents per orbit: median of five isolated measurements (PREREGISTRATION_B.md, section 3)
EXPLORE_DATA = "data_b/cr3bp_3d_training_data_accuracy_monotone_seed21.csv"
EXPLORE_POLICY = "out_b/policy_20261025T213540Z"
TARGETS = {
    "confirm": {"data": "data_b/cr3bp_3d_training_data_accuracy_monotone_seed22.csv", "out": "out_b/confirm"},
    "lock": {"data": "data_b/cr3bp_3d_training_data_accuracy_monotone_seed23.csv", "out": "out_b/confirm"},
    "dry": {"data": EXPLORE_DATA, "out": "out_b/dry"},
}


def _logged():
    if not os.path.exists(LOG):
        return []
    return [json.loads(l) for l in open(LOG) if l.strip()]


def _record(step, target, cmd, rc, reason=None):
    rec = {"utc": datetime.now(timezone.utc).isoformat(), "step": step, "target": target, "cmd": cmd, "returncode": rc,
           "rerun_reason": reason, "code_sha256": proto.code_fingerprint(ROOT)["combined_sha256"]}
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")


def _latest(out_dir, prefix):
    c = sorted(glob.glob(os.path.join(ROOT, out_dir, prefix + "_*")))
    c = [x for x in c if os.path.isdir(x)]
    return c[-1] if c else None


def step_commands(step, target, a):
    t = TARGETS[target]
    out = t["out"]
    if step == "policy":    # confirm: frozen D_explore model, fresh calibration on 2,000 rows of D_confirm, every other usable row evaluated
        return [PY, "run_failure_study_3d.py", "policy", "--data", t["data"], "--out_dir", out, "--n_cal", "2000", "--n_eval", "1500" if target == "dry" else "0",
                "--frozen_policy", EXPLORE_POLICY, "--c_pred_fixed", str(C_PRED_FIXED), "--n_jobs", str(a.n_jobs)]
    pol = _latest(out, "policy")
    if step == "policy_lock":   # D_lock: the deployed confirm policy, scales and rtols exactly as calibrated on D_confirm
        cpol = _latest(TARGETS["confirm"]["out"], "policy")
        return [PY, "run_failure_study_3d.py", "policy", "--data", TARGETS["lock"]["data"], "--out_dir", out, "--n_cal", "0",
                "--n_eval", "0", "--frozen_policy", cpol, "--frozen_calibration", "--c_pred_fixed", str(C_PRED_FIXED),
                "--n_jobs", str(a.n_jobs)]
    if step == "decompose":
        return [PY, "run_failure_study_3d.py", "decompose", "--ledger", os.path.join(pol, "ledger.csv"), "--out_dir", out]
    if step == "e1":
        return [PY, "run_failure_study_3d.py", "e1", "--data", t["data"], "--out_dir", out, "--n_jobs", str(a.n_jobs)] + \
            (["--n_ok", "30", "--n_doomed", "15"] if target == "dry" else [])
    if step == "e3a":
        return [PY, "run_failure_study_3d.py", "e3a", "--ledger", os.path.join(pol, "ledger.csv"), "--out_dir", out,
                "--n_jobs", str(a.n_jobs)] + (["--per_cell", "2", "--clones", "4"] if target == "dry" else [])
    if step in ("e5", "e9", "e10"):
        extra = ["--c_pred", str(C_PRED_FIXED)] if step == "e5" else []
        return [PY, "run_failure_experiments_3d.py", step, "--data", t["data"], "--policy_dir", pol, "--out_dir", out,
                "--n_jobs", str(a.n_jobs), "--train_data", EXPLORE_DATA, "--train_policy_dir", EXPLORE_POLICY] + extra
    if step == "sweep":
        return [PY, "make_sweep_b.py", "--source", t["data"], "--out", os.path.join("data_b", "sweeps", f"hsweep_{target}.csv"),
                "--policy_dir", pol, "--seed", "1", "--n_jobs", str(a.n_jobs)] + (["--n_ics", "120"] if target == "dry" else [])
    if step == "policy_sweep":
        return [PY, "run_failure_study_3d.py", "policy", "--data", os.path.join("data_b", "sweeps", f"hsweep_{target}.csv"),
                "--out_dir", os.path.join(out, "sweep"), "--n_cal", "0", "--n_eval", "0", "--frozen_policy", pol,
                "--frozen_calibration", "--c_pred_fixed", str(C_PRED_FIXED), "--n_jobs", str(a.n_jobs)]
    if step == "e11":
        lock_pol = _latest(TARGETS["lock"]["out"], "policy")
        return [PY, "run_failure_experiments_3d.py", "e11", "--gate_ledger", os.path.join(EXPLORE_POLICY, "ledger.csv"),
                "--lock_ledger", os.path.join(lock_pol, "ledger.csv"), "--out_dir", out]
    if step == "e11b":
        lock_pol = _latest(TARGETS["lock"]["out"], "policy")
        cpol = _latest(TARGETS["confirm"]["out"], "policy")
        return [PY, "run_failure_experiments_3d.py", "e11b", "--train_ledger", os.path.join(cpol, "ledger.csv"),
                "--test_ledger", os.path.join(lock_pol, "ledger.csv"), "--out_dir", out]
    raise SystemExit(f"unknown step {step}")


ORDER = ["policy", "decompose", "e1", "e3a", "e5", "e9", "e10", "sweep", "policy_sweep", "policy_lock", "e11", "e11b"]


def cmd_steps(a):
    done = {(r["step"], r["target"]) for r in _logged() if r["returncode"] == 0}
    for s in ORDER:
        print(f"{s:13s} confirm:{'done' if (s, 'confirm') in done else '-':5s} lock:{'done' if (s, 'lock') in done else '-'}")


def cmd_run(a):
    _guard(a.target)
    if a.target != "dry":
        prior = [r for r in _logged() if r["step"] == a.step and r["target"] == a.target and r["returncode"] == 0]
        if prior and not a.allow_rerun:
            raise SystemExit(f"step {a.step} on {a.target} already ran at {prior[-1]['utc']}. A rerun needs --allow_rerun '<reason>' "
                             f"(recorded; also add a dated entry under '## Deviations').")
    cmd = step_commands(a.step, a.target, a)
    print("RUN", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=ROOT)
    if a.target != "dry":
        _record(a.step, a.target, cmd, rc, a.allow_rerun)
    if rc:
        raise SystemExit(rc)


def _guard(target):
    """Confirmatory targets need the seal, and the code and pre-registration must still match it."""
    if target == "dry":
        return
    st = freeze_b.status()
    if st is None:
        raise SystemExit("Part B is not sealed (FREEZE_B.json missing): run `python freeze_b.py seal` first.")
    if st["problems"]:
        raise SystemExit("code or pre-registration changed since the seal:\n  - " + "\n  - ".join(st["problems"]))


def _load(path):
    return pd.read_csv(path, low_memory=False)


def cmd_tests(a):
    _guard(a.target)
    t = TARGETS[a.target]
    out = os.path.join(ROOT, t["out"])
    pol = _latest(t["out"], "policy")
    led = _load(os.path.join(pol, "ledger.csv"))
    led["cell"] = fs.assign_cells(led)
    rep = {"target": a.target, "utc": datetime.now(timezone.utc).isoformat(), "n_orbits": int(len(led))}
    p = {}
    rep["T1_paradigm"] = cb.t1_paradigm(led)
    p["T1_paradigm"] = rep["T1_paradigm"]["p"]
    e3a = _latest(t["out"], "e3a")
    sig = _load(os.path.join(e3a, "e3a_sigma.csv"))
    rep["T2_information"] = cb.t2_information(sig)
    p["T2_information"] = rep["T2_information"]["p"]
    e5 = _latest(t["out"], "e5")
    g = _load(os.path.join(e5, "e5_orbits_1_global.csv"))
    m = _load(os.path.join(e5, "e5_orbits_2_mondrian_regime.csv"))
    rep["T3_calibration_shape"] = cb.t3_calibration_shape(led, g, m)
    p["T3_calibration_shape"] = rep["T3_calibration_shape"]["p"]
    rep["T4_where"] = cb.t4_where(led)
    p["T4_where"] = rep["T4_where"]["p"]
    spol = _latest(os.path.join(t["out"], "sweep"), "policy")
    sled = _load(os.path.join(spol, "ledger.csv"))
    rep["T5_horizon_dial"] = cb.t5_horizon_dial(sled)
    p["T5_horizon_dial"] = rep["T5_horizon_dial"]["p"]
    rep["holm"] = cb.holm(p)
    path = os.path.join(out, f"confirm_tests_{a.target}.json")
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
