"""
Seal for the Part B pre-registration (part_b/PREREGISTRATION_B.md).

  python freeze_b.py seal    write FREEZE_B.json: code fingerprint (every *.py), SHA-256 of the pre-registration text above
                             '## Deviations', SHA-256 of the frozen model file and of the D_explore CSV, git commit, UTC time.
                             Refuses to run twice and refuses if the working tree has uncommitted *.py or pre-registration changes.
  python freeze_b.py check   report whether the code and the pre-registration body still match the seal (exit 1 if not)

The confirmatory runner (run_confirm_b_3d.py) calls `check` before every non-dry step.
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

from orbit_ml_cr3bp import protocol_3d as proto

ROOT = os.path.dirname(os.path.abspath(__file__))
PREREG = "part_b/PREREGISTRATION_B.md"
RECORD = os.path.join(ROOT, "FREEZE_B.json")
MODEL = "out_b/policy_20261025T213540Z/policy.joblib"
DATA = "data_b/cr3bp_3d_training_data_accuracy_monotone_seed21.csv"


def _git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT).decode().strip()


def status():
    if not os.path.exists(RECORD):
        return None
    rec = json.load(open(RECORD))
    problems = []
    cur = proto.code_fingerprint(ROOT)
    if cur["combined_sha256"] != rec["code_combined_sha256"]:
        old, new = rec["code_files"], cur["files"]
        ch = sorted(k for k in new if k in old and new[k] != old[k]) + ["+" + k for k in new if k not in old] + \
            ["-" + k for k in old if k not in new]
        problems.append("code differs from the seal: " + ", ".join(ch))
    if proto.prereg_body_sha256(os.path.join(ROOT, PREREG)) != rec["prereg_body_sha256"]:
        problems.append("pre-registration text above '## Deviations' changed")
    for rel, h in rec["frozen_files_sha256"].items():
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p) or proto.sha256_file(p) != h:
            problems.append(f"{rel} changed or missing")
    return {"record": rec, "problems": problems}


def cmd_seal(a):
    if os.path.exists(RECORD):
        sys.exit("FREEZE_B.json exists; a seal is written once. Add a dated entry under '## Deviations' instead.")
    dirty = _git("status", "--porcelain", "--", "*.py", PREREG, "tests")
    if dirty:
        sys.exit("uncommitted changes in code or the pre-registration; commit first:\n" + dirty)
    fp = proto.code_fingerprint(ROOT)
    rec = {"sealed_utc": datetime.now(timezone.utc).isoformat(), "prereg_file": PREREG,
           "prereg_body_sha256": proto.prereg_body_sha256(os.path.join(ROOT, PREREG)),
           "prereg_full_sha256": proto.sha256_file(os.path.join(ROOT, PREREG)),
           "code_combined_sha256": fp["combined_sha256"], "code_files": fp["files"],
           "frozen_files_sha256": {MODEL: proto.sha256_file(os.path.join(ROOT, MODEL)),
                                   DATA: proto.sha256_file(os.path.join(ROOT, DATA))},
           "git_commit": _git("rev-parse", "HEAD")}
    json.dump(rec, open(RECORD, "w"), indent=2)
    print("sealed at", rec["sealed_utc"], "git", rec["git_commit"][:12], "code", rec["code_combined_sha256"][:16])


def cmd_check(a):
    st = status()
    if st is None:
        print("not sealed")
        sys.exit(2)
    if st["problems"]:
        print("MISMATCH since the seal at", st["record"]["sealed_utc"])
        for p in st["problems"]:
            print("  -", p)
        sys.exit(1)
    print("OK: code, pre-registration body and frozen files match the seal of", st["record"]["sealed_utc"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("seal").set_defaults(fn=cmd_seal)
    sub.add_parser("check").set_defaults(fn=cmd_check)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
