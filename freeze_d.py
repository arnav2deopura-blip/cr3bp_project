"""
Seal for the Part D pre-registration (part_d/PREREGISTRATION_D.md). Same mechanism as freeze_c.py.

  python freeze_d.py seal    write FREEZE_D.json: code fingerprint (every *.py), SHA-256 of the pre-registration text above '## Deviations', SHA-256 of every file
                             listed in part_d/FROZEN_FILES.json (the frozen policies, the datasets, the JPL responses), the git HEAD and the list of files that
                             differ from it at the time of the seal. Refuses to run twice.
  python freeze_d.py check   report whether the code, the pre-registration body and the frozen files still match the seal (exit 1 if not)

Difference from freeze_c.py: the seal does not require a clean git tree (nothing is committed on the user's behalf); the code is identified by the SHA-256 of
every source file, and the dirty-file list is recorded next to the HEAD commit, so the sealed tree can be reconstructed from the commit plus the files.
The confirmatory runner (run_confirm_d_3d.py) calls `check` before every non-dry step.
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

from orbit_ml_cr3bp import protocol_3d as proto

ROOT = os.path.dirname(os.path.abspath(__file__))
PREREG = "part_d/PREREGISTRATION_D.md"
RECORD = os.path.join(ROOT, "FREEZE_D.json")
FROZEN_LIST = os.path.join(ROOT, "part_d", "FROZEN_FILES.json")


def _git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT).decode().strip()


def frozen_files():
    return json.load(open(FROZEN_LIST))


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
        sys.exit("FREEZE_D.json exists; a seal is written once. Add a dated entry under '## Deviations' instead.")
    fp = proto.code_fingerprint(ROOT)
    rec = {"sealed_utc": datetime.now(timezone.utc).isoformat(), "prereg_file": PREREG,
           "prereg_body_sha256": proto.prereg_body_sha256(os.path.join(ROOT, PREREG)),
           "prereg_full_sha256": proto.sha256_file(os.path.join(ROOT, PREREG)),
           "code_combined_sha256": fp["combined_sha256"], "code_files": fp["files"],
           "frozen_files_sha256": {rel: proto.sha256_file(os.path.join(ROOT, rel)) for rel in frozen_files()},
           "git_commit": _git("rev-parse", "HEAD"),
           "git_dirty_files": [l for l in _git("status", "--porcelain").splitlines()]}
    json.dump(rec, open(RECORD, "w"), indent=2)
    print("sealed at", rec["sealed_utc"], "git", rec["git_commit"][:12], f"(+{len(rec['git_dirty_files'])} uncommitted paths)", "code", rec["code_combined_sha256"][:16])


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
