"""
Seal for the Part E pre-registration (part_e/PREREGISTRATION_E.md). Same mechanism as freeze_d.py, with three differences required by the draft:

  python freeze_e.py seal     refuses unless
                                - FREEZE_E.json does not exist (a seal is written once),
                                - the git tree is clean (`git status --porcelain` is empty: everything committed or ignored),
                                - part_e/OSF_LINKS.json holds the OSF registration URL of the pre-registration (registered before the dry run),
                                - the pre-registration no longer carries the DRAFT status line,
                                - every file of part_e/FROZEN_FILES.json exists;
                              then writes FREEZE_E.json: the SHA-256 of the pre-registration text above '## Deviations' and of the whole file, the code
                              fingerprint (every *.py), the SHA-256 of every frozen file (frozen policies, gate, pinned overheads, the Part E datasets,
                              the JPL responses), the git HEAD and the OSF registration URL, and prints the `git bundle` command for the OSF upload.
  python freeze_e.py check    reports whether the code, the pre-registration body and the frozen files still match the seal (exit 1 if not)

After the seal: upload FREEZE_E.json and the bundle to the OSF project, then record their OSF links in part_e/OSF_LINKS.json ("seal_record",
"bundle") and commit that file. FREEZE_E.json itself is never edited (its hash is what was uploaded).
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

from orbit_ml_cr3bp import protocol_3d as proto

ROOT = os.path.dirname(os.path.abspath(__file__))
PREREG = "part_e/PREREGISTRATION_E.md"
RECORD = os.path.join(ROOT, "FREEZE_E.json")
FROZEN_LIST = os.path.join(ROOT, "part_e", "FROZEN_FILES.json")
OSF_LINKS = os.path.join(ROOT, "part_e", "OSF_LINKS.json")
DRAFT_MARK = "Status: **DRAFT"


def _git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT).decode().strip()


def frozen_files():
    return json.load(open(FROZEN_LIST))


def preconditions():
    """The list of reasons the seal would refuse (empty = ready)."""
    bad = []
    if os.path.exists(RECORD):
        bad.append("FREEZE_E.json exists; a seal is written once (add a dated entry under '## Deviations' instead)")
    dirty = _git("status", "--porcelain")
    if dirty:
        bad.append("the git tree is not clean:\n      " + "\n      ".join(dirty.splitlines()))
    if not os.path.exists(OSF_LINKS) or not json.load(open(OSF_LINKS)).get("registration", "").startswith("https://osf.io/"):
        bad.append("part_e/OSF_LINKS.json has no OSF registration URL ({\"registration\": \"https://osf.io/...\"})")
    if DRAFT_MARK in open(os.path.join(ROOT, PREREG)).read():
        bad.append("the pre-registration still carries the DRAFT status line")
    if not os.path.exists(FROZEN_LIST):
        bad.append("part_e/FROZEN_FILES.json is missing (run_part_e_3d.py frozen_list)")
    else:
        missing = [f for f in frozen_files() if not os.path.exists(os.path.join(ROOT, f))]
        if missing:
            bad.append("frozen files missing: " + ", ".join(missing))
    return bad


def status():
    if not os.path.exists(RECORD):
        return None
    rec = json.load(open(RECORD))
    problems = []
    cur = proto.code_fingerprint(ROOT)
    if cur["combined_sha256"] != rec["code_combined_sha256"]:
        old, new = rec["code_files"], cur["files"]
        ch = sorted(k for k in new if k in old and new[k] != old[k]) + ["+" + k for k in new if k not in old] + ["-" + k for k in old if k not in new]
        problems.append("code differs from the seal: " + ", ".join(ch))
    if proto.prereg_body_sha256(os.path.join(ROOT, PREREG)) != rec["prereg_body_sha256"]:
        problems.append("pre-registration text above '## Deviations' changed")
    for rel, h in rec["frozen_files_sha256"].items():
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p) or proto.sha256_file(p) != h:
            problems.append(f"{rel} changed or missing")
    return {"record": rec, "problems": problems}


def cmd_seal(a):
    bad = preconditions()
    if bad:
        sys.exit("refusing to seal:\n  - " + "\n  - ".join(bad))
    fp = proto.code_fingerprint(ROOT)
    rec = {"sealed_utc": datetime.now(timezone.utc).isoformat(), "prereg_file": PREREG,
           "prereg_body_sha256": proto.prereg_body_sha256(os.path.join(ROOT, PREREG)),
           "prereg_full_sha256": proto.sha256_file(os.path.join(ROOT, PREREG)),
           "code_combined_sha256": fp["combined_sha256"], "code_files": fp["files"],
           "frozen_files_sha256": {rel: proto.sha256_file(os.path.join(ROOT, rel)) for rel in frozen_files()},
           "git_commit": _git("rev-parse", "HEAD"), "git_clean": True,
           "osf_registration": json.load(open(OSF_LINKS))["registration"]}
    json.dump(rec, open(RECORD, "w"), indent=2)
    print("sealed at", rec["sealed_utc"], "git", rec["git_commit"][:12], "code", rec["code_combined_sha256"][:16], "OSF", rec["osf_registration"])
    print("next: commit FREEZE_E.json, then upload it and the bundle to the OSF project before any confirmatory run:")
    print("  git add FREEZE_E.json && git commit -m 'Seal Part E pre-registration (FREEZE_E.json)'")
    print("  git bundle create study_e_sealed.bundle HEAD")


def cmd_check(a):
    st = status()
    if st is None:
        print("not sealed")
        bad = preconditions()
        print("seal preconditions:", "ready" if not bad else "\n  - " + "\n  - ".join(bad))
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
