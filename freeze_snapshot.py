"""
Code snapshot and freeze record for the pre-registration (this folder is not a git repository).

  python3 freeze_snapshot.py snapshot   hash every *.py (CODE_SNAPSHOT.json) and archive a copy
                                        (snapshots/code_<hash12>.tar.gz); prints the hash to paste into
                                        PREREGISTRATION.md
  python3 freeze_snapshot.py seal       after the <<...>> placeholders in PREREGISTRATION.md are filled in: write
                                        FREEZE.json (date, code hash, SHA-256 of the pre-registration text above
                                        '## Deviations', SHA-256 of the data CSVs). Refuses to run twice.
  python3 freeze_snapshot.py check      report whether the code and the pre-registration still match the seal

After the seal, `run_rigor_suite_3d.py final` refuses to run if either changed (override: --allow_post_freeze_changes,
which is recorded in the final-evaluation log). Dated entries appended under '## Deviations' do not change the sealed hash.
"""
import argparse
import glob
import json
import os
import re
import sys
import tarfile
from datetime import datetime, timezone

from orbit_ml_cr3bp import protocol_3d as proto

ROOT = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT_DIR = os.path.join(ROOT, "snapshots")


def cmd_snapshot(a):
    snap_path = os.path.join(ROOT, proto.FROZEN_SNAPSHOT)
    if os.path.exists(os.path.join(ROOT, proto.FREEZE_RECORD)):
        sys.exit("FREEZE.json exists: the code is already sealed. Record changes under '## Deviations' instead.")
    if os.path.exists(snap_path) and not a.force:
        sys.exit(f"{proto.FROZEN_SNAPSHOT} already exists; pass --force to replace it (before sealing only).")
    fp = proto.code_fingerprint(ROOT)
    rec = {"created_utc": datetime.now(timezone.utc).isoformat(), "combined_sha256": fp["combined_sha256"],
           "files": fp["files"], "packages": proto._package_versions()}
    with open(snap_path, "w") as f:
        json.dump(proto.json_safe(rec), f, indent=2)
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    tar_path = os.path.join(SNAPSHOT_DIR, f"code_{fp['combined_sha256'][:12]}.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tf:
        for rel in sorted(fp["files"]):
            tf.add(os.path.join(ROOT, rel), arcname=rel)
    print(f"{len(fp['files'])} source files hashed.")
    print(f"code snapshot SHA-256: {fp['combined_sha256']}")
    print(f"wrote {proto.FROZEN_SNAPSHOT} and {os.path.relpath(tar_path, ROOT)}")
    print("Next: put this hash in PREREGISTRATION.md, fill in the other <<...>> placeholders, then run `seal`.")


def cmd_seal(a):
    rec_path = os.path.join(ROOT, proto.FREEZE_RECORD)
    if os.path.exists(rec_path):
        sys.exit("FREEZE.json already exists; a seal is written once. Add a dated entry under '## Deviations' instead.")
    snap = proto.compare_to_frozen(ROOT)
    if snap is None:
        sys.exit(f"no {proto.FROZEN_SNAPSHOT}; run `snapshot` first.")
    if not snap["matches"]:
        sys.exit("the code changed since `snapshot` (changed: %s; added: %s; removed: %s). Re-run `snapshot --force`."
                 % (snap["changed"] or "-", snap["added"] or "-", snap["removed"] or "-"))
    prereg = os.path.join(ROOT, a.prereg)
    body = open(prereg, encoding="utf-8").read().split(proto.DEVIATIONS_MARKER, 1)[0]
    left = sorted(set(re.findall(r"<<[A-Z0-9_ ]+>>", body)))
    if left:
        sys.exit(f"placeholders still present in {a.prereg}: {left}")
    if snap["frozen_combined_sha256"] not in body:
        sys.exit(f"{a.prereg} does not contain the code snapshot hash {snap['frozen_combined_sha256']}.")
    data_hashes = {os.path.relpath(p, ROOT): proto.sha256_file(p) for p in sorted(glob.glob(os.path.join(ROOT, "data", "*.csv")))}
    rec = {"sealed_utc": datetime.now(timezone.utc).isoformat(), "prereg_file": a.prereg,
           "prereg_body_sha256": proto.prereg_body_sha256(prereg), "prereg_full_sha256": proto.sha256_file(prereg),
           "code_combined_sha256": snap["frozen_combined_sha256"], "data_sha256": data_hashes}
    with open(rec_path, "w") as f:
        json.dump(proto.json_safe(rec), f, indent=2)
    print(f"sealed at {rec['sealed_utc']}")
    print(f"  code {rec['code_combined_sha256'][:16]}...  prereg body {rec['prereg_body_sha256'][:16]}...")


def cmd_check(a):
    st = proto.freeze_status(ROOT, os.path.join(ROOT, a.prereg))
    if st is None:
        snap = proto.compare_to_frozen(ROOT)
        print("not sealed." + ("" if snap is None else
              f" snapshot present; code {'matches' if snap['matches'] else 'DIFFERS from'} it."))
        return
    if st["problems"]:
        print("MISMATCH since the seal at", st["sealed_utc"])
        for p in st["problems"]:
            print("  -", p)
        sys.exit(1)
    print("OK: code and pre-registration body match the seal of", st["sealed_utc"])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot"); s.add_argument("--force", action="store_true"); s.set_defaults(fn=cmd_snapshot)
    z = sub.add_parser("seal"); z.add_argument("--prereg", default="PREREGISTRATION.md"); z.set_defaults(fn=cmd_seal)
    c = sub.add_parser("check"); c.add_argument("--prereg", default="PREREGISTRATION.md"); c.set_defaults(fn=cmd_check)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
