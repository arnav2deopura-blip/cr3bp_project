"""Supporting-information archives, built from tracked files only:

  si/S1_File.zip   the preregistration documents of Studies A-D as sealed, their seal manifests, and the definitions of the post hoc analyses,
                   with a MANIFEST.txt of SHA-256 hashes
  si/S1_Data.zip   the values behind every panel of Figs 1-10 (figures/data/*.csv), with a MANIFEST.txt
  si/SN_Table.zip  for N = 1..4, the CSV files of that supporting table (si/SN_Table_*.csv) as one file, because PLOS takes one file per
                   supporting item

The archives are deterministic: entries are sorted, carry a fixed timestamp and fixed permissions, and are stored with the same compression
level, so rebuilding from the same inputs gives the same bytes (the script prints each archive's SHA-256).

  python manuscript/scripts/si_archives.py
"""
from __future__ import annotations

import glob
import hashlib
import os
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
MAN = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(MAN, ".."))
FIXED = (2026, 1, 1, 0, 0, 0)

S1_FILE = [
    ("PREREGISTRATION.md", "study_A/PREREGISTRATION.md"),
    ("FREEZE.json", "study_A/FREEZE.json"),
    ("CODE_SNAPSHOT.json", "study_A/CODE_SNAPSHOT.json"),
    ("part_b/PREREGISTRATION_B.md", "study_B/PREREGISTRATION_B.md"),
    ("FREEZE_B.json", "study_B/FREEZE_B.json"),
    ("FREEZE_B.v1.json", "study_B/FREEZE_B.v1.json"),
    ("part_c/PREREGISTRATION_C.md", "study_C/PREREGISTRATION_C.md"),
    ("FREEZE_C.json", "study_C/FREEZE_C.json"),
    ("part_d/PREREGISTRATION_D.md", "study_D/PREREGISTRATION_D.md"),
    ("FREEZE_D.json", "study_D/FREEZE_D.json"),
    ("posthoc/README.md", "post_hoc/README.md"),
]
S1_FILE_NOTE = """S1 File. Preregistration documents and seal manifests.

study_A .. study_D   the preregistration of each study as sealed (text above "## Deviations" was hashed by the seal; later changes are the
                     dated entries below that heading) and the seal manifest written by the study's freeze script.
post_hoc             the definitions of the post hoc and exploratory analyses (P1-P23), committed before each computation.

The seals are local hash manifests; their timestamps cannot be verified independently (S1 Appendix, section 1).
"""


def sha(b):
    return hashlib.sha256(b).hexdigest()


def build(name, entries, note):
    """entries: list of (source path, name inside the archive)."""
    out = os.path.join(MAN, "si", name)
    payload = []
    for src, arc in sorted(entries, key=lambda e: e[1]):
        with open(src, "rb") as f:
            payload.append((arc, f.read()))
    manifest = note + "\nSHA-256                                                           bytes  file\n" + "".join(
        f"{sha(b)}  {len(b):>9}  {arc}\n" for arc, b in payload)
    payload.insert(0, ("MANIFEST.txt", manifest.encode("utf-8")))
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for arc, b in payload:
            zi = zipfile.ZipInfo(arc, date_time=FIXED)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            z.writestr(zi, b, compresslevel=9)
    with open(out, "rb") as f:
        digest = sha(f.read())
    print(f"si/{name}  {len(payload) - 1} files  {os.path.getsize(out) / 1e6:.2f} MB  sha256 {digest}")


if __name__ == "__main__":
    build("S1_File.zip", [(os.path.join(ROOT, s), a) for s, a in S1_FILE], S1_FILE_NOTE)
    data = sorted(glob.glob(os.path.join(MAN, "figures", "data", "Fig*.csv")))
    build("S1_Data.zip", [(p, os.path.basename(p)) for p in data],
          "S1 Data. The values behind every panel of Figs 1-10, written by manuscript/scripts/figNN.py next to each figure.\n")
    titles = {1: "Conditional certificate by stratum (D4).", 2: "Periodic-orbit families (D5).", 3: "Full regression results (D1 rules, T1 fit).",
              4: "Post hoc cost summaries of the sealed ledgers."}
    for n, title in titles.items():
        parts = sorted(glob.glob(os.path.join(MAN, "si", f"S{n}_Table_*.csv")))
        if not parts:
            print(f"si/S{n}_Table.zip  skipped: no si/S{n}_Table_*.csv (run the table scripts first)")
            continue
        build(f"S{n}_Table.zip", [(p, os.path.basename(p)) for p in parts], f"S{n} Table. {title} One CSV file per section of the table.\n")
