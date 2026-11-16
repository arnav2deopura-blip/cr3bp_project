"""Writes the files PLOS ONE asks for from a LaTeX submission into manuscript/build/submission/:

  main.tex      main.tex with the contents of build/main.bbl pasted in place of \\bibliography{references} (PLOS: one .tex file, references
                included in the file, no separate .bib), otherwise byte-identical to manuscript/main.tex
  Fig1.tif ...  the figure files, uploaded separately (the PDF contains no graphics)
  S*            the supporting-information files

build.sh compiles this main.tex on its own afterwards, which shows that it needs no other file. Run after `./build.sh pdf`.

  python manuscript/scripts/make_submission.py
"""
from __future__ import annotations

import glob
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
MAN = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(MAN, "build", "submission")

tex = open(os.path.join(MAN, "main.tex"), encoding="utf-8").read()
bbl_path = os.path.join(MAN, "build", "main.bbl")
if not os.path.exists(bbl_path):
    raise SystemExit("build/main.bbl not found: run ./build.sh pdf first")
bbl = open(bbl_path, encoding="utf-8").read().strip()
anchor = "\n\\bibliography{references}\n"            # the command on its own line (the header comment also names it)
assert tex.count(anchor) == 1, "expected exactly one \\bibliography{references} line"
assert r"\input{" not in tex and r"\include{" not in tex
if os.path.isdir(OUT):
    shutil.rmtree(OUT)
os.makedirs(os.path.join(OUT, "supporting_information"))
note = "% References pasted from main.bbl (BibTeX, plos2025.bst, references.bib) by scripts/make_submission.py.\n"
open(os.path.join(OUT, "main.tex"), "w", encoding="utf-8").write(tex.replace(anchor, "\n" + note + bbl + "\n"))
n_fig = 0
for p in sorted(glob.glob(os.path.join(MAN, "figures", "Fig*.tif"))):
    shutil.copy2(p, OUT)
    n_fig += 1
n_si = 0
for p in sorted(glob.glob(os.path.join(MAN, "si", "S*"))):
    if p.endswith((".zip", ".pdf")):                      # one file per supporting item: the appendix PDF and the archives
        shutil.copy2(p, os.path.join(OUT, "supporting_information"))
        n_si += 1
print(f"build/submission/: main.tex ({bbl.count(chr(92) + 'bibitem')} references pasted), {n_fig} figure files, {n_si} supporting files")
