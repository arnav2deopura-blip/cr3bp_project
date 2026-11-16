"""Checks main.tex against the PLOS ONE submission guidelines (journals.plos.org/plosone/s/submission-guidelines, read 2026-11-15) and against
the files of this folder. Prints a report, writes it to build/check_report.txt, and exits 1 when a requirement fails.

Requirements checked (FAIL):   title <= 250 characters; short title <= 100; abstract <= 300 words and without citations; section order
  (Abstract, Introduction, Materials and methods, Results, Discussion, Conclusions, Acknowledgments, References, Supporting information); at most
  three heading levels; no footnotes; line numbers and double spacing; figures and tables numbered consecutively, first cited in ascending
  order, each caption placed after the paragraph that first cites it; no graphics in the submission PDF; one FigN.tif per figure (RGB, no
  alpha, 300-600 dpi, at most 7.5 x 8.75 in, under 10 MB); every \cite key in references.bib; every supporting-information file present.
Reported (WARN): bibliography entries never cited; abbreviations defined in the text and used fewer than three times; supporting-information
  items never cited; author-only placeholders that remain.

  python manuscript/scripts/check_manuscript.py
"""
from __future__ import annotations

import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MAN = os.path.abspath(os.path.join(HERE, ".."))
TEX = open(os.path.join(MAN, "main.tex"), encoding="utf-8").read()
BIB = open(os.path.join(MAN, "references.bib"), encoding="utf-8").read()
SUB = os.path.join(MAN, "submission", "SUBMISSION_STATEMENTS.md")

fails, warns, lines = [], [], []


def ok(msg):
    lines.append(f"  ok    {msg}")


def fail(msg):
    fails.append(msg)
    lines.append(f"  FAIL  {msg}")


def warn(msg):
    warns.append(msg)
    lines.append(f"  warn  {msg}")


def check(cond, msg_ok, msg_fail=None):
    (ok if cond else fail)(msg_ok if cond else (msg_fail or msg_ok))


def strip_comments(t):
    return re.sub(r"(?<!\\)%.*", "", t)


def detex(t):
    """Rough plain text for counting words: maths becomes one token, commands are dropped, their text arguments kept."""
    t = strip_comments(t)
    t = re.sub(r"\$[^$]*\$", " MATH ", t)
    t = re.sub(r"\\(cite|ref|label|nameref)\{[^}]*\}", " ", t)
    t = re.sub(r"\\[a-zA-Z]+\*?", " ", t)
    t = t.replace("--", "-").replace("~", " ")
    return re.sub(r"[{}\\]", " ", t)


body = strip_comments(TEX[TEX.index(r"\begin{document}"):])

# ------------------------------------------------------------------------------------------------- title, short title, abstract
lines.append("Title page and abstract")
m = re.search(r"\\textbf\{(.+?)\}\s*\}\s*\\newline", body, re.S)
title = re.sub(r"\s+", " ", detex(m.group(1))).strip() if m else ""
check(0 < len(title) <= 250, f"title: {len(title)} characters (limit 250)")
if os.path.exists(SUB):
    sm = re.search(r"\*\*Short title[^*]*\*\*[:\s]*(.+)", open(SUB, encoding="utf-8").read())
    short = sm.group(1).strip() if sm else ""
    check(0 < len(short) <= 100, f"short title: {len(short)} characters (limit 100)", f"short title missing or too long ({len(short)})")
else:
    fail("submission/SUBMISSION_STATEMENTS.md not found (short title, statements for the submission form)")
a = body[body.index(r"\section*{Abstract}") + len(r"\section*{Abstract}"):body.index(r"\clearpage")]
n_words = len(detex(a).split())
check(n_words <= 300, f"abstract: {n_words} words (limit 300)")
check(r"\cite" not in a, "abstract has no citations")

# ------------------------------------------------------------------------------------------------- structure
lines.append("Structure")
want = ["Abstract", "Introduction", "Materials and methods", "Results", "Discussion", "Conclusions", "Acknowledgments", "Supporting information"]
got = re.findall(r"\\section\*\{([^}]*)\}", body)
check(got == want, "section order: " + " > ".join(got), f"section order is {got}, expected {want}")
check(body.index(r"\section*{Acknowledgments}") < body.index(r"\bibliography{") < body.index(r"\section*{Supporting information}"),
      "ending order: Acknowledgments, References, Supporting information captions")
check(not re.search(r"\\(paragraph|subparagraph)\*?\{", body[:body.index(r"\section*{Supporting information}")]),
      "at most three heading levels in the main text (section, subsection, subsubsection)")
check(not re.search(r"\\footnote(?!size)", body), "no footnotes")          # \footnotesize is a font size, not a footnote
check(r"\linenumbers" in body and r"\doublespacing" in TEX, "continuous line numbers and double spacing")
check(r"\input{" not in body and r"\include{" not in body, "single .tex file (no \\input or \\include)")
unguarded = [l for l in body.splitlines() if r"\includegraphics" in l and not l.strip().startswith(r"\ifdraftfigs")]
check(not unguarded, "no graphics in the submission PDF (every \\includegraphics is inside \\ifdraftfigs)")
check(re.search(r"^\\draftfigsfalse", TEX, re.M) is not None, "\\draftfigsfalse is the default")

# ------------------------------------------------------------------------------------------------- figures and tables
for kind, env, word, key in (("Figures", "figure", "Fig", "fig"), ("Tables", "table", "Table", "table")):
    lines.append(kind)
    labels = [int(x) for x in re.findall(r"\\label\{%s(\d+)\}" % key, body)]
    check(labels == list(range(1, len(labels) + 1)), f"{len(labels)} {kind.lower()}, labelled consecutively in order of appearance",
          f"{kind.lower()} labels out of order: {labels}")
    first = {}
    for mm in re.finditer(r"%s~\\ref\{%s(\d+)\}" % (word, key), body):
        first.setdefault(int(mm.group(1)), mm.start())
    order = sorted(first, key=first.get)
    check(order == sorted(order) and set(order) == set(labels), f"every {word} is cited, first citations in ascending order",
          f"first citations of {kind.lower()} in order {order}; labels {labels}")
    envs = [mm.start() for mm in re.finditer(r"\\begin\{%s\}" % env, body)]
    late = [n for n, pos in zip(labels, envs) if n in first and first[n] > pos]
    check(not late, f"each {word} caption follows the paragraph that first cites it", f"{word} {late} appear before their first citation")
    far = []
    for n, pos in zip(labels, envs):
        if n in first:
            between = body[first[n]:pos]
            if len(re.findall(r"\n\s*\n", between)) > 1 + between.count(r"\end{figure}") + between.count(r"\end{table}"):
                far.append(n)
    check(not far, f"each {word} is placed directly after its first-citing paragraph", f"{word} {far} are not directly after the citing paragraph")
    if key == "fig":
        try:
            from PIL import Image
        except ImportError:
            Image = None
            warn("Pillow not available: TIFF properties not checked")
        for n in labels:
            p = os.path.join(MAN, "figures", f"Fig{n}.tif")
            if not os.path.exists(p):
                fail(f"figures/Fig{n}.tif missing")
                continue
            if Image is None:
                continue
            im = Image.open(p)
            dpi = float(im.info.get("dpi", (0, 0))[0])
            w, h, mb = im.width / max(dpi, 1), im.height / max(dpi, 1), os.path.getsize(p) / 1e6
            good = im.mode == "RGB" and 300 <= dpi <= 600 and w <= 7.5 and h <= 8.75 and mb < 10
            check(good, f"Fig{n}.tif: {im.mode}, {dpi:.0f} dpi, {w:.2f} x {h:.2f} in, {mb:.2f} MB")

# ------------------------------------------------------------------------------------------------- references
lines.append("References")
cited = []
for mm in re.finditer(r"\\cite\{([^}]*)\}", body):
    for k in mm.group(1).split(","):
        if k.strip() not in cited:
            cited.append(k.strip())
keys = re.findall(r"^@\w+\{([^,]+),", BIB, re.M)
missing = [k for k in cited if k not in keys]
check(not missing, f"{len(cited)} cited keys, all in references.bib ({len(keys)} entries)", f"cited but not in references.bib: {missing}")
unused = [k for k in keys if k not in cited]
(warn if unused else ok)(f"bibliography entries never cited: {unused}" if unused else "every bibliography entry is cited")
check("plos2025" in TEX, "bibliography style plos2025.bst (Vancouver, numbered in order of appearance)")
nodoi = [k for k, blk in zip(keys, re.split(r"^@\w+\{", BIB, flags=re.M)[1:]) if "doi" not in blk.lower()]
lines.append(f"  note  entries without a DOI (books, reports, software, pre-DOI articles): {nodoi}")

# ------------------------------------------------------------------------------------------------- supporting information
lines.append("Supporting information")
si_labels = re.findall(r"\\paragraph\*\{(S\d+ [A-Za-z]+)\.\}\s*\\label\{([^}]+)\}", body)
files = {"S1 Appendix": ["si/S1_Appendix.pdf"], "S1 Table": ["si/S1_Table.zip"], "S2 Table": ["si/S2_Table.zip"], "S3 Table": ["si/S3_Table.zip"],
         "S4 Table": ["si/S4_Table.zip"], "S1 File": ["si/S1_File.zip"], "S1 Data": ["si/S1_Data.zip"]}
for name, lab in si_labels:
    pats = files.get(name)
    found = sum(len(glob.glob(os.path.join(MAN, p))) for p in pats) if pats else 0
    check(found > 0, f"{name}: {found} file(s)", f"{name}: no file found ({pats}); run ./build.sh si")
    if body.count(r"\nameref{%s}" % lab) == 0:
        warn(f"{name} is never cited in the text")
for p in glob.glob(os.path.join(MAN, "si", "*")):
    if os.path.getsize(p) > 20e6:
        fail(f"{os.path.relpath(p, MAN)} is over 20 MB")

# ------------------------------------------------------------------------------------------------- abbreviations and placeholders
lines.append("Abbreviations (PLOS: define at first use; non-standard abbreviations only if used at least three times)")
main_text = body[:body.index(r"\section*{Supporting information}")]
plain = detex(main_text)
abbr = sorted(set(re.findall(r"\(([A-Z][A-Za-z0-9]{1,7})\)", plain)) - {"A", "B", "C", "D", "E", "MATH"})
abbr = [x for x in abbr if not re.fullmatch(r"[A-Z]\d+[a-z]?", x)]          # names of tests (T3, C4, E1a), not abbreviations
for ab in abbr:
    n = len(re.findall(r"(?<![A-Za-z0-9])%s(?![A-Za-z0-9])" % re.escape(ab), plain))
    (ok if n >= 3 else warn)(f"{ab}: used {n} times")
lines.append("Placeholders that only the author can fill")
ph = re.findall(r"\[(?:TO BE WRITTEN[^\]]*|repository DOI[^\]]*|Department/Program|Institution|City|State|Country|corresponding author email)\]", body)
for p_ in sorted(set(ph)):
    warn(f"placeholder remains: {p_[:90]}")
if not ph:
    ok("none")

report = "\n".join(["PLOS ONE checks of manuscript/main.tex", ""] + lines + ["", f"{len(fails)} failed, {len(warns)} warnings"])
print(report)
os.makedirs(os.path.join(MAN, "build"), exist_ok=True)
open(os.path.join(MAN, "build", "check_report.txt"), "w", encoding="utf-8").write(report + "\n")
sys.exit(1 if fails else 0)
