#!/bin/zsh
# Rebuild everything in manuscript/ that is not a source file. Sources: main.tex, references.bib, plos2025.bst, si/S1_Appendix.tex,
# scripts/*.py, submission/SUBMISSION_STATEMENTS.md. Everything else here is written by one of the targets below (README.md has the table).
#
#   ./build.sh figures      Fig 1-10: figures/FigN.tif (the files to upload), figures/draft/FigN.png, figures/data/*.csv
#   ./build.sh si           S1, S2, S4 Tables (CSV), S1 Appendix (PDF), S1 File and S1 Data (zip)
#   ./build.sh si-full      the same plus S3 Table, whose bootstrap intervals take 100 refits (about 40 minutes)
#   ./build.sh pdf          main.pdf, the submission PDF (no graphics, as PLOS asks of LaTeX submissions)
#   ./build.sh draft        main_draft.pdf, the same text with the figures shown under their captions
#   ./build.sh submission   build/submission/: single-file main.tex with the references pasted in, compiled on its own, plus figures and SI
#   ./build.sh check        the PLOS ONE checks (scripts/check_manuscript.py); report in build/check_report.txt
#   ./build.sh all          figures, si, pdf, draft, submission, check, in that order
#   ./build.sh clean        delete build/ (intermediate LaTeX files, the submission copy, the check report)
#
# figures and si read the result files of the studies, including the git-ignored run folders out_c/ and out_d/ (deposited with the
# datasets); pdf, draft, submission and check need only this folder and a TeX distribution with latexmk.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-/Users/arnavdeopura/miniforge3/envs/orbit-ml/bin/python}
LATEXMK=(latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build)

figures() {
  (cd scripts && for f in fig01 fig02 fig03 fig04 fig05 fig06 fig07 fig08 fig09 fig10; do $PY $f.py; done)
}
appendix() {
  mkdir -p build/si
  (cd si && for i in 1 2; do pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../build/si S1_Appendix.tex > /dev/null; done)
  cp build/si/S1_Appendix.pdf si/S1_Appendix.pdf
  echo "si/S1_Appendix.pdf"
}
si() {
  $PY scripts/si_tables.py --only "$@"
  $PY scripts/si_archives.py
  appendix
}
pdf() {
  $LATEXMK main.tex > build/latexmk_main.log 2>&1 || { tail -40 build/latexmk_main.log; exit 1; }
  cp build/main.pdf main.pdf
  echo "main.pdf  $(sed -n 's/^Output written on .*(\([0-9]* pages\).*/\1/p' build/main.log | tail -1)"
}
draft() {
  $LATEXMK -jobname=main_draft -pdflatex='pdflatex %O "\def\drafton{}\input{%S}"' main.tex > build/latexmk_draft.log 2>&1 \
    || { tail -40 build/latexmk_draft.log; exit 1; }
  cp build/main_draft.pdf main_draft.pdf
  echo "main_draft.pdf  $(sed -n 's/^Output written on .*(\([0-9]* pages\).*/\1/p' build/main_draft.log | tail -1)"
}
submission() {
  [[ -f build/main.bbl ]] || pdf
  $PY scripts/make_submission.py
  (cd build/submission && for i in 1 2 3; do pdflatex -interaction=nonstopmode -halt-on-error main.tex > compile.log; done \
     && rm -f main.aux main.log main.out compile.log)
  echo "build/submission/main.pdf  compiled from the single file"
}
check() {
  $PY scripts/check_manuscript.py
}

mkdir -p build
case "${1:-}" in
  figures)    figures ;;
  si)         si s1 s2 s4 ;;
  si-full)    si s1 s2 s3 s4 ;;
  pdf)        pdf ;;
  draft)      draft ;;
  submission) submission ;;
  check)      check ;;
  all)        figures; si s1 s2 s4; pdf; draft; submission; check ;;
  clean)      rm -rf build ;;
  *)          sed -n '2,17p' "$0"; exit 1 ;;
esac
