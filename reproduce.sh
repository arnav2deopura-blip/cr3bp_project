#!/bin/zsh
# Rebuild figures and tables from the deposited data and run outputs (data_b/ data_c/ data_d/ out_c/ out_d/ and the tracked part_*/results).
# Nothing here reruns a confirmatory test or touches a sealed file: figure scripts write to reproduce_out/<study>/ (gitignored), the manuscript
# figure scripts rewrite manuscript/figures/ (deterministic), and the post hoc commands write new run folders under posthoc/results/.
#
#   ./reproduce.sh env          create the pinned conda environment (environment.lock.txt: exact builds, osx-arm64)
#   ./reproduce.sh tests        every test suite
#   ./reproduce.sh b|c|d        the figures of Study B, C or D
#   ./reproduce.sh manuscript   Fig 1-10 (TIFF + draft PNG + data CSVs) and main.pdf
#   ./reproduce.sh posthoc      the post hoc analyses of posthoc/README.md (P1-P8; P7 and P6 take about an hour each on 10 cores)
#   ./reproduce.sh check        the seal checks of Studies B, C and D (they list every file changed since each seal)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-/Users/arnavdeopura/miniforge3/envs/orbit-ml/bin/python}
OUT=reproduce_out
one() { local m=( ${~1} ); [[ ${#m} -eq 1 ]] || { echo "expected one match for $1, found ${#m}" >&2; exit 1; }; echo $m[1]; }

case "${1:-}" in
  env)
    conda create -y -p "${2:-./.env-orbit-ml}" --file environment.lock.txt ;;
  tests)
    for t in tests/test_phase0.py tests/test_failure_study.py tests/test_part_c.py tests/test_part_d.py tests/test_posthoc.py tests/test_hardening.py; do
      echo "== $t"; $PY $t
    done ;;
  b)
    mkdir -p $OUT/part_b
    $PY make_figures_b.py all --decompose $(one "out_b/confirm/decompose_*") --ledger $(one "out_b/confirm/policy_*")/ledger.csv --label D_confirm --out $OUT/part_b ;;
  c)
    mkdir -p $OUT/part_c
    $PY make_figures_c.py all --sund $(one "out_c/confirm/sund/apply_policy_*") --phys $(one "out_c/confirm/phys/apply_policy_*") \
        --global $(one "out_c/confirm/global/apply_policy_*") --out $OUT/part_c --label D_confirm_C ;;
  d)
    mkdir -p $OUT/part_d
    $PY make_figures_d.py all --root out_d/confirm --explore_root out_d/explore --out $OUT/part_d --label D_confirm_D ;;
  manuscript)
    (cd manuscript/scripts && for f in fig01 fig02 fig03 fig04 fig05 fig06 fig07 fig08 fig09 fig10; do $PY $f.py; done)
    (cd manuscript && latexmk -pdf -interaction=nonstopmode main.tex) ;;
  posthoc)
    $PY run_posthoc.py aggregates
    $PY run_posthoc.py fig4_check
    $PY run_posthoc.py three_quarters
    $PY run_posthoc.py symmetric --set d_replication
    $PY run_posthoc.py symmetric --set c_confirm
    $PY run_posthoc.py atol
    $PY run_posthoc.py seeds ;;
  check)
    $PY freeze_b.py check || true
    $PY freeze_c.py check || true
    $PY freeze_d.py check || true ;;
  *)
    sed -n '2,13p' "$0"; exit 1 ;;
esac
