#!/bin/zsh
# Defense-plan exploratory comparator runs (posthoc/README.md P20-P23), one after another, 8 workers each.
cd "/Users/arnavdeopura/School/Everything Portfolio/Research/extended-orbit-ML-2"
PY=/Users/arnavdeopura/miniforge3/envs/orbit-ml/bin/python
for c in sundman_adaptive learned_rtol multistep rk8s; do
  echo "$c start $(date -u +%FT%TZ)" >> posthoc/results/chain_defense.status
  $PY -W ignore run_posthoc.py $c > posthoc/results/$c.log 2>&1
  echo "$c exit $? $(date -u +%FT%TZ)" >> posthoc/results/chain_defense.status
done
