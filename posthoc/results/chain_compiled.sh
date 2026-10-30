#!/bin/zsh
cd "/Users/arnavdeopura/School/Everything Portfolio/Research/extended-orbit-ML-2"
PY=/Users/arnavdeopura/miniforge3/envs/orbit-ml/bin/python
until [[ -f posthoc/results/chain_atol.status ]]; do sleep 30; done
$PY run_posthoc.py d4b_fine > posthoc/results/d4b_fine.log 2>&1
$PY run_posthoc.py compiled > posthoc/results/compiled.log 2>&1
echo "exit $?" > posthoc/results/chain_compiled.status
