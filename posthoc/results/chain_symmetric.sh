#!/bin/zsh
cd "/Users/arnavdeopura/School/Everything Portfolio/Research/extended-orbit-ML-2"
PY=/Users/arnavdeopura/miniforge3/envs/orbit-ml/bin/python
$PY run_posthoc.py symmetric --set d_replication > posthoc/results/symmetric_d_replication.log 2>&1 && \
$PY run_posthoc.py symmetric --set c_confirm > posthoc/results/symmetric_c_confirm.log 2>&1
echo "exit $?" > posthoc/results/chain_symmetric.status
