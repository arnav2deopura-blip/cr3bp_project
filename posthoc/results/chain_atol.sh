#!/bin/zsh
cd "/Users/arnavdeopura/School/Everything Portfolio/Research/extended-orbit-ML-2"
until [[ -f posthoc/results/chain_symmetric.status ]]; do sleep 30; done
/Users/arnavdeopura/miniforge3/envs/orbit-ml/bin/python run_posthoc.py atol > posthoc/results/atol.log 2>&1
echo "exit $?" > posthoc/results/chain_atol.status
