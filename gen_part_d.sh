#!/bin/bash
# Part D data generation (pipeline v5, the Part C settings; only the seeds are new). Seeds: 52 select (10,000), 53 confirm (20,000),
# 54 lock (5,000), 55 replication (20,000). Seed 51 (train extension in the plan skeleton) is not used: both the ML policy and the physics rules are
# fitted on the Part C D_train. Run from the repository root with the project environment's python.
set -u
PY=${PY:-/Users/arnavdeopura/miniforge3/envs/orbit-ml/bin/python}
gen() { # name n seed
  mkdir -p data_d/$1
  echo "$(date -u +%FT%TZ) start $1" >> data_d/gen_part_d.log
  $PY generate_3d_data.py --num_orbits $2 --label_mode accuracy_monotone --pipeline v5 --tau_c 2.0 --base_seed $3 --n_workers 8 \
      --output_dir data_d/$1 --tol_km 100 --chunk_size 8 > data_d/gen_$1.log 2>&1
  echo "$(date -u +%FT%TZ) end $1 rc=$?" >> data_d/gen_part_d.log
}
mkdir -p data_d
gen select 10000 52
gen confirm 20000 53
gen lock 5000 54
gen replication 20000 55
echo done > data_d/gen_part_d.done
