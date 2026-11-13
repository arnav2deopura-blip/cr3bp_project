#!/bin/bash
# Part E data generation (pipeline v6 = v5 without the FTLE; part_e/PREREGISTRATION_E.md section 2). Run from the repository root with the
# project environment's python.
#   ./gen_part_e.sh select    E_select (seed 60) only: the dry run, the expectations, the E7 selection and the OOD threshold. Before the seal.
#   ./gen_part_e.sh confirm   every confirmation set (seeds 61-66, and seed 63 at 10 km and 1,000 km). ONLY AFTER THE SEAL; the files are then
#                             hashed into part_e/FROZEN_FILES.json and not opened before the confirmatory runner.
# The 100 km sets use --tol_km 100. E_tol (seed 63) is generated twice from the same seed, once per tolerance; the initial states are the same
# (tests/test_study_e.py::test_tolerance_relabelling_keeps_the_states).
set -u
PY=${PY:-/Users/arnavdeopura/miniforge3/envs/orbit-ml/bin/python}
gen() { # name n seed tol_km [extra args]
  local name=$1 n=$2 seed=$3 tol=$4; shift 4
  mkdir -p data_e/$name
  echo "$(date -u +%FT%TZ) start $name" >> data_e/gen_part_e.log
  $PY generate_3d_data.py --num_orbits $n --label_mode accuracy_monotone --pipeline v6 --tau_c 2.0 --base_seed $seed --n_workers 8 \
      --output_dir data_e/$name --tol_km $tol --chunk_size 8 "$@" > data_e/gen_$name.log 2>&1
  echo "$(date -u +%FT%TZ) end $name rc=$?" >> data_e/gen_part_e.log
}
case "${1:-}" in
  select)
    mkdir -p data_e
    gen select 10000 60 100
    ;;
  confirm)
    if [ ! -f FREEZE_E.json ]; then echo "refusing: FREEZE_E.json does not exist (seal first)"; exit 1; fi
    gen confirm 20000 61 100 --sim_time_min 1 --sim_time_max 3
    gen confirm_8 20000 62 100 --sim_time_min 1 --sim_time_max 8
    gen tol_10 10000 63 10 --sim_time_min 1 --sim_time_max 8
    gen tol_1000 10000 63 1000 --sim_time_min 1 --sim_time_max 8
    gen lock 5000 64 100 --sim_time_min 1 --sim_time_max 3
    gen confirm_20 20000 65 100
    gen lock_20 5000 66 100
    echo done > data_e/gen_part_e_confirm.done
    ;;
  *) echo "usage: $0 select|confirm"; exit 2 ;;
esac
