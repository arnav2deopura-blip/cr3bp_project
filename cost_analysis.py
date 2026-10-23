from __future__ import annotations

import argparse
import time
import numpy as np

def compute_break_even(
    n_training_orbits: int,
    training_generation_time_sec: float,   # total time to generate/label training data
    model_training_time_sec: float,        # time to train final model
    brute_force_time_per_orbit_sec: float, # average time for one brute-force search (3 integrators)
    ml_prediction_time_per_orbit_sec: float, # time for feature computation + model inference
):
    """
    Returns break-even number of future orbits.
    """
    upfront_cost = training_generation_time_sec + model_training_time_sec
    saving_per_orbit = brute_force_time_per_orbit_sec - ml_prediction_time_per_orbit_sec
    if saving_per_orbit <= 0:
        return np.inf, upfront_cost, saving_per_orbit
    n_break_even = upfront_cost / saving_per_orbit
    return n_break_even, upfront_cost, saving_per_orbit

def main():
    # Example placeholders; user should replace with actual timings.
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train", type=int, default=15000)
    parser.add_argument("--gen_time", type=float, help="Total training data generation time (sec)")
    parser.add_argument("--train_time", type=float, help="Model training time (sec)")
    parser.add_argument("--brute_per_orbit", type=float, help="Avg brute-force search time (sec/orbit)")
    parser.add_argument("--ml_per_orbit", type=float, help="Avg ML prediction time (sec/orbit)")
    args = parser.parse_args()

    if None in [args.gen_time, args.train_time, args.brute_per_orbit, args.ml_per_orbit]:
        print("Please provide all timing arguments. Example values:")
        print("  --gen_time 43200 --train_time 300 --brute_per_orbit 5.0 --ml_per_orbit 0.002")
        return

    n_be, upfront, saving = compute_break_even(
        args.n_train, args.gen_time, args.train_time,
        args.brute_per_orbit, args.ml_per_orbit
    )
    print(f"Upfront cost: {upfront:.1f} sec")
    print(f"Savings per orbit: {saving:.4f} sec")
    if np.isinf(n_be):
        print("ML never breaks even because savings <= 0.")
    else:
        print(f"Break-even after {n_be:.0f} future orbits.")

if __name__ == "__main__":
    main()