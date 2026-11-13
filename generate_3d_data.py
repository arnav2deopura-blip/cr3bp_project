"""
Generate the 3D CR3BP training dataset.

This is the "gen" half of the two-file pipeline: it samples candidate
orbits, brute-force searches for the largest stable timestep per integrator
(orbit_ml_cr3bp.search_3d), and writes the usable rows to CSV via
orbit_ml_cr3bp.data_3d.generate_3d_training_dataset. It does no model
training, evaluation, or figure generation -- run archive/study_a_tools/run_3d_analysis_from_data.py
on the resulting CSV for that.
"""
from __future__ import annotations

import argparse
import os

from orbit_ml_cr3bp.config import MU, N_WORKERS, CHUNK_SIZE, NUM_TRAINING_ORBITS
from orbit_ml_cr3bp.data_3d import generate_3d_training_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate the 3D CR3BP training dataset.")
    parser.add_argument("--num_orbits", type=int, default=NUM_TRAINING_ORBITS,
                         help=f"Number of candidate orbits to generate (default: {NUM_TRAINING_ORBITS})")
    parser.add_argument("--mu", type=float, default=MU,
                         help=f"CR3BP mass ratio (default: {MU}, Earth-Moon)")
    parser.add_argument("--n_workers", type=int, default=N_WORKERS,
                         help=f"Parallel worker processes (default: {N_WORKERS})")
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE,
                         help=f"multiprocessing.Pool chunksize (default: {CHUNK_SIZE})")
    parser.add_argument("--output_dir", type=str, default=".",
                         help="Directory to write the training CSV to (default: current directory)")
    parser.add_argument("--label_mode", type=str, default="jacobi",
                         choices=["jacobi", "accuracy", "accuracy_monotone"],
                         help="'jacobi' (default, original Jacobi-drift labels), 'accuracy' "
                              "(DOP853 position-error labels, 1-20 TU horizon, near-Moon resampling), or "
                              "'accuracy_monotone' (same, but the dt label must also pass for every tested "
                              "finer dt down to dt/4)")
    parser.add_argument("--tol_km", type=float, default=None,
                         help="Position-error tolerance in km for the accuracy label modes "
                              "(default: config.POSITION_TOL_KM)")
    parser.add_argument("--tol_du", type=float, default=None,
                         help="Position-error tolerance in this system's own length unit (DU). Use the same "
                              "--tol_du across systems for a like-for-like cross-mu comparison "
                              "(100 km Earth-Moon = 100/384400 DU). Ignored if --tol_km is given.")
    parser.add_argument("--base_seed", type=int, default=12345,
                         help="Seed offset; use different values to generate independent datasets "
                              "(output file gets a _seed<N> suffix when not 12345)")
    parser.add_argument("--pipeline", type=str, default="legacy", choices=["legacy", "v4", "v5", "v6"],
                         help="'legacy' = label_pipeline_version 3 exactly (Part A). 'v4' = Part B: physical step ceiling "
                              "(2 TU, clipped at the horizon), reference rtol 1e-13, per-orbit physics diagnostics, "
                              "success defined by the search's reason code. 'v5' = Part C: v4 plus the Sundman-time RK4 label (dt_rk4s, counted "
                              "cost), the osculating periapsis feature columns and optional --sampling_weights. 'v6' = Study E: v5 without the FTLE "
                              "integration (ftle_3d NaN; labels identical to v5).")
    parser.add_argument("--tau_c", type=float, default=None, help="v5: cap of the Sundman time-scaling function (default 1.0 TU)")
    parser.add_argument("--sampling_weights", type=float, nargs=3, default=None, metavar=("LAGRANGE", "WIDE", "NEAR_MOON"),
                         help="override the lagrange/wide/near_moon mixture (default 0.55 0.20 0.25); v5 only")
    parser.add_argument("--sim_time_min", type=float, default=None, help="Study E: horizon range lower end in TU (with --sim_time_max; log-uniform)")
    parser.add_argument("--sim_time_max", type=float, default=None, help="Study E: horizon range upper end in TU")
    args = parser.parse_args()
    if (args.sim_time_min is None) != (args.sim_time_max is None):
        parser.error("--sim_time_min and --sim_time_max go together")

    os.makedirs(args.output_dir, exist_ok=True)
    os.chdir(args.output_dir)

    print("=" * 70)
    print("3D CR3BP DATA GENERATION")
    print("=" * 70)
    print(f"Generating {args.num_orbits} candidate orbits (mu={args.mu:.6f}, "
          f"{args.n_workers} workers, chunk_size={args.chunk_size}, label_mode={args.label_mode}, "
          f"base_seed={args.base_seed})...")

    data, failure_rates, combo_report = generate_3d_training_dataset(
        num_orbits=args.num_orbits,
        n_workers=args.n_workers,
        chunk_size=args.chunk_size,
        mu=args.mu,
        label_mode=args.label_mode,
        tol_km=args.tol_km,
        base_seed=args.base_seed,
        tol_du=args.tol_du,
        pipeline=args.pipeline,
        tau_c=args.tau_c,
        sampling_weights=(None if args.sampling_weights is None else dict(zip(("lagrange", "wide", "near_moon"), args.sampling_weights))),
        sim_time_range=(None if args.sim_time_min is None else (args.sim_time_min, args.sim_time_max)),
    )

    print(f"\nUsable orbits: {combo_report['usable_count']} / {combo_report['num_candidates']}")
    print(f"All-failed (unusable) orbits: {combo_report['all_failed_count']}")
    print(f"Flagged as FTLE-NaN (kept in CSV with ftle_nan=True; excluded from the "
        f"counts above and from failure-rate denominators; should now be rare because the FTLE is "
        f"truncated at a crash instead of returning NaN): {combo_report['ftle_nan_count']}")
    print(f"Reference-divergent (accuracy modes only; DOP853 reference itself collided/ejected or the "
          f"state started inside a body; KEPT in the CSV with a reference_outcome, excluded from "
          f"failure-rate denominators): {combo_report['reference_divergent_count']}")
    print("\nPer-integrator failure rates:")
    for name, rate in failure_rates.items():
        print(f"  {name}: {rate:.3%}")

    if len(data) < 100:
        print("\nWARNING: fewer than 100 usable orbits generated. Downstream "
              "training/eval will be unreliable -- consider increasing --num_orbits.")

    print("\nDone. Study A analysed the CSV with archive/study_a_tools/run_3d_analysis_from_data.py; Studies B-D use run_*_3d.py.")


if __name__ == "__main__":
    main()