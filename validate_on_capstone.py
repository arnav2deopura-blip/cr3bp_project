"""
Pull a real Earth-Moon three-body trajectory (CAPSTONE, Horizons ID -1176)
from NASA JPL Horizons, convert it into this project's CR3BP synodic-frame
units, and run it through the saved integrator/timestep recommender.

Requires: pip install astroquery
Needs network access to ssd.jpl.nasa.gov.

CAPSTONE's NRHO orbital insertion was 2022-11-14; query epochs after that
to get real NRHO data rather than the trans-lunar cruise phase.

UPDATED:
  - default sampling is 30 min (was 6 h): at 6 h the closest sampled point to
    the Moon was ~8,700 km from its center, i.e. the NRHO perilune passes were
    skipped entirely.
  - --save_raw stores the raw Horizons text so the trajectory's provenance
    (reconstructed vs predicted, source file) can be read from its header;
    this script does not decide that for you and the result is NOT ground
    truth until you have checked it.
  - the CR3BP reference (DOP853) is run per row and reference_reached /
    reference_outcome are saved, so you can see which real states crash or
    escape under this model and collision convention.
  - --accuracy_compare adds the position-error-based dt per integrator
    (the same label definition as the training data); --brute_force_compare
    keeps the older Jacobi-drift search.
  - rows whose features are not finite are skipped for model prediction
    instead of crashing it.
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from orbit_ml_cr3bp.config import MU, POSITION_TOL_KM, collision_radii
from orbit_ml_cr3bp.physics_3d import (
    add_orbit_features_3d, reference_trajectory_dense_3d, REFERENCE_EVENT_NAMES,
)
from orbit_ml_cr3bp.search_3d import find_dt_for_integrator_cr3bp_3d, find_dt_for_integrator_accuracy_3d

AU_KM = 149597870.7


def query_vectors(target_id, location, epochs, save_raw=None):
    from astroquery.jplhorizons import Horizons

    obj = Horizons(id=target_id, location=location, epochs=epochs, id_type=None)
    if save_raw:
        try:
            raw = obj.vectors(refplane="earth", get_raw_response=True)
            with open(save_raw, "w") as f:
                f.write(getattr(raw, "text", str(raw)))
            print(f"Saved raw Horizons response to {save_raw}")
        except Exception as exc:
            print(f"Could not save raw Horizons response ({exc}); continuing.")
    vec = obj.vectors(refplane="earth")
    r = np.column_stack([vec["x"], vec["y"], vec["z"]]).astype(float) * AU_KM
    v = np.column_stack([vec["vx"], vec["vy"], vec["vz"]]).astype(float) * AU_KM / 86400.0
    jd = np.array(vec["datetime_jd"], dtype=float)
    return jd, r, v


def synodic_frame_from_moon_state(r_moon, v_moon):
    r_norm = np.linalg.norm(r_moon)
    x_hat = r_moon / r_norm
    h_vec = np.cross(r_moon, v_moon)
    h_norm = np.linalg.norm(h_vec)
    z_hat = h_vec / h_norm
    y_hat = np.cross(z_hat, x_hat)
    Q = np.vstack([x_hat, y_hat, z_hat])
    omega_z = h_norm / r_norm**2
    return Q, r_norm, omega_z


def inertial_to_cr3bp_synodic(r_sc, v_sc, r_moon, v_moon):
    Q, r_norm, omega_z = synodic_frame_from_moon_state(r_moon, v_moon)
    r_rot = Q @ r_sc
    v_rot = Q @ v_sc - np.cross(np.array([0.0, 0.0, omega_z]), r_rot)
    r_syn = r_rot / r_norm
    v_syn = v_rot / (r_norm * omega_z)
    return np.concatenate([r_syn, v_syn])


def main():
    from orbit_ml_cr3bp.modeling_3d import load_model, predict_hurdle

    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="-1176")
    parser.add_argument("--start", type=str, default="2023-01-01")
    parser.add_argument("--stop", type=str, default="2023-01-15")
    parser.add_argument("--step", type=str, default="30m")
    parser.add_argument("--sim_time", type=float, default=1.61,
                         help="normalized horizon; CAPSTONE's ~7-day NRHO period is ~1.61 TU")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--brute_force_compare", action="store_true",
                         help="older Jacobi-drift dt search (slow)")
    parser.add_argument("--accuracy_compare", action="store_true",
                         help="position-error dt per integrator vs a DOP853 reference (same label as training)")
    parser.add_argument("--tol_km", type=float, default=POSITION_TOL_KM)
    parser.add_argument("--save_raw", type=str, default=None,
                         help="path to store the raw Horizons text (read its header for provenance)")
    parser.add_argument("--output", type=str, default="capstone_validation.csv")
    parser.add_argument("--selfcheck", action="store_true",
                         help="run the Moon-self-consistency check instead of the spacecraft query")
    args = parser.parse_args()

    epochs = {"start": args.start, "stop": args.stop, "step": args.step}

    print("Querying the Moon relative to Earth...")
    jd_moon, r_moon, v_moon = query_vectors("301", "500@399", epochs)

    if args.selfcheck:
        print("Self-check: feeding the Moon's own barycentric state through the transform.")
        print(f"Expect synodic position ~= ({1.0 - MU:.6f}, 0, 0) with small residual velocity.")
        jd_sc, r_sc, v_sc = query_vectors("301", "500@3", epochs)
        target_label = "Moon (self-check)"
    else:
        print(f"Querying {args.target} relative to Earth-Moon barycenter...")
        jd_sc, r_sc, v_sc = query_vectors(args.target, "500@3", epochs, save_raw=args.save_raw)
        target_label = args.target

    assert len(jd_sc) == len(jd_moon), \
        "Spacecraft/Moon queries returned different epoch counts; check the date range and step."

    rows = []
    for i in range(len(jd_sc)):
        state = inertial_to_cr3bp_synodic(r_sc[i], v_sc[i], r_moon[i], v_moon[i])
        rows.append({
            "jd": jd_sc[i],
            "x": state[0], "y": state[1], "z": state[2],
            "vx": state[3], "vy": state[4], "vz": state[5],
            "sim_time": args.sim_time,
        })
    df = pd.DataFrame(rows)

    if args.selfcheck:
        print(df[["x", "y", "z", "vx", "vy", "vz"]].describe())
        return

    print(f"Computing physics features for {target_label}...")
    df = add_orbit_features_3d(df, MU, use_variational_ftle=True, ftle_T=args.sim_time)
    _r1c, r2c = collision_radii(MU)
    print(f"min r2_3d over sampled rows = {df['r2_3d'].min():.5f} DU "
          f"(Moon collision radius {r2c:.5f} DU); rows with FTLE truncated by a crash/escape/step cap: "
          f"{int((df['ftle_status'] != 0).sum())} of {len(df)}")

    if args.model_path:
        print(f"Loading model from {args.model_path}...")
        model = load_model(args.model_path)
        feat_cols = model["feature_columns"]
        good = df[feat_cols].notna().all(axis=1).to_numpy()
        print(f"{int(good.sum())} of {len(df)} rows have finite features; the rest get no prediction.")
        if good.any():
            pred = predict_hurdle(model, df[good])
            for integrator in ["euler", "rk4", "leapfrog"]:
                feas = np.full(len(df), np.nan)
                dt = np.full(len(df), np.nan)
                feas[good] = pred[integrator]["feasible"].astype(float)
                dt[good] = pred[integrator]["dt"]
                df[f"{integrator}_feasible_pred"] = feas
                df[f"{integrator}_dt_pred"] = dt

    if args.accuracy_compare:
        print("Reference trajectories and position-error dt per row (slow)...")
        reached, outcome = [], []
        dts = {k: [] for k in ("euler", "rk4", "leapfrog")}
        for _, row in df.iterrows():
            state = [row["x"], row["y"], row["z"], row["vx"], row["vy"], row["vz"]]
            ref = reference_trajectory_dense_3d(state, args.sim_time, MU)
            reached.append(bool(ref[1]))
            outcome.append(REFERENCE_EVENT_NAMES.get(int(getattr(ref[0], "reference_event", 9)), "solver_failure"))
            for integ in dts:
                dts[integ].append(find_dt_for_integrator_accuracy_3d(integ, state, args.sim_time, MU,
                                                                     tol_km=args.tol_km, reference=ref)
                                  if ref[1] else np.nan)
        df["reference_reached"] = reached
        df["reference_outcome"] = outcome
        for integ, vals in dts.items():
            df[f"{integ}_dt_accuracy"] = vals
        print(df["reference_outcome"].value_counts().to_string())

    if args.brute_force_compare:
        print("Running brute-force (Jacobi-drift) search per point (slow)...")
        for integrator in ["euler", "rk4", "leapfrog"]:
            dts = []
            for i, row in df.iterrows():
                state = [row["x"], row["y"], row["z"], row["vx"], row["vy"], row["vz"]]
                dts.append(find_dt_for_integrator_cr3bp_3d(integrator, state, args.sim_time, MU))
            df[f"{integrator}_dt_brute"] = dts

    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
