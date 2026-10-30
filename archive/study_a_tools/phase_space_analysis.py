from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from orbit_ml_cr3bp.config import MU
from orbit_ml_cr3bp.search_3d import find_dt_for_integrator_cr3bp_3d
from orbit_ml_cr3bp.physics_3d import (
    add_orbit_features_3d,
    compute_ftle_3d_variational,
    compute_ftle_3d_divergence,
    step_rk4_3d,
)
from orbit_ml_cr3bp.modeling_3d import predict_hurdle


def compute_grid(
    model=None,
    x_range=(-0.2, 1.2),
    y_range=(-0.6, 0.6),
    z_fixed=0.0,
    vx_fixed=0.0,
    vy_fixed=0.0,
    vz_fixed=0.0,
    nx=20,
    ny=20,
    sim_time=20.0,
    use_model=False,
    ftle_method="variational",
):
    rows = []
    xs = np.linspace(x_range[0], x_range[1], nx)
    ys = np.linspace(y_range[0], y_range[1], ny)
    for x in xs:
        for y in ys:
            state = [x, y, z_fixed, vx_fixed, vy_fixed, vz_fixed]
            dt_true = find_dt_for_integrator_cr3bp_3d("rk4", state, sim_time, MU)

            df_temp = pd.DataFrame([{"x": x, "y": y, "z": z_fixed,
                                     "vx": vx_fixed, "vy": vy_fixed, "vz": vz_fixed}])
            df_temp = add_orbit_features_3d(df_temp, MU, use_variational_ftle=False)

            if ftle_method == "variational":
                ftle = compute_ftle_3d_variational(state, MU, T=sim_time, dt=0.01)
            else:
                ftle = compute_ftle_3d_divergence(state, MU, T=sim_time, step_func=step_rk4_3d, dt=0.01)
            df_temp["ftle_3d"] = ftle
            df_temp["sim_time"] = sim_time

            dt_pred = np.nan
            if use_model and model is not None:
                pred = predict_hurdle(model, df_temp)
                if pred["rk4"]["feasible"][0]:
                    dt_pred = pred["rk4"]["dt"][0]

            rows.append({
                "x": x, "y": y, "dt_true": dt_true, "ftle": ftle,
                "jacobi": df_temp["jacobi_3d"].iloc[0], "dt_pred": dt_pred,
            })
    return pd.DataFrame(rows)


def plot_maps(df, output_prefix="phase_space_3d"):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    sc = ax.scatter(df["x"], df["y"], c=df["ftle"], cmap="plasma")
    plt.colorbar(sc, ax=ax, label="FTLE")
    ax.set_title("Finite-Time Lyapunov Exponent")

    ax = axes[0, 1]
    sc = ax.scatter(df["x"], df["y"], c=df["dt_true"], cmap="viridis")
    plt.colorbar(sc, ax=ax, label="Δt_max")
    ax.set_title("True Stable Timestep (RK4)")

    ax = axes[1, 0]
    jacobi_vals = df["jacobi"].values
    vmin, vmax = np.nanpercentile(jacobi_vals, [1, 99])
    sc = ax.scatter(df["x"], df["y"], c=jacobi_vals, cmap="coolwarm", vmin=vmin, vmax=vmax)
    plt.colorbar(sc, ax=ax, label="Jacobi Constant (clipped 1st–99th pct)")
    ax.set_title("Jacobi Constant")

    ax = axes[1, 1]
    if df["dt_pred"].notna().any():
        sc = ax.scatter(df["x"], df["y"], c=df["dt_pred"], cmap="viridis")
        plt.colorbar(sc, ax=ax, label="Predicted Δt")
        ax.set_title("Predicted Stable Timestep (ML)")
    else:
        ax.text(0.5, 0.5, "No model provided", ha="center", va="center")
        ax.set_title("ML Prediction")

    for a in axes.flat:
        a.set_xlabel("x")
        a.set_ylabel("y")
    plt.tight_layout()
    plt.savefig(f"{output_prefix}.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--nx", type=int, default=20)
    parser.add_argument("--ny", type=int, default=20)
    parser.add_argument("--sim_time", type=float, default=20.0)
    parser.add_argument("--use_model", action="store_true")
    parser.add_argument("--output_prefix", type=str, default="phase_space_3d")
    args = parser.parse_args()

    model = None
    if args.use_model and args.model_path:
        from orbit_ml_cr3bp.modeling_3d import load_model
        model = load_model(args.model_path)

    df = compute_grid(model=model, nx=args.nx, ny=args.ny, sim_time=args.sim_time, use_model=args.use_model)
    df.to_csv(f"{args.output_prefix}_data.csv", index=False)
    plot_maps(df, output_prefix=args.output_prefix)