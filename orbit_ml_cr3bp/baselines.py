from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from .config import MU

def _accel_magnitude(df: pd.DataFrame, mu: float = MU) -> np.ndarray:
    x = df["x"].to_numpy(float); y = df["y"].to_numpy(float); z = df["z"].to_numpy(float)
    vx = df["vx"].to_numpy(float); vy = df["vy"].to_numpy(float)
    r1 = np.maximum(np.sqrt((x + mu)**2 + y**2 + z**2), 1e-12)
    r2 = np.maximum(np.sqrt((x + mu - 1.0)**2 + y**2 + z**2), 1e-12)
    ax = 2.0*vy + x - (1.0 - mu)*(x + mu)/r1**3 - mu*(x + mu - 1.0)/r2**3
    ay = -2.0*vx + y - (1.0 - mu)*y/r1**3 - mu*y/r2**3
    az = -(1.0 - mu)*z/r1**3 - mu*z/r2**3
    return np.sqrt(ax**2 + ay**2 + az**2)


def _fit_linear(X: np.ndarray, y: np.ndarray) -> float:
    """Fit a single proportionality constant alpha such that y ≈ alpha * X."""
    alpha = np.dot(X, y) / max(np.dot(X, X), 1e-12)
    return alpha

def evaluate_physics_baselines(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    integrator: str = "rk4",
    sim_time_col: str = "sim_time",
) -> dict:
    """
    Evaluate four physics-based baselines for predicting the stable timestep
    of a given integrator (only on successful rows).

    Each baseline corresponds to a physical hypothesis about what controls
    numerical integration difficulty:

      1. Keplerian timescale:      dt ∝ sqrt(r^3)          (orbital period)
      2. Dynamical timescale:      dt ∝ sqrt(r / a)        (local acceleration)
      3. Crossing timescale:       dt ∝ r / v              (local crossing time)
      4. Instability timescale:    dt ∝ 1 / FTLE           (local chaos)

    The proportionality constant is fitted on the training set (successful
    rows only), then frozen before evaluating on the test set.

    Returns a dict with MAE/R² for each baseline.
    """
    dt_col = f"dt_{integrator}"
    success_train = train_data[f"{integrator}_success"] == 1
    success_test = test_data[f"{integrator}_success"] == 1

    if success_train.sum() == 0 or success_test.sum() == 0:
        return {"error": "No successful samples for training/testing"}

    # Training features
    X_train = train_data[success_train].copy()
    y_train = X_train[dt_col].values

    # Test features
    X_test = test_data[success_test].copy()
    y_test = X_test[dt_col].values

    # Feature columns needed
    r = X_train["r2_3d"].values
    v = np.sqrt(X_train["vx"].values**2 + X_train["vy"].values**2 + X_train["vz"].values**2)
    a_mag = _accel_magnitude(X_train)
    r_test = X_test["r2_3d"].values
    v_test = np.sqrt(X_test["vx"].values**2 + X_test["vy"].values**2 + X_test["vz"].values**2)
    a_mag_test = _accel_magnitude(X_test)

    # Baseline 1: Keplerian timescale
    X1_train = np.sqrt(r**3)
    alpha1 = _fit_linear(X1_train, y_train)
    y1_pred = alpha1 * np.sqrt(r_test**3)

    # Baseline 2: Dynamical timescale sqrt(r/a)
    X2_train = np.sqrt(r / np.maximum(a_mag, 1e-12))
    alpha2 = _fit_linear(X2_train, y_train)
    y2_pred = alpha2 * np.sqrt(r_test / np.maximum(a_mag_test, 1e-12))

    # Baseline 3: Crossing timescale r/v
    X3_train = r / np.maximum(v, 1e-12)
    alpha3 = _fit_linear(X3_train, y_train)
    y3_pred = alpha3 * (r_test / np.maximum(v_test, 1e-12))

    results = {
        "keplerian": {"mae": mean_absolute_error(y_test, y1_pred), "r2": r2_score(y_test, y1_pred)},
        "dynamical": {"mae": mean_absolute_error(y_test, y2_pred), "r2": r2_score(y_test, y2_pred)},
        "crossing": {"mae": mean_absolute_error(y_test, y3_pred), "r2": r2_score(y_test, y3_pred)},
    }

    # Baseline 4: Instability timescale (1/FTLE) if available
    if "ftle_3d" in X_train.columns and X_train["ftle_3d"].sum() > 0:
        ftle_train = X_train["ftle_3d"].values
        ftle_test = X_test["ftle_3d"].values
        mask_train = ftle_train > 1e-6
        mask_test = ftle_test > 1e-6
        if mask_train.sum() > 0 and mask_test.sum() > 0:
            X4_train = 1.0 / ftle_train[mask_train]
            y4_train = y_train[mask_train]
            alpha4 = _fit_linear(X4_train, y4_train)
            X4_test = 1.0 / ftle_test[mask_test]
            y4_test = y_test[mask_test]
            results["instability"] = {
                "mae": mean_absolute_error(y4_test, alpha4 * X4_test),
                "r2": r2_score(y4_test, alpha4 * X4_test),
            }
        else:
            results["instability"] = {"mae": float('nan'), "r2": float('nan')}
    else:
        results["instability"] = {"mae": float('nan'), "r2": float('nan')}

    return results