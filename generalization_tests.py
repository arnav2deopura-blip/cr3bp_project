from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from orbit_ml_cr3bp.config import PHYSICS_FEATURE_COLUMNS_3D
from orbit_ml_cr3bp.modeling_3d import (
    fit_hurdle_models,
    predict_hurdle,
    evaluate_hurdle_predictions,
    integrator_selection_stats,
)

def evaluate_transfer(data, train_mask, test_mask, feature_cols, label=""):
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        print(f"  Skipping {label}: empty train/test split (train={train_mask.sum()}, test={test_mask.sum()})")
        return None

    train_idx = np.where(train_mask)[0]
    models = fit_hurdle_models(data, feature_cols, train_idx, random_state=42, n_estimators=100)
    test_data = data[test_mask]
    pred = predict_hurdle(models, test_data)

    classifier_acc, regressor_r2, regressor_mae = evaluate_hurdle_predictions(test_data, pred)
    results = {
        integrator: {
            "classifier_acc": classifier_acc[integrator],
            "r2": regressor_r2[integrator] if regressor_r2[integrator] is not None else np.nan,
            "mae": regressor_mae[integrator] if regressor_mae[integrator] is not None else np.nan,
        }
        for integrator in ["euler", "rk4", "leapfrog"]
    }
    results["selection_acc"] = integrator_selection_stats(test_data, pred)["accuracy"]
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    args = parser.parse_args()
    data = pd.read_csv(args.data)

    # FIXED: same missing filter as run_3d_analysis_from_data.py -- rows
    # with ftle_nan=True have NaN ftle_3d (and NaN dt_* columns, since the
    # integrator search is skipped for them). Left in, they crash
    # fit_hurdle_models's RandomForest fit the moment such a row lands in
    # a train_idx split.
    if "ftle_nan" in data.columns:
        n_raw = len(data)
        data = data[~data["ftle_nan"]].reset_index(drop=True)
        n_dropped = n_raw - len(data)
        if n_dropped:
            print(f"Dropped {n_dropped} FTLE-NaN row(s) (excluded from modeling/analysis).")

    if "reference_divergent" in data.columns:
        n_raw = len(data)
        data = data[~data["reference_divergent"].fillna(False).astype(bool)].reset_index(drop=True)
        n_dropped = n_raw - len(data)
        if n_dropped:
            print(f"Dropped {n_dropped} reference-divergent row(s) (their dt labels are undefined).")

    feature_cols = [c for c in PHYSICS_FEATURE_COLUMNS_3D if c in data.columns]

    if len(data) < 200:
        print("Warning: dataset is very small; transfer metrics will be noisy and may be unreliable.")

    if "orbit_label" not in data.columns:
        print("No 'orbit_label' column found; cannot perform L1/L2 transfer.")
        return

    # L1 → L2 transfer
    l1_mask = data["orbit_label"] == "L1"
    l2_mask = data["orbit_label"] == "L2"
    print("L1 → L2 transfer:")
    res = evaluate_transfer(data, l1_mask, l2_mask, feature_cols, label="L1→L2")
    if res is not None:
        for k, v in res.items():
            if k != "selection_acc":
                print(f"  {k}: {v}")
        print(f"  selection_acc: {res['selection_acc']:.4f}")

    # Regular → Chaotic transfer
    # Use FTLE if it has variance; otherwise fall back to c_minus_c_l1_3d
    if "ftle_3d" in data.columns and data["ftle_3d"].std() > 1e-12:
        split_col = "ftle_3d"
        print("\nRegular → Chaotic transfer (by FTLE median):")
    elif "c_minus_c_l1_3d" in data.columns:
        split_col = "c_minus_c_l1_3d"
        print("\nRegular → Chaotic transfer (by C - C_L1 median; FTLE unavailable/zero):")
    else:
        split_col = None
        print("\nCannot perform regular/chaotic split: no FTLE or Jacobi-related column.")

    if split_col is not None:
        median_val = data[split_col].median()
        if split_col == "ftle_3d":
            regular_mask = data[split_col] < median_val
        else:
            # For c_minus_c_l1_3d, lower values may indicate more chaotic (higher energy)
            regular_mask = data[split_col] > median_val
        chaotic_mask = ~regular_mask

        res2 = evaluate_transfer(data, regular_mask, chaotic_mask, feature_cols, label="Regular→Chaotic")
        if res2 is not None:
            for k, v in res2.items():
                if k != "selection_acc":
                    print(f"  {k}: {v}")
            print(f"  selection_acc: {res2['selection_acc']:.4f}")

if __name__ == "__main__":
    main()