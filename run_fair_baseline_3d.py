import argparse
import warnings

import pandas as pd

from orbit_ml_cr3bp.config import MU, TRAINING_DATASET_3D_NAME, MODEL_3D_NAME
from orbit_ml_cr3bp.modeling_3d import load_model
from orbit_ml_cr3bp.fair_baseline_3d import matched_accuracy_baseline_3d


def main():
    parser = argparse.ArgumentParser(description="Matched-accuracy ML vs adaptive-solver comparison (3D CR3BP).")
    parser.add_argument("--data", type=str, default=TRAINING_DATASET_3D_NAME,
                        help="Training/test CSV. Use held-out rows only for a fair test.")
    parser.add_argument("--model_path", type=str, default=MODEL_3D_NAME)
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--output", type=str, default="3d_fair_baseline.csv")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

    data = pd.read_csv(args.data)
    if "ftle_nan" in data.columns:
        data = data[~data["ftle_nan"].astype(bool)].reset_index(drop=True)
    if "reference_divergent" in data.columns:
        data = data[~data["reference_divergent"].astype(bool)].reset_index(drop=True)

    model = load_model(args.model_path, expected_mu=MU)
    df, summary = matched_accuracy_baseline_3d(data, model, mu=MU, num_samples=args.num_samples)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()