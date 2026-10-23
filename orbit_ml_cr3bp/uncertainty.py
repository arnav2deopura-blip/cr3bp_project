from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any

from .modeling_3d import INTEGRATORS

def predict_hurdle_with_uncertainty(models: dict, features_df: pd.DataFrame) -> dict:
    """
    Returns predictions with ensemble disagreement (std across trees).
    For each integrator:
        out[integrator] = {
            "feasible": bool array,
            "dt_mean": float array (NaN where infeasible),
            "dt_std": float array (NaN where infeasible)
        }
    """
    feature_columns = models["feature_columns"]
    x = features_df[feature_columns].values
    out = {}
    for integrator in INTEGRATORS:
        clf = models[integrator]["classifier"]
        reg = models[integrator]["regressor"]
        dt_mean = np.full(len(x), np.nan)
        dt_std = np.full(len(x), np.nan)
        if reg is None:
            out[integrator] = {"feasible": np.zeros(len(x), dtype=bool),
                               "dt_mean": dt_mean, "dt_std": dt_std}
            continue
        feasible = clf.predict(x).astype(bool)

        if feasible.any():
            # Get individual tree predictions for regression
            tree_preds = np.array([tree.predict(x[feasible]) for tree in reg.estimators_])
            dt_mean[feasible] = tree_preds.mean(axis=0)
            dt_std[feasible] = tree_preds.std(axis=0, ddof=1)
        out[integrator] = {
            "feasible": feasible,
            "dt_mean": dt_mean,
            "dt_std": dt_std,
        }
    return out