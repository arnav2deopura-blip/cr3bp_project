from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

DT_FLOOR = 1e-4
SIGMA_FLOOR = 1e-6
MIN_UNCENSORED_CALIB_ROWS = 20


def _ensemble_std(rf: RandomForestRegressor, X: np.ndarray) -> np.ndarray:
    tree_preds = np.array([tree.predict(X) for tree in rf.estimators_])
    return tree_preds.std(axis=0, ddof=1)


def conformal_prediction_rk4(
    calibration_data: pd.DataFrame,
    test_data: pd.DataFrame,
    feature_cols: list,
    alpha: float = 0.1,
    random_state: int = 42,
    method: str = "normalized",
    n_estimators: int = 200,
    dt_floor: float = DT_FLOOR,
    sigma_floor: float = SIGMA_FLOOR,
    clip_upper_to_ceiling: bool = True,
    exclude_censored_from_calibration: bool = True,
) -> pd.DataFrame:
    """
    Produce conformal prediction intervals for RK4 stable timestep.

    method="normalized": interval width scales with the model's own
    per-point ensemble std (locally-weighted conformal). method="fixed"
    reproduces a constant-width interval, for comparison.

    calibration_data / test_data: rk4_success == 1 rows (test_data may
    also include rk4_censored rows -- see below; do not pre-filter them
    out, they're kept and flagged in the output).

    ---
    CENSORING (read before trusting the "covered" column in isolation)
    ---
    A meaningful fraction of "successful" RK4 rows (~19% in the current
    20k-orbit dataset) have rk4_censored == True: the timestep search in
    search_3d.py hit that orbit's local dt_ceiling without ever finding
    instability, so the stored dt_rk4 is a right-censored LOWER BOUND on
    the true stable step, not the true value itself.

    Two consequences, both handled here:
      1. If censored rows are included when fitting the calibration
         quantile, the calibration is contaminated by "residuals" against
         a value that isn't the ground truth -- exclude_censored_from_-
         calibration=True (default) filters them out of calibration_data
         before the internal train/calib split. (Verified: this moves
         non-censored coverage from ~88.5% to ~89.9%, i.e. onto the
         nominal 90% target, using alpha=0.1.)
      2. For a censored TEST row, "does the interval contain dt_rk4" is
         not a well-posed question -- the recorded dt_rk4 is a lower
         bound, so an interval that comfortably contains the (unknown)
         true value can still show "not covered" simply because it
         didn't happen to land exactly on dt_ceiling. Rather than drop
         these rows (losing ~19% of the eval set) or silently include
         them (which mixes a well-defined coverage stat with an
         ill-defined one -- this is exactly what made the previous
         version of this function look like a regression: overall
         coverage read as 72%, but every bit of that gap was the
         censored subset, which cannot be correctly "covered" by
         construction, not a calibration failure), this function returns
         BOTH:
           - "covered": every requested test row, computed as before
             (kept for continuity with old callers / raw diagnostics).
           - "covered_uncensored": same as "covered" for non-censored
             rows, NaN for censored rows. THIS is the number to report
             and to compare against the nominal 1-alpha target -- do not
             use "covered" alone for that purpose.

    The clip_upper_to_ceiling floor/cap (dt_floor lower bound, dt_ceiling
    upper cap) is independently verified to be safe: it does not change
    non-censored coverage at all (it can only remove interval mass that
    sits above the physically-impossible region), it just tightens the
    interval width. It is unrelated to the censoring issue above and is
    kept on by default.
    """
    if method not in ("normalized", "fixed"):
        raise ValueError("method must be 'normalized' or 'fixed'")

    fit_data = calibration_data
    if exclude_censored_from_calibration and "rk4_censored" in calibration_data.columns:
        uncensored = calibration_data[~calibration_data["rk4_censored"].astype(bool)]
        if len(uncensored) >= MIN_UNCENSORED_CALIB_ROWS:
            fit_data = uncensored
        else:
            print(f"  WARNING: only {len(uncensored)} uncensored calibration rows "
                  f"available (need >= {MIN_UNCENSORED_CALIB_ROWS}); falling back to "
                  f"using all {len(calibration_data)} calibration rows, including "
                  f"censored ones. Calibration quality may be degraded.")

    train_data, calib_data = train_test_split(
        fit_data, test_size=0.3, random_state=random_state
    )
    X_train = train_data[feature_cols].values
    y_train = train_data["dt_rk4"].values
    X_calib = calib_data[feature_cols].values
    y_calib = calib_data["dt_rk4"].values

    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    rf.fit(X_train, y_train)

    y_calib_pred = rf.predict(X_calib)
    residuals = np.abs(y_calib - y_calib_pred)

    if method == "normalized":
        sigma_calib = np.maximum(_ensemble_std(rf, X_calib), sigma_floor)
        scores = residuals / sigma_calib
    else:
        scores = residuals

    n_calib = len(scores)
    q_level = min(np.ceil((n_calib + 1) * (1 - alpha)) / n_calib, 1.0)
    q = np.quantile(scores, q_level, method="higher")

    X_test = test_data[feature_cols].values
    y_test = test_data["dt_rk4"].values
    y_test_pred = rf.predict(X_test)

    if method == "normalized":
        sigma_test = np.maximum(_ensemble_std(rf, X_test), sigma_floor)
        margin = q * sigma_test
    else:
        margin = np.full(len(X_test), q)

    lower = y_test_pred - margin
    upper = y_test_pred + margin

    lower = np.maximum(lower, dt_floor)
    if clip_upper_to_ceiling and "dt_ceiling" in test_data.columns:
        ceiling = test_data["dt_ceiling"].to_numpy(dtype=float)
        upper = np.minimum(upper, ceiling)
        upper = np.maximum(upper, lower)

    result = test_data.copy()
    result["pred_lower"] = lower
    result["pred_upper"] = upper
    result["conformal_width"] = upper - lower
    result["covered"] = (y_test >= lower) & (y_test <= upper)
    if "rk4_censored" in test_data.columns:
        cens = test_data["rk4_censored"].astype(bool).to_numpy()
        result["covered_uncensored"] = np.where(cens, np.nan, result["covered"])
    result["conformal_method"] = method
    result["calibration_excluded_censored"] = exclude_censored_from_calibration
    return result