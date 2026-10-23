from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

from .config import (
    MU,
    BASIC_FEATURE_COLUMNS_3D,
    PHYSICS_FEATURE_COLUMNS_3D,
    TARGET_COLUMNS_3D,
    MODEL_3D_NAME,
    INTEGRATOR_TOLERANCES_3D as TOLERANCES_3D,
    SAFETY_FACTOR_CANDIDATES_3D,
)
from .physics_3d import (
    EVALS_PER_STEP_3D,
    INTEGRATORS_3D,
    add_orbit_features_3d,
    cr3bp_derivatives_3d,
    jacobi_constant_3d,
    max_jacobi_drift_3d,
    step_rk4_3d,
)
from .search_3d import find_dt_for_integrator_cr3bp_3d

INTEGRATORS = ["euler", "rk4", "leapfrog"]


# =============================================================================
# HURDLE MODELS
# =============================================================================

def _rf_classifier(n_estimators: int = 100, random_state: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)


def _rf_regressor(n_estimators: int = 100, random_state: int = 42) -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)


def _hgb_quantile_regressor(quantile: float, random_state: int = 42):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        loss="quantile", quantile=quantile, max_iter=300, learning_rate=0.05, random_state=random_state)


def fit_hurdle_models(
    data: pd.DataFrame,
    feature_columns: list[str],
    train_idx: np.ndarray,
    random_state: int = 42,
    n_estimators: int = 100,
    regressor_kind: str = "rf",
    quantile: float | None = None,
) -> dict:
    """
    regressor_kind="rf" (default, unchanged): random-forest regressor on dt.
    regressor_kind="hgb_quantile": gradient-boosted regressor with quantile
    loss on log10(dt); `quantile` (default 0.1) is the target percentile of
    the safe step, so predictions deliberately aim LOW (conservative) instead
    of at the mean -- an alternative to predicting the mean and then shrinking
    it by a coarse global safety scale. Models trained this way do not have
    .estimators_, so uncertainty.py / analyze_single_orbit_3d (tree-spread
    uncertainty) only support "rf".
    """
    if regressor_kind not in ("rf", "hgb_quantile"):
        raise ValueError("regressor_kind must be 'rf' or 'hgb_quantile'")
    if regressor_kind == "hgb_quantile" and quantile is None:
        quantile = 0.1
    log_target = regressor_kind == "hgb_quantile"
    models = {}
    x_all = data[feature_columns].values

    for integrator in INTEGRATORS:
        success_col = f"{integrator}_success"
        dt_col = f"dt_{integrator}"

        y_success_train = data[success_col].values[train_idx]
        clf = _rf_classifier(n_estimators=n_estimators, random_state=random_state)
        clf.fit(x_all[train_idx], y_success_train)

        succ_mask = data[success_col].values[train_idx] == 1
        reg_train_idx = train_idx[succ_mask]

        if succ_mask.sum() < 2:
            models[integrator] = {"classifier": clf, "regressor": None, "log_target": log_target,
                                  "n_success_train": int(succ_mask.sum()), 
                                  "n_train": int(len(train_idx))}
            continue
        
        y_dt = data[dt_col].values[reg_train_idx]
        if log_target:
            reg = _hgb_quantile_regressor(quantile, random_state=random_state)
            reg.fit(x_all[reg_train_idx], np.log10(y_dt))
        else:
            reg = _rf_regressor(n_estimators=n_estimators, random_state=random_state)
            reg.fit(x_all[reg_train_idx], y_dt)

        models[integrator] = {
            "classifier": clf,
            "regressor": reg,
            "log_target": log_target,
            "n_success_train": int(succ_mask.sum()),
            "n_train": int(len(train_idx)),
        }

    models["feature_columns"] = list(feature_columns)
    models["regressor_kind"] = regressor_kind
    models["quantile"] = quantile
    return models


def predict_hurdle(models: dict, features_df: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    feature_columns = models["feature_columns"]
    x = features_df[feature_columns].values
    out = {}
    for integrator in INTEGRATORS:
        clf = models[integrator]["classifier"]
        reg = models[integrator]["regressor"]
        if reg is None:
            out[integrator] = {"feasible": np.zeros(len(x), dtype=bool),
                               "dt": np.full(len(x), np.nan)}
            continue
        feasible = clf.predict(x).astype(bool)
        dt_pred = np.full(len(x), np.nan)
        if feasible.any():
            raw = reg.predict(x[feasible])
            dt_pred[feasible] = np.power(10.0, raw) if models[integrator].get("log_target", False) else raw
        out[integrator] = {"feasible": feasible, "dt": dt_pred}
    return out


def cost_normalized_choice(
    dt_by_integrator: dict[str, float], exclude: list[str] | None = None
) -> tuple[str | None, dict[str, float]]:
    exclude_set = set(exclude) if exclude else set()
    scores = {}
    for integrator, dt in dt_by_integrator.items():
        if integrator in exclude_set:
            continue
        if dt is None or (isinstance(dt, float) and np.isnan(dt)):
            continue
        scores[integrator] = dt / EVALS_PER_STEP_3D[integrator]
    if not scores:
        return None, scores
    best = max(scores, key=scores.get)
    return best, scores


def true_best_integrator_row(row: pd.Series, exclude: list[str] | None = None) -> str | None:
    dts = {}
    for integrator in INTEGRATORS:
        if row[f"{integrator}_success"] == 1:
            dts[integrator] = row[f"dt_{integrator}"]
    choice, _ = cost_normalized_choice(dts, exclude=exclude)
    return choice


# =============================================================================
# GROUPED SPATIAL SPLIT (3D)
# =============================================================================

def group_orbits_by_region_3d(data: pd.DataFrame, n_bins: int = 5) -> np.ndarray:
    cols = ['r2_3d', 'ftle_3d', 'jacobi_3d']
    if not all(c in data.columns for c in cols):
        cols = ['x', 'y', 'z']
    coords = data[cols].values
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    binned = discretizer.fit_transform(coords).astype(int)
    group_ids = (
        binned[:, 0] * (n_bins**2) +
        binned[:, 1] * n_bins +
        binned[:, 2]
    ).astype(int)
    return group_ids


def create_grouped_split_3d(
    data: pd.DataFrame,
    test_size: float = 0.3,
    random_state: int = 42,
    n_bins: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    group_ids = group_orbits_by_region_3d(data, n_bins=n_bins)
    unique_groups = np.unique(group_ids)
    group_counts = {g: int(np.sum(group_ids == g)) for g in unique_groups}

    rng = np.random.RandomState(random_state)
    shuffled_groups = rng.permutation(unique_groups)

    train_groups = set()
    train_samples = 0
    target_train = int((1 - test_size) * len(data))

    for g in shuffled_groups:
        if train_samples < target_train:
            train_groups.add(g)
            train_samples += group_counts[g]
        else:
            break

    test_groups = set(shuffled_groups) - train_groups
    train_mask = np.array([g in train_groups for g in group_ids])
    test_mask = np.array([g in test_groups for g in group_ids])

    train_frac = train_mask.sum() / len(data)
    test_frac = test_mask.sum() / len(data)

    if train_frac < 0.2 or test_frac < 0.2 or len(unique_groups) < 10:
        print("Grouped split produced imbalanced sets. Falling back to random split.")
        train_idx, test_idx = train_test_split(
            np.arange(len(data)), test_size=test_size, random_state=random_state
        )
        return train_idx, test_idx

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]
    return train_idx, test_idx


# =============================================================================
# HELD-OUT EVALUATION (shared by run_3d_analysis_from_data.py and
# generalization_tests.py -- previously each reimplemented this loop)
# =============================================================================

def evaluate_hurdle_predictions(
    test_data: pd.DataFrame, pred: dict, integrators: list[str] = INTEGRATORS,
) -> tuple[dict, dict, dict]:
    """
    Per-integrator classifier accuracy plus regressor R2/MAE (on successful,
    finite predictions only). Returns (classifier_acc, regressor_r2,
    regressor_mae), each a dict keyed by integrator name; regressor_r2 and
    regressor_mae are None for an integrator when there are fewer than two
    successful, finite predictions to score.
    """
    classifier_acc, regressor_r2, regressor_mae = {}, {}, {}
    for integrator in integrators:
        y_true_succ = test_data[f"{integrator}_success"].values
        y_pred_succ = pred[integrator]["feasible"].astype(int)
        classifier_acc[integrator] = accuracy_score(y_true_succ, y_pred_succ)

        regressor_r2[integrator] = None
        regressor_mae[integrator] = None
        succ_mask = y_true_succ == 1
        if succ_mask.sum() > 0:
            y_true_dt = test_data[f"dt_{integrator}"].values[succ_mask]
            y_pred_dt = pred[integrator]["dt"][succ_mask]
            finite_mask = np.isfinite(y_pred_dt)
            if finite_mask.sum() > 1:
                regressor_r2[integrator] = r2_score(y_true_dt[finite_mask], y_pred_dt[finite_mask])
                regressor_mae[integrator] = mean_absolute_error(y_true_dt[finite_mask], y_pred_dt[finite_mask])
    return classifier_acc, regressor_r2, regressor_mae


# =============================================================================
# INTEGRATOR SELECTION STATISTICS
# =============================================================================

def integrator_selection_stats(
    data_test: pd.DataFrame, pred_phys: dict, exclude: list[str] | None = None
) -> dict:
    n = len(data_test)
    true_best = data_test.apply(lambda row: true_best_integrator_row(row, exclude=exclude), axis=1).values

    model_choices = []
    for i in range(n):
        dts = {integrator: pred_phys[integrator]["dt"][i] for integrator in INTEGRATORS}
        choice, _ = cost_normalized_choice(dts, exclude=exclude)
        model_choices.append(choice)
    model_choices = np.array(model_choices, dtype=object)

    always_rk4 = np.array(["rk4"] * n, dtype=object)

    accuracy = float(np.mean(model_choices == true_best))
    rk4_accuracy = float(np.mean(always_rk4 == true_best))

    classifier_accuracies = {}
    for integrator in INTEGRATORS:
        y_true = data_test[f"{integrator}_success"].values
        y_pred = pred_phys[integrator]["feasible"].astype(int)
        classifier_accuracies[integrator] = accuracy_score(y_true, y_pred)

    return {
        "n_test": n,
        "excluded": sorted(exclude) if exclude else [],
        "accuracy": accuracy,
        "rk4_accuracy": rk4_accuracy,
        "improvement": accuracy - rk4_accuracy,
        "true_none_feasible_fraction": float(np.mean(true_best == None)),  # noqa: E711
        "true_euler_fraction": float(np.mean(true_best == "euler")),
        "true_rk4_fraction": float(np.mean(true_best == "rk4")),
        "true_leapfrog_fraction": float(np.mean(true_best == "leapfrog")),
        "classifier_accuracies": classifier_accuracies,
    }


# =============================================================================
# TRAINING FINAL MODEL / SAVE
# =============================================================================

def train_final_model_3d(data: pd.DataFrame, random_state: int = 42):
    phys_cols_3d = [c for c in PHYSICS_FEATURE_COLUMNS_3D if c in data.columns]
    all_idx = np.arange(len(data))
    return fit_hurdle_models(data, phys_cols_3d, all_idx, random_state=random_state, n_estimators=200)


def save_model(model, destination: str = MODEL_3D_NAME, mu: float = MU) -> None:
    joblib.dump({"model": model, "mu": mu}, destination)


def load_model(source: str = MODEL_3D_NAME, expected_mu: float = MU, tol: float = 1e-9):
    payload = joblib.load(source)
    if isinstance(payload, dict) and "mu" in payload:
        saved_mu = payload["mu"]
        if abs(saved_mu - expected_mu) > tol:
            raise ValueError(f"Model trained with mu={saved_mu}, expected {expected_mu}.")
        return payload["model"]
    return payload


# =============================================================================
# FAILURE PREDICTION CLASSIFIER
# =============================================================================

def run_failure_classifier(
    data: pd.DataFrame,
    phys_cols: list,
    target_col: str = 'rk4_success',
    test_size: float = 0.3,
    random_state: int = 42,
) -> dict:
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn.model_selection import train_test_split

    X = data[phys_cols].values
    y = data[target_col].values

    unique, counts = np.unique(y, return_counts=True)
    print(f"\n  Class distribution for '{target_col}':")
    for cls, cnt in zip(unique, counts):
        print(f"    {cls}: {cnt} ({100*cnt/len(y):.1f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1,
        class_weight='balanced',
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"\n  Accuracy: {accuracy:.4f}")
    print("  Classification Report:")
    print(classification_report(y_test, y_pred))
    print("  Confusion Matrix:")
    print(conf_matrix)

    importances = clf.feature_importances_
    feat_imp = pd.DataFrame({'feature': phys_cols, 'importance': importances})
    feat_imp = feat_imp.sort_values('importance', ascending=False)

    return {
        'model': clf,
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': conf_matrix,
        'feature_importance': feat_imp,
        'y_test': y_test,
        'y_pred': y_pred,
    }


# =============================================================================
# VIOLATION AUDIT SUMMARY / ABLATION PRINTING
# =============================================================================

def summarize_violation_audit(audit_df: pd.DataFrame) -> dict:
    summary = {}
    for integrator in INTEGRATORS:
        subset = audit_df[audit_df["integrator"] == integrator]
        if len(subset) == 0:
            summary[integrator] = None
            continue
        summary[integrator] = {
            "n_evaluated": len(subset),
            "violation_rate": float((~subset["stable"]).mean()),
            "worst_jacobi_drift": float(subset["jacobi_drift"].max()),
            "median_jacobi_drift": float(subset["jacobi_drift"].median()),
        }
    return summary


def print_feature_ablation(report: dict) -> None:
    print("\n" + "=" * 60)
    print(f"FEATURE ABLATION REPORT (mean +/- std over {report['seed_count']} seeds)")
    print("=" * 60)

    print("\nStage 1: Classification (can this integrator find ANY stable step?)")
    print(f"Baseline features: {', '.join(BASIC_FEATURE_COLUMNS_3D)}")
    for integrator in INTEGRATORS:
        acc_mean, acc_std = report["baseline_clf_acc"][integrator]
        f1_mean, f1_std = report["baseline_clf_f1"][integrator]
        print(f"  {integrator.capitalize():9s}: accuracy = {acc_mean:.4f} +/- {acc_std:.4f}   F1 = {f1_mean:.4f} +/- {f1_std:.4f}")

    print(f"\nPhysics-aware features: {', '.join(PHYSICS_FEATURE_COLUMNS_3D)}")
    for integrator in INTEGRATORS:
        acc_mean, acc_std = report["phys_clf_acc"][integrator]
        f1_mean, f1_std = report["phys_clf_f1"][integrator]
        print(f"  {integrator.capitalize():9s}: accuracy = {acc_mean:.4f} +/- {acc_std:.4f}   F1 = {f1_mean:.4f} +/- {f1_std:.4f}")

    print("\nStage 2: Regression (predicted dt, successes-only / uncensored)")
    print(f"Baseline features: {', '.join(BASIC_FEATURE_COLUMNS_3D)}")
    for integrator in INTEGRATORS:
        r2_mean, r2_std = report["baseline_reg_r2"][integrator]
        mae_mean, mae_std = report["baseline_reg_mae"][integrator]
        print(f"  {integrator.capitalize():9s}: R\u00b2 = {r2_mean:.4f} +/- {r2_std:.4f}   MAE = {mae_mean:.5f} +/- {mae_std:.5f}")

    print(f"\nPhysics-aware features: {', '.join(PHYSICS_FEATURE_COLUMNS_3D)}")
    for integrator in INTEGRATORS:
        r2_mean, r2_std = report["phys_reg_r2"][integrator]
        mae_mean, mae_std = report["phys_reg_mae"][integrator]
        print(f"  {integrator.capitalize():9s}: R\u00b2 = {r2_mean:.4f} +/- {r2_std:.4f}   MAE = {mae_mean:.5f} +/- {mae_std:.5f}")
    print()


# =============================================================================
# CHAOTIC VS REGULAR HOLDOUT (3D)
# =============================================================================

def chaotic_holdout_3d(data: pd.DataFrame, random_state: int = 42, n_bins: int = 5) -> dict:
    group_ids = group_orbits_by_region_3d(data, n_bins=n_bins)
    train_mask = test_mask = None

    if "orbit_label" in data.columns and data["orbit_label"].nunique() > 1:
        chaotic_labels = {"L1", "L2", "L3", "wide"}
        is_chaotic = data["orbit_label"].isin(chaotic_labels).to_numpy()
        chaotic_frac = pd.Series(is_chaotic.astype(float)).groupby(group_ids).mean()
        chaotic_groups = set(chaotic_frac[chaotic_frac >= 0.5].index)
        candidate_test_mask = np.array([g in chaotic_groups for g in group_ids])
        candidate_train_mask = ~candidate_test_mask
        if candidate_train_mask.sum() >= 10 and candidate_test_mask.sum() >= 10 and len(chaotic_groups) >= 2:
            train_mask, test_mask = candidate_train_mask, candidate_test_mask
            split_col = "orbit_label (L1/L2/L3/wide vs L4/L5)"

    if train_mask is None:
        if "ftle_3d" in data.columns and data["ftle_3d"].std() > 1e-12:
            split_col, high_is_chaotic = "ftle_3d", True
        elif "c_minus_c_l1_3d" in data.columns:
            split_col, high_is_chaotic = "c_minus_c_l1_3d", False
        else:
            raise ValueError("No usable orbit_label, ftle_3d, or c_minus_c_l1_3d column in data")

        group_median = pd.Series(data[split_col].values).groupby(group_ids).median()
        cutoff = group_median.median()
        if high_is_chaotic:
            chaotic_groups = set(group_median[group_median >= cutoff].index)
        else:
            chaotic_groups = set(group_median[group_median <= cutoff].index)
        test_mask = np.array([g in chaotic_groups for g in group_ids])
        train_mask = ~test_mask

    if train_mask.sum() < 10 or test_mask.sum() < 10:
        raise ValueError(f"Chaotic holdout too small: train={train_mask.sum()}, test={test_mask.sum()}")

    basic_cols = [c for c in BASIC_FEATURE_COLUMNS_3D if c in data.columns]
    phys_cols = [c for c in PHYSICS_FEATURE_COLUMNS_3D if c in data.columns]
    train_idx = np.where(train_mask)[0]
    test_data = data[test_mask].reset_index(drop=True)

    models_basic = fit_hurdle_models(data, basic_cols, train_idx, random_state=random_state, n_estimators=100)
    models_phys = fit_hurdle_models(data, phys_cols, train_idx, random_state=random_state, n_estimators=100)
    pred_basic = predict_hurdle(models_basic, test_data)
    pred_phys = predict_hurdle(models_phys, test_data)

    return {
        "holdout_type": "grouped_chaotic_3d",
        "split_col": split_col,
        "train_size": int(train_mask.sum()),
        "test_size": int(test_mask.sum()),
        "chaotic_groups": len(chaotic_groups),
        "data_test": test_data,
        "pred_basic": pred_basic,
        "pred_phys": pred_phys,
        "models_basic": models_basic,
        "models_phys": models_phys,
    }


# =============================================================================
# FEATURE ABLATION (3D, grouped split)
# =============================================================================

def run_feature_ablation_grouped_3d(
    data: pd.DataFrame, seed_count: int = 10, n_bins: int = 5, n_estimators: int = 100
) -> dict:
    basic_cols = [c for c in BASIC_FEATURE_COLUMNS_3D if c in data.columns]
    phys_cols = [c for c in PHYSICS_FEATURE_COLUMNS_3D if c in data.columns]
    if not basic_cols or not phys_cols:
        raise ValueError("Missing required 3D columns in data")

    keys = [
        "baseline_clf_acc", "baseline_clf_f1", "baseline_reg_r2", "baseline_reg_mae",
        "phys_clf_acc", "phys_clf_f1", "phys_reg_r2", "phys_reg_mae",
    ]
    acc = {k: {i: [] for i in INTEGRATORS} for k in keys}

    for seed in range(seed_count):
        train_idx, test_idx = create_grouped_split_3d(data, test_size=0.3, random_state=seed, n_bins=n_bins)
        test_data = data.iloc[test_idx].reset_index(drop=True)

        for cols, prefix in [(basic_cols, "baseline"), (phys_cols, "phys")]:
            models = fit_hurdle_models(data, cols, train_idx, random_state=seed, n_estimators=n_estimators)
            pred = predict_hurdle(models, test_data)
            for integrator in INTEGRATORS:
                y_true = test_data[f"{integrator}_success"].values
                y_pred = pred[integrator]["feasible"].astype(int)
                acc[f"{prefix}_clf_acc"][integrator].append(accuracy_score(y_true, y_pred))
                acc[f"{prefix}_clf_f1"][integrator].append(f1_score(y_true, y_pred, zero_division=0))
                succ_mask = y_true == 1
                if succ_mask.sum() > 1:
                    y_true_dt = test_data[f"dt_{integrator}"].values[succ_mask]
                    y_pred_dt = pred[integrator]["dt"][succ_mask]
                    finite = np.isfinite(y_pred_dt)
                    if finite.sum() > 1:
                        acc[f"{prefix}_reg_r2"][integrator].append(r2_score(y_true_dt[finite], y_pred_dt[finite]))
                        acc[f"{prefix}_reg_mae"][integrator].append(mean_absolute_error(y_true_dt[finite], y_pred_dt[finite]))

    def _summ(d):
        return {i: (float(np.mean(v)), float(np.std(v))) if v else (float("nan"), float("nan")) for i, v in d.items()}

    report = {k: _summ(acc[k]) for k in keys}
    report["seed_count"] = seed_count
    report["holdout_type"] = "grouped_spatial_3d"
    return report


# =============================================================================
# LEARNING CURVE (3D)
# =============================================================================

def run_dataset_size_study_3d(full_data: pd.DataFrame, sizes=(100, 500, 1000, 2000, 5000, 10000, 20000)) -> pd.DataFrame:
    phys_cols = [c for c in PHYSICS_FEATURE_COLUMNS_3D if c in full_data.columns]
    succ_data = full_data[full_data["rk4_success"] == 1].reset_index(drop=True)
    X = succ_data[phys_cols].values
    y = succ_data["dt_rk4"].values

    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    usable_sizes = []
    max_train = len(X_train_full)
    for size in sizes:
        capped = min(size, max_train)
        if capped not in usable_sizes:
            usable_sizes.append(capped)

    results = []
    for size in usable_sizes:
        model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
        model.fit(X_train_full[:size], y_train_full[:size])
        preds = model.predict(X_test)
        mae_rk4 = mean_absolute_error(y_test, preds)
        r2_rk4 = r2_score(y_test, preds)
        tree_preds = np.array([tree.predict(X_test) for tree in model.estimators_])
        avg_variance = np.mean(np.var(tree_preds, axis=0))
        results.append({"size": size, "mae": mae_rk4, "r2": r2_rk4, "variance": avg_variance})

    return pd.DataFrame(results)


# =============================================================================
# SPEED COMPARISON (3D)
# =============================================================================

def speed_comparison_3d(final_model) -> dict:
    sample_inputs = pd.DataFrame({
        "x": [0.3, 0.5, 0.7, 0.9, 1.1],
        "y": [0.1, -0.1, 0.2, -0.2, 0.0],
        "z": [0.0, 0.05, -0.05, 0.02, 0.0],
        "vx": [0.0, 0.1, -0.1, 0.0, 0.05],
        "vy": [0.5, 0.6, 0.7, 0.8, 0.9],
        "vz": [0.0, 0.0, 0.01, -0.01, 0.0],
        "sim_time": [20.0] * 5,
    })
    raw_inputs = sample_inputs.copy()
    _ = add_orbit_features_3d(raw_inputs, MU, use_variational_ftle=True, ftle_T=20.0)  # JIT warmup

    n_reps = 5
    start = time.time()
    for _ in range(n_reps):
        feats = add_orbit_features_3d(raw_inputs, MU, use_variational_ftle=True, ftle_T=20.0)
        predict_hurdle(final_model, feats)
    ml_time = (time.time() - start) / n_reps

    sample_inputs = add_orbit_features_3d(raw_inputs, MU, use_variational_ftle=True, ftle_T=20.0)

    warm_state = [sample_inputs.iloc[0][c] for c in ["x", "y", "z", "vx", "vy", "vz"]]
    for name in ("euler", "rk4", "leapfrog"):
        find_dt_for_integrator_cr3bp_3d(name, warm_state, 20.0, MU)

    start = time.time()
    for idx in range(len(sample_inputs)):
        row = sample_inputs.iloc[idx]
        state = [row["x"], row["y"], row["z"], row["vx"], row["vy"], row["vz"]]
        sim_time = row["sim_time"]
        for name in ("euler", "rk4", "leapfrog"):
            find_dt_for_integrator_cr3bp_3d(name, state, sim_time, MU)
    brute_time = time.time() - start

    return {
        "ml_time_ms": ml_time * 1000.0,
        "brute_time_ms": brute_time * 1000.0,
        "speedup": brute_time / max(ml_time, 1e-12),
    }


# =============================================================================
# SINGLE ORBIT ANALYSIS (3D)
# =============================================================================

def analyze_single_orbit_3d(final_model, x=0.4, y=0.1, z=0.0, vx=0.0, vy=0.6, vz=0.0, sim_time=20.0) -> dict:
    input_df = pd.DataFrame([[x, y, z, vx, vy, vz, sim_time]],
                             columns=["x", "y", "z", "vx", "vy", "vz", "sim_time"])
    input_df = add_orbit_features_3d(input_df, MU, use_variational_ftle=True, ftle_T=sim_time)
    feature_columns = final_model["feature_columns"]
    features = input_df[feature_columns].values

    means, stds, feasible = {}, {}, {}
    for integrator in INTEGRATORS:
        clf = final_model[integrator]["classifier"]
        reg = final_model[integrator]["regressor"]
        is_feasible = bool(clf.predict(features)[0])
        feasible[integrator] = is_feasible
        if is_feasible and not hasattr(reg, "estimators_"):
            raise ValueError("analyze_single_orbit_3d needs the random-forest regressor (tree-spread uncertainty).")
        if is_feasible:
            tree_preds = np.array([tree.predict(features)[0] for tree in reg.estimators_])
            means[integrator] = float(tree_preds.mean())
            stds[integrator] = float(tree_preds.std())
        else:
            means[integrator] = float("nan")
            stds[integrator] = float("nan")

    dts_for_choice = {i: (means[i] if feasible[i] else None) for i in INTEGRATORS}
    best_name, efficiency_scores = cost_normalized_choice(dts_for_choice)

    return {
        "x": x, "y": y, "z": z, "vx": vx, "vy": vy, "vz": vz, "sim_time": sim_time,
        "means": means, "stds": stds, "feasible": feasible,
        "best_name": best_name, "efficiency_scores": efficiency_scores,
        "input_df": input_df,
    }


# =============================================================================
# FTLE CORRELATION (3D)
# =============================================================================

def run_ftle_correlation_analysis_3d(data: pd.DataFrame, final_model, ftle_column: str = "ftle_3d") -> dict:
    if ftle_column not in data.columns or data[ftle_column].sum() == 0:
        return {"correlation": 0.0, "error": "FTLE column not available or all zeros"}

    succ_mask = data["rk4_success"].values == 1
    if succ_mask.sum() == 0:
        return {"correlation": 0.0, "error": "No RK4 successes"}

    pred = predict_hurdle(final_model, data[succ_mask])
    y_pred = pred["rk4"]["dt"]
    y_true = data["dt_rk4"].values[succ_mask]
    finite = np.isfinite(y_pred)
    y_pred, y_true = y_pred[finite], y_true[finite]
    errors = np.abs(y_pred - y_true)
    ftle = data[ftle_column].values[succ_mask][finite]

    if len(ftle) < 3:
        return {"correlation": 0.0, "error": "Not enough finite predictions"}

    from scipy.stats import pearsonr
    corr, p_value = pearsonr(ftle, errors)

    ftle_quartiles = np.percentile(ftle, [0, 25, 50, 75, 100])
    quartile_errors = []
    for i in range(4):
        upper_inclusive = i == 3
        mask = (ftle >= ftle_quartiles[i]) & (ftle <= ftle_quartiles[i + 1] if upper_inclusive else ftle < ftle_quartiles[i + 1])
        quartile_errors.append(float(np.mean(errors[mask])) if mask.sum() > 0 else float("nan"))

    return {
        "correlation": corr,
        "p_value": p_value,
        "ftle_quartiles": ftle_quartiles,
        "quartile_errors": quartile_errors,
    }


# =============================================================================
# RKF45 COMPARISON (3D)
# =============================================================================

def compare_rkf45_3d(model, data: pd.DataFrame, phys_cols: list, num_samples: int = 100, sim_time: float = 20.0):
    from scipy.integrate import solve_ivp

    sample_data = data.sample(n=min(num_samples, len(data)), random_state=42)
    results = []
    for idx, row in sample_data.iterrows():
        row_sim_time = row["sim_time"] if "sim_time" in row and np.isfinite(row["sim_time"]) else sim_time
        state = [row["x"], row["y"], row["z"], row["vx"], row["vy"], row["vz"]]

        pred = predict_hurdle(model, pd.DataFrame([row[phys_cols].to_dict()]))
        pred_dt = pred["rk4"]["dt"][0] if pred["rk4"]["feasible"][0] else 0.001
        if not np.isfinite(pred_dt) or pred_dt <= 0:
            pred_dt = 0.001

        success, rk4_jacobi_drift, actual_t, _ = max_jacobi_drift_3d(step_rk4_3d, state, pred_dt, row_sim_time, MU)
        rk4_steps = int(round(actual_t / pred_dt)) if pred_dt > 0 else 0
        rk4_evals = rk4_steps * 4
        if not success:
            rk4_jacobi_drift = np.inf

        def cr3bp_derivs(t, y):
            return np.array(cr3bp_derivatives_3d(y.tolist(), MU))

        start_time = time.time()
        sol = solve_ivp(cr3bp_derivs, (0, row_sim_time), state, method="RK45", rtol=1e-6, atol=1e-9)
        wall_time = time.time() - start_time

        if sol.success:
            rkf45_steps = len(sol.t) - 1
            final_state = sol.y[:, -1]
            C_final = jacobi_constant_3d(final_state.tolist(), MU)
            C0 = jacobi_constant_3d(state, MU)
            rkf45_jacobi_drift = abs((C_final - C0) / C0) if abs(C0) > 1e-12 else abs(C_final - C0)
            rkf45_evals = int(sol.nfev)
        else:
            rkf45_steps = 0
            rkf45_jacobi_drift = np.inf
            wall_time = np.inf
            rkf45_evals = 0

        results.append({
            "rk4_steps": rk4_steps, "rk4_jacobi_drift": rk4_jacobi_drift, "rk4_evals": rk4_evals,
            "rkf45_steps": rkf45_steps, "rkf45_jacobi_drift": rkf45_jacobi_drift,
            "rkf45_wall_time": wall_time, "rkf45_evals": rkf45_evals, "pred_dt": pred_dt,
        })

    df_results = pd.DataFrame(results)
    valid = df_results[(df_results["rk4_steps"] > 0) & (df_results["rkf45_steps"] > 0)]
    if len(valid) == 0:
        return {"error": "No valid integrations"}, df_results

    summary = {
        "mean_rk4_steps": valid["rk4_steps"].mean(),
        "mean_rkf45_steps": valid["rkf45_steps"].mean(),
        "mean_rk4_jacobi_drift": valid["rk4_jacobi_drift"].mean(),
        "mean_rkf45_jacobi_drift": valid["rkf45_jacobi_drift"].mean(),
        "mean_rkf45_wall_time": valid["rkf45_wall_time"].mean(),
        "mean_rk4_evals": valid["rk4_evals"].mean(),
        "mean_rkf45_evals": valid["rkf45_evals"].mean(),
        "num_valid": len(valid),
    }
    speedup_evals = summary["mean_rkf45_evals"] / summary["mean_rk4_evals"] if summary["mean_rk4_evals"] > 0 else np.inf
    summary["speedup_evals"] = speedup_evals
    return summary, df_results


# =============================================================================
# HELD-OUT VIOLATION AUDIT (3D) + SAFETY FACTOR
# =============================================================================

def held_out_violation_audit_3d(
    data_test: pd.DataFrame, pred_phys: dict, safety_factor=1.0,
    sim_time: float = 20.0, tolerances: dict | None = None,
    integrators: list[str] | None = None,
) -> pd.DataFrame:
    tolerances = tolerances or TOLERANCES_3D
    integrators = list(integrators) if integrators is not None else list(INTEGRATORS)
    records = []
    for integrator in integrators:
        sf = safety_factor[integrator] if isinstance(safety_factor, dict) else safety_factor
        feasible = pred_phys[integrator]["feasible"]
        dt_pred = pred_phys[integrator]["dt"]
        step_func = INTEGRATORS_3D[integrator]
        tol = tolerances[integrator]
        for i in range(len(data_test)):
            if not feasible[i]:
                continue
            row = data_test.iloc[i]
            row_sim_time = row["sim_time"] if "sim_time" in row and np.isfinite(row["sim_time"]) else sim_time
            dt_to_test = dt_pred[i] * sf
            state = [row["x"], row["y"], row["z"], row["vx"], row["vy"], row["vz"]]
            success, drift, _, _ = max_jacobi_drift_3d(step_func, state, dt_to_test, row_sim_time, MU)
            records.append({
                "integrator": integrator,
                "x": row["x"], "y": row["y"], "z": row["z"],
                "vx": row["vx"], "vy": row["vy"], "vz": row["vz"],
                "dt_predicted": dt_to_test,
                "jacobi_drift": drift,
                "stable": bool(success and drift <= tol),
                "completed": bool(success),
            })
    return pd.DataFrame.from_records(records)


# FIXED: previously, if no candidate factor achieved <=5% violation on
# the tune half, the loop fell through silently to
# chosen_factor = min(candidate_factors) -- reported no differently
# from a factor that actually met the target. On the 20,000-orbit run
# this produced "Chosen safety factor: 0.8  eval coverage: 0.674"
# (32.6% eval violation) with nothing indicating 0.8 never cleared the
# 5% bar on tune either. Now every candidate's tune violation rate is
# kept, and target_met / tune_violation_at_chosen_factor make the
# fallback case visible instead of silent.
#
# EXTENDED: candidate_factors now goes down to 0.30 (was floored at
# 0.80, which never reached the 5% target for any integrator on the
# 20k-orbit run -- see SAFETY_FACTOR_CANDIDATES_3D in config.py), and
# a separate factor is now fit and reported per integrator instead of
# one factor pooled across all three.
def fit_and_validate_safety_factor_3d(
    data_test: pd.DataFrame, pred_phys: dict,
    candidate_factors=SAFETY_FACTOR_CANDIDATES_3D,
    tune_frac: float = 0.5, random_state: int = 42, sim_time: float = 20.0,
    tolerances: dict | None = None,
    target_violation: float = 0.05,
    integrators: list[str] | None = None,
) -> dict:
    integrators = list(integrators) if integrators is not None else list(INTEGRATORS)
    idx = np.arange(len(data_test))
    tune_idx, eval_idx = train_test_split(idx, test_size=1 - tune_frac, random_state=random_state)
    data_tune = data_test.iloc[tune_idx].reset_index(drop=True)
    data_eval = data_test.iloc[eval_idx].reset_index(drop=True)
    pred_tune = {i: {k: v[tune_idx] for k, v in pred_phys[i].items()} for i in INTEGRATORS}
    pred_eval = {i: {k: v[eval_idx] for k, v in pred_phys[i].items()} for i in INTEGRATORS}

    per_integrator = {}
    chosen_factors = {}
    for integrator in integrators:
        tune_violation_by_factor = {}
        chosen_factor = None
        for factor in sorted(candidate_factors, reverse=True):
            audit = held_out_violation_audit_3d(
                data_tune, pred_tune, safety_factor=factor, sim_time=sim_time,
                tolerances=tolerances, integrators=[integrator],
            )
            if len(audit) == 0:
                continue
            violation_rate = float((~audit["stable"]).mean())
            tune_violation_by_factor[factor] = violation_rate
            if violation_rate <= target_violation:
                chosen_factor = factor
                break
        target_met = chosen_factor is not None
        if chosen_factor is None:
            chosen_factor = min(candidate_factors)
        chosen_factors[integrator] = chosen_factor
        per_integrator[integrator] = {
            "chosen_factor": chosen_factor,
            "target_met": target_met,
            "tune_violation_at_chosen_factor": tune_violation_by_factor.get(chosen_factor, float("nan")),
            "tune_violation_by_factor": tune_violation_by_factor,
        }

    eval_audit = held_out_violation_audit_3d(
        data_eval, pred_eval, safety_factor=chosen_factors, sim_time=sim_time,
        tolerances=tolerances, integrators=integrators,
    )
    eval_summary = summarize_violation_audit(eval_audit)
    for integrator in integrators:
        stats = eval_summary.get(integrator)
        per_integrator[integrator]["eval_coverage"] = (
            1.0 - stats["violation_rate"] if stats is not None else float("nan")
        )
        per_integrator[integrator]["n_eval"] = stats["n_evaluated"] if stats is not None else 0

    return {
        "integrators": integrators,
        "per_integrator": per_integrator,
        "chosen_factors": chosen_factors,
        "n_tune": len(data_tune),
        "n_eval": len(data_eval),
        "eval_audit_summary": eval_summary,
    }