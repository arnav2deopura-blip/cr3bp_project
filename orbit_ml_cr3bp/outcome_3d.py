"""
Crash/escape ("will the reference trajectory hit a body or leave the system
before the horizon?") classifier.

Orbits whose DOP853 reference collides or escapes used to be dropped (either
via ftle_nan or via reference_divergent), so the hurdle models only ever saw
survivors and could never warn about a crash. They are now kept in the data
(reference_divergent=True, with a reference_outcome string) and this module
trains/evaluates a classifier for them with the same grouped-split protocol as
the rest of the pipeline.

Caveat to report with any result: the FTLE feature comes from integrating the
variational equations over the same horizon, so if the trajectory crashes the
FTLE input has already "seen" it (it is truncated at the crash). Evaluate with
and without ftle_3d (feature_sets below) before calling this a prediction made
from the initial state alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, roc_auc_score

from .config import PHYSICS_FEATURE_COLUMNS_3D
from .modeling_3d import create_grouped_split_3d


def feature_sets_for_crash(data):
    full = [c for c in PHYSICS_FEATURE_COLUMNS_3D if c in data.columns]
    return {"full": full, "no_ftle": [c for c in full if c != "ftle_3d"]}


def fit_crash_classifier(data, feature_cols, train_idx, random_state=42, n_estimators=150):
    y = data["reference_divergent"].astype(bool).to_numpy().astype(int)
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1,
                                 class_weight="balanced")
    clf.fit(data[feature_cols].to_numpy(dtype=float)[train_idx], y[train_idx])
    return clf


def _metrics(y_true, y_pred, p_crash):
    out = {
        "base_rate_test": float(np.mean(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "majority_baseline_acc": float(max(np.mean(y_true), 1.0 - np.mean(y_true))),
    }
    if len(np.unique(y_true)) == 2:
        out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
        out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        out["auc"] = float(roc_auc_score(y_true, p_crash))
    else:
        out["balanced_accuracy"] = out["precision"] = out["recall"] = out["auc"] = float("nan")
    return out


def crash_classifier_evaluation(data, seeds, n_estimators=150, n_bins=5, feature_sets=None):
    """
    data: accuracy-labeled rows INCLUDING reference_divergent rows (ftle_nan
    rows removed because features are undefined for them). Returns
    (per_seed_df, summary_df, recall_by_outcome_df).
    """
    from .rigor_3d import mean_ci

    if "reference_divergent" not in data.columns:
        raise ValueError("data has no reference_divergent column")
    data = data.reset_index(drop=True)
    feature_sets = feature_sets or feature_sets_for_crash(data)
    outcome = data["reference_outcome"].astype(str).to_numpy() if "reference_outcome" in data.columns \
        else np.where(data["reference_divergent"].astype(bool), "diverged", "ok")
    rows, outcome_rows = [], []
    for seed in seeds:
        tr, te = create_grouped_split_3d(data, test_size=0.3, random_state=seed, n_bins=n_bins)
        y_te = data["reference_divergent"].astype(bool).to_numpy()[te].astype(int)
        for name, cols in feature_sets.items():
            clf = fit_crash_classifier(data, cols, tr, random_state=seed, n_estimators=n_estimators)
            x_te = data[cols].to_numpy(dtype=float)[te]
            pred = clf.predict(x_te)
            if len(clf.classes_) == 2:
                p = clf.predict_proba(x_te)[:, list(clf.classes_).index(1)]
            else:
                p = np.full(len(te), float(clf.classes_[0]))
            m = {"seed": seed, "feature_set": name, "n_train": int(len(tr)), "n_test": int(len(te))}
            m.update(_metrics(y_te, pred, p))
            rows.append(m)
            for cat in sorted(set(outcome[te]) - {"ok"}):
                mask = outcome[te] == cat
                outcome_rows.append({"seed": seed, "feature_set": name, "outcome": cat, "n_test": int(mask.sum()),
                                     "recall": float(pred[mask].mean())})
    per_seed = pd.DataFrame(rows)
    summary = []
    for name, g in per_seed.groupby("feature_set"):
        for col in g.columns:
            if col in ("seed", "feature_set"):
                continue
            mean, lo, hi, n = mean_ci(g[col], 0.95, (0.0, 1.0) if col not in ("n_train", "n_test") else None)
            summary.append({"feature_set": name, "metric": col, "mean": mean, "ci_lo": lo, "ci_hi": hi, "n_seeds": n})
    by_outcome = pd.DataFrame(outcome_rows)
    if len(by_outcome):
        by_outcome = by_outcome.groupby(["feature_set", "outcome"]).agg(
            n_test_mean=("n_test", "mean"), recall_mean=("recall", "mean")).reset_index()
    return per_seed, pd.DataFrame(summary), by_outcome
