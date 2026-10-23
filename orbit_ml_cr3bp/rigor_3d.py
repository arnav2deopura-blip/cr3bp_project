"""
Multi-seed evaluation with confidence intervals and cross-mu transfer.

Seed-to-seed intervals reflect split + forest randomness on ONE generated
dataset (splits overlap, so they understate true sampling variance and do not
include dataset-generation variance). Regression metrics are reported on the
uncensored subset (ceiling-censored labels are lower bounds, not truths) and on
all successes. Features contain no mu, so cross-mu transfer is expected to be
weak unless the dynamics scale similarly; a poor transfer result is a finding,
not a bug.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

from .config import MU, POSITION_TOL_KM, PHYSICS_FEATURE_COLUMNS_3D, BASIC_FEATURE_COLUMNS_3D
from .modeling_3d import (
    create_grouped_split_3d,
    fit_hurdle_models,
    predict_hurdle,
    integrator_selection_stats,
)
from .audit_3d import ReferenceCache, position_error_audit_3d, INTEGRATOR_NAMES


def mean_ci(values, conf=0.95, bounds=None):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    m = float(arr.mean())
    if n < 2:
        return m, float("nan"), float("nan"), n
    h = float(arr.std(ddof=1) / math.sqrt(n) * student_t.ppf((1 + conf) / 2, n - 1))
    lo, hi = m - h, m + h
    if bounds is not None:
        lo, hi = max(lo, bounds[0]), min(hi, bounds[1])
    return m, lo, hi, n


def _bounds_for(col):
    if "_acc" in col or col.endswith("_rate_unscaled"):
        return (0.0, 1.0)
    if "_r2_" in col:
        return (-np.inf, 1.0)
    return None


def metrics_from_prediction(test, pred):
    out = {}
    for integ in INTEGRATOR_NAMES:
        y = test[f"{integ}_success"].to_numpy()
        out[f"{integ}_clf_acc"] = float(accuracy_score(y, pred[integ]["feasible"].astype(int)))
        cens = test[f"{integ}_censored"].fillna(False).astype(bool).to_numpy()
        ok = (y == 1) & np.isfinite(pred[integ]["dt"])
        truth = test[f"dt_{integ}"].to_numpy(dtype=float)
        for name, mask in (("all", ok), ("uncens", ok & ~cens)):
            if mask.sum() > 1:
                out[f"{integ}_r2_{name}"] = float(r2_score(truth[mask], pred[integ]["dt"][mask]))
                out[f"{integ}_mae_{name}"] = float(mean_absolute_error(truth[mask], pred[integ]["dt"][mask]))
            else:
                out[f"{integ}_r2_{name}"] = out[f"{integ}_mae_{name}"] = float("nan")
    st = integrator_selection_stats(test, pred)
    out["selection_acc"] = st["accuracy"]
    out["always_rk4_acc"] = st["rk4_accuracy"]
    out["majority_baseline_acc"] = max(st["true_euler_fraction"], st["true_rk4_fraction"], st["true_leapfrog_fraction"])
    st_ne = integrator_selection_stats(test, pred, exclude=["euler"])
    out["selection_acc_no_euler"] = st_ne["accuracy"]
    out["always_rk4_acc_no_euler"] = st_ne["rk4_accuracy"]
    out["majority_baseline_acc_no_euler"] = max(st_ne["true_rk4_fraction"], st_ne["true_leapfrog_fraction"])
    return out


def multiseed_evaluation(data, seeds, feature_cols=None, n_estimators=100, n_bins=5,
                         audit_rows=0, tol_km=None, mu=MU, conf=0.95, model_kw=None):
    """Per-seed grouped-split metrics; audit_rows > 0 adds the unscaled position-error violation rate on that many held-out rows."""
    if feature_cols is None:
        feature_cols = [c for c in PHYSICS_FEATURE_COLUMNS_3D if c in data.columns]
    if tol_km is None:
        tol_km = POSITION_TOL_KM
    rows = []
    for seed in seeds:
        tr, te = create_grouped_split_3d(data, test_size=0.3, random_state=seed, n_bins=n_bins)
        models = fit_hurdle_models(data, feature_cols, tr, random_state=seed, n_estimators=n_estimators,
                                   **(model_kw or {}))
        test = data.iloc[te].reset_index(drop=True)
        pred = predict_hurdle(models, test)
        m = {"seed": seed, "n_train": int(len(tr)), "n_test": int(len(te))}
        m.update(metrics_from_prediction(test, pred))
        if audit_rows > 0:
            sub = np.random.RandomState(seed).choice(len(test), size=min(audit_rows, len(test)), replace=False)
            aud = position_error_audit_3d(test, pred, tol_km, 1.0, mu, ReferenceCache(mu), INTEGRATOR_NAMES, sub)
            for integ, g in aud.groupby("integrator"):
                m[f"{integ}_violation_rate_unscaled"] = float((~g["stable"]).mean())
        rows.append(m)
    per_seed = pd.DataFrame(rows)
    summary = []
    for col in per_seed.columns:
        if col in ("seed", "n_train", "n_test"):
            continue
        mean, lo, hi, n = mean_ci(per_seed[col], conf, _bounds_for(col))
        summary.append({"metric": col, "mean": mean, "ci_lo": lo, "ci_hi": hi, "n_seeds": n})
    return per_seed, pd.DataFrame(summary)


def _summarize(per_seed, keys, conf=0.95):
    out = []
    for key, g in per_seed.groupby(keys):
        key = key if isinstance(key, tuple) else (key,)
        for col in per_seed.columns:
            if col in keys or col == "seed":
                continue
            mean, lo, hi, n = mean_ci(g[col], conf, _bounds_for(col))
            out.append({**dict(zip(keys, key)), "metric": col, "mean": mean, "ci_lo": lo, "ci_hi": hi,
                        "n_seeds": n})
    return pd.DataFrame(out)


def _default_feature_cols(datasets):
    common = set.intersection(*(set(d.columns) for d in datasets.values()))
    return [c for c in PHYSICS_FEATURE_COLUMNS_3D if c in common]


def cross_mu_evaluation(datasets, seeds, feature_cols=None, n_estimators=100, n_bins=5, conf=0.95, mus=None,
                        model_kw=None):
    """
    datasets: {name: DataFrame} of accuracy-labeled data at different mu (already
    filtered of ftle_nan / reference_divergent rows). Returns per-seed rows and a CI summary.

    Protocols (UPDATED so the numbers are comparable):
      kind="within":   grouped split inside one dataset (train on its train fold, test on its test fold).
      kind="transfer": train on ALL of the source dataset, test on the SAME grouped test fold of the target
                       that "within" uses (previously the whole target was the test set, so within and
                       transfer scores were computed on different rows and could not be compared).
                       No mu feature: the model cannot know which system it is in.
      kind="pooled":   only if mus={name: mu} is given. Train one model on the union of every dataset's
                       train fold WITH mu as an extra feature; test on each dataset's test fold. With only
                       two or three mu values the model can tell the systems apart but cannot
                       extrapolate to an unseen mu.
    Tolerances must already be expressed identically (same normalized tolerance) at generation time.
    """
    if feature_cols is None:
        feature_cols = _default_feature_cols(datasets)
    rows = []
    for seed in seeds:
        splits = {name: create_grouped_split_3d(d, test_size=0.3, random_state=seed, n_bins=n_bins)
                  for name, d in datasets.items()}
        for name, d in datasets.items():
            tr, te = splits[name]
            models = fit_hurdle_models(d, feature_cols, tr, random_state=seed, n_estimators=n_estimators,
                                       **(model_kw or {}))
            test = d.iloc[te].reset_index(drop=True)
            r = {"kind": "within", "source": name, "target": name, "seed": seed}
            r.update(metrics_from_prediction(test, predict_hurdle(models, test)))
            rows.append(r)
        for src, ds in datasets.items():
            models = fit_hurdle_models(ds, feature_cols, np.arange(len(ds)), random_state=seed,
                                       n_estimators=n_estimators, **(model_kw or {}))
            for tgt, dt in datasets.items():
                if tgt == src:
                    continue
                test = dt.iloc[splits[tgt][1]].reset_index(drop=True)
                r = {"kind": "transfer", "source": src, "target": tgt, "seed": seed}
                r.update(metrics_from_prediction(test, predict_hurdle(models, test)))
                rows.append(r)
        if mus is not None and len(datasets) > 1:
            parts = []
            for name, d in datasets.items():
                part = d.iloc[splits[name][0]].copy()
                part["mu"] = mus[name]
                parts.append(part)
            pool = pd.concat(parts, ignore_index=True)
            models = fit_hurdle_models(pool, list(feature_cols) + ["mu"], np.arange(len(pool)),
                                       random_state=seed, n_estimators=n_estimators, **(model_kw or {}))
            for name, d in datasets.items():
                test = d.iloc[splits[name][1]].copy()
                test["mu"] = mus[name]
                test = test.reset_index(drop=True)
                r = {"kind": "pooled", "source": "pooled", "target": name, "seed": seed}
                r.update(metrics_from_prediction(test, predict_hurdle(models, test)))
                rows.append(r)
    per_seed = pd.DataFrame(rows)
    return per_seed, _summarize(per_seed, ["kind", "source", "target"], conf)


def replication_evaluation(datasets, seeds, feature_cols=None, n_estimators=100, n_bins=5, conf=0.95,
                           model_kw=None):
    """
    datasets: {name: DataFrame}, SAME mu and settings, generated with different base seeds (independent draws of
    orbits). Everything is trained and scored the same way so the kinds are comparable:
      kind="within":         train on a random 70% of dataset A, test on the other 30% of A (same distribution).
      kind="replicate":      the SAME model (trained on A's 70%), tested on every row of an independent dataset B.
                             replicate - within is therefore the dataset-to-dataset effect (generation variance), not a
                             difference in split type or training size.
      kind="within_grouped": a region-shift reference (grouped split inside A, as in the other commands); reported for
                             context only, NOT comparable with the two kinds above.
    Returns (per_seed, summary_by_pair, summary_across_datasets). The across-dataset summary averages over seeds first
    and then takes a t-interval over dataset(-pair) means; kind="replicate_minus_within" is the paired difference per
    SOURCE dataset (mean over seeds and targets), with one value per dataset, so with 3 datasets its interval is wide
    by design. Pairs sharing a dataset are not independent: treat the pair-level summaries as descriptive.
    """
    if feature_cols is None:
        feature_cols = _default_feature_cols(datasets)
    rows = []
    for seed in seeds:
        for src, ds in datasets.items():
            tr, te = train_test_split(np.arange(len(ds)), test_size=0.3, random_state=seed)
            tr, te = np.sort(tr), np.sort(te)
            models = fit_hurdle_models(ds, feature_cols, tr, random_state=seed, n_estimators=n_estimators,
                                       **(model_kw or {}))
            within = ds.iloc[te].reset_index(drop=True)
            r = {"kind": "within", "source": src, "target": src, "seed": seed}
            r.update(metrics_from_prediction(within, predict_hurdle(models, within)))
            rows.append(r)
            for tgt, dt in datasets.items():
                if tgt == src:
                    continue
                test = dt.reset_index(drop=True)
                r = {"kind": "replicate", "source": src, "target": tgt, "seed": seed}
                r.update(metrics_from_prediction(test, predict_hurdle(models, test)))
                rows.append(r)
            trg, teg = create_grouped_split_3d(ds, test_size=0.3, random_state=seed, n_bins=n_bins)
            models_g = fit_hurdle_models(ds, feature_cols, trg, random_state=seed, n_estimators=n_estimators,
                                         **(model_kw or {}))
            test_g = ds.iloc[teg].reset_index(drop=True)
            r = {"kind": "within_grouped", "source": src, "target": src, "seed": seed}
            r.update(metrics_from_prediction(test_g, predict_hurdle(models_g, test_g)))
            rows.append(r)
    per_seed = pd.DataFrame(rows)
    by_pair = _summarize(per_seed, ["kind", "source", "target"], conf)
    across = []
    metric_cols = [c for c in per_seed.columns if c not in ("kind", "source", "target", "seed")]
    for kind, g in per_seed.groupby("kind"):
        pair_means = g.groupby(["source", "target"])[metric_cols].mean()
        for col in metric_cols:
            mean, lo, hi, n = mean_ci(pair_means[col], conf, _bounds_for(col))
            across.append({"kind": kind, "metric": col, "mean": mean, "ci_lo": lo, "ci_hi": hi, "n_pairs": n})
    w = per_seed[per_seed["kind"] == "within"].groupby("source")[metric_cols].mean()
    rp = per_seed[per_seed["kind"] == "replicate"].groupby("source")[metric_cols].mean()
    diff = (rp - w).dropna(how="all")
    for col in metric_cols:
        mean, lo, hi, n = mean_ci(diff[col], conf, None)
        across.append({"kind": "replicate_minus_within", "metric": col, "mean": mean, "ci_lo": lo, "ci_hi": hi,
                       "n_pairs": n})
    return per_seed, by_pair, pd.DataFrame(across)


def default_feature_sets(data, leave_one_out=True):
    full = [c for c in PHYSICS_FEATURE_COLUMNS_3D if c in data.columns]
    basic = [c for c in BASIC_FEATURE_COLUMNS_3D if c in data.columns]
    sets = {"full": full, "no_ftle": [c for c in full if c != "ftle_3d"], "state_only": basic}
    if leave_one_out:
        for c in full:
            if c not in basic:
                sets[f"drop_{c}"] = [x for x in full if x != c]
    return sets


def feature_ablation_evaluation(data, seeds, feature_sets=None, n_estimators=100, n_bins=5, conf=0.95,
                                model_kw=None):
    """
    Same grouped splits and seeds for every feature set. ftle_3d is the one
    feature that needs an integration over the whole horizon (its cost is not
    free), so "full" vs "no_ftle" is the key comparison; paired per-seed
    differences versus "full" are returned with t-intervals. Seed intervals
    reflect split + forest randomness on one dataset only.
    Returns (per_seed, summary, deltas_vs_full).
    """
    feature_sets = feature_sets or default_feature_sets(data)
    rows = []
    for seed in seeds:
        tr, te = create_grouped_split_3d(data, test_size=0.3, random_state=seed, n_bins=n_bins)
        test = data.iloc[te].reset_index(drop=True)
        for name, cols in feature_sets.items():
            models = fit_hurdle_models(data, cols, tr, random_state=seed, n_estimators=n_estimators,
                                       **(model_kw or {}))
            r = {"seed": seed, "feature_set": name}
            r.update(metrics_from_prediction(test, predict_hurdle(models, test)))
            rows.append(r)
    per_seed = pd.DataFrame(rows)
    summary = _summarize(per_seed, ["feature_set"], conf)
    deltas = []
    if "full" in feature_sets:
        full = per_seed[per_seed["feature_set"] == "full"].set_index("seed")
        metric_cols = [c for c in per_seed.columns if c not in ("seed", "feature_set")]
        for name in feature_sets:
            if name == "full":
                continue
            g = per_seed[per_seed["feature_set"] == name].set_index("seed")
            for col in metric_cols:
                mean, lo, hi, n = mean_ci((g[col] - full[col]).dropna(), conf, None)
                deltas.append({"feature_set": name, "metric": col, "delta_mean": mean, "delta_ci_lo": lo,
                               "delta_ci_hi": hi, "n_seeds": n})
    return per_seed, summary, pd.DataFrame(deltas)
