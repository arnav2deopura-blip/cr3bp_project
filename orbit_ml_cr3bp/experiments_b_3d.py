"""
Experiment helpers for the failure study (Part B): quantile calibration diagnostics (E4/E5), kNN out-of-distribution score (E4),
physics-rule baseline (E10), step-over analysis (E9). Pure functions on DataFrames/arrays; the integration passes live in
policy_b_3d.evaluate_orbits.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import MU
from .policy_b_3d import FEATS_STATE


# ---------------------------------------------------------------------------------------------------------------------
# quantile-model diagnostics
# ---------------------------------------------------------------------------------------------------------------------

def pinball(y, yhat, q):
    d = np.asarray(y, dtype=float) - np.asarray(yhat, dtype=float)
    return np.maximum(q * d, (q - 1.0) * d)


def fit_log_quantile(train, integrator, q, features=FEATS_STATE, seed=42):
    """HGB quantile regressor on log10 d* of one integrator, trained on rows where that integrator has a valid label."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    t = train[train[f"{integrator}_success"] == 1]
    reg = HistGradientBoostingRegressor(loss="quantile", quantile=q, random_state=seed)
    reg.fit(t[features].to_numpy(dtype=float), np.log10(t[f"dt_{integrator}"].to_numpy(dtype=float)))
    return reg


def learning_curve(train, ev, integrator, sizes, seeds, q=0.1, features=FEATS_STATE, group_col="cell"):
    """For each training size and seed: fit on a random subset of `train`, score on `ev` (rows with a valid label). Returns a
    long DataFrame (size, seed, group, n, pinball, over_rate, mae_dex)."""
    e = ev[ev[f"{integrator}_success"] == 1]
    y = np.log10(e[f"dt_{integrator}"].to_numpy(dtype=float))
    X = e[features].to_numpy(dtype=float)
    rows = []
    for n in sizes:
        for s in seeds:
            rng = np.random.RandomState(1000 * s + 17)
            sub = train.iloc[rng.choice(len(train), size=min(n, len(train)), replace=False)]
            reg = fit_log_quantile(sub, integrator, q, features, seed=s)
            yh = reg.predict(X)
            for g, idx in [("ALL", np.arange(len(e)))] + [(c, np.where(e[group_col].to_numpy() == c)[0])
                                                          for c in sorted(e[group_col].unique())]:
                if len(idx) == 0:
                    continue
                rows.append({"size": min(n, len(train)), "seed": s, "group": g, "n": int(len(idx)),
                             "pinball": float(pinball(y[idx], yh[idx], q).mean()),
                             "over_rate": float((yh[idx] > y[idx]).mean()),
                             "mae_dex": float(np.abs(yh[idx] - y[idx]).mean())})
    return pd.DataFrame(rows)


def learning_slope(lc, group, min_sizes=3):
    """Slope of mean pinball loss against log10(size) for one group, with a bootstrap over seeds. Negative = still improving."""
    g = lc[lc["group"] == group]
    if g["size"].nunique() < min_sizes:
        return {"group": group}
    piv = g.pivot_table(index="seed", columns="size", values="pinball")
    xs = np.log10(piv.columns.to_numpy(dtype=float))
    slopes = []
    rng = np.random.RandomState(0)
    for _ in range(500):
        sel = piv.iloc[rng.randint(0, len(piv), len(piv))].mean(axis=0).to_numpy()
        slopes.append(np.polyfit(xs, sel, 1)[0])
    mean_curve = piv.mean(axis=0).to_numpy()
    return {"group": group, "slope": float(np.polyfit(xs, mean_curve, 1)[0]),
            "slope_ci95": [float(np.quantile(slopes, 0.025)), float(np.quantile(slopes, 0.975))],
            "pinball_first": float(mean_curve[0]), "pinball_last": float(mean_curve[-1]), "n_eval": int(g["n"].iloc[0])}


def reliability(train, ev, integrator, qs=(0.02, 0.05, 0.1, 0.2, 0.5), features=FEATS_STATE, seed=42, group_col="regime"):
    """Nominal quantile vs the fraction of labels that fall BELOW the predicted quantile (should equal q), overall and by group,
    on rows with a valid label."""
    e = ev[ev[f"{integrator}_success"] == 1]
    y = np.log10(e[f"dt_{integrator}"].to_numpy(dtype=float))
    X = e[features].to_numpy(dtype=float)
    rows = []
    for q in qs:
        reg = fit_log_quantile(train, integrator, q, features, seed)
        yh = reg.predict(X)
        below = y < yh   # label below the predicted quantile -> the prediction over-estimates the step
        rows.append({"q": q, "group": "ALL", "n": len(y), "empirical": float(below.mean())})
        for g in sorted(e[group_col].unique()):
            m = e[group_col].to_numpy() == g
            rows.append({"q": q, "group": g, "n": int(m.sum()), "empirical": float(below[m].mean())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------------
# out-of-distribution score
# ---------------------------------------------------------------------------------------------------------------------

def knn_ood(train, cal, ev, features=FEATS_STATE, k=5, q=0.95):
    """Mean distance to the k nearest training rows in standardized feature space. The flag threshold is the q-quantile of the
    CALIBRATION rows' distances (so 5 % of in-distribution rows are flagged by construction)."""
    from sklearn.neighbors import NearestNeighbors
    Xt = train[features].to_numpy(dtype=float)
    mu, sd = Xt.mean(axis=0), Xt.std(axis=0)
    sd[sd == 0] = 1.0
    nn = NearestNeighbors(n_neighbors=k).fit((Xt - mu) / sd)

    def dist(df):
        d, _ = nn.kneighbors((df[features].to_numpy(dtype=float) - mu) / sd)
        return d.mean(axis=1)

    dc, de = dist(cal), dist(ev)
    thr = float(np.quantile(dc, q))
    return de, thr, de > thr


# ---------------------------------------------------------------------------------------------------------------------
# physics rule (E10)
# ---------------------------------------------------------------------------------------------------------------------

def tau0(df, mu=MU):
    r1, r2 = df["r1_3d"].to_numpy(dtype=float), df["r2_3d"].to_numpy(dtype=float)
    return np.minimum(np.sqrt(r1 ** 3 / (1.0 - mu)), np.sqrt(r2 ** 3 / mu))


def fit_physics_rule(train, integrator, q=0.1, mu=MU):
    """log10 d = a + log10 tau0 - b log10 T, fitted by linear quantile regression (two parameters per integrator)."""
    from sklearn.linear_model import QuantileRegressor
    t = train[train[f"{integrator}_success"] == 1]
    y = np.log10(t[f"dt_{integrator}"].to_numpy(dtype=float)) - np.log10(tau0(t, mu))
    X = np.log10(t["sim_time"].to_numpy(dtype=float))[:, None]
    m = QuantileRegressor(quantile=q, alpha=0.0, solver="highs").fit(X, y)
    return {"a": float(m.intercept_), "b": float(-m.coef_[0])}


def physics_predict(df, rule, mu=MU):
    return 10.0 ** (rule["a"] + np.log10(tau0(df, mu)) - rule["b"] * np.log10(df["sim_time"].to_numpy(dtype=float)))


def physics_pred_dict(df, rules, mu=MU):
    out = {}
    for k, r in rules.items():
        out[k] = {"feasible": np.ones(len(df), dtype=bool), "dt": physics_predict(df, r, mu)}
    return out


# ---------------------------------------------------------------------------------------------------------------------
# step-over (E9)
# ---------------------------------------------------------------------------------------------------------------------

def auc_score(pos, neg):
    """P(score_pos > score_neg) + 0.5 P(tie): Mann-Whitney AUC."""
    pos, neg = np.asarray(pos, dtype=float), np.asarray(neg, dtype=float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def guard_table(step_over, violated, thresholds):
    """Cheap runtime guard: flag a run when dt / tau_min exceeds k. Catch rate = flagged among violations, false-alarm rate =
    flagged among runs that met the tolerance."""
    so = np.asarray(step_over, dtype=float)
    v = np.asarray(violated, dtype=bool)
    rows = []
    for k in thresholds:
        flag = so > k
        rows.append({"k": float(k), "catch_rate": float(flag[v].mean()) if v.any() else float("nan"),
                     "false_alarm_rate": float(flag[~v].mean()) if (~v).any() else float("nan"),
                     "flag_rate": float(flag.mean())})
    return pd.DataFrame(rows)
