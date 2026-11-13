"""
Study E, draft section 7 item 5: the pre-run out-of-distribution gate of E5 and its evaluation on the JPL periodic-orbit families. Nothing
here is used by a sealed kernel.

Gate (P19's distance, made into a rule). Feature space: the 25 inputs of the frozen Sundman policy, each standardized by its mean and standard
deviation over the training rows (Study C D_train and D_train_enriched, 94,611 usable rows). Score of a point: mean Euclidean distance to its
k = 10 nearest training rows. Threshold: the split-conformal (1 - alpha) quantile of the scores of the usable E_select rows, alpha = 0.05, i.e.
the ceil((n + 1)(1 - alpha))-th smallest score. A run abstains when its score is above the threshold; an abstained run is integrated by the
certified comparator instead of the policy, and its violation is the comparator's.

Families. The gate uses no periodic-orbit data, so every JPL family is held out from it by construction; E5's "leave-one-family-out" is
evaluated as the per-family violation rates and their family-balanced mean (confirm_d_3d.d5_ood, the D5 statistic, H1: below 10 %). The members
are a third evenly spaced sample of each JPL stream, disjoint from the Study D dry and confirmation samples (select_indices_e).

Pilot (2026-11-13, Study D data, posthoc-style, no Study E data): with the threshold set on the Study D selection set at alpha = 0.05, the Study D
replication rows abstained 4.95 % overall, 3.90 % at 1-3 TU, 3.74 % at 1-8 TU and 7.86 % at 8-20 TU; so the in-distribution criterion of E5
(abstention <= 5 % on E_confirm, 1-3 TU) is expected to hold with about one point to spare.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["K_NN", "ALPHA_GATE", "KnnGate", "conformal_threshold", "select_indices_e", "gated_violations"]

K_NN = 10
ALPHA_GATE = 0.05


def conformal_threshold(scores, alpha=ALPHA_GATE):
    """The ceil((n + 1)(1 - alpha))-th smallest score (split conformal); +inf if that rank exceeds n."""
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    r = int(np.ceil((n + 1) * (1.0 - alpha)))
    return float(np.inf) if r > n else float(s[r - 1])


class KnnGate:
    """fit(train_df, features) -> .score(df) (mean distance to the k nearest training rows, standardized space) -> .calibrate(df) sets the
    threshold -> .abstain(df)."""

    def __init__(self, k=K_NN, alpha=ALPHA_GATE):
        self.k, self.alpha = int(k), float(alpha)
        self.threshold = None

    def fit(self, train_df, features):
        from sklearn.neighbors import NearestNeighbors
        self.features = list(features)
        X = train_df[self.features].to_numpy(dtype=float)
        self.mean_, self.sd_ = X.mean(axis=0), X.std(axis=0)
        self.sd_[self.sd_ == 0] = 1.0
        self.nn_ = NearestNeighbors(n_neighbors=self.k).fit((X - self.mean_) / self.sd_)
        self.n_train = int(len(X))
        return self

    def score(self, df):
        X = df[self.features].to_numpy(dtype=float)
        d, _ = self.nn_.kneighbors((X - self.mean_) / self.sd_)
        return d.mean(axis=1)

    def calibrate(self, df):
        sc = self.score(df)
        self.threshold = conformal_threshold(sc, self.alpha)
        self.n_cal = int(len(sc))
        return self.threshold

    def abstain(self, df):
        if self.threshold is None:
            raise RuntimeError("calibrate the gate first")
        return self.score(df) > self.threshold

    def spec(self):
        return {"k": self.k, "alpha": self.alpha, "threshold": self.threshold, "n_train": getattr(self, "n_train", None),
                "n_cal": getattr(self, "n_cal", None), "features": self.features}


def select_indices_e(n_total, n):
    """A third evenly spaced sample of a stream: round((j + 0.25) * step), minus every index of the Study D confirmation and dry samples
    (periodic_d.select_indices), so the Study E members are new."""
    from .periodic_d import select_indices
    step = (n_total - 1) / n
    used = set(select_indices(n_total, n, "confirm")) | set(select_indices(n_total, n, "dry"))
    return sorted({int(round((j + 0.25) * step)) for j in range(n)} - used)


def gated_violations(abstain, policy_violation, comparator_violation):
    """Per run: the comparator's violation where the gate abstained, the policy's elsewhere."""
    a = np.asarray(abstain, dtype=bool)
    return np.where(a, np.asarray(comparator_violation, dtype=bool), np.asarray(policy_violation, dtype=bool))


def build_e5_runs(jpl_dir, n_per_stream=50, n_jobs=8):
    """The E5 runs: the select_indices_e members of every JPL stream (the raw responses of fetch_jpl_d.py), the Study D member quality filter,
    horizons k x period (periodic_d.build_runs) and the two references with the trust flag (periodic_d.attach_references). Returns
    (members, runs, mu_jpl); the test rows are runs[in_range & ref_ok]. Mirrors run_part_d_3d.cmd_d5 up to the policy runs."""
    import os
    from multiprocessing import Pool
    from . import periodic_d as pdd
    from .config import collision_radii
    mus, members = set(), []
    for stream, group in pdd.STREAM_TO_GROUP.items():
        df, mu_j = pdd.load_stream(os.path.join(jpl_dir, stream + ".json"))
        if df is None or not len(df):
            continue
        mus.add(mu_j)
        idx = select_indices_e(len(df), n_per_stream)
        sub = df.iloc[idx].copy()
        sub["stream"], sub["group"], sub["idx"] = stream, group, idx
        members.append(sub)
    if len(mus) != 1:
        raise RuntimeError(f"JPL responses disagree on the mass ratio: {mus}")
    mu = mus.pop()
    M = pd.concat(members, ignore_index=True)
    rc1, rc2 = collision_radii(mu)
    jobs = [(M.loc[i, ["x", "y", "z", "vx", "vy", "vz"]].to_numpy(dtype=float), float(M.loc[i, "period"]), mu) for i in M.index]
    with Pool(n_jobs) as pool:
        diag = pool.map(pdd.member_diagnostics, jobs, chunksize=4)
    M = pd.concat([M, pd.DataFrame(diag)], axis=1)
    M["quality_ok"] = M["reached"] & (M["closure"] <= pdd.CLOSURE_TOL) & (M["min_r1"] > rc1) & (M["min_r2"] > rc2)
    runs = pdd.build_runs(M[M["quality_ok"]], mu)
    runs = pdd.attach_references(runs, mu, n_jobs)
    runs["cluster"] = runs["stream"].str.replace("_N", "", regex=False).str.replace("_S", "", regex=False) + "#" + runs["idx"].astype(str)
    return M, runs, mu
