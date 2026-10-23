"""
Lockstep (SIMD/GPU-style) cost MODEL for ensembles of orbits.

Not a GPU measurement. In a lockstep batch of B orbits every lane occupies a
slot until the slowest lane finishes, so batch cost = len(batch) * max(RHS
evals in batch). A fixed-step run's eval count is known BEFORE launch
(ceil(sim_time/dt) * evals_per_step), so orbits can be sorted into batches of
similar length; an adaptive solver's count is only known after it runs.

Scenarios per batch size B (counts taken from matched_accuracy_baseline_3d):
  ideal_*              sum of counts (no idle lanes; lower bound for each method)
  ml_sorted            ML counts, batched after sorting by predicted count
  ml_unsorted          ML counts, random batches
  adaptive_unsorted    adaptive nfev, random batches (the realistic case)
  adaptive_oracle      adaptive nfev sorted by true nfev (unachievable pre-launch)
Headline ratios: ml_sorted / adaptive_unsorted (<1 favors ML) and
ml_sorted / adaptive_oracle (pessimistic for ML).

Ignores per-evaluation cost differences between methods, adaptive step-accept
branching beyond idle lanes, memory traffic, and kernel-launch overhead. If
ML only wins under this model, the claim to test next is a measured GPU run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _batch_cost(counts, batch_size):
    c = np.asarray(counts, dtype=float)
    total = 0.0
    for i in range(0, len(c), batch_size):
        chunk = c[i:i + batch_size]
        total += len(chunk) * chunk.max()
    return float(total)


def lockstep_cost_model(df, adaptive_col, ml_col="ml_evals", batch_sizes=(32, 128),
                        n_shuffles=200, random_state=42):
    d = df.dropna(subset=[ml_col, adaptive_col])
    ml = d[ml_col].to_numpy(dtype=float)
    ad = d[adaptive_col].to_numpy(dtype=float)
    rng = np.random.RandomState(random_state)
    rows = []
    for B in batch_sizes:
        ml_uns, ad_uns = [], []
        for _ in range(n_shuffles):
            perm = rng.permutation(len(d))
            ml_uns.append(_batch_cost(ml[perm], B))
            ad_uns.append(_batch_cost(ad[perm], B))
        r = {
            "batch_size": B, "n_orbits": int(len(d)),
            "ideal_ml": float(ml.sum()), "ideal_adaptive": float(ad.sum()),
            "ml_sorted": _batch_cost(np.sort(ml), B), "ml_unsorted": float(np.mean(ml_uns)),
            "adaptive_unsorted": float(np.mean(ad_uns)), "adaptive_oracle": _batch_cost(np.sort(ad), B),
        }
        r["ratio_ml_sorted_over_adaptive_unsorted"] = r["ml_sorted"] / r["adaptive_unsorted"]
        r["ratio_ml_sorted_over_adaptive_oracle"] = r["ml_sorted"] / r["adaptive_oracle"]
        r["lane_efficiency_ml_sorted"] = r["ideal_ml"] / r["ml_sorted"]
        r["lane_efficiency_adaptive_unsorted"] = r["ideal_adaptive"] / r["adaptive_unsorted"]
        rows.append(r)
    return pd.DataFrame(rows)