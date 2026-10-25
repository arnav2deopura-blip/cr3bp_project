"""
Predictability dials (Part B, E7): the horizon sweep relabels the SAME initial states at several horizons T, so differences
between horizons cannot come from the sample. Output rows have the schema of a v4 dataset (features, labels, diagnostics) and a
metadata file, so they go through `run_failure_study_3d.py policy --frozen_policy` like any dataset.
"""
from __future__ import annotations

import json
import os
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd

from .config import MU, POSITION_TOL_KM, du_km_for_mu
from .data_3d import extract_features_3d, label_orbit_accuracy_3d
from .physics_3d import REASON_OK
from . import protocol_3d as proto

STATE_COLS = ["x", "y", "z", "vx", "vy", "vz"]
HORIZONS = (1.0, 2.0, 5.0, 10.0, 20.0)


def _row_for(state, T, mu, tol_km, orbit_label, idx):
    feats = extract_features_3d(state, mu, sim_time=T)
    row = {**feats, "sim_time": T, "orbit_label": orbit_label, "ftle_nan": not np.isfinite(feats.get("ftle_3d", 0.0)),
           "orbit_idx": idx, "mu": mu, "du_km": du_km_for_mu(mu), "tol_km": tol_km}
    lab = label_orbit_accuracy_3d(list(state), T, mu, tol_km=tol_km, monotone=True, ceiling_mode="physical", diagnostics=True)
    row.update(lab)
    for k in ("euler", "rk4", "leapfrog"):
        row[f"{k}_success"] = int(lab[f"{k}_reason"] == REASON_OK)
    return row


def _job(a, horizons, mu, tol_km):
    idx, state, orbit_label = a
    return [_row_for(state, T, mu, tol_km, orbit_label, idx) for T in horizons]


def horizon_sweep(source_csv, out_csv, n_ics=2600, horizons=HORIZONS, seed=0, mu=MU, tol_km=POSITION_TOL_KM, n_jobs=8,
                  require_longest=True, pool_row_ids=None):
    """Pick n_ics usable initial states of `source_csv` (seeded shuffle), relabel each at every horizon, and keep the ICs whose
    reference reaches the longest horizon (so every kept IC appears at every T). Returns the DataFrame written to out_csv."""
    d = pd.read_csv(source_csv, low_memory=False)
    for c in ("ftle_nan", "reference_divergent"):
        d[c] = d[c].map({True: True, False: False, "True": True, "False": False}).fillna(False).astype(bool)
    u = d[~d["ftle_nan"] & ~d["reference_divergent"]]
    if pool_row_ids is not None:     # row_id = position in the source CSV; e.g. the rows a frozen model never trained on
        u = u[u.index.isin(set(int(i) for i in pool_row_ids))]
    pick = u.iloc[np.random.RandomState(seed).choice(len(u), size=min(n_ics, len(u)), replace=False)]
    jobs = [(int(i), [r[c] for c in STATE_COLS], r["orbit_label"]) for i, r in zip(pick.index, pick.to_dict("records"))]
    rows = []
    with Pool(n_jobs) as pool:
        for k, rr in enumerate(pool.imap_unordered(partial(_job, horizons=horizons, mu=mu, tol_km=tol_km), jobs, chunksize=4)):
            rows.extend(rr)
            if (k + 1) % 250 == 0:
                print(f"  horizon sweep: {k + 1}/{len(jobs)} initial states", flush=True)
    df = pd.DataFrame(rows)
    if require_longest:
        tmax = max(horizons)
        ok_ic = df[(df["sim_time"] == tmax) & ~df["reference_divergent"] & ~df["ftle_nan"]]["orbit_idx"].unique()
        df = df[df["orbit_idx"].isin(ok_ic)]
        # an IC that survives tmax survives every shorter horizon; verify rather than assume
        bad = df[df["reference_divergent"] | df["ftle_nan"]]
        if len(bad):
            df = df[~df["orbit_idx"].isin(bad["orbit_idx"].unique())]
    df = df.sort_values(["orbit_idx", "sim_time"]).reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    meta = {"label_pipeline_version": 4, "pipeline": "v4", "label_mode": "accuracy_monotone", "mu": mu, "tol_km": tol_km,
            "horizons": list(horizons), "source_csv": os.path.basename(source_csv), "source_sha256": proto.sha256_file(source_csv),
            "seed": seed, "n_initial_states": int(df["orbit_idx"].nunique()), "kind": "horizon_sweep"}
    with open(out_csv.replace(".csv", "_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return df
