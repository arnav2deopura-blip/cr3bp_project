"""
mu dial (E7, exploratory, pre-registered 2026-10-26): the same study at several mass ratios, with the sampling geometry scaled by the
secondary's Hill radius r_H = (mu/3)^(1/3) and the local circular speed v_H = sqrt(mu / r_H), so the sample covers the same dynamical
neighbourhood at every mu. The constants are the Earth-Moon values of the old box sampling (0.1 DU offset, 0.15 DU/TU kick around L1/L2,
+-0.6 DU/TU near the secondary, inner radius 1.2 Moon radii), so mu = 0.01215 reproduces it up to the outer near-secondary radius
(0.15 DU became r_H = 0.159 DU). L3/L4/L5 and the "wide" box are primary-dominated and unchanged.
"""
from __future__ import annotations

import json
import math
import os
import random
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd

from .config import (MU, R_MOON_KM, DU_KM, POSITION_TOL_KM, SAMPLING_WEIGHTS_3D, ACCURACY_SIM_TIME_LOG_MIN,
                     ACCURACY_SIM_TIME_LOG_MAX, du_km_for_mu, collision_radii, system_params)
from .physics_3d import generate_3d_orbit
from .sweeps_b_3d import _row_for
from . import protocol_3d as proto

R_H_EM = (MU / 3.0) ** (1.0 / 3.0)
V_H_EM = math.sqrt(MU / R_H_EM)
A_L12 = 0.1 / R_H_EM                 # L1/L2 position offset in units of r_H            (0.628)
B_L12 = 0.15 / V_H_EM                # L1/L2 velocity kick in units of v_H              (0.543)
C_NEAR = 0.6 / V_H_EM                # near-secondary velocity half-range in units of v_H (2.17)
R_LO_OVER_RH = 1.2 * (R_MOON_KM / DU_KM) / R_H_EM   # inner near-secondary radius in units of r_H (0.034)
DIAL_MUS = (1e-6, 1e-4, 1e-3, MU, 0.05)


def hill_radius(mu):
    return (mu / 3.0) ** (1.0 / 3.0)


def hill_speed(mu):
    return math.sqrt(mu / hill_radius(mu))


def sample_state(mu):
    """One state and its label from the mu-scaled mixture; uses the global numpy RNG (seeded by the caller)."""
    rh, vh = hill_radius(mu), hill_speed(mu)
    roll = np.random.rand()
    w_l, w_w = SAMPLING_WEIGHTS_3D["lagrange"], SAMPLING_WEIGHTS_3D["wide"]
    if roll < w_l:
        point = str(np.random.choice(["L1", "L2", "L3", "L4", "L5"]))
        if point in ("L1", "L2"):
            return generate_3d_orbit(mu=mu, point=point, sampling_mode="lagrange", offset=A_L12 * rh,
                                     velocity_perturb=B_L12 * vh, return_label=True)
        return generate_3d_orbit(mu=mu, point=point, sampling_mode="lagrange", offset=0.1, velocity_perturb=0.15, return_label=True)
    if roll < w_l + w_w:
        return generate_3d_orbit(mu=mu, sampling_mode="wide", return_label=True)
    r_lo = R_LO_OVER_RH * rh
    r2 = float(np.exp(np.random.uniform(np.log(r_lo), np.log(rh))))
    u = np.random.normal(size=3)
    u = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-12 else np.array([1.0, 0.0, 0.0])
    x_sec = 1.0 - mu
    v = np.random.uniform(-C_NEAR * vh, C_NEAR * vh, size=3)
    return [x_sec + r2 * u[0], r2 * u[1], r2 * u[2], v[0], v[1], v[2]], "near_secondary"


def one_orbit(idx, mu, base_seed, tol_km):
    seed32 = int(np.random.SeedSequence([int(base_seed), int(idx)]).generate_state(1)[0])
    np.random.seed(seed32)
    random.seed(seed32)
    state, label = sample_state(mu)
    T = 10.0 ** random.uniform(ACCURACY_SIM_TIME_LOG_MIN, ACCURACY_SIM_TIME_LOG_MAX)
    return _row_for(list(state), T, mu, tol_km, label, idx)


def generate(mu, n, base_seed, out_csv, n_jobs=8, tol_km=POSITION_TOL_KM):
    r1c, r2c = collision_radii(mu)
    rows = []
    with Pool(n_jobs) as pool:
        for k, r in enumerate(pool.imap_unordered(partial(one_orbit, mu=mu, base_seed=base_seed, tol_km=tol_km), range(n), chunksize=8)):
            rows.append(r)
            if (k + 1) % 2000 == 0:
                print(f"  mu={mu:g}: {k + 1}/{n}", flush=True)
    df = pd.DataFrame(rows).sort_values("orbit_idx").reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    sp = system_params(mu)
    meta = {"label_pipeline_version": 4, "pipeline": "v4", "label_mode": "accuracy_monotone", "kind": "mu_dial", "mu": mu,
            "system": sp["name"], "generic_radii": sp["generic"], "collision_radius_primary_du": r1c, "collision_radius_secondary_du": r2c,
            "r_hill": hill_radius(mu), "v_hill": hill_speed(mu), "tol_km": tol_km, "du_km": du_km_for_mu(mu), "base_seed": base_seed,
            "num_orbits": n, "constants": {"A_L12": A_L12, "B_L12": B_L12, "C_NEAR": C_NEAR, "R_LO_OVER_RH": R_LO_OVER_RH},
            "sampling_weights": SAMPLING_WEIGHTS_3D, "sim_time_log_range": [ACCURACY_SIM_TIME_LOG_MIN, ACCURACY_SIM_TIME_LOG_MAX]}
    with open(out_csv.replace(".csv", "_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return df


# ---------------------------------------------------------------------------------------------------------------------
# Part C Track 4.1: the dial without the confound between mu and how close the orbits come to the secondary
# ---------------------------------------------------------------------------------------------------------------------

HILL_BAND_EDGES = (0.0, 1.0, 3.0, 10.0, np.inf)
HILL_BAND_LABELS = ("<1", "1-3", "3-10", ">10")


def hill_band(min_r2, mu):
    """Closest approach to the secondary in Hill radii, banded (plan 4.1.1)."""
    x = np.asarray(min_r2, dtype=float) / hill_radius(mu)
    return pd.cut(x, bins=list(HILL_BAND_EDGES), labels=list(HILL_BAND_LABELS), right=False).astype(object)


def stratify_by_hill(led, mu, comp="RK45_spec", min_n=50, n_boot=1000):
    """Per closest-approach band: n, share where a perfect fixed step is cheaper, median paradigm gap and integration-only ML gap (dex) with
    bootstrap intervals, violation rate. A mu effect WITHIN bands is evidence of mu itself; a difference only BETWEEN bands means the dial measured
    encounter depth."""
    from .audit_3d import clopper_pearson
    from .failure_study_3d import bootstrap_ci
    d = led.copy()
    d["band"] = hill_band(d["min_r2"], mu)
    rows = []
    for b in HILL_BAND_LABELS:
        g = d[d["band"] == b]
        if len(g) < min_n:
            rows.append({"mu": mu, "band": b, "n": int(len(g))})
            continue
        h = g[g[f"{comp}_nfev"].notna() & g["C_fix_star"].notna()]
        gap = np.log10(h["C_fix_star"] / h[f"{comp}_nfev"])
        k = int((h["C_fix_star"] < h[f"{comp}_nfev"]).sum())
        ml = g[g["ml_ok"].fillna(False).astype(bool) & g["ml_evals"].notna() & g[f"{comp}_nfev"].notna()]
        mgap = np.log10(ml["ml_evals"] / ml[f"{comp}_nfev"])
        nv = int(g["violation"].sum())
        nh = int(g["has_choice"].sum())
        rows.append({"mu": mu, "band": b, "n": int(len(g)), "fixed_cheaper_share": k / max(len(h), 1), "fixed_cheaper_ci95": clopper_pearson(k, len(h)),
                     "paradigm_gap_median": bootstrap_ci(gap, n_boot=n_boot), "ml_int_gap_median": bootstrap_ci(mgap, n_boot=n_boot),
                     "violation_rate": nv / max(nh, 1), "violation_ci95": clopper_pearson(nv, nh)})
    return pd.DataFrame(rows)


def matched_subsample(values_by_mu, target_edges=None, seed=0, min_per_bin=30):
    """Matched resampling (plan 4.1.2). values_by_mu: {mu: array of log10(min_r2 / r_H)}. The common target histogram is the pooled histogram
    restricted to bins that EVERY system populates (>= min_per_bin orbits); each mu is subsampled (without replacement, probability proportional
    to target / own density per bin, scaled so the largest ratio is 1) so the bin shares match the target. Returns {mu: boolean mask} and
    the target shares per bin."""
    rng = np.random.RandomState(seed)
    allv = np.concatenate([np.asarray(v) for v in values_by_mu.values()])
    edges = np.asarray(target_edges) if target_edges is not None else np.linspace(np.floor(allv.min()), np.ceil(allv.max()), 13)
    counts = {mu: np.histogram(v, bins=edges)[0] for mu, v in values_by_mu.items()}
    populated = np.all([c >= min_per_bin for c in counts.values()], axis=0)
    pooled = sum(counts.values()).astype(float)
    target = np.where(populated, pooled, 0.0)
    target = target / target.sum()
    masks = {}
    for mu, v in values_by_mu.items():
        v = np.asarray(v)
        c = counts[mu].astype(float)
        own = c / c.sum()
        ratio = np.where(populated & (own > 0), target / np.where(own > 0, own, 1.0), 0.0)
        ratio = ratio / ratio.max()
        b = np.clip(np.digitize(v, edges) - 1, 0, len(edges) - 2)
        masks[mu] = rng.rand(len(v)) < ratio[b]
    return masks, {"edges": edges.tolist(), "target_share": target.tolist(), "populated": populated.tolist()}


def dimensionless_features(df, mu):
    """Inputs that do not carry the system's units (plan 4.1.4): r1 (DU); r2 / r_H; the velocity components over the local circular speed of the
    nearer body; T (TU, universal in the CR3BP); (C - C_L1) / (C_L1 - C_L3). Returns a DataFrame of those columns."""
    from .physics_3d import jacobi_constant_3d, libration_point_location
    x1, _ = libration_point_location(mu, "L1")
    x3, _ = libration_point_location(mu, "L3")
    c_l1 = jacobi_constant_3d([x1, 0, 0, 0, 0, 0], mu)
    c_l3 = jacobi_constant_3d([x3, 0, 0, 0, 0, 0], mu)
    r1 = np.maximum(np.sqrt((df["x"] + mu) ** 2 + df["y"] ** 2 + df["z"] ** 2), 1e-9)
    r2 = np.maximum(np.sqrt((df["x"] + mu - 1.0) ** 2 + df["y"] ** 2 + df["z"] ** 2), 1e-9)
    near2 = (r2 / hill_radius(mu)) < (r1 / 1.0)
    gm = np.where(near2, mu, 1.0 - mu)
    r = np.where(near2, r2, r1)
    vc = np.sqrt(gm / r)
    out = pd.DataFrame({"r1": r1, "r2_over_rH": r2 / hill_radius(mu), "vx_over_vc": df["vx"] / vc, "vy_over_vc": df["vy"] / vc, "vz_over_vc": df["vz"] / vc,
                        "T": df["sim_time"], "c_rel": (df["jacobi_3d"] - c_l1) / (c_l1 - c_l3), "near_secondary": near2.astype(float)})
    return out
