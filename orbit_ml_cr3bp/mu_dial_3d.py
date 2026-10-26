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
