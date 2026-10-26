"""
Osculating two-body "periapsis" features (Part C plan 5.2.2): an ANALYTIC lookahead that costs a few arithmetic operations and no
right-hand-side evaluation. From the current state, the two-body orbit about each body predicts the next periapsis radius and the time
to reach it. It is exact for two-body motion and approximate inside each body's sphere of influence; E3b showed that where and how
close the orbit will pass a body is the information the state-only model lacks.

Frame: rotating, primary at (-mu, 0, 0) with GM1 = 1 - mu, secondary at (1 - mu, 0, 0) with GM2 = mu, angular velocity z-hat = 1.
Inertial velocity relative to a body: v = v_rot + omega x r = (vx - r_y, vy + r_x, vz), r the position relative to that body.

Per body i (columns osc_*_1 for the primary, osc_*_2 for the secondary):
  logrp   log10 of the periapsis radius r_p = h^2 / (GM (1 + e)), floored at 1e-8
  logtaup log10 of tau_p = sqrt(r_p^3 / GM), the dynamical time at periapsis
  tpn     min(t_p / T, 2): time to the next periapsis in horizons (2 = beyond, or no future periapsis)
  reach   1 if t_p <= T
  logrel  log10(|r| / r_H) for the secondary (r_H = (mu/3)^(1/3)); log10 |r1| for the primary
  ecc     osculating eccentricity
Combined:
  osc_logtau_hat   log10 of min over bodies of (tau_p if reach else sqrt(|r|^3 / GM))
Fallbacks (flag columns osc_flag_1/2, not model features): e < 1e-8 (circular): t_p = 0, r_p = |r|; |1 - e| < 1e-6 (parabolic): Barker's
equation for the time, receding parabola has no future periapsis.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from numba import njit

from .config import MU

OSC_FEATURE_COLS = ["osc_logrp_1", "osc_logtaup_1", "osc_tpn_1", "osc_reach_1", "osc_logrel_1", "osc_ecc_1",
                    "osc_logrp_2", "osc_logtaup_2", "osc_tpn_2", "osc_reach_2", "osc_logrel_2", "osc_ecc_2",
                    "osc_logtau_hat"]
OSC_FLAG_COLS = ["osc_flag_1", "osc_flag_2"]
RP_FLOOR = 1e-8
E_CIRCULAR = 1e-8
E_PARABOLIC = 1e-6


@njit(cache=True)
def _periapsis_two_body(rx, ry, rz, vx, vy, vz, GM):
    """r_p, t_p (inf if there is no future periapsis), e and a flag (0 normal, 1 circular, 2 parabolic) of the two-body orbit through
    (r, v) with gravitational parameter GM. r, v relative to the body, velocity inertial."""
    rm = max(math.sqrt(rx * rx + ry * ry + rz * rz), 1e-12)     # a state exactly on a body is outside the data; do not divide by zero
    v2 = vx * vx + vy * vy + vz * vz
    hx = ry * vz - rz * vy
    hy = rz * vx - rx * vz
    hz = rx * vy - ry * vx
    h2 = hx * hx + hy * hy + hz * hz
    rv = rx * vx + ry * vy + rz * vz
    # e = (v x h) / GM - r / |r|
    ex = (vy * hz - vz * hy) / GM - rx / rm
    ey = (vz * hx - vx * hz) / GM - ry / rm
    ez = (vx * hy - vy * hx) / GM - rz / rm
    e = math.sqrt(ex * ex + ey * ey + ez * ez)
    eps = 0.5 * v2 - GM / rm
    if e < E_CIRCULAR:
        return rm, 0.0, e, 1
    rp = h2 / (GM * (1.0 + e))
    if abs(1.0 - e) < E_PARABOLIC:
        p = h2 / GM
        cosnu = p / rm - 1.0
        cosnu = min(1.0, max(-1.0, cosnu))
        nu = math.acos(cosnu)
        if rv < 0.0:
            nu = -nu
        D = math.tan(0.5 * nu)
        t_from_peri = 0.5 * math.sqrt(p * p * p / GM) * (D + D * D * D / 3.0)
        tp = -t_from_peri if rv < 0.0 else np.inf
        return rp, tp, e, 2
    if eps < 0.0:                                   # elliptic
        a = -GM / (2.0 * eps)
        n = math.sqrt(GM / (a * a * a))
        sinE = rv / (e * math.sqrt(GM * a))
        cosE = (1.0 - rm / a) / e
        E = math.atan2(sinE, cosE)
        M = E - e * math.sin(E)
        tp = -M / n if M < 0.0 else (2.0 * math.pi - M) / n
        return rp, tp, e, 0
    a = GM / (2.0 * eps)                            # hyperbolic
    n = math.sqrt(GM / (a * a * a))
    sinhH = rv / (e * math.sqrt(GM * a))
    H = math.asinh(sinhH)
    M = e * sinhH - H
    tp = -M / n if rv < 0.0 else np.inf
    return rp, tp, e, 0


@njit(cache=True)
def osc_features_numba(states, T, mu):
    n = states.shape[0]
    out = np.empty((n, 13))
    flags = np.zeros((n, 2))
    GM1 = 1.0 - mu
    GM2 = mu
    rH = (mu / 3.0) ** (1.0 / 3.0)
    for i in range(n):
        x, y, z, vx, vy, vz = states[i, 0], states[i, 1], states[i, 2], states[i, 3], states[i, 4], states[i, 5]
        Ti = T[i]
        tau_hat = np.inf
        for b in range(2):
            px = -mu if b == 0 else 1.0 - mu
            GM = GM1 if b == 0 else GM2
            rx, ry, rz = x - px, y, z
            ivx, ivy, ivz = vx - ry, vy + rx, vz
            rm = max(math.sqrt(rx * rx + ry * ry + rz * rz), 1e-12)
            rp, tp, e, fl = _periapsis_two_body(rx, ry, rz, ivx, ivy, ivz, GM)
            rp = max(rp, RP_FLOOR)
            taup = math.sqrt(rp * rp * rp / GM)
            reach = 1.0 if tp <= Ti else 0.0
            tpn = min(tp / Ti, 2.0)
            logrel = math.log10(rm / rH) if b == 1 else math.log10(rm)
            o = b * 6
            out[i, o + 0] = math.log10(rp)
            out[i, o + 1] = math.log10(taup)
            out[i, o + 2] = tpn
            out[i, o + 3] = reach
            out[i, o + 4] = logrel
            out[i, o + 5] = e
            flags[i, b] = fl
            tloc = taup if reach > 0.5 else math.sqrt(rm * rm * rm / GM)
            if tloc < tau_hat:
                tau_hat = tloc
        out[i, 12] = math.log10(tau_hat)
    return out, flags


def osc_features(states, T, mu=MU):
    """states (n, 6) rotating-frame, T (n,) horizons -> (features (n, 13) in OSC_FEATURE_COLS order, flags (n, 2))."""
    return osc_features_numba(np.ascontiguousarray(states, dtype=np.float64), np.ascontiguousarray(T, dtype=np.float64), float(mu))


def add_osc_features(df, mu=MU):
    """Return a copy of df with the OSC_FEATURE_COLS and OSC_FLAG_COLS columns (needs x..vz and sim_time)."""
    f, fl = osc_features(df[["x", "y", "z", "vx", "vy", "vz"]].to_numpy(dtype=float), df["sim_time"].to_numpy(dtype=float), mu)
    out = df.copy()
    new = pd.DataFrame(np.hstack([f, fl]), columns=OSC_FEATURE_COLS + OSC_FLAG_COLS, index=df.index)
    for c in new.columns:
        out[c] = new[c]
    return out
