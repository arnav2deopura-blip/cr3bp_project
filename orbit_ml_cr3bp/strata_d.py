"""
Conditional safety near the Moon (Part D, D4). Part C certified ONE scale on average over orbits; the realized risk of the Sundman policy was 4.4 %
overall but 13.5 % where the trajectory later passed within 0.05 DU of the Moon. Here the scale is certified per STRATUM, the strata being defined from
quantities that are free before any integration (the two-body periapsis about the Moon and the free crash flag, the Hill regime, the horizon band), and a
stratum that cannot be certified abstains to the adaptive solver (SciPy RK45 at the certified rtol).

  base_cells / build_strata_map / assign_strata   the pre-specified 4 x 3 x 3 cells and Part B's fixed merge rule applied to the CALIBRATION counts
  certify_strata                                  fixed-sequence certificate per stratum at level delta / G (G = number of strata)
  deploy_strata                                   run the policy at the stratum scales, abstained rows through the adaptive fallback
  stratum_table                                   per stratum: violation rate with Clopper-Pearson interval, cost, shares, for the policy and the comparators
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binom

from .audit_3d import clopper_pearson
from .config import MU, POSITION_TOL_KM
from .doomed_3d import crash_flag_physical
from . import failure_study_3d as fs
from . import policy_c_3d as pc
from . import deploy_d as dd

MOON_EDGES = (0.05, 0.17)                        # DU, Part C's closest-approach bands (r2 < 0.05, 0.05-0.17)
MOON_CLASSES = ("crash", "peri<0.05", "peri0.05-0.17", "far")
REGIME_GROUPS = {"I": "I", "II": "II-III", "III": "II-III", "IV": "IV-V", "V": "IV-V"}
MIN_N_CELL = 400                                 # a base cell with fewer calibration rows is merged (fs.assign_cells rule)
MIN_N_CERT = 150                                 # a stratum with fewer calibration rows after merging abstains without a test
MIN_N_REPORT = 300                               # evaluation rows a stratum needs to enter the D4 hypothesis
UPPER_LEVEL = 0.07                               # the pre-set level of the hypothesis (CP upper bound of every reported stratum)


def moon_class(df, mu=MU):
    """Free closeness class: 'crash' = the physical crash flag (osculating periapsis inside a collision radius, reached before T), otherwise the
    osculating periapsis about the Moon, if reached before T, in the bands < 0.05 DU, 0.05-0.17 DU; 'far' = no close predicted passage."""
    _f1, _f2, anyf = crash_flag_physical(df, df["sim_time"], mu)
    rp2 = 10.0 ** df["osc_logrp_2"].to_numpy(dtype=float)
    reach2 = df["osc_reach_2"].to_numpy() == 1.0
    cls = np.where(anyf, MOON_CLASSES[0], np.where(reach2 & (rp2 < MOON_EDGES[0]), MOON_CLASSES[1],
                                                   np.where(reach2 & (rp2 < MOON_EDGES[1]), MOON_CLASSES[2], MOON_CLASSES[3])))
    return pd.Series(cls, index=df.index, dtype=object)


def base_cells(df, mu=MU):
    """DataFrame with the three cell coordinates and the base label 'regime-group|moon-class|horizon-band' (36 cells)."""
    d = pd.DataFrame(index=df.index)
    d["rg"] = df["regime"].map(REGIME_GROUPS).astype(object)
    d["mc"] = moon_class(df, mu)
    d["hb"] = df["h_band"].astype(object)
    d["base"] = d["rg"] + "|" + d["mc"] + "|" + d["hb"]
    return d


def build_strata_map(cal_cells, min_n=MIN_N_CELL):
    """{base label: stratum label} from the calibration rows, by the merge rule of failure_study_3d.assign_cells (along the horizon within
    (regime group, moon class); then along the moon class within the regime group; anything still small is 'pooled-small')."""
    lab = fs.assign_cells(cal_cells, min_n=min_n, regime_col="rg", r2_col="mc", h_col="hb")
    return dict(zip(cal_cells["base"], lab)), set(lab)


def assign_strata(cells, smap, labels):
    """Stratum label of every row; a base cell not seen in calibration falls back along the same merge chain."""
    out = []
    for b, rg, mc in zip(cells["base"], cells["rg"], cells["mc"]):
        if b in smap:
            out.append(smap[b])
        elif f"{rg}|{mc}|h*" in labels:
            out.append(f"{rg}|{mc}|h*")
        elif f"{rg}|r2*|h*" in labels:
            out.append(f"{rg}|r2*|h*")
        else:
            out.append("pooled-small")
    return np.array(out, dtype=object)


def certify_strata(runner, states, T, raw_ds, ref_pos, labels, tol_km=POSITION_TOL_KM, alpha=dd.ALPHA, delta=dd.DELTA, grid=pc.FIXED_SEQ_GRID,
                   min_n=MIN_N_CERT):
    """One fixed-sequence certificate per stratum, each at level delta / G (Bonferroni over the G strata; the order inside a stratum is the fixed
    ascending scale grid). A stratum with fewer than min_n calibration rows, or with no certified scale, abstains (scale None)."""
    groups = sorted(set(labels))
    G = len(groups)
    cache = {}

    def bad_at(s):
        if s not in cache:
            r = dd.run_rows(runner, states, T, raw_ds * s, ref_pos, tol_km)
            cache[s] = ~r["met"]
        return cache[s]

    out = {}
    for g in groups:
        idx = np.where(labels == g)[0]
        if len(idx) < min_n:
            out[g] = {"scale": None, "n_cal": int(len(idx)), "reason": f"fewer than {min_n} calibration rows", "chain": []}
            continue
        rep = pc.certify_sequence(lambda s, idx=idx: (int(bad_at(s)[idx].sum()), len(idx)), grid, alpha, delta / G, "fixed_sequence")
        out[g] = {"scale": rep["chosen_scale"], "n_cal": int(len(idx)), "reason": None if rep["chosen_scale"] is not None else "no scale certified",
                  "chain": rep["chain"]}
    return out, G


def stratum_table(led, group_col, comps=("RK45_eqr", "DOP853_eqr"), min_n=0):
    """Per-group table of the policy ledger `led` (needs meets_tol, C_ml_tot, <comp>_nfev, <comp>_ok, abstain flag column 'ml_abstain'):
    n, policy violations with the 95 % Clopper-Pearson interval, median total cost, abstention share, and for every comparator its violation rate (with
    interval) and the share of orbits where the policy met the tolerance AND cost less."""
    rows = []
    for g, d in led.groupby(group_col):
        n = len(d)
        k = int((~d["meets_tol"].astype(bool)).sum())
        lo, hi = clopper_pearson(k, n)
        r = {"group": g, "n": n, "violations": k, "rate": k / n, "cp_lo": lo, "cp_hi": hi, "abstain_share": float(d["ml_abstain"].mean()),
             "median_cost_met": float(d.loc[d["meets_tol"].astype(bool), "C_ml_tot"].median()) if d["meets_tol"].any() else float("nan")}
        for c in comps:
            ok = d[f"{c}_ok"].astype(bool)
            kc = int((~ok).sum())
            clo, chi = clopper_pearson(kc, n)
            has = d[f"{c}_nfev"].notna()
            r[f"{c}_rate"], r[f"{c}_cp_lo"], r[f"{c}_cp_hi"] = kc / n, clo, chi
            r[f"{c}_median_cost_met"] = float(d.loc[ok, f"{c}_nfev"].median()) if ok.any() else float("nan")
            r[f"share_cheaper_{c}"] = float((d["meets_tol"].astype(bool) & has & (d["C_ml_tot"] < d[f"{c}_nfev"])).sum() / max(has.sum(), 1))
        r["p_rate_ge_level"] = float(binom.cdf(k, n, UPPER_LEVEL))
        rows.append(r)
    return pd.DataFrame(rows)
