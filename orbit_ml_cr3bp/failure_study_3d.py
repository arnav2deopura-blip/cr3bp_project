"""
Failure-study analysis tools (Part B, plan sections 5.1-5.5): pre-specified cells, per-orbit cost-gap decomposition,
failure taxonomy and risk ledger. Pure functions on DataFrames; no integration happens here.

Cost-gap identity (exact in log space, up to the ceil rounding which is reported as its own residual R):

  log10(C_ML,tot / C_ad) = P + S + Q + K + O + R
    P  log10(C_fix* / C_ad)                                    paradigm: best possible single step vs adaptive
    S  log10[(e_k / d*_k) / (e_k* / d*_k*)]                    selection: picked the wrong integrator (>= 0)
    Q  log10(d*_k / d_hat_k)                                   prediction: > 0 under-predicted (costly), < 0 over-predicted (danger)
    K  log10(1 / s_k)                                          safety-scale tax
    O  log10(1 + (C_feat + C_pred) / C_ML,int)                 overhead
  with C_ML,int = e_k ceil(T / (s d_hat)), C_fix* = e_k* ceil(T / d*_k*), d* the oracle label of the named integrator.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import MU
from .diagnostics_3d import hill_regime
from .physics_3d import EVALS_PER_STEP_3D

TOL_KM = 100.0
R2_EDGES = (0.0, 0.05, 0.17, 0.8, np.inf)
R2_LABELS = ("<0.05", "0.05-0.17", "0.17-0.8", ">0.8")
H_EDGES = (0.0, 3.0, 8.0, np.inf)
H_LABELS = ("1-3", "3-8", "8-20")
MIN_CELL_N = 50
REGIMES = ("I", "II", "III", "IV", "V")
CATASTROPHIC = 10.0          # error / tol above which a violation is "catastrophic"
REF_FLAG_FRAC = 0.1          # F0-ref when the reference check differs by more than 0.1 * tol
COMPARATORS = ("RK45_spec", "DOP853_spec", "RK45_cal", "DOP853_cal", "RK45_eqr", "DOP853_eqr")


# ---------------------------------------------------------------------------------------------------------------------
# derived columns and the pre-specified cells
# ---------------------------------------------------------------------------------------------------------------------

def add_derived(df, mu=MU):
    d = df.copy()
    d["regime"] = hill_regime(d["jacobi_3d"].to_numpy(dtype=float), mu)
    d["r2_band"] = pd.cut(d["min_r2"], bins=list(R2_EDGES), labels=list(R2_LABELS), right=False).astype(object)
    d["h_band"] = pd.cut(d["sim_time"], bins=list(H_EDGES), labels=list(H_LABELS), right=False).astype(object)
    d["lambda_T"] = d["ftle_3d"] * d["sim_time"] * d.get("ftle_t_frac", 1.0)
    d["log_delta_phys"] = np.log10(d["delta_phys"])
    d["log_delta_emp"] = np.log10(d["delta_emp"])
    d["log_min_r2"] = np.log10(d["min_r2"])
    return d


def assign_cells(d, min_n=MIN_CELL_N, regime_col="regime", r2_col="r2_band", h_col="h_band"):
    """Pre-specified 5 x 4 x 3 = 60 cells, merged by a rule fixed in advance (plan 5.5): a base cell with fewer than `min_n`
    orbits joins, first, the other small cells of the same (regime, r2 band) along the horizon axis (label 'regime|r2|h*');
    if that union is still smaller than `min_n` it joins the small remainder of the same regime along the r2 axis
    ('regime|r2*|h*'); anything still smaller than `min_n` is labelled 'pooled-small'. Counts are those of the set passed in,
    the rule is not. Returns a Series of labels aligned with d."""
    lab = pd.Series(index=d.index, dtype=object)
    base = d[regime_col].astype(str) + "|" + d[r2_col].astype(str) + "|" + d[h_col].astype(str)
    n_base = base.map(base.value_counts())
    big = n_base >= min_n
    lab[big] = base[big]
    rest = d.index[~big]
    # stage 1: merge along horizon within (regime, r2 band)
    grp1 = (d.loc[rest, regime_col].astype(str) + "|" + d.loc[rest, r2_col].astype(str) + "|h*")
    n1 = grp1.map(grp1.value_counts())
    ok1 = n1 >= min_n
    lab[grp1.index[ok1]] = grp1[ok1]
    rest2 = grp1.index[~ok1]
    # stage 2: merge along r2 within regime
    grp2 = d.loc[rest2, regime_col].astype(str) + "|r2*|h*"
    n2 = grp2.map(grp2.value_counts())
    ok2 = n2 >= min_n
    lab[grp2.index[ok2]] = grp2[ok2]
    lab[grp2.index[~ok2]] = "pooled-small"
    return lab


# ---------------------------------------------------------------------------------------------------------------------
# oracle fixed step
# ---------------------------------------------------------------------------------------------------------------------

def oracle_fixed(d, candidates=("rk4", "leapfrog")):
    """C_fix* = min over candidate integrators with a valid label of e_k ceil(T / d*_k). Adds fix_cost_<k>, k_star, C_fix_star,
    dstar_<k>, cens_<k>."""
    out = pd.DataFrame(index=d.index)
    T = d["sim_time"].to_numpy(dtype=float)
    costs = {}
    for k in candidates:
        ok = d[f"{k}_success"].to_numpy() == 1
        dstar = np.where(ok, d[f"dt_{k}"].to_numpy(dtype=float), np.nan)
        out[f"dstar_{k}"] = dstar
        out[f"cens_{k}"] = d[f"{k}_censored"].fillna(False).astype(bool).to_numpy()
        costs[k] = EVALS_PER_STEP_3D[k] * np.ceil(T / dstar - 1e-9)
        out[f"fix_cost_{k}"] = costs[k]
    arr = np.vstack([costs[k] for k in candidates])
    arr_f = np.where(np.isfinite(arr), arr, np.inf)
    idx = np.argmin(arr_f, axis=0)
    best = arr_f[idx, np.arange(arr_f.shape[1])]
    out["C_fix_star"] = np.where(np.isfinite(best), best, np.nan)
    out["k_star"] = np.where(np.isfinite(best), np.array(candidates, dtype=object)[idx], None)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# ledger and decomposition
# ---------------------------------------------------------------------------------------------------------------------

def build_ledger(d, results, c_pred, c_feat=1.0, tol_km=TOL_KM, candidates=("rk4", "leapfrog")):
    """d: usable rows with labels, diagnostics and derived columns (must contain row_id). results: policy_b_3d.evaluate_orbits
    output (one row per row_id). c_pred: measured model-inference cost per orbit in RHS-equivalents."""
    led = d.merge(results, on="row_id", how="inner", suffixes=("", "_res"))
    fx = oracle_fixed(led, candidates)
    led = pd.concat([led, fx], axis=1)
    led["has_choice"] = led["ml_integrator"].notna()
    led["ml_ok"] = led["ml_ok"].fillna(False).astype(bool)
    led["meets_tol"] = led["has_choice"] & led["ml_ok"] & (led["ml_err_km"] <= tol_km)
    led["violation"] = led["has_choice"] & ~led["meets_tol"]
    led["abstain"] = ~led["has_choice"]
    led["err_over_tol"] = led["ml_err_km"] / tol_km
    led["C_feat"] = float(c_feat)
    led["C_pred"] = float(c_pred)
    led["C_ml_int"] = led["ml_evals"]
    led["C_ml_tot"] = led["ml_evals"] + led["C_feat"] + led["C_pred"]
    # label of the integrator the policy picked
    k = led["ml_integrator"]
    led["dstar_chosen"] = np.where(k == "rk4", led["dstar_rk4"], np.where(k == "leapfrog", led["dstar_leapfrog"], np.nan))
    led["cens_chosen"] = np.where(k == "rk4", led["cens_rk4"], np.where(k == "leapfrog", led["cens_leapfrog"], False))
    led["e_chosen"] = k.map(EVALS_PER_STEP_3D)
    led["over_pred"] = np.log10(led["ml_dt"] / led["dstar_chosen"])      # r: > 0 means the scaled step exceeds the label
    led["raw_over_label"] = np.log10(led["ml_raw_dt"] / led["dstar_chosen"])
    # step-over ratio (HB7): dt used against the shortest local timescale seen on the trajectory
    led["step_over"] = led["ml_dt"] / led["tau_min"]
    return led


def decompose(led, comparator, tol_km=TOL_KM):
    """Rows that met the tolerance and have every ingredient get P, S, Q, K, O, R, total (all in dex) and the dominant ceiling.
    Returns a DataFrame indexed like led (NaN elsewhere)."""
    col = f"{comparator}_nfev"
    ok = (led["meets_tol"] & led[col].notna() & led["C_fix_star"].notna() & led["dstar_chosen"].notna()
          & (led["ml_raw_dt"] > 0))
    out = pd.DataFrame(index=led.index, columns=["P", "S", "Q", "K", "O", "R", "total", "dominant", "paradigm_bound"],
                       dtype=object)
    L = led[ok]
    T = L["sim_time"]
    e_k = L["e_chosen"]
    ek_star = L["k_star"].map(EVALS_PER_STEP_3D)
    dstar_kstar = np.where(L["k_star"] == "rk4", L["dstar_rk4"], L["dstar_leapfrog"])
    c_ad = L[col]
    P = np.log10(L["C_fix_star"] / c_ad)
    S = np.log10((e_k / L["dstar_chosen"]) / (ek_star / dstar_kstar))
    Q = np.log10(L["dstar_chosen"] / L["ml_raw_dt"])
    K = np.log10(1.0 / L["ml_scale"])
    O = np.log10(1.0 + (L["C_feat"] + L["C_pred"]) / L["C_ml_int"])
    total = np.log10(L["C_ml_tot"] / c_ad)
    R = total - (P + S + Q + K + O)
    terms = pd.DataFrame({"P": P, "S": S, "Q": Q, "K": K, "O": O})
    dom = terms.idxmax(axis=1).where(terms.max(axis=1) > 0, "none")
    for name, v in (("P", P), ("S", S), ("Q", Q), ("K", K), ("O", O), ("R", R), ("total", total)):
        out.loc[L.index, name] = v.astype(float)
    out.loc[L.index, "dominant"] = dom
    out.loc[L.index, "paradigm_bound"] = (P > 0)
    for c in ("P", "S", "Q", "K", "O", "R", "total"):
        out[c] = out[c].astype(float)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# taxonomy (plan 5.3)
# ---------------------------------------------------------------------------------------------------------------------

def classify(led, comparator, tol_km=TOL_KM, sigma_eps=None, ood_flag=None):
    """Mutually exclusive class per orbit, applied in the plan's order:
       F0-ref, F0-label, abstain, violation:{information|epistemic|calibration|unattributed}, cost-loss:{P,S,Q,K,O,none}, ml-win,
       no-comparator (no adaptive cost for this orbit).
    sigma_eps: Series (aligned with led) of the IC-perturbation label spread in dex (E3a); ood_flag: boolean Series (E4).
    Without them a violation is 'violation:unattributed'. (F1-doomed orbits are not in the usable ledger by construction.)"""
    dec = decompose(led, comparator, tol_km)
    cls = pd.Series("unclassified", index=led.index, dtype=object)
    col = f"{comparator}_nfev"
    f0ref = led["ref_check_km"] > REF_FLAG_FRAC * tol_km
    f0lab = led["cens_chosen"].astype(bool)
    abst = led["abstain"]
    viol = led["violation"]
    ok = led["meets_tol"]
    cls[ok] = "ml-win"
    cls[ok & led[col].isna()] = "no-comparator"
    loss = ok & led[col].notna() & (led["C_ml_tot"] > led[col])
    cls[loss] = "cost-loss:" + dec.loc[loss, "dominant"].astype(str)
    cls[viol] = "violation:unattributed"
    if sigma_eps is not None:
        r = led["over_pred"]
        info = viol & (sigma_eps >= r)
        cls[info] = "violation:information"
        if ood_flag is not None:
            epi = viol & ~info & ood_flag.fillna(False)
            cls[epi] = "violation:epistemic"
            cls[viol & ~info & ~ood_flag.fillna(False)] = "violation:calibration"
        else:
            cls[viol & ~info] = "violation:unattributed"
    cls[abst] = "abstain"
    cls[f0lab & ~f0ref] = "F0-label"
    cls[f0ref] = "F0-ref"
    return cls, dec


# ---------------------------------------------------------------------------------------------------------------------
# risk ledger (plan 5.4) and statistics helpers
# ---------------------------------------------------------------------------------------------------------------------

def risk_ledger(led, tol_km=TOL_KM):
    """Violation summary: rate, CVaR90 of error/tol among violations (diverged runs count as +inf only in n_diverged),
    catastrophic share (> 10 tol), and the per-violation covariates the plan asks for."""
    v = led[led["violation"]]
    n = int(led["has_choice"].sum())
    e = (v["ml_err_km"] / tol_km).dropna()
    out = {"n_policy_runs": n, "n_violations": int(len(v)), "violation_rate": len(v) / n if n else float("nan"),
           "n_diverged": int((v["ml_ok"] == False).sum()),  # noqa: E712
           "abstain_rate": float(led["abstain"].mean())}
    if len(e):
        q90 = float(np.quantile(e, 0.9))
        out.update({"median_err_over_tol": float(e.median()), "p90_err_over_tol": q90,
                    "cvar90_err_over_tol": float(e[e >= q90].mean()), "max_err_over_tol": float(e.max()),
                    "catastrophic_share": float((e > CATASTROPHIC).mean())})
    return out


def bootstrap_ci(values, stat=np.median, n_boot=2000, seed=0, alpha=0.05):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return (float("nan"),) * 3
    rng = np.random.RandomState(seed)
    boots = np.array([stat(v[rng.randint(0, len(v), len(v))]) for _ in range(n_boot)])
    return float(stat(v)), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def spearman_bootstrap(x, y, n_boot=2000, seed=0):
    from scipy.stats import spearmanr
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 5:
        return float("nan"), float("nan"), float("nan"), int(len(x))
    rho = float(spearmanr(x, y).statistic)
    rng = np.random.RandomState(seed)
    bs = []
    for _ in range(n_boot):
        i = rng.randint(0, len(x), len(x))
        bs.append(spearmanr(x[i], y[i]).statistic)
    return rho, float(np.nanquantile(bs, 0.025)), float(np.nanquantile(bs, 0.975)), int(len(x))
