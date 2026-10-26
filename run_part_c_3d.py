"""
Part C commands (see part_c/PART_C_LOG.md for the decisions behind each one). Every command writes into its own timestamped
folder under --out_dir with a manifest.json, as in Part B.

  lever          C0: lever surface (share of orbits where the ML policy would beat each adaptive comparator for a given overhead O
                 and reduction delta of its integration cost) and the 50 % iso-lines, from a policy ledger
  e3c_paired     Track 0.1 / 0.2: paired (McNemar) and risk-matched re-analysis of E3c, f = 0 against f = 0.05

Run with the project's conda env:  /Users/arnavdeopura/miniforge3/envs/orbit-ml/bin/python run_part_c_3d.py <command> ...
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orbit_ml_cr3bp.config import MU, POSITION_TOL_KM  # noqa: E402
from orbit_ml_cr3bp import failure_study_3d as fs  # noqa: E402
from orbit_ml_cr3bp import policy_b_3d as pb  # noqa: E402
from run_failure_study_3d import make_run_dir, save_json  # noqa: E402


def read_ledger(path):
    led = pd.read_csv(path, low_memory=False)
    for c in ("meets_tol", "violation", "has_choice", "abstain", "ml_ok"):
        if c in led.columns:
            led[c] = led[c].astype(bool)
    return led


# ---------------------------------------------------------------------------------------------------------------------
# C0: lever surface
# ---------------------------------------------------------------------------------------------------------------------

def cmd_lever(a):
    out = make_run_dir(a.out_dir, "lever", a)
    led = read_ledger(a.ledger)
    if "regime" not in led.columns or "h_band" not in led.columns:
        led = fs.add_derived(led)
    led["deploy_cell"] = led["regime"].astype(str) + "|" + led["h_band"].astype(str)
    surf = fs.lever_surface(led)
    surf.to_csv(os.path.join(out, "lever_surface.csv"), index=False)
    cells = fs.lever_surface(led, by="deploy_cell", O_grid=(0.0, 10.0, 50.0), delta_grid=(0.0, 0.3, 0.6))
    cells.to_csv(os.path.join(out, "lever_cells.csv"), index=False)
    iso = fs.lever_isoline(led)
    iso.to_csv(os.path.join(out, "lever_isoline.csv"), index=False)
    # oracle fixed step (delta = everything, O = 0 on the paradigm side): C_fix_star against each comparator
    orc = {}
    for comp in fs.COMPARATORS:
        col = f"{comp}_nfev"
        if col not in led.columns:
            continue
        m = led[col].notna() & led["C_fix_star"].notna()
        orc[comp] = {"n": int(m.sum()), "share_oracle_fixed_cheaper": float((led.loc[m, "C_fix_star"] < led.loc[m, col]).mean())}
    # the table of the plan's section 2.1, wide
    wide = surf[surf["group"] == "ALL"].pivot_table(index="comparator", columns=["O", "delta_dex"], values="share")
    save_json({"ledger": a.ledger, "n_orbits": int(len(led)), "n_met_tolerance": int(led["meets_tol"].sum()),
               "oracle_fixed": orc, "isoline": iso.to_dict(orient="records")}, os.path.join(out, "lever_report.json"))
    pd.set_option("display.width", 200)
    sel = [(0.0, 0.0), (0.0, 0.3), (0.0, 0.6), (10.0, 0.6), (50.0, 0.6), (200.0, 0.6)]
    print((100 * wide[[c for c in sel if c in wide.columns]]).round(1).to_string())
    print("\nDelta (dex) needed for a 50 % share, by overhead O:")
    print(iso.pivot_table(index="comparator", columns="O", values="delta_needed_dex").round(2).to_string())
    print("wrote", out)


# ---------------------------------------------------------------------------------------------------------------------
# Track 0.1 / 0.2: E3c re-analysed pairwise and at matched risk
# ---------------------------------------------------------------------------------------------------------------------

def _eval_rk4(reg_feats, reg, ev, cal_cache_pred, scale, comps, tol_km, n_jobs, c_extra):
    """Run the RK4-only policy on `ev` at one scale; returns per-orbit frame with meets, c_tot (no model cost) and comparator costs."""
    def predict(df):
        dt = 10.0 ** reg.predict(df[reg_feats].to_numpy(dtype=float))
        n = len(df)
        return {"rk4": {"feasible": np.ones(n, dtype=bool), "dt": dt},
                "leapfrog": {"feasible": np.zeros(n, dtype=bool), "dt": np.full(n, np.nan)}}
    ev_pred = predict(ev)
    res = pb.evaluate_orbits(ev, ev_pred, {"rk4": scale, "leapfrog": None}, rtols={}, do_spec=False, tol_km=tol_km,
                             n_jobs=n_jobs, candidates=("rk4",), progress=False)
    m = ev[["row_id"] + [f"{c}_nfev" for c in comps]].rename(columns={f"{c}_nfev": c for c in comps}).merge(res, on="row_id", how="left")
    m["meets"] = m["ml_ok"].fillna(False).astype(bool) & (m["ml_err_km"] <= tol_km)
    m["c_tot"] = m["ml_evals"] + 1.0 + c_extra
    return m


def mcnemar_exact(b, c):
    """Two-sided exact binomial test on the discordant pairs (b: only first wins, c: only second wins)."""
    from scipy.stats import binomtest
    n = b + c
    return 1.0 if n == 0 else float(binomtest(b, n, 0.5, alternative="two-sided").pvalue)


def cmd_e3c_paired(a):
    from orbit_ml_cr3bp import lookahead_3d as la
    from orbit_ml_cr3bp import experiments_b_3d as xb
    from orbit_ml_cr3bp.audit_3d import ReferenceCache, clopper_pearson
    from run_failure_experiments_3d import load_policy_dir
    out = make_run_dir(a.out_dir, "e3c_paired", a)
    U, train, cal, led, rep = load_policy_dir(a)
    lk = pd.read_csv(a.lookahead, low_memory=False)
    train, cal, ev = (x.merge(lk, on="row_id", how="left") for x in (train, cal, led))
    comps = ["RK45_cal", "DOP853_cal", "RK45_spec"]
    arms = {0.0: list(xb.FEATS_STATE)}
    for f in a.fractions:
        arms[f] = list(xb.FEATS_STATE) + la.feature_names(f)
    regs = {f: xb.fit_log_quantile(train, "rk4", a.quantile, features=ft, seed=42) for f, ft in arms.items()}
    scales = list(a.scales)
    per, rows = {}, []
    t0 = time.time()
    for f, ft in arms.items():
        c_la = 0.0 if f == 0 else ev[f"la{la.tag(f)}_nfev"].to_numpy(dtype=float)
        for s in scales:
            m = _eval_rk4(ft, regs[f], ev, None, s, comps, a.tol_km, a.n_jobs, c_la)
            per[(f, s)] = m
            n = len(m)
            k = int((~m["meets"]).sum())
            row = {"f": f, "scale": s, "n": n, "violations": k, "violation_rate": k / n, "viol_lo": clopper_pearson(k, n)[0],
                   "viol_hi": clopper_pearson(k, n)[1], "median_c_tot": float(m.loc[m["meets"], "c_tot"].median())}
            for c in comps:
                w = m["meets"] & (m["c_tot"] < m[c])
                row[f"win_{c}"] = float(w.mean())
            rows.append(row)
            print(f"f={f:<5} scale {s:<4} viol {k / n:.4f}  win vs RK45-cal {row['win_RK45_cal']:.4f}  median cost {row['median_c_tot']:.0f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(out, "e3c_scale_grid.csv"), index=False)
    # --- 0.1: paired test at the Part-B certified scales (0.5 for f = 0, 0.6 for f = 0.05)
    paired = {}
    for comp in comps:
        for (s0, s1) in a.paired_scales:
            for f1 in a.fractions:
                m0, m1 = per[(0.0, s0)], per[(f1, s1)]
                w0 = (m0["meets"] & (m0["c_tot"] < m0[comp])).to_numpy()
                w1 = (m1["meets"] & (m1["c_tot"] < m1[comp])).to_numpy()
                b, c = int((w1 & ~w0).sum()), int((w0 & ~w1).sum())
                paired[f"{comp}|f0@{s0}|f{f1}@{s1}"] = {"wins_f0": int(w0.sum()), "wins_f1": int(w1.sum()), "only_f1": b,
                                                          "only_f0": c, "n": int(len(w0)), "diff": float(w1.mean() - w0.mean()),
                                                          "mcnemar_exact_p": mcnemar_exact(b, c)}
    # --- 0.2: equal realized risk. For each arm interpolate (violation rate -> win share, log median cost) along the scale grid.
    matched = {}
    for target in a.risk_targets:
        for f in arms:
            t = tab[tab["f"] == f].sort_values("violation_rate")
            x = t["violation_rate"].to_numpy()
            if not (x.min() <= target <= x.max()):
                matched[f"risk{target}|f{f}"] = None
                continue
            matched[f"risk{target}|f{f}"] = {
                "win_RK45_cal": float(np.interp(target, x, t["win_RK45_cal"].to_numpy())),
                "win_DOP853_cal": float(np.interp(target, x, t["win_DOP853_cal"].to_numpy())),
                "median_c_tot": float(10 ** np.interp(target, x, np.log10(t["median_c_tot"].to_numpy())))}
    save_json({"scales": scales, "paired_mcnemar": paired, "risk_matched": matched, "table": rows}, os.path.join(out, "e3c_paired_report.json"))
    for k, v in paired.items():
        if k.startswith("RK45_cal"):
            print(k, v)
    print("risk matched:")
    for k, v in matched.items():
        print(" ", k, v)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("lever")
    sp.add_argument("--ledger", required=True)
    sp.add_argument("--out_dir", default="out_c")
    sp.set_defaults(fn=cmd_lever)

    sp = sub.add_parser("e3c_paired")
    sp.add_argument("--data", required=True)
    sp.add_argument("--policy_dir", required=True)
    sp.add_argument("--lookahead", required=True)
    sp.add_argument("--out_dir", default="out_c")
    sp.add_argument("--quantile", type=float, default=0.1)
    sp.add_argument("--tol_km", type=float, default=POSITION_TOL_KM)
    sp.add_argument("--n_jobs", type=int, default=8)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--fractions", type=float, nargs="+", default=[0.05])
    sp.add_argument("--scales", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    sp.add_argument("--paired_scales", type=float, nargs=2, action="append", default=[[0.5, 0.6], [0.5, 0.5], [0.6, 0.6]],
                    help="(scale of f = 0, scale of f > 0) pairs for the McNemar test")
    sp.add_argument("--risk_targets", type=float, nargs="+", default=[0.03, 0.035, 0.04])
    sp.set_defaults(fn=cmd_e3c_paired)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
