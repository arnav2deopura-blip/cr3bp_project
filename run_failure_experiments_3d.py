"""
Failure-study experiments that read a `policy` run folder (run_failure_study_3d.py policy):

  e4   learning curves per cell (data-starved or at a floor), reliability of the quantile model, kNN out-of-distribution score
  e5   calibration alternatives at equal certified risk: global, Mondrian by regime, by predicted dispersion, abstention
  e9   step-over analysis of violations (error / tol against dt / tau_min) and a cheap runtime guard
  e10  physics-rule baseline, break-even inference cost per cell

All of these are exploratory on D_explore unless a pre-registration says otherwise.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orbit_ml_cr3bp.config import MU, POSITION_TOL_KM  # noqa: E402
from orbit_ml_cr3bp import failure_study_3d as fs  # noqa: E402
from orbit_ml_cr3bp import experiments_b_3d as xb  # noqa: E402
from orbit_ml_cr3bp import policy_b_3d as pb  # noqa: E402
from orbit_ml_cr3bp import protocol_3d as proto  # noqa: E402
from run_failure_study_3d import make_run_dir, save_json, load_partb  # noqa: E402


def load_policy_dir(a):
    d, meta = load_partb(a.data)
    U = fs.add_derived(pb.usable_rows(d).reset_index(drop=True))
    pdir = a.policy_dir
    led = pd.read_csv(os.path.join(pdir, "ledger.csv"), low_memory=False)
    led["cell"] = fs.assign_cells(led)
    train_ids = set(pd.read_csv(os.path.join(pdir, "train_row_ids.csv"))["row_id"])
    cal_ids = set(pd.read_csv(os.path.join(pdir, "cal_rows.csv"), low_memory=False)["row_id"])
    # row_id in the policy run is the position in the USABLE frame U (reset_index) -- see cmd_policy: U is rebuilt there and
    # row_id re-assigned inside evaluate_orbits from df["row_id"], which load_partb set on the FULL frame. Use that same id.
    train = U[U["row_id"].isin(train_ids)].reset_index(drop=True)
    cal = U[U["row_id"].isin(cal_ids)].reset_index(drop=True)
    rep = json.load(open(os.path.join(pdir, "policy_report.json")))
    return U, train, cal, led, rep


def cmd_e4(a):
    out = make_run_dir(a.out_dir, "e4", a)
    U, train, cal, led, rep = load_policy_dir(a)
    print(f"train {len(train)}, cal {len(cal)}, eval {len(led)}", flush=True)
    sizes = [s for s in (1000, 2000, 5000) if s < len(train)] + [len(train)]
    t0 = time.time()
    lc = xb.learning_curve(train, led, "rk4", sizes, range(a.seeds), q=a.quantile)
    lc.to_csv(os.path.join(out, "learning_curve_rk4.csv"), index=False)
    print(f"learning curves {time.time() - t0:.0f}s", flush=True)
    groups = ["ALL"] + sorted(led["cell"].unique())
    slopes = [xb.learning_slope(lc, g) for g in groups]
    pd.DataFrame(slopes).to_csv(os.path.join(out, "learning_slopes.csv"), index=False)
    rel = xb.reliability(train, led, "rk4", seed=a.seed)
    rel.to_csv(os.path.join(out, "reliability_rk4.csv"), index=False)
    dist, thr, flag = xb.knn_ood(train, cal, led)
    ood = pd.DataFrame({"row_id": led["row_id"], "ood_dist": dist, "ood_flag": flag})
    ood.to_csv(os.path.join(out, "ood.csv"), index=False)
    v = led["violation"].to_numpy()
    rep_out = {"n_train": len(train), "sizes": sizes, "slopes_all": slopes[0],
               "cells_still_improving": [s["group"] for s in slopes[1:] if s.get("slope_ci95") and s["slope_ci95"][1] < 0],
               "cells_flat": [s["group"] for s in slopes[1:] if s.get("slope_ci95") and s["slope_ci95"][0] <= 0 <= s["slope_ci95"][1]],
               "ood": {"threshold": thr, "flag_rate": float(flag.mean()), "violation_rate_flagged": float(v[flag].mean()) if flag.any() else None,
                       "violation_rate_unflagged": float(v[~flag].mean()), "auc_dist_for_violation": xb.auc_score(dist[v], dist[~v]),
                       "share_of_violations_flagged": float(flag[v].mean()) if v.any() else None},
               "reliability_all": rel[rel["group"] == "ALL"][["q", "n", "empirical"]].to_dict(orient="records")}
    save_json(rep_out, os.path.join(out, "e4_report.json"))
    print(json.dumps(proto.json_safe(rep_out), indent=1))
    print("wrote", out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--data", required=True)
        sp.add_argument("--policy_dir", required=True)
        sp.add_argument("--out_dir", default="out_b")
        sp.add_argument("--seed", type=int, default=42)
        sp.add_argument("--tol_km", type=float, default=POSITION_TOL_KM)
        sp.add_argument("--n_jobs", type=int, default=8)

    sp = sub.add_parser("e4")
    common(sp)
    sp.add_argument("--seeds", type=int, default=5)
    sp.add_argument("--quantile", type=float, default=0.1)
    sp.set_defaults(fn=cmd_e4)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
