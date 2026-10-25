"""Build a horizon-sweep dataset (same initial states relabelled at several horizons).
  python make_sweep_b.py --source data_b/<csv> --out data_b/sweeps/<name>.csv --policy_dir out_b/policy_<ts> [--seed 0] [--n_ics 2600]
--policy_dir restricts the pool to rows that policy run held out (calibration + evaluation rows), so a frozen model never saw them."""
import argparse
import time

import pandas as pd

from orbit_ml_cr3bp.sweeps_b_3d import horizon_sweep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy_dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_ics", type=int, default=2600)
    ap.add_argument("--n_jobs", type=int, default=8)
    a = ap.parse_args()
    ids = None
    if a.policy_dir:
        ids = list(pd.read_csv(a.policy_dir + "/cal_rows.csv", low_memory=False)["row_id"]) + \
            list(pd.read_csv(a.policy_dir + "/ledger.csv", low_memory=False)["row_id"])
    t0 = time.time()
    df = horizon_sweep(a.source, a.out, n_ics=a.n_ics, seed=a.seed, pool_row_ids=ids, n_jobs=a.n_jobs)
    print("done", round(time.time() - t0), "s;", len(df), "rows;", df["orbit_idx"].nunique(), "initial states;",
          df.groupby("sim_time").size().to_dict())


if __name__ == "__main__":
    main()
