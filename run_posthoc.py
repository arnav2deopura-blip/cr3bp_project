"""
Post hoc analyses of the sealed Studies A-D (posthoc/README.md). Every command writes a run folder with a manifest under posthoc/results/ and
never touches a sealed file. Numbers produced here are POST HOC and are reported next to the sealed numbers, never in their place.

  aggregates     P1-P3: share, aggregate cost ratio and mean log ratio for every sealed policy ledger and comparator, by horizon band, Hill
                 regime and sampling family, under the declared mixtures, and with one overhead charged on every Table 3 column
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from orbit_ml_cr3bp import posthoc_metrics as pm  # noqa: E402
from orbit_ml_cr3bp.config import MU  # noqa: E402
from run_part_d_3d import DATA, save_json, make_run_dir as _make_run_dir  # noqa: E402

OUT = "posthoc/results"
COMPARATORS = ("RK45_eqr", "RK45_cal", "DOP853_eqr", "DOP853_cal")
META_COLS = ["row_id", "sim_time", "orbit_label", "regime", "h_band", "r2_band", "min_r2"]
OVERHEAD_C, OVERHEAD_D = 15.036794812388587, 16.088492841244694      # pinned Study C value; Study D re-measurement (out_d/overheads)


def _one(pattern):
    f = sorted(glob.glob(os.path.join(ROOT, pattern)))
    if len(f) != 1:
        raise FileNotFoundError(f"{pattern}: {len(f)} matches")
    return f[0]


# name -> (ledger glob, dataset whose usable rows the ledger's row_id refers to (None: the ledger carries its metadata), Table 3 column)
LEDGERS = {
    "C_confirm_sund": ("out_c/confirm/sund/apply_policy_*/ledger.csv", None, True),
    "D_replication_sund": ("out_d/confirm/d3/sund/apply_policy_*/ledger.csv", None, True),
    "C_lock_sund": ("out_c/lock/sund/apply_policy_*/ledger.csv", None, True),
    "D_lock_sund": ("out_d/confirm/lock/lock_*/ledger_sund.csv.gz", "lock", True),
    "D_confirm_D1_sund": ("out_d/confirm/d1/d1_*/ledger_sund.csv.gz", "confirm", False),
    "D_confirm_D1_rule_THR": ("out_d/confirm/d1/d1_*/ledger_rule_THR.csv.gz", "confirm", False),
    "D_replication_D4_strat": ("out_d/confirm/d4/d4_*/ledger_strat.csv.gz", "replication", False),
    "D_replication_D4_onescale": ("out_d/confirm/d4/d4_*/ledger_marginal.csv.gz", "replication", False),
}


def load_ledger(name):
    """A sealed ledger with the metadata columns (joined from its dataset by row_id where the ledger does not carry them)."""
    pattern, dset, _t3 = LEDGERS[name]
    path = _one(pattern)
    led = pd.read_csv(path, low_memory=False)
    if dset is not None:
        import run_part_c_3d as rc
        U, _w = rc.load_c_frame([os.path.join(ROOT, DATA[dset])], MU)
        meta = U[[c for c in META_COLS if c in U.columns]]
        both = [c for c in meta.columns if c in led.columns and c != "row_id"]
        n0 = len(led)
        led = led.merge(meta, on="row_id", how="left", suffixes=("", "_meta"), indicator=True)
        if len(led) != n0 or (led["_merge"] != "both").any():
            raise ValueError(f"{name}: row_id join to {dset} failed")
        led = led.drop(columns="_merge")
        for c in both:                       # the ledger's own copy must agree with the dataset's
            if not (led[c].astype(str) == led[f"{c}_meta"].astype(str)).all():
                raise ValueError(f"{name}: column {c} disagrees with the dataset after the row_id join")
            led = led.drop(columns=f"{c}_meta")
    if "C_ml_int" not in led.columns:
        raise ValueError(f"{name}: no C_ml_int column")
    return led.copy(), os.path.relpath(path, ROOT)


def _flat(r, **keys):
    row = dict(keys)
    for k, v in r.items():
        if isinstance(v, list):
            row[f"{k}_lo"], row[f"{k}_hi"] = v
        else:
            row[k] = v
    return row


def cmd_aggregates(a):
    out = _make_run_dir(os.path.join(ROOT, a.out_dir), "aggregates", a)
    t0 = time.time()
    rows, by_rows, mix_rows, oh_rows, sources = [], [], [], [], {}
    for name in LEDGERS:
        led, src = load_ledger(name)
        sources[name] = src
        comps = [c for c in COMPARATORS if f"{c}_nfev" in led.columns]
        for comp in comps:
            rows.append(_flat(pm.stats_frame(led, comp, n_boot=a.n_boot), ledger=name, comparator=comp))
            for by in ("h_band", "regime", "orbit_label"):
                for r in pm.breakdown(led, comp, by, n_boot=a.n_boot):
                    g = r.pop(by)
                    by_rows.append(_flat(r, ledger=name, comparator=comp, by=by, group=g))
            for mix, target in pm.MIXTURES.items():
                if target is None:
                    continue
                w = pm.mixture_weights(led["orbit_label"].to_numpy(), target)
                r = pm.stats_frame(led, comp, weights=w, strata=led["orbit_label"].to_numpy(), n_boot=a.n_boot)
                mix_rows.append(_flat(r, ledger=name, comparator=comp, mixture=mix))
            if LEDGERS[name][2]:
                for h, tag in ((OVERHEAD_C, "15.04"), (OVERHEAD_D, "16.09")):
                    tmp = led.assign(C_alt=led["C_ml_int"] + h)
                    r = pm.stats_frame(tmp, comp, policy_col="C_alt", n_boot=a.n_boot)
                    oh_rows.append(_flat(r, ledger=name, comparator=comp, overhead=tag))
        print(f"{name}: {len(led)} orbits ({time.time() - t0:.0f}s)", flush=True)
    tabs = {"overall": rows, "by_group": by_rows, "mixtures": mix_rows, "overhead": oh_rows}
    for k, v in tabs.items():
        pd.DataFrame(v).to_csv(os.path.join(out, f"aggregates_{k}.csv"), index=False)
    save_json({"label": "POST HOC summaries of sealed ledgers (posthoc/README.md P1-P3)", "sources": sources, "n_boot": a.n_boot,
               "wall_seconds": time.time() - t0}, os.path.join(out, "aggregates_report.json"))
    ov = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(ov[["ledger", "comparator", "n", "n_no_comp", "share", "aggregate_ratio", "aggregate_ratio_ci95_lo", "aggregate_ratio_ci95_hi",
              "mean_log_ratio", "median_policy_all", "median_policy_met"]].round(4).to_string())
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("aggregates")
    sp.add_argument("--out_dir", default=OUT)
    sp.add_argument("--n_boot", type=int, default=pm.N_BOOT)
    sp.set_defaults(fn=cmd_aggregates)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
