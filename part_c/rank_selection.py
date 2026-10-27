"""Rank selection-stage candidates by the pre-registered rule (PART_C_LOG.md section 4): lowest median total cost per orbit that met the tolerance
(integration + pinned overhead), among candidates with overhead <= GATE and an evaluation-fold violation rate <= 6 %; ties within 2 % go to the simpler model.
  python part_c/rank_selection.py <select_run_dir> <overheads.json> <gate> [name_prefix ...]"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd


def complexity(name):
    if name.endswith("lin") or "_lin" in name:
        return 0
    import re
    m = re.search(r"t(\d+)(?:x(\d+))?", name)
    return int(m.group(1)) if m else 10 ** 6


def rank(run_dir, overheads, gate, prefixes=(), tie=0.02, max_viol=0.06):
    rows = pd.read_csv(os.path.join(run_dir, "select_rows.csv"))
    out = []
    for _, r in rows.iterrows():
        if prefixes and not any(r["name"].startswith(p) for p in prefixes):
            continue
        z = np.load(os.path.join(run_dir, f"orbits_{r['name']}.npz"))
        o = overheads.get(r["name"], {}).get("rhs_equiv", float("nan"))
        tot = float(np.median(z["evals"][z["met"]] + o))
        out.append({"name": r["name"], "scale": r["scale"], "violation_rate": r["violation_rate"], "median_c_int": r["median_c_int_met"], "overhead": o,
                    "median_total": tot, "eligible": bool(o <= gate and r["violation_rate"] <= max_viol), "complexity": complexity(r["name"])})
    t = pd.DataFrame(out).sort_values("median_total").reset_index(drop=True)
    el = t[t["eligible"]]
    winner = None
    if len(el):
        best = el["median_total"].min()
        near = el[el["median_total"] <= best * (1 + tie)]
        winner = near.sort_values(["complexity", "median_total"]).iloc[0]["name"]
    return t, winner


if __name__ == "__main__":
    run_dir, ov_path, gate = sys.argv[1], sys.argv[2], float(sys.argv[3])
    ov = json.load(open(ov_path))["costs"]
    t, w = rank(run_dir, ov, gate, tuple(sys.argv[4:]))
    pd.set_option("display.width", 200)
    print(t.round(3).to_string())
    print("winner:", w)
