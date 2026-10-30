"""Fig 8: D2 compiled wall-clock. (A) single-thread throughput ratio (comparator time / policy time) by horizon band and overall;
(B) batch throughput ratio vs batch size on 1 and 10 threads (median over 7 repeats; whisker down to the worst repeat)."""
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import style as S

D = json.load(open(S.path("part_d/results/confirm/d2_report.json")))
st = D["D2_single_thread"]
comps = [("DP54_PI_eqr", "DP5(4), PI controller", S.C["rk45"], "^", ""), ("DP54_scipy_eqr", "DP5(4), SciPy controller", "#F5C978", "v", "//"),
         ("DOP853_eqr", "DOP853", S.C["dop853"], "D", "..")]
groups = [("h_band=1-3", "1–3 TU"), ("h_band=3-8", "3–8 TU"), ("h_band=8-20", "8–20 TU"), (None, "All")]
rows = []
fig, (a, b) = plt.subplots(1, 2, figsize=(S.FULL_W, 3.2), gridspec_kw={"wspace": 0.3})
w = 0.26
for j, (key, lab, col, mk, hatch) in enumerate(comps):
    for i, (g, gl) in enumerate(groups):
        x = i + (j - 1) * w
        if g is None:
            v, (lo, hi), n = st[key]["throughput_ratio"], st[key]["ci95"], st[key]["n"]
        else:
            e = st[key]["by_group"][g]
            v, lo, hi, n = e["throughput_ratio"], np.nan, np.nan, e["n"]
        a.bar(x, v, width=w, color=col, hatch=hatch, edgecolor="#000000", linewidth=0.5, label=lab if i == 0 else None)
        if g is None:
            a.errorbar(x, v, yerr=[[v - lo], [hi - v]], fmt="none", ecolor="#000000", **S.ERR)
        rows.append({"panel": "A", "comparator": lab, "group": gl, "throughput_ratio": v, "ci95_lo": lo, "ci95_hi": hi, "n": n})
a.axhline(1, color="#000000", lw=0.75, ls=(0, (4, 3)))
a.set_xticks(range(len(groups)))
a.set_xticklabels([gl for _, gl in groups])
a.set_xlabel("Horizon band")
a.set_ylabel("Throughput ratio (comparator / policy)")
a.set_ylim(0, 2.0)
a.legend(loc="upper right", fontsize=8)
S.panel(a, "A", x=-0.15)

R = D["batch_throughput_ratios_adaptive_over_policy"]
sizes = [1000, 10000, 100000]
for key, lab, col, mk, _ in comps:
    for th, ls in ((1, "-"), (10, (0, (4, 2)))):
        med = np.array([R[f"{key}|{n}|{th}"]["median_ratio"] for n in sizes])
        worst = np.array([R[f"{key}|{n}|{th}"]["min_ratio_over_repeats"] for n in sizes])
        xs = np.array(sizes) * (1.0 if th == 1 else 1.08)
        b.errorbar(xs, med, yerr=[med - worst, np.zeros_like(med)], fmt=mk, ls=ls, color=col, mfc=col if th == 1 else "white", mec=col, ms=4.5,
                   lw=1.1, **S.ERR)
        for n, m_, w_ in zip(sizes, med, worst):
            rows.append({"panel": "B", "comparator": lab, "group": f"batch {n}, {th} thread(s)", "throughput_ratio": m_, "ci95_lo": w_,
                         "ci95_hi": np.nan, "n": n})
b.axhline(1, color="#000000", lw=0.75, ls=(0, (4, 3)))
b.set_xscale("log")
b.set_xticks(sizes)
b.set_xticklabels(["10$^3$", "10$^4$", "10$^5$"])
b.set_xlabel("Batch size (trajectories)")
b.set_ylabel("Batch throughput ratio")
b.set_ylim(0.3, 1.22)
h1 = [Line2D([], [], color=c, marker=m, ls="none", ms=4.5) for _, _, c, m, _ in comps]
l1 = b.legend(h1, [l for _, l, _, _, _ in comps], loc="lower left", fontsize=8, title="Method", title_fontsize=8, alignment="left")
b.add_artist(l1)
h2 = [Line2D([], [], color="#000000", ls="-", marker="o", ms=4), Line2D([], [], color="#000000", ls=(0, (4, 2)), marker="o", mfc="white", ms=4)]
b.legend(h2, ["1 thread", "10 threads"], loc="upper right", fontsize=8, ncol=2, handlelength=2.6)
S.panel(b, "B", x=-0.15)
S.save(fig, 8)
S.write_data(8, "throughput", pd.DataFrame(rows))
