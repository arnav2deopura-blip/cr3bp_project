"""Fig 9: D5, frozen Study C policy on the JPL periodic-orbit families. (A) violation rate per family (policy with 95 % Clopper-Pearson;
RK45 certified and equal-risk, DOP853 equal-risk); (B) median log10(policy raw ds / oracle ds) per family."""
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import style as S

G = pd.read_csv(S.path("part_d/results/confirm/d5_table_by_group.csv")).set_index("group")
D = json.load(open(S.path("part_d/results/confirm/d5_report.json")))
med = D["oracle_label"]["by_group_median"]
names = {"axial_L1": "Axial L1", "axial_L2": "Axial L2", "vertical_L1": "Vertical L1", "vertical_L2": "Vertical L2", "lyapunov_L1": "Lyapunov L1",
         "lyapunov_L2": "Lyapunov L2", "dro": "DRO", "halo_L1": "Halo L1", "halo_L2": "Halo L2", "butterfly": "Butterfly", "resonant": "Resonant"}
order = G.sort_values("rate", ascending=False).index.tolist()
y = np.arange(len(order))
fig, (a, b) = plt.subplots(1, 2, figsize=(S.FULL_W, 3.6), sharey=True, gridspec_kw={"width_ratios": [1.5, 1.0], "wspace": 0.08})
rows = []
spec = [("policy", "rate", S.C["sundman"], "o", S.C["sundman"], -0.27), ("RK45, certified", "RK45_cal_rate", S.C["rk45"], "^", S.C["rk45"], -0.09),
        ("RK45, equal-risk", "RK45_eqr_rate", S.C["rk45"], "^", "white", 0.09), ("DOP853, equal-risk", "DOP853_eqr_rate", S.C["dop853"], "D", "white", 0.27)]
for lab, col_, c, mk, fc, off in spec:
    n = G.loc[order, "n"].to_numpy()
    r = G.loc[order, col_].to_numpy()
    k = np.rint(r * n).astype(int)
    ci = np.array([S.cp_interval(kk, nn) for kk, nn in zip(k, n)])
    if lab == "policy":
        a.errorbar(100 * r, y + off, xerr=[100 * (r - ci[:, 0]), 100 * (ci[:, 1] - r)], fmt=mk, color=c, mfc=fc, mec=c, ms=5, label=lab, **S.ERR)
    else:
        a.plot(100 * r, y + off, mk, color=c, mfc=fc, mec=c, ms=4.5, label=lab)
    for g, kk, nn, rr, (lo, hi) in zip(order, k, n, r, ci):
        rows.append({"panel": "A", "family": names[g], "series": lab, "violations": kk, "runs": nn, "rate": rr, "cp_lo": lo, "cp_hi": hi})
a.axvline(10, color="#555555", lw=0.75, ls=(0, (4, 3)))
a.text(11, -0.62, "10% test level", fontsize=8, color="#555555", va="center")
a.set_yticks(y)
a.set_yticklabels([f"{names[g]} ({int(G.loc[g, 'n'])})" for g in order], fontsize=8.5)
a.set_ylim(len(order) - 0.5, -0.9)
a.set_xlim(0, 104)
a.set_xlabel("Violation rate (%)")
h, l = a.get_legend_handles_labels()
o = [l.index(x) for x in ("policy", "RK45, certified", "RK45, equal-risk", "DOP853, equal-risk")]
a.legend([h[i] for i in o], [l[i] for i in o], loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=4, fontsize=8, columnspacing=0.9,
         handletextpad=0.3)
S.panel(a, "A", x=-0.33, y=1.09)
v = np.array([med[g] for g in order])
b.barh(y, v, height=0.55, color=S.C["sundman"], edgecolor="#000000", linewidth=0.5)
b.axvline(0, color="#000000", lw=0.75)
for yy, vv in zip(y, v):
    b.text(vv + (0.015 if vv >= 0 else -0.015), yy, f"{vv:+.2f}", va="center", ha="left" if vv >= 0 else "right", fontsize=8)
    rows.append({"panel": "B", "family": names[order[int(yy)]], "series": "median log10(policy ds / oracle ds)", "rate": vv})
b.set_xlim(-0.2, 0.68)
b.set_xlabel(r"Median log$_{10}$(predicted / oracle step)")
S.panel(b, "B", x=-0.06, y=1.09)
S.save(fig, 9)
S.write_data(9, "families", pd.DataFrame(rows))
