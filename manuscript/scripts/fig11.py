"""Fig 11: pilot of a learned 8th-order fixed step in Sundman time (post hoc P25, exploratory; Study D replication set, n = 12,253).
(A) ratio of totals (comparator evaluations / policy evaluations) of the learned RK8 policy and of the sealed RK4 policy against the same four
adaptive comparators, each certified by the policies' own rule; (B) the learned RK8 policy against the four comparators at equal overall risk,
by horizon band. Error bars: 95% bootstrap intervals over trajectories. Source: posthoc/results/rk8s_learned_*/rk8s_learned_summary.csv."""
import glob

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import style as S

src = sorted(glob.glob(S.path("posthoc/results/rk8s_learned_*/rk8s_learned_summary.csv")))[-1]
D = pd.read_csv(src)
RK8_COL = "#003F5C"          # the 8th-order policy; the RK4 Sundman-time policy keeps its blue (style.C["sundman"])
comps = [("physical:RK45", "RK45,\nphysical time", S.C["rk45"], ""), ("sundman:RK45", "RK45,\nSundman time", S.C["rk45"], "//"),
         ("physical:DOP853", "DOP853,\nphysical time", S.C["dop853"], ""), ("sundman:DOP853", "DOP853,\nSundman time", S.C["dop853"], "//")]
cal_tag = {"physical:RK45": "physical:RK45_calsym (P7)", "sundman:RK45": "sundman:RK45_calsym (P20)",
           "physical:DOP853": "physical:DOP853_calsym (P7)", "sundman:DOP853": "sundman:DOP853_calsym (P20)"}


def row(comparison, band):
    r = D[(D["comparison"] == comparison) & (D["h_band"] == band)]
    assert len(r) == 1, (comparison, band, len(r))
    return r.iloc[0]


rows = []
fig, (a, b) = plt.subplots(1, 2, figsize=(S.FULL_W, 3.3), gridspec_kw={"wspace": 0.28, "width_ratios": [1, 1.25]})

w = 0.36
for i, (key, lab, _, _) in enumerate(comps):
    for j, (pol, col, hatch, plab) in enumerate((("RK8 policy", RK8_COL, "", "Learned 8th-order step"),
                                                 ("sealed RK4 policy", S.C["sundman"], "..", "Learned 4th-order step (Study C policy)"))):
        r = row(f"{pol} vs {cal_tag[key]}", "all")
        v, lo, hi = r["aggregate_ratio"], r["aggregate_ratio_ci95_lo"], r["aggregate_ratio_ci95_hi"]
        x = i + (j - 0.5) * w
        a.bar(x, v, width=w, color=col, hatch=hatch, edgecolor="#000000", linewidth=0.5, label=plab if i == 0 else None)
        a.errorbar(x, v, yerr=[[v - lo], [hi - v]], fmt="none", ecolor="#000000", **S.ERR)
        a.text(x, hi + 0.04, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)
        rows.append({"panel": "A", "policy": plab, "comparator": lab.replace("\n", " "), "comparator_setting": "certified by the policies' rule",
                     "h_band_TU": "all", "n": int(r["n"]), "ratio_of_totals": v, "ci95_lo": lo, "ci95_hi": hi,
                     "share_policy_cheaper": r["share"], "policy_violation_rate": r["policy_violation_rate"],
                     "comparator_violation_rate": r["comparator_violation_rate"]})
S.refline(a, y=1, color="#000000")
a.set_xticks(range(len(comps)))
a.set_xticklabels([c[1].replace(",\n", "\n").replace(" time", "") for c in comps], fontsize=8.5)
a.set_xlabel("Adaptive comparator and its time variable")
a.set_ylabel("Ratio of totals (comparator / policy)")
a.set_ylim(0, 2.95)
a.legend(loc="upper right", fontsize=8, handlelength=1.4)
S.panel(a, "A", x=-0.16)

bands = [("1-3", "1–3 TU"), ("3-8", "3–8 TU"), ("8-20", "8–20 TU"), ("all", "All")]
w = 0.2
for j, (key, lab, col, hatch) in enumerate(comps):
    for i, (band, bl) in enumerate(bands):
        r = row(f"RK8 policy vs {key}_eqr", band)
        v, lo, hi = r["aggregate_ratio"], r["aggregate_ratio_ci95_lo"], r["aggregate_ratio_ci95_hi"]
        x = i + (j - 1.5) * w
        b.bar(x, v, width=w, color=col, hatch=hatch, edgecolor="#000000", linewidth=0.5)
        b.errorbar(x, v, yerr=[[v - lo], [hi - v]], fmt="none", ecolor="#000000", **S.ERR)
        rows.append({"panel": "B", "policy": "Learned 8th-order step", "comparator": lab.replace("\n", " "),
                     "comparator_setting": "equal overall risk", "h_band_TU": band, "n": int(r["n"]), "ratio_of_totals": v, "ci95_lo": lo,
                     "ci95_hi": hi, "share_policy_cheaper": r["share"], "policy_violation_rate": r["policy_violation_rate"],
                     "comparator_violation_rate": r["comparator_violation_rate"]})
S.refline(b, y=1, color="#000000")
b.set_xticks(range(len(bands)))
b.set_xticklabels([bl for _, bl in bands])
b.set_xlabel("Horizon band")
b.set_ylabel("Ratio of totals (comparator / 8th-order policy)")
b.set_ylim(0, 3.75)
b.set_yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
b.legend([Patch(facecolor=c, hatch=h, edgecolor="#000000", linewidth=0.5) for _, _, c, h in comps], [l.replace("\n", " ") for _, l, _, _ in comps],
         loc="upper right", fontsize=8, ncol=2, handlelength=1.4, columnspacing=1.0)
S.panel(b, "B", x=-0.13)

S.save(fig, 11)
S.write_data(11, "rk8_pilot", pd.DataFrame(rows))
