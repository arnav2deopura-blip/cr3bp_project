"""Fig 6: (A) extra mean log10 total cost of each D1 arm relative to the 25-tree Sundman policy (paired, 95 % bootstrap CI);
(B) exploratory D6 drop-group refits, increase in test MSE for the Sundman and physical-time targets."""
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import style as S

T = json.load(open(S.path("part_d/results/confirm/confirm_tests_confirm.json")))
pw = T["D1_pairwise_vs_ml"]
arms = [("rule_T", "Rule: T", "rule"), ("rule_TH", "Rule: T, regime", "rule"), ("rule_TP", r"Rule: T, $\tau_p$", "rule"),
        ("rule_THP", r"Rule: T, regime, $\tau_p$", "rule"), ("rule_THPD", r"Rule: T, regime, $\tau_p$, C $-$ C$_{L_1}$", "rule"),
        ("rule_THR", r"Rule: T, regime, $r_p$ (test arm)", "rule"), ("lin33", "Linear model, 33 features", "linear")]
rows = []
for key, lab, kind in arms:
    d = pw[key]
    rows.append({"arm": lab, "kind": kind, "extra_cost_dex": d["mean_log10_saving"], "ci_lo": d["ci95"][0], "ci_hi": d["ci95"][1],
                 "n_paired": d["n_paired"], "dataset": "Study D confirmation"})
g = T["D3_also_C5_ml_vs_global"]
rows.append({"arm": "One global Sundman step", "kind": "global", "extra_cost_dex": g["mean_log10_saving"], "ci_lo": g["ci95"][0],
             "ci_hi": g["ci95"][1], "n_paired": g["n_paired"], "dataset": "Study D replication"})
A = pd.DataFrame(rows).sort_values("extra_cost_dex").reset_index(drop=True)

fig, (a, b) = plt.subplots(1, 2, figsize=(S.FULL_W, 3.5), gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.62})
fig.subplots_adjust(left=0.30, right=0.98, top=0.93, bottom=0.14)
sty = {"rule": (S.C["rules"], "s", "none"), "linear": (S.C["linear"], "v", S.C["linear"]), "global": (S.C["onescale"], "o", S.C["onescale"])}
for i, r in A.iterrows():
    col, mk, fc = sty[r["kind"]]
    a.errorbar(r["extra_cost_dex"], i, xerr=[[r["extra_cost_dex"] - r["ci_lo"]], [r["ci_hi"] - r["extra_cost_dex"]]], fmt=mk, color=col,
               mfc=fc if fc != "none" else "white", mec=col, ms=5, **S.ERR)
    a.text(r["ci_hi"] + 0.014, i, f"{r['extra_cost_dex']:+.3f}", va="center", fontsize=8)
S.refline(a, x=0)
a.set_yticks(range(len(A)))
a.set_yticklabels(A["arm"], fontsize=8.5)
a.set_xlabel("Extra cost relative to the 25-tree policy (dex)")
a.set_xlim(-0.05, 0.33)
a.invert_yaxis()
fig.text(0.005, 0.965, "A", fontsize=12, fontweight="bold", va="bottom")

d6 = json.load(open(S.path("part_d/results/explore/d6_report.json")))
groups = [("state", "State"), ("proximity", "Proximity"), ("jacobi", "Jacobi"), ("periapsis", "Periapsis"), ("horizon", "Horizon"),
          ("ftle (offline, needs an integration)", "FTLE (offline)")]
rowsB = []
for t, tl in (("sundman_time (log10 ds*)", "Sundman-time target"), ("physical_time (log10 dt*)", "physical-time target")):
    for gk, gl in groups:
        src = d6[t]["with_ftle_group"]["groups"] if gk.startswith("ftle") else d6[t]["deployable_groups"]["groups"]
        e = src[gk]
        rowsB.append({"target": tl, "group": gl, "drop_delta_mse": e["drop_delta_mse"], "ci_lo": e["drop_ci95"][0], "ci_hi": e["drop_ci95"][1]})
Bd = pd.DataFrame(rowsB)
hgt = 0.36
for j, (tl, col, mk) in enumerate((("Sundman-time target", S.C["sundman"], "o"), ("physical-time target", S.C["fixed"], "s"))):
    s = Bd[Bd["target"] == tl].reset_index(drop=True)
    y = np.arange(len(s)) + (j - 0.5) * hgt
    b.barh(y, s["drop_delta_mse"], height=hgt, color=col, edgecolor="#000000", linewidth=0.5, label=tl, hatch="" if j == 0 else "//")
    b.errorbar(s["drop_delta_mse"], y, xerr=[s["drop_delta_mse"] - s["ci_lo"], s["ci_hi"] - s["drop_delta_mse"]], fmt="none", ecolor="#000000", **S.ERR)
b.axvline(0, color="#000000", lw=0.75)
b.set_yticks(range(len(groups)))
b.set_yticklabels([gl for _, gl in groups], fontsize=8.5)
b.invert_yaxis()
b.set_xlabel("Increase in test MSE when removed")
b.legend(loc="lower right", fontsize=8)
fig.text(b.get_position().x0 - 0.135, 0.965, "B", fontsize=12, fontweight="bold", va="bottom")
S.save(fig, 6)
S.write_data(6, "A_arms", A)
S.write_data(6, "B_drop_group", Bd)
print(A.round(4).to_string(index=False))
