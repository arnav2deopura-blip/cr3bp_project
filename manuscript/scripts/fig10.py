"""Fig 10 (exploratory D7): (A) FTLE on the surface of section y = 0, z = vz = 0, vy > 0 at C = 3.10 with policy violations;
(B) violation rate by FTLE decile at C = 3.19, 3.10, 2.95."""
import glob
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import style as S

d7 = glob.glob(S.path("out_d/explore/d7_*"))[0]
P = pd.read_csv(f"{d7}/slice_C3.1.csv")
R = json.load(open(S.path("part_d/results/explore/d7_report.json")))
ok = P[P["ref_ok"].astype(bool)].copy()
print("points", len(P), "ref ok", len(ok), "violations", int(ok["violation"].sum()))
xs, vs = np.sort(P["x"].unique()), np.sort(P["vx"].unique())
grid = ok.pivot_table(index="vx", columns="x", values="ftle").reindex(index=vs, columns=xs)
dx, dv = np.diff(xs).mean(), np.diff(vs).mean()

fig, (a, b) = plt.subplots(1, 2, figsize=(S.FULL_W, 3.3), gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.6})
im = a.pcolormesh(np.append(xs - dx / 2, xs[-1] + dx / 2), np.append(vs - dv / 2, vs[-1] + dv / 2), grid.to_numpy(), cmap="cividis",
                  shading="flat", rasterized=True)
v = ok[ok["violation"].astype(bool)]
a.scatter(v["x"], v["vx"], s=2.2, facecolors="none", edgecolors=S.C["vermillion"], linewidths=0.35, label="policy violation")
a.set_xlabel("x (DU)")
a.set_ylabel(r"$\dot{x}$ (DU/TU)")
a.set_facecolor("white")
cb = fig.colorbar(im, ax=a, fraction=0.05, pad=0.02)
cb.set_label("FTLE over 8 TU (1/TU)", fontsize=9)
a.legend(loc="upper right", fontsize=8, markerscale=3, handletextpad=0.2, borderpad=0.2, frameon=True, framealpha=0.9, edgecolor="none")
a.set_title("C = 3.10", fontsize=9, loc="left")
S.panel(a, "A", x=-0.2)

rows = []
dec = np.arange(1, 11)
for (key, col, mk, lab) in (("C=3.19", S.C["sundman"], "o", "C = 3.19"), ("C=3.1", S.C["fixed"], "s", "C = 3.10"), ("C=2.95", S.C["dop853"], "D", "C = 2.95")):
    m = R["maps"][key]
    rates = 100 * np.array(m["violation_rate_by_ftle_decile"])
    b.plot(dec, rates, mk + "-", color=col, ms=4.5, lw=1.1, label=f"{lab[4:]}, AUC {m['auc_ftle_for_violation']:.2f}")
    for d_, r_ in zip(dec, rates):
        rows.append({"section": lab, "ftle_decile": d_, "violation_rate_pct": r_, "n_points_section": m["n_ref_ok"]})
b.set_xticks(dec)
b.set_xlabel("FTLE decile")
b.set_ylabel("Violation rate (%)")
b.set_ylim(-2, 102)
b.legend(loc="upper left", fontsize=8, title="Jacobi constant C", title_fontsize=8, alignment="left")
S.panel(b, "B", x=-0.2)
S.save(fig, 10)
S.write_data(10, "A_section_C3.10", ok[["x", "vx", "vy", "ftle", "violation"]])
S.write_data(10, "B_deciles", pd.DataFrame(rows))
