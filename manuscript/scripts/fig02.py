"""Fig 2: paradigm gap against the timescale range. (A) Study B, oracle fixed physical step; (B) Study C, fixed physical vs fixed Sundman step.
Gap = log10(cost at the oracle fixed step / RK45 nfev at the per-trajectory oracle tolerance)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import style as S

B = pd.read_csv(S.path("part_b/results/confirm/policy/ledger.csv.gz"), usecols=["delta_phys", "C_fix_star", "RK45_spec_nfev"])
B = B.dropna()
B = B[(B["C_fix_star"] > 0) & (B["RK45_spec_nfev"] > 0)]
xb = np.log10(B["delta_phys"].to_numpy())
yb = np.log10(B["C_fix_star"] / B["RK45_spec_nfev"]).to_numpy()
Cc = pd.read_csv(S.path("out_c/confirm/sund/apply_policy_20261027T013754Z/ledger.csv"),
                 usecols=["delta_phys", "C_fix_star", "C_s_star", "RK45_spec_nfev"]).dropna()
Cc = Cc[(Cc["C_fix_star"] > 0) & (Cc["C_s_star"] > 0) & (Cc["RK45_spec_nfev"] > 0)]
xc = np.log10(Cc["delta_phys"].to_numpy())
yc_f = np.log10(Cc["C_fix_star"] / Cc["RK45_spec_nfev"]).to_numpy()
yc_s = np.log10(Cc["C_s_star"] / Cc["RK45_spec_nfev"]).to_numpy()
print("n B", len(xb), "rho", round(spearmanr(xb, yb)[0], 3), "| n C", len(xc), "rho phys", round(spearmanr(xc, yc_f)[0], 3),
      "rho sund", round(spearmanr(xc, yc_s)[0], 3))

edges = np.quantile(np.concatenate([xb, xc]), np.linspace(0, 1, 13))
edges[0] -= 1e-9; edges[-1] += 1e-9


def binned(x, y):
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (x >= lo) & (x < hi)
        if k.sum() >= 20:
            q = np.quantile(y[k], [0.25, 0.5, 0.75])
            rows.append((np.median(x[k]), q[0], q[1], q[2], int(k.sum())))
    return pd.DataFrame(rows, columns=["x_median", "q25", "median", "q75", "n"])


fig, (a, b) = plt.subplots(1, 2, figsize=(S.FULL_W, 3.2), sharey=True, gridspec_kw={"wspace": 0.12})
ylim = (-1.2, 2.6)
hb = a.hexbin(xb, yb, gridsize=45, cmap="cividis", bins="log", mincnt=1, extent=(0, xb.max(), ylim[0], ylim[1]), linewidths=0)
bb = binned(xb, yb)
a.errorbar(bb["x_median"], bb["median"], yerr=[bb["median"] - bb["q25"], bb["q75"] - bb["median"]], fmt="s", color="#FFFFFF",
           mec="#000000", mew=0.8, ms=4.5, ecolor="#000000", **S.ERR)
S.refline(a, y=0)
a.set_ylim(*ylim)
a.set_xlabel(r"Timescale range, log$_{10}$ $\Delta_\tau$")
a.set_ylabel("Fixed-step cost / RK45 oracle cost (dex)")
cax = a.inset_axes([0.52, 0.1, 0.42, 0.04])
cb = fig.colorbar(hb, cax=cax, orientation="horizontal")
cb.set_label("Trajectories per bin", fontsize=8)
cb.ax.tick_params(labelsize=8)
a.text(0.03, 0.95, r"Study B, $\rho$ = 0.91", transform=a.transAxes, va="top", fontsize=9)
S.panel(a, "A", x=-0.2)

bf, bs = binned(xc, yc_f), binned(xc, yc_s)
for d, col, mk, lab, off in ((bf, S.C["fixed"], "s", r"fixed physical step ($\rho$ = 0.91)", -0.012),
                             (bs, S.C["sundman"], "o", r"fixed Sundman step ($\rho$ = 0.22)", 0.012)):
    a2 = b.errorbar(d["x_median"] + off, d["median"], yerr=[d["median"] - d["q25"], d["q75"] - d["median"]], fmt=mk + "-", color=col,
                    ms=4.5, lw=1.0, label=lab, **S.ERR)
S.refline(b, y=0)
b.set_xlabel(r"Timescale range, log$_{10}$ $\Delta_\tau$")
b.legend(loc="upper left", fontsize=8.5, title="Study C", title_fontsize=9, alignment="left")
S.panel(b, "B", x=-0.06)
S.save(fig, 2)
S.write_data(2, "A_studyB_binned", bb)
S.write_data(2, "B_studyC_binned_fixed_physical", bf)
S.write_data(2, "B_studyC_binned_fixed_sundman", bs)
S.write_data(2, "A_studyB_points", pd.DataFrame({"log10_delta": xb, "gap_dex": yb}))
S.write_data(2, "B_studyC_points", pd.DataFrame({"log10_delta": xc, "gap_fixed_physical_dex": yc_f, "gap_fixed_sundman_dex": yc_s}))
